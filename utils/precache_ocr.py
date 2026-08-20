"""
Forhåndscache PaddleOCR-resultater for et sett med dokumenter.

Kjører kun PaddleOCR (orientering + tekstgjenkjenning) og lagrer
resultatene i OCR-cachen. Ingen YOLO, ingen FNR-deteksjon, ingen
evaluering — bare den tyngste GPU-operasjonen isolert.

Arkitektur: hovedprosessen koordinerer bare, og alt arbeid skjer i
uavhengige prosesser som henter filer fra en delt kø. Grunnen er målt:
~77 % av tiden i én pipeline går til enkelttrådet CPU-forbehandling
(normalisering, bildekopier), og bare ~11 % til GPU-arbeid. Én prosess
klarer altså ikke å mette kortet — flere prosesser mot samme GPU gjør det.

  - Pre-rendrer neste PDF(er) mens OCR-en jobber
  - Flere parallelle GPU-pipeliner mot samme kort (--gpu-prosesser)
  - Kan kjøre CPU-OCR i egne prosesser i tillegg (--cpu-ocr)
  - Overvåker minne og tilpasser pipeline-dybden
  - Hopper automatisk over allerede cachede dokumenter

Bruk:
    python precache_ocr.py --mappe /sti/til/pdfer
    python precache_ocr.py --mappe /sti/til/pdfer --gpu-prosesser 4
    python precache_ocr.py --mappe /sti/til/pdfer --gpu-prosesser --profil
    python precache_ocr.py --velg-fra-fil filer.txt --mappe /sti/til/pdfer
    python precache_ocr.py --mappe /sti/til/pdfer --gpu-prosesser 4 --cpu-ocr 4
"""

import argparse
import multiprocessing as mp
import os
import sys
import threading
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

warnings.filterwarnings("ignore", message=".*ccache.*")
os.environ["GLOG_minloglevel"] = "3"
os.environ["GLOG_v"] = "0"
os.environ["FLAGS_call_stack_level"] = "0"

_UTILS = os.path.dirname(os.path.abspath(__file__))
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)
_APP = os.path.join(_UTILS, "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

import numpy as np

from config import SIDER_PER_OCR_BATCH
from file_selection import velg_filer
from load_pdf import les_sider
from ocr_cache import les_cache as les_ocr_cache, skriv_cache as skriv_ocr_cache

# Merk: orientering og paddle_ocr_model_fnr importeres IKKE her. De trekker
# inn paddle, og en fork() etter at CUDA er berørt gir ubrukelige
# CUDA-kontekster i barneprosessene. Motorene importerer dem selv.


# ── Logging ──────────────────────────────────────────────────────

_LOG_DIR = "/data2/tmp"


class _Tee:
    """Skriv til både konsoll og loggfil samtidig."""

    def __init__(self, loggfil_sti):
        self._stdout = sys.stdout
        self._fil = open(loggfil_sti, "a", encoding="utf-8")

    def write(self, data):
        self._stdout.write(data)
        self._fil.write(data)
        self._fil.flush()

    def flush(self):
        self._stdout.flush()
        self._fil.flush()

    def fileno(self):
        return self._stdout.fileno()


def _setup_loggfil():
    """Sett opp tee-logging til /data2/tmp/precache_ocr_<tidsstempel>.log."""
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        loggfil = os.path.join(_LOG_DIR, f"precache_ocr_{ts}.log")
        sys.stdout = _Tee(loggfil)
        print(f"Loggfil: {loggfil}")
        return loggfil
    except OSError as e:
        print(f"Advarsel: Kunne ikke opprette loggfil i {_LOG_DIR}: {e}")
        return None


# ── Tidsformatering ──────────────────────────────────────────────

def _fmt_tid(sekunder):
    """Formater sekunder til lesbar h:mm:ss / m:ss / Xs."""
    if sekunder < 0:
        return "?"
    s = int(sekunder)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}t {(s % 3600) // 60:02d}m {s % 60:02d}s"


# ── Ressursovervåking ────────────────────────────────────────────

def _frigjør_gpu_cache():
    """Frigjør ubrukt cachet GPU-minne tilbake til CUDA.

    PaddlePaddle sin CUDA-allokator cacher minneblokker for gjenbruk.
    Når bilder har varierende dimensjoner allokeres nye blokker som aldri
    frigjøres av seg selv — dette ser ut som en "leak" i nvidia-smi.
    Kall denne periodisk for å holde GPU-minnet stabilt.
    """
    try:
        import paddle
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        pass

def _minne_info():
    """Returnerer (brukt_gb, tilgjengelig_gb, prosent_brukt) for systemminne."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.used / 1e9, mem.available / 1e9, mem.percent
    except ImportError:
        pass
    # Fallback for macOS/Linux uten psutil
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for linje in f:
                deler = linje.split()
                if len(deler) >= 2:
                    info[deler[0].rstrip(":")] = int(deler[1]) * 1024
            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", 0)
            brukt = total - available
            return brukt / 1e9, available / 1e9, (brukt / total * 100) if total else 0
    except (FileNotFoundError, KeyError):
        return 0, 0, 0


def _gpu_minne_info():
    """Returnerer (brukt_mb, total_mb) for GPU, eller None hvis ikke tilgjengelig."""
    # Bevisst kun nvidia-smi: paddle-API-et her ville initialisert CUDA i
    # koordinatoren, og da blir fork() av GPU-prosessene ugyldig.
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            linje = result.stdout.strip().split("\n")[0]
            brukt, total = [int(x.strip()) for x in linje.split(",")]
            return brukt, total
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _er_minne_trygt(grense_prosent=88):
    """Sjekk om det er nok minne til å fortsette å pre-rendre."""
    _, _, prosent = _minne_info()
    if prosent == 0:
        return True  # klarte ikke å lese minne, anta OK
    return prosent < grense_prosent


# ── PDF-lasting i tråd ───────────────────────────────────────────

def _last_pdf(sti):
    """Last og rendr en PDF til bildeliste (CPU-bundet). Returnerer (sti, bilder) eller (sti, Exception)."""
    try:
        bilder = les_sider(sti)
        return sti, bilder
    except Exception as e:
        return sti, e


# ── Auto-tuning ──────────────────────────────────────────────────

def _antall_cpu_kjerner():
    """Returner antall tilgjengelige CPU-kjerner."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def _auto_prefetch():
    """Velg prefetch-dybde basert på tilgjengelig RAM.

    Hver pre-rendret PDF bruker ca. 50-200 MB RAM (avhengig av sideantall).
    Prefetch skal være >= 2× dokument-batch for å holde GPU-en mettet.
    """
    _, tilg, _ = _minne_info()
    if tilg <= 0:
        return 16  # fallback
    # Anta ~150 MB per PDF i snitt, bruk maks 10% av tilgjengelig RAM
    maks_fra_ram = int(tilg * 0.10 * 1000 / 150)
    return min(max(maks_fra_ram, 12), 64)


def _gpu_prosent_brukt():
    """Returner GPU-minnebruk i prosent, eller None."""
    gpu = _gpu_minne_info()
    if gpu and gpu[1] > 0:
        return gpu[0] / gpu[1] * 100
    return None


class _AdaptivBatchStørrelse:
    """Dynamisk GPU-batchstørrelse som tilpasser seg minnebruk.

    Starter konservativt og øker gradvis så lenge GPU-minnet er under
    en trygg grense. Reduserer umiddelbart hvis grensen overskrides.
    """

    def __init__(self, start=4, minimum=2, maksimum=16, gpu_grense_prosent=75):
        self.nåværende = start
        self.minimum = minimum
        self.maksimum = maksimum
        self.gpu_grense = gpu_grense_prosent
        self._forrige_gpu_pct = None
        self._stabil_teller = 0  # antall batches under grensen
        self._sperre = 0         # batches uten vekst etter tom-for-minne

    def oom(self):
        """Kalles etter tom-for-minne: halver og utsett ny vekst.

        nvidia-smi viser paddle-allokatorenes høyvannsmerke, ikke ledig
        minne akkurat nå. Etter en minnefeil er den målingen upålitelig,
        så vi holder oss nede en stund uansett hva den sier.
        """
        self.nåværende = max(self.minimum, self.nåværende // 2)
        self._stabil_teller = 0
        self._sperre = 10

    def neste(self):
        """Returner neste batchstørrelse, justert basert på GPU-minne."""
        if self._sperre > 0:
            self._sperre -= 1
            return self.nåværende

        gpu_pct = _gpu_prosent_brukt()
        if gpu_pct is None:
            return self.nåværende  # kan ikke lese GPU, behold nåværende

        self._forrige_gpu_pct = gpu_pct

        if gpu_pct > self.gpu_grense + 5:
            # Over grensen — krympe raskt
            self.nåværende = max(self.minimum, self.nåværende - 2)
            self._stabil_teller = 0
        elif gpu_pct > self.gpu_grense:
            # Nær grensen — krympe forsiktig
            self.nåværende = max(self.minimum, self.nåværende - 1)
            self._stabil_teller = 0
        else:
            # Under grensen — øk aggressivt når det er mye ledig
            self._stabil_teller += 1
            if gpu_pct < self.gpu_grense * 0.3:
                # Veldig lavt minne (< 30% av grensen) — øk raskt
                self.nåværende = min(self.maksimum, self.nåværende + 3)
            elif gpu_pct < self.gpu_grense * 0.5 and self._stabil_teller >= 2:
                # Under halvparten av grensen — øk moderat
                self.nåværende = min(self.maksimum, self.nåværende + 2)
                self._stabil_teller = 0
            elif self._stabil_teller >= 3 and gpu_pct < self.gpu_grense - 15:
                # Godt under grensen i 3+ batches — øk forsiktig
                self.nåværende = min(self.maksimum, self.nåværende + 1)
                self._stabil_teller = 0

        return self.nåværende

    def ok(self):
        """Batch gikk bra. (Her styres veksten av minnemålingen.)"""

    def __repr__(self):
        gpu_str = f" GPU:{self._forrige_gpu_pct:.0f}%" if self._forrige_gpu_pct else ""
        return f"dokument_batch={self.nåværende}{gpu_str}"


class _AIMDBatchStørrelse:
    """Batchstørrelse styrt av faktiske minnefeil, ikke av nvidia-smi.

    Når flere prosesser deler kortet med hvert sitt minnetak er den globale
    målingen ubrukelig som styringssignal: den ligger *ved* taket når alt
    går som planlagt, og en måling-styrt regulator bremser da hele tiden.

    Her vokser batchen ett hakk for hver batch som går bra, og halveres når
    prosessen går tom for minne — samme mønster som TCP. Siden en minnefeil
    bare koster en omkjøring av samme dokumenter, presser dette kortet så
    hardt det tåler uten å miste noe.
    """

    def __init__(self, start=2, minimum=1, maksimum=12):
        self.nåværende = start
        self.minimum = minimum
        self.maksimum = maksimum
        self._ok_siden = 0
        self._oom = 0

    def neste(self):
        if self._ok_siden >= 2 and self.nåværende < self.maksimum:
            self.nåværende += 1
            self._ok_siden = 0
        return self.nåværende

    def ok(self):
        self._ok_siden += 1

    def oom(self):
        self.nåværende = max(self.minimum, self.nåværende // 2)
        self._ok_siden = -4   # hold igjen noen batcher før ny vekst
        self._oom += 1

    def __repr__(self):
        return f"dokument_batch={self.nåværende} (AIMD, {self._oom} minnefeil)"


# ── Hovudlogikk ──────────────────────────────────────────────────

def _utled_cache_mappe(args):
    """Utled cache-mappe fra argumenter og miljøvariabler.

    Prioritet:
      1. Eksplisitt --cache
      2. $SLADD_CACHE/<uttrekk-navn>/ocr/  (server-standard: /data2/cache)
      3. Feil — krever at én av de to er satt
    """
    if args.cache:
        return args.cache
    cache_base = os.environ.get("SLADD_CACHE")
    if cache_base:
        uttrekk_navn = os.path.basename(os.path.normpath(args.mappe))
        return os.path.join(cache_base, uttrekk_navn, "ocr")
    return None


def _antall_cachet(filer, cache_mappe):
    """Tell hvor mange filer som allerede har gyldig cache."""
    n = 0
    for f in filer:
        navn = os.path.basename(f)
        cachet = les_ocr_cache(cache_mappe, navn)
        if cachet is not None:
            n += 1
    return n


def _filtrer_cachet(filer, cache_mappe):
    """Returner kun filer som IKKE allerede har gyldig cache."""
    mangler = []
    for f in filer:
        navn = os.path.basename(f)
        cachet = les_ocr_cache(cache_mappe, navn)
        if cachet is None:
            mangler.append(f)
    return mangler


def _skriv_ressursstatus():
    """Skriv ut ressursstatus."""
    brukt, tilg, pct = _minne_info()
    deler = []
    if pct > 0:
        deler.append(f"RAM: {brukt:.1f}/{brukt + tilg:.1f} GB ({pct:.0f}%)")
    gpu = _gpu_minne_info()
    if gpu:
        deler.append(f"GPU: {gpu[0]}/{gpu[1]} MB ({gpu[0] / gpu[1] * 100:.0f}%)")
    if deler:
        print(f"  Ressurser: {' | '.join(deler)}")


# ── Samplende profiler (py-spy-erstatning uten nettverk) ─────────

class _Profil:
    """Teller hvor hovedtråden står, ved å ta stakk-prøver utenfra.

    Den innerste Python-rammen sier hva tiden går til: paddle sine
    run/infer-kall er GPU-arbeid, mens resize, unclip, boxes_from_bitmap,
    rot90 og ascontiguousarray er enkelttrådet CPU-arbeid som GPU-en må
    vente på. Trengs fordi py-spy krever nettverk for å installeres.
    """

    def __init__(self, intervall=0.01):
        self._intervall = intervall
        self._teller = {}
        self._hovedtråd = threading.get_ident()
        self._stopp = threading.Event()

    def start(self):
        threading.Thread(target=self._løkke, daemon=True).start()

    def _løkke(self):
        while not self._stopp.wait(self._intervall):
            ramme = sys._current_frames().get(self._hovedtråd)
            if ramme is None:
                continue
            kode = ramme.f_code
            nøkkel = f"{kode.co_name}  ({os.path.basename(kode.co_filename)}:{ramme.f_lineno})"
            self._teller[nøkkel] = self._teller.get(nøkkel, 0) + 1

    def stopp(self):
        self._stopp.set()

    def snapshot(self):
        return dict(self._teller)


# ── Arbeidsfordeling mellom GPU og CPU ───────────────────────────

class _Arbeidsfordeler:
    """Delt filliste som GPU-hovedprosessen og CPU-workerne henter fra.

    Begge sider tar neste ledige fil fra samme teller, så fordelingen
    balanserer seg selv: den raskeste siden tar flest filer. En fast
    fordeling gjettet på forhånd gir alltid en hale der den ene siden
    er ferdig og venter på den andre.

    Telleren er en delt mp.Value; fillisten i hver worker er en
    fork()-kopi. Derfor må CPU-workerne startes med fork-kontekst.
    """

    def __init__(self, filer, ctx=None):
        self._filer = filer
        self.totalt = len(filer)
        self._delt = ctx.Value("i", 0) if ctx is not None else None
        self._lokal = 0

    def hent(self):
        """Returner neste filsti, eller None når alle filer er utdelt."""
        if self._delt is None:
            if self._lokal >= self.totalt:
                return None
            sti = self._filer[self._lokal]
            self._lokal += 1
            return sti
        with self._delt.get_lock():
            i = self._delt.value
            if i >= self.totalt:
                return None
            self._delt.value = i + 1
        return self._filer[i]

    def tom(self):
        """True når det ikke er flere filer å dele ut."""
        if self._delt is None:
            return self._lokal >= self.totalt
        return self._delt.value >= self.totalt

    def utdelt(self):
        """Antall filer delt ut så langt."""
        return self._lokal if self._delt is None else self._delt.value


# ── Feilmeldinger ────────────────────────────────────────────────

def _kort_feil(e, maks=200):
    """Kort én-linjes sammendrag av en paddle-feil.

    Paddle sine minnefeil er 4-5 KB C++-traceback. Uforkortet fyller de
    loggen fullstendig når flere prosesser treffer taket samtidig.
    """
    tekst = " ".join(str(e).split())
    for markør in ("Error Message Summary:", "Error Message Summary"):
        if markør in tekst:
            tekst = tekst.split(markør)[-1]
            break
    tekst = tekst.strip(" -:")
    if not tekst:
        tekst = repr(e)
    return tekst[:maks]


def _er_minnefeil(e):
    tekst = str(e)
    return ("Out of memory" in tekst or "ResourceExhausted" in tekst
            or "OutOfMemory" in tekst or isinstance(e, MemoryError))


# ── OCR-motor per enhet ──────────────────────────────────────────

def _lag_motor(device, cpu_tråder, ocr_batch, gpu_andel=None):
    """Bygg (orienter_batch, les_tokens) for «gpu» eller «cpu».

    GPU-varianten gjenbruker app-modulenes lazy singletons — paddle velger
    GPU selv. CPU-varianten lager egne instanser med cpu_threads satt;
    uten den tar hver prosess 10 tråder (Paddle-default) og maskinen blir
    overtegnet.
    """
    if device == "gpu":
        # Hver prosess får sin egen andel av kortet. Uten dette vokser
        # allokatorene uavhengig av hverandre til kortet er fullt, og da
        # feiler den prosessen som tilfeldigvis ber om minne sist.
        # Må settes før paddle initialiserer GPU-allokatoren.
        os.environ["FLAGS_allocator_strategy"] = "auto_growth"
        if gpu_andel:
            os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = f"{gpu_andel:.4f}"

        from orientering import finn_rotasjoner_batch
        from paddle_ocr_model_fnr import les_tokens_batched

        def orienter_batch(bilder):
            return finn_rotasjoner_batch(bilder)

        def les_tokens(bilder, batch=None):
            return les_tokens_batched(bilder, batch_size=batch or ocr_batch)

        return orienter_batch, les_tokens

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # tving CPU-modus
    from config import NEDSKALERING, MIN_KONFIDENS, DET_SIDE_LEN, REC_BATCH
    from paddleocr import PaddleOCR, DocImgOrientationClassification
    from paddle_ocr_model_fnr import _les_tokens

    reader = PaddleOCR(
        lang="en",
        device="cpu",
        cpu_threads=cpu_tråder,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_type="max",
        text_det_limit_side_len=DET_SIDE_LEN,
        text_recognition_batch_size=REC_BATCH,
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        text_detection_model_dir=os.path.join(_APP, "PP-OCRv6_medium_det_infer"),
        text_recognition_model_dir=os.path.join(_APP, "PP-OCRv6_medium_rec_infer"),
        enable_mkldnn=True,
    )
    ori_kwargs = dict(
        model_name="PP-LCNet_x1_0_doc_ori",
        model_dir=os.path.join(_APP, "PP-LCNet_x1_0_doc_ori_infer"),
        device="cpu",
    )
    try:
        orient = DocImgOrientationClassification(cpu_threads=cpu_tråder, **ori_kwargs)
    except (TypeError, ValueError):
        orient = DocImgOrientationClassification(**ori_kwargs)

    def orienter_batch(bilder):
        if not bilder:
            return []
        lite = [np.ascontiguousarray(b[::NEDSKALERING, ::NEDSKALERING]) for b in bilder]
        try:
            resultater = orient.predict(lite)
        except Exception:
            return [0] * len(bilder)
        rotasjoner = []
        for r in resultater:
            vinkel = int(r["label_names"][0])
            score = float(np.asarray(r["scores"]).reshape(-1)[0])
            rotasjoner.append(0 if score < MIN_KONFIDENS else (vinkel // 90) % 4)
        while len(rotasjoner) < len(bilder):
            rotasjoner.append(0)
        return rotasjoner

    def les_tokens(bilder, batch=None):
        steg = batch or ocr_batch
        ut = []
        for start in range(0, len(bilder), steg):
            chunk = bilder[start:start + steg]
            bgr = [np.ascontiguousarray(b[:, :, ::-1]) for b in chunk]
            for res in (reader.predict(bgr, return_word_box=True) or []):
                ut.append(_les_tokens(res))
            while len(ut) < start + len(chunk):
                ut.append([])
        return ut

    return orienter_batch, les_tokens


# ── Pipeline-worker (én prosess = én komplett OCR-pipeline) ───────

def _worker(oppgave):
    """Prosess-inngang: steng utskrift, kjør pipelinen, rapporter alltid.

    PaddlePaddle skriver modell- og oneDNN-meldinger direkte fra C++ til
    fd 1/2. Med flere prosesser drukner loggen i det, så alt går via
    resultatkøen i stedet.
    """
    kø = oppgave["resultat_kø"]
    wid = oppgave["id"]
    _devnull = open(os.devnull, "w")
    try:
        os.dup2(_devnull.fileno(), 1)
        os.dup2(_devnull.fileno(), 2)
    except OSError:
        pass
    sys.stdout = _devnull
    sys.stderr = _devnull

    antall = 0
    try:
        antall = _pipeline(oppgave) or 0
    except BaseException as e:  # rapporteres — ellers dør prosessen stille
        import traceback
        kø.put(("worker-feil", wid, "", 0, f"{e!r}\n{traceback.format_exc()}"))
    finally:
        kø.put(("ferdig", wid, "", antall, ""))


def _pipeline(oppgave):
    """Rendr PDF-er i tråder, kjør orientering + OCR i batch, skriv cache.

    Identisk arbeid uansett enhet — flere slike prosesser mot samme GPU er
    hele poenget: mens én prosess står i enkelttrådet CPU-forbehandling,
    kan en annen bruke GPU-en.
    """
    wid = oppgave["id"]
    device = oppgave["device"]
    kø = oppgave["resultat_kø"]
    fordeler = oppgave["fordeler"]
    cache_mappe = oppgave["cache_mappe"]
    ocr_batch = oppgave["ocr_batch"]
    prefetch = oppgave["prefetch"]
    minne_grense = oppgave["minne_grense"]

    orienter_batch, les_tokens = _lag_motor(
        device, oppgave["cpu_tråder"], ocr_batch, oppgave.get("gpu_andel"))

    # Varm opp modellene — engangskostnad som ikke skal med i estimatene
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    orienter_batch([dummy])
    les_tokens([dummy])
    kø.put(("klar", wid, device, 0, ""))

    # Batchstørrelse. CPU-prosesser tar ett dokument om gangen; adaptiv
    # styring leser GPU-minne og er meningsløs der.
    if device == "cpu":
        adaptiv, maks_batch_fast = None, oppgave["dokument_batch"] or 1
    elif oppgave["dokument_batch"]:
        adaptiv, maks_batch_fast = None, oppgave["dokument_batch"]
    elif oppgave.get("gpu_andel"):
        # Hardt minnetak per prosess ⇒ minnefeil er signalet, ikke nvidia-smi
        adaptiv = _AIMDBatchStørrelse(
            start=2, minimum=1, maksimum=oppgave["maks_batch"])
        maks_batch_fast = None
    else:
        adaptiv = _AdaptivBatchStørrelse(
            start=2, minimum=1, maksimum=oppgave["maks_batch"],
            gpu_grense_prosent=oppgave["gpu_grense"])
        maks_batch_fast = None

    profil = _Profil() if oppgave["profil"] else None
    if profil:
        profil.start()

    executor = ThreadPoolExecutor(max_workers=oppgave["workers"])
    prefetch_kø = deque()

    def _fyll_kø():
        while len(prefetch_kø) < prefetch and _er_minne_trygt(minne_grense):
            sti = fordeler.hent()
            if sti is None:
                break
            prefetch_kø.append((sti, executor.submit(_last_pdf, sti)))

    tider = {"vente": 0.0, "orient": 0.0, "rot": 0.0, "ocr": 0.0, "cache": 0.0}

    def _behandle(gruppe, sider_om_gangen=None):
        """Orientering + rotasjon + OCR for en gruppe dokumenter."""
        alle_bilder, dok_grenser = [], []
        for _navn, bilder in gruppe:
            dok_grenser.append((len(alle_bilder), len(bilder)))
            alle_bilder.extend(bilder)

        t0 = time.perf_counter()
        ori_steg = sider_om_gangen or max(4 * ocr_batch, 8)
        alle_rotasjoner = []
        for i in range(0, len(alle_bilder), ori_steg):
            alle_rotasjoner.extend(orienter_batch(alle_bilder[i:i + ori_steg]))
        tider["orient"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        rotert = [np.rot90(b, k) if k else b
                  for b, k in zip(alle_bilder, alle_rotasjoner)]
        tider["rot"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            alle_tokens = les_tokens(rotert, batch=sider_om_gangen)
        finally:
            tider["ocr"] += time.perf_counter() - t0
            del rotert
        return dok_grenser, alle_rotasjoner, alle_tokens

    def _rapporter(gruppe, dok_grenser, rotasjoner, tokens, tid_batch):
        """Skriv cache og meld ferdige dokumenter på køen."""
        n = 0
        batch_sider = sum(s for _, s in dok_grenser)
        for idx, (navn, _bilder) in enumerate(gruppe):
            start_idx, n_sider = dok_grenser[idx]
            rot = rotasjoner[start_idx:start_idx + n_sider]
            tok = tokens[start_idx:start_idx + n_sider]

            t0 = time.perf_counter()
            skriv_ocr_cache(cache_mappe, navn, rot, tok)
            tider["cache"] += time.perf_counter() - t0

            n += 1
            kø.put(("ok", wid, navn, n_sider, {
                "tokens": sum(len(ts) for ts in tok),
                "rot": list(rot),
                "batch_dok": len(gruppe),
                "batch_sider": batch_sider,
                "batch_tid": tid_batch,
            }))
        return n

    _fyll_kø()

    ferdig = 0
    kø_dybde_sum = 0
    antall_batches = 0
    sist_rapport = 0

    def _stats(ferdig_nå):
        return ("stats", wid, device, ferdig_nå, dict(
            tider, kø=kø_dybde_sum / max(antall_batches, 1),
            batch=maks_batch_fast or (adaptiv.nåværende if adaptiv else 0),
            profil=profil.snapshot() if profil else None))

    while True:
        _fyll_kø()
        if not prefetch_kø:
            if not fordeler.tom():
                time.sleep(2)  # minnetak nådd — vent på ledig RAM
                _fyll_kø()
                if not prefetch_kø:
                    sti = fordeler.hent()
                    if sti is not None:
                        prefetch_kø.append((sti, executor.submit(_last_pdf, sti)))
            if not prefetch_kø:
                break

        maks_batch = maks_batch_fast or adaptiv.neste()
        kø_dybde_sum += len(prefetch_kø)
        antall_batches += 1

        batch_docs = []
        t0 = time.perf_counter()
        while prefetch_kø and len(batch_docs) < maks_batch:
            sti, fut = prefetch_kø.popleft()
            sti, resultat = fut.result()
            navn = os.path.basename(sti)
            if isinstance(resultat, Exception):
                kø.put(("feil", wid, navn, 0, _kort_feil(resultat)))
                continue
            batch_docs.append((navn, resultat))
            _fyll_kø()
        tider["vente"] += time.perf_counter() - t0

        if not batch_docs:
            continue

        # Kjør batchen. Tom-for-minne håndteres ved å halvere gruppen og
        # prøve igjen — et dokument skal ikke gå tapt fordi en annen
        # prosess tilfeldigvis holdt minnet i det øyeblikket.
        arbeid = [batch_docs]
        while arbeid:
            gruppe = arbeid.pop()
            start_gruppe = time.perf_counter()
            try:
                dok_grenser, rot, tok = _behandle(gruppe)
            except Exception as e:
                _frigjør_gpu_cache()
                minnefeil = _er_minnefeil(e)
                if minnefeil and adaptiv:
                    adaptiv.oom()
                if minnefeil and len(gruppe) > 1:
                    midt = len(gruppe) // 2
                    arbeid.append(gruppe[midt:])
                    arbeid.append(gruppe[:midt])
                    kø.put(("oom", wid, "", len(gruppe),
                            f"deler batchen (→ {midt} + {len(gruppe) - midt} dok)"))
                    continue
                if minnefeil:
                    # Ett dokument alene: prøv én side om gangen
                    try:
                        dok_grenser, rot, tok = _behandle(gruppe, sider_om_gangen=1)
                    except Exception as e2:
                        _frigjør_gpu_cache()
                        kø.put(("feil", wid, gruppe[0][0], 0, _kort_feil(e2)))
                        continue
                    kø.put(("oom", wid, gruppe[0][0], 1, "kjørt side for side"))
                else:
                    for navn, _b in gruppe:
                        kø.put(("feil", wid, navn, 0, _kort_feil(e)))
                    continue
            if adaptiv:
                adaptiv.ok()
            ferdig += _rapporter(gruppe, dok_grenser, rot, tok,
                                 time.perf_counter() - start_gruppe)

        # Frigjør cachet GPU-minne — holder minnebruken flat over tid
        _frigjør_gpu_cache()

        if ferdig - sist_rapport >= 20:
            sist_rapport = ferdig
            kø.put(_stats(ferdig))

    executor.shutdown(wait=True)
    if profil:
        profil.stopp()
    kø.put(_stats(ferdig))
    return ferdig


def main():
    _setup_loggfil()

    p = argparse.ArgumentParser(
        description="Forhåndscache PaddleOCR-resultater. Kjører kun orientering + OCR, "
                    "lagrer til OCR-cache. Ingen YOLO eller FNR-deteksjon.")
    p.add_argument("--mappe", required=True,
                   help="mappe med PDF-filer")
    p.add_argument("--cache", default=None,
                   help="mappe for OCR-cache (default: $SLADD_CACHE/<uttrekk>/ocr/)")
    p.add_argument("--velg", nargs="*", default=[],
                   help="spesifikke filer (filnavn/delstreng)")
    p.add_argument("--velg-fra-fil", default=None,
                   help="les fil-IDer fra en tekstfil (én per linje)")
    p.add_argument("--antall", default="alle",
                   help="antall filer når --velg er tom (tall, eller 'alle')")
    p.add_argument("--gpu-prosesser", type=int, nargs="?", default=1, const=-1,
                   metavar="N",
                   help="antall parallelle GPU-pipeliner mot samme kort "
                        "(1=som før, uten tall = auto). Hver prosess har sin "
                        "egen CPU-forbehandling, som er den reelle flaskehalsen")
    p.add_argument("--workers", type=int, default=0,
                   help="tråder for PDF-rendering PER prosess (0=auto)")
    p.add_argument("--prefetch", type=int, default=0,
                   help="pre-rendrede PDF-er i kø per prosess (0=auto fra RAM)")
    p.add_argument("--minne-grense", type=int, default=88,
                   help="maks RAM-bruk i prosent før pre-rendering pauser (default: 88)")
    p.add_argument("--ocr-batch", type=int, default=None,
                   help=f"sider per OCR-batch (default: {SIDER_PER_OCR_BATCH} fra config)")
    p.add_argument("--dokument-batch", type=int, default=0,
                   help="maks dokumenter per GPU-batch per prosess (0=adaptiv)")
    p.add_argument("--gpu-grense", type=int, default=75,
                   help="maks GPU-minnebruk i prosent før batchstørrelse reduseres (default: 75)")
    p.add_argument("--hpi", action="store_true",
                   help="aktiver High Performance Inference (TensorRT) — krever kraftig GPU")
    p.add_argument("--cpu-ocr", type=int, nargs="?", default=0, const=-1,
                   metavar="N",
                   help="antall CPU-OCR-prosesser i tillegg (0=ingen, uten tall = auto)")
    p.add_argument("--profil", action="store_true",
                   help="mål hvor prosessene bruker tiden (faser + stakk-prøver)")
    p.add_argument("--vis-ressurser", action="store_true",
                   help="vis RAM/GPU-status underveis")
    p.add_argument("--force", action="store_true",
                   help="kjør på nytt selv om dokumentet allerede er cachet")
    args = p.parse_args()

    # ── Valider input ────────────────────────────────────────────
    if not os.path.isdir(args.mappe):
        print(f"FEIL: --mappe finnes ikke: {args.mappe}")
        return 1

    if args.velg_fra_fil and not os.path.isfile(args.velg_fra_fil):
        print(f"FEIL: --velg-fra-fil finnes ikke: {args.velg_fra_fil}")
        return 1

    cache_mappe = _utled_cache_mappe(args)
    if not cache_mappe:
        print("FEIL: Ingen cache-mappe angitt. Bruk --cache eller sett $SLADD_CACHE "
              "(server-standard: /data2/cache).")
        return 1
    os.makedirs(cache_mappe, exist_ok=True)

    # ── Bygg filliste ────────────────────────────────────────────
    velg = args.velg
    fra_fil = False
    if args.velg_fra_fil:
        with open(args.velg_fra_fil, encoding="utf-8") as f:
            velg = [linje.strip() for linje in f if linje.strip()]
        fra_fil = True
        print(f"Leste {len(velg)} IDer fra {args.velg_fra_fil}")

    filer = velg_filer(args.mappe, velg, args.antall, eksakt=fra_fil)

    if not filer:
        print("Ingen filer å behandle — sjekk --mappe / --velg / --antall.")
        return 1

    print(f"Filer funnet: {len(filer)}")
    print(f"Cache-mappe:  {cache_mappe}")

    if not args.force:
        opprinnelig = len(filer)
        filer = _filtrer_cachet(filer, cache_mappe)
        hoppet = opprinnelig - len(filer)
        if hoppet:
            print(f"Hopper over: {hoppet} allerede cachet, {len(filer)} gjenstår")

    if not filer:
        print("Alle dokumenter er allerede cachet!")
        return 0

    # ── Antall prosesser ─────────────────────────────────────────
    kjerner = _antall_cpu_kjerner()
    n_gpu = args.gpu_prosesser
    if n_gpu < 0:
        # Hver pipeline trenger ~1 kjerne til forbehandling + render-tråder.
        # Hold av litt til koordinatoren og la GPU-minnet være taket.
        n_gpu = max(min(kjerner // 12, 6), 1)
    n_gpu = max(n_gpu, 1)

    n_cpu = args.cpu_ocr
    if n_cpu < 0:
        n_cpu = max(min((kjerner - 6 * n_gpu) // 8, 8), 0)

    # ── OCR-batch ────────────────────────────────────────────────
    if args.ocr_batch:
        ocr_batch = args.ocr_batch
    else:
        gpu = _gpu_minne_info()
        if gpu and gpu[1] >= 24000:
            ocr_batch = 32
        elif gpu and gpu[1] >= 16000:
            ocr_batch = 16
        else:
            ocr_batch = SIDER_PER_OCR_BATCH
        if n_gpu > 1:  # del GPU-minnet mellom prosessene
            ocr_batch = max(ocr_batch // n_gpu, SIDER_PER_OCR_BATCH)

    # ── Tråder og prefetch per prosess ───────────────────────────
    # Del GPU-minnet mellom prosessene. Uten dette vokser allokatorene
    # uavhengig til kortet er fullt (målt: 30,9 av 32 GB), og da feiler
    # den prosessen som tilfeldigvis ber om minne sist.
    gpu_andel = None
    if n_gpu > 1:
        gpu = _gpu_minne_info()
        if gpu and gpu[1] > 0:
            # ~700 MB per prosess går til CUDA-kontekst og modellvekter
            til_deling = gpu[1] * args.gpu_grense / 100 - 700 * n_gpu
            gpu_andel = max(til_deling / gpu[1] / n_gpu, 0.05)
        else:
            gpu_andel = (args.gpu_grense / 100) / n_gpu

    workers = args.workers or min(max(kjerner // (4 * n_gpu), 2), 16)
    prefetch = args.prefetch or max(_auto_prefetch() // n_gpu, 8)
    # Med hardt minnetak per prosess trenger ikke taket være stramt —
    # AIMD-regulatoren og omkjøringen håndterer resten.
    maks_batch = 12 if gpu_andel else max(16 // n_gpu, 4)

    if args.hpi:
        os.environ["SLADD_HPI"] = "1"

    print(f"  GPU-pipeliner: {n_gpu} × (1 hovedtråd + {workers} render-tråder), "
          f"prefetch {prefetch}/prosess")
    if gpu_andel:
        gpu = _gpu_minne_info()
        mb = gpu[1] * gpu_andel if gpu else 0
        print(f"  GPU-minne:     {100 * gpu_andel:.0f}% per prosess"
              + (f" (~{mb:.0f} MB)" if mb else ""))
    if args.dokument_batch:
        print(f"  Dokument-batch: fast {args.dokument_batch} per prosess")
    elif gpu_andel:
        print(f"  Dokument-batch: AIMD (vokser til {maks_batch}, halveres ved "
              f"minnefeil)")
    else:
        print(f"  Dokument-batch: adaptiv (GPU-grense {args.gpu_grense}%, "
              f"maks {maks_batch} per prosess)")

    cpu_tråder = 1
    if n_cpu:
        ledige = max(kjerner - n_gpu * (workers + 1), n_cpu)
        cpu_tråder = max(ledige // n_cpu, 1)
        print(f"  CPU-OCR:       {n_cpu} prosesser × {cpu_tråder} tråder")
    print(f"  OCR-batch: {ocr_batch} sider  |  filene deles dynamisk mellom "
          f"{n_gpu + n_cpu} prosess(er)")
    _skriv_ressursstatus()

    # ── Start prosessene ─────────────────────────────────────────
    ctx = mp.get_context("fork")   # fordeleren deles via fork
    fordeler = _Arbeidsfordeler(filer, ctx=ctx)
    kø = ctx.Queue()

    felles = dict(
        resultat_kø=kø, fordeler=fordeler, cache_mappe=cache_mappe,
        workers=workers, prefetch=prefetch, minne_grense=args.minne_grense,
        ocr_batch=ocr_batch, dokument_batch=args.dokument_batch,
        gpu_grense=args.gpu_grense, maks_batch=maks_batch, profil=args.profil,
        gpu_andel=gpu_andel,
    )
    prosesser = []
    for i in range(n_gpu + n_cpu):
        device = "gpu" if i < n_gpu else "cpu"
        navn = f"{'g' if device == 'gpu' else 'c'}{i if device == 'gpu' else i - n_gpu}"
        oppgave = dict(felles, id=navn, device=device,
                       cpu_tråder=cpu_tråder if device == "cpu" else 1)
        pr = ctx.Process(target=_worker, args=(oppgave,), daemon=True)
        pr.start()
        prosesser.append(pr)

    totalt = len(filer)
    print(f"\nStarter OCR-caching av {totalt} dokumenter "
          f"({n_gpu} GPU + {n_cpu} CPU). Laster modeller...\n")

    # ── Koordinator: samle resultater og skriv framdrift ─────────
    ferdig = 0
    total_sider = 0
    feilet = []
    per_prosess = {}
    stats = {}
    klar = 0
    avsluttet = 0
    oom_hendelser = 0
    start_alle = None
    neste_status = 20
    neste_ressurs = 100

    try:
        while avsluttet < len(prosesser):
            try:
                status, wid, navn, n, data = kø.get(timeout=30)
            except Exception:
                if all(not pr.is_alive() for pr in prosesser):
                    print("!! Alle prosesser er borte — avslutter")
                    break
                continue

            if status == "klar":
                klar += 1
                print(f"  [{wid}] modeller lastet ({navn})")
                if klar == len(prosesser):
                    start_alle = time.perf_counter()
                    print(f"\nAlle {klar} prosesser i gang.\n")
                continue

            if status == "feil":
                feilet.append((navn, data))
                ferdig += 1
                print(f"[{ferdig}/{totalt}] ✗ {navn} [{wid}]: {data}")
                continue

            if status == "oom":
                oom_hendelser += 1
                print(f"  [{wid}] GPU tom for minne på {n} dok — {data}")
                continue

            if status == "worker-feil":
                print(f"  [{wid}] KRASJET: {data}")
                continue

            if status == "ferdig":
                avsluttet += 1
                print(f"  [{wid}] avsluttet — {n} dokumenter")
                continue

            if status == "stats":
                stats[wid] = data
                continue

            # status == "ok"
            if start_alle is None:
                start_alle = time.perf_counter()
            ferdig += 1
            total_sider += n
            teller = per_prosess.setdefault(wid, [0, 0])
            teller[0] += 1
            teller[1] += n

            elapsed = time.perf_counter() - start_alle
            snitt = elapsed / max(ferdig, 1)
            rot = data["rot"]
            rot_str = ""
            if any(k != 0 for k in rot):
                rot_str = f" rot=[{','.join(str(k * 90) + '°' for k in rot)}]"
            print(f"[{ferdig}/{totalt}] ✓ {navn} [{wid}]: "
                  f"{n} side(r), {data['tokens']} tokens{rot_str} — "
                  f"batch {data['batch_dok']} dok/{data['batch_sider']} sider "
                  f"på {data['batch_tid']:.1f}s "
                  f"(gått: {_fmt_tid(elapsed)}, "
                  f"gjenstår: {_fmt_tid(snitt * (totalt - ferdig))})")

            if ferdig >= neste_status:
                neste_status += 20
                _skriv_status(stats, per_prosess, elapsed, ferdig, total_sider,
                              args.profil)
            if args.vis_ressurser and ferdig >= neste_ressurs:
                neste_ressurs += 100
                _skriv_ressursstatus()
    except KeyboardInterrupt:
        print("\n!! Avbrutt — skriver oppsummering for det som er gjort")

    for pr in prosesser:
        pr.join(timeout=5)

    # ── Oppsummering ─────────────────────────────────────────────
    vegg_tid = time.perf_counter() - (start_alle or time.perf_counter())
    print(f"\n{'=' * 60}")
    print(f"Ferdig! {ferdig}/{totalt} dokumenter, {total_sider} sider")
    for wid in sorted(per_prosess):
        dok, sider = per_prosess[wid]
        print(f"  [{wid}] {dok} dok, {sider} sider "
              f"({100 * dok / max(ferdig, 1):.0f}%)")
    if oom_hendelser:
        print(f"  Tom-for-minne: {oom_hendelser} ganger (batchen ble delt og "
              f"kjørt om — ingen dokumenter tapt av det)")
    print(f"  Vegg-tid:   {_fmt_tid(vegg_tid)}")
    if vegg_tid > 0:
        print(f"  Throughput: {total_sider / vegg_tid:.2f} sider/s, "
              f"{ferdig / vegg_tid * 3600:.0f} dok/time")
    _skriv_status(stats, per_prosess, vegg_tid, ferdig, total_sider,
                  args.profil, n_profil=20)
    print(f"  Cache-mappe: {cache_mappe}")

    if feilet:
        print(f"\nFeilet ({len(feilet)}):")
        for navn, feil in feilet[:10]:
            print(f"  {navn}: {feil}")
        if len(feilet) > 10:
            print(f"  ... og {len(feilet) - 10} til")

    _skriv_ressursstatus()
    return 1 if feilet else 0


def _skriv_status(stats, per_prosess, elapsed, ferdig, sider, med_profil,
                  n_profil=12):
    """Skriv fasefordeling per prosess og samlet gjennomstrømning."""
    if not stats:
        return
    gpu_mem = _gpu_prosent_brukt()
    gpu_str = f" | GPU-minne: {gpu_mem:.0f}%" if gpu_mem else ""
    if elapsed > 0:
        print(f"  ⚡ {sider / elapsed:.2f} sider/s | "
              f"{ferdig / elapsed * 3600:.0f} dok/time{gpu_str}")
    for wid in sorted(stats):
        d = stats[wid]
        sum_fase = d["orient"] + d["rot"] + d["ocr"] + d["cache"]
        if sum_fase <= 0:
            continue
        print(f"     [{wid}] batch={d['batch']} kø={d['kø']:.0f} | "
              f"vente-render {100 * d['vente'] / (sum_fase + d['vente']):.0f}% | "
              f"orientering {100 * d['orient'] / sum_fase:.0f}% | "
              f"rotasjon {100 * d['rot'] / sum_fase:.0f}% | "
              f"OCR {100 * d['ocr'] / sum_fase:.0f}% | "
              f"cache {100 * d['cache'] / sum_fase:.0f}%")
    if not med_profil:
        return
    samlet = {}
    for d in stats.values():
        for navn, ant in (d.get("profil") or {}).items():
            samlet[navn] = samlet.get(navn, 0) + ant
    totalt = sum(samlet.values())
    if not totalt:
        return
    print(f"     Profil (alle prosesser) — {totalt} prøver:")
    for navn, ant in sorted(samlet.items(), key=lambda kv: -kv[1])[:n_profil]:
        print(f"       {100 * ant / totalt:5.1f}%  {navn}")


if __name__ == "__main__":
    sys.exit(main() or 0)
