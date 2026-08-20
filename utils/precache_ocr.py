"""
Forhåndscache PaddleOCR-resultater for et sett med dokumenter.

Kjører kun PaddleOCR (orientering + tekstgjenkjenning) og lagrer
resultatene i OCR-cachen. Ingen YOLO, ingen FNR-deteksjon, ingen
evaluering — bare den tyngste GPU-operasjonen isolert.

Optimalisert for å bruke så mye ressurser som mulig uten å krasje:
  - Pre-rendrer neste PDF(er) mens GPU jobber
  - Overvåker minne og tilpasser pipeline-dybden
  - Kan parallellisere PDF-rendering (CPU) med OCR (GPU)
  - Kan kjøre CPU-OCR i egne prosesser ved siden av GPU-en (--cpu-ocr)
  - Hopper automatisk over allerede cachede dokumenter

Bruk:
    python precache_ocr.py --mappe /sti/til/pdfer
    python precache_ocr.py --mappe /sti/til/pdfer --cache /sti/til/cache
    python precache_ocr.py --mappe /sti/til/pdfer --workers 4 --antall alle
    python precache_ocr.py --velg-fra-fil filer.txt --mappe /sti/til/pdfer
    python precache_ocr.py --mappe /sti/til/pdfer --cpu-ocr 10
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from config import PDF_DPI, SIDER_PER_OCR_BATCH
from file_selection import velg_filer
from load_pdf import les_sider
from ocr_cache import les_cache as les_ocr_cache, skriv_cache as skriv_ocr_cache
from orientering import finn_rotasjon, finn_rotasjoner_batch
from paddle_ocr_model_fnr import les_tokens_batched


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
    try:
        import paddle
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            # Paddle/CUDA bruker torch-aktig API ikke direkte, prøv nvidia-smi
            pass
    except Exception:
        pass
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


def _auto_workers():
    """Velg antall PDF-rendering-tråder basert på CPU-kjerner.

    PDF-rendering (PyMuPDF) frigir GIL og kan bruke ekte parallellisme.
    Bruk en fjerdedel av kjernene — nok til å holde GPU-en mettet uten
    å konkurrere om I/O. Min 2, maks 16.
    """
    kjerner = _antall_cpu_kjerner()
    return min(max(kjerner // 4, 2), 16)


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

    def neste(self):
        """Returner neste batchstørrelse, justert basert på GPU-minne."""
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

    def __repr__(self):
        gpu_str = f" GPU:{self._forrige_gpu_pct:.0f}%" if self._forrige_gpu_pct else ""
        return f"dokument_batch={self.nåværende}{gpu_str}"


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


# ── CPU OCR-worker (multiprocessing) ─────────────────────────────

def _cpu_worker(fordeler, resultat_kø, cache_mappe, worker_id, tråder_per_worker):
    """Prosess som kjører PaddleOCR på CPU for ett dokument om gangen.

    Wrapper som stenger av utskrift og rapporterer krasj tilbake på køen.
    PaddlePaddle skriver oneDNN-meldinger (ReduceMeanCheckIfOneDNNSupport
    o.l.) direkte fra C++ til fd 1/2 — med flere workers drukner loggen i
    det. Alt vi trenger å vite går via resultat_kø.
    """
    _devnull = open(os.devnull, "w")
    try:
        os.dup2(_devnull.fileno(), 1)
        os.dup2(_devnull.fileno(), 2)
    except OSError:
        pass
    sys.stdout = _devnull
    sys.stderr = _devnull

    try:
        _cpu_worker_kjør(fordeler, resultat_kø, cache_mappe, worker_id,
                         tråder_per_worker)
    except BaseException as e:  # rapporteres på køen — ellers dør workeren stille
        import traceback
        resultat_kø.put(("worker-feil", worker_id, "", 0,
                         f"{e!r}\n{traceback.format_exc()}"))
        resultat_kø.put(("ferdig", worker_id, "", 0, ""))


def _cpu_worker_kjør(fordeler, resultat_kø, cache_mappe, worker_id, tråder_per_worker):
    """Selve arbeidsløkken: hent fil, OCR på CPU, skriv cache, rapporter."""
    # Trådbegrensning. Merk: OMP_NUM_THREADS o.l. har ingen effekt her —
    # OpenMP-runtimet er allerede initialisert i foreldreprosessen og
    # arves gjennom fork(). Den virksomme knappen er cpu_threads under,
    # som Paddle setter med omp_set_num_threads() ved oppstart av
    # prediktoren. Uten den bruker HVER worker 10 tråder (Paddle-default)
    # og maskinen blir overtegnet: alt stopper nesten helt.
    tr = str(tråder_per_worker)
    os.environ["OMP_NUM_THREADS"] = tr
    os.environ["MKL_NUM_THREADS"] = tr
    os.environ["OPENBLAS_NUM_THREADS"] = tr
    os.environ["VECLIB_MAXIMUM_THREADS"] = tr
    os.environ["NUMEXPR_NUM_THREADS"] = tr

    # Sett opp paths
    _utils = os.path.dirname(os.path.abspath(__file__))
    _app = os.path.join(_utils, "..", "app")
    if _utils not in sys.path:
        sys.path.insert(0, _utils)
    if _app not in sys.path:
        sys.path.insert(0, _app)

    os.environ["GLOG_minloglevel"] = "3"
    os.environ["GLOG_v"] = "0"
    os.environ["FLAGS_call_stack_level"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # tving CPU-modus

    import numpy as np
    from load_pdf import les_sider
    from ocr_cache import skriv_cache as skriv_ocr_cache
    from config import NEDSKALERING, MIN_KONFIDENS, DET_SIDE_LEN, REC_BATCH, SIDER_PER_OCR_BATCH
    from paddleocr import PaddleOCR, DocImgOrientationClassification

    # Opprett CPU-basert PaddleOCR
    det_dir = os.path.join(_app, "PP-OCRv6_medium_det_infer")
    rec_dir = os.path.join(_app, "PP-OCRv6_medium_rec_infer")
    ori_dir = os.path.join(_app, "PP-LCNet_x1_0_doc_ori_infer")

    reader = PaddleOCR(
        lang="en",
        device="cpu",
        cpu_threads=tråder_per_worker,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_type="max",
        text_det_limit_side_len=DET_SIDE_LEN,
        text_recognition_batch_size=REC_BATCH,
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        text_detection_model_dir=det_dir,
        text_recognition_model_dir=rec_dir,
        enable_mkldnn=True,
    )
    ori_kwargs = dict(
        model_name="PP-LCNet_x1_0_doc_ori",
        model_dir=ori_dir,
        device="cpu",
    )
    try:
        orient = DocImgOrientationClassification(cpu_threads=tråder_per_worker,
                                                 **ori_kwargs)
    except (TypeError, ValueError):
        orient = DocImgOrientationClassification(**ori_kwargs)

    from paddle_ocr_model_fnr import _les_tokens

    def _orientering(bilde):
        lite = np.ascontiguousarray(bilde[::NEDSKALERING, ::NEDSKALERING])
        try:
            res = orient.predict(lite)
            r = res[0]
            vinkel = int(r["label_names"][0])
            score = float(np.asarray(r["scores"]).reshape(-1)[0])
        except Exception:
            return 0
        if score < MIN_KONFIDENS:
            return 0
        return (vinkel // 90) % 4

    resultat_kø.put(("klar", worker_id, "", 0, ""))

    ferdig = 0
    while True:
        sti = fordeler.hent()
        if sti is None:  # ingen filer igjen — GPU-siden tok resten
            break

        navn = os.path.basename(sti)
        t0 = time.perf_counter()
        try:
            bilder = les_sider(sti)
            rotasjoner = [_orientering(b) for b in bilder]
            bilder_ocr = [np.rot90(b, k) if k else b for b, k in zip(bilder, rotasjoner)]

            tokens_per_side = []
            for start in range(0, len(bilder_ocr), SIDER_PER_OCR_BATCH):
                chunk = bilder_ocr[start:start + SIDER_PER_OCR_BATCH]
                bgr_chunk = [np.ascontiguousarray(b[:, :, ::-1]) for b in chunk]
                resultater = reader.predict(bgr_chunk, return_word_box=True) or []
                for res in resultater:
                    tokens_per_side.append(_les_tokens(res))
                while len(tokens_per_side) < start + len(chunk):
                    tokens_per_side.append([])

            skriv_ocr_cache(cache_mappe, navn, rotasjoner, tokens_per_side)
            ferdig += 1
            n_tokens = sum(len(ts) for ts in tokens_per_side)
            resultat_kø.put(("ok", worker_id, navn, len(bilder),
                             (n_tokens, time.perf_counter() - t0)))
        except Exception as e:
            resultat_kø.put(("feil", worker_id, navn, 0, repr(e)))

    resultat_kø.put(("ferdig", worker_id, "", ferdig, ""))


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
    p.add_argument("--workers", type=int, default=0,
                   help="antall tråder for PDF-rendering (0=auto fra CPU-kjerner)")
    p.add_argument("--prefetch", type=int, default=0,
                   help="antall PDF-er å pre-rendre i kø (0=auto fra RAM)")
    p.add_argument("--minne-grense", type=int, default=88,
                   help="maks RAM-bruk i prosent før pre-rendering pauser (default: 88)")
    p.add_argument("--ocr-batch", type=int, default=None,
                   help=f"sider per OCR-batch (default: {SIDER_PER_OCR_BATCH} fra config)")
    p.add_argument("--dokument-batch", type=int, default=0,
                   help="maks dokumenter per GPU-batch (0=adaptiv basert på GPU-minne)")
    p.add_argument("--gpu-grense", type=int, default=75,
                   help="maks GPU-minnebruk i prosent før batchstørrelse reduseres (default: 75)")
    p.add_argument("--hpi", action="store_true",
                   help="aktiver High Performance Inference (TensorRT) — krever kraftig GPU")
    p.add_argument("--cpu-ocr", type=int, nargs="?", default=0, const=-1,
                   metavar="N",
                   help="antall ekstra CPU-prosesser for OCR (0=kun GPU, "
                        "--cpu-ocr uten tall = auto ut fra ledige kjerner)")
    p.add_argument("--vis-ressurser", action="store_true",
                   help="vis RAM/GPU-status for hvert dokument")
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

    # ── Filtrer allerede cachede ─────────────────────────────────
    if not args.force:
        opprinnelig = len(filer)
        filer = _filtrer_cachet(filer, cache_mappe)
        hoppet = opprinnelig - len(filer)
        if hoppet:
            print(f"Hopper over: {hoppet} allerede cachet, {len(filer)} gjenstår")

    if not filer:
        print("Alle dokumenter er allerede cachet!")
        return 0

    # ── Overstyr OCR-batch fra CLI ───────────────────────────────
    if args.ocr_batch:
        ocr_batch = args.ocr_batch
    else:
        # Auto-skalér basert på GPU-minne. Mer ledig GPU = større batches.
        gpu = _gpu_minne_info()
        if gpu and gpu[1] >= 24000:       # 24GB+ GPU
            ocr_batch = 32
        elif gpu and gpu[1] >= 16000:     # 16GB+ GPU
            ocr_batch = 16
        else:
            ocr_batch = SIDER_PER_OCR_BATCH

    # ── Resolve auto-verdier ──────────────────────────────────────
    workers = args.workers or _auto_workers()
    prefetch = args.prefetch or _auto_prefetch()
    dokument_batch_fast = args.dokument_batch  # 0 = adaptiv

    if dokument_batch_fast:
        adaptiv_batch = None
        print(f"  Dokument-batch: fast {dokument_batch_fast}")
    else:
        adaptiv_batch = _AdaptivBatchStørrelse(
            start=4, minimum=2, maksimum=16, gpu_grense_prosent=args.gpu_grense)
        print(f"  Dokument-batch: adaptiv (GPU-grense {args.gpu_grense}%, "
              f"start=4, maks=16)")

    # ── HPI-modus (TensorRT) ──────────────────────────────────────
    if args.hpi:
        os.environ["SLADD_HPI"] = "1"

    # ── CPU OCR-workers (parallelt med GPU) ───────────────────────
    # Workerne må startes FØR GPU-oppvarmingen: de forkes fra denne
    # prosessen, og en fork etter at CUDA er initialisert gir en ubrukelig
    # CUDA-kontekst i barnet.
    cpu_ocr = args.cpu_ocr
    if cpu_ocr < 0:  # --cpu-ocr uten tall = auto
        ledige = _antall_cpu_kjerner() - workers - 2
        cpu_ocr = max(min(ledige // 6, 12), 1)
    cpu_prosesser = []
    cpu_resultat_kø = None

    if cpu_ocr > 0:
        mp_ctx = mp.get_context("fork")  # fordeleren deles via fork
        fordeler = _Arbeidsfordeler(filer, ctx=mp_ctx)
        cpu_resultat_kø = mp_ctx.Queue()

        # Fordel kjerner: reserver noen til GPU-pipelinen (rendering + main).
        # tråder_per sendes videre som cpu_threads til Paddle — uten det
        # tar hver worker 10 tråder og maskinen blir overtegnet.
        kjerner_til_cpu = max(_antall_cpu_kjerner() - workers - 2, cpu_ocr)
        tråder_per = max(kjerner_til_cpu // cpu_ocr, 1)

        for i in range(cpu_ocr):
            p_worker = mp_ctx.Process(
                target=_cpu_worker,
                args=(fordeler, cpu_resultat_kø, cache_mappe, i, tråder_per),
                daemon=True
            )
            p_worker.start()
            cpu_prosesser.append(p_worker)

        print(f"  CPU OCR: {cpu_ocr} prosesser × {tråder_per} tråder = "
              f"{cpu_ocr * tråder_per} kjerner")
        print(f"  Filer deles dynamisk mellom GPU og CPU "
              f"({len(filer)} totalt) — raskeste side tar flest")
    else:
        fordeler = _Arbeidsfordeler(filer)

    # ── Skriv ressursstatus før start ────────────────────────────
    _skriv_ressursstatus()
    print(f"\nStarter OCR-caching av {len(filer)} dokumenter "
          f"({'GPU + ' + str(len(cpu_prosesser)) + ' CPU-workers' if cpu_prosesser else 'GPU'}) "
          f"(workers={workers}, prefetch={prefetch}, "
          f"ocr_batch={ocr_batch}, "
          f"minne_grense={args.minne_grense}%)\n")

    # ── Pipeline: pre-rendr PDF-er i tråder, OCR på GPU ──────────
    totalt = len(filer)
    ferdig = 0
    feilet = []
    total_sider = 0
    total_tid = 0

    # Warm up modellene med å kjøre orientering + OCR på et dummy-bilde
    print("Varmer opp PaddleOCR-modellen...")
    _warmup_start = time.perf_counter()
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    finn_rotasjon(dummy)
    les_tokens_batched([dummy])
    print(f"Oppvarming ferdig ({time.perf_counter() - _warmup_start:.1f}s)\n")

    # Start klokken ETTER warmup — warmup er engangskostnad og skal ikke
    # inflate estimatet for gjenstående tid.
    start_alle = time.perf_counter()

    # Prefetch-kø: rendre PDF-er i bakgrunnstråder
    render_executor = ThreadPoolExecutor(max_workers=workers)
    prefetch_kø = deque()  # inneholder Future-objekter for rendrede PDF-er

    def _fyll_kø():
        """Send filer til rendering så lenge køen ikke er full og minne er OK."""
        while len(prefetch_kø) < prefetch and _er_minne_trygt(args.minne_grense):
            sti = fordeler.hent()
            if sti is None:
                break
            prefetch_kø.append((sti, render_executor.submit(_last_pdf, sti)))

    # ── CPU-worker-resultater ─────────────────────────────────────
    cpu_ferdig = 0
    cpu_feilet = 0
    cpu_sider = 0
    cpu_tid = 0.0
    cpu_avsluttet = 0

    def _drener_cpu(blokkerende=False):
        """Hent resultater fra CPU-workerne.

        Må gjøres jevnlig underveis: resultatkøen ligger på en pipe med
        ~64 KB buffer, og en full pipe stopper workerne helt.
        """
        nonlocal cpu_ferdig, cpu_feilet, cpu_sider, cpu_tid, cpu_avsluttet
        if cpu_resultat_kø is None:
            return
        while cpu_avsluttet < len(cpu_prosesser):
            try:
                msg = (cpu_resultat_kø.get(timeout=30) if blokkerende
                       else cpu_resultat_kø.get_nowait())
            except Exception:
                if not blokkerende:
                    return
                if all(not pr.is_alive() for pr in cpu_prosesser):
                    return
                continue

            status, wid, navn, n, ekstra = msg
            if status == "ok":
                cpu_ferdig += 1
                cpu_sider += n
                if isinstance(ekstra, tuple):
                    cpu_tid += ekstra[1]
                if cpu_ferdig % 25 == 0:
                    snitt_cpu = cpu_tid / max(cpu_ferdig, 1)
                    print(f"  [CPU] {cpu_ferdig} dok, {cpu_sider} sider "
                          f"({snitt_cpu:.1f}s/dok per worker)")
            elif status == "feil":
                cpu_feilet += 1
                feilet.append((navn, ekstra))
            elif status == "klar":
                print(f"  [CPU {wid}] modeller lastet — jobber")
            elif status == "worker-feil":
                print(f"  [CPU {wid}] KRASJET: {ekstra}")
            elif status == "ferdig":
                cpu_avsluttet += 1
                print(f"  [CPU {wid}] avsluttet — {n} dokumenter")

    # Start pre-rendering
    _fyll_kø()

    # ── Bottleneck-tracking ───────────────────────────────────────
    tid_vente_cpu = 0.0   # tid brukt på å vente på at CPU rendrer ferdig
    tid_gpu = 0.0         # tid brukt på GPU (orientering + OCR)
    tid_io = 0.0          # tid brukt på cache-skriving
    antall_batches = 0
    kø_dybde_sum = 0      # sum av kødybde ved batch-start (for snitt)
    neste_status = 20     # neste dokument-tall som utløser flaskehals-print

    while True:
        # Hent unna CPU-workernes resultater før neste GPU-batch
        _drener_cpu()

        # Sørg for at køen er fylt opp
        _fyll_kø()

        # Hvis køen er tom men vi har flere filer, vent litt og prøv igjen
        if not prefetch_kø:
            if not fordeler.tom():
                # Minne er for høyt, vent til det frigjøres
                print("  ⏸  Venter på ledig minne for pre-rendering...")
                time.sleep(2)
                _fyll_kø()
                if not prefetch_kø:
                    # Tving gjennom én fil uansett
                    sti = fordeler.hent()
                    if sti is not None:
                        prefetch_kø.append(
                            (sti, render_executor.submit(_last_pdf, sti)))
            if not prefetch_kø:
                break  # alle filer er delt ut og rendret

        # ── Samle en batch med dokumenter for GPU-prosessering ────
        # Bestem batchstørrelse: fast eller adaptiv
        if dokument_batch_fast:
            maks_batch = dokument_batch_fast
        else:
            maks_batch = adaptiv_batch.neste()

        # Mål kødybde (indikator: er CPU foran eller bak GPU?)
        kø_dybde_sum += len(prefetch_kø)
        antall_batches += 1

        batch_docs = []  # [(navn, bilder), ...]
        start_vente = time.perf_counter()
        while prefetch_kø and len(batch_docs) < maks_batch:
            sti, fut = prefetch_kø.popleft()
            sti, resultat = fut.result()  # kan blokkere hvis CPU ikke er ferdig
            navn = os.path.basename(sti)

            if isinstance(resultat, Exception):
                feilet.append((navn, repr(resultat)))
                ferdig += 1
                print(f"[{ferdig + cpu_ferdig}/{totalt}] ✗ {navn}: {resultat!r}")
                continue
            batch_docs.append((navn, resultat))
            # Fyll køen mens vi venter på futures
            _fyll_kø()
        tid_vente_cpu += time.perf_counter() - start_vente

        if not batch_docs:
            continue

        start_batch = time.perf_counter()

        try:
            # ── 1. Samle alle sider fra alle dokumenter ───────────
            alle_bilder = []
            dok_grenser = []  # (start_idx, antall_sider) per dokument
            for navn, bilder in batch_docs:
                dok_grenser.append((len(alle_bilder), len(bilder)))
                alle_bilder.extend(bilder)

            # ── 2. Batch-orienteringsdeteksjon (alle sider samlet) ─
            alle_rotasjoner = finn_rotasjoner_batch(alle_bilder)

            # ── 3. Roter alle bilder ──────────────────────────────
            alle_bilder_ocr = [np.rot90(b, k) if k else b
                               for b, k in zip(alle_bilder, alle_rotasjoner)]

            # ── 4. Kjør PaddleOCR på hele batchen (GPU) ───────────
            alle_tokens = les_tokens_batched(alle_bilder_ocr, batch_size=ocr_batch)

            # Frigjør minne
            del alle_bilder, alle_bilder_ocr

            # ── 5. Splitt resultater tilbake per dokument og lagre ─
            for dok_idx, (navn, bilder) in enumerate(batch_docs):
                start_idx, n_sider = dok_grenser[dok_idx]
                rotasjoner = alle_rotasjoner[start_idx:start_idx + n_sider]
                tokens_per_side = alle_tokens[start_idx:start_idx + n_sider]

                skriv_ocr_cache(cache_mappe, navn, rotasjoner, tokens_per_side)
                total_sider += n_sider
                ferdig += 1

                n_tokens = sum(len(ts) for ts in tokens_per_side)
                tid_batch = time.perf_counter() - start_batch
                elapsed = time.perf_counter() - start_alle
                # Framdrift og ETA måles mot ALLE filer — CPU-workerne
                # jobber på samme filliste, så bare GPU-tellingen ville
                # gitt et estimat som er langt for pessimistisk.
                ferdig_totalt = ferdig + cpu_ferdig
                snitt = elapsed / max(ferdig_totalt, 1)
                gjenstår = snitt * (totalt - ferdig_totalt)

                rot_str = ""
                if any(k != 0 for k in rotasjoner):
                    rot_str = f" rot=[{','.join(str(k * 90) + '°' for k in rotasjoner)}]"

                print(f"[{ferdig_totalt}/{totalt}] ✓ {navn}: "
                      f"{n_sider} side(r), {n_tokens} tokens{rot_str} — "
                      f"batch {len(batch_docs)} dok/{sum(n for _, n in dok_grenser)} sider "
                      f"på {tid_batch:.1f}s "
                      f"(gått: {_fmt_tid(elapsed)}, gjenstår: {_fmt_tid(gjenstår)})")

        except Exception as e:
            # Hele batchen feilet — logg alle dokumenter
            for navn, _bilder in batch_docs:
                feilet.append((navn, repr(e)))
                ferdig += 1
                print(f"[{ferdig + cpu_ferdig}/{totalt}] ✗ {navn}: {e!r}")
            import traceback
            traceback.print_exc()
            continue

        tid_batch = time.perf_counter() - start_batch
        total_tid += tid_batch
        tid_gpu += tid_batch

        # Frigjør ubrukt cachet GPU-minne — forhindrer den gradvise veksten
        # som PaddlePaddle sin CUDA-allokator ellers viser.
        _frigjør_gpu_cache()

        # Logg adaptiv batchstørrelse og bottleneck-info
        if ferdig + cpu_ferdig >= neste_status:
            neste_status += 20
            elapsed = time.perf_counter() - start_alle
            total_tracked = tid_vente_cpu + tid_gpu
            if total_tracked > 0:
                cpu_pct = tid_vente_cpu / total_tracked * 100
                gpu_pct_tid = tid_gpu / total_tracked * 100
            else:
                cpu_pct = gpu_pct_tid = 0
            snitt_kø = kø_dybde_sum / max(antall_batches, 1)

            # Bestem flaskehals
            if cpu_pct > 60:
                flaskehals = "🔴 CPU (rendering)"
            elif gpu_pct_tid > 80:
                flaskehals = "🟡 GPU (OCR)"
            else:
                flaskehals = "🟢 balansert"

            gpu_mem = _gpu_prosent_brukt()
            gpu_mem_str = f" GPU-minne: {gpu_mem:.0f}%" if gpu_mem else ""

            print(f"  ⚡ Flaskehals: {flaskehals} | "
                  f"vente-CPU: {cpu_pct:.0f}% | GPU: {gpu_pct_tid:.0f}% | "
                  f"kø-dybde: {snitt_kø:.1f}{gpu_mem_str}")
            if cpu_prosesser:
                print(f"     Fordeling så langt: GPU {ferdig} dok / "
                      f"CPU {cpu_ferdig} dok ({len(cpu_prosesser) - cpu_avsluttet} "
                      f"workers aktive)")
            if adaptiv_batch:
                print(f"     {adaptiv_batch}")

        if args.vis_ressurser and ferdig % 10 == 0:
            _skriv_ressursstatus()

    render_executor.shutdown(wait=True)

    # ── Vent på CPU-workers og samle resten av resultatene ───────
    if cpu_prosesser:
        print(f"\nGPU ferdig med sin del. Venter på {len(cpu_prosesser)} "
              f"CPU-workers ({cpu_ferdig} dok ferdig så langt)...")
        _drener_cpu(blokkerende=True)
        for p_worker in cpu_prosesser:
            p_worker.join(timeout=30)

        print(f"  CPU-workers ferdig: {cpu_ferdig} dok, {cpu_sider} sider, "
              f"{cpu_feilet} feilet")

    # ── Oppsummering ─────────────────────────────────────────────
    vegg_tid = time.perf_counter() - start_alle
    totalt_ferdig = ferdig + cpu_ferdig
    totalt_sider = total_sider + cpu_sider
    print(f"\n{'=' * 60}")
    print(f"Ferdig! {totalt_ferdig} dokumenter, {totalt_sider} sider")
    if cpu_prosesser:
        print(f"  GPU: {ferdig} dok, {total_sider} sider")
        print(f"  CPU: {cpu_ferdig} dok, {cpu_sider} sider ({len(cpu_prosesser)} workers)")
    print(f"  GPU-tid:    {_fmt_tid(total_tid)} ({total_tid / max(ferdig, 1):.2f}s/dok, "
          f"{total_tid / max(total_sider, 1):.2f}s/side)")
    print(f"  Vegg-tid:   {_fmt_tid(vegg_tid)} (inkl. rendering, I/O)")
    if totalt_sider > 0:
        print(f"  Throughput: {totalt_sider / vegg_tid:.1f} sider/s, "
              f"{totalt_ferdig / vegg_tid * 3600:.0f} dok/time")

    # Flaskehals-analyse
    total_tracked = tid_vente_cpu + tid_gpu
    if total_tracked > 0:
        cpu_pct = tid_vente_cpu / total_tracked * 100
        gpu_pct_tid = tid_gpu / total_tracked * 100
        overhead = vegg_tid - total_tracked
        print(f"\n  Tidsfordeling:")
        print(f"    Vente på CPU (rendering): {_fmt_tid(tid_vente_cpu)} ({cpu_pct:.0f}%)")
        print(f"    GPU (orientering + OCR):  {_fmt_tid(tid_gpu)} ({gpu_pct_tid:.0f}%)")
        if overhead > 1:
            print(f"    Overhead (kø/IO/cache):   {_fmt_tid(overhead)}")
        print(f"    Snitt kødybde:            {kø_dybde_sum / max(antall_batches, 1):.1f} "
              f"(høy = GPU er flaskehals, lav = CPU er flaskehals)")
        if cpu_pct > 60:
            print(f"\n  💡 CPU-rendering er flaskehalsen. Prøv --workers {workers * 2}")
        elif gpu_pct_tid > 80 and not cpu_prosesser:
            print(f"\n  💡 GPU er flaskehalsen. Prøv --cpu-ocr for parallell CPU-inferens.")

    print(f"  Cache-mappe: {cache_mappe}")

    if feilet:
        print(f"\nFeilet ({len(feilet)}):")
        for navn, feil in feilet[:10]:
            print(f"  {navn}: {feil}")
        if len(feilet) > 10:
            print(f"  ... og {len(feilet) - 10} til")

    _skriv_ressursstatus()
    return 1 if feilet else 0


if __name__ == "__main__":
    sys.exit(main() or 0)



