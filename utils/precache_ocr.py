"""
Forhåndscache PaddleOCR-resultater for et sett med dokumenter.

Kjører kun PaddleOCR (orientering + tekstgjenkjenning) og lagrer
resultatene i OCR-cachen. Ingen YOLO, ingen FNR-deteksjon, ingen
evaluering — bare den tyngste GPU-operasjonen isolert.

Arkitektur: hovedprosessen koordinerer bare, og alt arbeid skjer i
uavhengige prosesser som henter filer fra en delt teller. Grunnen er målt
på V100S + Xeon 6230: ~77 % av tiden i én pipeline gikk til enkelttrådet
CPU-forbehandling (normalisering, bildekopier) og bare ~11 % til
GPU-arbeid, så kortet sto stille halve tiden. Fire prosesser mot samme
kort ga 3,3× gjennomstrømning og mettet GPU-en (228 av 250 W).

Minnestyringen følger av det: hver prosess får sitt eget tak på kortet
(FLAGS_gpu_memory_limit_mb), batchstørrelsen finner taket selv med
slow-start og halvering ved minnefeil, og en minnefeil koster en
omkjøring av de samme dokumentene — aldri et tapt dokument.

Bruk:
    python precache_ocr.py --mappe /sti/til/pdfer
    python precache_ocr.py --mappe /sti/til/pdfer --gpu-prosesser 4 --start-batch 8
    python precache_ocr.py --mappe /sti/til/pdfer --profil
    python precache_ocr.py --velg-fra-fil filer.txt --mappe /sti/til/pdfer
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
# CUDA-kontekster i barneprosessene. Arbeidsprosessene importerer dem selv.


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


class _AIMDBatchStørrelse:
    """Batchstørrelse styrt av faktiske minnefeil, ikke av nvidia-smi.

    Når flere prosesser deler kortet med hvert sitt minnetak er den globale
    målingen ubrukelig som styringssignal: den ligger *ved* taket når alt
    går som planlagt, og en måling-styrt regulator bremser da hele tiden.

    Samme mønster som TCP: dobling til første minnefeil (slow-start), så
    halvering og forsiktig lineær vekst videre. Doblingen finner taket i
    log₂-steg i stedet for ett hakk om gangen — 4→8→12 tar tre batcher der
    ren lineær vekst brukte tjue. Siden en minnefeil bare koster en
    omkjøring av de samme dokumentene, er det billig å lete oppover.
    """

    def __init__(self, start=4, minimum=1, maksimum=12):
        self.nåværende = max(min(start, maksimum), minimum)
        self.minimum = minimum
        self.maksimum = maksimum
        self._ok_siden = 0
        self._oom = 0
        self._slow_start = True

    def neste(self):
        if self.nåværende >= self.maksimum:
            return self.nåværende
        if self._slow_start:
            if self._ok_siden >= 1:
                self.nåværende = min(self.maksimum, self.nåværende * 2)
                self._ok_siden = 0
        elif self._ok_siden >= 2:
            self.nåværende += 1
            self._ok_siden = 0
        return self.nåværende

    def ok(self):
        self._ok_siden += 1

    def oom(self):
        self.nåværende = max(self.minimum, self.nåværende // 2)
        self._slow_start = False   # over til forsiktig lineær vekst
        self._ok_siden = -4        # hold igjen noen batcher først
        self._oom += 1

    def __repr__(self):
        fase = "slow-start" if self._slow_start else "lineær"
        return f"dokument_batch={self.nåværende} (AIMD/{fase}, {self._oom} minnefeil)"


# ── Cache og filvalg ─────────────────────────────────────────────

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


# ── Arbeidsfordeling mellom prosessene ───────────────────────────

class _Arbeidsfordeler:
    """Delt filliste som GPU-hovedprosessen og CPU-workerne henter fra.

    Begge sider tar neste ledige fil fra samme teller, så fordelingen
    balanserer seg selv: den raskeste siden tar flest filer. En fast
    fordeling gjettet på forhånd gir alltid en hale der den ene siden
    er ferdig og venter på den andre.

    Telleren er en delt mp.Value; fillisten i hver prosess er en
    fork()-kopi. Derfor må prosessene startes med fork-kontekst.
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

# ── Feilmeldinger ────────────────────────────────────────────────

def _kort_feil(e, maks=200):
    """Kort én-linjes sammendrag av en paddle-feil.

    Paddle sine minnefeil er 4-5 KB C++-traceback. Uforkortet fyller de
    loggen fullstendig når flere prosesser treffer taket samtidig.
    """
    tekst = " ".join(str(e).split())
    if "Error Message Summary" in tekst:
        tekst = tekst.split("Error Message Summary")[-1]
    tekst = tekst.strip(" -:")
    if not tekst:
        tekst = repr(e)
    return tekst[:maks]


def _er_minnefeil(e):
    tekst = str(e)
    return ("Out of memory" in tekst or "ResourceExhausted" in tekst
            or "OutOfMemory" in tekst or isinstance(e, MemoryError))


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
    kø = oppgave["resultat_kø"]
    fordeler = oppgave["fordeler"]
    cache_mappe = oppgave["cache_mappe"]
    ocr_batch = oppgave["ocr_batch"]
    prefetch = oppgave["prefetch"]
    minne_grense = oppgave["minne_grense"]

    # Forsøk på å gi hver prosess sin egen andel av kortet. Målt:
    # FLAGS_fraction_of_gpu_memory_to_use blir IKKE respektert av
    # inferens-prediktoren, mens FLAGS_gpu_memory_limit_mb blir det —
    # uten den vokser prosessene inn i hverandre til kortet er tomt.
    # Må settes før paddle rører GPU-en, altså før importene under.
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
    if oppgave.get("gpu_mb"):
        os.environ["FLAGS_gpu_memory_limit_mb"] = str(int(oppgave["gpu_mb"]))

    from orientering import finn_rotasjoner_batch as orienter_batch
    from paddle_ocr_model_fnr import les_tokens_batched

    def les_tokens(bilder, batch=None):
        return les_tokens_batched(bilder, batch_size=batch or ocr_batch)

    # Varm opp modellene — engangskostnad som ikke skal med i estimatene
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    orienter_batch([dummy])
    les_tokens([dummy])
    kø.put(("klar", wid, "", 0, ""))

    if oppgave["dokument_batch"]:
        adaptiv, maks_batch_fast = None, oppgave["dokument_batch"]
    else:
        adaptiv = _AIMDBatchStørrelse(
            start=oppgave["start_batch"] or 4, minimum=1,
            maksimum=oppgave["maks_batch"])
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
        return ("stats", wid, "", ferdig_nå, dict(
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
                    # Ett dokument alene. Trykket er forbigående — de andre
                    # prosessene frigir minne fortløpende — så her venter vi
                    # heller enn å gi opp dokumentet.
                    resultat = None
                    siste = e
                    for forsøk, pause in enumerate((0, 5, 15, 45), start=1):
                        if pause:
                            time.sleep(pause)
                        _frigjør_gpu_cache()
                        try:
                            resultat = _behandle(gruppe, sider_om_gangen=1)
                            break
                        except Exception as e2:
                            siste = e2
                            if not _er_minnefeil(e2):
                                break
                    if resultat is None:
                        kø.put(("feil", wid, gruppe[0][0], 0, _kort_feil(siste)))
                        continue
                    dok_grenser, rot, tok = resultat
                    kø.put(("oom", wid, gruppe[0][0], 1,
                            f"side for side, klarte det på forsøk {forsøk}"))
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
    p.add_argument("--gpu-prosesser", type=int, nargs="?", default=4, const=-1,
                   metavar="N",
                   help="antall parallelle pipeliner mot samme GPU (default: 4, "
                        "uten tall = auto). Målt på V100S: ~77 %% av tiden i én "
                        "pipeline er enkelttrådet CPU-forbehandling, så én "
                        "prosess klarer ikke å mette kortet")
    p.add_argument("--workers", type=int, default=0,
                   help="tråder for PDF-rendering PER prosess (0=auto)")
    p.add_argument("--prefetch", type=int, default=0,
                   help="pre-rendrede PDF-er i kø per prosess (0=auto fra RAM)")
    p.add_argument("--minne-grense", type=int, default=88,
                   help="maks RAM-bruk i prosent før pre-rendering pauser (default: 88)")
    p.add_argument("--ocr-batch", type=int, default=None,
                   help=f"sider per OCR-batch (default: {SIDER_PER_OCR_BATCH} fra config). "
                        "Dette er den største driveren av GPU-minnetoppen — "
                        "deteksjonsmodellen holder aktiveringer for hele batchen")
    p.add_argument("--rec-batch", type=int, default=0,
                   help="tekstlinjer per gjenkjennings-batch (0=config-default). "
                        "Påvirker kun minne og hastighet, ikke resultatet")
    p.add_argument("--dokument-batch", type=int, default=0,
                   help="maks dokumenter per GPU-batch per prosess (0=adaptiv)")
    p.add_argument("--start-batch", type=int, default=0,
                   help="startverdi for den adaptive batchen (0=4). Vet du fra "
                        "forrige kjøring hva maskinen tåler, hopp rett dit")
    p.add_argument("--gpu-grense", type=int, default=90,
                   help="hvor mye av GPU-minnet prosessene får dele, i prosent "
                        "(default: 90). Deles likt mellom dem")
    p.add_argument("--hpi", action="store_true",
                   help="aktiver High Performance Inference (TensorRT) — krever kraftig GPU")
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

    # ── GPU-minne per prosess ────────────────────────────────────
    # Uten et tak per prosess vokser allokatorene uavhengig til kortet er
    # fullt (målt: 30,9 av 32 GB), og da feiler den prosessen som
    # tilfeldigvis ber om minne sist.
    gpu = _gpu_minne_info()
    gpu_mb = None
    if gpu and gpu[1] > 0:
        # ~700 MB per prosess går til CUDA-kontekst og modellvekter
        gpu_mb = max((gpu[1] * args.gpu_grense / 100 - 700 * n_gpu) / n_gpu, 512)

    # ── Sider per OCR-batch ──────────────────────────────────────
    # Største driver av minnetoppen: deteksjonsmodellen holder
    # aktiveringer for hele sidebatchen samtidig.
    if args.ocr_batch:
        ocr_batch = args.ocr_batch
    elif gpu and gpu[1] >= 24000:
        ocr_batch = max(32 // n_gpu, 4)
    elif gpu and gpu[1] >= 16000:
        ocr_batch = max(16 // n_gpu, 4)
    else:
        ocr_batch = SIDER_PER_OCR_BATCH

    workers = args.workers or min(max(kjerner // (4 * n_gpu), 2), 16)
    prefetch = args.prefetch or max(_auto_prefetch() // n_gpu, 8)
    maks_batch = 12   # taket for AIMD; minnetaket over er den reelle grensen

    if args.hpi:
        os.environ["SLADD_HPI"] = "1"
    if args.rec_batch:
        # Leses av app/paddle_ocr_model_fnr.py når prediktoren bygges
        os.environ["SLADD_REC_BATCH"] = str(args.rec_batch)

    print(f"  GPU-pipeliner: {n_gpu} × (1 hovedtråd + {workers} render-tråder), "
          f"prefetch {prefetch}/prosess")
    if gpu_mb:
        print(f"  GPU-minne:     {gpu_mb:.0f} MB per prosess "
              f"(FLAGS_gpu_memory_limit_mb)")
    if args.dokument_batch:
        print(f"  Dokument-batch: fast {args.dokument_batch} per prosess")
    else:
        print(f"  Dokument-batch: AIMD fra {args.start_batch or 4}, dobler til "
              f"maks {maks_batch}, halveres ved minnefeil")

    print(f"  OCR-batch: {ocr_batch} sider  |  filene deles dynamisk mellom "
          f"{n_gpu} prosess(er)")
    _skriv_ressursstatus()

    # ── Start prosessene ─────────────────────────────────────────
    ctx = mp.get_context("fork")   # fordeleren deles via fork
    fordeler = _Arbeidsfordeler(filer, ctx=ctx)
    kø = ctx.Queue()

    felles = dict(
        resultat_kø=kø, fordeler=fordeler, cache_mappe=cache_mappe,
        workers=workers, prefetch=prefetch, minne_grense=args.minne_grense,
        ocr_batch=ocr_batch, dokument_batch=args.dokument_batch,
        maks_batch=maks_batch, start_batch=args.start_batch,
        gpu_mb=gpu_mb, profil=args.profil,
    )
    prosesser = []
    for i in range(n_gpu):
        oppgave = dict(felles, id=f"g{i}")
        pr = ctx.Process(target=_worker, args=(oppgave,), daemon=True)
        pr.start()
        prosesser.append(pr)

    totalt = len(filer)
    print(f"\nStarter OCR-caching av {totalt} dokumenter "
          f"i {n_gpu} prosess(er). Laster modeller...\n")

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
                print(f"  [{wid}] modeller lastet")
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
