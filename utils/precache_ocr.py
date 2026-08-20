"""
Forhåndscache PaddleOCR-resultater for et sett med dokumenter.

Kjører kun PaddleOCR (orientering + tekstgjenkjenning) og lagrer
resultatene i OCR-cachen. Ingen YOLO, ingen FNR-deteksjon, ingen
evaluering — bare den tyngste GPU-operasjonen isolert.

Optimalisert for å bruke så mye ressurser som mulig uten å krasje:
  - Pre-rendrer neste PDF(er) mens GPU jobber
  - Overvåker minne og tilpasser pipeline-dybden
  - Kan parallellisere PDF-rendering (CPU) med OCR (GPU)
  - Hopper automatisk over allerede cachede dokumenter

Bruk:
    python precache_ocr.py --mappe /sti/til/pdfer
    python precache_ocr.py --mappe /sti/til/pdfer --cache /sti/til/cache
    python precache_ocr.py --mappe /sti/til/pdfer --workers 4 --antall alle
    python precache_ocr.py --velg-fra-fil filer.txt --mappe /sti/til/pdfer
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


# ── CPU OCR-worker (multiprocessing) ─────────────────────────────

def _cpu_worker(fil_kø, resultat_kø, cache_mappe, worker_id, tråder_per_worker):
    """Prosess som kjører PaddleOCR på CPU for ett dokument om gangen.

    Hver worker har sin egen PaddleOCR-instans med device='cpu' og mkldnn.
    Leser filstier fra fil_kø, skriver resultater til resultat_kø.
    """
    # Begrens tråder per prosess — forhindrer oversubscription
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

    # Supprimér C++-nivå oneDNN-spam fra PaddlePaddle (ReduceMeanCheckIfOneDNNSupport osv.)
    _devnull = open(os.devnull, "w")
    os.dup2(_devnull.fileno(), 2)  # redirect stderr → /dev/null

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
    orient = DocImgOrientationClassification(
        model_name="PP-LCNet_x1_0_doc_ori",
        model_dir=ori_dir,
        device="cpu",
    )

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

    ferdig = 0
    while True:
        try:
            sti = fil_kø.get(timeout=5)
        except Exception:
            # Kø tom i 5s — sjekk om vi skal avslutte
            if fil_kø.empty():
                break
            continue

        if sti is None:  # poison pill
            break

        navn = os.path.basename(sti)
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
            resultat_kø.put(("ok", worker_id, navn, len(bilder), n_tokens))
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
    p.add_argument("--cpu-ocr", type=int, default=0,
                   help="antall ekstra CPU-prosesser for OCR (0=kun GPU, auto=basert på kjerner)")
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
    cpu_ocr = args.cpu_ocr
    cpu_prosesser = []
    cpu_fil_kø = None
    cpu_resultat_kø = None
    cpu_filer_sendt = 0

    if cpu_ocr > 0:
        # Splitt filer: gi en andel til CPU-workers, resten til GPU
        # CPU er ~5-10x tregere per side, så gi dem ~40% av filene
        # (10 CPU-workers × 0.1 GPU-hastighet ≈ 1× GPU-hastighet)
        cpu_andel = min(0.6, cpu_ocr * 0.06)  # ~6% per worker, maks 60%
        cpu_antall = int(len(filer) * cpu_andel)
        cpu_filer = filer[len(filer) - cpu_antall:]  # ta fra slutten
        filer = filer[:len(filer) - cpu_antall]       # GPU tar resten

        cpu_fil_kø = mp.Queue()
        cpu_resultat_kø = mp.Queue()

        # Legg filer i køen
        for f in cpu_filer:
            cpu_fil_kø.put(f)
        cpu_filer_sendt = len(cpu_filer)

        # Poison pills
        for _ in range(cpu_ocr):
            cpu_fil_kø.put(None)

        # Start CPU-workers
        # Fordel kjerner: reserver noen til GPU-pipeline (rendering + main)
        kjerner_til_cpu = max(_antall_cpu_kjerner() - workers - 2, cpu_ocr)
        tråder_per = max(kjerner_til_cpu // cpu_ocr, 1)

        for i in range(cpu_ocr):
            p = mp.Process(
                target=_cpu_worker,
                args=(cpu_fil_kø, cpu_resultat_kø, cache_mappe, i, tråder_per),
                daemon=True
            )
            p.start()
            cpu_prosesser.append(p)

        print(f"  CPU OCR: {cpu_ocr} prosesser × {tråder_per} tråder = "
              f"{cpu_ocr * tråder_per} kjerner, {cpu_filer_sendt} filer "
              f"({cpu_andel*100:.0f}%)")
        print(f"  GPU:     {len(filer)} filer ({(1-cpu_andel)*100:.0f}%)")

    # ── Skriv ressursstatus før start ────────────────────────────
    _skriv_ressursstatus()
    print(f"\nStarter OCR-caching av {len(filer)} dokumenter på GPU "
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
    fil_indeks = 0  # neste fil å sende til rendering

    def _fyll_kø():
        """Send filer til rendering så lenge køen ikke er full og minne er OK."""
        nonlocal fil_indeks
        while (fil_indeks < totalt
               and len(prefetch_kø) < prefetch
               and _er_minne_trygt(args.minne_grense)):
            fut = render_executor.submit(_last_pdf, filer[fil_indeks])
            prefetch_kø.append((filer[fil_indeks], fut))
            fil_indeks += 1

    # Start pre-rendering
    _fyll_kø()

    # ── Bottleneck-tracking ───────────────────────────────────────
    tid_vente_cpu = 0.0   # tid brukt på å vente på at CPU rendrer ferdig
    tid_gpu = 0.0         # tid brukt på GPU (orientering + OCR)
    tid_io = 0.0          # tid brukt på cache-skriving
    antall_batches = 0
    kø_dybde_sum = 0      # sum av kødybde ved batch-start (for snitt)
    neste_status = 20     # neste dokument-tall som utløser flaskehals-print

    while ferdig < totalt:
        # Sørg for at køen er fylt opp
        _fyll_kø()

        # Hvis køen er tom men vi har flere filer, vent litt og prøv igjen
        if not prefetch_kø:
            if fil_indeks < totalt:
                # Minne er for høyt, vent til det frigjøres
                print("  ⏸  Venter på ledig minne for pre-rendering...")
                time.sleep(2)
                _fyll_kø()
                if not prefetch_kø:
                    # Tving gjennom én fil uansett
                    fut = render_executor.submit(_last_pdf, filer[fil_indeks])
                    prefetch_kø.append((filer[fil_indeks], fut))
                    fil_indeks += 1
            else:
                break

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
                print(f"[{ferdig}/{totalt}] ✗ {navn}: {resultat!r}")
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
                snitt = elapsed / ferdig
                gjenstår = snitt * (totalt - ferdig)

                rot_str = ""
                if any(k != 0 for k in rotasjoner):
                    rot_str = f" rot=[{','.join(str(k * 90) + '°' for k in rotasjoner)}]"

                print(f"[{ferdig}/{totalt}] ✓ {navn}: "
                      f"{n_sider} side(r), {n_tokens} tokens{rot_str} — "
                      f"batch {len(batch_docs)} dok/{sum(n for _, n in dok_grenser)} sider "
                      f"på {tid_batch:.1f}s "
                      f"(gått: {_fmt_tid(elapsed)}, gjenstår: {_fmt_tid(gjenstår)})")

        except Exception as e:
            # Hele batchen feilet — logg alle dokumenter
            for navn, _bilder in batch_docs:
                feilet.append((navn, repr(e)))
                ferdig += 1
                print(f"[{ferdig}/{totalt}] ✗ {navn}: {e!r}")
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
        if ferdig >= neste_status:
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
            if adaptiv_batch:
                print(f"     {adaptiv_batch}")

        if args.vis_ressurser and ferdig % 10 == 0:
            _skriv_ressursstatus()

    render_executor.shutdown(wait=True)

    # ── Vent på CPU-workers og samle resultater ──────────────────
    cpu_ferdig = 0
    cpu_feilet = 0
    cpu_sider = 0
    if cpu_prosesser:
        print(f"\nGPU ferdig. Venter på {len(cpu_prosesser)} CPU-workers...")
        # Drain result queue
        workers_avsluttet = 0
        while workers_avsluttet < len(cpu_prosesser):
            try:
                msg = cpu_resultat_kø.get(timeout=30)
                status = msg[0]
                if status == "ok":
                    _, wid, navn, n_sider, n_tokens = msg
                    cpu_ferdig += 1
                    cpu_sider += n_sider
                    if cpu_ferdig % 50 == 0:
                        print(f"  [CPU] {cpu_ferdig}/{cpu_filer_sendt} ferdig")
                elif status == "feil":
                    _, wid, navn, _, feilmelding = msg
                    cpu_feilet += 1
                    feilet.append((navn, feilmelding))
                elif status == "ferdig":
                    workers_avsluttet += 1
            except Exception:
                # Timeout — sjekk om alle prosesser er døde
                if all(not p.is_alive() for p in cpu_prosesser):
                    break

        for p in cpu_prosesser:
            p.join(timeout=10)

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
            print(f"\n  💡 GPU er flaskehalsen. Prøv --cpu-ocr 10 for parallell CPU-inferens.")

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



