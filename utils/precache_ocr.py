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
import os
import sys
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", message=".*ccache.*")
os.environ["GLOG_minloglevel"] = "2"

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


def main():
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
    p.add_argument("--workers", type=int, default=4,
                   help="antall tråder for parallell PDF-rendering (default: 4)")
    p.add_argument("--prefetch", type=int, default=12,
                   help="antall PDF-er å pre-rendre i kø (default: 12)")
    p.add_argument("--minne-grense", type=int, default=88,
                   help="maks RAM-bruk i prosent før pre-rendering pauser (default: 88)")
    p.add_argument("--ocr-batch", type=int, default=None,
                   help=f"sider per OCR-batch (default: {SIDER_PER_OCR_BATCH} fra config)")
    p.add_argument("--dokument-batch", type=int, default=6,
                   help="antall dokumenter å samle på GPU samtidig (default: 6)")
    p.add_argument("--hpi", action="store_true",
                   help="aktiver High Performance Inference (TensorRT) — krever kraftig GPU")
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
    ocr_batch = args.ocr_batch or SIDER_PER_OCR_BATCH
    dokument_batch = args.dokument_batch

    # ── HPI-modus (TensorRT) ──────────────────────────────────────
    if args.hpi:
        os.environ["SLADD_HPI"] = "1"

    # ── Skriv ressursstatus før start ────────────────────────────
    _skriv_ressursstatus()
    print(f"\nStarter OCR-caching av {len(filer)} dokumenter "
          f"(workers={args.workers}, prefetch={args.prefetch}, "
          f"ocr_batch={ocr_batch}, dokument_batch={dokument_batch}, "
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
    render_executor = ThreadPoolExecutor(max_workers=args.workers)
    prefetch_kø = deque()  # inneholder Future-objekter for rendrede PDF-er
    fil_indeks = 0  # neste fil å sende til rendering

    def _fyll_kø():
        """Send filer til rendering så lenge køen ikke er full og minne er OK."""
        nonlocal fil_indeks
        while (fil_indeks < totalt
               and len(prefetch_kø) < args.prefetch
               and _er_minne_trygt(args.minne_grense)):
            fut = render_executor.submit(_last_pdf, filer[fil_indeks])
            prefetch_kø.append((filer[fil_indeks], fut))
            fil_indeks += 1

    # Start pre-rendering
    _fyll_kø()

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
        batch_docs = []  # [(navn, bilder), ...]
        while prefetch_kø and len(batch_docs) < dokument_batch:
            sti, fut = prefetch_kø.popleft()
            sti, resultat = fut.result()
            navn = os.path.basename(sti)

            if isinstance(resultat, Exception):
                feilet.append((navn, repr(resultat)))
                ferdig += 1
                print(f"[{ferdig}/{totalt}] ✗ {navn}: {resultat!r}")
                continue
            batch_docs.append((navn, resultat))
            # Fyll køen mens vi venter på futures
            _fyll_kø()

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
            alle_tokens = les_tokens_batched(alle_bilder_ocr)

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

        # Frigjør ubrukt cachet GPU-minne — forhindrer den gradvise veksten
        # som PaddlePaddle sin CUDA-allokator ellers viser.
        _frigjør_gpu_cache()

        if args.vis_ressurser and ferdig % 10 == 0:
            _skriv_ressursstatus()

    render_executor.shutdown(wait=True)

    # ── Oppsummering ─────────────────────────────────────────────
    vegg_tid = time.perf_counter() - start_alle
    print(f"\n{'=' * 60}")
    print(f"Ferdig! {ferdig} dokumenter, {total_sider} sider")
    print(f"  GPU-tid:    {_fmt_tid(total_tid)} ({total_tid / max(ferdig, 1):.2f}s/dok, "
          f"{total_tid / max(total_sider, 1):.2f}s/side)")
    print(f"  Vegg-tid:   {_fmt_tid(vegg_tid)} (inkl. rendering, I/O)")
    if total_sider > 0:
        print(f"  Throughput: {total_sider / vegg_tid:.1f} sider/s, "
              f"{ferdig / vegg_tid * 3600:.0f} dok/time")
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



