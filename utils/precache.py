"""Pre-fill the OCR and YOLO caches for a set of documents.

Renders each document once and fills both caches in the same pass, doing only
what is missing per document. A new model therefore costs only the YOLO part:
the rotations come free from the OCR cache, and the YOLO cache requires them
(it is invalidated when the per-page rotation does not match). With both caches
warm, utils/run.py skips OCR, YOLO and PDF rendering entirely.

Parallelism lives in parallel_pipeline.py.

Run:
    python precache.py --folder /path/to/pdfs
    python precache.py --folder /path/to/pdfs --only yolo --yolo-weights $SLADD_WEIGHTS/<model>/<model>.pt
    python precache.py --folder /path/to/pdfs --gpu-processes 4 --start-batch 8 --profile
"""

import argparse
import os
import sys
import time
import warnings

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

import parallel_pipeline as pp
from config import PAGES_PER_OCR_BATCH, default_weights
from file_selection import select_files
from ocr_cache import read_cache as read_ocr_cache, write_cache as write_ocr_cache
from yolo_cache import (read_cache as read_yolo_cache, write_cache as write_yolo_cache,
                        cache_dir_for_weights)

STANDARD_WEIGHTS = default_weights()


# ── Handler (runs in the worker process) ─────────────────────────

def make_handler(task):
    """Build the function that does OCR + YOLO for a group of documents.

    Called in the worker process, after fork and after the GPU memory flags
    are set, hence the heavy imports in here.
    """
    e = task["extra"]
    timings = task["timings"]
    ocr_dir, yolo_dir = e["ocr_dir"], e["yolo_dir"]
    ocr_batch = e["ocr_batch"]
    force = e["force"]

    from orientation import find_rotations_batch

    read_tokens_batched = None
    if ocr_dir:
        from paddle_ocr_model_fnr import read_tokens_batched

    find_yolo_boxes = None
    if yolo_dir:
        import yolo_fnr
        yolo_fnr.set_weights(e["weights"])
        find_yolo_boxes = yolo_fnr.find_yolo_boxes

    from config import YOLO_CACHE_CONF_FLOOR

    def _time(post, t0):
        timings[post] = timings.get(post, 0.0) + (time.perf_counter() - t0)

    # Warm up the models: a one-off cost that must stay out of the estimates.
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    find_rotations_batch([dummy])
    if read_tokens_batched:
        read_tokens_batched([dummy])
    if find_yolo_boxes:
        find_yolo_boxes(dummy, conf=YOLO_CACHE_CONF_FLOOR)

    def handle(group, pages_at_a_time=None):
        # ── 1. What is missing per document? ──────────────────────
        # Cached OCR gives the rotations for free, and YOLO needs them.
        workers = []
        for name, images in group:
            cached = None if force or not ocr_dir else read_ocr_cache(ocr_dir, name)
            rotations = list(cached[0]) if cached and len(cached[0]) == len(images) else None
            needs_yolo = bool(yolo_dir) and (
                force or rotations is None
                or read_yolo_cache(yolo_dir, name, rotations) is None)
            workers.append({
                "navn": name, "images": images, "rotations": rotations,
                "tokens": cached[1] if cached else None,
                "trenger_ocr": bool(ocr_dir) and cached is None,
                "needs_yolo": needs_yolo,
            })

        # ── 2. Orientation for those missing rotations ────────────
        missing_root = [j for j in workers if j["rotations"] is None]
        if missing_root:
            t0 = time.perf_counter()
            pages = [b for j in missing_root for b in j["images"]]
            steg = pages_at_a_time or max(4 * ocr_batch, 8)
            all_of = []
            for i in range(0, len(pages), steg):
                all_of.extend(find_rotations_batch(pages[i:i + steg]))
            _time("orientation", t0)
            i = 0
            for j in missing_root:
                j["rotations"] = all_of[i:i + len(j["images"])]
                i += len(j["images"])

        # ── 3. Rotate once: OCR and YOLO share the same image ────
        t0 = time.perf_counter()
        for j in workers:
            if j["trenger_ocr"] or j["needs_yolo"]:
                j["rotated"] = [np.rot90(b, k) if k else b
                               for b, k in zip(j["images"], j["rotations"])]
        _time("rotation", t0)

        # ── 4. OCR in one batch across every document missing it ──
        ocr_workers = [j for j in workers if j["trenger_ocr"]]
        if ocr_workers:
            t0 = time.perf_counter()
            pages = [b for j in ocr_workers for b in j["rotated"]]
            tokens = read_tokens_batched(pages, batch_size=pages_at_a_time or ocr_batch)
            _time("ocr", t0)
            i = 0
            for j in ocr_workers:
                j["tokens"] = tokens[i:i + len(j["images"])]
                i += len(j["images"])

        # ── 5. YOLO, one page at a time (the ultralytics API here) ─
        for j in workers:
            if not j["needs_yolo"]:
                continue
            t0 = time.perf_counter()
            j["yolo"] = [find_yolo_boxes(b, conf=YOLO_CACHE_CONF_FLOOR)
                         for b in j["rotated"]]
            _time("yolo", t0)

        # ── 6. Write the caches ───────────────────────────────────
        t0 = time.perf_counter()
        results = []
        for j in workers:
            if j["trenger_ocr"]:
                write_ocr_cache(ocr_dir, j["navn"], j["rotations"], j["tokens"])
            if j["needs_yolo"]:
                write_yolo_cache(yolo_dir, j["navn"], j["rotations"], j["yolo"])

            parts = []
            if j["tokens"] is not None:
                parts.append(f"{sum(len(t) for t in j['tokens'])} tokens"
                             + ("" if j["trenger_ocr"] else " (cached)"))
            if j.get("yolo") is not None:
                parts.append(f"{sum(len(b) for b in j['yolo'])} yolo boxes")
            elif j["needs_yolo"] is False and yolo_dir:
                parts.append("yolo cached")
            if any(k != 0 for k in j["rotations"]):
                parts.append("rot=[" + ",".join(f"{k * 90}°" for k in j["rotations"]) + "]")
            results.append({"navn": j["navn"], "sider": len(j["images"]),
                               "text": ", ".join(parts)})
        _time("cache", t0)
        return results

    def reset():
        """Tear the models down so the allocator returns the memory.

        Last resort when the process runs out of memory inside its own cap:
        then its own allocator cache is what is full, and empty_cache() alone
        will not release it. The models are lazy and are rebuilt on the next
        call (~10 s), which buys getting the document done.
        """
        import orientation
        orientation._orient = None
        if ocr_dir:
            import paddle_ocr_model_fnr
            paddle_ocr_model_fnr.reader = None
        if yolo_dir:
            import yolo_fnr as yf
            yf._model = None
        pp.free_gpu_cache()

    handle.reset = reset
    return handle


# ── File selection ───────────────────────────────────────────────

def _derive_cache_base(args):
    """Cache base folder: --cache, else $SLADD_CACHE/<uttrekk>/."""
    if args.cache:
        return args.cache
    base = os.environ.get("SLADD_CACHE")
    if base:
        return os.path.join(base, os.path.basename(os.path.normpath(args.folder)))
    return None


def _missing_cache(files, ocr_dir, yolo_dir):
    """Keep the files where at least one active cache is missing.

    The YOLO cache is validated against the rotations, so without an OCR cache
    YOLO counts as missing, because it cannot be read without them.
    """
    ut = []
    for f in files:
        name = os.path.basename(f)
        cached = read_ocr_cache(ocr_dir, name) if ocr_dir else None
        if ocr_dir and cached is None:
            ut.append(f)
            continue
        if yolo_dir:
            rotations = list(cached[0]) if cached else None
            if rotations is None or read_yolo_cache(yolo_dir, name, rotations) is None:
                ut.append(f)
    return ut


def main():
    pp.setup_logfile("precache")

    p = argparse.ArgumentParser(
        description="Pre-fill the OCR and YOLO caches. Renders each document "
                    "once and does only what is missing.")
    p.add_argument("--folder", required=True, help="folder of PDF files")
    p.add_argument("--cache", default=None,
                   help="base folder for the caches (default: $SLADD_CACHE/<uttrekk>/)")
    p.add_argument("--only", choices=("ocr", "yolo", "both"), default="both",
                   help="which caches to fill (default: begge)")
    p.add_argument("--yolo-weights", default=STANDARD_WEIGHTS,
                   help="YOLO weights file; the cache is per weights file")
    p.add_argument("--select", nargs="*", default=[],
                   help="specific files (filename/substring)")
    p.add_argument("--select-from-file", default=None,
                   help="read file IDs from a text file, one per line")
    p.add_argument("--count", default="all",
                   help="number of files when --select is empty (a number, or 'all')")
    p.add_argument("--ocr-batch", type=int, default=None,
                   help=f"pages per OCR batch (default: {PAGES_PER_OCR_BATCH} from "
                        "config, divided by the number of processes). Biggest "
                        "driver of peak GPU memory. The detection model holds "
                        "activations for the whole page batch at once")
    p.add_argument("--rec-batch", type=int, default=0,
                   help="text lines per recognition batch (0=auto from the "
                        "per-process memory cap). Affects memory and speed "
                        "only, not the result")
    p.add_argument("--hpi", action="store_true",
                   help="enable High Performance Inference (TensorRT)")
    p.add_argument("--force", action="store_true",
                   help="rerun even if the document is already cached")
    pp.add_arguments(p)
    args = p.parse_args()

    if not os.path.isdir(args.folder):
        print(f"ERROR: --folder does not exist: {args.folder}")
        return 1
    if args.select_from_file and not os.path.isfile(args.select_from_file):
        print(f"ERROR: --select-from-file does not exist: {args.select_from_file}")
        return 1

    cache_base = _derive_cache_base(args)
    if not cache_base:
        print("ERROR: no cache folder given. Use --cache or set $SLADD_CACHE "
              "(server default: /data2/cache).")
        return 1

    ocr_dir = os.path.join(cache_base, "ocr") if args.only in ("ocr", "both") else None
    yolo_dir = None
    if args.only in ("yolo", "both"):
        if not os.path.isfile(args.yolo_weights):
            print(f"ERROR: YOLO weights not found: {args.yolo_weights}")
            return 1
        yolo_dir = cache_dir_for_weights(os.path.join(cache_base, "yolo"),
                                            args.yolo_weights)
    for m in (ocr_dir, yolo_dir):
        if m:
            os.makedirs(m, exist_ok=True)

    # ── File list ────────────────────────────────────────────────
    select, from_file = args.select, False
    if args.select_from_file:
        with open(args.select_from_file, encoding="utf-8") as f:
            select = [line.strip() for line in f if line.strip()]
        from_file = True
        print(f"Read {len(select)} IDs from {args.select_from_file}")

    files = select_files(args.folder, select, args.count, exact=from_file)
    if not files:
        print("No files to process. Check --folder / --select / --count.")
        return 1
    print(f"Files found:  {len(files)}")
    if ocr_dir:
        print(f"OCR cache:    {ocr_dir}")
    if yolo_dir:
        print(f"YOLO cache:   {yolo_dir}")
        print(f"YOLO weights: {args.yolo_weights}")

    if not args.force:
        original = len(files)
        files = _missing_cache(files, ocr_dir, yolo_dir)
        if original - len(files):
            print(f"Skipping:     {original - len(files)} already cached, "
                  f"{len(files)} remaining")
    if not files:
        print("Everything is already cached!")
        return 0

    # ── Setup ────────────────────────────────────────────────────
    if args.hpi:
        os.environ["SLADD_HPI"] = "1"
    opts = pp.setup(args)
    n = opts["n_processes"]

    # The default rec batch does not fit under a small per-process cap; read by app/paddle_ocr_model_fnr.py.
    rec_batch = args.rec_batch
    if not rec_batch and opts["gpu_mb"]:
        rec_batch = 32 if opts["gpu_mb"] < 10000 else 64
    if rec_batch:
        os.environ["SLADD_REC_BATCH"] = str(rec_batch)
        print(f"  Rec batch:  {rec_batch} text lines"
              + ("" if args.rec_batch else " (auto from the memory cap)"))
    if args.ocr_batch:
        ocr_batch = args.ocr_batch
    else:
        gpu = pp.gpu_memory_info()
        reason = 32 if gpu and gpu[1] >= 24000 else 16 if gpu and gpu[1] >= 16000 else PAGES_PER_OCR_BATCH
        ocr_batch = max(reason // n, 4)
    opts["extra"] = dict(ocr_dir=ocr_dir, yolo_dir=yolo_dir,
                          weights=args.yolo_weights, ocr_batch=ocr_batch,
                          force=args.force)
    print(f"  OCR batch:  {ocr_batch} pages  |  files are shared dynamically "
          f"between {n} process(es)")
    pp.write_resource_status()

    done, _, failed = pp.run(files, make_handler, opts)
    if ocr_dir:
        print(f"  OCR cache:  {ocr_dir}")
    if yolo_dir:
        print(f"  YOLO cache: {yolo_dir}")
    # Unfinished counts as failure: if every process crashes the failed list is
    # empty, but the job is still not done.
    if failed or done < len(files):
        print(f"  Not done:   {len(files) - done} documents remaining")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
