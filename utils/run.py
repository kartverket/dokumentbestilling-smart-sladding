import argparse
import io
import os
import sys
import threading
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Silence PaddlePaddle ccache warnings and C++ logging.
warnings.filterwarnings("ignore", message=".*ccache.*")
os.environ["GLOG_minloglevel"] = "2"

import fitz

_UTILS = os.path.dirname(os.path.abspath(__file__))
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)

_APP = os.path.join(_UTILS, "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from file_selection import select_files
from model_main import run_model_on_pdf_bytes
from csv_export import write_csv_header, append_csv
from evaluation import evaluate_against_truth, read_truth_xywh, _doc_no
from visualization import draw_and_save
from redaction import sladd_files
from yolo_fnr import set_weights, active_weights
from yolo_cache import cache_dir_for_weights
from load_pdf import PDF_DPI
import traceback
from save_result import write_result_files

import time
import csv as csv_modul

from utils_config import (
    DOC_DIR, DEFAULT_FILE_COUNT, TRUTH_CSV, CSV_OUT, OCR_LOG_FILE,
    PNG_DIR, SLADD_DIR, Y_ORIGIN, HIT_THRESHOLD
)

SCALE = PDF_DPI / 72.0                     # PDF points -> pixels


def _pages_from_result(result, pdf_bytes):
    if isinstance(result, dict):
        return result.get("sider", [])

    per_page = defaultdict(list)
    for b in result:
        per_page[b["page"]].append(b)

    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for n in range(1, doc.page_count + 1):
            rect = doc[n - 1].rect
            bw = int(round(rect.width * SCALE))
            bh = int(round(rect.height * SCALE))
            boxes = []
            for b in per_page.get(n, []):
                box = {
                    "x0": b["x"] * SCALE,
                    "y0": b["y"] * SCALE,
                    "x1": (b["x"] + b["width"]) * SCALE,
                    "y1": (b["y"] + b["height"]) * SCALE,
                    "kilde": b.get("kilde", "paddle"),
                }
                # Kept apart: yolo_conf is detection confidence,
                # paddle_rec_score is OCR read quality. See csv_export.
                for field in ("yolo_conf", "paddle_rec_score", "trekk"):
                    if b.get(field) is not None:
                        box[field] = b[field]
                boxes.append(box)
            pages.append({"side": n, "bilde_bredde": bw, "bilde_hoyde": bh,
                          "boxes": boxes})
    return pages


def _write_ocr_log(ocr_lines, path):
    n_pages = 0
    with open(path, "w", encoding="utf-8") as log:
        for (name, si) in sorted(ocr_lines):
            lines = ocr_lines[(name, si)]
            log.write(f"\n===== {name} page {si} - {len(lines)} text lines =====\n")
            for li, (text, merker) in enumerate(lines, start=1):
                if merker:
                    merk = ", ".join(
                        f"{digits} (mod11 {'OK' if ok else 'FAIL'})" for digits, ok in merker)
                    log.write(f"  line {li:>2}: {text!r}   <-- FNR HIT: {merk}\n")
                else:
                    log.write(f"  line {li:>2}: {text!r}\n")
            n_pages += 1
    return n_pages


def _read_done_from_csv(csv_path):
    """Read already-processed filenames from an existing CSV (for --proceed)."""
    done = set()
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv_modul.DictReader(f):
                done.add(row["navn"])
    return done


def _read_files_from_file(path):
    """Read filenames/IDs from a text file, one per line."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _process_from_cache(pdf_path, ocr_cache, yolo_cache, elektronisk_tinglyst,
                        only_yolo, with_lines, rettsstiftelsestyper=None,
                        postfilter=True):
    """Handle one document in a worker process, from cache only.

    A cache hit is pure CPU, so workers need no models. A miss returns
    ("miss", ...) and the main process, which has the models, runs it.
    """
    name = os.path.basename(pdf_path)
    start = time.perf_counter()
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        result = run_model_on_pdf_bytes(
            pdf_bytes, write_time=False, with_lines=with_lines, name=name,
            elektronisk_tinglyst=elektronisk_tinglyst, only_yolo=only_yolo,
            cache_dir=ocr_cache, yolo_cache_dir=yolo_cache,
            only_cache=True, rettsstiftelsestyper=rettsstiftelsestyper,
            postfilter=postfilter)
        if result is None:
            return ("miss", name, None, 0.0)
        pages = _pages_from_result(result, pdf_bytes)
        return ("ok", name, pages, time.perf_counter() - start)
    except Exception:
        return ("feil", name, traceback.format_exc(), 0.0)


def _draw_continuous(name, pages, folder, png_dir, truth, y_origin, csv_boxes_doc, sladd_boxes_doc):
    """Draw PNGs for one document right after inference."""
    draw_and_save(sladd_boxes_doc, truth, folder, png_dir,
                  y_origin=y_origin, write_log=True, clean=False, sources=csv_boxes_doc)


class _ImageBudget:
    """Thread-safe counter for --max-error-images: at most N documents drawn."""

    def __init__(self, max_items):
        self._rest = max_items                   # None = unlimited
        self._laas = threading.Lock()

    def ta(self):
        if self._rest is None:
            return True
        with self._laas:
            if self._rest > 0:
                self._rest -= 1
                return True
            return False


def _evaluate_and_draw_error(sladd_doc, csv_doc, truth, folder, png_dir,
                           threshold, y_origin, budget=None):
    """Evaluate one document against truth and draw images for its errors.

    Writes to subfolders bom/ (missed detections) and oversladd/; a page with
    both kinds of error lands in both.
    """
    # write= keeps output off the shared sys.stdout; diagnostikk=False because
    # for a single document "no truth" only means unlabelled.
    doc_eval = evaluate_against_truth(sladd_doc, truth, folder, threshold=threshold,
                            y_origin=y_origin, sources=csv_doc,
                            write=lambda *a, **k: None, diagnostics=False)
    if not doc_eval:
        return

    has_miss = bool(doc_eval.get("miss_files"))
    has_over = bool(doc_eval.get("oversladd_files"))
    if not has_miss and not has_over:
        return
    if budget is not None and not budget.ta():
        return

    miss_indices = {
        (_doc_no(d["fil"]), d["side"], d["fasit_nr"] - 1)
        for d in doc_eval.get("details", [])
        if d["result"] == "MISSING"
    }
    oversladd = doc_eval.get("oversladd_boxes", None)

    miss_pages = {(bf["fil"], bf["side"]) for bf in doc_eval.get("miss_files", [])}
    over_pages = {(of["fil"], of["side"]) for of in doc_eval.get("oversladd_files", [])}

    if miss_pages:
        miss_dir = os.path.join(png_dir, "bom")
        os.makedirs(miss_dir, exist_ok=True)
        sladd_b = {k: v for k, v in sladd_doc.items() if k in miss_pages}
        csv_b = {k: v for k, v in csv_doc.items() if k in miss_pages}
        # Restrict truth to bom pages, else _sider_aa_tegne adds error-free
        # pages and they render as empty PNGs.
        miss_no_pages = {(_doc_no(name), si) for (name, si) in miss_pages}
        truth_miss = {k: v for k, v in truth.items() if k in miss_no_pages} if truth else None
        draw_and_save(sladd_b, truth_miss, folder, miss_dir,
                      y_origin=y_origin, write_log=False, clean=False, sources=csv_b,
                      oversladd_boxes=oversladd, miss_indices=miss_indices)

    if over_pages:
        over_dir = os.path.join(png_dir, "oversladd")
        os.makedirs(over_dir, exist_ok=True)
        sladd_o = {k: v for k, v in sladd_doc.items() if k in over_pages}
        csv_o = {k: v for k, v in csv_doc.items() if k in over_pages}
        over_no_pages = {(_doc_no(name), si) for (name, si) in over_pages}
        truth_over = {k: v for k, v in truth.items() if k in over_no_pages} if truth else None
        draw_and_save(sladd_o, truth_over, folder, over_dir,
                      y_origin=y_origin, write_log=False, clean=False, sources=csv_o,
                      oversladd_boxes=oversladd, miss_indices=miss_indices)


def main():
    p = argparse.ArgumentParser(
        description="Run the model locally as if the files were POSTs: "
                    "bytes -> run_model_on_pdf_bytes. Flags add CSV/PNG/truth/sladding.")
    p.add_argument("--folder", default=DOC_DIR, help="folder of PDFs (plays the role of the POST body)")
    p.add_argument("--select", nargs="*", default=[], help="specific files (filename/substring)")
    p.add_argument("--select-from-file", default=None,
                   help="read file IDs from a text file, one per line, used as --select")
    p.add_argument("--count", default=DEFAULT_FILE_COUNT, help="number of files when --select is empty (a number, or 'alle')")
    p.add_argument("--csv", action="store_true", help="write all found boxes to CSV")
    p.add_argument("--png", action="store_true", help="draw found + truth boxes to PNG")
    p.add_argument("--truth", action="store_true", help="measure recall against truth")
    p.add_argument("--sladd", action="store_true", help="produce actually sladdede PDFs")
    p.add_argument("--ocr-log", action="store_true", help="write the OCR text line by line to file")
    p.add_argument("--truth-csv", default=TRUTH_CSV, help="truth CSV")
    p.add_argument("--csv-out", default=CSV_OUT, help="where the box CSV is written")
    p.add_argument("--ocr-log-file", default=OCR_LOG_FILE, help="where the OCR log is written")
    p.add_argument("--png-dir", default=PNG_DIR, help="where the PNGs are stored")
    p.add_argument("--max-error-images", type=int, default=None,
                   dest="max_error_images", metavar="N",
                   help="draw error images for at most N documents (0 = no "
                        "images). The summary and result CSV are unaffected: "
                        "they are computed from the boxes alone")
    p.add_argument("--sladd-dir", default=SLADD_DIR, help="where sladdede PDFs are stored")
    p.add_argument("--threshold", type=float, default=HIT_THRESHOLD, help="fraction of truth area required for a hit")
    p.add_argument("--y-origin", choices=["top", "bottom"], default=Y_ORIGIN, help="CSV y origin")
    p.add_argument("--elektronisk-tinglyst", action="store_true",
                   help="treat as elektronisk tinglyst: no YOLO, with width filter")
    p.add_argument("--only-yolo", action="store_true",
                   help="run YOLO only, without Paddle OCR")
    p.add_argument("--only-error", action="store_true",
                   help="only generate PNGs for pages with bom or oversladding (needs truth)")
    p.add_argument("--time", action="store_true", help="print timing (render/ocr/postprocessing) per document")
    p.add_argument("--desc", default=None, help="optional suffix in the result folder name")
    p.add_argument("--yolo-weights", default=None,
                   help="path to the YOLO weights file; defaults to $SLADD_PRODWEIGHTS from server.env")
    p.add_argument("--proceed", action="store_true",
                   help="continue where the previous run stopped, skipping files already in the CSV")
    p.add_argument("--processes", type=int, default=None, metavar="N",
                   help="worker processes for cache hits (default: min(8, CPU "
                        "cores); 1 = sequential). Documents without a cache hit "
                        "run in the main process. The per-document table from "
                        "--time is only written sequentially or on a miss")
    p.add_argument("--result-dir", default=".",
                   help="folder where the result-* subfolder is created")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite an existing CSV without asking")
    p.add_argument("--ocr-cache", default=None,
                   help="folder for the per-document OCR cache (tokens + orientation). "
                        "Default: $SLADD_CACHE/<uttrekk name>/ocr/ derived from --folder")
    p.add_argument("--no-ocr-cache", action="store_true",
                   help="disable the OCR cache entirely")
    p.add_argument("--yolo-cache", default=None,
                   help="base folder for the per-document YOLO cache (raw boxes per "
                        "model). The weights hash is appended as a subfolder, so each "
                        "model gets its own cache. Default: $SLADD_CACHE/<uttrekk "
                        "name>/yolo/ derived from --folder")
    p.add_argument("--no-yolo-cache", action="store_true",
                   help="disable the YOLO cache entirely")
    p.add_argument("--without-postfilter", action="store_true",
                   dest="without_postfilter",
                   help="skip ALL postfilters (decimal, line evidence, paddle "
                        "window, profiles, geometry). Raw detection + mod11 + "
                        "dedup. Baseline measurement of what the rules "
                        "contribute; YOLO_CONF and lenient_check still apply")
    p.add_argument("--metadata-csv", default=None, metavar="FIL",
                   help="metadata CSV with rettsstiftelse types per document "
                        "(uttrekk_N.csv). Enables per-document-type rule "
                        "profiles (KOORDFAM_CODES in config) the way prod does "
                        "when the skip job sends the codes; without it, global "
                        "behaviour")
    args = p.parse_args()

    # ── Derive cache paths ───────────────────────────────────────
    if args.no_ocr_cache:
        args.ocr_cache = None
    if args.no_yolo_cache:
        args.yolo_cache = None
    cache_base = os.environ.get("SLADD_CACHE")
    if cache_base:
        uttrekk_name = os.path.basename(os.path.normpath(args.folder))
        if args.ocr_cache is None and not args.no_ocr_cache:
            args.ocr_cache = os.path.join(cache_base, uttrekk_name, "ocr")
        if args.yolo_cache is None and not args.no_yolo_cache:
            args.yolo_cache = os.path.join(cache_base, uttrekk_name, "yolo")

    # ── Validate inputs early ────────────────────────────────────
    if args.select_from_file and not os.path.isfile(args.select_from_file):
        print(f"ERROR: --select-from-file not found: {args.select_from_file}")
        return

    if args.metadata_csv and not os.path.isfile(args.metadata_csv):
        print(f"ERROR: --metadata-csv not found: {args.metadata_csv}")
        return

    if (args.truth or args.only_error) and not os.path.isfile(args.truth_csv):
        print(f"ERROR: --truth-csv not found: {args.truth_csv}")
        print(f"      Set the correct path with --truth-csv /path/to/labels.csv")
        return

    if args.yolo_weights and not os.path.isfile(args.yolo_weights):
        print(f"ERROR: --yolo-weights not found: {args.yolo_weights}")
        return

    if not os.path.isdir(args.folder):
        print(f"ERROR: --folder not found: {args.folder}")
        return

    set_weights(args.yolo_weights)

    # One cache folder per model: the weights hash becomes a subfolder, so a
    # new model never reads another model's boxes.
    if args.elektronisk_tinglyst:
        args.yolo_cache = None          # elektronisk tinglyst runs without YOLO
    if args.yolo_cache:
        if os.path.isfile(active_weights()):
            args.yolo_cache = cache_dir_for_weights(args.yolo_cache, active_weights())
        else:
            print(f"WARNING: weights file {active_weights()} not found - YOLO cache disabled")
            args.yolo_cache = None

    # ── Output folders ───────────────────────────────────────────
    output_mapper = [m for m in [
        os.path.dirname(args.csv_out) if args.csv else None,
        args.png_dir if (args.png or args.only_error) else None,
        args.sladd_dir if args.sladd else None,
        args.result_dir if args.truth else None,
    ] if m]

    if not args.proceed and not args.overwrite:
        existing = [m for m in output_mapper if os.path.isdir(m)]
        if existing:
            print("ERROR: output folder(s) already exist:")
            for m in existing:
                print(f"      {m}")
            print("      Use --proceed to resume, or --overwrite to start over.")
            return

    for folder in output_mapper:
        os.makedirs(folder, exist_ok=True)

    select = args.select
    from_file = False
    if args.select_from_file:
        select = _read_files_from_file(args.select_from_file)
        from_file = True
        print(f"Read {len(select)} IDs from {args.select_from_file}")

    files = select_files(args.folder, select, args.count, exact=from_file)

    if not files:
        print("No files to process - check --folder / --select / --count.")
        return

    def _count_cached(folder):
        return sum(1 for f in files
                   if os.path.isfile(os.path.join(folder,
                       os.path.splitext(os.path.basename(f))[0] + ".json")))

    if args.ocr_cache:
        os.makedirs(args.ocr_cache, exist_ok=True)
        print(f"OCR cache:  {args.ocr_cache} "
              f"({_count_cached(args.ocr_cache)}/{len(files)} documents cached)")

    if args.yolo_cache:
        os.makedirs(args.yolo_cache, exist_ok=True)
        print(f"YOLO cache: {args.yolo_cache} "
              f"({_count_cached(args.yolo_cache)}/{len(files)} documents cached)")

    # Resume: skip files already in the CSV.
    skipped = 0
    if args.proceed and args.csv and os.path.isfile(args.csv_out):
        done = _read_done_from_csv(args.csv_out)
        original = len(files)
        files = [f for f in files if os.path.basename(f) not in done]
        skipped = original - len(files)
        if skipped:
            print(f"--proceed: skipping {skipped} already processed, {len(files)} remaining")
    elif args.csv and not args.proceed:
        if os.path.isfile(args.csv_out) and os.path.getsize(args.csv_out) > 0 and not args.overwrite:
            n_existing = len(_read_done_from_csv(args.csv_out))
            if n_existing:
                print(f"ERROR: {args.csv_out} already contains {n_existing} documents.")
                print(f"      Use --proceed to resume, or --overwrite to start over.")
                return
        write_csv_header(args.csv_out)
        print(f"Writing continuously to {args.csv_out}")

    total_time = 0
    total_count = len(files)

    wants_artifact = args.csv or args.png or args.truth or args.sladd or args.only_error

    truth = None
    if args.png or args.only_error or args.truth:
        truth = read_truth_xywh(args.truth_csv) if os.path.isfile(args.truth_csv) else None
    if args.png or args.only_error:
        os.makedirs(args.png_dir, exist_ok=True)

    sladd_boxes, yolo_boxes, csv_boxes, failed = {}, {}, {}, []
    timings = {}
    ocr_lines = {}                          # (name, page) -> list of (text, marks)
    warned_about_lines = False

    # PNG rendering (CPU) in background threads while the GPU moves on.
    png_executor = ThreadPoolExecutor(max_workers=2) if (args.png and not args.only_error) else None
    png_futures = []
    feil_executor = (ThreadPoolExecutor(max_workers=2)
                     if args.only_error and args.max_error_images != 0 else None)
    image_budget = _ImageBudget(args.max_error_images)
    error_futures = []

    # ── Worker processes for cache hits ──────────────────────────
    processes = args.processes
    if processes is None:
        processes = min(8, os.cpu_count() or 1)
    if processes > 1 and not args.ocr_cache and not args.yolo_cache:
        print("Without a cache every document must go through the models - running sequentially.")
        processes = 1

    start_wall = time.perf_counter()

    def handle_finished(i, name, pages, time_used):
        """Everything that happens to a finished document: log, CSV, PNG, eval."""
        nonlocal total_time, warned_about_lines
        total_time += time_used
        timings[name] = time_used

        if args.ocr_log:
            if not warned_about_lines and pages and "linjer" not in pages[0]:
                print("  !! --ocr-log: model returns flat format without 'linjer' - the log stays empty.")
                warned_about_lines = True
            for page in pages:
                ocr_lines[(name, page["side"])] = page.get("linjer", [])

        n = sum(len(s["boxes"]) for s in pages)
        wall = time.perf_counter() - start_wall
        remains = wall / i * (total_count - i)

        if not wants_artifact:
            print(f"  {n} box(es), {len(pages)} page(s), {time_used:.2f}s (est. remaining: {remains:.0f}s)")
            return

        sladd_doc = {}
        csv_doc = {}
        for page in pages:
            boxes     = [(b["x0"], b["y0"], b["x1"], b["y1"]) for b in page["boxes"]]
            with_source  = [(b["x0"], b["y0"], b["x1"], b["y1"], b.get("kilde", "paddle"),
                           b.get("yolo_conf"), b.get("paddle_rec_score"), b.get("trekk"))
                          for b in page["boxes"]]
            yolo_only  = [(b["x0"], b["y0"], b["x1"], b["y1"]) for b in page["boxes"]
                          if b.get("kilde") in ("yolo", "begge")]
            sladd_boxes[(name, page["side"])] = (
                page["bilde_bredde"], page["bilde_hoyde"], boxes)
            sladd_doc[(name, page["side"])] = (
                page["bilde_bredde"], page["bilde_hoyde"], boxes)
            csv_boxes[(name, page["side"])] = (
                page["bilde_bredde"], page["bilde_hoyde"], with_source)
            csv_doc[(name, page["side"])] = (
                page["bilde_bredde"], page["bilde_hoyde"], with_source)
            if yolo_only:
                yolo_boxes[(name, page["side"])] = (
                    page["bilde_bredde"], page["bilde_hoyde"], yolo_only)

        if args.csv:
            append_csv(csv_doc, args.csv_out)

        # --only-error PNGs wait for the eval instead.
        if png_executor:
            fut = png_executor.submit(
                _draw_continuous, name, pages, args.folder, args.png_dir,
                truth, args.y_origin, csv_doc, sladd_doc)
            png_futures.append(fut)

        if feil_executor and truth:
            fut = feil_executor.submit(
                _evaluate_and_draw_error, sladd_doc, csv_doc,
                truth, args.folder, args.png_dir, args.threshold, args.y_origin,
                image_budget)
            error_futures.append(fut)

        print(f"  {n} box(es), {len(pages)} page(s), {time_used:.2f}s (est. remaining: {remains:.0f}s)")

    # ── Rettsstiftelse types per document (rule profiles, as in prod) ──
    rs_per_doc = {}
    if args.metadata_csv:
        from rettsstiftelse_stat import read_metadata
        meta, _desc = read_metadata(args.metadata_csv)
        rs_per_doc = {doc: codes for doc, (codes, _el) in meta.items()}
        print(f"Metadata: rettsstiftelse types for {len(rs_per_doc)} "
              f"documents, rule profiles (KOORDFAM_CODES) active as in prod")

    def rs_for(name):
        return rs_per_doc.get(_doc_no(name)) if rs_per_doc else None

    def run_in_main_process(pdf_bytes, name):
        """Full run with the models. Returns pages, or None on error."""
        try:
            result = run_model_on_pdf_bytes(pdf_bytes, write_time=args.time, with_lines=args.ocr_log, name=name,
                                              elektronisk_tinglyst=args.elektronisk_tinglyst,
                                              only_yolo=args.only_yolo,
                                              cache_dir=args.ocr_cache,
                                              yolo_cache_dir=args.yolo_cache,
                                              rettsstiftelsestyper=rs_for(name),
                                              postfilter=not args.without_postfilter)
        except Exception as e:
            failed.append((name, repr(e)))
            traceback.print_exc()
            return None
        return _pages_from_result(result, pdf_bytes)

    if processes > 1:
        # ── Parallel cache reads ─────────────────────────────────
        # Results are consumed in submission order, so CSV and output match a
        # sequential run.
        print(f"Parallel cache reads: {processes} processes")
        with ProcessPoolExecutor(max_workers=processes) as pool:
            futures = [pool.submit(_process_from_cache, pdf_path,
                                   args.ocr_cache, args.yolo_cache,
                                   args.elektronisk_tinglyst, args.only_yolo,
                                   args.ocr_log,
                                   rs_for(os.path.basename(pdf_path)),
                                   not args.without_postfilter)
                       for pdf_path in files]
            for i, (pdf_path, fut) in enumerate(zip(files, futures), start=1):
                status, name, payload, time_used = fut.result()
                print(f"\n[{i}/{total_count}] → {name}")
                if status == "feil":
                    failed.append((name, payload))
                    print(payload)
                    continue
                if status == "miss":
                    start = time.perf_counter()
                    try:
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                    except OSError as e:
                        failed.append((name, repr(e)))
                        traceback.print_exc()
                        continue
                    pages = run_in_main_process(pdf_bytes, name)
                    if pages is None:
                        continue
                    time_used = time.perf_counter() - start
                else:
                    pages = payload
                handle_finished(i, name, pages, time_used)
    else:
        # ── Sequential run ───────────────────────────────────────
        # Pre-read the next file while the GPU works.
        next_bytes = None
        if files:
            with open(files[0], "rb") as f:
                next_bytes = f.read()

        for i, pdf_path in enumerate(files, start=1):
            start = time.perf_counter()

            name = os.path.basename(pdf_path)
            print(f"\n[{i}/{total_count}] → {name}")

            pdf_bytes = next_bytes

            if i < total_count:
                with open(files[i], "rb") as f:
                    next_bytes = f.read()

            pages = run_in_main_process(pdf_bytes, name)
            if pages is None:
                continue

            handle_finished(i, name, pages, time.perf_counter() - start)

    wall_time = time.perf_counter() - start_wall
    print(f"\nDone! {total_count} documents in {wall_time:.1f}s ({wall_time/max(total_count,1):.2f}s/doc)")

    if failed:
        print(f"Failed ({len(failed)}):", failed[:5])

    if args.ocr_log:
        n = _write_ocr_log(ocr_lines, args.ocr_log_file)
        print(f"Wrote OCR lines for {n} page(s) to {args.ocr_log_file}")

    if not wants_artifact and not args.only_error:
        return

    eval_result = None
    if args.truth or args.only_error:
        buf = io.StringIO()
        eval_result = evaluate_against_truth(sladd_boxes, truth, args.folder, threshold=args.threshold,
                                     y_origin=args.y_origin, sources=csv_boxes,
                                     write=lambda *a, **k: print(*a, **k, file=buf))
        log = buf.getvalue()
        print(log, end="")
        if args.truth:
            time_lines = "".join(f"  {n}: {t:.2f}s\n" for n, t in sorted(timings.items()))
            header = (
                f"Folder:     {os.path.abspath(args.folder)}\n"
                f"Truth CSV:  {os.path.abspath(args.truth_csv)}\n"
                f"Total time: {total_time:.2f}s\n"
                f"Time per document:\n{time_lines}\n"
            )
            write_result_files(eval_result, folder=args.result_dir,
                               description=args.desc, log=header + log)

    # Wait for the background images only now: the summary above is computed
    # from the boxes alone and must not block on PNG rendering.
    if png_futures:
        print(f"\nWaiting for {len(png_futures)} background PNG jobs...")
        for fut in png_futures:
            try:
                fut.result()
            except Exception as e:
                print(f"  PNG error: {e!r}")
        png_executor.shutdown(wait=False)

    if error_futures:
        print(f"\nWaiting for {len(error_futures)} background error-image jobs...")
        for fut in error_futures:
            try:
                fut.result()
            except Exception as e:
                print(f"  Error-image error: {e!r}")
        feil_executor.shutdown(wait=False)

    # Fallback for --only-error without the executor. With --max-error-images 0
    # the executor is dropped on purpose, so the fallback must not draw either.
    if args.only_error and eval_result and not error_futures \
            and args.max_error_images != 0:
        miss_indices = {
            (_doc_no(d["fil"]), d["side"], d["fasit_nr"] - 1)
            for d in eval_result.get("details", [])
            if d["result"] == "MISSING"
        }
        oversladd = eval_result.get("oversladd_boxes", None)
        miss_pages = {(bf["fil"], bf["side"]) for bf in eval_result.get("miss_files", [])}
        over_pages = {(of["fil"], of["side"]) for of in eval_result.get("oversladd_files", [])}
        every_error_pages = miss_pages | over_pages
        print(f"\n--only-error: drawing {len(every_error_pages)} page(s) with errors (bom/ and oversladd/)")

        if miss_pages:
            miss_dir = os.path.join(args.png_dir, "bom")
            os.makedirs(miss_dir, exist_ok=True)
            sladd_b = {k: v for k, v in sladd_boxes.items() if k in miss_pages}
            csv_b = {k: v for k, v in csv_boxes.items() if k in miss_pages}
            miss_no_pages = {(_doc_no(name), si) for (name, si) in miss_pages}
            truth_miss = {k: v for k, v in truth.items() if k in miss_no_pages} if truth else None
            draw_and_save(sladd_b, truth_miss, args.folder, miss_dir,
                          y_origin=args.y_origin, sources=csv_b,
                          oversladd_boxes=oversladd, miss_indices=miss_indices)

        if over_pages:
            over_dir = os.path.join(args.png_dir, "oversladd")
            os.makedirs(over_dir, exist_ok=True)
            sladd_o = {k: v for k, v in sladd_boxes.items() if k in over_pages}
            csv_o = {k: v for k, v in csv_boxes.items() if k in over_pages}
            over_no_pages = {(_doc_no(name), si) for (name, si) in over_pages}
            truth_over = {k: v for k, v in truth.items() if k in over_no_pages} if truth else None
            draw_and_save(sladd_o, truth_over, args.folder, over_dir,
                          y_origin=args.y_origin, sources=csv_o,
                          oversladd_boxes=oversladd, miss_indices=miss_indices)

    if args.sladd:
        sladd_files(sladd_boxes, args.folder, args.sladd_dir)


if __name__ == "__main__":
    main()