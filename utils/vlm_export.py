"""Exports image crops of proposed sladdebokser for VLM judging.

Step 1 of the VLM verifier pilot. Reuses build_dataset from filter_common and
the crop machinery from filter_review, and writes one PNG crop per prediction
plus a manifest.

The selection is ASYMMETRIC on purpose: all BOM boxes (the oversladdinger)
plus a random sample of covering boxes (TREFF/SLURV). The sampling factors go
to utvalg.json next to the manifest. vlm_evaluate needs them to scale the
loss estimate up to the full uttrekk.

Run:
    python utils/vlm_export.py \
        --res-csv  $SLADD_VALIDATION/uttrekk6_frreg/resultat.csv \
        --truth-csv $SLADD_LABELS/uttrekk_6.csv \
        --folder    $SLADD_UTTREKK/uttrekk_6 \
        --out-dir /data2/vlm/uttrekk6_kalibrering \
        --processed-list /data2/tmp/rs_lister/rs_TR_MAS.txt \
        --hit-sample 0 \
        --ocr-cache $SLADD_CACHE/uttrekk_6/ocr
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz
from PIL import Image, ImageDraw

from filter_common import (CRITERIA, PDF_DPI, SCALE, STD_CRITERION,
                           STD_SLOPPINESS_FACTOR, HIT_THRESHOLD, build_dataset,
                           doc_no, read_truth_boxes, read_truth_rows,
                           read_processed_docs, read_predictions,
                           write_summary)
from filter_review import _read_ocr_cache, rotate_box, _render_page

MARKER = (230, 20, 20)        # red frame around the box being judged
MARKER_WIDTH = 3
MARKER_SPACE = 3               # px outside the box, so digits stay visible

STD_MARGIN = 60.0             # context margin in PDF points
# The smaller max-px, the faster the model (but lower quality image is used)
STD_MAX_PX = 800             # crops wider than this are scaled down
STD_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))

MANIFEST_FIELD = [
    "nr", "utsnitt", "fil", "doc_no", "side", "klasse", "kilde", "conf",
    "paddle_rec", "label_id", "dekkere",
    "x0", "y0", "x1", "y1", "bredde_pt", "hoyde_pt", "elongation",
    "kortside_pt", "langside_pt",
    "har_tokens", "n_siffer", "siffer_run", "har_fnr_kandidat",
    "har_desimal_naer", "rec_min", "rec_min_linje",
    "ocr_tekst", "ocr_linje", "ocr_blokk",
    "utsnitt_bredde", "utsnitt_hoyde", "m_x0", "m_y0", "m_x1", "m_y1",
    "vurdering",
]


# ── Selection ─────────────────────────────────────────────────

def select_boxes(ds, hit_sample, seed=42, max_miss=None, sources=None):
    """All BOM + a deterministic sample of covering boxes.

    Returns (selected, stats). "Covering" is TREFF and SLURV together: both
    cover a fasit box, and both cost recall if the VLM says no.
    """
    sources = {k.lower() for k in sources} if sources else None
    i_scope = [p for p in ds.pred
               if sources is None or p["kilde"].lower() in sources]

    miss = [p for p in i_scope if p["klasse"] == "BOM"]
    covering = [p for p in i_scope if p["klasse"] != "BOM"]

    # Sort before drawing: the sample must be identical across runs no matter
    # what order the rows come in.
    sort_key = lambda p: (p["navn"], p["side"], p["px"][1], p["px"][0], p["kilde"])
    miss.sort(key=sort_key)
    covering.sort(key=sort_key)

    rng = random.Random(seed)
    selected_miss = miss
    if max_miss is not None and max_miss < len(miss):
        selected_miss = sorted(rng.sample(miss, max_miss), key=sort_key)

    chosen_covering = covering
    if hit_sample is not None and hit_sample < len(covering):
        chosen_covering = sorted(rng.sample(covering, hit_sample), key=sort_key)

    stat = {
        "n_bom_total": len(miss),
        "n_bom_exported": len(selected_miss),
        "n_covering_total": len(covering),
        "n_covering_exported": len(chosen_covering),
        "bom_factor": (len(miss) / len(selected_miss)) if selected_miss else 0.0,
        "hit_factor": ((len(covering) / len(chosen_covering))
                         if chosen_covering else 0.0),
        "seed": seed,
        "sources": sorted(sources) if sources else "all",
    }
    return selected_miss + chosen_covering, stat


# ── OCR context ───────────────────────────────────────────────

def _lines(tokens):
    """Groups loose OCR tokens into text lines, top to bottom."""
    rest = sorted((t for t in tokens if t.text.strip()), key=lambda t: t.y0)
    lines = []
    for t in rest:
        for lin in lines:
            h = min(lin["y1"] - lin["y0"], t.y1 - t.y0)
            if min(lin["y1"], t.y1) - max(lin["y0"], t.y0) > 0.3 * max(h, 1.0):
                lin["tokens"].append(t)
                lin["y0"] = min(lin["y0"], t.y0)
                lin["y1"] = max(lin["y1"], t.y1)
                break
        else:
            lines.append({"y0": t.y0, "y1": t.y1, "tokens": [t]})
    for lin in lines:
        lin["tokens"].sort(key=lambda t: t.x0)
        lin["text"] = " ".join(t.text for t in lin["tokens"])
    lines.sort(key=lambda l: l["y0"])
    return lines


def _ocr_context(tokens, rect, n_lines=0):
    """(text inside the box, text on its line, block of neighbouring lines).

    rekt must be in the ROTATED pixel space, the same space the tokens are in.
    """
    x0, y0, x1, y1 = rect
    height = max(y1 - y0, 1.0)
    i_box, i_line = [], []
    for t in tokens:
        if not t.text.strip():
            continue
        v = min(y1, t.y1) - max(y0, t.y0)
        if v <= 0.3 * min(height, max(t.y1 - t.y0, 1.0)):
            continue
        i_line.append(t)
        if min(x1, t.x1) - max(x0, t.x0) > 0:
            i_box.append(t)
    i_box.sort(key=lambda t: t.x0)
    i_line.sort(key=lambda t: t.x0)
    box_text = " ".join(t.text for t in i_box)
    line_text = " ".join(t.text for t in i_line)

    block = ""
    if n_lines > 0 and tokens:
        lines = _lines(tokens)
        mid = max(range(len(lines)),
                   key=lambda i: min(y1, lines[i]["y1"]) - max(y0, lines[i]["y0"]),
                   default=None) if lines else None
        if mid is not None:
            lo = max(0, mid - n_lines)
            block = "\n".join(l["text"] for l in lines[lo:mid + n_lines + 1])
    return box_text, line_text, block


# ── Export ────────────────────────────────────────────────────

def _job_for_file(task):
    """Renders, crops and saves every box in ONE PDF.

    Runs in its own process, so it takes and returns plain values only.
    Warnings are collected and printed by the parent, otherwise they would
    interleave on screen.
    """
    (name, preds, folder, crop_dir, margin_x, margin_y, full_width,
     from_top, max_px, ocr_dir, roter, n_lines) = task

    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        return [], [f"Cannot find {path}, skipping"], len(preds)
    try:
        doc = fitz.open(path)
    except Exception as e:
        return [], [f"Could not open {name}: {e!r}"], len(preds)

    cache = _read_ocr_cache(ocr_dir, name) if ocr_dir else None
    per_page = defaultdict(list)
    for p in preds:
        per_page[p["side"]].append(p)

    rows, warnings, dropped = [], [], 0
    for si in sorted(per_page):
        page_pred = per_page[si]
        if not 1 <= si <= len(doc):
            dropped += len(page_pred)
            continue
        image = _render_page(doc, si)
        w0, h0 = image.width, image.height
        # bw/bh = 0 is the against-truth sentinel: the coordinates are already in
        # render space (pt × SCALE).
        bw = page_pred[0]["bw"] or w0
        bh = page_pred[0]["bh"] or h0
        sx, sy = w0 / bw, h0 / bh

        # Same rotation the pipeline OCR-ed with, or the text sits sideways
        # in crops of a landscape scan.
        k, tokens = 0, []
        if cache:
            rotations, tokens_per_page = cache
            if si <= len(rotations):
                k = (rotations[si - 1] or 0) if roter else 0
                tokens = tokens_per_page[si - 1]
        if k:
            image = image.rotate(90 * k, expand=True)

        mx = margin_x * SCALE
        m_up, m_ned = margin_y[0] * SCALE, margin_y[1] * SCALE
        for p in page_pred:
            px = p["px"]
            r = rotate_box([px[0] * sx, px[1] * sy, px[2] * sx, px[3] * sy],
                           k, w0, h0)
            # A label drawn past the page edge, an odd CropBox: one strange
            # row must not sink an export of tens of thousands of boxes.
            if (r[2] <= 0 or r[3] <= 0
                    or r[0] >= image.width or r[1] >= image.height):
                dropped += 1
                warnings.append(
                    f"{name} p{si}: box outside the page "
                    f"({r[0]:.0f},{r[1]:.0f},{r[2]:.0f},{r[3]:.0f} "
                    f"in {image.width}x{image.height}), skipped")
                continue
            left = 0 if full_width else max(0, int(r[0] - mx))
            right = (image.width if full_width
                     else min(image.width, int(r[2] + mx)))
            top = 0 if from_top else max(0, int(r[1] - m_up))
            box = (left, top, right,
                    min(image.height, int(r[3] + m_ned)))
            if box[2] <= box[0] or box[3] <= box[1]:
                dropped += 1
                continue
            ut = image.crop(box).convert("RGB")

            m = [r[0] - box[0], r[1] - box[1],
                 r[2] - box[0], r[3] - box[1]]
            if max_px and ut.width > max_px:
                f = max_px / ut.width
                ut = ut.resize((max_px, max(1, int(ut.height * f))),
                               Image.LANCZOS)
                m = [v * f for v in m]
            # A 3 px frame disappears in a full-page crop; the padding grows
            # with the line so the digits stay visible.
            stroke = max(MARKER_WIDTH, round(ut.width / 400))
            pad = stroke + 2
            m = [max(0, m[0] - pad), max(0, m[1] - pad),
                 min(ut.width - 1, m[2] + pad),
                 min(ut.height - 1, m[3] + pad)]
            if m[2] <= m[0] or m[3] <= m[1]:
                dropped += 1
                warnings.append(f"{name} p{si}: marker has no area after "
                                 f"clipping, skipped")
                continue
            ImageDraw.Draw(ut).rectangle(m, outline=MARKER, width=stroke)

            base = os.path.splitext(os.path.basename(name))[0]
            filename = f"{p['_nr']:05d}_{p['klasse']}_{base}_s{si}.png"
            ut.save(os.path.join(crop_dir, filename))

            i_box = i_line = i_block = ""
            if tokens:
                # Rotation is linear, so it can be applied straight to the
                # CSV coordinates instead of via the rendering.
                i_box, i_line, i_block = _ocr_context(
                    tokens, rotate_box(list(px), k, bw, bh), n_lines)

            rows.append({
                "nr": p["_nr"], "utsnitt": filename, "fil": name,
                "doc_no": p["doc_no"], "side": si, "klasse": p["klasse"],
                "kilde": p["kilde"],
                "conf": p["conf"] if p["conf"] is not None else "",
                "paddle_rec": (p["paddle_rec"]
                               if p["paddle_rec"] is not None else ""),
                "label_id": p["_label_id"], "dekkere": p["_dekkere"],
                "x0": round(px[0], 1), "y0": round(px[1], 1),
                "x1": round(px[2], 1), "y1": round(px[3], 1),
                "bredde_pt": round(p["w"], 1), "hoyde_pt": round(p["h"], 1),
                "elongation": round(p["elongation"], 2),
                "kortside_pt": round(p["short_side"], 1),
                "langside_pt": round(p["long_side"], 1),
                "har_tokens": _number(p.get("har_tokens")),
                "n_siffer": _number(p.get("n_siffer")),
                "siffer_run": _number(p.get("siffer_run")),
                "har_fnr_kandidat": _number(p.get("har_fnr_kandidat")),
                "har_desimal_naer": _number(p.get("har_desimal_naer")),
                "rec_min": _number(p.get("rec_min")),
                "rec_min_linje": _number(p.get("rec_min_linje")),
                "ocr_tekst": i_box, "ocr_linje": i_line,
                "ocr_blokk": i_block,
                "utsnitt_bredde": ut.width, "utsnitt_hoyde": ut.height,
                "m_x0": round(m[0], 1), "m_y0": round(m[1], 1),
                "m_x1": round(m[2], 1), "m_y1": round(m[3], 1),
                "vurdering": "",
            })
    doc.close()
    return rows, warnings, dropped


def export(ds, selected, folder, out_dir, margin_x=STD_MARGIN,
              margin_y=STD_MARGIN, full_width=False, from_top=False,
              max_px=STD_MAX_PX, ocr_dir=None, roter=True, workers=1,
              n_lines=0):
    """Renders, crops and writes the manifest. Returns the manifest rows.

    The margin is asymmetric because the context is: the ledetekst that says
    whether a number is a fnr or a coordinate sits on the same line or above
    it, so width buys more than height and up more than down. margin_y is one
    number or an (up, down) pair. Work is split per DOCUMENT across processes.
    """
    try:
        margin_y = (float(margin_y[0]), float(margin_y[1]))
    except TypeError:
        margin_y = (float(margin_y), float(margin_y))
    crop_dir = os.path.join(out_dir, "utsnitt")
    os.makedirs(crop_dir, exist_ok=True)

    # Numbering is assigned here, before the work is distributed, so it does
    # not depend on which process finishes first.
    selected = sorted(selected, key=lambda p: (p["navn"], p["side"],
                                           p["px"][1], p["px"][0], p["kilde"]))
    per_file = defaultdict(list)
    for nr, p in enumerate(selected, 1):
        p["_nr"] = nr
        if "_label_id" not in p:
            p["_label_id"] = ";".join(
                i for i in (ds.truth_boxes[j]["label_id"]
                            for j in p["covers"]) if i)
            p["_dekkere"] = (min(ds.coverage_before[j] for j in p["covers"])
                             if p["covers"] else 0)
        per_file[p["navn"]].append(p)

    tasks = [(name, per_file[name], folder, crop_dir, margin_x, margin_y,
                 full_width, from_top, max_px, ocr_dir, roter,
                 n_lines)
                for name in sorted(per_file)]
    print(f"\n  {len(selected)} boxes in {len(tasks)} documents"
          f"  ({workers} parallel processes)")

    rows, warnings, dropped = [], [], 0
    t0 = time.monotonic()
    n_file = 0

    def progress(nye):
        nonlocal n_file, dropped
        r, a, d = nye
        rows.extend(r)
        warnings.extend(a)
        dropped += d
        n_file += 1
        if n_file % 25 == 0 or n_file == len(tasks):
            gone = time.monotonic() - t0
            speed = n_file / gone if gone else 0
            eta = (len(tasks) - n_file) / speed if speed else 0
            print(f"    {n_file:>6}/{len(tasks)} doc  {gone:6.0f}s  "
                  f"{speed:5.1f} doc/s  {len(rows):>6} crops  "
                  f"ETA {eta:5.0f}s", flush=True)

    if workers <= 1:
        for task in tasks:
            progress(_job_for_file(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for nye in pool.map(_job_for_file, tasks, chunksize=1):
                progress(nye)

    for melding in warnings:
        print(f"  ⚠ {melding}")
    if dropped:
        print(f"  ⚠ {dropped} boxes dropped, missing PDF, invalid page "
              f"number or empty crop")

    rows.sort(key=lambda r: r["nr"])
    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELD, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  {len(rows)} crops in {crop_dir}")
    print(f"  Manifest: {manifest_path}  ({len(rows)} rows)")
    return rows


def _number(v):
    return "" if v is None else v


# ── CLI ───────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Exports PNG crops of proposed sladdebokser + a manifest, "
                    "as input to VLM judging (vlm_judge.py).")
    p.add_argument("--res-csv", default=None,
                   help="Result CSV from the model (not needed with "
                        "--against-truth)")
    p.add_argument("--against-truth", action="store_true", dest="against_truth",
                   help="Export the FASIT boxes (labels) instead of the "
                        "model's predictions, the VLM as fasit auditor. All "
                        "labels are included (no sampling factors). Rotation "
                        "and OCR context need --ocr-cache.")
    p.add_argument("--truth-csv", required=True, help="Labels CSV (fasit)")
    p.add_argument("--folder", required=True, help="Directory with the PDFs")
    p.add_argument("--out-dir", required=True, help="Directory for crops+manifest")

    p.add_argument("--hit-sample", type=int, default=1000, metavar="N",
                   help="Covering boxes (TREFF/SLURV) to draw. 0 = none "
                        "(pure BOM calibration), -1 = all. Default 1000.")
    p.add_argument("--max-miss", type=int, default=None, metavar="N",
                   help="Take only N random BOM boxes (default: all)")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed (42)")
    p.add_argument("--source", nargs="+", default=None, metavar="KILDE",
                   help="Limit to these sources (yolo/paddle/begge)")

    p.add_argument("--margin", type=float, default=STD_MARGIN, metavar="PT",
                   help=f"Context margin around the box in points "
                        f"(default {STD_MARGIN:g}). Overridden per axis by "
                        f"--margin-x/--margin-y.")
    p.add_argument("--margin-x", type=float, default=None, metavar="PT",
                   help="Sideways margin. The ledetekst that reveals what the "
                        "number is usually sits on the same line, so width is "
                        "worth more than height. 200-250 covers a text line.")
    p.add_argument("--margin-y", type=float, default=None, metavar="PT",
                   help="Margin up/down. 90 gives ~2 text lines each way, "
                        "enough to see a column heading.")
    p.add_argument("--margin-up", type=float, default=None, metavar="PT",
                   help="Margin upwards only. Headings sit ABOVE the number, "
                        "so more up than down buys context without as much "
                        "junk below.")
    p.add_argument("--margin-down", type=float, default=None, metavar="PT",
                   help="Margin downwards only.")
    p.add_argument("--from-top", action="store_true",
                   help="Include everything from the top of the page down to "
                        "the box, for form headings too far up to reach with "
                        "--margin-y. Gives tall crops and many image tokens, "
                        "raise the server's context to match.")
    p.add_argument("--full-width", action="store_true",
                   help="Take the whole page width instead of --margin-x. "
                        "Always catches the full line, at the cost of "
                        "resolution after downscaling, consider --max-px.")
    p.add_argument("--max-px", type=int, default=STD_MAX_PX, metavar="PX",
                   help=f"Scale down crops wider than this "
                        f"(default {STD_MAX_PX}). 0 = no scaling.")
    p.add_argument("--ocr-cache", default=None, metavar="STI",
                   help="OCR cache directory ($SLADD_CACHE/uttrekk_N/ocr). "
                        "Fills ocr_tekst/ocr_linje in the manifest and "
                        "straightens sideways scans.")
    p.add_argument("--workers", type=int, default=STD_WORKERS, metavar="N",
                   help=f"Parallel processes, one PDF at a time each "
                        f"(default {STD_WORKERS} on this machine). Rendering "
                        f"is CPU-bound, so this is the only lever that "
                        f"matters for export time.")
    p.add_argument("--ocr-lines", type=int, default=0, metavar="N",
                   help="Include N OCR lines above and below the box's own "
                        "line in the ocr_blokk column. A fnr can be split "
                        "across a line break; 2 is a sensible start. "
                        "Requires --ocr-cache.")
    p.add_argument("--no-rotate", dest="roter", action="store_false",
                   help="Do not straighten the page even if the OCR cache "
                        "has a rotation")

    p.add_argument("--processed-list", default=None, metavar="FIL",
                   help="File listing the documents the model ran on, same "
                        "semantics as in filter_review/filter_sweep. Use "
                        "rs_<CODE>.txt for one rettsstiftelse type.")
    p.add_argument("--criterion", default=STD_CRITERION, choices=sorted(CRITERIA),
                   help=f"Coverage match rule (default {STD_CRITERION})")
    p.add_argument("--threshold", type=float, default=HIT_THRESHOLD,
                   help=f"Overlap threshold (default {HIT_THRESHOLD})")
    p.add_argument("--oversize-factor", type=float, default=STD_SLOPPINESS_FACTOR,
                   help=f"SLURV limit (default {STD_SLOPPINESS_FACTOR})")
    p.add_argument("--include-unlabelled", action="store_true", default=True,
                   help="(default on) Processed documents without fasit rows "
                        "were reviewed and hold zero fnr, predictions there "
                        "are BOM")
    p.add_argument("--exclude-unlabelled", dest="include_unlabelled",
                   action="store_false",
                   help="Old behaviour: only documents with fasit rows")
    a = p.parse_args()

    processed = read_processed_docs(a.processed_list) if a.processed_list else None
    if a.against_truth:
        # Every label is a sladd a human made, so a «nei» here is a claim of
        # label noise. Read the outcome in gjennomgang_label.md.
        ds = None
        rows_f, excluded, _col = read_truth_rows(a.truth_csv)
        name_for = {}
        for fn in sorted(os.listdir(a.folder)):
            if fn.lower().endswith(".pdf"):
                nr = doc_no(fn)
                if nr is not None:
                    name_for.setdefault(nr, fn)
        selected, missing = [], 0
        for r in rows_f:
            if processed is not None and r["doc_no"] not in processed:
                continue
            name = name_for.get(r["doc_no"])
            if name is None:
                missing += 1
                continue
            x0, y0, x1, y1 = r["box"]
            selected.append({
                "navn": name, "doc_no": r["doc_no"], "side": r["side"],
                "px": [x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE],
                "bw": 0, "bh": 0, "klasse": "FASIT", "kilde": "fasit",
                "conf": None, "paddle_rec": None,
                "w": r["w"], "h": r["h"], "short_side": r["short_side"],
                "long_side": r["long_side"], "elongation": r["elongation"],
                "_label_id": (r["row"].get("id") or "").strip(),
                "_dekkere": "",
            })
        stat = {"against_truth": True,
                "n_bom_total": 0, "n_bom_exported": 0, "bom_factor": 0.0,
                "n_covering_total": len(selected),
                "n_covering_exported": len(selected), "hit_factor": 1.0,
                "seed": a.seed, "sources": "fasit"}
        print(f"MOT-FASIT: exporting labels, not predictions")
        print(f"  Labels read:    {len(rows_f)}"
              + (f"  (excluded: "
                 + ", ".join(f"{k}={v}" for k, v in sorted(excluded.items()))
                 + ")" if excluded else ""))
        if processed is not None:
            print(f"  --processed-list: {len(selected) + missing} within the list")
        if missing:
            print(f"  No PDF in {a.folder}: {missing}, skipped")
        print(f"  Exporting:      {len(selected)}")
    else:
        if not a.res_csv:
            p.error("--res-csv is required (except with --against-truth)")
        truth = read_truth_boxes(a.truth_csv)
        pred = read_predictions(a.res_csv)
        ds = build_dataset(truth, pred, threshold=a.threshold,
                           oversize_factor=a.oversize_factor,
                           include_unlabelled=a.include_unlabelled,
                           processed_doc=processed, criterion=a.criterion)
        write_summary(ds)

        hit_sample = None if a.hit_sample < 0 else a.hit_sample
        selected, stat = select_boxes(ds, hit_sample, seed=a.seed,
                                   max_miss=a.max_miss, sources=a.source)
        print(f"\nSelection for VLM judging:")
        print(f"  BOM:      {stat['n_bom_exported']:>6} of "
              f"{stat['n_bom_total']}   (factor {stat['bom_factor']:.2f})")
        print(f"  Covering: {stat['n_covering_exported']:>6} of "
              f"{stat['n_covering_total']}   (factor {stat['hit_factor']:.2f})")
    if not selected:
        print("  Nothing to export.")
        return

    os.makedirs(a.out_dir, exist_ok=True)
    margin_x = a.margin if a.margin_x is None else a.margin_x
    margin_y = a.margin if a.margin_y is None else a.margin_y
    margin_up = margin_y if a.margin_up is None else a.margin_up
    margin_down = margin_y if a.margin_down is None else a.margin_down
    rows = export(ds, selected, a.folder, a.out_dir, margin_x=margin_x,
                      margin_y=(margin_up, margin_down),
                      full_width=a.full_width,
                      from_top=a.from_top,
                      max_px=a.max_px or 0, ocr_dir=a.ocr_cache,
                      roter=a.roter, workers=a.workers,
                      n_lines=a.ocr_lines)

    # Recount from what was ACTUALLY written: boxes in missing PDFs drop out
    # above, and the original factors would then scale the loss estimate wrong.
    written = defaultdict(int)
    for r in rows:
        written["bom" if r["klasse"] == "BOM" else "covering"] += 1
    stat["n_bom_exported"] = written["bom"]
    stat["n_covering_exported"] = written["covering"]
    stat["bom_factor"] = (stat["n_bom_total"] / written["bom"]
                          if written["bom"] else 0.0)
    stat["hit_factor"] = (stat["n_covering_total"] / written["covering"]
                            if written["covering"] else 0.0)
    stat.update({
        "margin_x_pt": margin_x, "margin_y_pt": [margin_up, margin_down],
        "full_bredde": a.full_width, "from_top": a.from_top,
        "maks_px": a.max_px,
        "res_csv": os.path.abspath(a.res_csv) if a.res_csv else None,
        "fasit_csv": os.path.abspath(a.truth_csv),
        "folder": os.path.abspath(a.folder),
        "kjorte_liste": (os.path.abspath(a.processed_list)
                         if a.processed_list else None),
        "kriterium": a.criterion, "threshold": a.threshold,
        "slurv_faktor": a.oversize_factor,
        "inkluder_ulabelte": a.include_unlabelled,
        "n_fasit_i_scope": ds.n_truth if ds else len(selected),
        "n_covered_before": ds.covered_before if ds else None,
        "n_doc_in_scope": (len(ds.scope_doc) if ds
                          else len({p["doc_no"] for p in selected})),
    })
    with open(os.path.join(a.out_dir, "utvalg.json"), "w",
              encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)
    print(f"  Sampling factors: {os.path.join(a.out_dir, 'utvalg.json')}")
    print("\n  Next step:  python utils/vlm_judge.py --manifest "
          f"{os.path.join(a.out_dir, 'manifest.csv')} --url ... --model ...")


if __name__ == "__main__":
    main()
