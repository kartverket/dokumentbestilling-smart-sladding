"""
Visual review of the boxes a filter configuration removes.

Scored ground-truth-centrically like filter_sweep: removing a prediction only
costs recall if no other prediction still covers the same ground-truth box.
Pages are sorted and foldered by that distinction.

Colors: magenta = ground-truth box that lost all coverage, red = the removed
prediction that was its only cover, orange = removed but redundant cover,
green = removed oversladding, grey = kept, blue = still-covered ground truth.

Run:
    python utils/filter_review.py \\
        --truth-csv labels.csv --res-csv resultat.csv --folder /path/to/pdfs \\
        --per-source "paddle:e=1.8,h=80,b=150" "yolo:e=1.5,h=40,b=150,c=0.5"
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz
from PIL import Image, ImageDraw, ImageFont

from filter_common import (CRITERIA, CRITERION_FIELDS, CRITERION_LOW_IS_GOOD,
                           PDF_DPI, SCALE, STD_CRITERION, STD_SLOPPINESS_FACTOR,
                           HIT_THRESHOLD,
                           build_dataset, doc_no, measure_filter, rejection_reasons,
                           match_metrics,
                           make_filter, make_filter_per_source, read_truth_boxes, read_processed_docs,
                           read_truth_rows, read_predictions, parse_per_source,
                           write_summary, FILTER_PARAMS, FILTER_PARAMETERS)
from geometry import intersection_area

# ── Colors ────────────────────────────────────────────────────
LOST_TRUTH         = (230, 0, 200, 255)     # magenta
CRITICAL_REMOVED    = (220, 30, 30, 230)     # red
REDUNDANT_REMOVED  = (235, 150, 20, 230)    # orange
OVERSLADD_REMOVED  = (30, 180, 30, 220)     # green
MISS                = (255, 90, 0, 235)      # orange
KEPT            = (160, 160, 160, 100)   # grey
TRUTH              = (30, 80, 220, 140)     # blue

_FONT_SMALL = None


def _font_small():
    global _FONT_SMALL
    if _FONT_SMALL is None:
        _FONT_SMALL = ImageFont.load_default(size=16)
    return _FONT_SMALL


def _draw_text(drawer, r, text, color, over=True):
    y = max(r[1] - 24, 2) if over else r[3] + 4
    drawer.text((r[0] + 2, y), text, fill=color, font=_font_small())


# ── Labels ─────────────────────────────────────────────────

def _label(kwargs):
    """Human-readable filter, e.g. "e≥2, h≤20, no-orgnr"."""
    parts = [fp.label.format(kwargs[fp.name]) for fp in FILTER_PARAMS
             if kwargs.get(fp.name) is not None]
    return ", ".join(parts) if parts else "no filter"


def _label_per_source(per_source):
    return " + ".join(f"{k}({_label(kw)})" for k, kw in sorted(per_source.items()))


def _dir_name(kwargs):
    """Output folder for one filter. Uses dir_code, which is shorter than the
    --per-source code and does not always match it; see FILTER_PARAMS."""
    parts = [f"{fp.dir_code}{kwargs[fp.name]:g}" for fp in FILTER_PARAMS
             if kwargs.get(fp.name) is not None]
    return "_".join(parts) if parts else "ingen_filter"


def _dir_name_per_source(per_source):
    return "__".join(f"{k}_{_dir_name(kw)}" for k, kw in sorted(per_source.items()))


# ── Rendering ─────────────────────────────────────────────────

def _render_page(doc, si):
    pix = doc[si - 1].get_pixmap(dpi=PDF_DPI)
    mode = "RGBA" if pix.n == 4 else "RGB"
    return Image.frombytes(mode, (pix.w, pix.h), pix.samples).convert("RGB")


def _overlay(image):
    """Fresh transparent layer over a rendered page: (base, overlay, drawer)."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    return base, overlay, ImageDraw.Draw(overlay)


def _flatten(base, overlay):
    return Image.alpha_composite(base, overlay).convert("RGB")


def _crop_box(image, rect, margin):
    """rect grown by margin and clamped to the image, None if nothing is left.

    margin is one number, or (x, y) where the page scale differs per axis.
    """
    mx, my = margin if isinstance(margin, tuple) else (margin, margin)
    box = (max(0, int(rect[0] - mx)), max(0, int(rect[1] - my)),
           min(image.width, int(rect[2] + mx)),
           min(image.height, int(rect[3] + my)))
    return box if box[2] > box[0] and box[3] > box[1] else None


def _save_crop(image, rect, margin, path):
    """Writes one crop; False when the box has no area left after clamping."""
    box = _crop_box(image, rect, margin)
    if box is None:
        return False
    image.crop(box).save(path)
    return True


def _write_manifest(out_dir, filename, rows, fields):
    """Writes a review manifest under out_dir and returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def iter_pdf_pages(folder, work, file_key=None, page_key=None, on_missing=None):
    """Opens each PDF once and yields (name, page number, items, doc) per page.

    work maps file name -> {page number: items}, where items is whatever the
    caller needs on that page and comes straight back out. All pages of one
    file arrive together, so a per-file cache can be rebuilt when the name
    changes. file_key orders the files, page_key(name, si) orders one file's
    pages; both fall back to natural order.

    A file that is missing or unreadable, and a page number outside the
    document, is reported and skipped. on_missing(name, items) then runs once
    per skipped page, so the caller can count what it never got.
    """
    for name in sorted(work, key=file_key):
        pages = work[name]
        path = os.path.join(folder, name)
        doc = problem = None
        if not os.path.isfile(path):
            problem = f"  ⚠ Cannot find {path}, skipping"
        else:
            try:
                doc = fitz.open(path)
            except Exception as e:
                problem = f"  ⚠ Could not open {name}: {e!r}"
        if problem:
            print(problem)
            for items in pages.values():
                if on_missing:
                    on_missing(name, items)
            continue
        try:
            for si in sorted(pages, key=lambda s: page_key(name, s) if page_key else s):
                if 1 <= si <= len(doc):
                    yield name, si, pages[si], doc
                elif on_missing:
                    on_missing(name, pages[si])
        finally:
            doc.close()


def generate_images(ds, folder, out_dir, filter_kwargs=None, per_source=None,
                   only_lost=False, select=None, max_pages=None,
                   crop_margin=60.0):
    """Draws the removed boxes on the original PDFs, grouped by severity."""
    if per_source:
        label = _label_per_source(per_source)
        removed = make_filter_per_source(per_source)
        reasons_for = lambda p: (rejection_reasons(p, **per_source[p["kilde"].lower()])
                                 if p["kilde"].lower() in per_source else [])
    else:
        kw = filter_kwargs or {}
        label = _label(kw)
        removed = make_filter(**kw)
        reasons_for = lambda p: rejection_reasons(p, **kw)

    m = measure_filter(ds, removed, collect_lost=True)
    lost_ids = set(m.lost_boxes or ())

    print(f"\nFilter: {label}")
    print(f"  Ground-truth boxes lost (all coverage gone): {m.lost}"
          f"   ({m.lost_pct:.2f}% of {ds.covered_before} covered)")
    print(f"  Oversladdinger removed:                      {m.ov_rm}"
          f"   ({m.ov_pct:.1f}% of {ds.n_miss})")
    print(f"  Covering boxes removed without loss:         {m.red_rm}   (free)")
    print(f"  Removed in total:                            {m.n_rm}"
          f"   ({m.area_rm:,.0f} pt²)".replace(",", " "))
    print(f"  Recall after:  {m.recall_after:.2f}%"
          f"   Precision after: {m.prec_after:.1f}%")

    if not m.n_rm:
        print("  No boxes removed, nothing to draw.")
        return

    for p in ds.pred:
        p["_reasons"] = reasons_for(p) if removed(p) else []
        if not p["_reasons"]:
            p["_kat"] = None
        elif not p["covers"]:
            p["_kat"] = "oversladd"
        elif any(j in lost_ids for j in p["covers"]):
            p["_kat"] = "critical"
        else:
            p["_kat"] = "redundant"

    # ── Manifest of every lost ground-truth box, for manual review ──
    removed_for = defaultdict(list)
    for p in ds.pred:
        if p["_kat"] in ("critical",):
            for j in p["covers"]:
                if j in lost_ids:
                    removed_for[j].append(p)

    name_per_doc = {}
    for p in ds.pred:
        name_per_doc.setdefault(p["doc_no"], p["navn"])

    manifest = []
    for j in sorted(lost_ids):
        fb = ds.truth_boxes[j]
        fx0, fy0, fx1, fy1 = fb["box"]
        for p in removed_for.get(j, [{}]):
            fa = fb["norm_areal"]
            coverage = (intersection_area(p["norm"], fb["norm"]) / fa * 100
                       if p and fa > 0 else "")
            manifest.append({
                "coverage_pct": round(coverage, 1) if coverage != "" else "",
                "fil": name_per_doc.get(fb["doc_no"], f"{fb['doc_no']}"),
                "side": fb["side"],
                "label_id": fb.get("label_id", ""),
                "fasit_x0": round(fx0, 1), "fasit_y0": round(fy0, 1),
                "fasit_bredde_pt": round(fx1 - fx0, 1),
                "fasit_hoyde_pt": round(fy1 - fy0, 1),
                "dekkere_foer": ds.coverage_before[j],
                "kilde": p.get("kilde", ""),
                "conf": p.get("conf") if p.get("conf") is not None else "",
                "pred_bredde_pt": round(p["w"], 1) if p else "",
                "pred_hoyde_pt": round(p["h"], 1) if p else "",
                "elongation": round(p["elongation"], 2) if p else "",
                "kortside_pt": round(p["short_side"], 1) if p else "",
                "langside_pt": round(p["long_side"], 1) if p else "",
                "reason": "; ".join(p.get("_reasons", ())),
                "_j": j,
                "vurdering": "",
            })
    # Group identical causes together, so the review goes faster
    manifest.sort(key=lambda r: (r["reason"], r["fil"], r["side"]))
    crop_name = {}
    for nr, row in enumerate(manifest, 1):
        row["nr"] = nr
        base = os.path.splitext(os.path.basename(row["fil"]))[0]
        row["utsnitt"] = f"{nr:04d}_{base}_side{row['side']}.png"
        crop_name.setdefault(row["_j"], []).append(row["utsnitt"])

    if manifest:
        field = ["nr", "fil", "side", "label_id", "reason", "coverage_pct",
                "kilde", "conf",
                "elongation", "kortside_pt", "langside_pt",
                "pred_bredde_pt", "pred_hoyde_pt", "fasit_bredde_pt",
                "fasit_hoyde_pt", "dekkere_foer", "fasit_x0", "fasit_y0",
                "utsnitt", "vurdering"]
        manifest_path = _write_manifest(out_dir, "lost.csv", manifest, field)
        print(f"  Manifest of lost boxes: {manifest_path}")
        print(f"    {len(manifest)} rows for {len(lost_ids)} lost boxes"
              + ("  (a box can have several removed covers)"
                 if len(manifest) != len(lost_ids) else ""))
        per_reason = defaultdict(int)
        for row in manifest:
            per_reason[row["reason"]] += 1
        print("  Loss grouped by the rule that triggered the removal:")
        for reason, n in sorted(per_reason.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>5}  {reason}")

    lost_per_page = defaultdict(int)
    for j in lost_ids:
        fb = ds.truth_boxes[j]
        lost_per_page[(fb["doc_no"], fb["side"])] += 1

    pages = defaultdict(lambda: {"critical": 0, "redundant": 0, "oversladd": 0})
    for p in ds.pred:
        if p["_kat"]:
            pages[(p["navn"], p["side"])][p["_kat"]] += 1

    # A lost ground-truth box always has a critical removal on its own page.
    relevant = []
    for (name, si), tally in pages.items():
        n_lost = lost_per_page.get((doc_no(name), si), 0)
        if only_lost and not n_lost:
            continue
        relevant.append((n_lost, tally["critical"], tally["oversladd"], name, si))

    if select:
        select_apply = {os.path.basename(v) for v in select}
        relevant = [a for a in relevant if os.path.basename(a[3]) in select_apply]

    if not relevant:
        print("  No pages to draw.")
        return

    relevant.sort(key=lambda a: (-a[0], -a[1], -a[2], a[3], a[4]))  # worst first
    if max_pages:
        relevant = relevant[:max_pages]

    mapper = {k: os.path.join(out_dir, k) for k in
              ("lost", "redundant_fjernet", "oversladd_fjernet")}
    for path in mapper.values():
        os.makedirs(path, exist_ok=True)

    per_page = defaultdict(list)
    for p in ds.pred:
        per_page[(p["navn"], p["side"])].append(p)
    truth_per_page = defaultdict(list)
    for j, fb in enumerate(ds.truth_boxes):
        truth_per_page[(fb["doc_no"], fb["side"])].append(j)

    # Worst page first, so the lost ones are ready early; iter_pdf_pages still
    # opens each PDF once.
    rang = {(name, si): i for i, (_, _, _, name, si) in enumerate(relevant)}
    work = defaultdict(dict)
    for (_, _, _, name, si) in relevant:
        work[name][si] = per_page[(name, si)]

    tally = defaultdict(int)
    n_drawn = 0
    for name, si, page_pred, doc in iter_pdf_pages(
            folder, work,
            file_key=lambda n: min(rang[(n, s)] for s in work[n]),
            page_key=lambda n, s: rang[(n, s)]):
        if not page_pred:
            continue
        image = _render_page(doc, si)
        base, overlay, drawer = _overlay(image)
        sx = image.width / page_pred[0]["bw"]
        sy = image.height / page_pred[0]["bh"]

        for j in truth_per_page.get((doc_no(name), si), ()):
            fx0, fy0, fx1, fy1 = ds.truth_boxes[j]["box"]
            r = [fx0 * SCALE * sx, fy0 * SCALE * sy,
                 fx1 * SCALE * sx, fy1 * SCALE * sy]
            if j in lost_ids:
                # Inflate a little, else the prediction hides it
                outer = [r[0] - 6, r[1] - 6, r[2] + 6, r[3] + 6]
                drawer.rectangle(outer, outline=LOST_TRUTH, width=5)
                drawer.text((outer[0] + 2, max(outer[1] - 44, 2)),
                            "LOST TRUTH", fill=LOST_TRUTH,
                            font=_font_small())
            else:
                drawer.rectangle(r, outline=TRUTH, width=2)

        for p in page_pred:
            if p["_kat"] is None:
                px = p["px"]
                drawer.rectangle([px[0] * sx, px[1] * sy,
                                  px[2] * sx, px[3] * sy],
                                 outline=KEPT, width=1)

        colors = {"critical": CRITICAL_REMOVED,
                  "redundant": REDUNDANT_REMOVED,
                  "oversladd": OVERSLADD_REMOVED}
        has = set()
        for p in page_pred:
            if p["_kat"] is None:
                continue
            px = p["px"]
            r = [px[0] * sx, px[1] * sy, px[2] * sx, px[3] * sy]
            color = colors[p["_kat"]]
            has.add(p["_kat"])
            drawer.rectangle(r, outline=color, width=4)
            mark = p["_kat"].upper()
            if p["klasse"] == "SLURV":
                mark += "/SLURV"
            _draw_text(drawer, r, f"{mark} [{p['kilde']}]", color, over=True)
            _draw_text(drawer, r, "; ".join(p["_reasons"]), color, over=False)
            if p["conf"] is not None:
                drawer.text((r[0] + 2, max(r[1] + 2, 2)),
                            f"conf={p['conf']:.2f}", fill=color,
                            font=_font_small())

        image = _flatten(base, overlay)

        if crop_margin:
            margin = crop_margin * SCALE
            for j in truth_per_page.get((doc_no(name), si), ()):
                if j not in lost_ids:
                    continue
                fx0, fy0, fx1, fy1 = ds.truth_boxes[j]["box"]
                box = _crop_box(image, [fx0 * SCALE * sx, fy0 * SCALE * sy,
                                        fx1 * SCALE * sx, fy1 * SCALE * sy],
                                margin)
                if box is None:
                    continue
                ut = os.path.join(mapper["lost"], "utsnitt")
                os.makedirs(ut, exist_ok=True)
                for filename_u in crop_name.get(j, ()):
                    image.crop(box).save(os.path.join(ut, filename_u))
                    tally["utsnitt"] += 1

        filename = f"{os.path.splitext(name)[0]}_side{si}.png"
        if "critical" in has:
            image.save(os.path.join(mapper["lost"], filename))
            tally["lost"] += 1
        if "redundant" in has:
            image.save(os.path.join(mapper["redundant_fjernet"], filename))
            tally["redundant_fjernet"] += 1
        if "oversladd" in has:
            image.save(os.path.join(mapper["oversladd_fjernet"], filename))
            tally["oversladd_fjernet"] += 1
        n_drawn += 1

    for name, path in mapper.items():
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)

    print(f"  Drew {n_drawn} pages to {out_dir}")
    for name in ("lost", "redundant_fjernet", "oversladd_fjernet"):
        if tally[name]:
            print(f"    {tally[name]:>5} page(s) in {name}/")
    if tally["utsnitt"]:
        print(f"    {tally['utsnitt']:>5} crops in lost/utsnitt/ "
              f"— one per lost box, same order as lost.csv")


# ── OCR text layer for triage ─────────────────────────────────
# --ocr-text fades the original and draws the pipeline's CACHED tokens on top,
# colored by rec score. Tokens live in the ROTATED image's pixel space (the
# pipeline OCRs the rotated page), so predictions and ground truth are
# transformed forward with the inverse of orientation.unrotate_box.

OCR_REC_HIGH = (0, 115, 0)        # green
OCR_REC_MID = (25, 45, 170)      # blue
OCR_REC_LOW = (200, 30, 30)      # red
OCR_REC_UNKNOWN = (130, 130, 130)    # grey

_FONTER = {}
_OCR_READ_CACHE = None


def _font_str(px):
    px = max(9, min(int(px), 44))
    if px not in _FONTER:
        _FONTER[px] = ImageFont.load_default(size=px)
    return _FONTER[px]


def _rec_color(rec):
    if rec is None:
        return OCR_REC_UNKNOWN
    if rec >= 0.98:
        return OCR_REC_HIGH
    if rec >= 0.90:
        return OCR_REC_MID
    return OCR_REC_LOW


def rotate_box(r, k, w0, h0):
    """Unrotated pixel rect -> rotated space (inverse of orientation.unrotate_box)."""
    if not k:
        return list(r)
    x0, y0, x1, y1 = r
    if k == 1:
        pts = [(y0, w0 - x0), (y1, w0 - x1)]
    elif k == 2:
        pts = [(w0 - x0, h0 - y0), (w0 - x1, h0 - y1)]
    else:
        pts = [(h0 - y0, x0), (h0 - y1, x1)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _read_ocr_cache(ocr_dir, name):
    """Read (rotations, tokens_per_page) from the pipeline's OCR cache."""
    global _OCR_READ_CACHE
    if _OCR_READ_CACHE is None:
        app = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "app"))
        if app not in sys.path:
            sys.path.insert(0, app)
        from ocr_cache import read_cache
        _OCR_READ_CACHE = read_cache
    return _OCR_READ_CACHE(ocr_dir, name)


def _draw_tokens(drawer, tokens, srx, sry):
    """Draws each token inside its own box, scaled to it, colored by rec."""
    for t in tokens:
        if not t.text.strip():
            continue
        r = [t.x0 * srx, t.y0 * sry, t.x1 * srx, t.y1 * sry]
        color = _rec_color(t.rec_score)
        drawer.rectangle(r, outline=color + (70,), width=1)
        str_h = (r[3] - r[1]) * 0.8
        str_b = (r[2] - r[0]) / (max(len(t.text), 1) * 0.55)
        drawer.text((r[0] + 1, r[1]), t.text, fill=color + (255,),
                    font=_font_str(min(str_h, str_b)))


def triage_bom(ds, folder, out_dir, select=None, max_pages=None, source=None,
               ocr_dir=None, ocr_opacity=0.15):
    """Draws ALL BOM predictions, regardless of filter.

    A BOM box hits no ground truth, but the ground truth is human-made: it may
    equally be an fnr the case handler missed, and geometry cannot tell. Pages
    with 'begge' boxes come first. Where both models agree, the ground truth
    is most likely the one at fault.
    """
    miss = [p for p in ds.pred if p["klasse"] == "BOM"]
    if source:
        miss = [p for p in miss if p["kilde"].lower() == source.lower()]
    print(f"\nTRIAGE of BOM boxes (hitting no ground-truth box)"
          + (f", only kilde «{source}»" if source else ""))
    print(f"  Total: {len(miss)}")
    per_source = defaultdict(int)
    for p in miss:
        per_source[p["kilde"]] += 1
    for k in sorted(per_source):
        print(f"    {k:>8}: {per_source[k]:>5}")
    if not miss:
        return

    priority = {"begge": 0, "yolo": 1, "paddle": 2}
    pages = defaultdict(list)
    for p in miss:
        pages[(p["navn"], p["side"])].append(p)

    if select:
        select_apply = {os.path.basename(v) for v in select}
        pages = {k: v for k, v in pages.items()
                 if os.path.basename(k[0]) in select_apply}

    ranked = sorted(
        pages.items(),
        key=lambda kv: (min(priority.get(p["kilde"], 9) for p in kv[1]),
                        -len(kv[1]), kv[0]))
    if max_pages:
        ranked = ranked[:max_pages]

    per_page = defaultdict(list)
    for p in ds.pred:
        per_page[(p["navn"], p["side"])].append(p)
    truth_per_page = defaultdict(list)
    for fb in ds.truth_boxes:
        truth_per_page[(fb["doc_no"], fb["side"])].append(fb)

    # Ranked order (begge pages first); iter_pdf_pages still opens each PDF once.
    rang = {sort_key: i for i, (sort_key, _) in enumerate(ranked)}
    work = defaultdict(dict)
    for (name, si), _ in ranked:
        work[name][si] = per_page[(name, si)]

    tally = defaultdict(int)
    n_drawn = 0
    cache = cache_for = None
    for name, si, page_pred, doc in iter_pdf_pages(
            folder, work,
            file_key=lambda n: min(rang[(n, s)] for s in work[n]),
            page_key=lambda n, s: rang[(n, s)]):
        if name != cache_for:
            cache_for = name
            cache = _read_ocr_cache(ocr_dir, name) if ocr_dir else None
            if ocr_dir and cache is None:
                print(f"  ⚠ No OCR cache for {name}, drawing without text layer")
        image = _render_page(doc, si)
        w0, h0 = image.width, image.height
        k, tokens = 0, []
        if cache:
            rotations, tokens_per_page = cache
            if si <= len(rotations):
                k = rotations[si - 1] or 0
                tokens = tokens_per_page[si - 1]
            if k:
                # Same rotation the pipeline OCR'd with (np.rot90 = CCW)
                image = image.rotate(90 * k, expand=True)
            image = Image.blend(
                Image.new("RGB", image.size, (255, 255, 255)),
                image, ocr_opacity)
        base, overlay, drawer = _overlay(image)
        sx = w0 / page_pred[0]["bw"]
        sy = h0 / page_pred[0]["bh"]
        if cache and tokens:
            bw, bh = page_pred[0]["bw"], page_pred[0]["bh"]
            rot_w, rot_h = (bh, bw) if k % 2 else (bw, bh)
            _draw_tokens(drawer, tokens,
                         base.width / rot_w, base.height / rot_h)

        for fb in truth_per_page.get((doc_no(name), si), ()):
            fx0, fy0, fx1, fy1 = fb["box"]
            r = rotate_box([fx0 * SCALE * sx, fy0 * SCALE * sy,
                            fx1 * SCALE * sx, fy1 * SCALE * sy], k, w0, h0)
            drawer.rectangle(r, outline=TRUTH, width=2)

        sources_here = set()
        for p in page_pred:
            px = p["px"]
            r = rotate_box([px[0] * sx, px[1] * sy, px[2] * sx, px[3] * sy],
                           k, w0, h0)
            if p["klasse"] == "BOM":
                sources_here.add(p["kilde"])
                drawer.rectangle(r, outline=MISS, width=4)
                mark = f"BOM [{p['kilde']}]"
                if p["conf"] is not None:
                    mark += f" conf={p['conf']:.2f}"
                _draw_text(drawer, r, mark, MISS, over=True)
                _draw_text(drawer, r,
                            f"{p['w']:.0f}x{p['h']:.0f}pt e={p['elongation']:.1f}",
                            MISS, over=False)
            else:
                drawer.rectangle(r, outline=KEPT, width=1)

        image = _flatten(base, overlay)
        filename = f"{os.path.splitext(name)[0]}_side{si}.png"
        for source in sources_here:
            subdir = os.path.join(out_dir, "bom", source)
            os.makedirs(subdir, exist_ok=True)
            image.save(os.path.join(subdir, filename))
            tally[source] += 1
        n_drawn += 1

    print(f"  Drew {n_drawn} pages to {os.path.join(out_dir, 'bom')}")
    for k in sorted(tally):
        print(f"    {tally[k]:>5} page(s) in bom/{k}/")
    print("\n  Blue = ground truth (the case handler's sladding), "
          "orange = BOM, grey = hit.")
    print("  The question per orange box: is there an fnr there?")
    print("    yes → the case handler missed it; the model is right")
    print("    no  → real oversladding")
    if ocr_dir:
        print("  Text layer: Paddle's cached tokens over a faded original —")
        print("    green = rec ≥ 0.98, blue = 0.90–0.98, red = < 0.90, "
              "grey = no score.")


BAND_TRUTH = (30, 120, 255, 255)     # blue  = ground truth
BAND_PRED  = (230, 30, 30, 255)      # red   = the prediction in the band
BAND_OTHER = (150, 150, 150, 140)    # grey  = other boxes on the page


def band_review(ds, folder, out_dir, criterion, lo, hi, max_items=None,
                crop_margin=25.0, out_csv=None):
    """Draws CROPS of every (prediction, ground truth) pair scoring in [lo, hi).

    Only the band changes side when the threshold moves from lo to hi. The crops
    show the document content, not just the frames, because the ground truth is
    human-made and may itself be wrong.
    """
    field = CRITERION_FIELDS[criterion]
    low_er_bra = criterion in CRITERION_LOW_IS_GOOD

    truth_per_page = defaultdict(list)
    for j, fb in enumerate(ds.truth_boxes):
        truth_per_page[(fb["doc_no"], fb["side"])].append((j, fb))
    pred_per_page = defaultdict(list)
    for p in ds.pred:
        pred_per_page[(p["doc_no"], p["side"])].append(p)

    # All overlapping pairs, so the distribution can be reported too
    pair = []
    for spec_key, truth_here in truth_per_page.items():
        for p in pred_per_page.get(spec_key, ()):
            for j, fb in truth_here:
                m = match_metrics(p["norm"], fb["norm"], fb["horizontal"])
                if m is None:
                    continue
                pair.append((m[field], m, p, j, fb))

    i_band = [t for t in pair if lo <= t[0] < hi]
    i_band.sort(key=lambda t: t[0], reverse=low_er_bra)

    print(f"\nBAND REVIEW, criterion «{criterion}» ({field}) in [{lo:.0%}, {hi:.0%})")
    print(f"  Overlapping pairs in total:   {len(pair)}")
    print(f"  Pairs in the band:            {len(i_band)}")
    print(f"  Ground-truth boxes affected:  {len({t[3] for t in i_band})}")
    print(f"  Predictions affected:         {len({id(t[2]) for t in i_band})}")
    if not i_band:
        print("  No pairs in the band. Any threshold between these two "
              "values changes nothing.")
        return

    # Only boxes lacking another prediction that clears hi actually change side.
    manages_hi = defaultdict(bool)
    for value, _m, _p, j, _fb in pair:
        if (value < hi) if low_er_bra else (value >= hi):
            manages_hi[j] = True
    vipper = {t[3] for t in i_band if not manages_hi[t[3]]}
    print(f"  Ground-truth boxes that actually flip: {len(vipper)}  "
          f"(the rest are covered by another prediction regardless)")

    per_source = defaultdict(int)
    for t in i_band:
        per_source[t[2]["kilde"]] += 1
    print("  Per kilde: " + ", ".join(f"{k}={v}" for k, v in sorted(per_source.items())))

    selected = i_band[:max_items] if max_items else i_band
    if max_items and len(i_band) > max_items:
        print(f"  ⚠ Drawing only the first {max_items} of {len(i_band)} "
              f"(--max-pages controls this)")

    dir_name = os.path.join(
        out_dir, f"band_{criterion}_{lo:.2f}-{hi:.2f}".replace(".", ""))
    os.makedirs(dir_name, exist_ok=True)

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # The column is named "value", not after the field: for the area
            # criterion field == "cov_area", which would appear twice.
            w.writerow(["utsnitt", "fil", "side", "fasit_idx", "vipper", "kilde",
                        "conf", "value", "cov_area", "cov_short", "cov_long", "iou",
                        "center_short", "pred_wpt", "pred_hpt"])
            for n, (value, m, p, j, fb) in enumerate(selected, 1):
                w.writerow([f"{n:04d}", p["navn"], p["side"], j,
                            "ja" if j in vipper else "nei", p["kilde"],
                            f"{p['conf']:.3f}" if p["conf"] is not None else "",
                            f"{value:.4f}",
                            *[f"{m[k]:.4f}" for k in ("cov_area", "cov_short",
                                                      "cov_long", "iou",
                                                      "center_short")],
                            f"{p['w']:.1f}", f"{p['h']:.1f}"])
        print(f"  Metrics written to {out_csv}")

    # ── Draw the crops; every pair on a page shares one render ──
    work = defaultdict(lambda: defaultdict(list))
    for n, t in enumerate(selected, 1):
        work[t[2]["navn"]][t[2]["side"]].append((n, t))

    n_drawn = 0
    for name, si, items, doc in iter_pdf_pages(folder, work):
        image = _render_page(doc, si)
        for n, (value, m, p, j, fb) in sorted(items):
            sx = image.width / p["bw"]
            sy = image.height / p["bh"]

            base, overlay, drawer = _overlay(image)

            for other in pred_per_page.get((p["doc_no"], si), ()):
                if other is p:
                    continue
                a = other["px"]
                drawer.rectangle([a[0] * sx, a[1] * sy, a[2] * sx, a[3] * sy],
                                 outline=BAND_OTHER, width=2)
            for jj, other_fb in truth_per_page.get((p["doc_no"], si), ()):
                if jj == j:
                    continue
                b = other_fb["box"]
                drawer.rectangle([b[0] * SCALE * sx, b[1] * SCALE * sy,
                                  b[2] * SCALE * sx, b[3] * SCALE * sy],
                                 outline=BAND_OTHER, width=2)

            fx = [fb["box"][0] * SCALE * sx, fb["box"][1] * SCALE * sy,
                  fb["box"][2] * SCALE * sx, fb["box"][3] * SCALE * sy]
            px = [p["px"][0] * sx, p["px"][1] * sy,
                  p["px"][2] * sx, p["px"][3] * sy]
            drawer.rectangle(fx, outline=BAND_TRUTH, width=3)
            drawer.rectangle(px, outline=BAND_PRED, width=3)

            flat = _flatten(base, overlay)

            margin = crop_margin * SCALE
            u = _crop_box(flat, [min(fx[0], px[0]), min(fx[1], px[1]),
                                 max(fx[2], px[2]), max(fx[3], px[3])],
                          (margin * sx, margin * sy))
            if u is None:
                continue
            crop = flat.crop(u)

            # Metrics are burned in below the crop, so it stands alone
            text = (f"{criterion} {value:.3f}  |  dek_f={m['cov_area']:.2f} "
                     f"kort={m['cov_short']:.2f} lang={m['cov_long']:.2f} "
                     f"iou={m['iou']:.2f} senter={m['center_short']:.2f}")
            text2 = (f"blue=truth red=pred   {name} s{si}  {p['kilde']}"
                      + (f" conf={p['conf']:.2f}" if p["conf"] is not None else "")
                      + f"  pred {p['w']:.0f}x{p['h']:.0f}pt"
                      + f"  fasit {fb['box'][2]-fb['box'][0]:.0f}"
                      + f"x{fb['box'][3]-fb['box'][1]:.0f}pt"
                      + ("   VIPPER" if j in vipper else ""))
            # Wide enough that the caption is not clipped
            bottom = Image.new("RGB", (max(crop.width, 780), crop.height + 48),
                             (255, 255, 255))
            bottom.paste(crop, (0, 0))
            d = ImageDraw.Draw(bottom)
            d.text((4, crop.height + 4), text, fill=(0, 0, 0), font=_font_small())
            d.text((4, crop.height + 26), text2, fill=(90, 90, 90),
                   font=_font_small())

            tilt = "vipper_" if j in vipper else ""
            filename = (f"{n:04d}_{value:.3f}_{tilt}"
                       f"{os.path.splitext(name)[0]}_s{si}_f{j}.png")
            bottom.save(os.path.join(dir_name, filename))
            n_drawn += 1

    print(f"\n  Drew {n_drawn} crops to {dir_name}/")
    print(f"  File names start with a sequence number and the {field} value, "
          f"so they sort ascending.")
    print("  Blue = ground truth, red = the prediction in the band, "
          "grey = other boxes.")
    print("  Question per crop:")
    print("    1. Is the ground-truth box correct? (a misdrawn one is not our fault)")
    print("    2. Does the prediction point at the SAME field, or the next line?")
    print("    3. Does it cover enough of the digits to be a real sladding?")
    print("  Three yeses means the pair should be accepted. The threshold is too high.")


def test_against_fasit(truth_csv, folder, out_dir, filter_kwargs, max_pages=None,
                   crop_margin=60.0, select=None, ds=None):
    """Applies the filter DIRECTLY to the case handlers' sladdinger.

    Every ground-truth box is correct by definition, so a rejection means the
    filter refuses that shape, independent of what the model predicted, which
    is why all labels can be judged, not just the processed documents.
    """
    rows, excluded, columns = read_truth_rows(truth_csv)
    n_example = sum(excluded.values())

    print(f"GROUND-TRUTH TEST: filter applied directly to the sladdinger")
    print(f"  Columns in the labels CSV: {', '.join(columns)}")
    print(f"  Labels read:      {len(rows)}  (correct sladdinger)")
    if excluded:
        print(f"  Excluded:         {n_example}  "
              + ", ".join(f"{k}: {v}" for k, v in sorted(excluded.items())))

    for field in ("ml_status", "type"):
        distribution = defaultdict(int)
        for r in rows:
            distribution[r[field]] += 1
        if len(distribution) > 1 or field == "ml_status":
            print(f"  {field}: "
                  + ", ".join(f"{k}={v}" for k, v in
                              sorted(distribution.items(), key=lambda kv: -kv[1])))

    # Shape of ALL correct sladdinger: the margin a threshold has against real
    # data, not just against the model's boxes.
    def _pct(rows_sorted, p):
        if not rows_sorted:
            return 0.0
        i = (len(rows_sorted) - 1) * p / 100.0
        low, high = int(i), min(int(i) + 1, len(rows_sorted) - 1)
        return rows_sorted[low] + (rows_sorted[high] - rows_sorted[low]) * (i - low)

    PCT = (0.01, 0.1, 1, 50, 99, 99.9, 99.99)
    print("")
    print("  Shape of all correct sladdinger "
          "(your limits live here, measure against the tails):")
    header = f"    {'metric':<14}" + "".join(f" {('p' + format(p, 'g')):>9}" for p in PCT)
    print(header)
    print(f"    {'-' * (len(header) - 4)}")
    for sort_key, name, decimals in (("elongation", "elongation", 2),
                              ("short_side", "short side (pt)", 1),
                              ("long_side", "long side (pt)", 1),
                              ("areal_px", "area (px²)", 0)):
        rows_sorted = sorted(r[sort_key] for r in rows)
        print(f"    {name:<14}"
              + "".join(f" {_pct(rows_sorted, p):>9.{decimals}f}" for p in PCT))

    label = _label(filter_kwargs)
    print(f"\n  Filter: {label}")
    print(f"  Note: ground-truth boxes have no conf, so the conf gate never "
          f"fires. This shows what geometry alone rejects.")

    discarded = []
    for r in rows:
        reasons = rejection_reasons(r, **filter_kwargs)
        if reasons:
            r["_reasons"] = reasons
            discarded.append(r)

    share = len(discarded) / len(rows) * 100 if rows else 0
    print(f"\n  Sladdinger the filter would reject: {len(discarded)} "
          f"({share:.3f}% of {len(rows)})")
    if not discarded:
        print("  None. The filter rejects no sladding made by a case handler.")
        return

    for field, title in (("_reason", "rule (a box can break several)"),
                         ("ml_status", "ml_status"), ("type", "type")):
        distribution, denominator = defaultdict(int), defaultdict(int)
        for r in discarded:
            if field == "_reason":
                for g in r["_reasons"]:
                    distribution[re.sub(r"[\d.]+", "N", g, count=1)] += 1
            else:
                distribution[r[field]] += 1
        if field != "_reason":
            for r in rows:
                denominator[r[field]] += 1
        if len(distribution) > 1 or field == "_reason":
            print(f"\n  Grouped by {title}:")
            for k, v in sorted(distribution.items(), key=lambda kv: -kv[1]):
                tot = denominator.get(k)
                share = f"  of {tot} = {v / tot * 100:.3f}%" if tot else ""
                print(f"    {v:>6}  {k}{share}")

    # An illegal ground-truth shape costs nothing if the model's OWN box for
    # the same field has a legal shape and survives the filter.
    if ds is not None:
        m = measure_filter(ds, make_filter(**filter_kwargs), collect_lost=True)
        lost_ids = set(m.lost_boxes or ())
        i_scope = {}
        for j, fb in enumerate(ds.truth_boxes):
            i_scope[(fb["doc_no"], fb["side"],
                     round(fb["box"][0], 1), round(fb["box"][1], 1))] = j
        outside = shape_and_lost = form_men_covered = 0
        for r in discarded:
            j = i_scope.get((r["doc_no"], r["side"],
                             round(r["box"][0], 1), round(r["box"][1], 1)))
            if j is None:
                r["_status"] = "utenfor_scope"
                outside += 1
            elif j in lost_ids:
                r["_status"] = "MISTET_DEKNING"
                shape_and_lost += 1
            else:
                r["_status"] = "fortsatt_dekket"
                form_men_covered += 1
        i_scope_discarded = shape_and_lost + form_men_covered
        print("")
        print("  Cross-check against the model's own boxes "
              f"({len(ds.truth_boxes)} labels on processed documents):")
        print(f"    Illegal shape AND lost coverage:  {shape_and_lost:>5}"
              "   ← real risk")
        print(f"    Illegal shape, still covered:     {form_men_covered:>5}"
              "   ← artifact: the model's box has a legal shape")
        if outside:
            print(f"    Out of scope (document not run):  {outside:>5}"
                  "   ← cannot be judged")
        if i_scope_discarded:
            print(f"    Artifact share of the judgeable: "
                  f"{form_men_covered / i_scope_discarded * 100:.0f}%")
        print(f"    Total coverage lost under the same filter: {m.lost}")

    # ── Manifest ──
    rang = {"MISTET_DEKNING": 0, "": 1, "utenfor_scope": 2, "fortsatt_dekket": 3}
    discarded.sort(key=lambda r: (rang.get(r.get("_status", ""), 1),
                                  "; ".join(r["_reasons"]),
                                  r["doc_no"], r["side"]))
    manifest = []
    for nr, r in enumerate(discarded, 1):
        x0, y0, x1, y1 = r["box"]
        manifest.append({
            "nr": nr, "fil_revisjon_id": r["doc_no"], "side": r["side"],
            "reason": "; ".join(r["_reasons"]),
            "ml_status": r["ml_status"], "type": r["type"],
            "elongation": round(r["elongation"], 2),
            "kortside_pt": round(r["short_side"], 1),
            "langside_pt": round(r["long_side"], 1),
            "bredde_pt": round(r["w"], 1), "hoyde_pt": round(r["h"], 1),
            "areal_px": round(r["areal_px"]),
            "status": r.get("_status", ""),
            "x0": round(x0, 1), "y0": round(y0, 1),
            "utsnitt": f"{nr:04d}_{r['doc_no']}_side{r['side']}.png",
            "vurdering": "",
        })
        r["_utsnitt"] = manifest[-1]["utsnitt"]
    manifest_path = _write_manifest(out_dir, "forkastede_sladdinger.csv",
                                    manifest, list(manifest[0]))
    print(f"\n  Manifest: {manifest_path}")

    # ── Crops ──
    if not folder:
        return
    if select:
        select_apply = {os.path.basename(v) for v in select}
    per_doc = defaultdict(list)
    for r in (discarded[:max_pages] if max_pages else discarded):
        per_doc[r["doc_no"]].append(r)

    files = {}
    for name in os.listdir(folder):
        if name.lower().endswith(".pdf"):
            n = doc_no(name)
            if n is not None:
                files.setdefault(n, name)

    ut = os.path.join(out_dir, "utsnitt")
    os.makedirs(ut, exist_ok=True)
    n_drawn = missing = 0

    work = defaultdict(lambda: defaultdict(list))
    for nr in sorted(per_doc):
        name = files.get(nr)
        if name is None or (select and os.path.basename(name) not in select_apply):
            missing += len(per_doc[nr])
            continue
        for r in per_doc[nr]:
            work[name][r["side"]].append(r)

    def count_missing(_name, rows):
        nonlocal missing
        missing += len(rows)

    margin = crop_margin * SCALE
    for name, si, rows, doc in iter_pdf_pages(folder, work, file_key=doc_no,
                                              on_missing=count_missing):
        image = _render_page(doc, si)
        for r in rows:
            base, overlay, drawer = _overlay(image)
            x0, y0, x1, y1 = r["box"]
            rr = [x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE]
            drawer.rectangle(rr, outline=CRITICAL_REMOVED, width=4)
            _draw_text(drawer, rr, "REJECTED BY FILTER", CRITICAL_REMOVED, True)
            _draw_text(drawer, rr, "; ".join(r["_reasons"]),
                        CRITICAL_REMOVED, False)
            done = _flatten(base, overlay)
            if _save_crop(done, rr, margin, os.path.join(ut, r["_utsnitt"])):
                n_drawn += 1

    print(f"  Drew {n_drawn} crops to {ut}")
    if missing:
        print(f"  {missing} without a crop (PDF missing in {folder})")


SWEEP_CONFIGURE = [
    {"min_elongation": 1.5},
    {"min_elongation": 2.0},
    {"max_height": 40},
    {"max_height": 50},
    {"max_width": 100},
    {"max_width": 120},
    {"min_elongation": 1.5, "max_height": 50, "max_width": 100},
    {"min_elongation": 1.5, "max_height": 50, "max_width": 120},
    {"min_elongation": 1.5, "max_height": 60, "max_width": 120},
    {"min_elongation": 2.0, "max_height": 50, "max_width": 100},
]



# ── Uncovered review ──────────────────────────────────────────
# What the model never found, as opposed to what the filter removes. Ground
# truth is split on ml_generated: ML-accepted boxes are the model's own
# approved suggestions and a re-run nearly always refinds them (circular), so
# real detection ability is measured on the manually drawn boxes.

def _is_ml_generated(row):
    return (row.get("ml_generated") or "").strip().lower() in ("true", "t", "1")


def _p10_p50_p90(values):
    v = sorted(values)
    n = len(v)
    return v[int(0.10 * (n - 1))], v[int(0.50 * (n - 1))], v[int(0.90 * (n - 1))]


def review_uncovered(truth_csv, res_csv, folder, out_dir,
                        criterion=STD_CRITERION, threshold=HIT_THRESHOLD,
                        good_coverage=0.90, processed_doc=None, also_ml=False,
                        max_crop=None, crop_margin=60.0):
    """Catalogues and draws ground-truth boxes without (good enough) coverage."""
    if criterion in CRITERION_LOW_IS_GOOD:
        raise SystemExit(f"--uncovered does not support criterion {criterion!r} "
                         f"(low is good there, use areal/kortside/iou)")
    field = CRITERION_FIELDS[criterion]

    truth_rows, _discarded, columns = read_truth_rows(truth_csv)
    if "ml_generated" not in columns:
        print("⚠ The labels CSV has no ml_generated. Every box is treated as "
              "manual. Include the column in the next export.")
    pred = read_predictions(res_csv)

    processed = (set(processed_doc) if processed_doc is not None
              else {p["doc_no"] for p in pred})
    scope = {r["doc_no"] for r in truth_rows} & processed

    page_str = {}
    per_page_pred = defaultdict(list)
    name_per_doc = {}
    for p in pred:
        key = (p["doc_no"], p["side"])
        page_str.setdefault(key, (p["bw"] / SCALE, p["bh"] / SCALE))
        per_page_pred[key].append(p)
        name_per_doc.setdefault(p["doc_no"], p["navn"])

    groups = {"udekket": [], "daarlig_dekket": [], "ok": []}
    for r in truth_rows:
        if r["doc_no"] not in scope:
            continue
        x0, y0, x1, y1 = r["box"]
        pw, ph = page_str.get((r["doc_no"], r["side"]), (595.0, 842.0))
        fn = (x0 / pw, y0 / ph, x1 / pw, y1 / ph)
        horizontal = (x1 - x0) >= (y1 - y0)
        best_v, best_p = 0.0, None
        for p in per_page_pred.get((r["doc_no"], r["side"]), ()):
            m = match_metrics(p["norm"], fn, horizontal)
            if m is not None and m[field] > best_v:
                best_v, best_p = m[field], p
        r["_coverage"] = best_v
        r["_beste_p"] = best_p
        r["_ml"] = _is_ml_generated(r["row"])
        group = ("udekket" if best_v < threshold
                  else "daarlig_dekket" if best_v < good_coverage else "ok")
        groups[group].append(r)

    n_scope = sum(len(g) for g in groups.values())
    print(f"\nUncovered review  (criterion «{criterion}», threshold "
          f"{threshold:g}, good coverage ≥ {good_coverage:g})")
    print(f"  Scope: {len(scope)} documents, {n_scope} ground-truth boxes")
    for ml, name in ((False, "manually drawn"), (True, "ML-accepted   ")):
        tally = {k: sum(1 for r in g if r["_ml"] == ml)
                for k, g in groups.items()}
        n = sum(tally.values())
        if n:
            print(f"  {name} ({n:>5} boxes): "
                  f"uncovered {tally['udekket']:>4} ({100*tally['udekket']/n:4.1f}%)  "
                  f"poor {tally['daarlig_dekket']:>4} "
                  f"({100*tally['daarlig_dekket']/n:4.1f}%)  "
                  f"good {tally['ok']:>5} ({100*tally['ok']/n:4.1f}%)")

    sample = [r for r in groups["udekket"] + groups["daarlig_dekket"]
              if also_ml or not r["_ml"]]
    if not sample:
        print("  Nothing to draw.")
        return

    upright = sum(1 for r in sample if r["h"] > r["w"])
    print(f"\n  Selected for review: {len(sample)} boxes "
          f"({'manual + ML' if also_ml else 'manually drawn only'})")
    print(f"    upright (h > w):   {upright} ({100*upright/len(sample):.1f}%)")
    for target in ("short_side", "long_side"):
        p10, p50, p90 = _p10_p50_p90([r[target] for r in sample])
        print(f"    {target:>8} (pt):    p10 {p10:5.1f}   p50 {p50:5.1f}   "
              f"p90 {p90:5.1f}")
    per_type = defaultdict(int)
    for r in sample:
        per_type[r["type"]] += 1
    print("    per type:          "
          + "  ".join(f"{t}={n}" for t, n in
                      sorted(per_type.items(), key=lambda kv: -kv[1])))
    per_doc = defaultdict(int)
    for r in sample:
        per_doc[r["doc_no"]] += 1
    n_one = sum(1 for n in per_doc.values() if n == 1)
    print(f"    spread over {len(per_doc)} documents "
          f"({n_one} with a single box); worst offenders:")
    for dnr, n in sorted(per_doc.items(), key=lambda kv: -kv[1])[:15]:
        print(f"      {n:>4}  {name_per_doc.get(dnr, f'{dnr}.pdf')}")

    # ── Manifest ──
    sample.sort(key=lambda r: (r["_coverage"] >= threshold, name_per_doc.get(
        r["doc_no"], str(r["doc_no"])), r["side"], r["box"][1]))
    if max_crop:
        sample = sample[:max_crop]

    manifest = []
    for nr, r in enumerate(sample, 1):
        bp = r["_beste_p"]
        file = name_per_doc.get(r["doc_no"], f"{r['doc_no']}.pdf")
        base = os.path.splitext(os.path.basename(file))[0]
        manifest.append({
            "nr": nr, "fil": file, "side": r["side"],
            "group": ("udekket" if r["_coverage"] < threshold
                       else "daarlig_dekket"),
            "coverage_pct": round(100 * r["_coverage"], 1),
            "ml_generated": int(r["_ml"]),
            "ml_status": r["ml_status"], "type": r["type"],
            "upright": int(r["h"] > r["w"]),
            "fasit_bredde_pt": round(r["w"], 1),
            "fasit_hoyde_pt": round(r["h"], 1),
            "beste_kilde": bp["kilde"] if bp else "",
            "beste_conf": (bp["conf"] if bp and bp["conf"] is not None
                           else ""),
            "beste_bredde_pt": round(bp["w"], 1) if bp else "",
            "beste_hoyde_pt": round(bp["h"], 1) if bp else "",
            "fasit_x0": round(r["box"][0], 1),
            "fasit_y0": round(r["box"][1], 1),
            "utsnitt": f"{nr:04d}_{base}_side{r['side']}.png",
            "vurdering": "",
            "_r": r,
        })

    field_csv = ["nr", "fil", "side", "group", "coverage_pct", "ml_generated",
                "ml_status", "type", "upright", "fasit_bredde_pt",
                "fasit_hoyde_pt", "beste_kilde", "beste_conf",
                "beste_bredde_pt", "beste_hoyde_pt", "fasit_x0", "fasit_y0",
                "utsnitt", "vurdering"]
    manifest_path = _write_manifest(out_dir, "uncovered.csv", manifest, field_csv)
    print(f"\n  Manifest: {manifest_path}  ({len(manifest)} rows)")

    page_dir = os.path.join(out_dir, "sider")
    crop_dir = os.path.join(out_dir, "utsnitt")
    os.makedirs(page_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)

    per_file = defaultdict(lambda: defaultdict(list))
    for row in manifest:
        per_file[row["fil"]][row["side"]].append(row)

    n_pages = n_crop = 0
    for file, si, rows, doc in iter_pdf_pages(folder, per_file):
        image = _render_page(doc, si)
        page_rect = doc[si - 1].rect
        skx = image.width / page_rect.width if page_rect.width else SCALE
        sky = image.height / page_rect.height if page_rect.height else SCALE
        base, overlay, drawer = _overlay(image)
        dnr = doc_no(file)
        chosen = {id(row["_r"]) for row in rows}

        page_pred = per_page_pred.get((dnr, si), ())
        for p2 in page_pred:
            px = p2["px"]
            sx2 = image.width / p2["bw"]
            sy2 = image.height / p2["bh"]
            drawer.rectangle([px[0] * sx2, px[1] * sy2,
                              px[2] * sx2, px[3] * sy2],
                             outline=KEPT, width=1)

        for r2 in groups["udekket"] + groups["daarlig_dekket"] + groups["ok"]:
            if (r2["doc_no"], r2["side"]) != (dnr, si) or id(r2) in chosen:
                continue
            fx0, fy0, fx1, fy1 = r2["box"]
            drawer.rectangle([fx0 * skx, fy0 * sky, fx1 * skx, fy1 * sky],
                             outline=TRUTH, width=2)

        # The selected ones: ground truth magenta, best suggestion orange
        for row in rows:
            r2 = row["_r"]
            fx0, fy0, fx1, fy1 = r2["box"]
            rr = [fx0 * skx - 6, fy0 * sky - 6,
                  fx1 * skx + 6, fy1 * sky + 6]
            drawer.rectangle(rr, outline=LOST_TRUTH, width=5)
            mark = (f"{row['group'].upper()} {row['coverage_pct']:g}% "
                     f"[{'ML' if r2['_ml'] else 'manual'}]")
            _draw_text(drawer, rr, mark, LOST_TRUTH, over=True)
            bp = r2["_beste_p"]
            if bp is not None:
                px = bp["px"]
                sx2 = image.width / bp["bw"]
                sy2 = image.height / bp["bh"]
                rp = [px[0] * sx2, px[1] * sy2, px[2] * sx2, px[3] * sy2]
                drawer.rectangle(rp, outline=MISS, width=3)
                text = f"BEST SUGGESTION [{bp['kilde']}]"
                if bp["conf"] is not None:
                    text += f" conf={bp['conf']:.2f}"
                _draw_text(drawer, rp, text, MISS, over=False)

        done = _flatten(base, overlay)
        done.save(os.path.join(
            page_dir, f"{os.path.splitext(file)[0]}_side{si}.png"))
        n_pages += 1

        margin = crop_margin * skx
        for row in rows:
            fx0, fy0, fx1, fy1 = row["_r"]["box"]
            rect = [fx0 * skx, fy0 * sky, fx1 * skx, fy1 * sky]
            if _save_crop(done, rect, margin,
                          os.path.join(crop_dir, row["utsnitt"])):
                n_crop += 1

    print(f"  Drew {n_pages} pages to {page_dir}")
    print(f"  {n_crop} crops in {crop_dir}, same order as uncovered.csv, "
          f"ready for the vurdering column")


def main():
    p = argparse.ArgumentParser(
        description="Visual review of the boxes a filter configuration "
                    "removes, grouped by whether the removal actually costs "
                    "recall.")
    p.add_argument("--truth-csv", required=True, help="Labels CSV (ground truth)")
    p.add_argument("--res-csv", default=None,
                   help="Result CSV from the model (not needed with --against-truth)")
    p.add_argument("--folder", default=None, help="Folder with the PDF documents")
    p.add_argument("--out-dir", default="filter_review",
                   help="Folder for PNG output (default: filter_review)")
    p.add_argument("--criterion", default=STD_CRITERION,
                   choices=sorted(CRITERIA),
                   help=f"Match rule for coverage (default: {STD_CRITERION})")
    p.add_argument("--threshold", type=float, default=HIT_THRESHOLD,
                   help=f"Overlap threshold for coverage (default: {HIT_THRESHOLD})")
    p.add_argument("--oversize-factor", type=float, default=STD_SLOPPINESS_FACTOR,
                   help=f"SLURV limit (default: {STD_SLOPPINESS_FACTOR})")
    p.add_argument("--include-unlabelled", action="store_true", default=True,
                   help="(default on) Include processed documents with no rows "
                        "in the labels CSV. The labels file covers the whole "
                        "uttrekk, so those were reviewed with zero fnr and any "
                        "prediction there is a real oversladding")
    p.add_argument("--exclude-unlabelled", dest="include_unlabelled",
                   action="store_false",
                   help="Old behaviour: keep documents without ground-truth "
                        "rows out of scope (for older labels files that did "
                        "not cover the whole uttrekk)")

    p.add_argument("--processed-list", default=None, metavar="FILE",
                   help="File listing the documents the model ran on (one name "
                        "or number per line). Without it the documents in the "
                        "result CSV are assumed, and a document where the model "
                        "found nothing counts as not run.")
    groups = {
        "geometry": p.add_argument_group(
            "Filter parameters (give at least one, or use --per-source/--sweep)"),
        "ocr": p.add_argument_group(
            "OCR features (stricter lenient_check; hits only kilde «yolo» with "
            "text in the box. See _ocr_reason in filter_common.py)"),
    }
    # One flag per entry in FILTER_PARAMS, so a new parameter needs no edit here.
    for fp in FILTER_PARAMS:
        if fp.arg == "flag":
            groups[fp.group].add_argument(
                fp.flag, action="store_const", const=1, default=None,
                dest=fp.name, help=fp.help)
        else:
            groups[fp.group].add_argument(
                fp.flag, type=float, default=None, dest=fp.name,
                metavar=fp.unit, help=fp.help,
                **({"nargs": "?", "const": 1} if fp.arg == "optional" else {}))

    p.add_argument("--per-source", nargs="+", metavar="SPEC",
                   help='Independent filters per kilde: "kilde:e=V,h=V,b=V,a=V,c=V"')
    p.add_argument("--against-truth", action="store_true", dest="against_truth",
                   help="Apply the filter DIRECTLY to the case handlers' "
                        "sladdinger (all labels, not just documents the model "
                        "ran on). Answers how many correct sladdinger the "
                        "filter would reject. Does not need --res-csv.")
    p.add_argument("--miss", nargs="?", const="all", default=None,
                   metavar="KILDE",
                   help="Triage mode: draw ALL BOM boxes (hitting no "
                        "ground-truth box) regardless of filter, kilde 'begge' "
                        "first. With a value (begge/paddle/yolo) only that "
                        "kilde is drawn. Answers whether they are oversladding "
                        "or fnr the case handler missed.")
    p.add_argument("--ocr-text", action="store_true", dest="ocr_text",
                   help="--miss: fade the original and draw Paddle's cached "
                        "OCR tokens on top, colored by rec score (green ≥0.98, "
                        "blue ≥0.90, red <0.90). Shows what the OCR actually "
                        "read where the box was placed.")
    p.add_argument("--ocr-cache", default=None, metavar="PATH",
                   dest="ocr_cache",
                   help="OCR cache folder for --ocr-text "
                        "(default: $SLADD_CACHE/<folder>/ocr)")
    p.add_argument("--ocr-opacity", type=float, default=0.15,
                   dest="ocr_opacity", metavar="FRACTION",
                   help="Opacity of the original behind the text layer "
                        "(default 0.15)")
    p.add_argument("--uncovered", action="store_true",
                   help="Review mode: catalogue and draw ground-truth boxes the "
                        "model does not cover (well enough). Splits on "
                        "ml_generated, real detection ability is measured on "
                        "manually drawn boxes. Needs no filter flag.")
    p.add_argument("--good-coverage", type=float, default=0.90,
                   dest="good_coverage", metavar="FRACTION",
                   help="--uncovered: coverage below this counts as «poorly "
                        "covered» even when the threshold is met (default 0.90)")
    p.add_argument("--uncovered-also-ml", action="store_true",
                   dest="uncovered_also_ml",
                   help="--uncovered: draw ML-accepted boxes too (otherwise "
                        "manually drawn only. The ML boxes are circular)")
    p.add_argument("--max-crop", type=int, default=None,
                   dest="max_crop", metavar="N",
                   help="--uncovered: max number of boxes to draw "
                        "(worst coverage first)")
    p.add_argument("--band", nargs=3, default=None,
                   metavar=("CRITERION", "LO", "HI"),
                   help="Draw a crop of every (prediction, ground truth) pair "
                        "scoring in [LO, HI). The grey zone the threshold "
                        "actually decides. E.g. «--band areal 0.40 0.45». "
                        f"Criteria: {', '.join(sorted(CRITERIA))}")
    p.add_argument("--band-csv", default=None, metavar="FILE",
                   help="Write the band metrics to CSV")
    p.add_argument("--sweep", action="store_true",
                   help="Run a set of predefined configurations")
    p.add_argument("--only-lost", action="store_true",
                   dest="only_lost",
                   help="Only draw pages where a ground-truth box lost all coverage")
    p.add_argument("--crop-margin", type=float, default=60.0, metavar="PT",
                   help="Margin around the crops of lost boxes, in points. "
                        "0 turns crops off (default: 60)")
    p.add_argument("--max-pages", type=int, default=None,
                   help="Max number of pages to draw (worst first)")
    p.add_argument("--select", nargs="+", metavar="PDF",
                   help="Restrict to these PDF files")
    args = p.parse_args()

    kw_every = {n: getattr(args, n) for n in FILTER_PARAMETERS
               if getattr(args, n) is not None}

    if args.uncovered:
        if not args.res_csv or not args.folder:
            p.error("--uncovered requires --res-csv and --folder")
        processed = read_processed_docs(args.processed_list) if args.processed_list else None
        review_uncovered(args.truth_csv, args.res_csv, args.folder,
                            args.out_dir, criterion=args.criterion,
                            threshold=args.threshold,
                            good_coverage=args.good_coverage, processed_doc=processed,
                            also_ml=args.uncovered_also_ml,
                            max_crop=args.max_crop,
                            crop_margin=args.crop_margin)
        print("\nDone!")
        return

    if args.against_truth:
        if not kw_every:
            p.error("--against-truth requires at least one filter "
                    "(--elongation, --max-elongation, --min-short-side, ...)")
        ds_cross = None
        if args.res_csv:
            processed = (read_processed_docs(args.processed_list)
                      if args.processed_list else None)
            ds_cross = build_dataset(
                read_truth_boxes(args.truth_csv), read_predictions(args.res_csv),
                threshold=args.threshold, oversize_factor=args.oversize_factor,
                include_unlabelled=args.include_unlabelled, processed_doc=processed,
                criterion=args.criterion)
        test_against_fasit(args.truth_csv, args.folder, args.out_dir, kw_every,
                       max_pages=args.max_pages,
                       crop_margin=args.crop_margin, select=args.select,
                       ds=ds_cross)
        print("\nDone!")
        return

    if not args.res_csv:
        p.error("--res-csv is required (except with --against-truth)")
    if not args.folder:
        p.error("--folder is required")

    processed = read_processed_docs(args.processed_list) if args.processed_list else None
    ds = build_dataset(read_truth_boxes(args.truth_csv),
                       read_predictions(args.res_csv),
                       threshold=args.threshold, oversize_factor=args.oversize_factor,
                       include_unlabelled=args.include_unlabelled,
                       processed_doc=processed, criterion=args.criterion)
    write_summary(ds)

    common = dict(only_lost=args.only_lost, select=args.select,
                  max_pages=args.max_pages,
                  crop_margin=args.crop_margin)

    if args.band:
        criterion, lo, hi = args.band[0], float(args.band[1]), float(args.band[2])
        if criterion not in CRITERIA:
            p.error(f"unknown criterion {criterion!r}, "
                    f"valid: {', '.join(sorted(CRITERIA))}")
        if not lo < hi:
            p.error(f"LO must be smaller than HI (got {lo} and {hi})")
        band_review(ds, args.folder, args.out_dir, criterion, lo, hi,
                    max_items=args.max_pages,
                    crop_margin=(args.crop_margin
                                    if args.crop_margin != 60.0 else 25.0),
                    out_csv=args.band_csv)
    elif args.miss:
        ocr_dir = None
        if args.ocr_text:
            ocr_dir = args.ocr_cache
            if not ocr_dir:
                base = os.environ.get("SLADD_CACHE")
                if not base:
                    p.error("--ocr-text: give --ocr-cache, or set "
                            "$SLADD_CACHE (source activate.sh)")
                ocr_dir = os.path.join(
                    base, os.path.basename(os.path.normpath(args.folder)),
                    "ocr")
            if not os.path.isdir(ocr_dir):
                p.error(f"--ocr-text: cannot find the cache folder {ocr_dir}")
        triage_bom(ds, args.folder, args.out_dir, select=args.select,
                   max_pages=args.max_pages,
                   source=None if args.miss == "all" else args.miss,
                   ocr_dir=ocr_dir, ocr_opacity=args.ocr_opacity)
    elif args.per_source:
        per_source = parse_per_source(args.per_source)
        unknown = set(per_source) - {k.lower() for k in ds.sources()}
        if unknown:
            print(f"  ⚠ No predictions from kilde(r): {', '.join(sorted(unknown))}")
        ut = os.path.join(args.out_dir, _dir_name_per_source(per_source))
        generate_images(ds, args.folder, ut, per_source=per_source, **common)
    elif args.sweep:
        for kw in SWEEP_CONFIGURE:
            ut = os.path.join(args.out_dir, _dir_name(kw))
            generate_images(ds, args.folder, ut, filter_kwargs=kw, **common)
    else:
        kw = kw_every
        if not kw:
            p.error("Give at least one filter (--elongation, --max-height, "
                    "--max-width, --max-area, --conf-threshold), --per-source, "
                    "--miss or --sweep")
        generate_images(ds, args.folder, args.out_dir, filter_kwargs=kw, **common)

    print("\nDone!")


if __name__ == "__main__":
    main()
