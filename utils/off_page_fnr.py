"""
Whether the labels outside the page left an fnr uncovered.

Takes the CSV from off_page_labels.py --out and, for every page it names, finds
the fnr candidates in the cached OCR tokens and checks each against the labels
that are on the page. A candidate counts as covered when one burned label
(manual, or ML ACCEPTED) covers COVER_SHARE of it. An uncovered candidate on
the same text line as an off-page label is the case that matters: the label
was probably meant for it, and nothing on the page redacts it.

Prints no digits. A candidate is reported by position, digit shape and whether
it passes mod-11, so the output can be pasted outside the server. --png-dir
writes each page with the labels drawn on a canvas widened to fit the
off-page ones; those images show the document and stay on the server.

Edges. Only the latest label versjon per document is read, which is the one
the frontend and PdfSladder use. A document without an OCR cache entry is
reported and skipped: fill it with precache.py --only ocr. Pages the
orientation step turned are skipped too, since their tokens are not in page
coordinates.

Run:
    python utils/off_page_fnr.py --off-page off_page_u6.csv \\
        --labels $SLADD_LABELS/uttrekk_6.csv --folder $SLADD_UTTREKK/uttrekk_6
    python utils/off_page_fnr.py --off-page off_page_u6.csv \\
        --labels $SLADD_LABELS/uttrekk_6.csv --folder $SLADD_UTTREKK/uttrekk_6 \\
        --png-dir off_page_png
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import fitz
from PIL import Image, ImageDraw

_UTILS = os.path.dirname(os.path.abspath(__file__))
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)

_APP = os.path.join(_UTILS, "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from config import PDF_DPI
from ocr_cache import read_cache as read_ocr_cache
from paddle_ocr_model_fnr import _normalize_ocr, build_lines, find_fnr, valid_mod11

PX_TO_PT = 72 / PDF_DPI
COVER_SHARE = 0.5      # share of the candidate's area one label must cover
LINE_SLACK_PT = 4.0    # pt; y slack when matching a label to a text line
TOLERANCE_PT = 1.0     # pt; same as off_page_labels.py


def _derive_ocr_cache(args):
    if args.ocr_cache:
        return args.ocr_cache
    base = os.environ.get("SLADD_CACHE")
    if base:
        return os.path.join(base, os.path.basename(os.path.normpath(args.folder)), "ocr")
    return None


def _f(row, key):
    return float(row[key])


def _box(row):
    x, y, w, h = _f(row, "x"), _f(row, "y"), _f(row, "width"), _f(row, "height")
    x0, x1 = sorted((x, x + w))
    y0, y1 = sorted((y, y + h))
    return (x0, y0, x1, y1)


def _is_true(value):
    return (value or "").strip().lower() in ("true", "t", "1")


def _burned(row):
    if not _is_true(row.get("ml_generated")):
        return True
    return (row.get("ml_status") or "").strip().upper() == "ACCEPTED"


def _who(row):
    if not _is_true(row.get("ml_generated")):
        return "manual"
    return "ML " + ((row.get("ml_status") or "").strip() or "unprocessed")


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _on_page(b, w, h):
    return b[2] > 0 and b[3] > 0 and b[0] < w and b[1] < h


def _inside(b, w, h):
    t = TOLERANCE_PT
    return b[0] >= -t and b[1] >= -t and b[2] <= w + t and b[3] <= h + t


def _same_line(a, b):
    return a[1] - LINE_SLACK_PT < b[3] and b[1] - LINE_SLACK_PT < a[3]


def _candidates(tokens):
    """[(box in pt, digit count, passes mod11)] for every fnr-shaped run."""
    out = []
    for _, text, map_ in build_lines(tokens):
        for hit in find_fnr(text, require_mod11=False):
            digits = [d for d in map_[hit.start:hit.end] if d is not None]
            if not digits:
                continue
            digit_str = _normalize_ocr("".join(
                ch for ch, d in zip(text[hit.start:hit.end], map_[hit.start:hit.end])
                if d is not None))
            box = (min(d.left for d in digits) * PX_TO_PT, min(d.top for d in digits) * PX_TO_PT,
                   max(d.right for d in digits) * PX_TO_PT, max(d.bottom for d in digits) * PX_TO_PT)
            out.append((box, len(digits), valid_mod11(digit_str)))
    return out


def _latest_labels(path, wanted):
    """{doc: [rows]} with only the highest versjon per document."""
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            doc = row.get("fil_revisjon_id", "").strip()
            if doc in wanted:
                rows[doc].append(row)
    latest = {}
    for doc, doc_rows in rows.items():
        top = max(int(float(r.get("versjon") or 1)) for r in doc_rows)
        latest[doc] = [r for r in doc_rows if int(float(r.get("versjon") or 1)) == top]
    return latest


def _fmt_box(b):
    return f"x={b[0]:.0f}..{b[2]:.0f} y={b[1]:.0f}..{b[3]:.0f}"


def _draw(page, labels, cands, w_pt, h_pt, dpi, path):
    pix = page.get_pixmap(dpi=dpi)
    page_img = Image.frombytes("RGB", (pix.w, pix.h), pix.samples)
    s = dpi / 72
    right = max([w_pt] + [_box(r)[2] for r in labels]) + 20
    canvas = Image.new("RGB", (int(right * s), pix.h), (255, 230, 230))
    canvas.paste(page_img, (0, 0))
    d = ImageDraw.Draw(canvas)
    for b, _, ok in cands:
        d.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], outline=(0, 0, 255), width=2)
    for r in labels:
        b = _box(r)
        if not _inside(b, w_pt, h_pt):
            colour = (220, 0, 0)
        else:
            colour = (0, 160, 0) if _burned(r) else (160, 160, 160)
        d.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], outline=colour, width=3)
    d.line([w_pt * s, 0, w_pt * s, pix.h], fill=(0, 0, 0), width=1)
    canvas.save(path)


def main():
    p = argparse.ArgumentParser(description="Check whether off-page labels left an fnr uncovered.")
    p.add_argument("--off-page", required=True, help="CSV from off_page_labels.py --out")
    p.add_argument("--labels", required=True, help="labels CSV with versjon, ml_generated, ml_status")
    p.add_argument("--folder", required=True, help="folder of PDF files")
    p.add_argument("--ocr-cache", default=None, help="OCR cache (default $SLADD_CACHE/<uttrekk>/ocr)")
    p.add_argument("--png-dir", default=None, help="write each page with labels drawn here")
    p.add_argument("--dpi", type=int, default=100, help="dpi for --png-dir (default: 100)")
    args = p.parse_args()

    ocr_dir = _derive_ocr_cache(args)
    if not ocr_dir or not os.path.isdir(ocr_dir):
        print(f"ERROR: no OCR cache folder: {ocr_dir}")
        return 1
    if args.png_dir:
        os.makedirs(args.png_dir, exist_ok=True)

    with open(args.off_page, newline="", encoding="utf-8") as f:
        off = list(csv.DictReader(f))
    pages = sorted({(r["fil_revisjon_id"], int(float(r["sidetall"]))) for r in off},
                   key=lambda t: (int(t[0]), t[1]))
    labels = _latest_labels(args.labels, {doc for doc, _ in pages})

    uncovered_on_line = 0
    for doc, side in pages:
        print(f"\n== {doc} side {side}")
        files = glob.glob(os.path.join(args.folder, doc + "*"))
        if not files:
            print("   no PDF in --folder")
            continue
        cache = read_ocr_cache(ocr_dir, os.path.basename(files[0]))
        if cache is None:
            print("   no OCR cache entry")
            continue
        rotations, tokens_per_page = cache
        if side > len(tokens_per_page):
            print("   page not in OCR cache")
            continue
        if rotations[side - 1]:
            print(f"   page turned by the orientation step (k={rotations[side - 1]}), skipped")
            continue

        with fitz.open(files[0]) as pdf:
            page = pdf[side - 1]
            w_pt, h_pt = page.rect.width, page.rect.height
            page_labels = [r for r in labels.get(doc, []) if int(float(r["sidetall"])) == side]
            cands = _candidates(tokens_per_page[side - 1])
            if args.png_dir:
                _draw(page, page_labels, cands, w_pt, h_pt, args.dpi,
                      os.path.join(args.png_dir, f"{doc}_s{side}.png"))

        burned_on_page = [_box(r) for r in page_labels
                          if _burned(r) and _on_page(_box(r), w_pt, h_pt)]
        print(f"   page {w_pt:.0f} x {h_pt:.0f} pt, {len(page_labels)} labels "
              f"({len(burned_on_page)} burned on the page), {len(cands)} fnr candidates")

        cand_state = []
        for b, n, ok in cands:
            cover = max((_intersection(b, lb) / _area(b) for lb in burned_on_page if _area(b)),
                        default=0.0)
            cand_state.append((b, n, ok, cover >= COVER_SHARE))
            print(f"   candidate {_fmt_box(b)} digits={n} mod11={'yes' if ok else 'no'} "
                  f"{'covered' if cover >= COVER_SHARE else 'UNCOVERED'} ({cover:.0%})")

        for r in page_labels:
            b = _box(r)
            if _inside(b, w_pt, h_pt):
                continue
            kind = "partial" if _on_page(b, w_pt, h_pt) else "outside"
            line = [c for c in cand_state if _same_line(b, c[0])]
            uncovered = [c for c in line if not c[3]]
            uncovered_on_line += bool(uncovered)
            verdict = ("UNCOVERED fnr on this line" if uncovered
                       else "line candidates all covered" if line else "no candidate on this line")
            print(f"   {kind:<8}{_who(r):<14}{_fmt_box(b)}  w={b[2] - b[0]:.0f}  -> {verdict}")
            for c in line:
                ratio = b[0] / c[0][0] if c[0][0] else float("nan")
                print(f"      line candidate x={c[0][0]:.0f}..{c[0][2]:.0f} "
                      f"{'covered' if c[3] else 'UNCOVERED'}, label x0 / candidate x0 = {ratio:.2f}")

    print(f"\nOff-page labels with an uncovered fnr candidate on their line: {uncovered_on_line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
