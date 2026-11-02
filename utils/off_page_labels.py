"""
Pages whose geometry splits the model's coordinates from the application's,
and the labels that end up outside the page because of it.

The model reports boxes on the page as PyMuPDF shows it: the CropBox, with
/Rotate applied (page.rect). The saksbehandling frontend draws on a stage the
size of the MediaBox from pdf-lib, which ignores both, and PdfSladder burns
into the MediaBox the same way. On a page with /Rotate 90 or 270, or a CropBox
smaller than the MediaBox, the two spaces differ.

Without --labels the tool only counts pages: per /Rotate, and how many have a
CropBox that is not the MediaBox. With --labels it also places every label
row on its page in both spaces and classifies it inside, partial or outside.
A label outside the frontend stage but inside the model page is one the
rotation or CropBox explains. Labels on a page the PDF does not have are
counted separately, since that would be a different cause.

Reads every label row, REJECTED and ugyldige_labels.txt included: the point is
the state the frontend sees, not the fasit. «Unprocessed» means
ml_generated=true with an empty ml_status, which is what the frontend counts.

Edges. A PDF is matched to labels by the fil_revisjon_id at the start of its
filename, so a renamed file is silently unmatched. Labels for documents not in
--folder are ignored and counted. Coordinates are read with --y-origin
(default Y_ORIGIN from utils_config).

Run:
    python utils/off_page_labels.py --folder /path/to/uttrekk_6
    python utils/off_page_labels.py --folder /path/to/uttrekk_6 --labels labels.csv
    python utils/off_page_labels.py --folder /path/to/uttrekk_6 --labels labels.csv \\
        --out off_page.csv --examples 50
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

import fitz

_UTILS = os.path.dirname(os.path.abspath(__file__))
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)

from file_selection import select_files

try:
    from utils_config import Y_ORIGIN
except ImportError:
    Y_ORIGIN = "top"

TOLERANCE_PT = 1.0   # pt; rounding in the pipeline must not count as outside

OUT_FIELDS = ["fil_revisjon_id", "sidetall", "id", "ml_generated", "ml_status",
              "ml_kilde", "x", "y", "width", "height", "rotate",
              "mediabox_w", "mediabox_h", "cropbox_w", "cropbox_h",
              "cropbox_x", "cropbox_y", "stage", "model"]


def _doc_no(name):
    m = re.match(r"0*(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


def _page_info(page):
    mb = page.mediabox
    cb = page.cropbox
    pos = page.cropbox_position
    return {
        "rotate": page.rotation,
        "mediabox_w": mb.width, "mediabox_h": mb.height,
        "cropbox_w": cb.width, "cropbox_h": cb.height,
        "cropbox_x": pos.x, "cropbox_y": pos.y,
        "view_w": page.rect.width, "view_h": page.rect.height,
    }


def _cropbox_differs(info):
    return (abs(info["cropbox_w"] - info["mediabox_w"]) > TOLERANCE_PT
            or abs(info["cropbox_h"] - info["mediabox_h"]) > TOLERANCE_PT
            or abs(info["cropbox_x"]) > TOLERANCE_PT
            or abs(info["cropbox_y"]) > TOLERANCE_PT)


def _classify(box, width, height):
    """inside, partial or outside for a box against (0, 0, width, height)."""
    x0, y0, x1, y1 = box
    t = TOLERANCE_PT
    if x0 >= -t and y0 >= -t and x1 <= width + t and y1 <= height + t:
        return "inside"
    if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
        return "outside"
    return "partial"


def _label_box(row, page_h, y_origin):
    try:
        x, y = float(row["x"]), float(row["y"])
        w, h = float(row["width"]), float(row["height"])
    except (TypeError, ValueError, KeyError):
        return None
    x0, x1 = sorted((x, x + w))
    y0, y1 = sorted((y, y + h))
    if y_origin == "bottom":
        y0, y1 = page_h - y1, page_h - y0
    return (x0, y0, x1, y1)


def _is_true(value):
    return (value or "").strip().lower() in ("true", "t", "1")


def _scan_pdfs(files):
    """{doc_no: [page info, ...]} plus the files that could not be opened."""
    pages, failed = {}, []
    for i, path in enumerate(files, start=1):
        nr = _doc_no(path)
        if nr is None:
            continue
        try:
            with fitz.open(path) as doc:
                pages[nr] = [_page_info(p) for p in doc]
        except Exception as e:
            failed.append((os.path.basename(path), repr(e)))
        if i % 1000 == 0:
            print(f"   {i}/{len(files)} PDFs read")
    return pages, failed


def _print_page_summary(pages):
    per_rotate = Counter()
    cropbox_pages = rotated_docs = cropbox_docs = total = 0
    for infos in pages.values():
        total += len(infos)
        rotated = cropped = False
        for info in infos:
            per_rotate[info["rotate"]] += 1
            if info["rotate"] in (90, 270):
                rotated = True
            if _cropbox_differs(info):
                cropbox_pages += 1
                cropped = True
        rotated_docs += rotated
        cropbox_docs += cropped

    def share(n, of):
        return f"({100 * n / of:.2f} %)" if of else ""

    print()
    print(f"Documents                       {len(pages):>8}")
    print(f"Pages                           {total:>8}")
    for deg in sorted(per_rotate):
        n = per_rotate[deg]
        print(f"  /Rotate {deg:>3}                    {n:>8}  {share(n, total)}")
    print(f"  CropBox differs from MediaBox {cropbox_pages:>8}  {share(cropbox_pages, total)}")
    print(f"Documents with /Rotate 90/270   {rotated_docs:>8}  {share(rotated_docs, len(pages))}")
    print(f"Documents with a CropBox        {cropbox_docs:>8}  {share(cropbox_docs, len(pages))}")


def _page_kind(info):
    kind = f"rotate {info['rotate']}"
    if _cropbox_differs(info):
        kind += " + cropbox"
    return kind


def _check_labels(path, pages, y_origin):
    tally = Counter()
    by_kind = defaultdict(Counter)
    flagged = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                nr = int(float(row["fil_revisjon_id"]))
                side = int(float(row["sidetall"]))
            except (TypeError, ValueError, KeyError):
                tally["unreadable row"] += 1
                continue
            infos = pages.get(nr)
            if infos is None:
                tally["document not in --folder"] += 1
                continue
            if not 1 <= side <= len(infos):
                tally["page not in PDF"] += 1
                flagged.append({**row, "stage": "no such page", "model": ""})
                continue
            info = infos[side - 1]
            # The frontend stage and the model page differ in height on a
            # rotated page, so a bottom origin has to be resolved per space.
            stage_box = _label_box(row, info["mediabox_h"], y_origin)
            model_box = _label_box(row, info["view_h"], y_origin)
            if stage_box is None:
                tally["unreadable row"] += 1
                continue
            tally["placed"] += 1
            stage = _classify(stage_box, info["mediabox_w"], info["mediabox_h"])
            model = _classify(model_box, info["view_w"], info["view_h"])
            ml = _is_true(row.get("ml_generated"))
            unprocessed = ml and not (row.get("ml_status") or "").strip()
            key = ("ML" if ml else "manual") + (" unprocessed" if unprocessed else "")
            by_kind[_page_kind(info)][(key, stage, model)] += 1
            if stage != "inside":
                flagged.append({**row, **{k: info[k] for k in OUT_FIELDS if k in info},
                                "stage": stage, "model": model})
    return tally, by_kind, flagged


def _print_label_summary(tally, by_kind):
    print()
    print("Label rows")
    for reason in ("placed", "page not in PDF", "document not in --folder",
                   "unreadable row"):
        if tally[reason]:
            print(f"  {reason:<30}{tally[reason]:>8}")
    print()
    print("stage = frontend (MediaBox, no /Rotate); model = page.rect (CropBox, /Rotate)")
    print(f"  {'page':<22}{'label':<20}{'stage':<10}{'model':<10}{'n':>8}")
    for kind in sorted(by_kind):
        for (who, stage, model), n in sorted(by_kind[kind].items()):
            print(f"  {kind:<22}{who:<20}{stage:<10}{model:<10}{n:>8}")


def _fmt(value):
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def _print_examples(flagged, limit):
    if not flagged or not limit:
        return
    print()
    print(f"Labels not fully on the frontend stage (first {min(limit, len(flagged))} of {len(flagged)})")
    for r in flagged[:limit]:
        print("  " + "  ".join(f"{k}={_fmt(r.get(k, ''))}" for k in
                               ("fil_revisjon_id", "sidetall", "ml_generated", "ml_status",
                                "x", "y", "width", "height", "rotate",
                                "mediabox_w", "mediabox_h", "stage", "model")))


def _write_out(path, flagged):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in flagged:
            w.writerow({k: _fmt(r.get(k, "")) for k in OUT_FIELDS})
    print(f"\nWrote {len(flagged)} rows to {path}")


def main():
    p = argparse.ArgumentParser(
        description="Count pages with /Rotate or a CropBox, and find labels "
                    "that fall outside the page the frontend draws.")
    p.add_argument("--folder", required=True, help="folder of PDF files")
    p.add_argument("--labels", default=None,
                   help="labels CSV (fil_revisjon_id, sidetall, x, y, width, height, ...)")
    p.add_argument("--y-origin", choices=("top", "bottom"), default=Y_ORIGIN,
                   help=f"origin of y in the labels CSV (default: {Y_ORIGIN})")
    p.add_argument("--out", default=None,
                   help="write every label not fully on the stage to this CSV")
    p.add_argument("--examples", type=int, default=20,
                   help="print this many of those labels (default: 20)")
    p.add_argument("--select", nargs="*", default=[],
                   help="specific files (filename/substring)")
    p.add_argument("--select-from-file", default=None,
                   help="read file IDs from a text file, one per line")
    p.add_argument("--count", default="all",
                   help="number of files when --select is empty (a number, or 'all')")
    args = p.parse_args()

    if not os.path.isdir(args.folder):
        print(f"ERROR: --folder does not exist: {args.folder}")
        return 1
    if args.labels and not os.path.isfile(args.labels):
        print(f"ERROR: --labels does not exist: {args.labels}")
        return 1

    select, from_file = args.select, False
    if args.select_from_file:
        with open(args.select_from_file, encoding="utf-8") as f:
            select = [line.strip() for line in f if line.strip()]
        from_file = True

    files = [f for f in select_files(args.folder, select, args.count, exact=from_file)
             if f.lower().endswith(".pdf")]
    if not files:
        print("No PDF files to read. Check --folder / --select / --count.")
        return 1

    pages, failed = _scan_pdfs(files)
    for name, err in failed[:10]:
        print(f"!! {name}: could not open ({err})")
    if failed:
        print(f"!! {len(failed)} PDF(s) could not be opened")
    _print_page_summary(pages)

    if args.labels:
        tally, by_kind, flagged = _check_labels(args.labels, pages, args.y_origin)
        _print_label_summary(tally, by_kind)
        _print_examples(flagged, args.examples)
        if args.out:
            _write_out(args.out, flagged)
    return 0


if __name__ == "__main__":
    sys.exit(main())
