"""Records a fødselsnummer the fasit is missing, from an over-sladding review.

The opposite of ugyldige_labels.txt: that one removes label rows, this one
adds them. Reviewing a page in <run>/error_images/oversladd/ you sometimes
find that the «over-sladding» is a real fnr the labelling never got. This
writes that box into manglende_labels.csv, converted from the result CSV's
pixels at PDF_DPI to the labels CSV's points, so nobody types coordinates.

Point it at the PNG you are looking at. Document, page and the run's result
CSV all follow from that path, and every prediction box in the image carries
its «#N» in the corner, so --box N is read off the picture.

The box you pick is the DETECTION box, which is usually padded. --coords takes
a tighter one, in the PNG's own pixel coordinates: the page is rendered at
PDF_DPI and the result CSV stores the same space, so what a picture viewer
shows you is what this wants. A tight box matters, because HIT_THRESHOLD
measures coverage of the truth area.

Kjør:
    python utils/note_missing_label.py --png <run>/error_images/oversladd/0104822_rev3_side3.png
    python utils/note_missing_label.py --png <same>.png --box 2 \
        --coords 640 1210 960 1252 --kommentar "fnr i tabellkolonne, ingen label"

    # uten bildet, om du har tallene fra før
    python utils/note_missing_label.py --res-csv <run>/resultat.csv --doc 104822 --side 3
"""

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from filter_common import (MISSING_LABELS_FILE, MISSING_LABEL_FIELD, doc_no,
                           label_row_from_prediction, missing_label_id,
                           read_missing_label_rows)


def _from_png(path):
    """(doc_no, side) from an error image named <pdf>_side<N>.png."""
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"^(.*)_side(\d+)$", base)
    if not m:
        sys.exit(f"Cannot read document and page from {os.path.basename(path)}. "
                 f"Expected <pdf>_side<N>.png.")
    nr = doc_no(m.group(1))
    if nr is None:
        sys.exit(f"No document number in {m.group(1)}")
    return nr, int(m.group(2))


def _find_res_csv(png_path):
    """resultat.csv from the run the image belongs to.

    Error images live in <run>/error_images/<bom|oversladd>/, so the CSV is a
    couple of levels up. Searched rather than assumed, since --png-dir can be
    pointed anywhere.
    """
    d = os.path.dirname(os.path.abspath(png_path))
    for _ in range(4):
        candidate = os.path.join(d, "resultat.csv")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _rows_on_page(res_csv, doc, side):
    with open(res_csv, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f)
                if doc_no(r.get("navn", "")) == doc
                and str(r.get("side", "")).strip() == str(side)]


def _list(rows, doc, side):
    if not rows:
        print(f"No predictions for doc {doc} page {side} in the result CSV.")
        return
    print(f"  {len(rows)} prediction(s) on doc {doc} page {side}:\n")
    for i, r in enumerate(rows):
        lab = label_row_from_prediction(r)
        print(f"  --box {i}  kilde={r.get('kilde', '?'):6} "
              f"conf={r.get('yolo_conf') or '-':>6}  "
              f"px=({float(r['x0']):.0f},{float(r['y0']):.0f})-"
              f"({float(r['x1']):.0f},{float(r['y1']):.0f})  "
              f"pt=x{lab['x']} y{lab['y']} w{lab['width']} h{lab['height']}")
    print("\n  Pick one with --box N. Use --coords X0 Y0 X1 Y1 (result pixels) "
          "for a tighter box.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--png", default=None, metavar="STI",
                   help="the error image you are looking at. Gives document, "
                        "page and the run's result CSV in one argument")
    p.add_argument("--res-csv", default=None, help="resultat.csv from the run")
    p.add_argument("--doc", type=int, default=None, metavar="NR",
                   help="fil_revisjon_id (the document number)")
    p.add_argument("--side", type=int, default=None, help="page number")
    p.add_argument("--box", type=int, default=None, metavar="N",
                   help="index from the listing. Without it the page is listed")
    p.add_argument("--coords", type=float, nargs=4, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="tighter box in the result CSV's pixel space, "
                        "replacing the detection box's padding")
    p.add_argument("--type", default="", help="truth type, if the labels use one")
    p.add_argument("--kommentar", default="", help="why this row was added")
    p.add_argument("--fil", default=MISSING_LABELS_FILE,
                   help=f"where to append (default {MISSING_LABELS_FILE})")
    a = p.parse_args()

    if a.png:
        png_doc, png_side = _from_png(a.png)
        a.doc = a.doc if a.doc is not None else png_doc
        a.side = a.side if a.side is not None else png_side
        a.res_csv = a.res_csv or _find_res_csv(a.png)
        if not a.res_csv:
            sys.exit(f"Found no resultat.csv above {a.png}. Give --res-csv.")
    if not a.res_csv or a.doc is None or a.side is None:
        sys.exit("Need --png, or --res-csv with --doc and --side.")

    rows = _rows_on_page(a.res_csv, a.doc, a.side)
    if a.box is None:
        _list(rows, a.doc, a.side)
        return
    if not 0 <= a.box < len(rows):
        sys.exit(f"--box {a.box} is outside 0..{len(rows) - 1}")

    row = dict(rows[a.box])
    if a.coords:
        row["x0"], row["y0"], row["x1"], row["y1"] = a.coords
    ny = label_row_from_prediction(row)
    ny["type"] = a.type
    ny["kommentar"] = a.kommentar
    label_id = missing_label_id(ny)

    if any(r["id"] == label_id for r in read_missing_label_rows(a.fil)):
        print(f"  Already recorded: {label_id}")
        return

    new_file = not os.path.isfile(a.fil) or os.path.getsize(a.fil) == 0
    with open(a.fil, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MISSING_LABEL_FIELD)
        if new_file:
            w.writeheader()
        w.writerow(ny)

    print(f"  Added to {a.fil}:")
    print(f"    doc {ny['fil_revisjon_id']} page {ny['sidetall']}  "
          f"x={ny['x']} y={ny['y']} w={ny['width']} h={ny['height']}")
    print(f"    id {label_id}  (put it in ugyldige_labels.txt to retract)")


if __name__ == "__main__":
    main()
