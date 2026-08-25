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

One sladding drawn over two numbers is the same job twice. The oversized
label covers neither number well enough for HIT_THRESHOLD, so it counts as a
miss while both correct predictions count as oversladding. --erstatt retracts
that label and puts this box in its place; run it once per number.

Kjør:
    python utils/note_missing_label.py --png <run>/error_images/oversladd/0104822_rev3_side3.png
    python utils/note_missing_label.py --png <same>.png --box 2 \
        --coords 640 1210 960 1252 --kommentar "fnr i tabellkolonne, ingen label"

    # én fasit-sladding over to numre: kjør én gang per boks
    python utils/note_missing_label.py --png <same>.png --box 2 --erstatt \
        --truth-csv $SLADD_LABELS/uttrekk_4.csv
    python utils/note_missing_label.py --png <same>.png --box 3 --erstatt \
        --truth-csv $SLADD_LABELS/uttrekk_4.csv

    # uten bildet, om du har tallene fra før
    python utils/note_missing_label.py --res-csv <run>/resultat.csv --doc 104822 --side 3
"""

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from filter_common import (INVALID_LABELS_FILE, MISSING_LABELS_FILE,
                           MISSING_LABEL_FIELD, doc_no, label_row_from_prediction,
                           missing_label_id, read_invalid_label_ids,
                           read_missing_label_rows, _label_box, _label_key,
                           _same_box)


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


def _labels_on_page(truth_csv, doc, side):
    """Raw label rows for one page, ids included, nothing filtered away."""
    with open(truth_csv, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if _label_key(r) == (doc, side)]


def _covering_label(truth_csv, doc, side, box):
    """The fasit label that swallows `box`, when a caseworker drew one
    sladding over two numbers. Returns (row, [all candidates])."""
    hits = [r for r in _labels_on_page(truth_csv, doc, side)
            if (_label_box(r) or None) and _same_box(box, _label_box(r))]
    return (hits[0] if len(hits) == 1 else None), hits


def _retract(label_id, comment, path=INVALID_LABELS_FILE):
    """Adds a label id to ugyldige_labels.txt. Already listed is not an error.

    The file has historically ended without a newline, and appending to that
    glues the new id onto the last one, where neither parses.
    """
    if label_id in read_invalid_label_ids(path):
        return False
    ends_clean = True
    if os.path.isfile(path) and os.path.getsize(path):
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            ends_clean = f.read(1) == b"\n"
    with open(path, "a", encoding="utf-8") as f:
        if not ends_clean:
            f.write("\n")
        f.write(f"{label_id}\t# {comment}\n" if comment else f"{label_id}\n")
    return True


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
    p.add_argument("--truth-csv", default=None, metavar="STI",
                   help="labels CSV, needed by --erstatt")
    p.add_argument("--erstatt", action="store_true",
                   help="the fasit has ONE sladding over this number and a "
                        "neighbour, so it scores as both a miss and an "
                        "oversladding. Retracts that label into "
                        "ugyldige_labels.txt and records this box instead. "
                        "Run once per number in the shared box.")
    p.add_argument("--fil", default=MISSING_LABELS_FILE,
                   help=f"where to append (default {MISSING_LABELS_FILE})")
    p.add_argument("--ugyldige-fil", default=INVALID_LABELS_FILE,
                   help="where --erstatt retracts (default ugyldige_labels.txt)")
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

    if a.erstatt:
        if not a.truth_csv:
            sys.exit("--erstatt needs --truth-csv to find the label to retract.")
        one, alle = _covering_label(a.truth_csv, a.doc, a.side, _label_box(ny))
        if not alle:
            sys.exit(f"No fasit label covers this box on doc {a.doc} page "
                     f"{a.side}. Then it is a plain missing label, drop --erstatt.")
        if one is None:
            print(f"  {len(alle)} labels cover this box, expected one:")
            for r in alle:
                print(f"    {r.get('id')}  x={r['x']} y={r['y']} "
                      f"w={r['width']} h={r['height']}")
            sys.exit("  Retract the right one by hand in ugyldige_labels.txt.")
        why = a.kommentar or f"én sladding over flere numre, erstattet av {label_id}"
        if _retract(one["id"], why, a.ugyldige_fil):
            print(f"  Retracted {one['id']} in {a.ugyldige_fil}")
        else:
            print(f"  {one['id']} was already retracted")

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
