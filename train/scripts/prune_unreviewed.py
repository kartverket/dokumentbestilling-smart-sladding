"""Removes unreviewed documents from a dataset directory.

A page without boxes only works as a training negative when the document was
actually reviewed: it has fasit rows (iter_label_rows, so manglende counts) or
ml_behandlet set in a metadata CSV. A document with neither may hold unmarked
fnr, and its empty labels teach the model to miss. Dry run by default.

Kjør:
    python train/scripts/prune_unreviewed.py --dataset DIR \
        --labels u4.csv u6.csv --metadata u4.csv u6.csv --apply
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_UTILS = str(Path(__file__).resolve().parents[2] / "utils")
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)
from filter_common import iter_label_rows


def _norm_id(value):
    try:
        return str(int(float(str(value).strip())))
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help="dataset dir with images_all/labels_all")
    p.add_argument("--labels", nargs="+", required=True, help="labels CSVs")
    p.add_argument("--metadata", nargs="+", required=True, help="metadata CSVs")
    p.add_argument("--apply", action="store_true", help="delete instead of listing")
    args = p.parse_args()

    reviewed = set()
    for path in args.labels:
        for r in iter_label_rows(path):
            doc = _norm_id(r.get("fil_revisjon_id"))
            if doc:
                reviewed.add(doc)
    n_labeled = len(reviewed)
    for path in args.metadata:
        m = pd.read_csv(path)
        ids = m["fil_revisjon_id"].map(_norm_id)
        reviewed |= set(ids[m["ml_behandlet"].notna()].dropna())

    images_all = Path(args.dataset) / "images_all"
    labels_all = Path(args.dataset) / "labels_all"
    per_doc = {}
    for png in images_all.glob("*_p*.png"):
        per_doc.setdefault(png.stem.rsplit("_p", 1)[0], []).append(png)

    doomed = sorted(set(per_doc) - reviewed)
    n_files = sum(len(per_doc[d]) for d in doomed)
    print(f"{len(per_doc)} documents in {images_all} "
          f"({n_labeled} labeled, {len(reviewed)} reviewed in total)")
    print(f"{len(doomed)} unreviewed documents with {n_files} page-images"
          + (":" if doomed else "."))
    for doc in doomed[:10]:
        print(f"  {doc} ({len(per_doc[doc])} pages)")
    if len(doomed) > 10:
        print(f"  ... and {len(doomed) - 10} more")

    if not args.apply:
        if doomed:
            print("Dry run; use --apply to delete.")
        return
    for doc in doomed:
        for png in per_doc[doc]:
            (labels_all / (png.stem + ".txt")).unlink(missing_ok=True)
            png.unlink()
    print(f"Deleted {n_files} page-images and their labels.")


if __name__ == "__main__":
    main()
