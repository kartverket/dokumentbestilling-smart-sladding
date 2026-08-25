"""
Fnr and oversladding statistics per rettsstiftelse type.

Joins the metadata CSV (rettsstiftelse types per document) against the truth
labels and the result CSV, and shows per XX_YYY code how fnr-dense the documents
are and where the oversladding (BOM) comes from, that is which rettsstiftelser are
worth their own filter profiles. Coordinate-heavy types (målebrev and the like)
should show high BOM density, hjemmelsdokumenter high fnr density.

A document with several codes is counted under each of them, so the columns sum to
MORE than the totals at the bottom.

--write-lists DIR writes one ID list per code (rs_<CODE>.txt) for measuring a rule
on a single type:

    python utils/filter_sweep.py ... --processed-list DIR/rs_SR_ERK.txt
    python utils/filter_review.py ... --processed-list DIR/rs_SR_ERK.txt

Run:
    python utils/rettsstiftelse_stat.py \
        --metadata-csv $SLADD_METADATA/uttrekk_6.csv \
        --truth-csv $SLADD_LABELS/uttrekk_6.csv \
        --res-csv $SLADD_VALIDATION/<run>/resultat.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from filter_common import (STD_CRITERION, STD_SLOPPINESS_FACTOR, STD_THRESHOLD,
                           build_dataset, read_truth_boxes, read_predictions)


def read_metadata(path):
    """fil_revisjon_id -> (koder, elektronisk); kode -> beskrivelse."""
    per_doc, desc = {}, {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                doc = int(r["fil_revisjon_id"])
            except (KeyError, TypeError, ValueError):
                continue
            codes = []
            for part in (r.get("rettsstiftelsestyper") or "").split(","):
                part = part.strip()
                if not part:
                    continue
                code, _, text = part.partition(" - ")
                code = code.strip()
                if code:
                    codes.append(code)
                    if text.strip():
                        desc.setdefault(code, text.strip())
            elektronisk = ((r.get("er_elektronisk_tinglyst") or "")
                           .strip().lower() in ("true", "t", "1"))
            per_doc[doc] = (codes, elektronisk)
    return per_doc, desc


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--truth-csv", required=True)
    p.add_argument("--res-csv", required=True)
    p.add_argument("--threshold", type=float, default=STD_THRESHOLD)
    p.add_argument("--criterion", default=STD_CRITERION)
    p.add_argument("--oversize-factor", type=float, default=STD_SLOPPINESS_FACTOR)
    p.add_argument("--exclude-unlabelled", dest="include_unlabelled",
                   action="store_false", default=True,
                   help="Old behaviour: only documents with truth rows")
    p.add_argument("--top", type=int, default=None, metavar="N",
                   help="Show only the N codes with the most BOM")
    p.add_argument("--min-doc", type=int, default=1, metavar="N",
                   help="Hide codes with fewer than N processed documents")
    p.add_argument("--write-lists", default=None, metavar="DIR",
                   help="Write one ID list per code (rs_<CODE>.txt) for use "
                        "with --processed-list in sweep/review")
    args = p.parse_args()

    meta, desc = read_metadata(args.metadata_csv)
    ds = build_dataset(read_truth_boxes(args.truth_csv),
                       read_predictions(args.res_csv),
                       threshold=args.threshold, oversize_factor=args.oversize_factor,
                       include_unlabelled=args.include_unlabelled,
                       criterion=args.criterion)

    truth_per_doc = defaultdict(int)
    for fb in ds.truth_boxes:
        truth_per_doc[fb["doc_no"]] += 1

    pred_per_doc = defaultdict(list)
    for pr in ds.pred:
        pred_per_doc[pr["doc_no"]].append(pr)

    SOURCES = ("begge", "yolo", "paddle")

    def _new():
        return {"dok_meta": 0, "dok_kjort": 0, "fasit": 0, "treff": 0,
                "bom": 0, **{f"bom_{k}": 0 for k in SOURCES}}

    per_code = defaultdict(_new)
    elektronisk = {False: _new(), True: _new()}
    without_meta = _new()

    def _count(row, doc, ran):
        row["dok_meta"] += 1
        if not ran:
            return
        row["dok_kjort"] += 1
        row["fasit"] += truth_per_doc.get(doc, 0)
        for pr in pred_per_doc.get(doc, ()):
            if pr["klasse"] == "BOM":
                row["bom"] += 1
                sort_key = f"bom_{pr['kilde'].lower()}"
                if sort_key in row:
                    row[sort_key] += 1
            elif pr["klasse"] == "TREFF":
                row["treff"] += 1

    for doc, (codes, el) in meta.items():
        ran = doc in ds.scope_doc
        _count(elektronisk[el], doc, ran)
        for code in set(codes):
            _count(per_code[code], doc, ran)

    for doc in ds.scope_doc:
        if doc not in meta:
            _count(without_meta, doc, True)
    without_meta["dok_meta"] = without_meta["dok_kjort"]

    total_miss = ds.n_miss or 1

    print(f"Metadata: {len(meta)} documents, {len(per_code)} unique codes")
    print(f"Scope:    {len(ds.scope_doc)} processed documents, "
          f"{len(ds.truth_boxes)} truth boxes, {ds.n_miss} BOM in total")
    print("A document counts under EACH of its codes, the columns "
          "sum to more than the total.\n")

    header = (f"  {'code':<8} {'docs':>6} {'run':>6} {'truth':>6} "
            f"{'fnr/doc':>8} {'hits':>6} {'bom':>6} {'bom/doc':>8} "
            f"{'bom%':>6}  {'b/y/p':>13}  description")
    print(header)
    print(f"  {'─' * (len(header) + 8)}")

    def _row(name, r, text=""):
        kj = r["dok_kjort"]
        print(f"  {name:<8} {r['dok_meta']:>6} {kj:>6} {r['fasit']:>6} "
              f"{(r['fasit'] / kj if kj else 0):>8.2f} "
              f"{r['treff']:>6} {r['bom']:>6} "
              f"{(r['bom'] / kj if kj else 0):>8.2f} "
              f"{r['bom'] / total_miss * 100:>5.1f}%  "
              f"{r['bom_begge']:>4}/{r['bom_yolo']:>4}/{r['bom_paddle']:>3}  "
              f"{text[:44]}")

    rows = [(k, r) for k, r in per_code.items()
             if r["dok_kjort"] >= args.min_doc]
    rows.sort(key=lambda kr: -kr[1]["bom"])
    hidden = len(per_code) - len(rows)
    if args.top:
        hidden += max(0, len(rows) - args.top)
        rows = rows[:args.top]

    for code, r in rows:
        _row(code, r, desc.get(code, ""))
    if without_meta["dok_kjort"]:
        _row("(none)", without_meta, "processed, but no metadata row")
    if hidden:
        print(f"  ({hidden} codes hidden, --top/--min-doc)")

    print("\n  Elektronisk tinglyst vs. scanned:")
    for el, name in ((False, "scanned"), (True, "electr.")):
        _row(name[:8], elektronisk[el])

    if args.write_lists:
        os.makedirs(args.write_lists, exist_ok=True)
        n = 0
        for code, r in per_code.items():
            if r["dok_kjort"] < args.min_doc:
                continue
            doc_ids = sorted(d for d, (codes, _el) in meta.items()
                             if code in codes and d in ds.scope_doc)
            path = os.path.join(args.write_lists, f"rs_{code}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(str(d) for d in doc_ids) + "\n")
            n += 1
        print(f"\n  {n} ID lists written to {args.write_lists}/rs_<CODE>.txt")
        print("  Use: filter_sweep/filter_review ... --processed-list "
              f"{args.write_lists}/rs_<CODE>.txt")


if __name__ == "__main__":
    main()
