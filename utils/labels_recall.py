"""Recall and precision of the current solution, straight from a labels CSV.

Run:
    python utils/labels_recall.py --csv smartsladding_uttrekk_labels_2_29_06_26.csv
"""

import argparse
import csv
from collections import Counter

from filter_common import iter_label_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="smartsladding_uttrekk_labels_2_29_06_26.csv")
    args = p.parse_args()

    total = 0
    tp = fp = fn = 0
    unresolved = 0                     # ml_generated=true with empty/unknown status
    type_count = {}                 # type -> [tp, fp, fn]
    cross_table = Counter()          # (ml_generated, status) -> count

    info = {}
    for row in iter_label_rows(args.csv, exclude_status=(), info=info):
        total += 1
        status = (row.get("ml_status") or "").strip().upper()
        typ = (row.get("type") or "").strip()
        ml = (row.get("ml_generated") or "").strip().lower() == "true"

        cross_table[("true" if ml else "false", status or "(empty)")] += 1

        t = type_count.setdefault(typ, [0, 0, 0])
        if ml and status == "ACCEPTED":
            tp += 1
            t[0] += 1
        elif ml and status == "REJECTED":
            fp += 1
            t[1] += 1
        elif not ml:
            fn += 1
            t[2] += 1
        else:
            unresolved += 1
    invalid_skipped = info["discarded"]["(ugyldig-listet)"]

    print(f"\n=== Total: {total} boxes ===")
    if invalid_skipped:
        print(f"  ({invalid_skipped} boxes in ugyldige_labels.txt excluded)")
    print(f"  TP (ml + ACCEPTED)    : {tp:>6}")
    print(f"  FP (ml + REJECTED)    : {fp:>6}")
    print(f"  FN (added by hand)    : {fn:>6}")
    if unresolved:
        print(f"  Unresolved (ml, other status): {unresolved} - left out of the numbers")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"\n  Precision: {precision:.1%}")
    print(f"  Recall   : {recall:.1%}")

    print("\n=== Cross: ml_generated x ml_status ===")
    for (ml, status), n in sorted(cross_table.items()):
        print(f"  ml_generated={ml:<5}  status={status:<10}  {n}")

    print("\n=== Per type ===")
    for typ, (a, r, m) in sorted(type_count.items()):
        pre = a / (a + r) if a + r else 0.0
        rec = a / (a + m) if a + m else 0.0
        print(f"  {typ or '(empty)':<20}  TP: {a:>5}  FP: {r:>5}  FN: {m:>5}"
              f"  precision: {pre:6.1%}  recall: {rec:6.1%}")


if __name__ == "__main__":
    main()