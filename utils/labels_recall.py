import argparse
import csv
from collections import Counter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="smartsladding_uttrekk_labels_2_29_06_26.csv")
    args = p.parse_args()

    totalt = 0
    tp = fp = fn = 0
    uavklart = 0                     # ml_generated=true med tom/ukjent status
    type_teller = {}                 # type -> [tp, fp, fn]
    krysstabell = Counter()          # (ml_generated, status) -> antall

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        for rad in csv.DictReader(f):
            totalt += 1
            status = (rad.get("ml_status") or "").strip().upper()
            typ = (rad.get("type") or "").strip()
            ml = (rad.get("ml_generated") or "").strip().lower() == "true"

            krysstabell[("true" if ml else "false", status or "(tom)")] += 1

            t = type_teller.setdefault(typ, [0, 0, 0])
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
                uavklart += 1       
    print(f"\n=== Totalt: {totalt} bokser ===")
    print(f"  TP (ml + ACCEPTED)    : {tp:>6}")
    print(f"  FP (ml + REJECTED)    : {fp:>6}")
    print(f"  FN (manuelt lagt til) : {fn:>6}")
    if uavklart:
        print(f"  Uavklart (ml, annen status): {uavklart} - holdt utenfor tallene")

    presisjon = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"\n  Presisjon: {presisjon:.1%}")
    print(f"  Recall   : {recall:.1%}")

    print("\n=== Kryss: ml_generated x ml_status ===")
    for (ml, status), n in sorted(krysstabell.items()):
        print(f"  ml_generated={ml:<5}  status={status:<10}  {n}")

    print("\n=== Per type ===")
    for typ, (a, r, m) in sorted(type_teller.items()):
        pre = a / (a + r) if a + r else 0.0
        rec = a / (a + m) if a + m else 0.0
        print(f"  {typ or '(tom)':<20}  TP: {a:>5}  FP: {r:>5}  FN: {m:>5}"
              f"  presisjon: {pre:6.1%}  recall: {rec:6.1%}")


if __name__ == "__main__":
    main()