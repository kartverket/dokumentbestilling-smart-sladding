"""
Analyserer dimensjonene til fasit-bokser fra labels-CSV for å forstå
typiske størrelser og finne gode terskler for filtrering.

Kjør:
    python utils/analyse_label_storrelse.py --csv smartsladding_uttrekk_labels_5_29_07_26.csv
"""

import argparse
import csv
import statistics


def les(sti):
    bokser = []
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                w = float(r["width"])
                h = float(r["height"])
                ml = (r.get("ml_generated") or "").strip().lower() == "true"
                status = (r.get("ml_status") or "").strip().upper()
            except (TypeError, ValueError, KeyError):
                continue
            bokser.append({"w": w, "h": h, "ml": ml, "status": status,
                           "areal": abs(w * h), "ratio": w / h if h != 0 else 0})
    return bokser


def persentiler(verdier, navn):
    verdier = sorted(verdier)
    n = len(verdier)
    ps = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  {navn} (n={n}):")
    print(f"    min={verdier[0]:.2f}  maks={verdier[-1]:.2f}  "
          f"snitt={statistics.mean(verdier):.2f}  std={statistics.stdev(verdier):.2f}")
    for p in ps:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        print(f"    P{p:02d} = {verdier[idx]:.2f}")


def main():
    p = argparse.ArgumentParser(description="Analysér dimensjoner i labels-CSV")
    p.add_argument("--csv", default="smartsladding_uttrekk_labels_5_29_07_26.csv")
    args = p.parse_args()

    alle = les(args.csv)
    print(f"Totalt {len(alle)} bokser lest fra {args.csv}")

    # Del opp i kategorier
    accepted = [b for b in alle if b["ml"] and b["status"] == "ACCEPTED"]
    rejected = [b for b in alle if b["ml"] and b["status"] == "REJECTED"]
    manuell = [b for b in alle if not b["ml"]]

    # Negative dimensjoner
    neg = [b for b in alle if b["w"] <= 0 or b["h"] <= 0]
    print(f"\nBokser med negativ bredde eller høyde: {len(neg)}")
    if neg:
        for b in neg[:5]:
            print(f"  w={b['w']:.1f} h={b['h']:.1f} ml={b['ml']} status={b['status']}")
        if len(neg) > 5:
            print(f"  ... og {len(neg) - 5} til")

    # Bare positive dimensjoner fra nå av
    pos = [b for b in alle if b["w"] > 0 and b["h"] > 0]
    pos_accepted = [b for b in accepted if b["w"] > 0 and b["h"] > 0]

    print(f"\n{'=' * 60}")
    print(f"ALLE positive bokser (n={len(pos)}):")
    persentiler([b["w"] for b in pos], "Bredde (pt)")
    persentiler([b["h"] for b in pos], "Høyde (pt)")
    persentiler([b["areal"] for b in pos], "Areal (pt²)")
    persentiler([b["ratio"] for b in pos], "Ratio (w/h)")

    print(f"\n{'=' * 60}")
    print(f"Kun ACCEPTED (ml_generated + godkjent, n={len(pos_accepted)}):")
    persentiler([b["w"] for b in pos_accepted], "Bredde (pt)")
    persentiler([b["h"] for b in pos_accepted], "Høyde (pt)")
    persentiler([b["areal"] for b in pos_accepted], "Areal (pt²)")
    persentiler([b["ratio"] for b in pos_accepted], "Ratio (w/h)")

    # Konverter til piksler (300 DPI)
    skala = 300 / 72.0
    print(f"\n{'=' * 60}")
    print(f"ACCEPTED i piksler (300 DPI, skala={skala:.3f}):")
    persentiler([b["w"] * skala for b in pos_accepted], "Bredde (px)")
    persentiler([b["h"] * skala for b in pos_accepted], "Høyde (px)")
    persentiler([b["w"] * b["h"] * skala**2 for b in pos_accepted], "Areal (px²)")

    # Sammenlign med nåværende terskler
    min_areal_px = 965
    maks_bredde_pt = 50
    maks_bredde_px = maks_bredde_pt * skala

    print(f"\n{'=' * 60}")
    print("Sammenligning med nåværende filtre:")
    print(f"  MIN_BOKS_AREAL = {min_areal_px} px²")
    under_min = sum(1 for b in pos_accepted if b["w"] * b["h"] * skala**2 < min_areal_px)
    print(f"    ACCEPTED-bokser som ville blitt filtrert bort: {under_min}/{len(pos_accepted)}")

    print(f"  MAKS_BREDDE_PT = {maks_bredde_pt} pt ({maks_bredde_px:.0f} px)")
    over_maks = sum(1 for b in pos_accepted if b["w"] > maks_bredde_pt)
    print(f"    ACCEPTED-bokser som ville blitt filtrert bort: {over_maks}/{len(pos_accepted)}")

    # Forslag til nye terskler
    print(f"\n{'=' * 60}")
    print("Mulige nye filtre (basert på data):")

    # Maks høyde
    hoyder = sorted(b["h"] for b in pos_accepted)
    p99_h = hoyder[int(len(hoyder) * 0.99)]
    print(f"  Maks høyde: P99={p99_h:.1f} pt -> forslag: {p99_h * 1.5:.0f} pt")

    # Min bredde
    bredder = sorted(b["w"] for b in pos_accepted)
    p01_w = bredder[int(len(bredder) * 0.01)]
    print(f"  Min bredde: P01={p01_w:.1f} pt -> forslag: {p01_w * 0.7:.0f} pt")

    # Ratio
    ratioer = sorted(b["ratio"] for b in pos_accepted)
    p01_r = ratioer[int(len(ratioer) * 0.01)]
    p99_r = ratioer[int(len(ratioer) * 0.99)]
    print(f"  Ratio (w/h): P01={p01_r:.2f}  P99={p99_r:.2f}")
    print(f"    -> forslag min ratio: {p01_r * 0.7:.1f} (bredde skal alltid > høyde for FNR)")

    # Maks areal
    arealer = sorted(b["areal"] for b in pos_accepted)
    p99_a = arealer[int(len(arealer) * 0.99)]
    print(f"  Maks areal: P99={p99_a:.0f} pt² -> forslag: {p99_a * 1.5:.0f} pt²")


if __name__ == "__main__":
    main()

