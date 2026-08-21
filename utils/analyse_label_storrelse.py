"""
Analyserer dimensjonene til fasit-bokser fra labels-CSV for å forstå
typiske størrelser og finne gode terskler for filtrering.

Kjør:
    python utils/analyse_label_storrelse.py --csv smartsladding_uttrekk_labels_5_29_07_26.csv

Eller mot en resultat-CSV (pikselkoordinater):
    python utils/analyse_label_storrelse.py --res-csv utils/magnusruler.csv
"""

import argparse
import csv
import statistics

SKALA = 300 / 72.0  # PDF-punkt -> piksel ved 300 DPI


def les_labels(sti):
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
                           "areal": abs(w * h), "ratio": w / h if h != 0 else 0,
                           "kilde": "ml" if ml else "manuell"})
    return bokser


def les_resultat(sti):
    """Leser en resultat-CSV med pikselkoordinater (x0,y0,x1,y1,bilde_bredde,bilde_hoyde)."""
    bokser = []
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                x0, y0 = float(r["x0"]), float(r["y0"])
                x1, y1 = float(r["x1"]), float(r["y1"])
                kilde = r.get("kilde", "ukjent")
                # yolo_conf i nytt format, «conf» i gamle resultat-CSV-er
                raa_conf = r.get("yolo_conf") or r.get("conf")
                conf = float(raa_conf) if raa_conf else None
            except (TypeError, ValueError, KeyError):
                continue
            w_px = abs(x1 - x0)
            h_px = abs(y1 - y0)
            # Konverter til PDF-punkt
            w_pt = w_px / SKALA
            h_pt = h_px / SKALA
            bokser.append({"w": w_pt, "h": h_pt, "w_px": w_px, "h_px": h_px,
                           "areal": w_pt * h_pt, "kilde": kilde, "conf": conf,
                           "ratio": w_pt / h_pt if h_pt > 0 else 0,
                           "ml": True, "status": "PRED"})
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


def _analyser_gruppe(bokser, tittel):
    """Skriv persentiler for en gruppe bokser (allerede filtrert til positive)."""
    if not bokser:
        print(f"\n  (ingen bokser i '{tittel}')")
        return
    print(f"\n{'=' * 60}")
    print(f"{tittel} (n={len(bokser)}):")
    persentiler([b["w"] for b in bokser], "Bredde (pt)")
    persentiler([b["h"] for b in bokser], "Høyde (pt)")
    persentiler([b["areal"] for b in bokser], "Areal (pt²)")
    persentiler([b["ratio"] for b in bokser], "Ratio (w/h)")


def _analyser_per_kilde(bokser):
    """Skriv persentiler gruppert per kilde (paddle/yolo/begge)."""
    kilder = sorted(set(b["kilde"] for b in bokser))
    for kilde in kilder:
        gruppe = [b for b in bokser if b["kilde"] == kilde]
        _analyser_gruppe(gruppe, f"Kilde: {kilde}")


def main():
    p = argparse.ArgumentParser(description="Analysér dimensjoner i labels-CSV og/eller resultat-CSV")
    p.add_argument("--csv", default=None,
                   help="Labels-CSV (fasit, med width/height i PDF-punkt)")
    p.add_argument("--res-csv", default=None,
                   help="Resultat-CSV fra modellen (pikselkoordinater: x0,y0,x1,y1)")
    args = p.parse_args()

    if not args.csv and not args.res_csv:
        args.csv = "smartsladding_uttrekk_labels_5_29_07_26.csv"

    # ── Analysér labels-CSV (fasit) ──────────────────────────────
    if args.csv:
        alle = les_labels(args.csv)
        print(f"Totalt {len(alle)} bokser lest fra {args.csv}")

        accepted = [b for b in alle if b["ml"] and b["status"] == "ACCEPTED"]
        rejected = [b for b in alle if b["ml"] and b["status"] == "REJECTED"]
        manuell = [b for b in alle if not b["ml"]]

        neg = [b for b in alle if b["w"] <= 0 or b["h"] <= 0]
        print(f"\nBokser med negativ bredde eller høyde: {len(neg)}")
        if neg:
            for b in neg[:5]:
                print(f"  w={b['w']:.1f} h={b['h']:.1f} ml={b['ml']} status={b['status']}")
            if len(neg) > 5:
                print(f"  ... og {len(neg) - 5} til")

        pos = [b for b in alle if b["w"] > 0 and b["h"] > 0]
        pos_accepted = [b for b in accepted if b["w"] > 0 and b["h"] > 0]

        _analyser_gruppe(pos, "ALLE positive bokser")
        _analyser_gruppe(pos_accepted, "Kun ACCEPTED (ml_generated + godkjent)")

        # Piksler
        print(f"\n{'=' * 60}")
        print(f"ACCEPTED i piksler (300 DPI, skala={SKALA:.3f}):")
        persentiler([b["w"] * SKALA for b in pos_accepted], "Bredde (px)")
        persentiler([b["h"] * SKALA for b in pos_accepted], "Høyde (px)")
        persentiler([b["w"] * b["h"] * SKALA**2 for b in pos_accepted], "Areal (px²)")

        # Sammenlign med nåværende terskler
        min_areal_px = 965
        maks_bredde_pt = 50

        print(f"\n{'=' * 60}")
        print("Sammenligning med nåværende filtre:")
        print(f"  MIN_BOKS_AREAL = {min_areal_px} px²")
        under_min = sum(1 for b in pos_accepted if b["w"] * b["h"] * SKALA**2 < min_areal_px)
        print(f"    ACCEPTED-bokser som ville blitt filtrert bort: {under_min}/{len(pos_accepted)}")

        print(f"  MAKS_BREDDE_PT = {maks_bredde_pt} pt ({maks_bredde_pt * SKALA:.0f} px)")
        over_maks = sum(1 for b in pos_accepted if b["w"] > maks_bredde_pt)
        print(f"    ACCEPTED-bokser som ville blitt filtrert bort: {over_maks}/{len(pos_accepted)}")

        # Forslag til nye terskler
        print(f"\n{'=' * 60}")
        print("Mulige nye filtre (basert på fasit-data):")
        hoyder = sorted(b["h"] for b in pos_accepted)
        p99_h = hoyder[int(len(hoyder) * 0.99)]
        print(f"  Maks høyde: P99={p99_h:.1f} pt -> forslag: {p99_h * 1.5:.0f} pt")

        bredder = sorted(b["w"] for b in pos_accepted)
        p01_w = bredder[int(len(bredder) * 0.01)]
        print(f"  Min bredde: P01={p01_w:.1f} pt -> forslag: {p01_w * 0.7:.0f} pt")

        ratioer = sorted(b["ratio"] for b in pos_accepted)
        p01_r = ratioer[int(len(ratioer) * 0.01)]
        p99_r = ratioer[int(len(ratioer) * 0.99)]
        print(f"  Ratio (w/h): P01={p01_r:.2f}  P99={p99_r:.2f}")
        print(f"    -> forslag min ratio: {p01_r * 0.7:.1f} (bredde skal alltid > høyde for FNR)")

        arealer = sorted(b["areal"] for b in pos_accepted)
        p99_a = arealer[int(len(arealer) * 0.99)]
        print(f"  Maks areal: P99={p99_a:.0f} pt² -> forslag: {p99_a * 1.5:.0f} pt²")

    # ── Analysér resultat-CSV (prediksjoner) ─────────────────────
    if args.res_csv:
        pred = les_resultat(args.res_csv)
        print(f"\n\n{'#' * 60}")
        print(f"RESULTAT-CSV: {args.res_csv}")
        print(f"Totalt {len(pred)} predikerte bokser")

        pos_pred = [b for b in pred if b["w"] > 0 and b["h"] > 0]
        _analyser_gruppe(pos_pred, "Alle prediksjoner (pt)")

        # Per kilde
        _analyser_per_kilde(pos_pred)

        # Piksler
        print(f"\n{'=' * 60}")
        print("Prediksjoner i piksler:")
        persentiler([b["w_px"] for b in pos_pred], "Bredde (px)")
        persentiler([b["h_px"] for b in pos_pred], "Høyde (px)")
        persentiler([b["w_px"] * b["h_px"] for b in pos_pred], "Areal (px²)")

        # Sammenlign med nåværende terskler
        min_areal_px = 965
        maks_bredde_pt = 50
        print(f"\n{'=' * 60}")
        print("Nåværende filtre mot prediksjoner:")
        print(f"  MIN_BOKS_AREAL = {min_areal_px} px²")
        under_min = sum(1 for b in pos_pred if b["w_px"] * b["h_px"] < min_areal_px)
        print(f"    Prediksjoner som VILLE blitt filtrert: {under_min}/{len(pos_pred)}")

        print(f"  MAKS_BREDDE_PT = {maks_bredde_pt} pt (kun elektronisk)")
        over_maks = sum(1 for b in pos_pred if b["w"] > maks_bredde_pt)
        print(f"    Prediksjoner bredere enn {maks_bredde_pt} pt: {over_maks}/{len(pos_pred)}")

        # Vis outliers
        print(f"\n{'=' * 60}")
        print("Potensielle outliers i prediksjoner:")
        outliers = [b for b in pos_pred if b["ratio"] < 1.2 or b["areal"] > 3300 or b["h"] > 50]
        print(f"  Bokser med ratio < 1.2 ELLER areal > 3300 pt² ELLER høyde > 50 pt: {len(outliers)}/{len(pos_pred)}")
        for b in sorted(outliers, key=lambda x: -x["areal"])[:15]:
            print(f"    w={b['w']:.1f}pt h={b['h']:.1f}pt areal={b['areal']:.0f}pt² "
                  f"ratio={b['ratio']:.2f} kilde={b['kilde']} conf={b['conf']}")

        # Sammenlign med fasit hvis begge er gitt
        if args.csv:
            print(f"\n{'=' * 60}")
            print("SAMMENLIGNING fasit vs. prediksjoner (median):")
            print(f"  {'':20} {'Fasit (ACCEPTED)':>20} {'Prediksjoner':>20}")
            if pos_accepted and pos_pred:
                fw = statistics.median(b["w"] for b in pos_accepted)
                pw = statistics.median(b["w"] for b in pos_pred)
                print(f"  {'Bredde (pt)':<20} {fw:>20.1f} {pw:>20.1f}")
                fh = statistics.median(b["h"] for b in pos_accepted)
                ph = statistics.median(b["h"] for b in pos_pred)
                print(f"  {'Høyde (pt)':<20} {fh:>20.1f} {ph:>20.1f}")
                fa = statistics.median(b["areal"] for b in pos_accepted)
                pa = statistics.median(b["areal"] for b in pos_pred)
                print(f"  {'Areal (pt²)':<20} {fa:>20.0f} {pa:>20.0f}")
                fr = statistics.median(b["ratio"] for b in pos_accepted)
                pr = statistics.median(b["ratio"] for b in pos_pred)
                print(f"  {'Ratio (w/h)':<20} {fr:>20.2f} {pr:>20.2f}")


if __name__ == "__main__":
    main()



