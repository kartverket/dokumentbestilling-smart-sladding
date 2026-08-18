"""
Systematisk evaluering av filterkonfigurasjoner for boks-dimensjoner.

Krysssjekker fasit-labels (ACCEPTED = korrekt, REJECTED = feil) og
modellprediksjoner for å finne optimale terskler.

Kjør:
    python utils/filter_sweep.py \
        --csv /path/to/labels.csv \
        --res-csv /path/to/resultat.csv
"""

import argparse
import csv
from itertools import product

SKALA = 300 / 72.0


# ── Datainnlesing ────────────────────────────────────────────

def les_labels(sti):
    bokser = []
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                w = abs(float(r["width"]))
                h = abs(float(r["height"]))
                ml = (r.get("ml_generated") or "").strip().lower() == "true"
                status = (r.get("ml_status") or "").strip().upper()
            except (TypeError, ValueError, KeyError):
                continue
            if w <= 0 or h <= 0:
                continue
            # Klassifisér: korrekt = ACCEPTED eller manuelt lagt til,
            #              feil    = REJECTED (ML foreslo, menneske avviste)
            if ml and status == "REJECTED":
                kategori = "feil"
            else:
                kategori = "korrekt"
            bokser.append({"w": w, "h": h, "ratio": w / h,
                           "areal": w * h, "kategori": kategori})
    return bokser


def les_resultat(sti):
    bokser = []
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                x0, y0 = float(r["x0"]), float(r["y0"])
                x1, y1 = float(r["x1"]), float(r["y1"])
                kilde = r.get("kilde", "ukjent")
            except (TypeError, ValueError, KeyError):
                continue
            w_px, h_px = abs(x1 - x0), abs(y1 - y0)
            w_pt, h_pt = w_px / SKALA, h_px / SKALA
            if w_pt <= 0 or h_pt <= 0:
                continue
            bokser.append({"w": w_pt, "h": h_pt, "ratio": w_pt / h_pt,
                           "areal": w_pt * h_pt, "kilde": kilde})
    return bokser


# ── Filtrering ───────────────────────────────────────────────

def _filtrert(boks, min_ratio, maks_hoyde, maks_bredde, maks_areal):
    """Returnerer True hvis boksen FJERNES av filteret."""
    if min_ratio is not None and boks["ratio"] < min_ratio:
        return True
    if maks_hoyde is not None and boks["h"] > maks_hoyde:
        return True
    if maks_bredde is not None and boks["w"] > maks_bredde:
        return True
    if maks_areal is not None and boks["areal"] > maks_areal:
        return True
    return False


def _evaluer(bokser, min_ratio, maks_hoyde, maks_bredde, maks_areal):
    fjernet = [b for b in bokser if _filtrert(b, min_ratio, maks_hoyde, maks_bredde, maks_areal)]
    return len(fjernet)


# ── Enkeltparameter-sweeps ───────────────────────────────────

def _sweep_en_param(fasit, pred, navn, verdier, filter_fn):
    """Sweep én parameter, hold resten på None (ingen filtrering)."""
    korrekt = [b for b in fasit if b["kategori"] == "korrekt"]
    feil = [b for b in fasit if b["kategori"] == "feil"]
    pred_paddle = [b for b in pred if b["kilde"] == "paddle"]
    pred_yolo = [b for b in pred if b["kilde"] == "yolo"]
    pred_begge = [b for b in pred if b["kilde"] == "begge"]

    print(f"\n{'─' * 90}")
    print(f"Sweep: {navn}")
    print(f"{'─' * 90}")
    print(f"  {'Verdi':>10} │ {'Korrekt':>10} {'mistet':>8} │ {'Feil':>10} {'fanget':>8} │"
          f" {'Pred':>7} {'paddle':>8} {'yolo':>8} {'begge':>8}")
    print(f"  {'':>10} │ {'':>10} {'':>8} │ {'':>10} {'':>8} │"
          f" {'fjernet':>7} {'fjernet':>8} {'fjernet':>8} {'fjernet':>8}")
    print(f"  {'─' * 10}─┼─{'─' * 19}─┼─{'─' * 19}─┼─{'─' * 35}")

    for v in verdier:
        kwargs = filter_fn(v)
        n_korrekt_mistet = _evaluer(korrekt, **kwargs)
        n_feil_fanget = _evaluer(feil, **kwargs)
        n_pred = _evaluer(pred, **kwargs)
        n_paddle = _evaluer(pred_paddle, **kwargs)
        n_yolo = _evaluer(pred_yolo, **kwargs)
        n_begge = _evaluer(pred_begge, **kwargs)

        v_str = f"{v:g}" if v is not None else "av"
        k_pct = f"({n_korrekt_mistet / len(korrekt) * 100:.2f}%)" if korrekt else ""
        f_pct = f"({n_feil_fanget / len(feil) * 100:.1f}%)" if feil else ""

        print(f"  {v_str:>10} │ {len(korrekt):>6}{n_korrekt_mistet:>5} {k_pct:>8} │"
              f" {len(feil):>6}{n_feil_fanget:>5} {f_pct:>8} │"
              f" {n_pred:>7} {n_paddle:>8} {n_yolo:>8} {n_begge:>8}")


# ── Kombinasjons-sweep ───────────────────────────────────────

def _sweep_kombinasjoner(fasit, pred, ratio_verdier, hoyde_verdier, bredde_verdier):
    korrekt = [b for b in fasit if b["kategori"] == "korrekt"]
    feil = [b for b in fasit if b["kategori"] == "feil"]

    print(f"\n{'═' * 100}")
    print("KOMBINASJONS-SWEEP (min_ratio × maks_hoyde × maks_bredde)")
    print(f"{'═' * 100}")
    print(f"  {'ratio':>6} {'hoyde':>7} {'bredde':>8} │"
          f" {'korr.mistet':>12} {'%':>7} │ {'feil fanget':>12} {'%':>7} │"
          f" {'pred fjernet':>13} {'netto':>7}")
    print(f"  {'─' * 23}─┼─{'─' * 20}─┼─{'─' * 20}─┼─{'─' * 21}")

    beste = []
    for min_r, maks_h, maks_b in product(ratio_verdier, hoyde_verdier, bredde_verdier):
        n_km = _evaluer(korrekt, min_r, maks_h, maks_b, None)
        n_ff = _evaluer(feil, min_r, maks_h, maks_b, None)
        n_pred = _evaluer(pred, min_r, maks_h, maks_b, None)

        km_pct = n_km / len(korrekt) * 100 if korrekt else 0
        ff_pct = n_ff / len(feil) * 100 if feil else 0
        # netto = feil fanget minus korrekte mistet (høyere er bedre)
        netto = n_ff - n_km

        r_str = f"{min_r:g}" if min_r is not None else "av"
        h_str = f"{maks_h:g}" if maks_h is not None else "av"
        b_str = f"{maks_b:g}" if maks_b is not None else "av"

        beste.append((netto, km_pct, ff_pct, n_km, n_ff, n_pred, r_str, h_str, b_str))

    # Sortér: høyest netto først, ved likt → lavest korrekte mistet
    beste.sort(key=lambda x: (-x[0], x[1]))

    for netto, km_pct, ff_pct, n_km, n_ff, n_pred, r_str, h_str, b_str in beste:
        markør = " ◀" if km_pct == 0 and netto > 0 else ""
        print(f"  {r_str:>6} {h_str:>7} {b_str:>8} │"
              f" {n_km:>12} {km_pct:>6.2f}% │ {n_ff:>12} {ff_pct:>6.1f}% │"
              f" {n_pred:>13} {netto:>+7d}{markør}")


def main():
    p = argparse.ArgumentParser(
        description="Sweep filterkonfigurasjoner mot fasit + prediksjoner")
    p.add_argument("--csv", required=True,
                   help="Labels-CSV (fasit)")
    p.add_argument("--res-csv", required=True,
                   help="Resultat-CSV fra modellen")
    args = p.parse_args()

    fasit = les_labels(args.csv)
    pred = les_resultat(args.res_csv)

    n_korrekt = sum(1 for b in fasit if b["kategori"] == "korrekt")
    n_feil = sum(1 for b in fasit if b["kategori"] == "feil")
    print(f"Fasit:       {len(fasit)} bokser ({n_korrekt} korrekte, {n_feil} feil/REJECTED)")
    print(f"Prediksjoner: {len(pred)} bokser")

    pred_kilder = {}
    for b in pred:
        pred_kilder[b["kilde"]] = pred_kilder.get(b["kilde"], 0) + 1
    for k, n in sorted(pred_kilder.items()):
        print(f"  {k}: {n}")

    # ── Enkeltparameter-sweeps ──
    _sweep_en_param(fasit, pred, "MIN_BOKS_RATIO (w/h)",
                    [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0],
                    lambda v: {"min_ratio": v, "maks_hoyde": None,
                               "maks_bredde": None, "maks_areal": None})

    _sweep_en_param(fasit, pred, "MAKS_BOKS_HOYDE_PT",
                    [25, 30, 35, 40, 45, 50, 60, 80, 100],
                    lambda v: {"min_ratio": None, "maks_hoyde": v,
                               "maks_bredde": None, "maks_areal": None})

    _sweep_en_param(fasit, pred, "MAKS_BREDDE_PT (alle dok, ikke bare e-tinglyst)",
                    [50, 60, 70, 80, 90, 100, 120, 150],
                    lambda v: {"min_ratio": None, "maks_hoyde": None,
                               "maks_bredde": v, "maks_areal": None})

    _sweep_en_param(fasit, pred, "MAKS_AREAL_PT²",
                    [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000],
                    lambda v: {"min_ratio": None, "maks_hoyde": None,
                               "maks_bredde": None, "maks_areal": v})

    # ── Kombinasjons-sweep ──
    ratio_verdier = [None, 0.8, 1.0, 1.2]
    hoyde_verdier = [None, 40, 50, 60]
    bredde_verdier = [None, 80, 100, 120]

    _sweep_kombinasjoner(fasit, pred, ratio_verdier, hoyde_verdier, bredde_verdier)


if __name__ == "__main__":
    main()

