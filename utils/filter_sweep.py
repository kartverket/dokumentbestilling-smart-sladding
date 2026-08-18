"""
Overlapp-basert evaluering av filterkonfigurasjoner.

Matcher prediksjoner mot fasit via overlapp for å klassifisere hver
prediksjon som RIKTIG (treffer fasit) eller OVERSLADD (ingen treff).
Sweeper deretter filterkonfigurasjoner og måler:
  - Riktige fjernet   = recall-tap
  - Oversladdinger fjernet = oversladding-reduksjon

Kjør:
    python utils/filter_sweep.py \
        --fasit-csv /path/to/labels.csv \
        --res-csv /path/to/resultat.csv \
        --terskel 0.15
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from itertools import product

SKALA = 300 / 72.0   # PDF-punkt → piksel ved 300 DPI


# ── Hjelpefunksjoner ─────────────────────────────────────────

def _dok_nr(navn):
    m = re.match(r"0*(\d+)", os.path.basename(navn))
    return int(m.group(1)) if m else None


def _overlap(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return (ix1 - ix0) * (iy1 - iy0) if (ix1 > ix0 and iy1 > iy0) else 0.0


def _areal(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


# ── Datainnlesing ────────────────────────────────────────────

def les_fasit(sti):
    """Leser fasit-labels (ACCEPTED + manuell, ekskluderer REJECTED).
    Returnerer dict: (dok_nr, side) -> [(norm_x0, norm_y0, norm_x1, norm_y1), ...]
    Normalisering gjøres ved matching-tid fordi vi trenger sidestørrelse."""
    fasit = defaultdict(list)
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("ml_status") or "").strip().upper() == "REJECTED":
                continue
            try:
                nr = int(r["fil_revisjon_id"])
                side = int(r["sidetall"])
                x, y = float(r["x"]), float(r["y"])
                w, h = float(r["width"]), float(r["height"])
            except (TypeError, ValueError, KeyError):
                continue
            # Sortér koordinater (håndterer negative w/h)
            x0, x1 = sorted((x, x + w))
            y0, y1 = sorted((y, y + h))
            fasit[(nr, side)].append((x0, y0, x1, y1))
    return fasit


def les_prediksjoner(sti):
    """Leser resultat-CSV med pikselkoordinater.
    Returnerer liste av dicts med normaliserte koordinater og dimensjoner i pt."""
    pred = []
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                navn = r["navn"]
                side = int(r["side"])
                bw, bh = int(r["bilde_bredde"]), int(r["bilde_hoyde"])
                x0, y0 = float(r["x0"]), float(r["y0"])
                x1, y1 = float(r["x1"]), float(r["y1"])
                kilde = r.get("kilde", "ukjent")
                conf_s = r.get("conf", "")
                conf = float(conf_s) if conf_s else None
            except (TypeError, ValueError, KeyError):
                continue
            # Normalisér til [0,1] for overlapp-matching
            norm = (x0 / bw, y0 / bh, x1 / bw, y1 / bh)
            # Dimensjoner i PDF-punkt for filtrering
            w_pt = abs(x1 - x0) / SKALA
            h_pt = abs(y1 - y0) / SKALA
            if w_pt <= 0 or h_pt <= 0:
                continue
            pred.append({
                "navn": navn, "side": side, "dok_nr": _dok_nr(navn),
                "bw": bw, "bh": bh,
                "norm": norm,
                "w": w_pt, "h": h_pt,
                "ratio": w_pt / h_pt if h_pt > 0 else 0,
                "areal": w_pt * h_pt,
                "kilde": kilde, "conf": conf,
            })
    return pred


# ── Matching ─────────────────────────────────────────────────

def match_prediksjoner(pred_liste, fasit, terskel=0.15):
    """Matcher prediksjoner mot fasit via normalisert overlapp.
    Legger til 'riktig'-nøkkel på hver prediksjon."""

    # Normaliser fasit til [0,1] med estimert sidestørrelse
    # Grupper prediksjoner for å finne sidestørrelse per (dok_nr, side)
    side_str = {}
    for p in pred_liste:
        key = (p["dok_nr"], p["side"])
        if key not in side_str:
            # Estimér sidestørrelse i punkt fra piksel / skala
            side_str[key] = (p["bw"] / SKALA, p["bh"] / SKALA)

    # Forhåndsnormaliser fasit
    fasit_norm = {}
    for (nr, si), bokser in fasit.items():
        pw, ph = side_str.get((nr, si), (595, 842))  # fallback A4
        fasit_norm[(nr, si)] = [(x0/pw, y0/ph, x1/pw, y1/ph) for (x0,y0,x1,y1) in bokser]

    # Match
    n_riktig = n_oversladd = n_uten_fasit = 0
    for p in pred_liste:
        key = (p["dok_nr"], p["side"])
        fbokser = fasit_norm.get(key, [])
        if not fbokser:
            # Ingen fasit for denne siden — kan ikke vurdere
            p["riktig"] = None
            n_uten_fasit += 1
            continue

        pn = p["norm"]
        best_dek = 0.0
        for fb in fbokser:
            ov = _overlap(pn, fb)
            fa = _areal(fb)
            dek = ov / fa if fa > 0 else 0.0
            if dek > best_dek:
                best_dek = dek

        p["riktig"] = best_dek >= terskel
        if p["riktig"]:
            n_riktig += 1
        else:
            n_oversladd += 1

    # Sider uten fasit = ingen FNR der, alt er oversladding
    for p in pred_liste:
        if p["riktig"] is None:
            p["riktig"] = False
            n_oversladd += 1
            n_uten_fasit += 1

    return n_riktig, n_oversladd, n_uten_fasit


# ── Filtrering ───────────────────────────────────────────────

def _filtrert(p, min_ratio, maks_hoyde, maks_bredde, maks_areal):
    if min_ratio is not None and p["ratio"] < min_ratio:
        return True
    if maks_hoyde is not None and p["h"] > maks_hoyde:
        return True
    if maks_bredde is not None and p["w"] > maks_bredde:
        return True
    if maks_areal is not None and p["areal"] > maks_areal:
        return True
    return False


def _tell_filtrerte(bokser, **kwargs):
    return sum(1 for b in bokser if _filtrert(b, **kwargs))


# ── Sweeps ───────────────────────────────────────────────────

def _sweep_en_param(riktige, oversladdinger, navn, verdier, filter_fn):
    print(f"\n{'─' * 95}")
    print(f"Sweep: {navn}")
    print(f"{'─' * 95}")
    print(f"  {'Verdi':>8} │ {'Riktige':>8} {'fjernet':>8} {'%':>7} │"
          f" {'Oversladd':>9} {'fjernet':>8} {'%':>7} │"
          f" {'Netto':>7} {'Presisjon etter':>16}")
    print(f"  {'─' * 8}─┼─{'─' * 24}─┼─{'─' * 25}─┼─{'─' * 24}")

    for v in verdier:
        kwargs = filter_fn(v)
        n_rik_fj = _tell_filtrerte(riktige, **kwargs)
        n_ov_fj = _tell_filtrerte(oversladdinger, **kwargs)

        r_pct = n_rik_fj / len(riktige) * 100 if riktige else 0
        o_pct = n_ov_fj / len(oversladdinger) * 100 if oversladdinger else 0
        netto = n_ov_fj - n_rik_fj

        # Presisjon etter filtrering
        rik_etter = len(riktige) - n_rik_fj
        ov_etter = len(oversladdinger) - n_ov_fj
        totalt_etter = rik_etter + ov_etter
        pres_etter = rik_etter / totalt_etter * 100 if totalt_etter > 0 else 0

        v_str = f"{v:g}" if v is not None else "av"
        print(f"  {v_str:>8} │ {len(riktige):>5} {n_rik_fj:>7} {r_pct:>6.2f}% │"
              f" {len(oversladdinger):>6} {n_ov_fj:>7} {o_pct:>6.1f}% │"
              f" {netto:>+7d} {pres_etter:>14.1f}%")


def _sweep_kombinasjoner(riktige, oversladdinger, ratio_v, hoyde_v, bredde_v):
    # Utgangs-presisjon
    totalt_foer = len(riktige) + len(oversladdinger)
    pres_foer = len(riktige) / totalt_foer * 100 if totalt_foer else 0

    print(f"\n{'═' * 110}")
    print(f"KOMBINASJONS-SWEEP  (utgangspunkt: {len(riktige)} riktige + "
          f"{len(oversladdinger)} oversladd = {totalt_foer} pred, presisjon {pres_foer:.1f}%)")
    print(f"{'═' * 110}")
    print(f"  {'ratio':>6} {'hoyde':>6} {'bredde':>7} │"
          f" {'rik.fj':>7} {'%':>6} │ {'ov.fj':>7} {'%':>6} │"
          f" {'netto':>7} {'rik etter':>10} {'ov etter':>9} {'pres%':>7}")
    print(f"  {'─' * 21}─┼─{'─' * 14}─┼─{'─' * 14}─┼─{'─' * 35}")

    rader = []
    for min_r, maks_h, maks_b in product(ratio_v, hoyde_v, bredde_v):
        n_rk = _tell_filtrerte(riktige, min_ratio=min_r, maks_hoyde=maks_h,
                               maks_bredde=maks_b, maks_areal=None)
        n_ov = _tell_filtrerte(oversladdinger, min_ratio=min_r, maks_hoyde=maks_h,
                               maks_bredde=maks_b, maks_areal=None)

        rk_pct = n_rk / len(riktige) * 100 if riktige else 0
        ov_pct = n_ov / len(oversladdinger) * 100 if oversladdinger else 0
        netto = n_ov - n_rk

        rik_etter = len(riktige) - n_rk
        ov_etter = len(oversladdinger) - n_ov
        totalt_etter = rik_etter + ov_etter
        pres_etter = rik_etter / totalt_etter * 100 if totalt_etter > 0 else 0

        r_str = f"{min_r:g}" if min_r is not None else "av"
        h_str = f"{maks_h:g}" if maks_h is not None else "av"
        b_str = f"{maks_b:g}" if maks_b is not None else "av"

        rader.append((netto, rk_pct, ov_pct, n_rk, n_ov, rik_etter, ov_etter,
                      pres_etter, r_str, h_str, b_str))

    rader.sort(key=lambda x: (-x[0], x[1]))

    for (netto, rk_pct, ov_pct, n_rk, n_ov, rik_etter, ov_etter,
         pres_etter, r_str, h_str, b_str) in rader:
        markør = " ◀" if rk_pct == 0 and netto > 0 else ""
        print(f"  {r_str:>6} {h_str:>6} {b_str:>7} │"
              f" {n_rk:>7} {rk_pct:>5.2f}% │ {n_ov:>7} {ov_pct:>5.1f}% │"
              f" {netto:>+7d} {rik_etter:>10} {ov_etter:>9} {pres_etter:>6.1f}%{markør}")


# ── Hovedprogram ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Overlapp-basert evaluering av filterkonfigurasjoner")
    p.add_argument("--fasit-csv", required=True,
                   help="Labels-CSV (ACCEPTED + manuell = fasit, REJECTED ekskluderes)")
    p.add_argument("--res-csv", required=True,
                   help="Resultat-CSV fra modellen (pikselkoordinater)")
    p.add_argument("--terskel", type=float, default=0.15,
                   help="Overlapp-terskel for å klassifisere prediksjon som riktig (default: 0.15)")
    # Bakoverkompatibilitet
    p.add_argument("--csv", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.csv and not args.fasit_csv:
        args.fasit_csv = args.csv

    # Last data
    fasit = les_fasit(args.fasit_csv)
    pred = les_prediksjoner(args.res_csv)

    n_fasit_bokser = sum(len(v) for v in fasit.values())
    print(f"Fasit:        {n_fasit_bokser} bokser i {len(fasit)} (dok, side)-grupper")
    print(f"Prediksjoner: {len(pred)} bokser")
    pred_kilder = {}
    for b in pred:
        pred_kilder[b["kilde"]] = pred_kilder.get(b["kilde"], 0) + 1
    for k, n in sorted(pred_kilder.items()):
        print(f"  {k}: {n}")

    # Match prediksjoner mot fasit
    print(f"\nMatcher med overlapp-terskel {args.terskel:.0%} ...")
    n_riktig, n_oversladd, n_uten = match_prediksjoner(pred, fasit, args.terskel)

    # Splitt i grupper
    riktige = [p for p in pred if p.get("riktig") is True]
    oversladdinger = [p for p in pred if p.get("riktig") is False]

    totalt = len(riktige) + len(oversladdinger)
    pres = len(riktige) / totalt * 100 if totalt else 0
    print(f"\nResultat:")
    print(f"  Riktige prediksjoner (treffer fasit):   {len(riktige)}")
    print(f"  Oversladdinger (ingen fasit-treff):     {len(oversladdinger)}")
    print(f"    herav på sider uten fasit:            {n_uten}")
    print(f"  Presisjon (riktige / totalt):           {pres:.1f}%")

    # Vis oversladdinger per kilde
    print(f"\n  Oversladdinger per kilde:")
    for kilde in sorted(pred_kilder):
        n_ov_k = sum(1 for p in oversladdinger if p["kilde"] == kilde)
        n_tot_k = pred_kilder[kilde]
        print(f"    {kilde:>8}: {n_ov_k:>5} / {n_tot_k:>5} ({n_ov_k/n_tot_k*100:.1f}%)")

    # ── Enkeltparameter-sweeps ──
    _sweep_en_param(riktige, oversladdinger,
                    "MIN_BOKS_RATIO (w/h)",
                    [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0],
                    lambda v: {"min_ratio": v, "maks_hoyde": None,
                               "maks_bredde": None, "maks_areal": None})

    _sweep_en_param(riktige, oversladdinger,
                    "MAKS_BOKS_HOYDE_PT",
                    [25, 30, 35, 40, 45, 50, 60, 80, 100],
                    lambda v: {"min_ratio": None, "maks_hoyde": v,
                               "maks_bredde": None, "maks_areal": None})

    _sweep_en_param(riktige, oversladdinger,
                    "MAKS_BOKS_BREDDE_PT (universell)",
                    [60, 80, 100, 120, 150, 200, 250],
                    lambda v: {"min_ratio": None, "maks_hoyde": None,
                               "maks_bredde": v, "maks_areal": None})

    _sweep_en_param(riktige, oversladdinger,
                    "MAKS_AREAL_PT²",
                    [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000],
                    lambda v: {"min_ratio": None, "maks_hoyde": None,
                               "maks_bredde": None, "maks_areal": v})

    # ── Kombinasjons-sweep ──
    ratio_verdier = [None, 0.8, 1.0, 1.2]
    hoyde_verdier = [None, 40, 50, 60]
    bredde_verdier = [None, 80, 100, 120]

    _sweep_kombinasjoner(riktige, oversladdinger,
                         ratio_verdier, hoyde_verdier, bredde_verdier)


if __name__ == "__main__":
    main()

