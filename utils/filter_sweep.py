"""
Overlapp-basert evaluering av filterkonfigurasjoner.

Matcher prediksjoner mot fasit via overlapp for å klassifisere hver
prediksjon som RIKTIG (treffer fasit) eller OVERSLADD (ingen treff).
Sweeper deretter filterkonfigurasjoner og måler:
  - Riktige fjernet   = recall-tap
  - Oversladdinger fjernet = oversladding-reduksjon

Filtrering baseres på geometriske egenskaper (elongation, høyde, bredde, areal)
og kan gated av YOLO-confidence: prediksjoner med conf ≥ conf_terskel beholdes
uansett geometri (vi stoler på høy-confidence-deteksjoner).

Kjør:
    python utils/filter_sweep.py \\
        --fasit-csv /path/to/labels.csv \\
        --res-csv /path/to/resultat.csv \\
        --terskel 0.15 \\
        --sort ov/rik \\
        --min-ov-rik 2.0
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
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
            ratio = w_pt / h_pt if h_pt > 0 else 0
            pred.append({
                "navn": navn, "side": side, "dok_nr": _dok_nr(navn),
                "bw": bw, "bh": bh,
                "norm": norm,
                "w": w_pt, "h": h_pt,
                "ratio": ratio,
                "elongation": max(ratio, 1/ratio) if ratio > 0 else 0,
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

    return n_riktig, n_oversladd, n_uten_fasit


# ── Filtrering ───────────────────────────────────────────────

def _filtrert(p, min_ratio, maks_hoyde, maks_bredde, maks_areal,
              min_elongation=None, conf_terskel=None):
    # Høy confidence → stol på prediksjonen, ikke filtrer
    if conf_terskel is not None and p["conf"] is not None and p["conf"] >= conf_terskel:
        return False
    if min_ratio is not None and p["ratio"] < min_ratio:
        return True
    if min_elongation is not None and p["elongation"] < min_elongation:
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


# ── Sortering ────────────────────────────────────────────────

SORT_FNS = {
    "netto": lambda x: (-x[0], x[1]),
    "ov.fj": lambda x: (-x[4], x[1]),
    "rik.fj": lambda x: (x[3], -x[4]),
    "pres": lambda x: (-x[7], -x[0]),
    "ov/rik": lambda x: (-(x[4] / x[3] if x[3] > 0 else float('inf')), x[1]),
}


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


def _sweep_kombinasjoner(riktige, oversladdinger, elong_v, hoyde_v, bredde_v,
                         conf_v=None, sort_key="netto", tittel=None,
                         min_ov_rik=None, maks_rik_pst=None):
    """Sweep alle kombinasjoner av elongation/høyde/bredde (og evt. conf_terskel)."""
    totalt_foer = len(riktige) + len(oversladdinger)
    pres_foer = len(riktige) / totalt_foer * 100 if totalt_foer else 0

    if conf_v is None:
        conf_v = [None]

    overskrift = tittel or "KOMBINASJONS-SWEEP"
    filter_info = ""
    if min_ov_rik:
        filter_info += f"  [filter: ov/rik > {min_ov_rik:g}]"
    if maks_rik_pst is not None:
        filter_info += f"  [filter: rik.fj ≤ {maks_rik_pst:g}%]"
    print(f"\n{'═' * 130}")
    print(f"{overskrift}  (utgangspunkt: {len(riktige)} riktige + "
          f"{len(oversladdinger)} oversladd = {totalt_foer} pred, presisjon {pres_foer:.1f}%)"
          f"  [sortert etter: {sort_key}]{filter_info}")
    print(f"{'═' * 130}")
    har_conf = any(c is not None for c in conf_v)
    if har_conf:
        print(f"  {'elong':>6} {'hoyde':>6} {'bredde':>7} {'conf≥':>6} │"
              f" {'rik.fj':>7} {'%':>6} │ {'ov.fj':>7} {'%':>6} │"
              f" {'netto':>7} {'ov/rik':>7} {'rik etter':>10} {'ov etter':>9} {'pres%':>7}")
        print(f"  {'─' * 28}─┼─{'─' * 14}─┼─{'─' * 14}─┼─{'─' * 42}")
    else:
        print(f"  {'elong':>6} {'hoyde':>6} {'bredde':>7} │"
              f" {'rik.fj':>7} {'%':>6} │ {'ov.fj':>7} {'%':>6} │"
              f" {'netto':>7} {'ov/rik':>7} {'rik etter':>10} {'ov etter':>9} {'pres%':>7}")
        print(f"  {'─' * 21}─┼─{'─' * 14}─┼─{'─' * 14}─┼─{'─' * 42}")

    rader = []
    for min_e, maks_h, maks_b, c_t in product(elong_v, hoyde_v, bredde_v, conf_v):
        n_rk = _tell_filtrerte(riktige, min_ratio=None, maks_hoyde=maks_h,
                               maks_bredde=maks_b, maks_areal=None,
                               min_elongation=min_e, conf_terskel=c_t)
        n_ov = _tell_filtrerte(oversladdinger, min_ratio=None, maks_hoyde=maks_h,
                               maks_bredde=maks_b, maks_areal=None,
                               min_elongation=min_e, conf_terskel=c_t)

        rk_pct = n_rk / len(riktige) * 100 if riktige else 0
        ov_pct = n_ov / len(oversladdinger) * 100 if oversladdinger else 0
        netto = n_ov - n_rk

        rik_etter = len(riktige) - n_rk
        ov_etter = len(oversladdinger) - n_ov
        totalt_etter = rik_etter + ov_etter
        pres_etter = rik_etter / totalt_etter * 100 if totalt_etter > 0 else 0

        e_str = f"{min_e:g}" if min_e is not None else "av"
        h_str = f"{maks_h:g}" if maks_h is not None else "av"
        b_str = f"{maks_b:g}" if maks_b is not None else "av"
        c_str = f"{c_t:g}" if c_t is not None else "av"

        rader.append((netto, rk_pct, ov_pct, n_rk, n_ov, rik_etter, ov_etter,
                      pres_etter, e_str, h_str, b_str, c_str))

    rader.sort(key=SORT_FNS.get(sort_key, SORT_FNS["netto"]))

    n_skjult = 0
    for (netto, rk_pct, ov_pct, n_rk, n_ov, rik_etter, ov_etter,
         pres_etter, e_str, h_str, b_str, c_str) in rader:
        # Filtrer på ov/rik-ratio
        if min_ov_rik is not None:
            ov_rik_val = (n_ov / n_rk) if n_rk > 0 else float('inf') if n_ov > 0 else 0
            if ov_rik_val <= min_ov_rik:
                n_skjult += 1
                continue
        # Filtrer på maks % riktige fjernet
        if maks_rik_pst is not None and rk_pct > maks_rik_pst:
            n_skjult += 1
            continue
        markør = " ◀" if rk_pct == 0 and netto > 0 else ""
        ratio_str = f"{n_ov / n_rk:.1f}" if n_rk > 0 else "∞" if n_ov > 0 else "–"
        if har_conf:
            print(f"  {e_str:>6} {h_str:>6} {b_str:>7} {c_str:>6} │"
                  f" {n_rk:>7} {rk_pct:>5.2f}% │ {n_ov:>7} {ov_pct:>5.1f}% │"
                  f" {netto:>+7d} {ratio_str:>7} {rik_etter:>10} {ov_etter:>9} {pres_etter:>6.1f}%{markør}")
        else:
            print(f"  {e_str:>6} {h_str:>6} {b_str:>7} │"
                  f" {n_rk:>7} {rk_pct:>5.2f}% │ {n_ov:>7} {ov_pct:>5.1f}% │"
                  f" {netto:>+7d} {ratio_str:>7} {rik_etter:>10} {ov_etter:>9} {pres_etter:>6.1f}%{markør}")

    if n_skjult:
        filters = []
        if min_ov_rik is not None:
            filters.append(f"ov/rik ≤ {min_ov_rik:g}")
        if maks_rik_pst is not None:
            filters.append(f"rik.fj > {maks_rik_pst:g}%")
        print(f"  ({n_skjult} rader skjult: {' eller '.join(filters)})")


def _sweep_kryss_kilder(riktige, oversladdinger, kilder, elong_v, hoyde_v, bredde_v,
                        conf_v=None, sort_key="netto", min_ov_rik=None,
                        maks_rik_pst=None):
    """Sweep med uavhengige filterparametre per kilde.

    Finner topp-kandidater per kilde, deretter kombinerer på tvers
    for å finne optimal konfigurasjon med ulike filtre per kilde."""

    if conf_v is None:
        conf_v = [None]

    # Bygg resultater per kilde (kun conf-sweep for kilder med conf-data)
    per_kilde_res = {}
    for kilde in kilder:
        rik_k = [p for p in riktige if p["kilde"] == kilde]
        ov_k = [p for p in oversladdinger if p["kilde"] == kilde]
        if not rik_k and not ov_k:
            continue
        kilde_har_conf = any(p["conf"] is not None for p in rik_k + ov_k)
        kilde_conf = conf_v if kilde_har_conf else [None]
        resultater = []
        for min_e, maks_h, maks_b, c_t in product(elong_v, hoyde_v, bredde_v, kilde_conf):
            n_rk = _tell_filtrerte(rik_k, min_ratio=None, maks_hoyde=maks_h,
                                   maks_bredde=maks_b, maks_areal=None,
                                   min_elongation=min_e, conf_terskel=c_t)
            n_ov = _tell_filtrerte(ov_k, min_ratio=None, maks_hoyde=maks_h,
                                   maks_bredde=maks_b, maks_areal=None,
                                   min_elongation=min_e, conf_terskel=c_t)
            resultater.append((min_e, maks_h, maks_b, c_t, n_rk, n_ov))
        # Sortér: maks netto (ov-rik) med minimalt riktige-tap
        resultater.sort(key=lambda x: (-(x[5] - x[4]), x[4]))
        per_kilde_res[kilde] = resultater[:8]

    if len(per_kilde_res) < 2:
        return

    kilde_liste = sorted(per_kilde_res.keys())
    kandidat_lister = [per_kilde_res[k] for k in kilde_liste]

    totalt_foer = len(riktige) + len(oversladdinger)
    pres_foer = len(riktige) / totalt_foer * 100 if totalt_foer else 0

    print(f"\n{'═' * 130}")
    print(f"KRYSS-KILDE SWEEP  (uavhengige parametre per kilde)")
    print(f"  Utgangspunkt: {len(riktige)} riktige + {len(oversladdinger)} oversladd"
          f" = {totalt_foer} pred, presisjon {pres_foer:.1f}%"
          f"  [sortert etter: {sort_key}]")
    print(f"{'═' * 130}")

    # Overskrift — vis conf-kolumn per kilde kun om den kilden har conf
    har_conf = any(c is not None for c in conf_v)
    kilde_hdrs_list = []
    for k in kilde_liste:
        rik_k = [p for p in riktige if p["kilde"] == k]
        ov_k = [p for p in oversladdinger if p["kilde"] == k]
        k_har_conf = har_conf and any(p["conf"] is not None for p in rik_k + ov_k)
        if k_har_conf:
            kilde_hdrs_list.append(f"{k:>8} (e/h/b/c)")
        else:
            kilde_hdrs_list.append(f"{k:>8} (e/h/b)")
    kilde_hdrs = "  │  ".join(kilde_hdrs_list)
    print(f"  {kilde_hdrs}  │ {'rik.fj':>7} {'ov.fj':>7} {'netto':>7}"
          f" {'ov/rik':>7} {'pres%':>7}")
    sep_len = len(kilde_liste) * 26 + 45
    print(f"  {'─' * sep_len}")

    rader = []
    for kombo in product(*kandidat_lister):
        tot_rk = 0
        tot_ov = 0
        params = []
        for i, kilde in enumerate(kilde_liste):
            min_e, maks_h, maks_b, c_t, n_rk_k, n_ov_k = kombo[i]
            tot_rk += n_rk_k
            tot_ov += n_ov_k
            params.append((kilde, min_e, maks_h, maks_b, c_t))

        netto = tot_ov - tot_rk
        rk_pct = tot_rk / len(riktige) * 100 if riktige else 0
        rik_etter = len(riktige) - tot_rk
        ov_etter = len(oversladdinger) - tot_ov
        totalt_etter = rik_etter + ov_etter
        pres_etter = rik_etter / totalt_etter * 100 if totalt_etter > 0 else 0

        rader.append((netto, rk_pct, 0, tot_rk, tot_ov, rik_etter, ov_etter,
                      pres_etter, params))

    # Bruk samme sort-logikk (indeks 0-7 matcher SORT_FNS)
    sort_fn = SORT_FNS.get(sort_key, SORT_FNS["netto"])
    rader.sort(key=lambda x: sort_fn(x[:8]))

    # Vis topp 30 (med ov/rik-filter)
    # Cache per-kilde conf-info
    kilde_har_conf_map = {}
    for k in kilde_liste:
        rik_k = [p for p in riktige if p["kilde"] == k]
        ov_k = [p for p in oversladdinger if p["kilde"] == k]
        kilde_har_conf_map[k] = har_conf and any(p["conf"] is not None for p in rik_k + ov_k)

    n_vist = 0
    n_skjult = 0
    for rad in rader:
        if n_vist >= 30:
            break
        netto, rk_pct, _, tot_rk, tot_ov, rik_etter, ov_etter, pres_etter, params = rad
        # Filtrer på ov/rik-ratio
        if min_ov_rik is not None:
            ov_rik_val = (tot_ov / tot_rk) if tot_rk > 0 else float('inf') if tot_ov > 0 else 0
            if ov_rik_val <= min_ov_rik:
                n_skjult += 1
                continue
        # Filtrer på maks % riktige fjernet
        if maks_rik_pst is not None and rk_pct > maks_rik_pst:
            n_skjult += 1
            continue
        ratio_str = f"{tot_ov / tot_rk:.1f}" if tot_rk > 0 else "∞" if tot_ov > 0 else "–"
        param_strs = []
        for kilde, min_e, maks_h, maks_b, c_t in params:
            e_s = f"{min_e:g}" if min_e is not None else "–"
            h_s = f"{maks_h:g}" if maks_h is not None else "–"
            b_s = f"{maks_b:g}" if maks_b is not None else "–"
            if kilde_har_conf_map.get(kilde, False):
                c_s = f"{c_t:g}" if c_t is not None else "–"
                param_strs.append(f"{e_s:>4}/{h_s:>3}/{b_s:>4}/{c_s:>4}")
            else:
                param_strs.append(f"{e_s:>4}/{h_s:>3}/{b_s:>4}")
        kilde_info = "  │  ".join(
            f"{k:>8} {ps}" for (k, _, _, _, _), ps in zip(params, param_strs))
        markør = " ◀" if rk_pct == 0 and netto > 0 else ""
        print(f"  {kilde_info}  │ {tot_rk:>7} {tot_ov:>7} {netto:>+7d}"
              f" {ratio_str:>7} {pres_etter:>6.1f}%{markør}")
        n_vist += 1

    if n_skjult:
        filters = []
        if min_ov_rik is not None:
            filters.append(f"ov/rik ≤ {min_ov_rik:g}")
        if maks_rik_pst is not None:
            filters.append(f"rik.fj > {maks_rik_pst:g}%")
        print(f"  ({n_skjult} rader skjult: {' eller '.join(filters)})")


def _sweep_terskel(pred_liste, fasit, terskler):
    """Viser hvordan baseline-statistikken endrer seg med overlapp-terskelen."""
    print(f"\n{'─' * 95}")
    print("Sweep: OVERLAPP-TERSKEL (baseline uten geometrifiltre)")
    print(f"{'─' * 95}")
    print(f"  {'Terskel':>8} │ {'Riktige':>8} {'Oversladd':>10} {'Uten fasit':>11} │"
          f" {'Presisjon':>10} {'Tot. pred':>10}")
    print(f"  {'─' * 8}─┼─{'─' * 30}─┼─{'─' * 20}")
    for t in terskler:
        # Nullstill riktig-flagg
        for p in pred_liste:
            p.pop("riktig", None)
        n_rik, n_ov, n_uten = match_prediksjoner(pred_liste, fasit, t)
        totalt = n_rik + n_ov
        pres = n_rik / totalt * 100 if totalt else 0
        markør = " ◀" if abs(t - 0.15) < 1e-9 else ""
        print(f"  {t:>8.2f} │ {n_rik:>8} {n_ov:>10} {n_uten:>11} │"
              f" {pres:>9.1f}% {totalt:>10}{markør}")




def main():
    p = argparse.ArgumentParser(
        description="Overlapp-basert evaluering av filterkonfigurasjoner")
    p.add_argument("--fasit-csv", required=True,
                   help="Labels-CSV (ACCEPTED + manuell = fasit, REJECTED ekskluderes)")
    p.add_argument("--res-csv", required=True,
                   help="Resultat-CSV fra modellen (pikselkoordinater)")
    p.add_argument("--terskel", type=float, default=0.15,
                   help="Overlapp-terskel for å klassifisere prediksjon som riktig (default: 0.15)")
    p.add_argument("--sort", default="netto",
                   choices=["netto", "ov.fj", "rik.fj", "pres", "ov/rik"],
                   help="Sorteringskolonne for kombinasjons-sweep (default: netto)")
    p.add_argument("--min-ov-rik", type=float, default=None,
                   help="Vis kun rader der ov.fj/rik.fj > denne verdien (f.eks. 1.0)")
    p.add_argument("--maks-rik-pst", type=float, default=None,
                   help="Skjul rader der mer enn denne %% av riktige fjernes (f.eks. 0.5)")
    p.add_argument("--ut", default=None, metavar="FIL",
                   help="Skriv resultat til fil (default: auto-generert filnavn)")
    # Bakoverkompatibilitet
    p.add_argument("--csv", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.csv and not args.fasit_csv:
        args.fasit_csv = args.csv

    # ── Output-fil ───────────────────────────────────────────
    if args.ut is None:
        tidsstempel = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        ut_fil = f"filter_sweep_{tidsstempel}.txt"
    else:
        ut_fil = args.ut

    # Tee: skriv til fil + stdout for oppsummering
    class _Tee:
        """Skriver til fil, printer kun oppsummeringslinjer til terminal."""
        def __init__(self, filobj, terminal):
            self.fil = filobj
            self.terminal = terminal
            self._i_oppsummering = True  # start med oppsummering synlig

        def write(self, tekst):
            self.fil.write(tekst)
            if self._i_oppsummering:
                self.terminal.write(tekst)

        def flush(self):
            self.fil.flush()
            self.terminal.flush()

    fil = open(ut_fil, "w", encoding="utf-8")
    tee = _Tee(fil, sys.stdout)
    sys.stdout = tee

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

    # ── Slå av terminal-output for sweep-tabeller ──
    tee._i_oppsummering = False

    # ── Terskel-sweep (baseline) ──
    _sweep_terskel(pred, fasit,
                   [0.15, 0.25, 0.30, 0.35])

    # Tilbakestill til valgt terskel etter terskel-sweep
    for p in pred:
        p.pop("riktig", None)
    match_prediksjoner(pred, fasit, args.terskel)
    riktige = [p for p in pred if p.get("riktig") is True]
    oversladdinger = [p for p in pred if p.get("riktig") is False]

    # ── Enkeltparameter-sweeps ──
    # Merk: MIN_BOKS_RATIO (kun horisontal) er fjernet — overlapper med MIN_ELONGATION
    # Merk: MAKS_AREAL_PT² er fjernet — dekkes av kombinasjonen høyde × bredde

    _sweep_en_param(riktige, oversladdinger,
                    "MIN_ELONGATION max(w/h, h/w) — begge retninger",
                    [1.1, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0],
                    lambda v: {"min_ratio": None, "maks_hoyde": None,
                               "maks_bredde": None, "maks_areal": None,
                               "min_elongation": v})

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

    # Confidence-terskel sweep (kun relevant om conf finnes)
    if any(p["conf"] is not None for p in pred):
        _sweep_en_param(riktige, oversladdinger,
                        "CONF_TERSKEL (conf≥V → behold uansett geometri, "
                        "kombinert med e=1.5/h=50/b=120)",
                        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        lambda v: {"min_ratio": None, "maks_hoyde": 50,
                                   "maks_bredde": 120, "maks_areal": None,
                                   "min_elongation": 1.5, "conf_terskel": v})

    # ── Kombinasjons-sweep (samlet) ──
    elong_verdier = [None, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0]
    hoyde_verdier = [None, 40, 50, 60, 80]
    bredde_verdier = [None, 80, 100, 120, 150]
    conf_verdier = [None, 0.5]

    # Sjekk om det finnes conf-verdier i dataene
    har_conf_data = any(p["conf"] is not None for p in pred)
    aktiv_conf = conf_verdier if har_conf_data else [None]

    _sweep_kombinasjoner(riktige, oversladdinger,
                         elong_verdier, hoyde_verdier, bredde_verdier,
                         conf_v=aktiv_conf,
                         sort_key=args.sort, min_ov_rik=args.min_ov_rik,
                         maks_rik_pst=args.maks_rik_pst)

    # ── Per-kilde kombinasjons-sweep ──
    kilder = sorted(set(p["kilde"] for p in pred))
    if len(kilder) > 1:
        for kilde in kilder:
            rik_k = [p for p in riktige if p["kilde"] == kilde]
            ov_k = [p for p in oversladdinger if p["kilde"] == kilde]
            if not rik_k and not ov_k:
                continue
            # Kun conf-sweep for kilder som har conf
            kilde_har_conf = any(p["conf"] is not None for p in rik_k + ov_k)
            kilde_conf = conf_verdier if kilde_har_conf else [None]
            _sweep_kombinasjoner(rik_k, ov_k,
                                 elong_verdier, hoyde_verdier, bredde_verdier,
                                 conf_v=kilde_conf,
                                 sort_key=args.sort,
                                 tittel=f"PER KILDE: {kilde.upper()}",
                                 min_ov_rik=args.min_ov_rik,
                                 maks_rik_pst=args.maks_rik_pst)

        # ── Kryssvalidert sweep: uavhengige parametre per kilde ──
        _sweep_kryss_kilder(riktige, oversladdinger, kilder,
                            elong_verdier, hoyde_verdier, bredde_verdier,
                            conf_v=aktiv_conf,
                            sort_key=args.sort, min_ov_rik=args.min_ov_rik,
                            maks_rik_pst=args.maks_rik_pst)

    # ── Lukk output-fil og vis melding ──
    sys.stdout = tee.terminal
    fil.close()
    filstr = os.path.getsize(ut_fil)
    print(f"\n✓ Sweep-resultater skrevet til: {ut_fil} ({filstr // 1024} KB)")


if __name__ == "__main__":
    main()
