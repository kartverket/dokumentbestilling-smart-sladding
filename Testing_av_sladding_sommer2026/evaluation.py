from save_result import lagre_resultat
import os
import re
from collections import defaultdict

import fitz


def _dok_nr(navn):
    m = re.match(r"0*(\d+)", os.path.basename(navn))
    return int(m.group(1)) if m else None


def _norm_csv(x, y, w, h, pw, ph, y_origin):
    x0, x1 = sorted((x, x + w))
    y0, y1 = sorted((y, y + h))
    if y_origin == "bunn":
        y0, y1 = ph - y1, ph - y0
    return (x0 / pw, y0 / ph, x1 / pw, y1 / ph)


def _overlap(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return (ix1 - ix0) * (iy1 - iy0) if (ix1 > ix0 and iy1 > iy0) else 0.0


def _areal(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _sidestr(navn, si, mappe, sladd_bokser):
    fil = os.path.join(mappe, navn)
    if fil.lower().endswith(".pdf"):
        d = fitz.open(fil)
        r = d[si - 1].rect
        d.close()
        return r.width, r.height
    iw, ih, _ = sladd_bokser[(navn, si)]
    return iw, ih


def mal_overlapp(sladd_bokser, fasit, mappe, terskel=0.40, y_origin="topp"):
    if fasit is None:
        print("Ingen fasit — hopper over måling.")
        return None

    sum_fasit = sum_truffet = sum_pred = sum_overflod = 0
    sum_ov_areal = sum_fa_areal = 0.0
    pr_type = defaultdict(lambda: [0, 0])
    bom_filer = defaultdict(lambda: [0, 0])

    for (navn, si) in sorted(sladd_bokser):
        nr = _dok_nr(navn)
        iw, ih, raw = sladd_bokser[(navn, si)]
        pw, ph = _sidestr(navn, si, mappe, sladd_bokser)
        pred = [(x0 / iw, (y0 - 2) / ih, x1 / iw, (y1 + 2) / ih) for (x0, y0, x1, y1) in raw]
        fbokser = [(_norm_csv(x, y, w, h, pw, ph, y_origin), t)
                   for (x, y, w, h, t) in fasit.get((nr, si), [])]

        sum_pred += len(pred)
        sum_fasit += len(fbokser)
        truffet_pred = set()
        if fbokser:
            print(f"\n{navn}  (dok_nr={nr}, side {si})")
        for fi, (fb, t) in enumerate(fbokser):
            fa = _areal(fb)
            best_dek = best_iou = best_ov = 0.0
            best_pi = -1
            for pi, pb in enumerate(pred):
                ov = _overlap(fb, pb)
                if ov > best_ov:
                    best_ov = ov
                    best_dek = ov / fa if fa else 0.0
                    best_iou = ov / (fa + _areal(pb) - ov)
                    best_pi = pi
            truffet = best_dek >= terskel
            pr_type[t][1] += 1
            sum_fa_areal += fa
            sum_ov_areal += best_ov
            bom_filer[(navn, si)][1] += 1
            if truffet:
                sum_truffet += 1
                pr_type[t][0] += 1
                truffet_pred.add(best_pi)
            else:
                bom_filer[(navn, si)][0] += 1
            print(f"   fasit#{fi + 1} {t:<22} dekning={best_dek:5.0%}  IoU={best_iou:5.0%}  "
                  f"-> {'TRUFFET' if truffet else 'MANGLER'}")
        sum_overflod += len(pred) - len(truffet_pred)


    nr_til_navn = {}
    for (navn, _si) in sladd_bokser:
        nr_til_navn.setdefault(_dok_nr(navn), navn)
    for (nr, side), fb in fasit.items():
        navn = nr_til_navn.get(nr)
        if navn is None or (navn, side) in sladd_bokser:
            continue                       # dok ikke prosessert, eller side alt talt
        print(f"\n{navn}  (dok_nr={nr}, side {side})  -- ingen prediksjon paa sida")
        for (_x, _y, _w, _h, t) in fb:
            sum_fasit += 1
            pr_type[t][1] += 1
            bom_filer[(navn, side)][1] += 1
            bom_filer[(navn, side)][0] += 1
            print(f"   fasit {t:<22} dekning=  0%  IoU=  0%  -> MANGLER")

    print("\n" + "=" * 64)
    rec = sum_truffet / sum_fasit if sum_fasit else 0.0
    print(f"Recall (truffet / fasit):         {sum_truffet}/{sum_fasit} = {rec:.0%}")
    print(f"Samlet overlapp (areal):          {(sum_ov_areal / sum_fa_areal if sum_fa_areal else 0):.0%}")
    print(f"Sladde-bokser totalt:             {sum_pred}")
    print(f"Over-sladding (uten fasit-treff): {sum_overflod}")
    print(f"Terskel for treff:                {terskel:.0%} av fasit-boksens areal")
    print("Recall per type:")
    for t, (tr, tot) in sorted(pr_type.items()):
        print(f"   {t or '(tom)':<22} {tr}/{tot} = {tr / tot:.0%}")

    feil = sorted((k for k, (b, _tot) in bom_filer.items() if b > 0))
    print("\n" + "=" * 64)
    if feil:
        print(f"Filer med bom ({len(feil)} side(r) med minst én MANGLER):")
        for (navn, si) in feil:
            bom, tot = bom_filer[(navn, si)]
            print(f"   {navn}  side {si}:  {bom}/{tot} fasit-bokser bommet")
    else:
        print("Ingen bom — alle fasit-bokser ble truffet. 🎉")


    resultat = mal_overlapp(sladd_bokser, fasit, mappe)
    lagre_resultat(resultat)                       # saves to current directory
    lagre_resultat(resultat, mappe="resultater/")  # or a specific folder

    return {
        "recall": rec, "truffet": sum_truffet, "fasit": sum_fasit,
        "pred": sum_pred, "overflod": sum_overflod,
        "pr_type": {t: tuple(v) for t, v in pr_type.items()},
    }

import csv
from collections import defaultdict


def les_fasit(csv_sti):
    fasit = defaultdict(list)
    try:
        with open(csv_sti, newline="", encoding="utf-8-sig") as f:
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
                fasit[(nr, side)].append((x, y, w, h, (r.get("type") or "").strip()))
    except FileNotFoundError:
        print(f"!! Fant ikke CSV: {csv_sti} — fasit tegnes ikke, og treff måles ikke.")
        return None

    print(f"Fasit: {sum(len(v) for v in fasit.values())} boks(er) i "
          f"{len(fasit)} (dok_nr, side)-grupper fra {csv_sti}.")
    return fasit