import os
import re
import sys
from collections import defaultdict

import fitz

from save_result import lagre_resultat


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
        try:
            d = fitz.open(fil)
            r = d[si - 1].rect
            d.close()
            return r.width, r.height
        except Exception:
            pass
    iw, ih, _ = sladd_bokser[(navn, si)]
    return iw, ih

def mal_overlapp(sladd_bokser, fasit, mappe, terskel=0.32, y_origin="topp", kilder=None, yolo_bokser=None):
    if fasit is None:
        print("Ingen fasit — hopper over måling.")
        return None

    sum_fasit = sum_truffet = sum_pred = sum_overflod = 0
    sum_ov_areal = sum_fa_areal = 0.0
    pr_type = defaultdict(lambda: [0, 0])
    bom_filer = defaultdict(lambda: [0, 0])
    overflod_filer = defaultdict(int)
    oversladd_bokser = {}   # (navn, si) -> (iw, ih, [(x0,y0,x1,y1)])
    detaljer = []

    for (navn, si) in sorted(sladd_bokser):
        nr = _dok_nr(navn)
        iw, ih, raw = sladd_bokser[(navn, si)]
        # YOLO-bokser for denne siden (som koordinat-sett for oppslag)
        yolo_coords = set()
        if yolo_bokser and (navn, si) in yolo_bokser:
            _, _, yolo_raw = yolo_bokser[(navn, si)]
            yolo_coords = set(yolo_raw)
        kilde_liste = []
        conf_liste = []
        if kilder and (navn, si) in kilder:
            _, _, med_kilde = kilder[(navn, si)]
            kilde_liste = [b[4] if len(b) > 4 else "paddle" for b in med_kilde]
            conf_liste  = [b[5] if len(b) > 5 else None for b in med_kilde]
        pw, ph = _sidestr(navn, si, mappe, sladd_bokser)
        pred = [(b[0] / iw, (b[1] - 2) / ih, b[2] / iw, (b[3] + 2) / ih) for b in raw]
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
            if kilde_liste:
                kilde = kilde_liste[best_pi] if (0 <= best_pi < len(kilde_liste)) else ""
                kilde = kilde or "ukjent"   # prod-CSV har ingen kilde-kolonne
                conf  = conf_liste[best_pi]  if (0 <= best_pi < len(conf_liste)) else None
            else:
                kilde = "yolo" if (best_pi >= 0 and raw[best_pi] in yolo_coords) else "paddle"
                conf  = None
            pr_type[t][1] += 1
            sum_fa_areal += fa
            sum_ov_areal += best_ov
            bom_filer[(navn, si)][1] += 1
            detaljer.append({
                "fil": navn, "side": si, "fasit_nr": fi + 1, "type": t,
                "dekning_pst": round(best_dek * 100, 1),
                "resultat": "TRUFFET" if truffet else "MANGLER",
                "kilde": kilde if truffet else "",
                "conf": round(conf, 3) if (truffet and conf is not None) else "",
                "fasit_x0": round(fb[0], 6),
                "fasit_y0": round(fb[1], 6),
                "fasit_x1": round(fb[2], 6),
                "fasit_y1": round(fb[3], 6),
            })
            if truffet:
                sum_truffet += 1
                pr_type[t][0] += 1
                truffet_pred.add(best_pi)
            else:
                bom_filer[(navn, si)][0] += 1
            print(f"   fasit#{fi + 1} {t:<22} dekning={best_dek:5.0%}  IoU={best_iou:5.0%}  "
                  f"-> {'TRUFFET' if truffet else 'MANGLER'}")
        n_overflod = len(pred) - len(truffet_pred)
        sum_overflod += n_overflod
        if n_overflod > 0:
            overflod_filer[(navn, si)] += n_overflod
            over_bokser = [raw[i] for i in range(len(raw)) if i not in truffet_pred]
            oversladd_bokser[(navn, si)] = (iw, ih, over_bokser)

    print("\n" + "=" * 64)
    rec = sum_truffet / sum_fasit if sum_fasit else 0.0
    print(f"Recall (truffet / fasit):         {sum_truffet}/{sum_fasit} = {rec:.0%}")
    print(f"Samlet overlapp (areal):          {(sum_ov_areal / sum_fa_areal if sum_fa_areal else 0):.0%}")
    print(f"Sladde-bokser totalt:             {sum_pred}")
    print(f"Over-sladding (uten fasit-treff): {sum_overflod}")
    print(f"Terskel for treff:                {terskel:.0%} av fasit-boksens areal")

    # Diagnostikk: vis om fasit-oppslag feilet
    eval_dok_nrs = {_dok_nr(n) for (n, _) in sladd_bokser}
    eval_dok_nrs.discard(None)
    fasit_dok_nrs = {nr for (nr, _) in fasit}
    felles = eval_dok_nrs & fasit_dok_nrs
    n_fasit_totalt = sum(len(v) for v in fasit.values())

    if sum_fasit == 0 and n_fasit_totalt > 0 and eval_dok_nrs:
        print(f"\n!! ADVARSEL: Ingen av de {len(eval_dok_nrs)} evaluerte dokumentene "
              f"matcher de {len(fasit_dok_nrs)} dokumentene i fasit.")
        print(f"   Evaluert (dok_nr fra filnavn):  {sorted(eval_dok_nrs)[:5]}")
        print(f"   Fasit (fil_revisjon_id):        {sorted(fasit_dok_nrs)[:5]}")
        eval_navneksempler = sorted({n for (n, _) in sladd_bokser})[:3]
        print(f"   Filnavn-eksempler:              {eval_navneksempler}")
        print(f"   Sjekk at filnavnene i --mappe samsvarer med fil_revisjon_id i --fasit-csv.")
    elif felles and len(felles) < len(eval_dok_nrs):
        n_uten = len(eval_dok_nrs) - len(felles)
        print(f"\n   Info: {len(felles)}/{len(eval_dok_nrs)} evaluerte dokumenter har fasit "
              f"({n_uten} uten fasit-bokser)")

    print("Recall per type:")
    for t, (tr, tot) in sorted(pr_type.items()):
        print(f"   {t or '(tom)':<22} {tr}/{tot} = {tr / tot:.0%}")

    feil = sorted((k for k, (b, _tot) in bom_filer.items() if b > 0))
    print("\n" + "=" * 64)
    if sum_fasit == 0:
        if n_fasit_totalt > 0:
            print(f"Ingen fasit-bokser matchet de evaluerte dokumentene "
                  f"({n_fasit_totalt} fasit-bokser finnes, men for andre dokumenter).")
        else:
            print("Ingen fasit-bokser lastet — kan ikke måle recall.")
    elif feil:
        print(f"Filer med bom ({len(feil)} side(r) med minst én MANGLER):")
       
        for (navn, si) in feil:
            bom, tot = bom_filer[(navn, si)]
            print(f"   {navn}  side {si}:  {bom}/{tot} fasit-bokser bommet")
    else:
        print("Ingen bom — alle fasit-bokser ble truffet. 🎉")
                   

    return {
        "recall": rec, "truffet": sum_truffet, "fasit": sum_fasit,
        "pred": sum_pred, "overflod": sum_overflod,
        "samlet_overlapp": sum_ov_areal / sum_fa_areal if sum_fa_areal else 0.0,
        "terskel": terskel,
        "pr_type": {t: tuple(v) for t, v in pr_type.items()},
        "detaljer": detaljer,
        "bom_filer": [
            {"fil": navn, "side": si, "bom": b, "fasit_totalt": tot}
            for (navn, si), (b, tot) in sorted(bom_filer.items()) if b > 0
        ],
        "oversladd_bokser": oversladd_bokser,
        "overflod_filer": [
            {"fil": navn, "side": si, "oversladd": n}
            for (navn, si), n in sorted(overflod_filer.items())
        ],
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
