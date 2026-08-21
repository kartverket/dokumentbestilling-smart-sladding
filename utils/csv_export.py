import csv
import os
import sys

_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from config import TREKK_FELT

# yolo_conf og paddle_rec_score er BEVISST to kolonner. De måler ulike ting:
# yolo_conf er deteksjonssikkerhet, paddle_rec_score er hvor sikkert OCR-en
# leste tegnene. Slås de sammen til én «conf»-kolonne, tolker
# filter_felles.er_filtrert en paddle-boks med høy lesekvalitet som en
# høykonfident deteksjon og lar den hoppe over geometrifiltrene — stikk i
# strid med regelen i config.py om at paddle-bokser alltid filtreres.
BASISFELT = ["navn", "side", "bilde_bredde", "bilde_hoyde", "x0", "y0", "x1", "y1",
             "kilde", "yolo_conf", "paddle_rec_score"]

# Trekkene er tomme for alle andre kilder enn «yolo» — se boks_trekk.
FELT = BASISFELT + list(TREKK_FELT)


def initialiser_csv(sti):
    with open(sti, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(FELT)


def _felt(boks, i, standard=""):
    return boks[i] if len(boks) > i and boks[i] is not None else standard


def append_csv(grupper, sti):
    n = 0
    with open(sti, "a", newline="", encoding="utf-8") as f:
        skriv = csv.writer(f)
        for (navn, si) in sorted(grupper):
            bw, bh, bokser = grupper[(navn, si)]
            for boks in bokser:
                x0, y0, x1, y1 = boks[:4]
                kilde = _felt(boks, 4, "paddle")
                yolo_conf = _felt(boks, 5)
                paddle_rec = _felt(boks, 6)
                trekk = boks[7] if len(boks) > 7 and boks[7] else {}
                skriv.writerow(
                    [navn, si, bw, bh, x0, y0, x1, y1, kilde, yolo_conf, paddle_rec]
                    + [_trekkverdi(trekk, felt) for felt in TREKK_FELT])
                n += 1
    return n


def _trekkverdi(trekk, felt):
    v = trekk.get(felt)
    return "" if v is None else v


def les_csv(sti):
    """Leser resultat-CSV tilbake til tegne-bokser.

    Tåler både nytt format (yolo_conf/paddle_rec_score) og gamle filer med
    én sammenslått «conf»-kolonne.
    """
    sladd_bokser = {}
    with open(sti, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            navn, si = r["navn"], int(r["side"])
            bw, bh = int(r["bilde_bredde"]), int(r["bilde_hoyde"])
            x0, y0, x1, y1 = float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])
            kilde = r.get("kilde", "paddle") or "paddle"
            yolo_conf = _flyt(r.get("yolo_conf"))
            paddle_rec = _flyt(r.get("paddle_rec_score"))
            if yolo_conf is None and paddle_rec is None:
                yolo_conf = _flyt(r.get("conf"))       # gammelt format
            trekk = {felt: _flyt(r.get(felt)) for felt in TREKK_FELT
                     if r.get(felt) not in (None, "")}
            boks = (x0, y0, x1, y1, kilde, yolo_conf, paddle_rec, trekk or None)
            sladd_bokser.setdefault((navn, si), (bw, bh, []))[2].append(boks)
    return sladd_bokser


def _flyt(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None
