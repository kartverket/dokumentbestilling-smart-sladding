import json
import os
import re

import numpy as np
from ultralytics import YOLO

from config import (
    YOLO_CONF, VERTIKAL_FAKTOR, YOLO_IMGSZ, MIN_SIFFER, MAKS_BOKSTAVER,
    MIN_BOKS_AREAL, MIN_ELONGATION, MAKS_ELONGATION, MIN_KORTSIDE_PT,
    MIN_KORTSIDE_YOLO_PT, MIN_LANGSIDE_YOLO_PT,
    PDF_DPI, standard_vekter)

YOLO_VEKTER = standard_vekter()
MIN_KORTSIDE_PX = MIN_KORTSIDE_PT * PDF_DPI / 72.0
MIN_KORTSIDE_YOLO_PX = MIN_KORTSIDE_YOLO_PT * PDF_DPI / 72.0
MIN_LANGSIDE_YOLO_PX = MIN_LANGSIDE_YOLO_PT * PDF_DPI / 72.0

_modell = None
_vekter_sti = YOLO_VEKTER


def sett_vekter(sti):
    global _vekter_sti, _modell
    if sti:
        _vekter_sti = sti
        _modell = None          # tving ny lasting hvis modellen alt er lastet


def aktive_vekter():
    return _vekter_sti


def modell_info():
    """Metadataen som ble publisert sammen med vektene, hvis den finnes.

    modell.json ligger ved siden av vektfilen (i vektlageret, og bygget inn
    i imaget av ./deploy.sh). Mangler den, er modellen kopiert på egen hånd
    og vi vet ikke hva den er trent på.
    """
    sti = os.path.join(os.path.dirname(os.path.abspath(_vekter_sti)), "modell.json")
    try:
        with open(sti, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _hent_modell():
    global _modell
    if _modell is None:
        if not os.path.isfile(_vekter_sti):
            raise FileNotFoundError(f"Fant ikke YOLO-vekter: {_vekter_sti}")
        info = modell_info()
        navn = info.get("navn")
        print(f"Laster YOLO-vekter fra: {_vekter_sti}"
              + (f"  (modell: {navn}, trent {info.get('trent', {}).get('dato', 'ukjent dato')})"
                 if navn else "  (ingen modell.json — ukjent opphav)"))
        _modell = YOLO(_vekter_sti)
    return _modell


def finn_yolo_bokser(bilde, conf=None):
    """Kjør YOLO paa ett bilde. conf=None gir predict-terskelen YOLO_CONF."""
    bgr = np.ascontiguousarray(bilde[:, :, ::-1])    # RGB (PyMuPDF) -> BGR (ultralytics)
    res = _hent_modell().predict(bgr, conf=YOLO_CONF if conf is None else conf,
                                 imgsz=YOLO_IMGSZ, verbose=False)
    ut = []
    for boks in res[0].boxes:
        x0, y0, x1, y1 = boks.xyxy[0].tolist()
        ut.append((x0, y0, x1, y1, boks.conf[0].item()))
    return ut


def _overlapp_andel(t, boks):
    bx0, by0, bx1, by1 = boks
    ix0, iy0 = max(t.x0, bx0), max(t.y0, by0)
    ix1, iy1 = min(t.x1, bx1), min(t.y1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    ta = (t.x1 - t.x0) * (t.y1 - t.y0)
    return ((ix1 - ix0) * (iy1 - iy0)) / ta if ta else 0.0


def tokens_i_boks(tokens, boks, terskel=0.3):
    return [t for t in tokens if _overlapp_andel(t, boks) > terskel]


def tell_siffer_bokstaver(tokens):
    """(siffer, bokstaver) i tokens som inneholder minst ett siffer.

    Rene ord-tokens hoppes over helt, slik at en etikett ved siden av tallet
    ikke teller som bokstaver. Skilt ut fra snill_sjekk fordi boks_trekk
    skriver de samme to tallene til resultat-CSV-en: deler de kode, kan
    tallene i sweepen ikke komme i utakt med regelen i produksjon.
    """
    n_siffer = 0
    bokstaver = 0
    for token in tokens:
        if not any(ch.isdigit() for ch in token.tekst):
            continue
        n_siffer += sum(ch.isdigit() for ch in token.tekst)
        bokstaver += sum(ch.isalpha() for ch in token.tekst)
    return n_siffer, bokstaver


def snill_sjekk(tokens, boks):
    n_siffer, bokstaver = tell_siffer_bokstaver(tokens_i_boks(tokens, boks))
    return n_siffer >= MIN_SIFFER and bokstaver <= MAKS_BOKSTAVER


def er_vertikal(boks):
    x0, y0, x1, y1 = boks[:4]
    return (y1 - y0) > VERTIKAL_FAKTOR * (x1 - x0)


def er_for_liten(boks):
    x0, y0, x1, y1 = boks[:4]
    return (x1 - x0) * (y1 - y0) < MIN_BOKS_AREAL


def har_feil_ratio(boks):
    """Forkaster bokser som er for kvadratiske eller for langstrakte.

    En sladding av 5 sifre har et smalt formområde. Under MIN_ELONGATION er
    boksen nesten kvadratisk (typisk falsk positiv); over MAKS_ELONGATION er
    den 3-4x bredere enn selve feltet og sladder unødvendig mye.
    Orienteringsuavhengig, så stående sladdinger behandles likt.
    """
    x0, y0, x1, y1 = boks[:4]
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return True
    elongation = max(w / h, h / w)
    return not (MIN_ELONGATION <= elongation <= MAKS_ELONGATION)


def er_for_tynn(boks):
    """Forkaster bokser der korteste side er for smal til å være tekst.

    Måler korteste side, ikke høyden, slik at stående sladdinger ikke rammes.
    """
    x0, y0, x1, y1 = boks[:4]
    return min(x1 - x0, y1 - y0) < MIN_KORTSIDE_PX


def har_yolo_stoyform(boks):
    """Strengere formkrav for rene YOLO-bokser — se MIN_*_YOLO_PT i config.

    Orienteringsuavhengig som er_for_tynn: kortside/langside, ikke bredde/høyde.
    """
    x0, y0, x1, y1 = boks[:4]
    kort, lang = sorted((x1 - x0, y1 - y0))
    return kort < MIN_KORTSIDE_YOLO_PX or lang < MIN_LANGSIDE_YOLO_PX



def overlapp_andel_boks(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    ov = (ix1 - ix0) * (iy1 - iy0)
    minst = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ov / minst if minst else 0.0