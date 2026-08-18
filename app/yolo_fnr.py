import os
import re

import numpy as np
from ultralytics import YOLO

from config import (
    YOLO_CONF, VERTIKAL_FAKTOR, YOLO_IMGSZ, MIN_SIFFER, MAKS_BOKSTAVER,
    MIN_BOKS_AREAL, MIN_ELONGATION, MAKS_BOKS_HOYDE_PT, MAKS_BOKS_BREDDE_PT,
    MAKS_BREDDE_ELEKTRONISK_PT, PDF_DPI)

YOLO_VEKTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best.pt")
MAKS_BOKS_HOYDE_PX = MAKS_BOKS_HOYDE_PT * PDF_DPI / 72.0
MAKS_BOKS_BREDDE_PX = MAKS_BOKS_BREDDE_PT * PDF_DPI / 72.0
MAKS_BREDDE_ELEKTRONISK_PX = MAKS_BREDDE_ELEKTRONISK_PT * PDF_DPI / 72.0

_modell = None
_vekter_sti = YOLO_VEKTER


def sett_vekter(sti):
    global _vekter_sti, _modell
    if sti:
        _vekter_sti = sti
        _modell = None          # tving ny lasting hvis modellen alt er lastet


def _hent_modell():
    global _modell
    if _modell is None:
        if not os.path.isfile(_vekter_sti):
            raise FileNotFoundError(f"Fant ikke YOLO-vekter: {_vekter_sti}")
        print(f"Laster YOLO-vekter fra: {_vekter_sti}")
        _modell = YOLO(_vekter_sti)
    return _modell


def finn_yolo_bokser(bilde):
    bgr = np.ascontiguousarray(bilde[:, :, ::-1])    # RGB (PyMuPDF) -> BGR (ultralytics)
    res = _hent_modell().predict(bgr, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)
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


def snill_sjekk(tokens, boks):
    n_siffer = 0
    bokstaver = 0
    for token in tokens_i_boks(tokens, boks):
        if not any(ch.isdigit() for ch in token.tekst):
            continue
        n_siffer += sum(ch.isdigit() for ch in token.tekst)
        bokstaver += sum(ch.isalpha() for ch in token.tekst)
    return n_siffer >= MIN_SIFFER and bokstaver <= MAKS_BOKSTAVER


def er_vertikal(boks):
    x0, y0, x1, y1 = boks[:4]
    return (y1 - y0) > VERTIKAL_FAKTOR * (x1 - x0)


def er_for_liten(boks):
    x0, y0, x1, y1 = boks[:4]
    return (x1 - x0) * (y1 - y0) < MIN_BOKS_AREAL


def har_feil_ratio(boks):
    """Forkaster nesten-kvadratiske bokser (elongation < MIN_ELONGATION)."""
    x0, y0, x1, y1 = boks[:4]
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return True
    elongation = max(w / h, h / w)
    return elongation < MIN_ELONGATION


def er_for_hoy(boks):
    x0, y0, x1, y1 = boks[:4]
    return (y1 - y0) > MAKS_BOKS_HOYDE_PX


def er_for_bred(boks, elektronisk=False):
    x0, y0, x1, y1 = boks[:4]
    grense = MAKS_BREDDE_ELEKTRONISK_PX if elektronisk else MAKS_BOKS_BREDDE_PX
    return (x1 - x0) > grense



def overlapp_andel_boks(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    ov = (ix1 - ix0) * (iy1 - iy0)
    minst = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ov / minst if minst else 0.0