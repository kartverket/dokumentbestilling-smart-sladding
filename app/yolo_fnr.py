import os
import re

import numpy as np
from ultralytics import YOLO

from config import (
    YOLO_CONF, VERTIKAL_FAKTOR, YOLO_IMGSZ, MIN_SIFFER, MAKS_BOKSTAVER
)

YOLO_VEKTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "weights", "best.pt")

_modell = None


def _hent_modell():
    global _modell
    if _modell is None:
        _modell = YOLO(YOLO_VEKTER)
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


def overlapp_andel_boks(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    ov = (ix1 - ix0) * (iy1 - iy0)
    minst = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ov / minst if minst else 0.0