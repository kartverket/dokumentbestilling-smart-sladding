import os
import numpy as np
from paddleocr import DocImgOrientationClassification

from config import NEDSKALERING, MIN_KONFIDENS

ORI_MODELL = "PP-LCNet_x1_0_doc_ori"
_MODELL_MAPPE = os.path.dirname(os.path.abspath(__file__))
ORI_MODELL_DIR = os.path.join(_MODELL_MAPPE, "PP-LCNet_x1_0_doc_ori_infer")

_orient = None


def _hent_orient():
    global _orient
    if _orient is None:
        _orient = DocImgOrientationClassification(
            model_name=ORI_MODELL,
            model_dir=ORI_MODELL_DIR,
        )
    return _orient


def finn_rotasjoner_batch(bilder):
    """Kjør orienteringsdeteksjon på en liste med bilder i én batch.

    Returnerer en liste med rotasjonsverdier (0-3) for hvert bilde.
    Ett modellkall for hele listen — per bilde ble GPU-en startet og
    stoppet én gang per side.
    """
    if not bilder:
        return []
    lite_bilder = [np.ascontiguousarray(b[::NEDSKALERING, ::NEDSKALERING]) for b in bilder]
    try:
        resultater = _hent_orient().predict(lite_bilder)
        rotasjoner = []
        for r in resultater:
            vinkel = int(r["label_names"][0])
            score = float(np.asarray(r["scores"]).reshape(-1)[0])
            if score < MIN_KONFIDENS:
                rotasjoner.append(0)
            else:
                rotasjoner.append((vinkel // 90) % 4)
        return rotasjoner
    except Exception as e:
        print(f"!! batch-orienteringssjekk feilet ({e!r}) - antar 0 grader for alle")
        return [0] * len(bilder)


def boks_tilbake(boks, k, w0, h0):
    if not k:
        return boks
    x0, y0, x1, y1 = boks
    if k == 1:
        pts = [(w0 - y0, x0), (w0 - y1, x1)]
    elif k == 2:
        pts = [(w0 - x0, h0 - y0), (w0 - x1, h0 - y1)]
    else: 
        pts = [(y0, h0 - x0), (y1, h0 - x1)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))