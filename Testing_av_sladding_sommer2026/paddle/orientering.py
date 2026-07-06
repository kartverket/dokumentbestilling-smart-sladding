import os
import numpy as np
from paddleocr import DocImgOrientationClassification

ORI_MODELL = "PP-LCNet_x1_0_doc_ori"
_MODELL_MAPPE = os.path.dirname(os.path.abspath(__file__))
ORI_MODELL_DIR = os.path.join(_MODELL_MAPPE, "PP-LCNet_x1_0_doc_ori_infer")

NEDSKALERING = 4        # klassifiser paa hver 4. piksel - holder fint og er raskt
MIN_KONFIDENS = 0.7     # under dette: stol ikke paa gjetningen, ikke roter

_orient = None


def _hent_orient():
    global _orient
    if _orient is None:
        _orient = DocImgOrientationClassification(
            model_name=ORI_MODELL,
            model_dir=ORI_MODELL_DIR,
        )
    return _orient


def finn_rotasjon(bilde):
    lite = np.ascontiguousarray(bilde[::NEDSKALERING, ::NEDSKALERING])
    try:
        res = _hent_orient().predict(lite)
        r = res[0]
        vinkel = int(r["label_names"][0])            # "0"/"90"/"180"/"270"
        score = float(np.asarray(r["scores"]).reshape(-1)[0])
    except Exception as e:
        print(f"!! orienteringssjekk feilet ({e!r}) - antar 0 grader")
        return 0
    if score < MIN_KONFIDENS:
        return 0
    return (vinkel // 90) % 4


def boks_tilbake(boks, k, w0, h0):
    if not k:
        return boks
    x0, y0, x1, y1 = boks
    if k == 1:
        pts = [(w0 - y0, x0), (w0 - y1, x1)]
    elif k == 2:
        pts = [(w0 - x0, h0 - y0), (w0 - x1, h0 - y1)]
    else:  # k == 3
        pts = [(y0, h0 - x0), (y1, h0 - x1)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))