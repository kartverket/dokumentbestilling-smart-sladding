import numpy as np
from paddleocr import DocImgOrientationClassification

_orient = None


def _hent_orient():
    global _orient
    if _orient is None:
        _orient = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")
    return _orient


def finn_rotasjon(bilde):
    res = _hent_orient().predict(np.ascontiguousarray(bilde[::4, ::4]))
    try:
        vinkel = int(res[0]["label_names"][0])      
        return (vinkel // 90) % 4
    except Exception:
        return 0


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
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))
