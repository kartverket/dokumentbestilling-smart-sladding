import os
import numpy as np

from config import ORIENTATION_DOWNSCALE, ORIENTATION_MIN_CONFIDENCE

ORIENTATION_MODEL = "PP-LCNet_x1_0_doc_ori"
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ORIENTATION_MODEL_DIR = os.path.join(_MODEL_DIR, "PP-LCNet_x1_0_doc_ori_infer")

_orient = None


def _fetch_orient():
    global _orient
    if _orient is None:
        # Imported here, not at the top: cache readers should not pay for
        # the paddleocr import.
        from paddleocr import DocImgOrientationClassification

        _orient = DocImgOrientationClassification(
            model_name=ORIENTATION_MODEL,
            model_dir=ORIENTATION_MODEL_DIR,
        )
    return _orient


def find_rotations_batch(images):
    """Rotation (0-3) per image, in one batched model call.

    One call for the whole list: per image the GPU was spun up and down once
    per page.
    """
    if not images:
        return []
    lite_images = [np.ascontiguousarray(b[::ORIENTATION_DOWNSCALE, ::ORIENTATION_DOWNSCALE]) for b in images]
    try:
        results = _fetch_orient().predict(lite_images)
        rotations = []
        for r in results:
            angle = int(r["label_names"][0])
            score = float(np.asarray(r["scores"]).reshape(-1)[0])
            if score < ORIENTATION_MIN_CONFIDENCE:
                rotations.append(0)
            else:
                rotations.append((angle // 90) % 4)
        return rotations
    except Exception as e:
        print(f"!! batch orientation check failed ({e!r}) - assuming 0 degrees for all")
        return [0] * len(images)


def unrotate_box(box, k, w0, h0):
    if not k:
        return box
    x0, y0, x1, y1 = box
    if k == 1:
        pts = [(w0 - y0, x0), (w0 - y1, x1)]
    elif k == 2:
        pts = [(w0 - x0, h0 - y0), (w0 - x1, h0 - y1)]
    else: 
        pts = [(y0, h0 - x0), (y1, h0 - x1)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))