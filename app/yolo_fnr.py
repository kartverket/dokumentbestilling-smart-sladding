import json
import os
import re

import numpy as np

from config import (
    YOLO_CONF, VERTICAL_FACTOR, YOLO_IMGSZ, MIN_DIGITS, MAX_LETTERS,
    MIN_BOX_AREA, MIN_ELONGATION, MAX_ELONGATION, MIN_SHORT_SIDE_PT,
    MIN_SHORT_SIDE_YOLO_PT, MIN_LONG_SIDE_YOLO_PT,
    MIN_SHORT_SIDE_PADDLE_PT, MIN_LONG_SIDE_PADDLE_PT, MAX_ELONGATION_PADDLE,
    PDF_DPI, default_weights)
from geometry import covered_share

YOLO_WEIGHTS = default_weights()
MIN_SHORT_SIDE_PX = MIN_SHORT_SIDE_PT * PDF_DPI / 72.0
MIN_SHORT_SIDE_YOLO_PX = MIN_SHORT_SIDE_YOLO_PT * PDF_DPI / 72.0
MIN_LONG_SIDE_YOLO_PX = MIN_LONG_SIDE_YOLO_PT * PDF_DPI / 72.0
MIN_SHORT_SIDE_PADDLE_PX = MIN_SHORT_SIDE_PADDLE_PT * PDF_DPI / 72.0
MIN_LONG_SIDE_PADDLE_PX = MIN_LONG_SIDE_PADDLE_PT * PDF_DPI / 72.0

_model = None
_weights_path = YOLO_WEIGHTS


def set_weights(path):
    global _weights_path, _model
    if path:
        _weights_path = path
        _model = None          # force a reload if a model is already loaded


def active_weights():
    return _weights_path


def model_info():
    """Metadata published alongside the weights, if present.

    modell.json sits next to the weight file. If it is missing, the model was
    copied by hand and we do not know what it was trained on.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(_weights_path)), "modell.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _fetch_model():
    global _model
    if _model is None:
        # Imported here, not at the top: cache readers use only the geometry
        # functions and should not pay for the ultralytics import.
        from ultralytics import YOLO

        if not os.path.isfile(_weights_path):
            raise FileNotFoundError(f"YOLO weights not found: {_weights_path}")
        info = model_info()
        name = info.get("name")
        print(f"Loading YOLO weights from: {_weights_path}"
              + (f"  (model: {name}, trained {info.get('trained', {}).get('date', 'unknown date')})"
                 if name else "  (no modell.json, unknown origin)"))
        _model = YOLO(_weights_path)
    return _model


def find_yolo_boxes(image, conf=None):
    """Run YOLO on one image. conf=None uses the YOLO_CONF threshold."""
    bgr = np.ascontiguousarray(image[:, :, ::-1])    # RGB (PyMuPDF) -> BGR (ultralytics)
    res = _fetch_model().predict(bgr, conf=YOLO_CONF if conf is None else conf,
                                 imgsz=YOLO_IMGSZ, verbose=False)
    ut = []
    for box in res[0].boxes:
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        ut.append((x0, y0, x1, y1, box.conf[0].item()))
    return ut


def tokens_in_box(tokens, box, threshold=0.3):
    return [t for t in tokens
            if covered_share((t.x0, t.y0, t.x1, t.y1), box) > threshold]


def count_digits_and_letters(tokens):
    """(digits, letters) in tokens that contain at least one digit.

    Pure word tokens are skipped so a label next to the number does not count
    as letters. Shared with box_features, which writes the same two numbers to
    the result CSV: sharing the code keeps the sweep in step with the rule
    running in production.
    """
    n_digits = 0
    letters = 0
    for token in tokens:
        if not any(ch.isdigit() for ch in token.text):
            continue
        n_digits += sum(ch.isdigit() for ch in token.text)
        letters += sum(ch.isalpha() for ch in token.text)
    return n_digits, letters


def lenient_check(tokens, box):
    n_digits, letters = count_digits_and_letters(tokens_in_box(tokens, box))
    return n_digits >= MIN_DIGITS and letters <= MAX_LETTERS


def is_vertical(box):
    x0, y0, x1, y1 = box[:4]
    return (y1 - y0) > VERTICAL_FACTOR * (x1 - x0)


def is_too_small(box):
    x0, y0, x1, y1 = box[:4]
    return (x1 - x0) * (y1 - y0) < MIN_BOX_AREA


def has_wrong_ratio(box):
    """Reject boxes that are too square or too elongated.

    Below MIN_ELONGATION the box is nearly square (typical false positive);
    above MAX_ELONGATION it is 3-4x wider than the field and sladder far more
    than needed. Orientation-free, so vertical sladdinger are treated alike.
    """
    x0, y0, x1, y1 = box[:4]
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return True
    elongation = max(w / h, h / w)
    return not (MIN_ELONGATION <= elongation <= MAX_ELONGATION)


def is_too_thin(box):
    """Reject boxes whose shortest side is too thin to be text.

    Measures the shortest side, not the height, so vertical sladdinger survive.
    """
    x0, y0, x1, y1 = box[:4]
    return min(x1 - x0, y1 - y0) < MIN_SHORT_SIDE_PX


def is_too_narrow_yolo(box):
    """Short side below MIN_SHORT_SIDE_YOLO_PT is noise at any confidence."""
    x0, y0, x1, y1 = box[:4]
    return min(x1 - x0, y1 - y0) < MIN_SHORT_SIDE_YOLO_PX


def is_too_short_yolo(box):
    """Long side below MIN_LONG_SIDE_YOLO_PT is too short for 5 digits.

    High confidence exempts (see _hopp_over_geometrifilter), unlike the short
    side limit above.
    """
    x0, y0, x1, y1 = box[:4]
    return max(x1 - x0, y1 - y0) < MIN_LONG_SIDE_YOLO_PX


def has_paddle_noise_shape(box):
    """Stricter shape for paddle boxes. See MIN_*_PADDLE_* in config.

    Paddle is never exempted by confidence, so all three limits always apply.
    """
    x0, y0, x1, y1 = box[:4]
    short, long = sorted((x1 - x0, y1 - y0))
    if short <= 0:
        return True
    return (short < MIN_SHORT_SIDE_PADDLE_PX or long < MIN_LONG_SIDE_PADDLE_PX
            or long / short > MAX_ELONGATION_PADDLE)
