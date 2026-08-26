import csv
import os
import sys

_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from config import FEATURE_FIELDS

# Two columns on purpose: merged into one "conf", filter_common.is_filtered would
# read a well-read paddle box as a high-confidence detection and let it skip the
# geometry filters, against the config.py rule that paddle boxes are always filtered.
BASE_FIELD = ["navn", "side", "bilde_bredde", "bilde_hoyde", "x0", "y0", "x1", "y1",
             "kilde", "yolo_conf", "paddle_rec_score"]

# Features are empty for every kilde other than "yolo". See box_features.
FIELD = BASE_FIELD + list(FEATURE_FIELDS)


def write_csv_header(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(FIELD)


def _field(box, i, standard=""):
    return box[i] if len(box) > i and box[i] is not None else standard


def append_csv(groups, path):
    n = 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        write = csv.writer(f)
        for (name, si) in sorted(groups):
            bw, bh, boxes = groups[(name, si)]
            for box in boxes:
                x0, y0, x1, y1 = box[:4]
                source = _field(box, 4, "paddle")
                yolo_conf = _field(box, 5)
                paddle_rec = _field(box, 6)
                features = box[7] if len(box) > 7 and box[7] else {}
                write.writerow(
                    [name, si, bw, bh, x0, y0, x1, y1, source, yolo_conf, paddle_rec]
                    + [_feature_value(features, field) for field in FEATURE_FIELDS])
                n += 1
    return n


def _feature_value(features, field):
    v = features.get(field)
    return "" if v is None else v


def read_result_csv(path):
    """Reads a result CSV back into drawable boxes.

    Handles both the new format (yolo_conf/paddle_rec_score) and old files with a
    single merged "conf" column.
    """
    sladd_boxes = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name, si = r["navn"], int(r["side"])
            bw, bh = int(r["bilde_bredde"]), int(r["bilde_hoyde"])
            x0, y0, x1, y1 = float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])
            source = r.get("kilde", "paddle") or "paddle"
            yolo_conf = _float(r.get("yolo_conf"))
            paddle_rec = _float(r.get("paddle_rec_score"))
            if yolo_conf is None and paddle_rec is None:
                yolo_conf = _float(r.get("conf"))       # old format
            features = {field: _float(r.get(field)) for field in FEATURE_FIELDS
                     if r.get(field) not in (None, "")}
            box = (x0, y0, x1, y1, source, yolo_conf, paddle_rec, features or None)
            sladd_boxes.setdefault((name, si), (bw, bh, []))[2].append(box)
    return sladd_boxes


def _float(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None
