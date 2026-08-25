"""
Cache for YOLO detections per document and per model, so a new run with the
same weight file skips GPU inference, and when the OCR cache also hits the
whole PDF rendering. See ocr_cache.py for tokens and orientation.

Keyed by sha256 of the weight file (in the folder name) + document name:
/data2/cache/uttrekk_5/yolo/a1b2c3d4e5f6a7b8/1000039999.json, so a new model
gets its own cache without shadowing the old one.

Invalidation: imgsz, DPI, confidence floor and per-page rotation are stored in
each file and checked on lookup. See write_cache for the file layout.

Boxes are stored down to YOLO_CACHE_CONF_FLOOR and filtered against YOLO_CONF
on read, so changing YOLO_CONF, the geometry filters, the matching or the
evaluation threshold still hits, as long as the new threshold is not below
the floor the cache was written with.

"rotation" is the rotation of the image YOLO was given, i.e. the output of the
orientation step. It is part of the key so a changed orientation model gives a
miss instead of boxes in the wrong coordinate space.
"""

import hashlib
import json
import os

from config import PDF_DPI, YOLO_CACHE_CONF_FLOOR, YOLO_CONF, YOLO_IMGSZ

CACHE_VERSION = 1

# Hex digits of the weight hash used in the folder name. 16 hex = 64 bit,
# ample against collision between a manageable number of models.
HASH_LENGTH = 16

# (path, mtime, size) -> hash. The weight file is large enough (~50 MB) that
# we do not want to re-read it for every document.
_hash_cache = {}


def weights_hash(weights_path):
    """sha256 prefix of a weight file, memoised on mtime + size."""
    st = os.stat(weights_path)
    key = (os.path.abspath(weights_path), st.st_mtime_ns, st.st_size)
    if key not in _hash_cache:
        h = hashlib.sha256()
        with open(weights_path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _hash_cache[key] = h.hexdigest()[:HASH_LENGTH]
    return _hash_cache[key]


def cache_dir_for_weights(base_dir, weights_path):
    """Model-specific cache folder: {base}/{weight-hash}."""
    return os.path.join(base_dir, weights_hash(weights_path))


def _cache_path(cache_dir, doc_name):
    doc_id = os.path.splitext(os.path.basename(doc_name))[0]
    return os.path.join(cache_dir, f"{doc_id}.json")


def read_cache(cache_dir, doc_name, rotations):
    """Cached YOLO boxes for a document, or None if missing or invalid.

    `rotations` is the rotation of the image YOLO will run on, one per page,
    and must match what was stored. Returns one list of
    (x0, y0, x1, y1, conf) per page, filtered against YOLO_CONF.
    """
    path = _cache_path(cache_dir, doc_name)
    if not os.path.isfile(path):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if data.get("version") != CACHE_VERSION:
        return None
    if data.get("imgsz") != YOLO_IMGSZ:
        return None
    if data.get("dpi") != PDF_DPI:
        return None
    # The floor must be at or below today's threshold, or the cache is
    # missing boxes we now want.
    floor = data.get("conf_floor")
    if floor is None or floor > YOLO_CONF:
        return None

    pages = data["pages"]
    if len(pages) != len(rotations):
        return None
    if any(page["rotation"] != k for page, k in zip(pages, rotations)):
        return None

    return [
        [tuple(b) for b in page["boxes"] if b[4] >= YOLO_CONF]
        for page in pages
    ]


def write_cache(cache_dir, doc_name, rotations, boxes_per_page):
    """Write a document's raw YOLO boxes to the cache.

    `boxes_per_page` must come from a predict at YOLO_CACHE_CONF_FLOOR. Store
    a stricter selection and the floor in the file becomes a lie, so later
    runs with a lower YOLO_CONF hit an incomplete cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, doc_name)

    pages = []
    for si, (k, boxes) in enumerate(zip(rotations, boxes_per_page), start=1):
        pages.append({
            "page": si,
            "rotation": k,
            # Stored unrounded: json writes the shortest exactly
            # round-tripping string, so a cached run gives bit-identical boxes
            # to an uncached one. Rounding could tip a box over YOLO_CONF.
            "boxes": [list(box) for box in boxes],
        })

    data = {
        "version": CACHE_VERSION,
        "imgsz": YOLO_IMGSZ,
        "dpi": PDF_DPI,
        "conf_floor": YOLO_CACHE_CONF_FLOOR,
        "pages": pages,
    }

    # Write to a temp file first so an interrupt cannot leave a corrupt file
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
