"""
Cache for PaddleOCR tokens and orientation results per document, so expensive
GPU work is skipped when the same document turns up in a new run.

One file per document, keyed by document name (fil_revisjon_id), under one
folder per uttrekk: /data2/cache/uttrekk_4/ocr/123456789.json

Invalidation: OCR model version + DPI are stored in each file and checked on
lookup, so a change misses every entry automatically. See write_cache for the
file layout.
"""

import json
import os
from collections import namedtuple

from cache_path import cache_path
from config import PADDLE_MODEL_SET, PDF_DPI

# Identical to Token in paddle_ocr_model_fnr.py, duplicated here to avoid
# importing PaddleOCR just to read the cache.
Token = namedtuple("Token", ["text", "x0", "y0", "x1", "y1", "rec_score"])

CACHE_VERSION = 2


def read_cache(cache_dir, doc_name):
    """Cached OCR result for a document, or None if missing or invalid.

    Raises ValueError on a document name that cannot be a cache key.
    """
    path = cache_path(cache_dir, doc_name)
    if not os.path.isfile(path):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if data.get("version") != CACHE_VERSION:
        return None
    if data.get("ocr_model") != PADDLE_MODEL_SET:
        return None
    if data.get("dpi") != PDF_DPI:
        return None

    rotations = []
    tokens_per_page = []
    for page in data["pages"]:
        rotations.append(page["rotation"])
        tokens = [
            Token(t["text"], t["x0"], t["y0"], t["x1"], t["y1"], t.get("rec_score"))
            for t in page["tokens"]
        ]
        tokens_per_page.append(tokens)

    return rotations, tokens_per_page


def write_cache(cache_dir, doc_name, rotations, tokens_per_page):
    """Write a document's OCR result to the cache.

    Raises ValueError on a document name that cannot be a cache key.
    """
    path = cache_path(cache_dir, doc_name)
    os.makedirs(cache_dir, exist_ok=True)

    pages = []
    for si, (rot, tokens) in enumerate(zip(rotations, tokens_per_page), start=1):
        pages.append({
            "page": si,
            "rotation": rot,
            "tokens": [
                {"text": t.text, "x0": t.x0, "y0": t.y0, "x1": t.x1, "y1": t.y1,
                 "rec_score": t.rec_score}
                for t in tokens
            ],
        })

    data = {
        "version": CACHE_VERSION,
        "ocr_model": PADDLE_MODEL_SET,
        "dpi": PDF_DPI,
        "pages": pages,
    }

    # Write to a temp file first so an interrupt cannot leave a corrupt file
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)




