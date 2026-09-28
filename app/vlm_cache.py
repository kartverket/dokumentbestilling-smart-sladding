"""Cache for VLM judgements per crop, prompt, config and model.

vlm_judge answers are expensive GPU work and deterministic at temperature 0,
so a re-run with the same prompt, config and model can reuse them. The cache
is content-addressed on both sides of the call:

    folder: {base}/{fingerprint}/       one folder per (prompt, templates,
                                        model, mode, temperature, max_tokens,
                                        thinking) — the prompt VERSION
    file:   {folder}/{item_key}.json    one file per judged input: sha256 of
                                        the crop bytes and the OCR text the
                                        model was shown

The folder name doubles as the prompt version: any edit to the prompt or the
config gives a new fingerprint, so a stale judgement can never shadow a new
experiment — the same guarantee cache_dir_for_weights gives the YOLO cache.
meta.json inside the folder holds the full prompt and the parameters, so a
fingerprint can always be traced back to what produced it.

Only the raw model answer is stored, not the parsed fields: parsing is cheap
and local, and a parser fix then applies to cached answers too. Answers that
did not parse are not stored — resume must retry those.

NOTE: the raw answers transcribe real fødselsnumre — the cache is exactly as
sensitive as the judgement CSVs and must stay on the same disk.
"""

import hashlib
import json
import os
import threading

CACHE_VERSION = 1

# Hex digits of the fingerprint. 16 hex = 64 bit, ample against collision
# between a manageable number of prompt versions.
FINGERPRINT_LENGTH = 16

# Item keys count in the hundreds of thousands per fingerprint — more bits.
ITEM_KEY_LENGTH = 24


def fingerprint(prompt, templates, model, mode, temperature, max_tokens,
                thinking):
    """Version id for a (prompt, config, model) combination.

    `templates` is the OCR suffix the mode appends to the prompt — part of
    what the model reads, hence part of the version.
    """
    payload = json.dumps(
        {"prompt": prompt, "templates": templates, "model": model,
         "mode": mode, "temperature": temperature, "max_tokens": max_tokens,
         "thinking": thinking},
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


def item_key(image_b64, ocr):
    """Key for one judged input: what the model was shown, nothing else.

    Content-addressed on purpose: the same crop exported under a new run name
    still hits, and a re-rendered crop with different pixels misses.
    """
    if not image_b64 and not ocr:
        raise ValueError("item_key needs an image, OCR text or both")
    h = hashlib.sha256()
    if image_b64:
        h.update(image_b64.encode("ascii"))
    if ocr:
        h.update(json.dumps(ocr, ensure_ascii=False,
                            sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:ITEM_KEY_LENGTH]


def _item_path(cache_dir, key):
    return os.path.join(cache_dir, f"{key}.json")


def _write_atomic(path, data):
    # Unique temp name: concurrent judge runs may share a cache folder, and
    # writing the same answer twice must stay harmless.
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def write_meta(cache_dir, meta):
    """Write meta.json describing the fingerprint, first run only."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "meta.json")
    if os.path.isfile(path):
        return
    _write_atomic(path, {"version": CACHE_VERSION, **meta})


def read_cache(cache_dir, key):
    """Cached raw model answer for an input, or None if missing or invalid."""
    path = _item_path(cache_dir, key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("version") != CACHE_VERSION:
        return None
    raw = data.get("raw")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw


def write_cache(cache_dir, key, utsnitt, raw, seconds):
    """Write one judged answer. utsnitt and seconds are for humans only."""
    os.makedirs(cache_dir, exist_ok=True)
    _write_atomic(_item_path(cache_dir, key),
                  {"version": CACHE_VERSION, "utsnitt": utsnitt,
                   "seconds": round(seconds, 2), "raw": raw})
