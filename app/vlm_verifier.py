"""VLM verifier inside the pipeline: judges proposed sladdebokser, drops «nei».

Off unless it is turned on (VLM_ENABLED / --vlm / ?vlm=true). When on, every
box in the stratum is cropped with a red frame around it, sent to the model
one box at a time, and dropped only on a clear «nei». Everything else — «ja»,
a timeout, an unparsable answer, a missing endpoint — keeps the box, so the
verifier can only remove sladdinger, never add or move one.

Two guards sit in front of the «nei»:

  stratum   Only kilder in VLM_SOURCES, and only in documents that get no rule
            profile. A box the rules already handle is not worth the GPU.
  fnr guard PaddleOCR's line and the model's own transcription are re-read by
            find_fnr. A valid 11-digit run, or a fnr ledetekst next to a
            five-digit run, overrules the «nei». The model reads better than
            it infers, so the code does the inferring.

The crop geometry mirrors utils/vlm_export.py, or the model would see other
images in prod than in the runs it was measured on.
"""

import base64
import io
import os
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

from config import (PDF_DPI, VLM_API_KEY, VLM_CONCURRENT, VLM_ENABLED,
                    VLM_MARGIN_PT, VLM_MAX_PX, VLM_MAX_TOKENS, VLM_MODEL,
                    VLM_SOURCES, VLM_TIMEOUT, VLM_URL)
import vlm_cache
from vlm_client import (STD_PROMPT, _THINKING, _build_melding, call_model,
                        fnr_protects, parse_answer)

SCALE = PDF_DPI / 72.0            # PDF points -> pixels
MARKER = (230, 20, 20)            # red frame around the box being judged
MARKER_WIDTH = 3                  # px, grown with the crop width


class VlmConfig:
    """What the verifier needs to run. `urls` may hold several backends.

    Built once per process (from_env or from the run.py flags) and passed
    down through run_model_on_pdf_bytes, so nothing reads the environment
    per document.
    """

    def __init__(self, urls, model, timeout=VLM_TIMEOUT,
                 concurrent=VLM_CONCURRENT, max_tokens=VLM_MAX_TOKENS,
                 api_key=VLM_API_KEY, sources=VLM_SOURCES,
                 margin_pt=VLM_MARGIN_PT, max_px=VLM_MAX_PX, cache_dir=None,
                 prompt=STD_PROMPT, thinking="none"):
        self.urls = [urls] if isinstance(urls, str) else list(urls)
        self.model = model
        self.timeout = timeout
        self.concurrent = max(1, int(concurrent))
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.sources = frozenset(sources)
        self.margin_pt = margin_pt
        self.max_px = max_px
        self.prompt = prompt
        # Sent AND fingerprinted, so a cached answer is never one the server
        # produced under a different setting.
        self.thinking = thinking
        self.cache_dir = None
        if cache_dir:
            fp = vlm_cache.fingerprint(prompt, "", model, "image", 0.0,
                                       max_tokens, thinking)
            self.cache_dir = os.path.join(cache_dir, fp)

    @property
    def fingerprint(self):
        return os.path.basename(self.cache_dir) if self.cache_dir else None


def config_from_env():
    """The prod switch: VLM_ENABLED plus a URL and a model, or None."""
    if not VLM_ENABLED or not VLM_URL or not VLM_MODEL:
        return None
    return VlmConfig([u.strip() for u in VLM_URL.split(",") if u.strip()],
                     VLM_MODEL)


def _crop(image, box, margin_px, max_px):
    """Crop around the box with a red frame on it, as PNG bytes.

    `image` is the page as the OCR saw it (already rotated), `box` its
    coordinates in that same pixel space. Returns None when the box falls
    outside the page or has no area left after clipping — the caller then
    keeps the box unjudged.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    x0, y0, x1, y1 = box
    if x1 <= 0 or y1 <= 0 or x0 >= image.width or y0 >= image.height:
        return None
    left = max(0, int(x0 - margin_px))
    top = max(0, int(y0 - margin_px))
    right = min(image.width, int(x1 + margin_px))
    bottom = min(image.height, int(y1 + margin_px))
    if right <= left or bottom <= top:
        return None

    ut = image.crop((left, top, right, bottom)).convert("RGB")
    m = [x0 - left, y0 - top, x1 - left, y1 - top]
    if max_px and ut.width > max_px:
        f = max_px / ut.width
        ut = ut.resize((max_px, max(1, int(ut.height * f))), Image.LANCZOS)
        m = [v * f for v in m]
    # A 3 px frame disappears in a wide crop; the padding grows with the line
    # so the digits stay visible.
    stroke = max(MARKER_WIDTH, round(ut.width / 400))
    pad = stroke + 2
    m = [max(0, m[0] - pad), max(0, m[1] - pad),
         min(ut.width - 1, m[2] + pad), min(ut.height - 1, m[3] + pad)]
    if m[2] <= m[0] or m[3] <= m[1]:
        return None
    ImageDraw.Draw(ut).rectangle(m, outline=MARKER, width=stroke)

    buffer = io.BytesIO()
    ut.save(buffer, format="PNG")
    return buffer.getvalue()


def _line_text(lines, box):
    """The text of the OCR line the box sits on, for the fnr guard.

    `lines` is build_lines(tokens). Picks the line with the largest vertical
    overlap; empty when the box holds no text at all, which is normal for
    boxes YOLO found in map graphics.
    """
    x0, y0, x1, y1 = box
    best, best_overlap = "", 0.0
    for entry in lines or ():
        tokens = entry[0]
        if not tokens:
            continue
        top = min(t.y0 for t in tokens)
        bottom = max(t.y1 for t in tokens)
        overlap = min(y1, bottom) - max(y0, top)
        if overlap > best_overlap:
            best_overlap, best = overlap, entry[1]
    return best


def _judge(crop_png, a, i):
    """One box -> «ja» or «nei», plus the model's transcription.

    Never raises, and never answers «nei» on a failure: the caller must be
    able to keep the box whatever goes wrong.
    """
    image_b64 = base64.b64encode(crop_png).decode("ascii")
    key = vlm_cache.item_key(image_b64, None) if a.cache_dir else None
    if key:
        try:
            cached = vlm_cache.read_cache(a.cache_dir, key)
        except OSError:
            cached = None
        if cached is not None:
            answer, number, _rationale, parse_error, check = parse_answer(cached)
            if not parse_error:
                return answer, number, check.get("linjen", "")

    url = a.urls[i % len(a.urls)]
    messages = _build_melding(a.prompt, image_b64)
    try:
        raw = call_model(url, a.model, messages, api_key=a.api_key,
                         timeout=a.timeout, temperature=0.0,
                         max_tokens=a.max_tokens,
                         thinking=_THINKING["value"])
    except urllib.error.HTTPError as e:
        # An endpoint that does not know reasoning_effort answers 400 to every
        # box. Without this the verifier would answer «ja» to everything and
        # quietly do nothing at all.
        if not (e.code == 400 and _THINKING["value"]
                and "reasoning" in str(e).lower()):
            return "ja", "", ""
        _THINKING["value"] = None
        try:
            raw = call_model(url, a.model, messages, api_key=a.api_key,
                             timeout=a.timeout, temperature=0.0,
                             max_tokens=a.max_tokens, thinking=None)
        except Exception:
            return "ja", "", ""
    except Exception:
        return "ja", "", ""
    answer, number, _rationale, parse_error, check = parse_answer(raw)
    # Once the field has been dropped the answers no longer match the
    # fingerprint, so they stay out of the cache.
    if key and not parse_error and _THINKING["value"] == a.thinking:
        try:
            vlm_cache.write_cache(a.cache_dir, key, "", raw, 0.0)
        except OSError:
            pass
    return answer, number, check.get("linjen", "")


def in_stratum(source, a, koordfam=False, seksjonering=False):
    """Whether this box is one the verifier is allowed to judge.

    Documents that get a rule profile are left alone: the profiles already
    remove the oversladding there, and the measured gain sits in the boxes no
    rule touches.
    """
    if koordfam or seksjonering:
        return False
    return source in a.sources


def verify_page(boxes_with_source, image, lines, a, koordfam=False,
                seksjonering=False):
    """Judge one page's boxes and return the survivors.

    `boxes_with_source` is the internal per-box list from model_main,
    `image` the page as the OCR saw it. Boxes outside the stratum are
    returned untouched and never reach the model.
    """
    if not boxes_with_source or image is None:
        return boxes_with_source, 0, 0

    margin_px = a.margin_pt * SCALE
    tasks = []
    for i, pair in enumerate(boxes_with_source):
        if not in_stratum(pair[1], a, koordfam, seksjonering):
            continue
        crop_png = _crop(image, pair[0], margin_px, a.max_px)
        if crop_png is not None:
            tasks.append((i, crop_png))
    if not tasks:
        return boxes_with_source, 0, 0

    with ThreadPoolExecutor(max_workers=min(a.concurrent, len(tasks))) as pool:
        verdicts = list(pool.map(
            lambda t: _judge(t[1], a, t[0]), tasks))

    dropped = set()
    for (i, _crop_png), (answer, number, own_line) in zip(tasks, verdicts):
        if answer != "nei":
            continue
        if fnr_protects([number, own_line, _line_text(lines, boxes_with_source[i][0])]):
            continue
        dropped.add(i)

    survivors = [pair for i, pair in enumerate(boxes_with_source)
                 if i not in dropped]
    return survivors, len(tasks), len(dropped)
