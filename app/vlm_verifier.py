"""VLM verifier inside the pipeline: judges proposed sladdebokser, drops «nei».

Off unless it is turned on (VLM_ENABLED / --vlm / ?vlm=true). When on, every
box in the stratum is cropped with a red frame around it, sent to the model
one box at a time, and dropped only on a clear «nei». Everything else — «ja»,
a timeout, an unparsable answer, a missing endpoint — keeps the box, so the
verifier can only remove sladdinger, never add or move one.

Two guards sit in front of the «nei»:

  stratum   Only kilder in VLM_SOURCES reach the model.
  fnr guard PaddleOCR's line and the model's own transcription are re-read by
            find_fnr. A valid 11-digit run, or a fnr ledetekst next to a
            five-digit run, overrules the «nei». The model reads better than
            it infers, so the code does the inferring.

The crop itself is vlm_client.crop_with_marker, shared with
utils/vlm_export.py, and the geometry is the VLM_MARGIN_*/VLM_MAX_PX block in
config, which the export CLI defaults to as well: if prod and the export drift
apart the model sees other images in prod than in the runs it was measured on.
"""

import base64
import io
import os
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

from config import (PDF_DPI, VLM_API_KEY, VLM_BREAKER_COOLDOWN,
                    VLM_BREAKER_FAILURES, VLM_CACHE, VLM_CONCURRENT, VLM_ENABLED,
                    VLM_MARGIN_DOWN_PT, VLM_MARGIN_LEFT_PT,
                    VLM_MARGIN_RIGHT_PT, VLM_MARGIN_UP_PT, VLM_MAX_PX,
                    VLM_MAX_TOKENS, VLM_MODEL, VLM_SOURCES, VLM_TIMEOUT,
                    VLM_URL)
import vlm_cache
from vlm_client import (FULL, STD_PROMPT, _THINKING, _build_melding,
                        call_model, crop_with_marker, fnr_protects,
                        parse_answer)

SCALE = PDF_DPI / 72.0            # PDF points -> pixels

# Consecutive failures, and the clock until calls resume. Per process: each
# gunicorn worker finds out on its own, which is cheaper than sharing state.
_BREAKER = {"failures": 0, "open_until": 0.0}
_BREAKER_LOCK = threading.Lock()


def _breaker_open():
    """True while the verifier is backing off from a failing endpoint."""
    if not VLM_BREAKER_FAILURES:
        return False
    with _BREAKER_LOCK:
        return time.monotonic() < _BREAKER["open_until"]


def _breaker_note(ok):
    """One call's outcome. A success anywhere closes the breaker again."""
    if not VLM_BREAKER_FAILURES:
        return
    with _BREAKER_LOCK:
        if ok:
            _BREAKER["failures"] = 0
            _BREAKER["open_until"] = 0.0
            return
        _BREAKER["failures"] += 1
        if _BREAKER["failures"] >= VLM_BREAKER_FAILURES:
            _BREAKER["open_until"] = time.monotonic() + VLM_BREAKER_COOLDOWN
            _BREAKER["failures"] = 0


# parse_answer's «feil» texts, shortened for the run report
_SHORT = {"not JSON, read by keyword": "answer was not JSON",
          "unparsable answer": "answer unreadable",
          "empty answer": "answer empty",
          "the model omitted the «svar» field": "no «svar» field"}


class VlmConfig:
    """What the verifier needs to run. `urls` may hold several backends.

    Built once per process (from_env or from the run.py flags) and passed
    down through run_model_on_pdf_bytes, so nothing reads the environment
    per document.
    """

    def __init__(self, urls, model, timeout=VLM_TIMEOUT,
                 concurrent=VLM_CONCURRENT, max_tokens=VLM_MAX_TOKENS,
                 api_key=VLM_API_KEY, sources=VLM_SOURCES,
                 margin_up_pt=VLM_MARGIN_UP_PT,
                 margin_down_pt=VLM_MARGIN_DOWN_PT,
                 margin_left_pt=VLM_MARGIN_LEFT_PT,
                 margin_right_pt=VLM_MARGIN_RIGHT_PT,
                 max_px=VLM_MAX_PX, cache_dir=None,
                 prompt=STD_PROMPT, thinking="none"):
        self.urls = [urls] if isinstance(urls, str) else list(urls)
        self.model = model
        self.timeout = timeout
        self.concurrent = max(1, int(concurrent))
        self.max_tokens = max_tokens
        self.api_key = api_key
        # A bare string would dissolve into letters and empty the stratum.
        self.sources = frozenset((sources,) if isinstance(sources, str)
                                 else sources)
        self.margin_up_pt = margin_up_pt
        self.margin_down_pt = margin_down_pt
        self.margin_left_pt = margin_left_pt
        self.margin_right_pt = margin_right_pt
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
                     VLM_MODEL, cache_dir=VLM_CACHE or None)


def _px(margin_pt):
    """A margin in points as pixels, leaving FULL alone."""
    return margin_pt if margin_pt == FULL else margin_pt * SCALE


def _crop(image, box, a):
    """The crop the model is sent, as PNG bytes.

    `image` is the page as the OCR saw it (already rotated), `box` its
    coordinates in that same pixel space. The geometry comes from the config
    `a`, so it is the same one utils/vlm_export.py cuts with. Returns None
    when the box falls outside the page or has no area left after clipping —
    the caller then keeps the box unjudged.
    """
    ut, _marker = crop_with_marker(
        image, box,
        margin_up=_px(a.margin_up_pt), margin_down=_px(a.margin_down_pt),
        margin_left=_px(a.margin_left_pt), margin_right=_px(a.margin_right_pt),
        max_px=a.max_px)
    if ut is None:
        return None
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
    """One box -> «ja» or «nei», the model's transcription, and a cache note.

    The note is «hit», «written», «off», or the reason the answer was not
    stored. An answer the cache refuses is judged again on every run, so the
    reason has to reach the caller instead of being dropped here.

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
                return answer, number, check.get("linjen", ""), "hit"

    if _breaker_open():
        return "ja", "", "", "breaker open"

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
            _breaker_note(False)
            return "ja", "", "", "call failed"
        _THINKING["value"] = None
        try:
            raw = call_model(url, a.model, messages, api_key=a.api_key,
                             timeout=a.timeout, temperature=0.0,
                             max_tokens=a.max_tokens, thinking=None)
        except Exception:
            _breaker_note(False)
            return "ja", "", "", "call failed"
    except Exception:
        _breaker_note(False)
        return "ja", "", "", "call failed"
    _breaker_note(True)
    answer, number, _rationale, parse_error, check = parse_answer(raw)
    note = "off" if not key else ""
    if key:
        # Once the field has been dropped the answers no longer match the
        # fingerprint, so they stay out of the cache.
        if parse_error:
            note = _SHORT.get(parse_error.split("—")[0].strip(), "unparsed")
        elif _THINKING["value"] != a.thinking:
            note = "reasoning_effort dropped"
        else:
            note = "written"
            try:
                vlm_cache.write_cache(a.cache_dir, key, "", raw, 0.0)
            except OSError:
                note = "cache not writable"
    return answer, number, check.get("linjen", ""), note


def in_stratum(source, a):
    """Whether this box is one the verifier is allowed to judge.

    Rule profiles change the rules, not the verifier: it runs after them,
    judges every kilde in VLM_SOURCES and can only remove.
    """
    return source in a.sources


def needs_image(boxes_with_source, a):
    """Whether the page holds a box the verifier may judge.

    The caller asks before rasterising: a page with nothing in the stratum
    never needs an image.
    """
    return any(in_stratum(pair[1], a) for pair in boxes_with_source)


def verify_page(boxes_with_source, image, lines, a, stats=None):
    """Judge one page's boxes and return the survivors.

    `boxes_with_source` is the internal per-box list from model_main,
    `image` the page as the OCR saw it. Boxes outside the stratum are
    returned untouched and never reach the model.

    stats  an optional dict the counters are added to: judged, dropped,
             cache_hits, and the same split per kilde. The caller owns the
             dict and can accumulate over pages and documents.
    """
    if not boxes_with_source or image is None:
        return boxes_with_source, 0, 0

    tasks = []
    for i, pair in enumerate(boxes_with_source):
        if not in_stratum(pair[1], a):
            continue
        crop_png = _crop(image, pair[0], a)
        if crop_png is not None:
            tasks.append((i, crop_png))
    if not tasks:
        return boxes_with_source, 0, 0

    with ThreadPoolExecutor(max_workers=min(a.concurrent, len(tasks))) as pool:
        verdicts = list(pool.map(
            lambda t: _judge(t[1], a, t[0]), tasks))

    dropped = set()
    notes = []
    for (i, _crop_png), (answer, number, own_line, note) in zip(tasks, verdicts):
        notes.append(note)
        if answer != "nei":
            continue
        if fnr_protects([number, own_line, _line_text(lines, boxes_with_source[i][0])]):
            continue
        dropped.add(i)

    if stats is not None:
        _count(stats, tasks, dropped, notes, boxes_with_source)

    survivors = [pair for i, pair in enumerate(boxes_with_source)
                 if i not in dropped]
    return survivors, len(tasks), len(dropped)


def _count(stats, tasks, dropped, notes, boxes_with_source):
    """Add one page's verdicts to the caller's counter dict."""
    stats["judged"] = stats.get("judged", 0) + len(tasks)
    stats["dropped"] = stats.get("dropped", 0) + len(dropped)
    stats["cache_hits"] = (stats.get("cache_hits", 0)
                           + sum(1 for n in notes if n == "hit"))
    not_cached = stats.setdefault("not_cached", {})
    for note in notes:
        if note in ("hit", "written", "off"):
            continue
        not_cached[note] = not_cached.get(note, 0) + 1
    judged_per_kilde = stats.setdefault("judged_per_kilde", {})
    dropped_per_kilde = stats.setdefault("dropped_per_kilde", {})
    for i, _crop_png in tasks:
        source = boxes_with_source[i][1]
        judged_per_kilde[source] = judged_per_kilde.get(source, 0) + 1
        if i in dropped:
            dropped_per_kilde[source] = dropped_per_kilde.get(source, 0) + 1
