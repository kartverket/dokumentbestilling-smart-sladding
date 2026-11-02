import math
import os
import re
import statistics
from collections import namedtuple

import numpy as np

from config import SLADD_DIGITS, PAD_X_FACTOR, PAD_Y_FACTOR, MAX_HEIGHT_FACTOR, PADDLE_MODEL_SET, DET_PAGE_LEN, REC_BATCH, PAGES_PER_OCR_BATCH, PDF_DPI

MOD11_WEIGHTS_1 = [3, 7, 6, 1, 8, 9, 4, 5, 2]
MOD11_WEIGHTS_2 = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]

_OCR_DIGIT_MAP = str.maketrans("oOsSlIbB", "00551166")


def _normalize_ocr(text):
    return text.translate(_OCR_DIGIT_MAP)

Token = namedtuple("Token", ["text", "x0", "y0", "x1", "y1", "rec_score"])
DigitBox = namedtuple("DigitBox", ["left", "right", "top", "bottom", "rec_score"])
Hit = namedtuple("Hit", ["start", "end"])


_NAME = {
    "v5": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    "v6": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
}
DET_MODEL, REC_MODEL = _NAME[PADDLE_MODEL_SET]
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DET_MODEL_DIR = os.path.join(_MODEL_DIR, DET_MODEL + "_infer")
REC_MODEL_DIR = os.path.join(_MODEL_DIR, REC_MODEL + "_infer")


reader = None


def _has_gpu():
    try:
        import paddle
        return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def fetch_reader():
    """The project's shared PaddleOCR reader, built on first use."""
    global reader
    if reader is None:
        # Imported here, not at the top: cache readers use only the pure text
        # functions and should not pay for the paddleocr import.
        from paddleocr import PaddleOCR

        gpu = _has_gpu()
        print(f"GPU available: {gpu}")

        kwargs = dict(
            lang="en",
            device="gpu" if gpu else "cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_type="max",
            text_det_limit_side_len=DET_PAGE_LEN,
            # A larger rec batch on GPU gives better throughput.
            # SLADD_REC_BATCH can lower it when several processes share the
            # card; batch size affects memory and speed, not the result.
            text_recognition_batch_size=int(
                os.environ.get("SLADD_REC_BATCH")
                or (REC_BATCH * 2 if gpu else REC_BATCH)),
        )
        kwargs["text_detection_model_name"] = DET_MODEL
        kwargs["text_recognition_model_name"] = REC_MODEL
        kwargs["text_detection_model_dir"] = DET_MODEL_DIR
        kwargs["text_recognition_model_dir"] = REC_MODEL_DIR
        if gpu:
            kwargs["precision"] = "fp16"
        else:
            # PaddleX enables oneDNN by default, and 3.3.1 crashes in that path.
            kwargs["enable_mkldnn"] = False
        if os.environ.get("SLADD_HPI") == "1":
            kwargs["enable_hpi"] = True

        reader = PaddleOCR(**kwargs)
    return reader


def _check_digit(weights, digits):
    rest = sum(weight * number for weight, number in zip(weights, digits)) % 11
    return (11 - rest) % 11


def valid_mod11(number_str):
    if len(number_str) != 11 or not number_str.isdigit():
        return False
    digits = [int(c) for c in number_str]
    check_1 = _check_digit(MOD11_WEIGHTS_1, digits[:9])
    check_2 = _check_digit(MOD11_WEIGHTS_2, digits[:10])
    return check_1 == digits[9] and check_2 == digits[10]


def er_fnr_form(number_str):
    "Accepts both fnr (day 01-31) and d-nummer (day 41-71)."
    if len(number_str) != 11 or not number_str.isdigit():
        return False
    day = int(number_str[0:2])
    month = int(number_str[2:4])
    if 41 <= day <= 71:          # d-nummer: 40 added to the day
        day -= 40
    return 1 <= day <= 31 and 1 <= month <= 12


def find_fnr(text, require_mod11=True):
    """11-digit runs in the text that look like an fnr/d-nummer.

    krev_mod11=False drops the check-digit test and yields *candidates*: runs
    with the right shape but not necessarily a valid number. box_features uses
    it to ask "is there an 11-digit number here at all?". A single misread
    digit breaks mod11, so shape alone is the right question when the point is
    to reject boxes that CANNOT be an fnr.
    """
    norm = _normalize_ocr(text)
    pos = [i for i, ch in enumerate(norm) if ch.isdigit()]
    hits, i = [], 0
    while i + 11 <= len(pos):
        start, stop = pos[i], pos[i + 10] + 1
        between = norm[start:stop]
        digit_str = re.sub(r"\D", "", between)
        luker = re.findall(r"\D+", between)
        ok = (
            len(luker) <= 3                                   # OCR may split the fnr into pieces
            and all(set(g) <= set(" .-,_") for g in luker)
            and all(len(g) <= 2 for g in luker)
            and er_fnr_form(digit_str)
            and (not require_mod11 or valid_mod11(digit_str))
        )
        if ok:
            hits.append(Hit(start, stop))
            i += 11                                           # skip the whole fnr
        else:
            i += 1                                            # slide one digit on
    return hits


def _read_tokens(res):
    tokens = []
    if not res:
        return tokens

    word_per_line = res.get("text_word")
    box_per_line = res.get("text_word_boxes")
    if word_per_line and box_per_line:
        scores_per_line = res.get("text_word_scores") or []
        # PaddleOCR 3.7+ does not always give per-word scores, but does give
        # line-level rec_scores. Fall back to the line score for every word.
        line_rec_scores = res.get("rec_scores") or []
        for line_idx, (word_names, box_names) in enumerate(zip(word_per_line, box_per_line)):
            line_scores = scores_per_line[line_idx] if line_idx < len(scores_per_line) else []
            fallback_rec = float(line_rec_scores[line_idx]) if line_idx < len(line_rec_scores) else None
            for ord_idx, (text, box) in enumerate(zip(word_names, box_names)):
                if not text.strip():
                    continue
                if ord_idx < len(line_scores):
                    rec_score = float(line_scores[ord_idx])
                else:
                    rec_score = fallback_rec
                x0, y0, x1, y1 = (float(v) for v in np.asarray(box).reshape(-1)[:4])
                tokens.append(Token(text, min(x0, x1), min(y0, y1),
                                    max(x0, x1), max(y0, y1), rec_score))
        if tokens:
            return tokens
    # Fallback: line level (four corner points per box).
    texts = res.get("rec_texts") or []
    polys = res.get("rec_polys")
    if polys is None:
        polys = res.get("dt_polys") or []
    scores = res.get("rec_scores") or []
    for idx, (text, poly) in enumerate(zip(texts, polys)):
        rec_score = float(scores[idx]) if idx < len(scores) else None
        pts = np.asarray(poly, dtype=float)
        tokens.append(Token(text, float(pts[:, 0].min()), float(pts[:, 1].min()),
                            float(pts[:, 0].max()), float(pts[:, 1].max()), rec_score))
    return tokens


def _groups_to_lines(tokens):
    # The line's min-y0/max-y1 are kept running: recomputing them per
    # membership test is quadratic on token-heavy pages.
    lines = []                 # [tokens, min_y0, max_y1] per line
    for token in sorted(tokens, key=lambda t: ((t.y0 + t.y1) / 2, t.x0)):
        center_y = (token.y0 + token.y1) / 2
        placed = False
        for line in lines:
            if line[1] <= center_y <= line[2]:
                line[0].append(token)
                if token.y0 < line[1]:
                    line[1] = token.y0
                if token.y1 > line[2]:
                    line[2] = token.y1
                placed = True
                break
        if not placed:
            lines.append([[token], token.y0, token.y1])
    return [line[0] for line in lines]


def _build_line_text(line):
    chars, map_ = [], []
    for token_nr, token in enumerate(sorted(line, key=lambda t: t.x0)):
        if token_nr > 0:
            chars.append(" ")
            map_.append(None)
        width = token.x1 - token.x0
        count = len(token.text)
        for position, ch in enumerate(token.text):
            chars.append(ch)
            if _normalize_ocr(ch).isdigit():
                left = token.x0 + width * position / count
                right = token.x0 + width * (position + 1) / count
                map_.append(DigitBox(left, right, token.y0, token.y1, token.rec_score))
            else:
                map_.append(None)
    return "".join(chars), map_


def build_lines(tokens):
    """Text lines on one page: [(tokens, text, map)].

    Split out because box_features needs the same lines per YOLO box, and
    grouping per box would cost O(boxes x tokens) instead of once per page.
    """
    ut = []
    for line in _groups_to_lines(tokens):
        text, map_ = _build_line_text(line)
        ut.append((line, text, map_))
    return ut


def _sladd_box(digit_boxes):
    if len(digit_boxes) <= SLADD_DIGITS:
        return None
    last = digit_boxes[-SLADD_DIGITS:]            # the 5 to cover
    anchor = digit_boxes[-SLADD_DIGITS - 1]         # the digit before, NOT covered

    median_width = statistics.median(b.right - b.left for b in last)

    top = statistics.median(b.top for b in digit_boxes)
    bottom = statistics.median(b.bottom for b in digit_boxes)

    mx = PAD_X_FACTOR * median_width
    my = PAD_Y_FACTOR * (bottom - top)

    limit = (anchor.right + last[0].left) / 2
    left = max(limit - mx, (anchor.left + anchor.right) / 2)

    right = max(b.right for b in last) + mx
    top -= my
    bottom += my

    cap = MAX_HEIGHT_FACTOR * median_width
    if (bottom - top) > cap:
        center = (top + bottom) / 2
        top, bottom = center - cap / 2, center + cap / 2

    return (math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom))



# A real fnr has separators at FIXED positions: the date periods after digit
# 2 and 4 ("01.01.50") and the field separator after digit 6 ("010150 12345").
# Gaps there prove nothing; coordinate seams put their gaps elsewhere.
_LEGAL_GAP_POS = frozenset((2, 4, 6))


def _window_features(window, digit_boxes):
    """Features of the 11-digit window a sladd box was built from.

    Coordinate columns ("6626630.58 549810.29") get stitched into a window
    across decimal points and column gaps: the gap rules allow it, and the
    line text has exactly ONE space between tokens whatever the physical
    distance, so even numbers at opposite ends of a sketch are sewn together.

      maks_luke        largest distance between neighbouring digits outside a
                       legal position, in median digit widths
      har_desimal_luke 1 if a gap with "." or "," sits outside a legal position
    """
    has_decimal = 0
    pos = 0                      # digits read before this character
    for ch in window:
        if ch.isdigit():
            pos += 1
        elif ch in ".," and pos not in _LEGAL_GAP_POS:
            has_decimal = 1
    widths = sorted(b.right - b.left for b in digit_boxes)
    median = widths[len(widths) // 2] or 1.0
    gap = max((digit_boxes[j + 1].left - digit_boxes[j].right
               for j in range(len(digit_boxes) - 1)
               if (j + 1) not in _LEGAL_GAP_POS), default=0.0)
    return {"maks_luke": round(max(gap, 0.0) / median, 2),
            "har_desimal_luke": has_decimal}


def sladd_boxes_from_tokens(tokens, lines=None):
    """Sladd boxes from OCR tokens.

    `lines` can be passed in from build_lines when the caller has already
    grouped the page, so grouping is not done twice.
    """
    boxes = []
    for _line, text, map_ in (lines if lines is not None else build_lines(tokens)):
        for hit in find_fnr(text):
            digit_boxes = [map_[i] for i in range(hit.start, hit.end) if map_[i] is not None]
            box = _sladd_box(digit_boxes)
            if box is None:
                continue
            window = _normalize_ocr(text[hit.start:hit.end])
            digit_str = re.sub(r"\D", "", window)
            rec_scores = [sb.rec_score for sb in digit_boxes if sb.rec_score is not None]
            rec_score = round(min(rec_scores), 3) if rec_scores else None
            boxes.append((box, valid_mod11(digit_str), rec_score,
                           _window_features(window, digit_boxes)))
    return boxes


def lines_with_fnr_marks(tokens):
    lines_out = []
    lines = _groups_to_lines(tokens)
    for line in sorted(lines, key=lambda l: min(t.y0 for t in l)):
        text, _map = _build_line_text(line)
        text = text.strip()
        if not text:
            continue
        merker = []
        for tr in find_fnr(text):
            digit_str = re.sub(r"\D", "", _normalize_ocr(text[tr.start:tr.end]))
            merker.append((digit_str, valid_mod11(digit_str)))
        lines_out.append((text, merker))
    return lines_out


def read_tokens_batched(images, batch_size=None):
    reader = fetch_reader()
    chunk_size = batch_size or PAGES_PER_OCR_BATCH

    tokens_per_page = []
    for start in range(0, len(images), chunk_size):
        chunk = images[start:start + chunk_size]
        bgr_chunk = [np.ascontiguousarray(b[:, :, ::-1]) for b in chunk]
        results = reader.predict(bgr_chunk, return_word_box=True) or []
        for res in results:
            tokens_per_page.append(_read_tokens(res))
        while len(tokens_per_page) < start + len(chunk):
            tokens_per_page.append([])

    return tokens_per_page