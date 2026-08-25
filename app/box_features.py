"""
Features describing what Paddle read in and around a YOLO box.

They exist so stricter variants of lenient_check can be swept in the result
CSV instead of rerunning the pipeline. That is valid because _godta_yolo_boks
runs only for YOLO boxes with no overlapping Paddle box, and
MIN_DIGITS=1 / MAX_LETTERS=1 is the loosest possible setting, so anything a
stricter variant would keep is already in the CSV.

Computed for kilde "yolo" only. Paddle and "begge" boxes have a mod11-valid
number behind them and get empty features, so a global sweep filter can never
hit them by accident; "yolo_vertikal" reads no tokens and the line logic below
assumes horizontal text.

Features whose meaning is not obvious from the name:

    har_tokens        1 if Paddle read anything. Text-less boxes are governed
                      by YOLO_CONF_NO_TEXT, not lenient_check, and must stay
                      out of the digit rules.
    siffer_run        longest contiguous digit run overlapping the box
    har_fnr_kandidat  1 if an 11-digit run with valid fnr SHAPE (no mod11)
                      overlaps the box
    har_desimal_naer  1 if a decimal separator overlaps the box. Coordinates
                      have one, an fnr never does
    har_00_run        1 if a 10-12 digit run overlapping the box starts with
                      "00": orgnr are zero-padded to 11 digits in the
                      grunnbok, and day 00 is invalid in an fnr
    har_orgnr         1 if a 9-digit run (or the last 9 of a 00-padded
                      11-run) starts with 8/9 and passes orgnr mod11
    har_org_ord       1 if the box's line or its neighbours contain a company
                      form word (AS, Borettslag, Sameie, Org.nr, ...)
    lang_run          longest digit run WITH gaps (find_fnr's gap rules). A
                      real fnr sladd covers the last 5 of an 11-run; covering
                      the end of a 6-digit run instead cannot be an fnr

har_fnr_kandidat alone does not separate coordinates from fnr: find_fnr
accepts "." and "," as gaps, so on a coordinate line it stitches two
neighbouring numbers into a run of valid fnr shape. Use it with
har_desimal_naer.
"""

import re
import statistics

from config import FEATURE_FIELDS                                   # noqa: F401, re-export
from paddle_ocr_model_fnr import find_fnr
from yolo_fnr import count_digits_and_letters, tokens_in_box

_DECIMAL = re.compile(r"\d[.,]\d")

_GAP_DRAW = set(" .-,_")

# Short company-form acronyms need uppercase and a word boundary; DA/SA/BA are
# left out deliberately: they collide with ordinary words in all-caps text.
# The long words match case-insensitively and without a word boundary, so
# "Sparebanken" and "borettslaget" hit.
_ORG_ORD = re.compile(
    r"\b(AS|ASA|ANS|KS|NUF|IKS|BBL|BRL)\b"
    r"|(?i:borettslag|boligbyggelag|sameie|stiftelse|forening|kommune"
    r"|sparebank|bank|forsikring|org\.?\s*nr|organisasjonsn|foretak)")


def _overlap_area(t, box):
    ix0, iy0 = max(t.x0, box[0]), max(t.y0, box[1])
    ix1, iy1 = min(t.x1, box[2]), min(t.y1, box[3])
    return (ix1 - ix0) * (iy1 - iy0) if (ix1 > ix0 and iy1 > iy0) else 0.0


def _select_line(lines, box):
    """The line the box sits on: the one with the largest token overlap."""
    best, best_ov = None, 0.0
    for post in lines:
        ov = sum(_overlap_area(t, box) for t in post[0])
        if ov > best_ov:
            best, best_ov = post, ov
    return best


def _spans_over(map_, start, stop, box):
    """Do the digit boxes in [start, slutt) overlap the box horizontally?"""
    digits = [map_[i] for i in range(start, stop) if i < len(map_) and map_[i] is not None]
    if not digits:
        return False
    left = min(s.left for s in digits)
    right = max(s.right for s in digits)
    return right > box[0] and left < box[2]


def _digits_run(text, map_, box):
    """Longest contiguous digit run in the text overlapping the box.

    An fnr is sladdet on the last 5 of 11 digits, so a bare 5-7 digit run,
    a coordinate, is distinguishable from a real hit.
    """
    longest = 0
    for m in re.finditer(r"\d+", text):
        if _spans_over(map_, m.start(), m.end(), box):
            longest = max(longest, m.end() - m.start())
    return longest


def _has_decimal(text, map_, box):
    for m in _DECIMAL.finditer(text):
        if _spans_over(map_, m.start(), m.end(), box):
            return 1
    return 0


def _number_runs(text):
    """Maximal digit runs, allowing gaps of <=2 characters from " .-,_".

    find_fnr's gap rules without the shape requirement: which contiguous
    NUMBER is written there ("00 987 654 321" is one run), not whether it
    could be an fnr. Returns (start, slutt, cifre) tuples.
    """
    runs = []
    i, n = 0, len(text)
    while i < n:
        if not text[i].isdigit():
            i += 1
            continue
        start, stop = i, i + 1
        j = i + 1
        while j < n:
            if text[j].isdigit():
                stop = j + 1
                j += 1
            else:
                k = j
                while k < n and not text[k].isdigit():
                    k += 1
                if k < n and k - j <= 2 and set(text[j:k]) <= _GAP_DRAW:
                    j = k
                else:
                    break
        runs.append((start, stop, re.sub(r"\D", "", text[start:stop])))
        i = stop
    return runs


def valid_orgnr_mod11(digit_str):
    """Check digit of a 9-digit orgnr (Brønnøysund)."""
    if len(digit_str) != 9 or not digit_str.isdigit():
        return False
    weights = (3, 2, 7, 6, 5, 4, 3, 2)
    rest = sum(v * int(c) for v, c in zip(weights, digit_str[:8])) % 11
    check = 0 if rest == 0 else 11 - rest
    return check < 10 and check == int(digit_str[8])


def _has_00_run(text, map_, box):
    for start, stop, digit_str in _number_runs(text):
        if (10 <= len(digit_str) <= 12 and digit_str.startswith("00")
                and _spans_over(map_, start, stop, box)):
            return 1
    return 0


def _has_orgnr(text, map_, box):
    for start, stop, digit_str in _number_runs(text):
        if len(digit_str) == 9:
            candidate = digit_str
        elif len(digit_str) == 11 and digit_str.startswith("00"):
            candidate = digit_str[2:]
        else:
            continue
        if (candidate[0] in "89" and valid_orgnr_mod11(candidate)
                and _spans_over(map_, start, stop, box)):
            return 1
    return 0


def _has_org_ord(lines, box):
    """Company-form words on the box's line or within 1.5 box heights of it.

    The name ("X Borettslag") typically sits on the same line as the number or
    directly above or below.
    """
    y0, y1 = box[1], box[3]
    margin = 1.5 * (y1 - y0)
    for line_tokens, text, _map in lines:
        if not line_tokens:
            continue
        ymid = (min(t.y0 for t in line_tokens)
                + max(t.y1 for t in line_tokens)) / 2
        if y0 - margin <= ymid <= y1 + margin and _ORG_ORD.search(text):
            return 1
    return 0


def _rec_values(tokens):
    return [t.rec_score for t in tokens if t.rec_score is not None]


def features_for_box(tokens, lines, box):
    """Features for one YOLO box. `lines` comes from build_lines(tokens)."""
    i_box = tokens_in_box(tokens, box)
    n_digits, n_bokstaver = count_digits_and_letters(i_box)
    rec = _rec_values(i_box)

    features = {
        "har_tokens": 1 if i_box else 0,
        "n_siffer": n_digits,
        "n_bokstaver": n_bokstaver,
        # 5 decimals, not 3: the scores saturate towards 1, and thresholds
        # like 0.9999 are meaningless if everything above 0.9995 rounds to 1.
        "rec_min": round(min(rec), 5) if rec else None,
        "rec_median": round(statistics.median(rec), 5) if rec else None,
        "rec_min_linje": None,
        "n_siffer_linje": 0,
        "siffer_run": 0,
        "har_fnr_kandidat": 0,
        "har_desimal_naer": 0,
        "har_00_run": 0,
        "har_orgnr": 0,
        "har_org_ord": 0,
        "lang_run": 0,
    }

    # Independent of the box's own line: the words may be above or below.
    features["har_org_ord"] = _has_org_ord(lines, box)

    post = _select_line(lines, box)
    if post is None:
        return features
    line_tokens, text, map_ = post

    rec_line = _rec_values(line_tokens)
    features["rec_min_linje"] = round(min(rec_line), 5) if rec_line else None
    features["n_siffer_linje"] = sum(ch.isdigit() for ch in text)
    features["siffer_run"] = _digits_run(text, map_, box)
    features["har_desimal_naer"] = _has_decimal(text, map_, box)
    features["har_00_run"] = _has_00_run(text, map_, box)
    features["har_orgnr"] = _has_orgnr(text, map_, box)
    features["lang_run"] = max(
        (len(digit_str) for start, stop, digit_str in _number_runs(text)
         if _spans_over(map_, start, stop, box)), default=0)
    features["har_fnr_kandidat"] = 1 if any(
        _spans_over(map_, tr.start, tr.end, box)
        for tr in find_fnr(text, require_mod11=False)) else 0
    return features
