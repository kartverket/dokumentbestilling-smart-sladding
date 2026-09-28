from functools import lru_cache
from typing import NamedTuple

from config import PDF_DPI


# ── Filter parameter registry ────────────────────────────────

class FilterParam(NamedTuple):
    """One filter parameter, described once for every consumer.

    filter_sweep and filter_review build their short codes, CLI flags, labels
    and directory names from FILTER_PARAMS instead of each keeping its own
    copy of the list.

    Naming:
        name      keyword the filter functions take
        code      short code in a --per-source spec ("yolo:kmin=5")
        flag      CLI flag in filter_review.py
        label     format template for the human-readable filter label
        dir_code  short code in output directory names. Eight of these do NOT
                  match `code`: eM/hm/bm/km/kM/lm/lM/apx against emaks/hmin/
                  bmin/kmin/kmaks/lmin/lmaks/amin. They are kept as they are so
                  folders from earlier runs still sort next to new ones. Set
                  dir_code = code here to unify them, at the price of new
                  folder names.

    Argument handling:
        group     "geometry" or "ocr". The OCR rules run first and are not
                  covered by the conf gate, as in prod
        arg       "value" takes a number, "flag" is store_const 1, "optional"
                  takes an optional number with const 1
        unit      what the number means; doubles as the argparse metavar
        help      argparse help text

    Geometry comparison, used by rejection_reasons and compile_filter:
        key       field on the prediction to read. None for the OCR rules, and
                  for conf_threshold, which is a gate rather than a bound
        is_min    True rejects below the limit, False rejects above it
        noun      what the value is called in a rejection reason
        fmt       format spec for the value in that reason
        suffix    written straight after the value in that reason
    """
    name: str
    code: str
    flag: str
    label: str
    dir_code: str
    group: str
    arg: str
    unit: str
    help: str
    key: str = None
    is_min: bool = False
    noun: str = ""
    fmt: str = ""
    suffix: str = ""


FILTER_PARAMS = (
    FilterParam("min_elongation", "e", "--elongation", "e≥{:g}", "e",
                "geometry", "value", "RATIO",
                "MIN_ELONGATION",
                key="elongation", is_min=True, noun="elong", fmt=".1f"),
    FilterParam("max_elongation", "emaks", "--max-elongation", "e≤{:g}", "eM",
                "geometry", "value", "RATIO",
                "Max elongation, removes thin, long strokes",
                key="elongation", noun="elong", fmt=".1f"),
    FilterParam("max_height", "h", "--max-height", "h≤{:g}", "h",
                "geometry", "value", "PT",
                "Max box height in points",
                key="h", noun="height", fmt=".0f"),
    FilterParam("min_height", "hmin", "--min-height", "h≥{:g}", "hm",
                "geometry", "value", "PT",
                "Min box height in points",
                key="h", is_min=True, noun="height", fmt=".1f"),
    FilterParam("max_width", "b", "--max-width", "b≤{:g}", "b",
                "geometry", "value", "PT",
                "Max box width in points",
                key="w", noun="width", fmt=".0f"),
    FilterParam("min_width", "bmin", "--min-width", "b≥{:g}", "bm",
                "geometry", "value", "PT",
                "Min box width in points",
                key="w", is_min=True, noun="width", fmt=".1f"),
    FilterParam("min_short_side", "kmin", "--min-short-side", "short≥{:g}", "km",
                "geometry", "value", "PT",
                "Min short side in points (orientation independent, does not "
                "hit upright boxes the way --min-width does)",
                key="short_side", is_min=True, noun="short side", fmt=".1f"),
    FilterParam("max_short_side", "kmaks", "--max-short-side", "short≤{:g}", "kM",
                "geometry", "value", "PT",
                "Max short side in points",
                key="short_side", noun="short side", fmt=".1f"),
    FilterParam("min_long_side", "lmin", "--min-long-side", "long≥{:g}", "lm",
                "geometry", "value", "PT",
                "Min long side in points. Too short for 5 digits",
                key="long_side", is_min=True, noun="long side", fmt=".1f"),
    FilterParam("max_long_side", "lmaks", "--max-long-side", "long≤{:g}", "lM",
                "geometry", "value", "PT",
                "Max long side in points",
                key="long_side", noun="long side", fmt=".1f"),
    FilterParam("max_area", "a", "--max-area", "a≤{:g}", "a",
                "geometry", "value", "PT2",
                "Max box area in pt²",
                key="area", noun="area", fmt=".0f"),
    FilterParam("min_area_px", "amin", "--min-area-px", "apx≥{:g}", "apx",
                "geometry", "value", "PX2",
                "Min box area in PIXELS² (like MIN_BOX_AREA)",
                key="areal_px", is_min=True, noun="area", fmt=".0f", suffix="px²"),
    FilterParam("conf_threshold", "c", "--conf-threshold", "c≥{:g}→keep", "c",
                "geometry", "value", "CONF",
                "conf ≥ this value is kept regardless of geometry"),

    FilterParam("min_digits", "smin", "--min-digits", "digits≥{:g}", "smin",
                "ocr", "value", "N",
                "Require at least N digits in the box (prod today: 1)"),
    FilterParam("max_letters", "bmaks", "--max-letters", "letters≤{:g}", "bmaks",
                "ocr", "value", "N",
                "Allow at most N letters in the box (prod today: 1)"),
    FilterParam("min_digits_run", "rmin", "--min-digits-run", "run≥{:g}", "rmin",
                "ocr", "value", "N",
                "Require the longest digit run over the box to be ≥ N"),
    FilterParam("require_fnr_candidate", "fnr", "--require-fnr-candidate",
                "fnr-candidate", "fnr", "ocr", "flag", "",
                "Require an 11-digit run with valid fnr shape on the line "
                "(without mod11)"),
    FilterParam("reject_decimal", "des", "--reject-decimal", "no-decimal", "des",
                "ocr", "flag", "",
                "Reject boxes with a decimal separator in the number"),
    FilterParam("rec_veto", "rveto", "--rec-veto", "rec≥{:g}→applies", "rveto",
                "ocr", "value", "REC",
                "Turn the OCR rules on only when rec_min ≥ V. Below V Paddle "
                "read badly, and a missing fnr proves nothing."),
    FilterParam("ocr_conf_exempt", "cfritak", "--ocr-conf-exempt",
                "c≥{:g}→OCR-exempt", "cfritak", "ocr", "value", "CONF",
                "The OCR rules yield for boxes with detection conf ≥ V: a "
                "confident YOLO detection beats text evidence."),
    FilterParam("reject_00_run", "r00", "--reject-00-run", "no-00-run", "r00",
                "ocr", "flag", "",
                "Reject boxes where a 10-12 digit run starts with 00 — orgnr "
                "padded to fnr width; day 00 is invalid in an fnr"),
    FilterParam("reject_orgnr", "orgnr", "--reject-orgnr", "no-orgnr", "orgnr",
                "ocr", "flag", "",
                "Reject boxes with a valid orgnr mod11 (9 digits starting with "
                "8/9, possibly 00-padded)"),
    FilterParam("reject_org_ord", "orgord", "--reject-org-ord",
                "no-org-word({:g})", "orgord", "ocr", "value", "{1,2}",
                "Reject boxes with a company-form word nearby (AS, Borettslag, "
                "Org.nr, …). 1=always, 2=only when the box also lacks an fnr "
                "candidate"),
    FilterParam("line_veto", "lveto", "--line-veto", "linerec≥{:g}→applies",
                "lveto", "ocr", "value", "REC",
                "Turn the OCR rules on only when rec_min_linje ≥ V, fnr "
                "candidate and run length depend on the WHOLE line being read "
                "correctly, not just the box"),
    FilterParam("reject_run_6_10", "run610", "--reject-run-6-10", "no-run-6-10",
                "run610", "ocr", "optional", "MAX",
                "Reject boxes over digit runs of 6..MAX (with gaps); without a "
                "value = 6..10. Use 9: 10-runs are often fnr with a "
                "single-digit day/month or a lost char"),
    FilterParam("without_text_conf", "utconf", "--without-text-conf",
                "noText-c≥{:g}", "utconf", "ocr", "value", "CONF",
                "Boxes WITHOUT text (har_tokens=0) require conf ≥ V — stricter "
                "than prod's YOLO_CONF_NO_TEXT (0.40)"),
    FilterParam("max_gap", "luke", "--max-gap", "gap<{:g}", "luke",
                "ocr", "value", "WIDTHS",
                "Reject paddle/begge boxes where the largest physical gap in "
                "the 11-digit window is ≥ V digit widths, windows stitched "
                "across a column gap"),
    FilterParam("reject_decimal_gap", "desluke", "--reject-decimal-gap",
                "no-decimal-gap", "desluke", "ocr", "flag", "",
                "Reject paddle/begge boxes where the 11-window is stitched "
                "over a decimal separator (. or ,)"),
)

PARAM_BY_NAME = {p.name: p for p in FILTER_PARAMS}
PARAM_BY_CODE = {p.code: p for p in FILTER_PARAMS}

GEOMETRY_PARAMETERS = tuple(p.name for p in FILTER_PARAMS if p.group == "geometry")
# Stricter variants of lenient_check; see _ocr_reason for the semantics.
OCR_PARAMS = tuple(p.name for p in FILTER_PARAMS if p.group == "ocr")
FILTER_PARAMETERS = GEOMETRY_PARAMETERS + OCR_PARAMS

# Geometry rules that are one comparison against one field. min_area_px is held
# out of the loop because it is checked before the conf gate, not after it.
_AREA_CHECK = PARAM_BY_NAME["min_area_px"]
_GEOMETRY_CHECKS = tuple(p for p in FILTER_PARAMS
                         if p.key and p.name != "min_area_px")


# ── Filtering ────────────────────────────────────────────────

def _ocr_reason(p, min_digits=None, max_letters=None, min_digits_run=None,
               require_fnr_candidate=None, reject_decimal=None, rec_veto=None,
               ocr_conf_exempt=None,
               reject_00_run=None, reject_orgnr=None, reject_org_ord=None,
               line_veto=None, reject_run_6_10=None, without_text_conf=None,
               max_gap=None, reject_decimal_gap=None):
    """Why a YOLO box is rejected by a stricter lenient_check, or None.

    Mirrors _accept_yolo_box in app/model_main.py: the box needs text features
    (only kilde "yolo" has them) and har_tokens must be 1, since boxes without
    text are governed by YOLO_CONF_NO_TEXT instead. The window rules target
    paddle/begge and therefore run before that gate.

    rec_veto: the OCR rules may only reject when Paddle read the box
    confidently; on a bad read a missing fnr proves nothing. ocr_conf_exempt
    mirrors the geometry conf gate: a confident detection beats text evidence.
    """
    # Graphics/map detections have no text to defend themselves with, so only
    # confidence separates them from fnr on scanned pages where OCR read nothing.
    if without_text_conf is not None and p.get("har_tokens") is not None \
            and not p["har_tokens"]:
        if p.get("conf") is None or p["conf"] < without_text_conf:
            return f"no text and conf < {without_text_conf:g}"
    # Window rules must run BEFORE the text gate below. Paddle boxes, stitched
    # from an 11-digit window across "." and column gaps, have no har_tokens.
    if reject_decimal_gap and p.get("har_desimal_luke"):
        return "decimal separator in a gap of the 11-digit window"
    if max_gap is not None and p.get("maks_luke") is not None \
            and p["maks_luke"] >= max_gap:
        return f"gap {p['maks_luke']:g} >= {max_gap:g} digit widths"
    if p.get("har_tokens") is None or not p["har_tokens"]:
        return None
    if ocr_conf_exempt is not None and p.get("conf") is not None \
            and p["conf"] >= ocr_conf_exempt:
        return None
    if rec_veto is not None:
        rec = p.get("rec_min")
        if rec is None or rec < rec_veto:
            return None
    # rec_veto for the WHOLE line: the fnr-candidate and run-length features rely
    # on the neighbouring numbers being read correctly too.
    if line_veto is not None:
        rec_line = p.get("rec_min_linje")
        if rec_line is None or rec_line < line_veto:
            return None

    if min_digits is not None and (p["n_siffer"] or 0) < min_digits:
        return f"digits {p['n_siffer'] or 0:.0f} < {min_digits:g}"
    if max_letters is not None and (p["n_bokstaver"] or 0) > max_letters:
        return f"letters {p['n_bokstaver'] or 0:.0f} > {max_letters:g}"
    if min_digits_run is not None and (p["siffer_run"] or 0) < min_digits_run:
        return f"digit run {p['siffer_run'] or 0:.0f} < {min_digits_run:g}"
    if require_fnr_candidate and not p.get("har_fnr_kandidat"):
        return "no 11-digit fnr candidate on the line"
    if reject_decimal and p.get("har_desimal_naer"):
        return "decimal separator inside the number"
    if reject_00_run and p.get("har_00_run"):
        return "11-digit run starts with 00"
    if reject_orgnr and p.get("har_orgnr"):
        return "valid orgnr in the box"
    # At 2 the box must also lack an fnr candidate: an org name alone must not
    # condemn a number that really does have fnr shape.
    if reject_org_ord and p.get("har_org_ord") \
            and (reject_org_ord < 2 or not p.get("har_fnr_kandidat")):
        return "org word near the box"
    # Runs too long for a bare personnummer and too short for an fnr: dagbok
    # numbers, amounts, coordinates. 10-runs are often real fnr with a
    # single-digit day/month or an OCR-dropped character.
    if reject_run_6_10 and p.get("lang_run") is not None:
        max_items = 10 if reject_run_6_10 == 1 else reject_run_6_10
        if 6 <= p["lang_run"] <= max_items:
            return f"digit run of {p['lang_run']:.0f} cannot be an fnr"
    return None


def _split_parameters(kwargs):
    """Splits a flat filter config into (geometry, ocr), dropping unset values."""
    geometry, ocr = {}, {}
    for name, value in kwargs.items():
        if value is None:
            continue
        param = PARAM_BY_NAME.get(name)
        if param is None:
            raise TypeError(f"Unknown filter parameter {name!r}. "
                            f"Valid: {', '.join(FILTER_PARAMETERS)}")
        (geometry if param.group == "geometry" else ocr)[name] = value
    return geometry, ocr


def _geometry_reason(check, p, limit):
    """Text for a failed geometry comparison, or None if the box passes it."""
    value = p[check.key]
    if (value < limit) if check.is_min else (value > limit):
        return (f"{check.noun} {value:{check.fmt}}{check.suffix} "
                f"{'<' if check.is_min else '>'} {limit:g}")
    return None


def rejection_reasons(p, **kwargs):
    """Reasons the box is filtered out; an empty list means it is kept.

    min_area_px is in PIXELS² to match MIN_BOX_AREA, and is checked BEFORE the
    conf gate since the prod noise floor applies to every box. The OCR rules run
    first of all and are NOT covered by the conf gate, as in prod.

    compile_filter answers the same question without building text. Both walk
    _GEOMETRY_CHECKS, so the two cannot drift apart.
    """
    geometry, ocr = _split_parameters(kwargs)
    if reason := _ocr_reason(p, **ocr):
        return [reason]
    min_area = geometry.get("min_area_px")
    if min_area is not None:
        if reason := _geometry_reason(_AREA_CHECK, p, min_area):
            return [reason]
    conf_threshold = geometry.get("conf_threshold")
    if conf_threshold is not None and p.get("conf") is not None \
            and p["conf"] >= conf_threshold:
        return []
    reasons = []
    for check in _GEOMETRY_CHECKS:
        limit = geometry.get(check.name)
        if limit is not None:
            if reason := _geometry_reason(check, p, limit):
                reasons.append(reason)
    return reasons


# Cached on the sorted parameters so is_filtered stays cheap in a loop; a sweep
# goes through make_filter and compiles once anyway.
@lru_cache(maxsize=4096)
def _compile_filter(items):
    geometry, ocr = _split_parameters(dict(items))
    min_area = geometry.get("min_area_px")
    conf_threshold = geometry.get("conf_threshold")
    checks = tuple((c.key, c.is_min, geometry[c.name]) for c in _GEOMETRY_CHECKS
                   if c.name in geometry)

    def removed(p):
        if ocr and _ocr_reason(p, **ocr):
            return True
        if min_area is not None and p[_AREA_CHECK.key] < min_area:
            return True
        if conf_threshold is not None and p.get("conf") is not None \
                and p["conf"] >= conf_threshold:
            return False
        for key, is_min, limit in checks:
            value = p[key]
            if (value < limit) if is_min else (value > limit):
                return True
        return False

    return removed


def compile_filter(kwargs):
    """Predicate p -> bool for one filter config, resolved once.

    A sweep runs the same config over every prediction, so the parameters are
    looked up here and the closure walks only the checks that are actually set.
    """
    return _compile_filter(tuple(sorted(kwargs.items())))


def is_filtered(p, **kwargs):
    """Whether the filter drops this box; rejection_reasons without the text."""
    return compile_filter(kwargs)(p)


def make_filter(**kwargs):
    """Predicate p -> bool for one shared filter config."""
    return compile_filter(kwargs)


# ── Prod adapter ─────────────────────────────────────────────

_PT_PER_PX = 72.0 / PDF_DPI


def rule_input(box_px, features, conf):
    """Prediction dict for one prod box: features + conf + geometry in points.

    Same keys as read_predictions in utils/filter_common.py derives from the
    result CSV, limited to what the RULE_* configs actually read.
    """
    p = dict(features) if features else {}
    p["conf"] = conf
    x0, y0, x1, y1 = box_px[:4]
    short, long = sorted(((x1 - x0) * _PT_PER_PX, (y1 - y0) * _PT_PER_PX))
    p["short_side"] = short
    p["long_side"] = long
    # Degenerate boxes get 0, so any min_elongation bound rejects them.
    p["elongation"] = long / short if short > 0 else 0.0
    return p
