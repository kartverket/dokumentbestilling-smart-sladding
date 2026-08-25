"""
Shared loading, matching, filtering and measurement for filter_sweep and
filter_review.

Measurement is truth-centric: a config is judged by how many *truth boxes lose
all coverage*, not by how many predictions it drops. Removing one correct
prediction while another still covers the same box costs nothing. TREFF covers a
truth box at roughly its size, SLURV covers one but is far too large, BOM covers
none (pure oversladding); the report columns are lost (boxes that lost all
coverage), ov.rm (BOM removed) and red.rm (covering predictions removed with
nothing lost).

Predictions on documents missing from the truth CSV are out of scope by default;
including them inflates every oversladding number with never-labelled documents.
"""

import csv
import os
import re
import sys
from collections import defaultdict

_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

try:
    from config import PDF_DPI, FEATURE_FIELDS
except ImportError:  # running outside the repo
    PDF_DPI = 300
    FEATURE_FIELDS = ("har_tokens", "n_siffer", "n_bokstaver", "rec_min", "rec_median",
                  "rec_min_linje", "n_siffer_linje", "siffer_run",
                  "har_fnr_kandidat", "har_desimal_naer",
                  "har_00_run", "har_orgnr", "har_org_ord", "lang_run",
                  "maks_luke", "har_desimal_luke")

SCALE = PDF_DPI / 72.0   # PDF points -> pixels

# Set by MANUAL review of borderline crops (filter_review.py --band areal LO HI),
# not geometry: false hits on the neighbouring line are rare (12 of 20019 on
# uttrekk 4), so discarding real hits costs more. Do not change without a review.
STD_THRESHOLD = 0.32       # min coverage of a truth box for a prediction to "hit"
STD_SLOPPINESS_FACTOR = 3.0   # pred area > factor x covered truth area => SLURV


# ── Geometry ─────────────────────────────────────────────────

def doc_no(name):
    m = re.match(r"0*(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


def overlap(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return (ix1 - ix0) * (iy1 - iy0) if (ix1 > ix0 and iy1 > iy0) else 0.0


def area(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


# ── Match metrics ────────────────────────────────────────────

def match_metrics(pn, fn, truth_horizontal):
    """All metrics for one (prediction, truth) pair; boxes in normalised coords.
    Returns None when the boxes do not overlap.

    Neighbouring text lines sit offset along the sladd's SHORT side, which is the
    height only when the page is upright, so the short-side axis is measured
    explicitly rather than y. truth_horizontal must be decided in POINT space by
    the caller: normalising skews w/h, so a near-square box can flip orientation.
    """
    o = overlap(pn, fn)
    if o <= 0:
        return None
    fa, pa = area(fn), area(pn)
    if fa <= 0:
        return None
    ox = max(0.0, min(pn[2], fn[2]) - max(pn[0], fn[0]))
    oy = max(0.0, min(pn[3], fn[3]) - max(pn[1], fn[1]))
    if truth_horizontal:
        short, o_short = fn[3] - fn[1], oy
        long, o_long = fn[2] - fn[0], ox
        d_center = abs((pn[1] + pn[3]) / 2 - (fn[1] + fn[3]) / 2)
    else:
        short, o_short = fn[2] - fn[0], ox
        long, o_long = fn[3] - fn[1], oy
        d_center = abs((pn[0] + pn[2]) / 2 - (fn[0] + fn[2]) / 2)
    return {
        "cov_area": o / fa,                                  # current criterion
        "dek_p": o / pa if pa > 0 else 0.0,
        "iou": o / (fa + pa - o),
        "cov_short": o_short / short if short > 0 else 0.0,
        "cov_long": o_long / long if long > 0 else 0.0,
        "center_short": d_center / short if short > 0 else 9.9,
    }


# Decides whether prediction and truth box are the SAME FIELD. For "center" a low
# value is good, so its threshold is a ceiling, not a floor.
CRITERIA = {
    "area":    lambda m, t: m["cov_area"] >= t,
    "short_side": lambda m, t: m["cov_short"] >= t,
    "long_side": lambda m, t: m["cov_long"] >= t,
    "iou":      lambda m, t: m["iou"] >= t,
    "center":   lambda m, t: m["center_short"] <= t,
}

# Which match_metrics field each criterion reads; needed by the band review.
CRITERION_FIELDS = {"area": "cov_area", "short_side": "cov_short",
                  "long_side": "cov_long", "iou": "iou", "center": "center_short"}
CRITERION_LOW_IS_GOOD = {"center"}
STD_CRITERION = "area"

# Measured on label pairs from uttrekk 4 + 5. "long_side" is absent on purpose: it
# separates neighbouring lines poorly (shared long-side extent) and is uncalibrated.
RECOMMENDED_THRESHOLDS = {"area": 0.32, "short_side": 0.60, "iou": 0.20, "center": 0.40}


# ── Loading ──────────────────────────────────────────────────

# Label rows judged to be truth noise in manual review. iter_label_rows (and
# with it every read_truth*) skips them, so the labels file needs no cleaning
# and a re-export resets nothing.
INVALID_LABELS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "ugyldige_labels.txt")


def read_invalid_label_ids(path=None):
    """Label ids from ugyldige_labels.txt: one per line, "#" starts a comment."""
    path = path or INVALID_LABELS_FILE
    ids = set()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    ids.add(line)
    return ids


def reclassify_invalid_covering(rows):
    """Covering rows (klasse != BOM) whose fasit labels are ALL listed in
    ugyldige_labels.txt become BOM: the box covers only noise. label_id is
    «;»-joined, as vlm_export writes it. Runs at read time so a grown
    ugyldige_labels.txt applies to old manifests without a re-export.
    Returns the number of rows changed."""
    invalid = read_invalid_label_ids()
    if not invalid:
        return 0
    n = 0
    for r in rows:
        if r.get("klasse") == "BOM":
            continue
        ids = [i.strip() for i in (r.get("label_id") or "").split(";")
               if i.strip()]
        if ids and all(i in invalid for i in ids):
            r["klasse"] = "BOM"
            n += 1
    return n


_WARNED_WITHOUT_ID = False


def _warn_missing_id_column(invalid, columns):
    global _WARNED_WITHOUT_ID
    if invalid and "id" not in (columns or []) and not _WARNED_WITHOUT_ID:
        _WARNED_WITHOUT_ID = True
        print(f"⚠ ugyldige_labels.txt has {len(invalid)} ids, but the labels CSV "
              f"has no id column. No rows are filtered. "
              f"Include id in the next export.")


def iter_label_rows(path, exclude_status=("REJECTED",), info=None):
    """Yields rows from a labels CSV with the fasit policy applied.

    The one place that decides what counts as fasit input: rows listed in
    ugyldige_labels.txt are ALWAYS skipped (with the one-time warning when
    the CSV has no id column); statuses in exclude_status are skipped — pass
    () to keep REJECTED, as the stats tools do to measure prod's false
    positives.

    `info`, if given, is a dict that receives "columns" (the CSV header) and
    "discarded" (a per-reason tally, incl. «(ugyldig-listet)»). Both are set
    once iteration starts, so read them after the loop.
    """
    exclude = {str(e).strip().upper() for e in exclude_status}
    invalid = read_invalid_label_ids()
    tally = defaultdict(int)
    with open(path, newline="", encoding="utf-8-sig") as f:
        leser = csv.DictReader(f)
        _warn_missing_id_column(invalid, leser.fieldnames)
        if info is not None:
            info["columns"] = leser.fieldnames or []
            info["discarded"] = tally
        for r in leser:
            status = (r.get("ml_status") or "").strip().upper()
            if status in exclude:
                tally[status or "(empty)"] += 1
                continue
            label_id = (r.get("id") or "").strip()
            if label_id and label_id in invalid:
                tally["(ugyldig-listet)"] += 1
                continue
            yield r


def read_truth_boxes(path):
    """Truth labels (ACCEPTED + manual; REJECTED and ugyldige_labels.txt skipped)
    as (doc_no, side) -> [(x0, y0, x1, y1, label_id), ...] in PDF points.
    """
    truth = defaultdict(list)
    for r in iter_label_rows(path):
        try:
            nr = int(r["fil_revisjon_id"])
            page = int(r["sidetall"])
            x, y = float(r["x"]), float(r["y"])
            w, h = float(r["width"]), float(r["height"])
        except (TypeError, ValueError, KeyError):
            continue
        x0, x1 = sorted((x, x + w))
        y0, y1 = sorted((y, y + h))
        truth[(nr, page)].append((x0, y0, x1, y1, (r.get("id") or "").strip()))
    return truth


def read_predictions(path):
    """Result CSV with pixel coordinates (drawing), plus normalised coordinates
    (matching) and PDF-point dimensions (filtering) per prediction.
    """
    pred = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                name = r["navn"]
                page = int(r["side"])
                bw, bh = int(r["bilde_bredde"]), int(r["bilde_hoyde"])
                x0, y0 = float(r["x0"]), float(r["y0"])
                x1, y1 = float(r["x1"]), float(r["y1"])
                source = r.get("kilde", "unknown")
                # OCR read quality must never share the conf column, or paddle
                # boxes slip past the geometry filters. Old CSVs: "conf" = yolo_conf.
                conf_s = r.get("yolo_conf")
                if conf_s is None:
                    conf_s = r.get("conf", "")
                conf = float(conf_s) if conf_s else None
                rec_s = r.get("paddle_rec_score", "")
                paddle_rec = float(rec_s) if rec_s else None
            except (TypeError, ValueError, KeyError):
                continue
            w_pt = abs(x1 - x0) / SCALE
            h_pt = abs(y1 - y0) / SCALE
            if w_pt <= 0 or h_pt <= 0 or bw <= 0 or bh <= 0:
                continue
            ratio = w_pt / h_pt
            norm = (x0 / bw, y0 / bh, x1 / bw, y1 / bh)
            pred.append({
                "navn": name, "side": page, "doc_no": doc_no(name),
                "bw": bw, "bh": bh,
                "px": (x0, y0, x1, y1),
                "norm": norm,
                "norm_areal": area(norm),
                "w": w_pt, "h": h_pt,
                "ratio": ratio,
                "elongation": max(ratio, 1 / ratio),
                "short_side": min(w_pt, h_pt),
                "long_side": max(w_pt, h_pt),
                "area": w_pt * h_pt,
                "areal_px": abs(x1 - x0) * abs(y1 - y0),
                "kilde": source, "conf": conf, "paddle_rec": paddle_rec,
                **_read_features(r),
            })
    return pred


def _read_features(row):
    """Feature columns from the result CSV, as floats or None.

    Empty means "not computed" (a kilde other than "yolo", or an old CSV without
    the columns); the OCR filters then leave the box alone.
    """
    ut = {}
    for field in FEATURE_FIELDS:
        value = row.get(field, "")
        try:
            ut[field] = float(value) if value not in (None, "") else None
        except ValueError:
            ut[field] = None
    return ut


def read_processed_docs(path):
    """Documents the model has been run on: one per line, filename (00123.pdf) or
    bare number. Blank lines and lines starting with # are skipped.
    """
    dokumenter = set()
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            nr = doc_no(line)
            if nr is not None:
                dokumenter.add(nr)
    return dokumenter


def read_truth_rows(path, exclude=("REJECTED",)):
    """Truth labels as full rows, with geometry and metadata.

    Unlike read_truth_boxes all columns are kept, so a filter can be tested
    DIRECTLY against the caseworkers' sladdinger without any result CSV. Truth
    boxes have no conf, so the conf gate never fires: the test shows what the
    geometry rules alone would have rejected.
    """
    rows = []
    info = {}
    for r in iter_label_rows(path, exclude_status=exclude, info=info):
        status = (r.get("ml_status") or "").strip().upper()
        try:
            nr = int(r["fil_revisjon_id"])
            page = int(r["sidetall"])
            x, y = float(r["x"]), float(r["y"])
            w, h = float(r["width"]), float(r["height"])
        except (TypeError, ValueError, KeyError):
            info["discarded"]["(ugyldig rad)"] += 1
            continue
        x0, x1 = sorted((x, x + w))
        y0, y1 = sorted((y, y + h))
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            info["discarded"]["(null areal)"] += 1
            continue
        ratio = bw / bh
        rows.append({
            "doc_no": nr, "side": page,
            "box": (x0, y0, x1, y1),
            "w": bw, "h": bh,
            "short_side": min(bw, bh), "long_side": max(bw, bh),
            "elongation": max(ratio, 1 / ratio),
            "area": bw * bh,
            "areal_px": bw * bh * SCALE * SCALE,
            "conf": None,
            "ml_status": status or "(empty)",
            "type": (r.get("type") or "").strip() or "(empty)",
            "row": r,
        })
    return rows, dict(info.get("discarded", {})), info.get("columns", [])


# ── Dataset with truth-centric index ─────────────────────────

class Dataset:
    """Predictions and truth boxes joined by coverage.

    utenfor = predictions on documents with no truth (excluded from scope),
    covered_before = truth boxes with at least one coverer, n_miss = BOM count.
    """

    def __init__(self, pred, outside, truth_boxes, coverage_before,
                 threshold, oversize_factor, n_truth=None, name="all",
                 scope_doc=None, n_truth_unprocessed=0, n_doc_unprocessed=0,
                 criterion=STD_CRITERION):
        self.pred = pred
        self.outside = outside
        self.truth_boxes = truth_boxes
        self.coverage_before = coverage_before
        self.threshold = threshold
        self.oversize_factor = oversize_factor
        self.criterion = criterion
        self.name = name
        self.n_truth_unprocessed = n_truth_unprocessed
        self.n_doc_unprocessed = n_doc_unprocessed
        self.scope_doc = (scope_doc if scope_doc is not None
                          else {p["doc_no"] for p in pred})
        # Coverer indices stay global after a split; only a subset is in scope.
        self.n_truth = len(truth_boxes) if n_truth is None else n_truth
        self.covered_before = sum(1 for d in coverage_before if d > 0)
        self.n_miss = sum(1 for p in pred if p["klasse"] == "BOM")
        self.n_hit = sum(1 for p in pred if p["klasse"] == "TREFF")
        self.n_oversize = sum(1 for p in pred if p["klasse"] == "SLURV")
        self.n_dekkende = self.n_hit + self.n_oversize
        self.per_source = defaultdict(list)
        for p in pred:
            self.per_source[p["kilde"]].append(p)

    def sources(self):
        return sorted({p["kilde"] for p in self.pred})


def build_dataset(truth, pred, threshold=STD_THRESHOLD,
                  oversize_factor=STD_SLOPPINESS_FACTOR, include_unlabelled=False,
                  processed_doc=None, criterion=STD_CRITERION):
    """Joins predictions to truth boxes, setting p["dekker"] (truth-box indices
    covered >= threshold), p["klasse"] ("TREFF" | "SLURV" | "BOM") and p["riktig"].

    Scope is LABELLED ∩ PROCESSED: truth on documents the model never ran on is
    unmeasured, not missed, and counting it would depress recall. Without
    processed_doc the documents present in the result CSV are assumed, so a
    document where the model found nothing counts as unprocessed.

    The threshold is interpreted by criterion (see CRITERIA), so 0.4 means
    different things per rule.
    """
    labelled_doc = {nr for (nr, _page) in truth}
    processed = (set(processed_doc) if processed_doc is not None
              else {p["doc_no"] for p in pred})
    # The labels file covers the whole uttrekk, so a processed document without
    # truth rows was reviewed and has zero fnr, so its predictions are real BOM.
    scope_doc = processed if include_unlabelled else labelled_doc & processed

    page_str = {}
    for p in pred:
        key = (p["doc_no"], p["side"])
        if key not in page_str:
            page_str[key] = (p["bw"] / SCALE, p["bh"] / SCALE)

    # Flat normalised truth list + (doc, page) -> [indices] lookup
    truth_boxes = []
    per_page = defaultdict(list)
    n_truth_unprocessed = 0
    for (nr, si), boxes in sorted(truth.items()):
        if nr not in scope_doc:
            n_truth_unprocessed += len(boxes)
            continue
        pw, ph = page_str.get((nr, si), (595, 842))   # A4 fallback
        for (x0, y0, x1, y1, *rest) in boxes:
            n = (x0 / pw, y0 / ph, x1 / pw, y1 / ph)
            per_page[(nr, si)].append(len(truth_boxes))
            truth_boxes.append({
                "doc_no": nr, "side": si,
                "box": (x0, y0, x1, y1),
                "label_id": rest[0] if rest else "",
                "norm": n,
                "norm_areal": area(n),
                # Decided in point space: normalising skews w/h
                "horizontal": (x1 - x0) >= (y1 - y0),
            })

    # Limited to scope_doc, not to labelled/all: with an explicit --processed-list
    # the old condition counted hits outside the list as BOM (1 % precision).
    inside, outside = [], []
    for p in pred:
        if p["doc_no"] in scope_doc:
            inside.append(p)
        else:
            outside.append(p)

    try:
        passer = CRITERIA[criterion]
    except KeyError:
        raise ValueError(f"unknown criterion {criterion!r}, "
                         f"valid: {', '.join(sorted(CRITERIA))}")

    coverage_before = [0] * len(truth_boxes)
    for p in inside:
        pn = p["norm"]
        covers = []
        covered_area = 0.0
        for j in per_page.get((p["doc_no"], p["side"]), ()):
            fb = truth_boxes[j]
            fa = fb["norm_areal"]
            if fa <= 0:
                continue
            m = match_metrics(pn, fb["norm"], fb["horizontal"])
            if m is not None and passer(m, threshold):
                covers.append(j)
                covered_area += fa
                coverage_before[j] += 1
        p["covers"] = covers
        if not covers:
            p["klasse"] = "BOM"
        elif p["norm_areal"] > oversize_factor * covered_area:
            p["klasse"] = "SLURV"
        else:
            p["klasse"] = "TREFF"
        p["riktig"] = bool(covers)

    for p in outside:
        p["covers"] = []
        p["klasse"] = "BOM"
        p["riktig"] = False

    return Dataset(inside, outside, truth_boxes, coverage_before,
                    threshold, oversize_factor, scope_doc=scope_doc,
                    n_truth_unprocessed=n_truth_unprocessed,
                    n_doc_unprocessed=len(labelled_doc - processed),
                    criterion=criterion)


def split_dataset(ds, doc_apply, name):
    """Datasett restricted to doc_apply. Truth indices stay global (p["dekker"]
    still points correctly), but coverage is recounted from the subset alone.
    """
    pred = [p for p in ds.pred if p["doc_no"] in doc_apply]
    n_truth = sum(1 for fb in ds.truth_boxes if fb["doc_no"] in doc_apply)
    coverage = [0] * len(ds.truth_boxes)
    for p in pred:
        for j in p["covers"]:
            coverage[j] += 1
    return Dataset(pred, [], ds.truth_boxes, coverage, ds.threshold,
                    ds.oversize_factor, n_truth=n_truth, name=name,
                    scope_doc=doc_apply & ds.scope_doc,
                    criterion=ds.criterion)


def split_by_document(ds, share, seed=42):
    """Splits DOCUMENTS into (training, holdout). Splitting on predictions would
    leak the same page into both sets."""
    import random
    # Split the WHOLE scope; documents where the model found nothing must not vanish.
    dokumenter = sorted(ds.scope_doc)
    shuffled = list(dokumenter)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * share))
    test = set(shuffled[:n_test])
    training = set(shuffled[n_test:])
    return (split_dataset(ds, training, "training"),
            split_dataset(ds, test, "holdout"))


def pareto_front(rows, target=lambda r: (r.m.lost, r.m.ov_rm)):
    """Non-dominated rows: the highest `ov.rm` at each level of `lost`."""
    front, best = [], -1
    for r in sorted(rows, key=lambda r: (target(r)[0], -target(r)[1])):
        _lost, ov = target(r)
        if ov > best:
            front.append(r)
            best = ov
    return front


# ── Filtering ────────────────────────────────────────────────

GEOMETRY_PARAMETERS = (
    "min_elongation", "max_elongation", "max_height", "min_height",
    "max_width", "min_width", "min_short_side", "max_short_side",
    "min_long_side", "max_long_side", "max_area", "min_area_px",
    "conf_threshold")

# Stricter variants of lenient_check; see _ocr_grunn for the semantics.
OCR_PARAMS = (
    "min_digits", "max_letters", "min_digits_run", "require_fnr_candidate",
    "reject_decimal", "rec_veto", "ocr_conf_exempt", "reject_00_run",
    "reject_orgnr", "reject_org_ord", "line_veto", "reject_run_6_10",
    "without_text_conf", "max_gap", "reject_decimal_gap")

FILTER_PARAMETERS = GEOMETRY_PARAMETERS + OCR_PARAMS


def _ocr_reason(p, min_digits=None, max_letters=None, min_digits_run=None,
               require_fnr_candidate=None, reject_decimal=None, rec_veto=None,
               ocr_conf_exempt=None,
               reject_00_run=None, reject_orgnr=None, reject_org_ord=None,
               line_veto=None, reject_run_6_10=None, without_text_conf=None,
               max_gap=None, reject_decimal_gap=None):
    """Why a YOLO box is rejected by a stricter lenient_check, or None.

    Mirrors _godta_yolo_boks in app/model_main.py: the box needs text features
    (only kilde "yolo" has them) and har_tokens must be 1, since boxes without
    text are governed by YOLO_CONF_NO_TEXT instead. The window rules target
    paddle/begge and therefore run before that gate.

    rec_veto is the hypothesis under test: OCR rules may only reject when Paddle
    read the box confidently. On a bad read a missing fnr proves nothing.
    ocr_conf_fritak mirrors the geometry conf gate: manual review (uttrekk 6) put
    real fnr almost all at conf >= 0.5, coordinates and kontonummer below 0.4.
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
    # numbers, amounts, coordinates. Manual review on uttrekk 6 showed 10-runs are
    # OFTEN real fnr (single-digit day/month, or a character lost in OCR).
    if reject_run_6_10 and p.get("lang_run") is not None:
        max_items = 10 if reject_run_6_10 == 1 else reject_run_6_10
        if 6 <= p["lang_run"] <= max_items:
            return f"digit run of {p['lang_run']:.0f} cannot be an fnr"
    return None


def rejection_reasons(p, min_elongation=None, max_elongation=None,
                   max_height=None, min_height=None,
                   max_width=None, min_width=None,
                   min_short_side=None, max_short_side=None,
                   min_long_side=None, max_long_side=None,
                   max_area=None, min_area_px=None, conf_threshold=None,
                   **ocr):
    """Reasons the box is filtered out; an empty list means it is kept.

    min_area_px is in PIXELS² to match MIN_BOX_AREA, and is checked BEFORE the
    conf gate since the prod noise floor applies to every box. The OCR rules run
    first of all and are NOT covered by the conf gate, as in prod.
    """
    reasons = []
    if reason := _ocr_reason(p, **ocr):
        reasons.append(reason)
        return reasons
    if min_area_px is not None and p["areal_px"] < min_area_px:
        reasons.append(f"area {p['areal_px']:.0f}px² < {min_area_px:g}")
        return reasons
    # High confidence: trust the prediction and skip the rest of the geometry
    if conf_threshold is not None and p.get("conf") is not None \
            and p["conf"] >= conf_threshold:
        return []
    if min_elongation is not None and p["elongation"] < min_elongation:
        reasons.append(f"elong {p['elongation']:.1f} < {min_elongation:g}")
    if max_elongation is not None and p["elongation"] > max_elongation:
        reasons.append(f"elong {p['elongation']:.1f} > {max_elongation:g}")
    if max_height is not None and p["h"] > max_height:
        reasons.append(f"height {p['h']:.0f} > {max_height:g}")
    if min_height is not None and p["h"] < min_height:
        reasons.append(f"height {p['h']:.1f} < {min_height:g}")
    if max_width is not None and p["w"] > max_width:
        reasons.append(f"width {p['w']:.0f} > {max_width:g}")
    if min_width is not None and p["w"] < min_width:
        reasons.append(f"width {p['w']:.1f} < {min_width:g}")
    if min_short_side is not None and p["short_side"] < min_short_side:
        reasons.append(f"short side {p['short_side']:.1f} < {min_short_side:g}")
    if max_short_side is not None and p["short_side"] > max_short_side:
        reasons.append(f"short side {p['short_side']:.1f} > {max_short_side:g}")
    if min_long_side is not None and p["long_side"] < min_long_side:
        reasons.append(f"long side {p['long_side']:.1f} < {min_long_side:g}")
    if max_long_side is not None and p["long_side"] > max_long_side:
        reasons.append(f"long side {p['long_side']:.1f} > {max_long_side:g}")
    if max_area is not None and p["area"] > max_area:
        reasons.append(f"area {p['area']:.0f} > {max_area:g}")
    return reasons


def er_filtered(p, min_elongation=None, max_elongation=None,
                max_height=None, min_height=None,
                max_width=None, min_width=None,
                min_short_side=None, max_short_side=None,
                min_long_side=None, max_long_side=None,
                max_area=None, min_area_px=None, conf_threshold=None,
                **ocr):
    """Fast variant of rejection_reasons that builds no text."""
    if ocr and _ocr_reason(p, **ocr):
        return True
    if min_area_px is not None and p["areal_px"] < min_area_px:
        return True
    if conf_threshold is not None and p["conf"] is not None \
            and p["conf"] >= conf_threshold:
        return False
    if min_elongation is not None and p["elongation"] < min_elongation:
        return True
    if max_elongation is not None and p["elongation"] > max_elongation:
        return True
    if max_height is not None and p["h"] > max_height:
        return True
    if min_height is not None and p["h"] < min_height:
        return True
    if max_width is not None and p["w"] > max_width:
        return True
    if min_width is not None and p["w"] < min_width:
        return True
    if min_short_side is not None and p["short_side"] < min_short_side:
        return True
    if max_short_side is not None and p["short_side"] > max_short_side:
        return True
    if min_long_side is not None and p["long_side"] < min_long_side:
        return True
    if max_long_side is not None and p["long_side"] > max_long_side:
        return True
    if max_area is not None and p["area"] > max_area:
        return True
    return False


def make_filter(**kwargs):
    """Predicate p -> bool for one shared filter config."""
    return lambda p: er_filtered(p, **kwargs)


def make_filter_per_source(per_source, only_source=None):
    """Predicate p -> bool with separate parameters per kilde.

    per_source:   {kilde: {filter params}}; a missing kilde is not filtered.
    only_source:  filter only this kilde, leaving the others untouched.
    """
    def _removed(p):
        source = p["kilde"].lower()
        if only_source is not None and source != only_source.lower():
            return False
        kw = per_source.get(source)
        return er_filtered(p, **kw) if kw else False
    return _removed


def parse_per_source(spec_list):
    """Parses "kilde:e=V,h=V,b=V,c=V,..." -> {kilde: {params}}; keys in param_map.

    Units: h/hmin/b/bmin in pt, a in pt², amin in px² (like MIN_BOX_AREA).
    The OCR rules apply to kilde "yolo" only and the window rules (luke, desluke)
    to paddle/begge; see _ocr_grunn. Keys carrying a bound rather than 0/1:
    rveto/lveto = OCR rules apply only above that read quality, cfritak = they
    stand down above that detection conf, run610 = reject digit runs 6..V
    (1 = 6..10), luke = reject a gap of >= V digit widths, utconf = boxes without
    text need conf >= V (prod 0.40), orgord = 2 rejects only without fnr candidate.
    """
    param_map = {
        "e": "min_elongation",      "emaks": "max_elongation",
        "h": "max_height",          "hmin": "min_height",
        "b": "max_width",         "bmin": "min_width",
        "kmin": "min_short_side",     "kmaks": "max_short_side",
        "lmin": "min_long_side",     "lmaks": "max_long_side",
        "a": "max_area",          "amin": "min_area_px",
        "c": "conf_threshold",
        "smin": "min_digits",       "bmaks": "max_letters",
        "rmin": "min_digits_run",   "fnr": "require_fnr_candidate",
        "des": "reject_decimal",     "rveto": "rec_veto",
        "cfritak": "ocr_conf_exempt",
        "r00": "reject_00_run",      "orgnr": "reject_orgnr",
        "orgord": "reject_org_ord",
        "lveto": "line_veto",      "run610": "reject_run_6_10",
        "utconf": "without_text_conf",
        "luke": "max_gap",        "desluke": "reject_decimal_gap",
    }
    result = {}
    for spec in spec_list:
        if ":" not in spec:
            raise ValueError(f"Invalid per-kilde format: {spec!r} "
                             f"(expected 'kilde:e=V,h=V,...')")
        source, param_str = spec.split(":", 1)
        kwargs = {}
        for bit in param_str.split(","):
            bit = bit.strip()
            if not bit:
                continue
            if "=" not in bit:
                raise ValueError(f"Invalid parameter: {bit!r} in {spec!r}")
            key, value = bit.split("=", 1)
            key = key.strip().lower()
            if key not in param_map:
                raise ValueError(f"Unknown parameter {key!r} in {spec!r}. "
                                 f"Valid: {', '.join(param_map)}")
            kwargs[param_map[key]] = float(value)
        result[source.strip().lower()] = kwargs
    return result


# ── Measurement ──────────────────────────────────────────────

class Measurement:
    """Result of applying one filter config to a dataset."""

    __slots__ = (
        "lost", "lost_pct", "ov_rm", "ov_pct", "red_rm", "critical_rm",
        "oversize_rm", "n_rm", "area_rm", "ov_area_rm", "covered_after",
        "recall_after", "prec_after", "correct_after", "ov_after", "net",
        "ov_per_lost", "lost_boxes")

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw.get(name))


def measure_filter(ds, removed, cost=1.0, collect_lost=False, candidates=None):
    """Measures one filter config truth-centrically.

    cost:        removed oversladdinger one lost truth box is worth;
                 net = ov.rm − cost × lost.
    candidates:  restrict the predicate to this subset of ds.pred, leaving the
                 rest untouched, which makes per-kilde sweeps faster.
    """
    loss_count = defaultdict(int)
    removed_covering = []          # removed predictions that cover truth
    ov_rm = oversize_rm = n_rm = 0
    area_rm = ov_area_rm = 0.0

    for p in (ds.pred if candidates is None else candidates):
        if not removed(p):
            continue
        n_rm += 1
        area_rm += p["area"]
        if p["covers"]:
            removed_covering.append(p)
            if p["klasse"] == "SLURV":
                oversize_rm += 1
            for j in p["covers"]:
                loss_count[j] += 1
        else:
            ov_rm += 1
            ov_area_rm += p["area"]

    lost_ids = {j for j, c in loss_count.items() if c == ds.coverage_before[j]}
    critical_rm = sum(1 for p in removed_covering
                     if any(j in lost_ids for j in p["covers"]))

    lost = len(lost_ids)
    covered_after = ds.covered_before - lost
    keep_tot = len(ds.pred) - n_rm
    correct_after = ds.n_dekkende - len(removed_covering)

    return Measurement(
        lost=lost,
        lost_pct=lost / ds.covered_before * 100 if ds.covered_before else 0.0,
        ov_rm=ov_rm,
        ov_pct=ov_rm / ds.n_miss * 100 if ds.n_miss else 0.0,
        red_rm=len(removed_covering) - critical_rm,
        critical_rm=critical_rm,
        oversize_rm=oversize_rm,
        n_rm=n_rm,
        area_rm=area_rm,
        ov_area_rm=ov_area_rm,
        covered_after=covered_after,
        recall_after=covered_after / ds.n_truth * 100 if ds.n_truth else 0.0,
        prec_after=correct_after / keep_tot * 100 if keep_tot else 0.0,
        correct_after=correct_after,
        ov_after=keep_tot - correct_after,
        net=ov_rm - cost * lost,
        ov_per_lost=(ov_rm / lost) if lost else float("inf") if ov_rm else 0.0,
        lost_boxes=sorted(lost_ids) if collect_lost else None,
    )


def baseline(ds):
    """Measurement with no filtering: the starting point."""
    return measure_filter(ds, lambda p: False)


# ── Summary ──────────────────────────────────────────────────

def write_summary(ds, write=print):
    """Writes the starting point: truth coverage, class distribution, scope."""
    b = baseline(ds)
    write(f"Truth in scope: {ds.n_truth} boxes "
          f"in {len({(f['doc_no'], f['side']) for f in ds.truth_boxes})} "
          f"(doc, page) groups")
    write(f"Predictions:    {len(ds.pred)} in scope"
          + (f"  ({len(ds.outside)} excluded, document missing from truth)"
             if ds.outside else ""))
    write(f"Scope:          {len(ds.scope_doc)} documents"
          + ("  (labelled AND processed)" if ds.outside else
             "  (all processed, documents without truth rows count as zero fnr)"))
    if ds.n_doc_unprocessed:
        write(f"                {ds.n_doc_unprocessed} labelled documents were not "
              f"processed, {ds.n_truth_unprocessed} truth boxes held out")

    per_source = defaultdict(lambda: [0, 0, 0, 0])
    for p in ds.pred:
        row = per_source[p["kilde"]]
        row[0] += 1
        row[{"TREFF": 1, "SLURV": 2, "BOM": 3}[p["klasse"]]] += 1

    write("")
    write(f"  {'source':>10} {'count':>8} {'TREFF':>8} {'SLURV':>8} "
          f"{'BOM':>8} {'BOM%':>7}")
    for k in sorted(per_source):
        n, t, s, bo = per_source[k]
        write(f"  {k:>10} {n:>8} {t:>8} {s:>8} {bo:>8} {bo/n*100:>6.1f}%")
    write(f"  {'SUM':>10} {len(ds.pred):>8} {ds.n_hit:>8} {ds.n_oversize:>8} "
          f"{ds.n_miss:>8} "
          f"{ds.n_miss/len(ds.pred)*100 if ds.pred else 0:>6.1f}%")

    uncovered = ds.n_truth - ds.covered_before
    write("")
    write(f"  Truth boxes covered by at least one prediction: "
          f"{ds.covered_before} / {ds.n_truth} "
          f"({b.recall_after:.1f}% recall before filtering)")
    write(f"  Never covered (model missed):                   {uncovered}")
    if ds.covered_before:
        write(f"  Mean coverers per covered truth box:            "
              f"{sum(ds.coverage_before) / ds.covered_before:.2f}"
              f"   ← duplicates; the higher, the more 'correctly removed' is free")
    write(f"  Precision (predictions that hit truth):         {b.prec_after:.1f}%")
    return b
