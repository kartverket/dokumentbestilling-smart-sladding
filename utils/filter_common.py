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
import hashlib
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

from geometry import intersection_area, area
from filter_rules import (FilterParam, FILTER_PARAMS, PARAM_BY_NAME,
                          PARAM_BY_CODE, GEOMETRY_PARAMETERS, OCR_PARAMS,
                          FILTER_PARAMETERS, compile_filter, is_filtered,
                          make_filter, rejection_reasons, rule_input)

try:
    from utils_config import Y_ORIGIN, HIT_THRESHOLD
except ImportError:      # running outside the repo
    Y_ORIGIN = "top"
    HIT_THRESHOLD = 0.32

SCALE = PDF_DPI / 72.0   # PDF points -> pixels

STD_SLOPPINESS_FACTOR = 3.0   # pred area > factor x covered truth area => SLURV


# ── Document ids ─────────────────────────────────────────────

def doc_no(name):
    m = re.match(r"0*(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


# ── Match metrics ────────────────────────────────────────────

def match_metrics(pn, fn, truth_horizontal):
    """All metrics for one (prediction, truth) pair; boxes in normalised coords.
    Returns None when the boxes do not overlap.

    Neighbouring text lines sit offset along the sladd's SHORT side, which is the
    height only when the page is upright, so the short-side axis is measured
    explicitly rather than y. truth_horizontal must be decided in POINT space by
    the caller: normalising skews w/h, so a near-square box can flip orientation.
    """
    o = intersection_area(pn, fn)
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
RECOMMENDED_THRESHOLDS = {"area": HIT_THRESHOLD, "short_side": 0.60,
                          "iou": 0.20, "center": 0.40}


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


# The other half of ugyldige_labels.txt: fødselsnumre the fasit does NOT have.
# A missing label has no id to point at, so the file carries the geometry
# itself, in the labels CSV's own columns, units and origin.
MISSING_LABELS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "manglende_labels.csv")

MISSING_LABEL_FIELD = ["fil_revisjon_id", "sidetall", "x", "y", "width",
                       "height", "type", "kommentar"]


def missing_label_id(row):
    """Stable id for an added row, derived from what identifies it.

    Gives code that groups by id something to hold, and lets a row that turns
    out to be wrong be retracted through ugyldige_labels.txt like any other.
    """
    parts = [str(int(float(row["fil_revisjon_id"]))),
             str(int(float(row["sidetall"])))]
    parts += [f"{float(row[k]):.2f}" for k in ("x", "y", "width", "height")]
    key = "|".join(parts).encode("utf-8")
    return "mangler-" + hashlib.sha256(key).hexdigest()[:12]


def read_missing_label_rows(path=None):
    """manglende_labels.csv as rows shaped like labels-CSV rows.

    Lines starting with «#» are comments. The file is global, like
    ugyldige_labels.txt: fil_revisjon_id names a document revision, so a row
    for a document outside the current run is never looked up.
    """
    path = path or MISSING_LABELS_FILE
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = (ln for ln in f
                 if ln.strip() and not ln.lstrip().startswith("#"))
        for r in csv.DictReader(lines):
            try:
                out = {"fil_revisjon_id": str(int(float(r["fil_revisjon_id"]))),
                       "sidetall": str(int(float(r["sidetall"]))),
                       "x": f'{float(r["x"]):.2f}',
                       "y": f'{float(r["y"]):.2f}',
                       "width": f'{float(r["width"]):.2f}',
                       "height": f'{float(r["height"]):.2f}'}
            except (TypeError, ValueError, KeyError):
                continue
            out["type"] = (r.get("type") or "").strip()
            out["ml_status"] = "MANUAL"
            out["id"] = missing_label_id(out)
            rows.append(out)
    return rows


def label_row_from_prediction(row, y_origin=None):
    """A resultat.csv row -> a manglende_labels.csv row.

    Result coordinates are pixels at PDF_DPI with the origin at the top; the
    labels CSV is PDF points with the origin Y_ORIGIN names. The page height
    needed for a «bottom» origin comes from bilde_hoyde in the same row.
    """
    x0, x1 = sorted((float(row["x0"]) / SCALE, float(row["x1"]) / SCALE))
    y0, y1 = sorted((float(row["y0"]) / SCALE, float(row["y1"]) / SCALE))
    if (y_origin or Y_ORIGIN) == "bottom":
        page_h = float(row["bilde_hoyde"]) / SCALE
        y0, y1 = page_h - y1, page_h - y0
    return {"fil_revisjon_id": str(doc_no(row["navn"])),
            "sidetall": str(int(row["side"])),
            "x": f"{x0:.2f}", "y": f"{y0:.2f}",
            "width": f"{x1 - x0:.2f}", "height": f"{y1 - y0:.2f}"}


def _label_box(row):
    """(x0, y0, x1, y1) from a labels row, or None if the geometry is unusable."""
    try:
        x, y = float(row["x"]), float(row["y"])
        w, h = float(row["width"]), float(row["height"])
    except (TypeError, ValueError, KeyError):
        return None
    x0, x1 = sorted((x, x + w))
    y0, y1 = sorted((y, y + h))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _label_key(row):
    """(doc, page) normalised, so a CSV «104822» and an added «104822.0» meet."""
    try:
        return (int(float(row["fil_revisjon_id"])), int(float(row["sidetall"])))
    except (TypeError, ValueError, KeyError):
        return None


def _same_box(a, b):
    """Whether either box's centre falls inside the other."""
    for one, other in ((a, b), (b, a)):
        cx, cy = (one[0] + one[2]) / 2.0, (one[1] + one[3]) / 2.0
        if other[0] <= cx <= other[2] and other[1] <= cy <= other[3]:
            return True
    return False


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


def reclassify_missing_covered(rows):
    """BOM rows whose box covers a manglende_labels.csv row become TREFF:
    the fasit was missing, not the fødselsnummer. label_id gets the added
    row's id, so a retraction through ugyldige_labels.txt applies to it like
    any other label. Runs at read time so a grown manglende_labels.csv
    applies to old manifests without a re-export. Run it AFTER
    reclassify_invalid_covering: a box over both an invalid label and a
    missing number must end up covering. With Y_ORIGIN «bottom» the manifest
    lacks the page height the conversion needs, and no rows are changed.
    Returns the number of rows changed."""
    if Y_ORIGIN == "bottom":
        return 0
    invalid = read_invalid_label_ids()
    per_page = defaultdict(list)
    for m in read_missing_label_rows():
        if m["id"] in invalid:
            continue
        box, key = _label_box(m), _label_key(m)
        if box and key:
            per_page[key].append((tuple(v * SCALE for v in box), m["id"]))
    if not per_page:
        return 0
    n = 0
    for r in rows:
        if r.get("klasse") != "BOM":
            continue
        try:
            key = (int(r["doc_no"]), int(r["side"]))
            box = tuple(float(r[k]) for k in ("x0", "y0", "x1", "y1"))
        except (TypeError, ValueError, KeyError):
            continue
        hit = [mid for mbox, mid in per_page.get(key, ())
               if _same_box(box, mbox)]
        if hit:
            r["klasse"] = "TREFF"
            r["label_id"] = ";".join(hit)
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

    Rows from manglende_labels.csv are yielded after the file's own, as fasit
    the labelling never got. One that the CSV has since caught up with is
    dropped, so the file retires itself rather than double-counting.

    `info`, if given, is a dict that receives "columns" (the CSV header),
    "discarded" (a per-reason tally, incl. «(ugyldig-listet)») and "added"
    (rows from manglende_labels.csv). All are set once iteration starts, so
    read them after the loop.
    """
    exclude = {str(e).strip().upper() for e in exclude_status}
    invalid = read_invalid_label_ids()
    tally = defaultdict(int)
    added = defaultdict(int)
    seen = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        leser = csv.DictReader(f)
        _warn_missing_id_column(invalid, leser.fieldnames)
        if info is not None:
            info["columns"] = leser.fieldnames or []
            info["discarded"] = tally
            info["added"] = added
        for r in leser:
            status = (r.get("ml_status") or "").strip().upper()
            if status in exclude:
                tally[status or "(empty)"] += 1
                continue
            label_id = (r.get("id") or "").strip()
            if label_id and label_id in invalid:
                tally["(ugyldig-listet)"] += 1
                continue
            box, key = _label_box(r), _label_key(r)
            if box and key:
                seen[key].append(box)
            yield r

    for r in read_missing_label_rows():
        if r["id"] in invalid:
            tally["(ugyldig-listet)"] += 1
            continue
        box, key = _label_box(r), _label_key(r)
        if box and any(_same_box(box, o) for o in seen.get(key, [])):
            tally["(allerede i fasit)"] += 1
            continue
        added["manglende_labels"] += 1
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

    outside = predictions on documents with no truth (excluded from scope),
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


def build_dataset(truth, pred, threshold=HIT_THRESHOLD,
                  oversize_factor=STD_SLOPPINESS_FACTOR, include_unlabelled=False,
                  processed_doc=None, criterion=STD_CRITERION):
    """Joins predictions to truth boxes, setting p["covers"] (truth-box indices
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

    # Limited to scope_doc: predictions outside an explicit --processed-list
    # are out of scope, not BOM.
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
    """Dataset restricted to doc_apply. Truth indices stay global (p["covers"]
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
# The rule engine lives in app/filter_rules.py, shared with prod, and is
# re-exported here so the sweep tools keep importing from filter_common.

def make_filter_per_source(per_source, only_source=None):
    """Predicate p -> bool with separate parameters per kilde.

    per_source:   {kilde: {filter params}}; a missing kilde is not filtered.
    only_source:  filter only this kilde, leaving the others untouched.
    """
    compiled = {k: compile_filter(kw) for k, kw in per_source.items() if kw}
    only = only_source.lower() if only_source is not None else None

    def _removed(p):
        source = p["kilde"].lower()
        if only is not None and source != only:
            return False
        removed = compiled.get(source)
        return removed(p) if removed else False
    return _removed


def parse_per_source(spec_list):
    """Parses "kilde:e=V,h=V,b=V,c=V,..." -> {kilde: {params}}.

    The keys are the short codes in FILTER_PARAMS; see that registry for the
    full list and the unit of each one.

    Units: h/hmin/b/bmin in pt, a in pt², amin in px² (like MIN_BOX_AREA).
    The OCR rules apply to kilde "yolo" only and the window rules (luke, desluke)
    to paddle/begge; see _ocr_reason in filter_rules. Keys carrying a bound rather than 0/1:
    rveto/lveto = OCR rules apply only above that read quality, cfritak = they
    stand down above that detection conf, run610 = reject digit runs 6..V
    (1 = 6..10), luke = reject a gap of >= V digit widths, utconf = boxes without
    text need conf >= V (prod 0.40), orgord = 2 rejects only without fnr candidate.
    """
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
            if key not in PARAM_BY_CODE:
                raise ValueError(f"Unknown parameter {key!r} in {spec!r}. "
                                 f"Valid: {', '.join(PARAM_BY_CODE)}")
            kwargs[PARAM_BY_CODE[key].name] = float(value)
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
