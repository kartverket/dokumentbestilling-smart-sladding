"""
Ground-truth-centric evaluation of filter configurations.

    lost    = ground-truth boxes that lose ALL coverage (removing one
              prediction while another still covers the box costs nothing)
    ov.rm   = pure oversladdinger (BOM) removed
    red.rm  = covering predictions removed without loss, free gain

The Pareto front keeps only the non-dominated configurations: per level of
`lost`, the one removing most oversladdinger. Without --holdout, "best of 500
configurations" is mostly overfitting.

Run:
    python utils/filter_sweep.py --truth-csv labels.csv --res-csv resultat.csv \\
        --cost 20 --holdout 0.3 --out /tmp/sweep.txt
"""

import argparse
import os
import sys
from collections import namedtuple
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from filter_common import (FILTER_PARAMS, RECOMMENDED_THRESHOLDS, CRITERIA,
                           STD_CRITERION,
                           STD_SLOPPINESS_FACTOR, HIT_THRESHOLD, baseline,
                           build_dataset, measure_filter, make_filter,
                           make_filter_per_source, read_truth_boxes, read_processed_docs, read_predictions,
                           match_metrics, pareto_front, write_summary,
                           split_by_document)

# Default curve for --threshold-list. HIT_THRESHOLD is in it on purpose: the
# curve only reads as a curve when it passes through the operating point.
STD_THRESHOLD_LIST = sorted({0.15, 0.25, HIT_THRESHOLD, 0.40, 0.50, 0.70, 0.90})

# spec = {None: kwargs} for a global filter, else {kilde: kwargs}.
Row = namedtuple("Row", "m label spec")


def make_predicate(spec):
    if None in spec:
        return make_filter(**spec[None])
    return make_filter_per_source(spec)


# ── Sorting ────────────────────────────────────────────────

SORT_FNS = {
    "net":   lambda m: (-m.net, m.lost),
    "ov.rm":   lambda m: (-m.ov_rm, m.lost),
    "lost":    lambda m: (m.lost, -m.ov_rm),
    "recall":  lambda m: (-m.recall_after, -m.ov_rm),
    "prec":    lambda m: (-m.prec_after, -m.net),
    "ov/lost": lambda m: (-m.ov_per_lost, m.lost),
}
SORT_ALIAS = {"correct.rm": "lost", "ov/correct": "ov/lost"}


def _sort_fn(name):
    return SORT_FNS[SORT_ALIAS.get(name, name)]


# ── Table format ─────────────────────────────────────────────

HEADER_TARGET = (f" {'lost':>6} {'lost%':>7} │ {'ov.rm':>7} {'ov%':>6} │"
             f" {'red.rm':>7} │ {'net':>9} {'ov/lost':>8} │"
             f" {'recall%':>8} {'pres%':>7}")


def _target_cells(m):
    ov_lost = f"{m.ov_per_lost:.1f}" if m.lost else ("∞" if m.ov_rm else "–")
    return (f" {m.lost:>6} {m.lost_pct:>6.2f}% │ {m.ov_rm:>7} {m.ov_pct:>5.1f}% │"
            f" {m.red_rm:>7} │ {m.net:>+9.0f} {ov_lost:>8} │"
            f" {m.recall_after:>7.2f}% {m.prec_after:>6.1f}%")


def _g(v):
    return f"{v:g}" if v is not None else "off"


def _hidden(m, max_lost, max_lost_pct, min_ov_lost):
    return ((max_lost is not None and m.lost > max_lost)
            or (max_lost_pct is not None and m.lost_pct > max_lost_pct)
            or (min_ov_lost is not None and m.ov_per_lost <= min_ov_lost))


def _hidden_text(n, max_lost, max_lost_pct, min_ov_lost):
    requirements = []
    if max_lost is not None:
        requirements.append(f"lost > {max_lost:g}")
    if max_lost_pct is not None:
        requirements.append(f"lost > {max_lost_pct:g}%")
    if min_ov_lost is not None:
        requirements.append(f"ov/lost ≤ {min_ov_lost:g}")
    return f"  ({n} rows hidden: {' or '.join(requirements)})"


def review_command(spec):
    """Rebuilds the filter as arguments to filter_review.py."""
    def _pair(kw):
        return ",".join(f"{fp.code}={kw[fp.name]:g}" for fp in FILTER_PARAMS
                        if kw.get(fp.name) is not None)
    if None in spec:
        kw = spec[None]
        parts = [fp.flag if fp.arg == "flag" else f"{fp.flag} {kw[fp.name]:g}"
                 for fp in FILTER_PARAMS if kw.get(fp.name) is not None]
        return " ".join(parts) or "(no filter)"
    parts = [f'"{k}:{_pair(kw)}"' for k, kw in sorted(spec.items()) if _pair(kw)]
    return ("--per-source " + " ".join(parts)) if parts else "(no filter)"


# ── Sweeps ───────────────────────────────────────────────────

def _sweep_one_param(ds, name, values, filter_fn, cost):
    print(f"\n{'─' * 118}")
    print(f"Sweep: {name}")
    print(f"{'─' * 118}")
    print(f"  {'Value':>8} │{HEADER_TARGET}")
    print(f"  {'─' * 8}─┼{'─' * 106}")
    for v in values:
        m = measure_filter(ds, make_filter(**filter_fn(v)), cost=cost)
        print(f"  {_g(v):>8} │{_target_cells(m)}")


STD_FIELD = ("min_elongation", "max_height", "max_width", "conf_threshold")
STD_HEADERS = ("elong", "height", "width", "conf≥")


def _sweep_combinations(ds, elong_v, height_v, width_v, conf_v, cost,
                         sort_key, field=STD_FIELD, headers=STD_HEADERS,
                         title=None, only_source=None, only_front=True,
                         max_lost=None, max_lost_pct=None, min_ov_lost=None,
                         csv_rows=None, max_rows=None, label_prefix=""):
    """Sweeps all combinations. only_source filters just that kilde but
    measures globally, since coverage comes from all kilder combined.

    label_prefix marks the rows in the shared Pareto table: the four axes
    mean different things per grid, so "off/6/off/off" is otherwise unreadable.
    """
    candidates = ds.per_source[only_source] if only_source else None

    filter_info = ""
    if max_lost is not None:
        filter_info += f"  [lost ≤ {max_lost:g}]"
    if max_lost_pct is not None:
        filter_info += f"  [lost ≤ {max_lost_pct:g}%]"
    if min_ov_lost is not None:
        filter_info += f"  [ov/lost > {min_ov_lost:g}]"

    has_conf = any(c is not None for c in conf_v)
    h0, h1, h2, h3 = headers
    param_header = (f"  {h0:>6} {h1:>6} {h2:>7} {h3:>6} │"
                  if has_conf else f"  {h0:>6} {h1:>6} {h2:>7} │")

    rows = []
    for min_e, max_h, max_b, c_t in product(elong_v, height_v, width_v, conf_v):
        kw = dict(zip(field, (min_e, max_h, max_b, c_t)))
        m = measure_filter(ds, make_filter(**kw), cost=cost, candidates=candidates)
        label = (label_prefix
                   + f"{_g(min_e)}/{_g(max_h)}/{_g(max_b)}/{_g(c_t)}"
                   + (f" [{only_source}]" if only_source else ""))
        rows.append(Row(m, label, {only_source: kw} if only_source else {None: kw}))
        if csv_rows is not None:
            row = {"scope": only_source or "all"}
            for fp in FILTER_PARAMS:
                row[fp.name] = kw.get(fp.name)
            csv_rows.append({
                **row,
                "lost": m.lost, "lost_pct": round(m.lost_pct, 4),
                "ov_rm": m.ov_rm, "ov_pct": round(m.ov_pct, 3),
                "red_rm": m.red_rm, "slurv_rm": m.oversize_rm,
                "critical_rm": m.critical_rm, "n_rm": m.n_rm,
                "ov_area_rm_pt2": round(m.ov_area_rm),
                "net": round(m.net, 2),
                "recall_after": round(m.recall_after, 4),
                "prec_after": round(m.prec_after, 3),
            })

    relevant = [r for r in rows
                if not _hidden(r.m, max_lost, max_lost_pct, min_ov_lost)]
    n_hidden = len(rows) - len(relevant)

    if only_front:
        show = pareto_front(relevant)
        note = (f"Pareto front: {len(show)} of {len(rows)} configurations "
                f"— the rest are dominated or equivalent")
    else:
        show = sorted(relevant, key=lambda r: _sort_fn(sort_key)(r.m))
        note = f"all {len(show)} configurations, sorted by: {sort_key}"
    if max_rows is not None:
        show = show[:max_rows]

    print(f"\n{'═' * 145}")
    print(f"{title or 'COMBINATION SWEEP'}"
          f"   ({ds.covered_before} covered ground-truth boxes, "
          f"{ds.n_miss} oversladdinger"
          + (f", filter only on '{only_source}' ({len(candidates)} pred), "
             f"rest untouched" if only_source else "") + ")")
    print(f"  {note}   [cost {cost:g}]{filter_info}")
    print(f"{'═' * 145}")
    print(param_header + HEADER_TARGET)
    print(f"  {'─' * (len(param_header) - 4)}┼{'─' * 106}")

    for row in show:
        e, h, b, c = (row.spec[only_source if only_source else None][n]
                      for n in field)
        params = (f"  {_g(e):>6} {_g(h):>6} {_g(b):>7} {_g(c):>6} │"
                  if has_conf else f"  {_g(e):>6} {_g(h):>6} {_g(b):>7} │")
        print(params + _target_cells(row.m))

    if n_hidden:
        print(_hidden_text(n_hidden, max_lost, max_lost_pct, min_ov_lost))
    return rows


def _sweep_cross_sources(ds, per_source_rows, cost, sort_key, max_candidates=8,
                        field=STD_FIELD,
                        max_lost=None, max_lost_pct=None, min_ov_lost=None):
    """Combines the best candidates per kilde and measures globally.

    Candidates are pruned with the SAME objective the final table sorts on,
    else the optimum can be pruned away before the cross product.
    """
    sources = sorted(per_source_rows)
    if len(sources) < 2:
        return []

    sort_fn = _sort_fn(sort_key)
    candidates = []
    for k in sources:
        best = sorted(per_source_rows[k], key=lambda r: sort_fn(r.m))[:max_candidates]
        candidates.append([(k, r.spec[k]) for r in best])

    rows = []
    for combo in product(*candidates):
        spec = {k: kw for (k, kw) in combo}
        m = measure_filter(ds, make_filter_per_source(spec), cost=cost)
        label = "  ".join(
            f"{k} " + "/".join(_g(kw.get(n)) for n in field)
            for k, kw in sorted(spec.items()))
        rows.append(Row(m, label, spec))

    print(f"\n{'═' * 145}")
    print("CROSS-KILDE SWEEP  (independent parameters per kilde, measured globally)")
    print(f"  Pareto front of {len(rows)} combinations "
          f"[cost {cost:g}, top {max_candidates} candidates per kilde]")
    print(f"{'═' * 145}")

    axis_name = "/".join(n.split("_")[-1][:4] for n in field)
    column = max(24, max(len(k) for k in sources) + len(axis_name) + 6)
    header = "  " + "  │  ".join(f"{k + f' ({axis_name})':>{column}}"
                               for k in sources)
    print(header + "  │" + HEADER_TARGET)
    print(f"  {'─' * (len(header) + 106)}")

    relevant = [r for r in rows
                if not _hidden(r.m, max_lost, max_lost_pct, min_ov_lost)]
    for row in pareto_front(relevant)[:15]:
        cells = "  │  ".join(
            f"{f'{k} ' + '/'.join(_g(kw.get(n)) for n in field):>{column}}"
            for k, kw in sorted(row.spec.items()))
        print("  " + cells + "  │" + _target_cells(row.m))
    n_hidden = len(rows) - len(relevant)
    if n_hidden:
        print(_hidden_text(n_hidden, max_lost, max_lost_pct, min_ov_lost))
    return rows


def _sweep_threshold(truth, pred, thresholds, chosen, oversize_factor,
                   include_unlabelled, processed, criterion=STD_CRITERION):
    """Shows how the baseline moves with the overlap threshold.

    MONOTONE within one criterion: a box covered at 40 % is necessarily covered
    at 15 %, so raising the threshold only REMOVES hits. To see hits appear, the
    rule itself must change. See --criterion-diff.
    """
    print(f"\n{'─' * 118}")
    print(f"Sweep: OVERLAP THRESHOLD (criterion «{criterion}», "
          f"baseline without geometry filters)")
    print(f"{'─' * 118}")
    print(f"  {'Thresh':>8} │ {'TREFF':>8} {'SLURV':>8} {'BOM':>8} │"
          f" {'covered':>8} {'missing':>8} {'recall%':>8} {'pres%':>7} "
          f"{'covers/box':>13}")
    print(f"  {'─' * 8}─┼─{'─' * 88}")
    for t in thresholds:
        d = build_dataset(truth, pred, threshold=t, oversize_factor=oversize_factor,
                          include_unlabelled=include_unlabelled,
                          processed_doc=processed, criterion=criterion)
        b = baseline(d)
        mean = sum(d.coverage_before) / d.covered_before if d.covered_before else 0
        marker = " ◀" if abs(t - chosen) < 1e-9 else ""
        print(f"  {t:>8.2f} │ {d.n_hit:>8} {d.n_oversize:>8} {d.n_miss:>8} │"
              f" {d.covered_before:>8} {d.n_truth - d.covered_before:>8}"
              f" {b.recall_after:>7.2f}% {b.prec_after:>6.1f}% {mean:>13.2f}"
              f"{marker}")


# ── Criterion diff ────────────────────────────────────────────

def _state(truth, pred, criterion, threshold, oversize_factor,
              include_unlabelled, processed):
    """Snapshot of one match rule: who is covered, and how each pred is classified.

    build_dataset MUTATES the prediction dicts, so the snapshot must be taken
    BEFORE the next rule is built, or a rule is compared with itself.
    Ground-truth indices stay stable because scope depends on the
    labelled/processed documents, not on the threshold.
    """
    ds = build_dataset(truth, pred, threshold=threshold, oversize_factor=oversize_factor,
                       include_unlabelled=include_unlabelled, processed_doc=processed,
                       criterion=criterion)
    # In-scope predictions only: build_dataset forces predictions on unlabelled
    # documents to BOM, which would inflate the BOM cells and make the sums
    # disagree with ds.n_miss.
    i_scope = {id(p) for p in ds.pred}
    return {
        "ds": ds,
        "covered": [d > 0 for d in ds.coverage_before],
        "klasse": [p.get("klasse") if id(p) in i_scope else None for p in pred],
        "label": f"{criterion} ≥ {threshold:.0%}" if criterion != "center"
                   else f"{criterion} ≤ {threshold:.0%}",
    }


def _diff_criteria(truth, pred, spec_a, spec_b, oversize_factor,
                    include_unlabelled, processed, out_csv=None):
    """What moves when the match rule is switched from A to B.

    Changing the rule form (areal -> kortside) can both remove and add hits.
    The ones that COME IN are interesting: today they count as oversladding.
    """
    a = _state(truth, pred, *spec_a, oversize_factor, include_unlabelled, processed)
    b = _state(truth, pred, *spec_b, oversize_factor, include_unlabelled, processed)
    ds_a, ds_b = a["ds"], b["ds"]

    print(f"\n{'═' * 118}")
    print(f"CRITERION DIFF   A = {a['label']}   →   B = {b['label']}")
    print(f"{'═' * 118}")

    # ── Ground-truth boxes: 2x2 ──
    begge = only_a = only_b = none = 0
    for da, db in zip(a["covered"], b["covered"]):
        if da and db:
            begge += 1
        elif da:
            only_a += 1
        elif db:
            only_b += 1
        else:
            none += 1
    n = len(a["covered"])
    print(f"\n  GROUND-TRUTH BOXES ({n} in scope), counted as hit:")
    print(f"    {'':<26}{'B: hit':>14}{'B: missing':>14}")
    print(f"    {'A: hit':<26}{begge:>14}{only_a:>14}   ← {only_a} DROPPED OUT")
    print(f"    {'A: missing':<26}{only_b:>14}{none:>14}")
    print(f"    {'':<26}{'↑':>14}")
    print(f"    {only_b} CAME IN: these count as BOM/oversladding today")
    print(f"\n    recall A: {(begge + only_a) / n * 100:.2f}%    "
          f"recall B: {(begge + only_b) / n * 100:.2f}%    "
          f"net: {(only_b - only_a) / n * 100:+.2f} pp")

    # ── Predictions: 3x3 ──
    classes = ("TREFF", "SLURV", "BOM")
    cross = {(x, y): 0 for x in classes for y in classes}
    for ka, kb in zip(a["klasse"], b["klasse"]):
        if ka in classes and kb in classes:
            cross[(ka, kb)] += 1
    print(f"\n  PREDICTIONS, klasse under A (row) vs B (column):")
    # Kept out of the f-string: Python < 3.12 forbids a backslash inside the
    # expression part of an f-string.
    header = "A \\ B"
    print(f"    {header:<10}" + "".join(f"{k:>10}" for k in classes) + f"{'sum':>10}")
    for ka in classes:
        row = [cross[(ka, kb)] for kb in classes]
        print(f"    {ka:<10}" + "".join(f"{v:>10}" for v in row) + f"{sum(row):>10}")
    print(f"    {'sum':<10}" + "".join(
        f"{sum(cross[(ka, kb)] for ka in classes):>10}" for kb in classes))

    miss_to_hit = sum(cross[("BOM", k)] for k in ("TREFF", "SLURV"))
    hit_to_miss = sum(cross[(k, "BOM")] for k in ("TREFF", "SLURV"))
    print(f"\n    BOM → covering: {miss_to_hit:>7}  "
          f"(counted as oversladding today, but hits a field under B)")
    print(f"    covering → BOM: {hit_to_miss:>7}  "
          f"(counted as a hit today, but is the wrong field under B)")
    print(f"    oversladding:   {ds_a.n_miss} → {ds_b.n_miss}  "
          f"({(ds_b.n_miss - ds_a.n_miss) / ds_a.n_miss * 100:+.1f}%)"
          if ds_a.n_miss else "")

    # ── CSV for follow-up: every box that moved ──
    if out_csv:
        import csv as _csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["hva", "retning", "doc_no", "side", "x0", "y0", "x1", "y1",
                        "cov_area", "cov_short", "iou", "center_short"])
            for j, (da, db) in enumerate(zip(a["covered"], b["covered"])):
                if da == db:
                    continue
                fb = ds_a.truth_boxes[j]
                # Best prediction on the same page, to show WHAT hit
                best, bm = None, None
                for pp in ds_a.pred:
                    if (pp["doc_no"], pp["side"]) != (fb["doc_no"], fb["side"]):
                        continue
                    m = match_metrics(pp["norm"], fb["norm"], fb["horizontal"])
                    if m and (bm is None or m["cov_area"] > bm["cov_area"]):
                        best, bm = pp, m
                w.writerow(["fasit", "falt_ut" if da else "kom_til",
                            fb["doc_no"], fb["side"], *[f"{v:.2f}" for v in fb["box"]],
                            *([f"{bm[k]:.3f}" for k in
                               ("cov_area", "cov_short", "iou", "center_short")]
                              if bm else ["", "", "", ""])])
            for pp, ka, kb in zip(pred, a["klasse"], b["klasse"]):
                if ka == kb or ka is None or kb is None:
                    continue
                w.writerow(["pred", f"{ka}->{kb}", pp["doc_no"], pp["side"],
                            *[f"{v:.1f}" for v in pp["px"]], "", "", "", ""])
        print(f"\n    Moved boxes written to {out_csv}")
        print(f"    Draw them for inspection with filter_review.py --select <the pdfs>")


# ── Shape analysis ──────────────────────────────────────────────

TARGET = (("elongation", "elongation", 2), ("short_side", "kortside (pt)", 1),
        ("long_side", "langside (pt)", 1), ("areal_px", "area (px²)", 0))

PERCENTILES = (0.1, 1, 50, 99, 99.9)


def _percentile(rows_sorted, pct):
    if not rows_sorted:
        return 0.0
    i = (len(rows_sorted) - 1) * pct / 100.0
    low, high = int(i), min(int(i) + 1, len(rows_sorted) - 1)
    return rows_sorted[low] + (rows_sorted[high] - rows_sorted[low]) * (i - low)


def _sweep_distribution(ds):
    """Shape percentiles, per kilde and klasse.

    A sladd box covers the last 5 digits of an fnr, so its shape is physically
    bounded: a BOM box outside every TREFF box ever seen is impossible, not
    merely unusual.
    """
    print(f"\n{'═' * 145}")
    print("SHAPE DISTRIBUTION  (what a 5-digit sladding actually looks like)")
    print(f"{'═' * 145}")
    header = f"  {'kilde':>8} {'klasse':>7} {'n':>7} │ {'metric':<12}"
    for pct in PERCENTILES:
        header += f" {('p' + format(pct, 'g')):>9}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for source in ds.sources():
        for klasse in ("TREFF", "SLURV", "BOM"):
            group = [p for p in ds.per_source[source] if p["klasse"] == klasse]
            if len(group) < 20:      # too few to say anything about tails
                continue
            for nr, (spec_key, name, des) in enumerate(TARGET):
                rows_sorted = sorted(p[spec_key] for p in group)
                left = (f"  {source:>8} {klasse:>7} {len(group):>7} │"
                           if nr == 0 else f"  {'':>8} {'':>7} {'':>7} │")
                row = left + f" {name:<12}"
                for pct in PERCENTILES:
                    row += f" {_percentile(rows_sorted, pct):>9.{des}f}"
                print(row)
            print(f"  {'·' * 100}")


def _derive_bounds(ds, pct, use_conf=None):
    """Per-kilde limits derived from the TREFF distribution, not from net.

    Lower = TREFF percentile `pct` from the bottom, upper = from the top.
    Nothing is fitted against ov.rm. The limits only describe which shapes
    correct sladdinger have had.
    """
    spec = {}
    for source in ds.sources():
        hit = [p for p in ds.per_source[source]
                 if p["klasse"] in ("TREFF", "SLURV")]
        if len(hit) < 100:          # too few to estimate tails
            continue
        kw = {}
        for spec_key, field_min, field_max in (
                ("elongation", "min_elongation", "max_elongation"),
                ("short_side", "min_short_side", "max_short_side"),
                ("long_side", "min_long_side", "max_long_side")):
            rows_sorted = sorted(p[spec_key] for p in hit)
            kw[field_min] = round(_percentile(rows_sorted, pct), 2)
            kw[field_max] = round(_percentile(rows_sorted, 100 - pct), 2)
        area = sorted(p["areal_px"] for p in hit)
        kw["min_area_px"] = round(_percentile(area, pct))
        if use_conf is not None:
            kw["conf_threshold"] = use_conf
        spec[source] = kw
    return spec


def _report_bounds(ds, ds_test, pct, cost):
    """Measures the derived shape limit on training and holdout."""
    print(f"\n{'═' * 145}")
    print(f"SHAPE LIMIT DERIVED FROM TREFF  (lower = p{pct:g}, "
          f"upper = p{100 - pct:g} of correct boxes per kilde)")
    print("  The limits are NOT fitted to ov.rm. They only describe which")
    print("  shapes correct 5-digit sladdinger have had.")
    print(f"{'═' * 145}")

    for mark, conf in (("without conf gate", None), ("with conf≥0.5 gate", 0.5)):
        spec = _derive_bounds(ds, pct, use_conf=conf)
        if not spec:
            print("  (too few TREFF boxes per kilde to estimate tails)")
            return
        print(f"\n  {mark}:")
        for source, kw in sorted(spec.items()):
            print(f"    {source:>8}  elong [{kw['min_elongation']:g}, "
                  f"{kw['max_elongation']:g}]  kortside [{kw['min_short_side']:g}, "
                  f"{kw['max_short_side']:g}]  langside [{kw['min_long_side']:g}, "
                  f"{kw['max_long_side']:g}]  area ≥ {kw['min_area_px']:g}px²")
        m = measure_filter(ds, make_filter_per_source(spec), cost=cost)
        print(f"    training: {_target_cells(m)}")
        if ds_test is not None:
            t = measure_filter(ds_test, make_filter_per_source(spec), cost=cost)
            print(f"    holdout:  {_target_cells(t)}")
        print(f"    → filter_review.py {review_command(spec)}")


# ── Pareto-front ─────────────────────────────────────────────

def _pareto_table(rows, cost, ds_test=None, title="PARETO-FRONT",
                   max_lost=None, max_lost_pct=None, min_ov_lost=None):
    """The non-dominated configurations: for each level of `lost`, the one
    removing most oversladdinger. With ds_test each is measured on the holdout
    set as well, so overfitting becomes visible."""
    front = pareto_front(rows, target=lambda r: (r.m.lost, r.m.ov_rm))
    front = [r for r in front
             if not _hidden(r.m, max_lost, max_lost_pct, min_ov_lost)]
    if not front:
        return []

    width = max(28, max(len(r.label) for r in front) + 2)
    print(f"\n{'═' * 145}")
    print(f"{title}   ({len(front)} non-dominated of {len(rows)} "
          f"configurations)   [cost {cost:g}]")
    if ds_test is not None:
        print("  Left block = training (where the configuration was chosen), "
              "right = holdout (independent documents).")
        print("  Δ is holdout minus training in percentage points, large "
              "negative Δov% or positive Δlost% means overfitting.")
    print(f"{'═' * 145}")

    if ds_test is None:
        print(f"  {'configuration':<{width}}│{HEADER_TARGET}")
        print(f"  {'─' * width}┼{'─' * 106}")
        for r in front:
            print(f"  {r.label:<{width}}│{_target_cells(r.m)}")
        return front

    print(f"  {'configuration':<{width}}│"
          f" {'lost':>5} {'lost%':>7} {'ov.rm':>7} {'ov%':>6} {'net':>8} │"
          f" {'lost':>5} {'lost%':>7} {'ov.rm':>7} {'ov%':>6} {'net':>8} │"
          f" {'Δlost%':>7} {'Δov%':>7}")
    print(f"  {' ' * width}│{'  training'.ljust(38)} │"
          f"{'  holdout'.ljust(38)} │")
    print(f"  {'─' * width}┼{'─' * 39}┼{'─' * 39}┼{'─' * 17}")

    result = []
    for r in front:
        t = measure_filter(ds_test, make_predicate(r.spec), cost=cost)
        print(f"  {r.label:<{width}}│"
              f" {r.m.lost:>5} {r.m.lost_pct:>6.2f}% {r.m.ov_rm:>7} "
              f"{r.m.ov_pct:>5.1f}% {r.m.net:>+8.0f} │"
              f" {t.lost:>5} {t.lost_pct:>6.2f}% {t.ov_rm:>7} "
              f"{t.ov_pct:>5.1f}% {t.net:>+8.0f} │"
              f" {t.lost_pct - r.m.lost_pct:>+6.2f}p {t.ov_pct - r.m.ov_pct:>+6.1f}p")
        result.append((r, t))
    return result


def _recommendation(front, cost, ds_test=None):
    if not front:
        return
    if ds_test is not None:
        best = max(front, key=lambda rt: rt[1].net)   # choose on holdout
        r, t = best
        print(f"\n  Best net on HOLDOUT (cost {cost:g}): {r.label}")
        print(f"    training: lost {r.m.lost} ({r.m.lost_pct:.2f}%), "
              f"ov.rm {r.m.ov_rm} ({r.m.ov_pct:.1f}%), recall {r.m.recall_after:.2f}%")
        print(f"    holdout:  lost {t.lost} ({t.lost_pct:.2f}%), "
              f"ov.rm {t.ov_rm} ({t.ov_pct:.1f}%), recall {t.recall_after:.2f}%")
        spec = r.spec
    else:
        r = max(front, key=lambda r: r.m.net)
        print(f"\n  Best net on the front (cost {cost:g}): {r.label}")
        print(f"    lost {r.m.lost} ({r.m.lost_pct:.2f}%), "
              f"ov.rm {r.m.ov_rm} ({r.m.ov_pct:.1f}%), "
              f"red.rm {r.m.red_rm}, recall {r.m.recall_after:.2f}%")
        spec = r.spec
    print(f"    filter_review.py ... {review_command(spec)}")


# ── Main ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Ground-truth-centric evaluation of filter configurations")
    p.add_argument("--truth-csv", required=True,
                   help="Labels CSV (ACCEPTED + manual = ground truth, "
                        "REJECTED excluded)")
    p.add_argument("--res-csv", required=True,
                   help="Result CSV from the model (pixel coordinates)")
    p.add_argument("--threshold", type=float, default=HIT_THRESHOLD,
                   help=f"Overlap threshold for coverage (default: {HIT_THRESHOLD})")
    p.add_argument("--criterion", default=STD_CRITERION,
                   choices=sorted(CRITERIA),
                   help="Rule for whether a prediction and a ground-truth box "
                        "are the SAME FIELD. areal = one-sided area coverage of "
                        "the ground-truth box (current). kortside = overlap "
                        "along its short side, regardless of page rotation. "
                        f"(default: {STD_CRITERION})")
    p.add_argument("--threshold-list", default=None, metavar="T,T,...",
                   help="Thresholds in the threshold sweep, comma separated "
                        f"(default: {','.join(f'{t:g}' for t in STD_THRESHOLD_LIST)})")
    p.add_argument("--criterion-diff", nargs=2, default=None,
                   metavar=("A", "B"),
                   help="Compare two match rules and show how many hits drop "
                        "out and come in. Each spec is CRITERION:THRESHOLD, "
                        "e.g. areal:0.15 kortside:0.60")
    p.add_argument("--diff-csv", default=None, metavar="FILE",
                   help="Write the boxes that moved in --criterion-diff to CSV")
    p.add_argument("--oversize-factor", type=float, default=STD_SLOPPINESS_FACTOR,
                   help="Pred area > factor × covered ground-truth area ⇒ SLURV "
                        f"(default: {STD_SLOPPINESS_FACTOR})")
    p.add_argument("--include-unlabelled", action="store_true", default=True,
                   help="(default on) Include predictions on processed "
                        "documents with no rows in the labels CSV, the labels "
                        "file covers the whole uttrekk, so those were reviewed "
                        "with zero fnr and predictions there are real "
                        "oversladdinger")
    p.add_argument("--exclude-unlabelled", dest="include_unlabelled",
                   action="store_false",
                   help="Old behaviour: keep documents without ground-truth "
                        "rows out of scope (for older labels files that did "
                        "not cover the whole uttrekk)")
    p.add_argument("--form-pct", type=float, default=0.1, metavar="P",
                   help="Percentile for the shape limit derived from TREFF "
                        "boxes: lower = pP, upper = p(100-P). Lower is more "
                        "conservative (default: 0.1)")
    p.add_argument("--processed-list", default=None, metavar="FILE",
                   help="File listing the documents the model ran on (one name "
                        "or number per line). Without it the documents in the "
                        "result CSV are assumed, and a document where the model "
                        "found nothing counts as not run.")
    p.add_argument("--cost", type=float, default=1.0,
                   help="How many removed oversladdinger one lost ground-truth "
                        "box is worth. net = ov.rm − cost × lost (default: 1)")
    p.add_argument("--holdout", type=float, default=None, metavar="FRACTION",
                   help="Hold back this fraction of the DOCUMENTS for "
                        "independent measurement (e.g. 0.3). The sweep runs on "
                        "the rest; the Pareto front is measured on both.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the holdout split (default: 42)")
    p.add_argument("--sort", default="net",
                   choices=sorted(set(SORT_FNS) | set(SORT_ALIAS)),
                   help="Sort column (default: net)")
    p.add_argument("--max-lost", type=float, default=None,
                   help="Hide rows losing more than N ground-truth boxes")
    p.add_argument("--max-lost-pct", "--max-correct-pct", type=float, default=None,
                   dest="max_lost_pct",
                   help="Hide rows losing more than this %% of covered boxes")
    p.add_argument("--min-ov-lost", type=float, default=None,
                   dest="min_ov_lost",
                   help="Only show rows where ov.rm/lost exceeds this value")
    p.add_argument("--all-rows", action="store_true",
                   help="Print every configuration in the combination tables, "
                        "not just the Pareto front. Makes a much bigger file.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Max rows per table")
    p.add_argument("--out", default=None, metavar="FILE",
                   help="Write the report to a file (default: generated name)")
    p.add_argument("--out-csv", default=None, metavar="FILE",
                   help="Write all sweep rows to CSV for further analysis")
    args = p.parse_args()

    if args.holdout is not None and not 0 < args.holdout < 1:
        p.error("--holdout must be between 0 and 1 (e.g. 0.3)")

    out_file = args.out or (
        f"filter_sweep_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.txt")

    class _Tee:
        """Writes everything to file, only selected parts to the terminal."""

        def __init__(self, file_obj, terminal):
            self.file, self.terminal = file_obj, terminal
            self.to_terminal = True

        def write(self, text):
            self.file.write(text)
            if self.to_terminal:
                self.terminal.write(text)

        def flush(self):
            self.file.flush()
            self.terminal.flush()

    file = open(out_file, "w", encoding="utf-8")
    tee = _Tee(file, sys.stdout)
    sys.stdout = tee
    try:
        truth = read_truth_boxes(args.truth_csv)
        pred = read_predictions(args.res_csv)
        processed = read_processed_docs(args.processed_list) if args.processed_list else None
        ds_full = build_dataset(truth, pred, threshold=args.threshold,
                                oversize_factor=args.oversize_factor,
                                include_unlabelled=args.include_unlabelled,
                                processed_doc=processed, criterion=args.criterion)

        print(f"Criterion «{args.criterion}», threshold {args.threshold:.0%}, "
              f"slurv factor {args.oversize_factor:g}, "
              f"cost {args.cost:g}")
        if args.criterion in RECOMMENDED_THRESHOLDS and abs(
                args.threshold - RECOMMENDED_THRESHOLDS[args.criterion]) > 1e-9:
            print(f"  Note: recommended threshold for «{args.criterion}» is "
                  f"{RECOMMENDED_THRESHOLDS[args.criterion]:.0%} "
                  f"(measured on label pairs from uttrekk 4 + 5)")
        print("")
        write_summary(ds_full)

        ds_test = None
        if args.holdout is not None:
            ds, ds_test = split_by_document(ds_full, args.holdout, args.seed)
            n_doc_train = len({p["doc_no"] for p in ds.pred})
            n_doc_test = len({p["doc_no"] for p in ds_test.pred})
            print(f"\n  Holdout split (seed {args.seed}, fraction "
                  f"{args.holdout:g}, split by document):")
            print(f"    training: {n_doc_train:>6} docs, {len(ds.pred):>7} pred, "
                  f"{ds.covered_before:>6} covered ground-truth boxes, "
                  f"{ds.n_miss:>6} oversladdinger")
            print(f"    holdout:  {n_doc_test:>6} docs, {len(ds_test.pred):>7} pred, "
                  f"{ds_test.covered_before:>6} covered ground-truth boxes, "
                  f"{ds_test.n_miss:>6} oversladdinger")
        else:
            ds = ds_full
            print("\n  (no holdout, use --holdout 0.3 to see whether the "
                  "chosen configuration holds on independent documents)")

        tee.to_terminal = False

        thresholds = ([float(t) for t in args.threshold_list.split(",")]
                    if args.threshold_list
                    else STD_THRESHOLD_LIST)
        _sweep_threshold(truth, pred, thresholds, args.threshold,
                       args.oversize_factor, args.include_unlabelled, processed,
                       criterion=args.criterion)

        if args.criterion_diff:
            spec_text = []
            for raw in args.criterion_diff:
                name, _, t = raw.partition(":")
                if name not in CRITERIA:
                    p.error(f"unknown criterion {name!r} in --criterion-diff, "
                            f"valid: {', '.join(sorted(CRITERIA))}")
                if not t:
                    p.error(f"missing threshold in {raw!r}, write e.g. {name}:0.40")
                spec_text.append((name, float(t)))
            _diff_criteria(truth, pred, spec_text[0], spec_text[1], args.oversize_factor,
                            args.include_unlabelled, processed, args.diff_csv)

        # build_dataset mutates the predictions, so rebuild with the chosen threshold
        ds_full = build_dataset(truth, pred, threshold=args.threshold,
                                oversize_factor=args.oversize_factor,
                                include_unlabelled=args.include_unlabelled,
                                processed_doc=processed, criterion=args.criterion)
        if args.holdout is not None:
            ds, ds_test = split_by_document(ds_full, args.holdout, args.seed)
        else:
            ds = ds_full

        _sweep_one_param(ds, "MIN_ELONGATION max(w/h, h/w)",
                        [1.1, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0],
                        lambda v: {"min_elongation": v}, args.cost)
        _sweep_one_param(ds, "MAX_BOX_HEIGHT_PT",
                        [25, 30, 35, 40, 45, 50, 60, 80, 100],
                        lambda v: {"max_height": v}, args.cost)
        _sweep_one_param(ds, "MAX_BOX_WIDTH_PT",
                        [60, 80, 100, 120, 150, 200, 250],
                        lambda v: {"max_width": v}, args.cost)

        has_conf = any(x["conf"] is not None for x in ds.pred)
        if has_conf:
            _sweep_one_param(
                ds, "CONF_THRESHOLD (conf≥V kept regardless of geometry; "
                    "combined with e=1.5/h=50/b=120)",
                [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                lambda v: {"min_elongation": 1.5, "max_height": 50,
                           "max_width": 120, "conf_threshold": v},
                args.cost)

        _sweep_one_param(ds, "MIN_BOX_AREA (px²), tiny boxes",
                        [500, 700, 965, 1200, 1600, 2200, 3000],
                        lambda v: {"min_area_px": v}, args.cost)
        _sweep_one_param(ds, "MIN_SHORT_SIDE_PT: too thin to be text "
                            "(orientation independent: upright boxes are safe)",
                        [3, 4, 5, 6, 7, 8, 10],
                        lambda v: {"min_short_side": v}, args.cost)
        _sweep_one_param(ds, "MIN_LONG_SIDE_PT: too short to hold 5 digits",
                        [10, 15, 20, 25, 30, 40, 50],
                        lambda v: {"min_long_side": v}, args.cost)
        _sweep_one_param(ds, "MAX_ELONGATION: thin, long strokes",
                        [6, 8, 10, 12, 15, 20, 30, 50],
                        lambda v: {"max_elongation": v}, args.cost)

        # ── OCR features: stricter variants of lenient_check ────────
        # Only kilde «yolo» with text in the box is affected. See _ocr_reason
        # in filter_rules. Everything else is untouched.
        n_features = sum(1 for x in ds.pred if x.get("har_tokens"))
        if not n_features:
            print("\n(no feature columns in the result CSV, the OCR sweep is "
                  "skipped. Re-run run.py to get them.)")
        else:
            print(f"\n{'═' * 145}")
            print("OCR FEATURES: stricter lenient_check")
            print(f"  {n_features} yolo boxes with text are in play; the other "
                  f"{len(ds.pred) - n_features} predictions are untouched")
            print(f"{'═' * 145}")

            _sweep_one_param(ds, "MIN_DIGITS in the box (current: 1)",
                            [2, 3, 4, 5, 6, 8, 11],
                            lambda v: {"min_digits": v}, args.cost)
            _sweep_one_param(ds, "MAX_LETTERS in the box (current: 1)",
                            [0, 1, 2, 3],
                            lambda v: {"max_letters": v}, args.cost)
            _sweep_one_param(ds, "MIN_DIGITS_RUN: longest digit run "
                                "overlapping the box (a coordinate = 5-7)",
                            [6, 7, 8, 9, 10, 11],
                            lambda v: {"min_digits_run": v}, args.cost)
            _sweep_one_param(ds, "REC_VETO: min_digits=2 applies only where "
                                "Paddle read the box confidently",
                            [None, 0.80, 0.90, 0.95, 0.98],
                            lambda v: {"min_digits": 2, "rec_veto": v},
                            args.cost)
            _sweep_one_param(ds, "REQUIRE_FNR_CANDIDATE: 11-digit fnr shape on "
                                "the line, with a rec_veto gate",
                            [None, 0.80, 0.90, 0.95, 0.98],
                            lambda v: {"require_fnr_candidate": 1, "rec_veto": v},
                            args.cost)
            _sweep_one_param(ds, "REJECT_DECIMAL: decimal separator in the "
                                "number, with a rec_veto gate",
                            [None, 0.80, 0.90, 0.95, 0.98],
                            lambda v: {"reject_decimal": 1, "rec_veto": v},
                            args.cost)
            _sweep_one_param(ds, "REJECT_DECIMAL + CONF EXEMPTION: the decimal "
                                "rule (rec_veto 0.98) yields at detection "
                                "conf ≥ V",
                            [None, 0.40, 0.45, 0.50, 0.60, 0.70],
                            lambda v: {"reject_decimal": 1, "rec_veto": 0.98,
                                       "ocr_conf_exempt": v},
                            args.cost)
            _sweep_one_param(ds, "REJECT_00_RUN: a 10-12 digit run starts with "
                                "00 (orgnr padded to fnr width), with a "
                                "rec_veto gate",
                            [None, 0.80, 0.90, 0.95, 0.98],
                            lambda v: {"reject_00_run": 1, "rec_veto": v},
                            args.cost)
            _sweep_one_param(ds, "REJECT_ORGNR: valid orgnr mod11 in the box, "
                                "with a rec_veto gate",
                            [None, 0.80, 0.90, 0.95, 0.98],
                            lambda v: {"reject_orgnr": 1, "rec_veto": v},
                            args.cost)
            _sweep_one_param(ds, "REJECT_ORG_ORD=1: org word near the box, "
                                "with a rec_veto gate",
                            [None, 0.80, 0.90, 0.95, 0.98],
                            lambda v: {"reject_org_ord": 1, "rec_veto": v},
                            args.cost)
            _sweep_one_param(ds, "REJECT_ORG_ORD=2: org word near the box AND "
                                "no fnr candidate, with a rec_veto gate",
                            [None, 0.80, 0.90, 0.95, 0.98],
                            lambda v: {"reject_org_ord": 2, "rec_veto": v},
                            args.cost)
            _sweep_one_param(ds, "REQUIRE_FNR_CANDIDATE + LINE_VETO: as "
                                "fnr candidate (rec_veto 0.98), but the rule "
                                "applies only once the WHOLE line is read with "
                                "rec_min_linje ≥ V (thresholds above 0.999 "
                                "need features with 5 decimals)",
                            [None, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999],
                            lambda v: {"require_fnr_candidate": 1,
                                       "rec_veto": 0.98, "line_veto": v},
                            args.cost)
            _sweep_one_param(ds, "REJECT_RUN_BAND: upper limit on run length "
                                "(6..V) with rec 0.98/line 0.99; 10-runs are "
                                "often fnr with a single-digit day/month",
                            [None, 8, 9, 10],
                            lambda v: {"reject_run_6_10": v,
                                       "rec_veto": 0.98, "line_veto": 0.99},
                            args.cost)
            _sweep_one_param(ds, "REJECT_RUN_6_9 + LINE_VETO: band 6-9, with "
                                "rec_veto 0.98 and line_veto ≥ V",
                            [None, 0.98, 0.99, 0.995, 0.999, 0.9999],
                            lambda v: {"reject_run_6_10": 9,
                                       "rec_veto": 0.98, "line_veto": v},
                            args.cost)
            _sweep_one_param(ds, "WITHOUT_TEXT_CONF: boxes without OCR text "
                                "require conf ≥ V (prod: 0.40; hits the "
                                "graphics/map detections the text rules never "
                                "see)",
                            [None, 0.45, 0.50, 0.60, 0.70, 0.80],
                            lambda v: {"without_text_conf": v},
                            args.cost)

        # ── Window features: paddle boxes stitched over decimals/column gaps ──
        n_window = sum(1 for x in ds.pred if x.get("maks_luke") is not None)
        if not n_window:
            print("\n(no window features in the result CSV, the paddle "
                  "window sweep is skipped. Re-run run.py to get them.)")
        else:
            print(f"\n{'═' * 145}")
            print("PADDLE WINDOW: the 11-digit window the box was built from")
            print(f"  {n_window} paddle/begge boxes with window features are "
                  f"in play; everything else is untouched")
            print(f"{'═' * 145}")
            _sweep_one_param(ds, "MAX_GAP (GLOBAL, incl. begge loss, see the "
                                "per-kilde grid for the decision), largest "
                                "physical gap in the window, in digit widths. "
                                "Coordinate stitching («6626630.58 549810.29») "
                                "and sketch measurements give wide gaps; a real "
                                "fnr sits edge to edge",
                            [1.5, 2, 3, 4, 6, 8, 12],
                            lambda v: {"max_gap": v}, args.cost)
            _sweep_one_param(ds, "REJECT_DECIMAL_GAP: a gap in the window "
                                "contains a decimal separator (. or ,)",
                            [1],
                            lambda v: {"reject_decimal_gap": v}, args.cost)
            _sweep_one_param(ds, "DECIMAL_GAP + MAX_GAP combined",
                            [None, 1.5, 2, 3, 4, 6],
                            lambda v: {"reject_decimal_gap": 1,
                                       "max_gap": v}, args.cost)

        _sweep_distribution(ds)
        _report_bounds(ds, ds_test, args.form_pct, args.cost)

        elong_v = [None, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0]
        height_v = [None, 40, 50, 60, 80]
        width_v = [None, 80, 100, 120, 150]
        conf_v = [None, 0.5] if has_conf else [None]

        csv_rows = [] if args.out_csv else None
        bounds = dict(max_lost=args.max_lost,
                       max_lost_pct=args.max_lost_pct,
                       min_ov_lost=args.min_ov_lost)
        common = dict(cost=args.cost, sort_key=args.sort,
                      csv_rows=csv_rows, max_rows=args.max_rows,
                      only_front=not args.all_rows, **bounds)

        all_rows = _sweep_combinations(ds, elong_v, height_v, width_v,
                                          conf_v, **common)

        # OCR axes only hit yolo boxes with text, but their rows join the
        # shared Pareto front so they compete against the geometry limits.
        ocr_rows = []
        if n_features:
            # reject_decimal is in BOTH grids on purpose: find_fnr accepts "."
            # and "," as a gap between digit pieces, which is right for an fnr
            # the OCR split up, but on a coordinate line it stitches two
            # neighbouring numbers into a valid-looking 11-digit run
            # («370600.83 -56912.29»). There the decimal is the discriminator,
            # not the candidate.
            ocr_rows = _sweep_combinations(
                ds, [None, 2, 3, 5, 8, 11], [None, 6, 7, 9, 11],
                [None, 1], [None, 0.80, 0.90, 0.95, 0.98],
                field=("min_digits", "min_digits_run", "reject_decimal",
                      "rec_veto"),
                headers=("s.min", "run", "des", "rec≥"),
                title="OCR FEATURES COMBINED: digit requirements "
                       "(hits only «yolo» with text)",
                label_prefix="ocr-siffer ", **common)
            ocr_rows += _sweep_combinations(
                ds, [None, 6, 7, 9, 11], [None, 1], [None, 1],
                [None, 0.80, 0.90, 0.95, 0.98],
                field=("min_digits_run", "require_fnr_candidate", "reject_decimal",
                      "rec_veto"),
                headers=("run", "fnr", "des", "rec≥"),
                title="OCR FEATURES COMBINED: fnr candidate "
                       "(hits only «yolo» with text)",
                label_prefix="ocr-fnr ", **common)
            ocr_rows += _sweep_combinations(
                ds, [None, 4, 5, 6], [None, 0.95, 0.98], [None, 0.99],
                [None, 0.5, 0.6],
                field=("min_digits", "rec_veto", "line_veto",
                      "ocr_conf_exempt"),
                headers=("s.min", "rec≥", "linje≥", "cfrit"),
                title="OCR FEATURES COMBINED: digit requirement with line "
                       "veto and conf exemption (hits only «yolo» with text)",
                label_prefix="ocr-smin ", **common)
            # Orgnr rules with line veto: mod11 is worthless if one digit is
            # misread, so the line veto is the natural gate.
            ocr_rows += _sweep_combinations(
                ds, [None, 1], [None, 1], [None, 0.99, 0.999],
                [None, 0.5],
                field=("reject_00_run", "reject_orgnr", "line_veto",
                      "ocr_conf_exempt"),
                headers=("00run", "orgnr", "linje≥", "cfrit"),
                title="OCR FEATURES COMBINED: orgnr rules with line veto "
                       "(hits only «yolo» with text)",
                label_prefix="ocr-orglinje ", **common)
            # fnr candidate and run length 6-10, gated on the read quality of
            # the whole line; cfritak lets a high YOLO conf protect a real fnr.
            ocr_rows += _sweep_combinations(
                ds, [None, 1], [None, 9, 10], [None, 0.99, 0.995, 0.999, 0.9999],
                [None, 0.5, 0.6],
                field=("require_fnr_candidate", "reject_run_6_10", "line_veto",
                      "ocr_conf_exempt"),
                headers=("fnr", "r.maks", "linje≥", "cfrit"),
                title="OCR FEATURES COMBINED: line evidence, fnr candidate "
                       "and run length with line veto (the line veto covers "
                       "the box's own tokens and is therefore stricter than "
                       "rec_veto; hits only «yolo» with text)",
                label_prefix="ocr-linje ", **common)
            ocr_rows += _sweep_combinations(
                ds, [None, 1], [None, 1], [None, 1, 2],
                [None, 0.90, 0.95, 0.98],
                field=("reject_00_run", "reject_orgnr", "reject_org_ord",
                      "rec_veto"),
                headers=("00run", "orgnr", "orgord", "rec≥"),
                title="OCR FEATURES COMBINED: orgnr rules "
                       "(hits only «yolo» with text)",
                label_prefix="ocr-org ", **common)
            # Decimal rule with conf exemption: real fnr losses sit at high
            # detection conf, the coordinates the rule targets sit low. The
            # fourth axis is a dummy.
            ocr_rows += _sweep_combinations(
                ds, [None, 1], [None, 0.90, 0.95, 0.98],
                [None, 0.40, 0.45, 0.50, 0.60, 0.70], [None],
                field=("reject_decimal", "rec_veto", "ocr_conf_exempt",
                      "min_digits"),
                headers=("des", "rec≥", "cfrit", "-"),
                title="OCR FEATURES COMBINED: decimal rule with conf "
                       "exemption (hits only «yolo» with text)",
                label_prefix="ocr-cfrit ", **common)
            # Paddle window: boxes from 11-windows stitched over decimal
            # separators and column gaps. ONLY kilde paddle, globally the
            # rules also hit the begge hits, because a real fnr is written in
            # date form («01.01.50 12345» gives decimal gaps after digit 2 and
            # 4), OCR adds noise dots, and form fields split fnr physically.
            if any(x.get("maks_luke") is not None for x in ds.pred):
                ocr_rows += _sweep_combinations(
                    ds, [None, 1], [None, 1.5, 2, 3, 4, 6, 8], [None], [None],
                    field=("reject_decimal_gap", "max_gap",
                          "min_digits", "rec_veto"),
                    headers=("desluke", "luke≥", "-", "-"),
                    title="PADDLE WINDOW COMBINED: decimal gap and physical "
                           "gap width (ONLY kilde paddle)",
                    only_source="paddle",
                    label_prefix="p-vindu ", **common)
            all_rows += ocr_rows

        NOISE_FIELD_G = ("min_short_side", "min_long_side", "max_elongation",
                       "conf_threshold")
        noise_rows = _sweep_combinations(
            ds, [None, 4, 5, 6, 7], [None, 15, 20, 25, 30],
            [None, 6, 8, 10, 12, 15], conf_v,
            field=NOISE_FIELD_G, headers=("k.min", "l.min", "e.maks", "conf≥"),
            title="NOISE FILTERS: too small or too thin to be 5 digits",
            **common)

        NOISE_FIELD = ("min_short_side", "min_long_side", "max_elongation",
                     "conf_threshold")
        NOISE_HEADERS = ("k.min", "l.min", "e.maks", "conf≥")
        NOISE_AXES = ([None, 4, 5, 6, 7], [None, 15, 20, 25, 30],
                      [None, 6, 8, 10, 12, 15])

        sources = ds.sources()
        # Per kilde on the noise axes: paddle boxes are tight 5-digit boxes
        # with a narrow shape range, yolo boxes are raw detections. A threshold
        # that is free for one kilde can cost on another.
        noise_per_source = {}
        for source in sources:
            k_conf = ([None, 0.5]
                      if any(x["conf"] is not None for x in ds.per_source[source])
                      else [None])
            noise_per_source[source] = _sweep_combinations(
                ds, *NOISE_AXES, k_conf, field=NOISE_FIELD, headers=NOISE_HEADERS,
                title=f"NOISE FILTERS PER KILDE: {source.upper()}",
                only_source=source, **common)
            all_rows += noise_per_source[source]

        cross_rows = []
        if len(sources) > 1:
            per_source_rows = {}
            for source in sources:
                k_conf = ([None, 0.5]
                          if any(x["conf"] is not None for x in ds.per_source[source])
                          else [None])
                per_source_rows[source] = _sweep_combinations(
                    ds, elong_v, height_v, width_v, k_conf,
                    title=f"PER KILDE: {source.upper()}", only_source=source,
                    **common)
                all_rows += per_source_rows[source]
            cross_rows = _sweep_cross_sources(
                ds, per_source_rows, args.cost, args.sort, **bounds)
            cross_rows += _sweep_cross_sources(
                ds, noise_per_source, args.cost, args.sort,
                field=NOISE_FIELD, **bounds)

        # ── Pareto fronts: the only section that also goes to the terminal
        tee.to_terminal = True
        front = _pareto_table(
            all_rows + noise_rows + cross_rows, args.cost, ds_test=ds_test,
            title="PARETO FRONT: all configurations (shared, per kilde "
                   "and cross-kilde)", **bounds)
        _recommendation(front, args.cost, ds_test=ds_test)
        tee.to_terminal = False

        if args.out_csv and csv_rows:
            import csv as _csv
            field_name = list(csv_rows[0])
            for r in csv_rows:
                for k in r:
                    if k not in field_name:
                        field_name.append(k)
            with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=field_name)
                w.writeheader()
                w.writerows(csv_rows)
    finally:
        sys.stdout = tee.terminal
        file.close()

    print(f"\n✓ Report written to: {out_file} "
          f"({os.path.getsize(out_file) // 1024} KB)")
    if args.out_csv:
        print(f"✓ Sweep rows written to: {args.out_csv}")


if __name__ == "__main__":
    main()
