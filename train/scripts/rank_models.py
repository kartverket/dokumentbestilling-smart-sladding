"""Ranks candidate models on the miss-versus-oversladd economy.

Each model's resultat.csv must come from a validation run over the SAME
document list. Predictions join fasit through build_dataset under the
whole-uttrekk regime: a processed document without rows counts as zero fnr,
so --processed-list is required, or documents where the model found nothing
would silently leave the scope. The YOLO confidence cutoff is swept per model;
cost = cost_w * missed fnr + oversladd boxes, and the winner is the
(model, conf) pair. Misses are also reported on manual-only fasit, since
ML-accepted rows are the incumbent's own suggestions and the manual slice is
the number that escapes that circularity, plus per decade and per
rettsstiftelseskode (a doc with several codes counts under each), with
WORST_DECADE/WORST_CODE lines for the specialist probe to read.

Kjør:
    python train/scripts/rank_models.py \
        --truth-csv "$SLADD_LABELS/uttrekk_6.csv" \
        --processed-list "$SLADD_LISTS/uttrekk_6_holdout48.txt" \
        --metadata "$SLADD_METADATA/uttrekk_6.csv" \
        --run 48t_l=/data2/validering/48t_l_.../resultat.csv \
        --out rangering.csv
"""

import argparse
import csv
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for sub in ("utils", "app"):
    if str(_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(_ROOT / sub))
from filter_common import (build_dataset, iter_label_rows, read_predictions,
                           read_processed_docs, read_truth_boxes)

try:
    from config import YOLO_CONF as STD_CONF
except ImportError:
    STD_CONF = 0.12

MAX_SWEEP_POINTS = 400


def _conf(p):
    """Boxes without a YOLO conf (other kilder) survive every cutoff."""
    return p["conf"] if p["conf"] is not None else 1.0


def _manual_ids(truth_csv):
    manual = set()
    for r in iter_label_rows(truth_csv):
        status = (r.get("ml_status") or "").strip().upper()
        gen = (r.get("ml_generated") or "").strip().lower()
        if status == "MANUAL" or gen not in ("true", "t", "1"):
            manual.add((r.get("id") or "").strip())
    manual.discard("")
    return manual


def _measure(name, res_csv, truth, processed, manual, decade_of, codes_of, cost_w):
    ds = build_dataset(truth, read_predictions(res_csv),
                       include_unlabelled=True, processed_doc=processed)

    maxconf = [None] * len(ds.truth_boxes)
    for p in ds.pred:
        c = _conf(p)
        for j in p["covers"]:
            if maxconf[j] is None or c > maxconf[j]:
                maxconf[j] = c
    covered = sorted(-1.0 if c is None else c for c in maxconf)
    bom = sorted(_conf(p) for p in ds.pred if p["klasse"] == "BOM")

    def at(c):
        miss = bisect_left(covered, c)
        ov = len(bom) - bisect_left(bom, c)
        return miss, ov, cost_w * miss + ov

    points = sorted({round(v, 4) for v in covered + bom if v >= 0} | {STD_CONF})
    if len(points) > MAX_SWEEP_POINTS:
        step = len(points) / MAX_SWEEP_POINTS
        points = sorted({points[int(i * step)] for i in range(MAX_SWEEP_POINTS)}
                        | {STD_CONF})
    best_c = min(points, key=lambda c: (at(c)[2], at(c)[0], c))

    def slice_counts(c):
        miss_manual = 0
        miss_decade = defaultdict(int)
        miss_code = defaultdict(int)
        for j, fb in enumerate(ds.truth_boxes):
            mc = maxconf[j]
            if mc is not None and mc >= c:
                continue
            if fb["label_id"] in manual:
                miss_manual += 1
            miss_decade[decade_of.get(fb["doc_no"], "ukjent")] += 1
            for kode in codes_of.get(fb["doc_no"], ("ukjent",)):
                miss_code[kode] += 1
        return miss_manual, dict(miss_decade), dict(miss_code)

    truth_decade = defaultdict(int)
    truth_code = defaultdict(int)
    for fb in ds.truth_boxes:
        truth_decade[decade_of.get(fb["doc_no"], "ukjent")] += 1
        for kode in codes_of.get(fb["doc_no"], ("ukjent",)):
            truth_code[kode] += 1
    n_manual = sum(1 for fb in ds.truth_boxes if fb["label_id"] in manual)

    miss, ov, total = at(best_c)
    miss_manual, miss_decade, miss_code = slice_counts(best_c)
    miss_s, ov_s, total_s = at(STD_CONF)
    miss_manual_s, _, _ = slice_counts(STD_CONF)
    worst = max(miss_decade, key=lambda k: miss_decade[k], default="")
    worst_code = max((k for k in miss_code if k != "ukjent"),
                     key=lambda k: miss_code[k], default="")

    return {
        "navn": name, "res_csv": res_csv,
        "n_fasit": len(ds.truth_boxes), "n_manuell": n_manual,
        "n_dok_scope": len(ds.scope_doc),
        "beste_conf": best_c, "tapte": miss, "tapte_manuell": miss_manual,
        "oversladd": ov, "kostnad": total,
        "conf_std": STD_CONF, "tapte_std": miss_s,
        "tapte_manuell_std": miss_manual_s, "oversladd_std": ov_s,
        "kostnad_std": total_s,
        "verste_tiar": worst, "tapte_verste_tiar": miss_decade.get(worst, 0),
        "tapte_per_tiar": ";".join(
            f"{k}:{miss_decade[k]}/{truth_decade[k]}"
            for k in sorted(truth_decade, key=str) if k in miss_decade),
        "verste_kode": worst_code,
        "tapte_verste_kode": miss_code.get(worst_code, 0),
        "tapte_per_kode": ";".join(
            f"{k}:{miss_code[k]}/{truth_code[k]}"
            for k in sorted(miss_code, key=lambda k: -miss_code[k])[:8]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--truth-csv", required=True)
    p.add_argument("--processed-list", required=True,
                   help="the document list the validation ran over")
    p.add_argument("--metadata", nargs="+", default=[],
                   help="metadata CSVs for the decade slices")
    p.add_argument("--run", action="append", required=True, metavar="NAVN=RESULTAT.CSV")
    p.add_argument("--cost", type=float, default=20.0,
                   help="oversladd boxes one missed fnr is worth (default 20)")
    p.add_argument("--out", default=None, help="CSV to write the ranking to")
    args = p.parse_args()

    truth = read_truth_boxes(args.truth_csv)
    processed = read_processed_docs(args.processed_list)
    manual = _manual_ids(args.truth_csv)
    decade_of = {}
    codes_of = {}
    for path in args.metadata:
        m = pd.read_csv(path)
        years = pd.to_numeric(m["dokument_aar"], errors="coerce")
        rst = (m["rettsstiftelsestyper"] if "rettsstiftelsestyper" in m.columns
               else pd.Series([""] * len(m)))
        for doc, y, codes in zip(m["fil_revisjon_id"], years, rst.fillna("")):
            try:
                doc = int(doc)
            except (TypeError, ValueError):
                continue
            if not pd.isna(y):
                decade_of[doc] = int(y) // 10 * 10
            parsed = {part.strip().partition(" - ")[0].strip()
                      for part in str(codes).split(",") if part.strip()}
            if parsed:
                codes_of[doc] = parsed

    rows = []
    for spec in args.run:
        name, _, res_csv = spec.partition("=")
        if not res_csv:
            raise SystemExit(f"--run must be NAVN=RESULTAT.CSV, got: {spec}")
        rows.append(_measure(name, res_csv, truth, processed, manual,
                             decade_of, codes_of, args.cost))
    rows.sort(key=lambda r: r["kostnad"])

    hdr = f"{'navn':<24} {'conf':>5} {'tapte':>5} {'man':>4} {'oversl':>6} {'kostnad':>7}"
    print(f"\nRanking, cost = {args.cost:g} * tapte + oversladd, "
          f"{rows[0]['n_fasit']} fasit boxes ({rows[0]['n_manuell']} manual), "
          f"{rows[0]['n_dok_scope']} docs in scope")
    print(hdr + "   | ved std-conf " + f"{STD_CONF:g}: tapte/oversl/kostnad")
    for r in rows:
        print(f"{r['navn']:<24} {r['beste_conf']:>5.2f} {r['tapte']:>5} "
              f"{r['tapte_manuell']:>4} {r['oversladd']:>6} {r['kostnad']:>7.0f}"
              f"   | {r['tapte_std']}/{r['oversladd_std']}/{r['kostnad_std']:.0f}")
        print(f"{'':<24} tiår: {r['tapte_per_tiar']}")
        print(f"{'':<24} kode: {r['tapte_per_kode']}")
    print(f"\nWORST_DECADE {rows[0]['navn']} {rows[0]['verste_tiar']}")
    print(f"WORST_CODE {rows[0]['navn']} {rows[0]['verste_kode']}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
