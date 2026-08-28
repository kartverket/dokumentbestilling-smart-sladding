"""Builds train and holdout document lists for one uttrekk, before training.

Labeled documents (fasit via iter_label_rows, so manglende_labels.csv counts)
are split per decade at document level. Zero-fnr documents (no fasit rows;
human review covers every document in an uttrekk, so no rows means no fnr)
are sampled per decade into both sides. Documents in
--force-train, typically the overlap with another training uttrekk, never land
in holdout: labeled ones go to the train list, unlabeled ones are dropped
because the other uttrekk already carries them. The split is seeded, so the
holdout is reproducible and can be written once and left alone.

Kjør:
    python train/scripts/make_holdout_lists.py \
        --labels "$SLADD_LABELS/uttrekk_6.csv" \
        --metadata "$SLADD_METADATA/uttrekk_6.csv" \
        --force-train overlapp.txt \
        --out-train "$SLADD_LISTS/uttrekk_6_tren48.txt" \
        --out-holdout "$SLADD_LISTS/uttrekk_6_holdout48.txt"
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_UTILS = str(Path(__file__).resolve().parents[2] / "utils")
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)
from filter_common import iter_label_rows


def _norm_id(value):
    try:
        return str(int(float(str(value).strip().removesuffix(".pdf"))))
    except (TypeError, ValueError):
        return None


def _read_id_file(path):
    ids = set()
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            nid = _norm_id(line)
            if nid:
                ids.add(nid)
    return ids


def _is_manual(row):
    if (row.get("ml_status") or "").strip().upper() == "MANUAL":
        return True
    return (row.get("ml_generated") or "").strip().lower() not in ("true", "t", "1")


def _decade_table(title, docs, decade_of):
    per = defaultdict(int)
    for d in docs:
        per[decade_of.get(d, "ukjent")] += 1
    cells = "  ".join(f"{k}:{per[k]}" for k in sorted(per, key=str))
    print(f"  {title}: {len(docs)} dokumenter  [{cells}]")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--labels", required=True, help="labels CSV for the uttrekk")
    p.add_argument("--metadata", required=True, help="metadata CSV (dokument_aar)")
    p.add_argument("--force-train", default=None,
                   help="file of ids that must never enter the holdout")
    p.add_argument("--holdout-share", type=float, default=0.4,
                   help="share of labeled documents held out (default 0.4)")
    p.add_argument("--neg-holdout", type=int, default=2000,
                   help="verified-negative documents in the holdout (default 2000)")
    p.add_argument("--neg-train", type=int, default=1000,
                   help="verified-negative documents in the train list (default 1000)")
    p.add_argument("--seed", type=int, default=48)
    p.add_argument("--out-train", required=True)
    p.add_argument("--out-holdout", required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    forced = _read_id_file(args.force_train) if args.force_train else set()

    labeled = set()
    manual_docs = set()
    for r in iter_label_rows(args.labels):
        doc = _norm_id(r.get("fil_revisjon_id"))
        if not doc:
            continue
        labeled.add(doc)
        if _is_manual(r):
            manual_docs.add(doc)

    meta = pd.read_csv(args.metadata)
    meta_ids = meta["fil_revisjon_id"].map(_norm_id)
    alle_dok = set(meta_ids.dropna())
    years = pd.to_numeric(meta["dokument_aar"], errors="coerce")
    decade_of = {d: int(y) // 10 * 10 for d, y in zip(meta_ids, years)
                 if d and not pd.isna(y)}

    if labeled - alle_dok:
        print(f"NB: {len(labeled - alle_dok)} labeled docs missing from "
              f"metadata (kept, decade 'ukjent')")

    def stratified_split(docs, take):
        """Seeded per-decade shuffle; returns (taken, rest)."""
        per = defaultdict(list)
        for d in sorted(docs):
            per[decade_of.get(d, "ukjent")].append(d)
        taken, rest = [], []
        total = len(docs)
        for key in sorted(per, key=str):
            group = per[key]
            rng.shuffle(group)
            n = round(len(group) * take / total) if total else 0
            taken.extend(group[:n])
            rest.extend(group[n:])
        return set(taken), set(rest)

    splittable = labeled - forced
    n_hold = round(len(splittable) * args.holdout_share)
    hold_labeled, train_labeled = stratified_split(splittable, n_hold)
    train_labeled |= labeled & forced

    neg_pool = alle_dok - labeled - forced
    n_hold_neg = min(args.neg_holdout, len(neg_pool))
    hold_neg, neg_rest = stratified_split(neg_pool, n_hold_neg)
    n_train_neg = min(args.neg_train, len(neg_rest))
    train_neg, _ = stratified_split(neg_rest, n_train_neg)

    print(f"Fasit: {len(labeled)} labeled docs ({len(manual_docs)} with manual boxes) "
          f"of {len(alle_dok)} in the uttrekk, {len(neg_pool)} zero-fnr docs, "
          f"{len(forced)} forced to train")
    print("Holdout:")
    _decade_table("labeled", hold_labeled, decade_of)
    _decade_table("manual-docs", hold_labeled & manual_docs, decade_of)
    _decade_table("negative", hold_neg, decade_of)
    print("Train:")
    _decade_table("labeled", train_labeled, decade_of)
    _decade_table("negative", train_neg, decade_of)

    Path(args.out_holdout).write_text(
        "\n".join(sorted(hold_labeled | hold_neg, key=int)) + "\n")
    Path(args.out_train).write_text(
        "\n".join(sorted(train_labeled | train_neg, key=int)) + "\n")
    print(f"Wrote {args.out_holdout} ({len(hold_labeled | hold_neg)} ids) and "
          f"{args.out_train} ({len(train_labeled | train_neg)} ids)")

    overlap = (hold_labeled | hold_neg) & (train_labeled | train_neg)
    if overlap:
        raise SystemExit(f"BUG: {len(overlap)} ids in both lists")


if __name__ == "__main__":
    main()
