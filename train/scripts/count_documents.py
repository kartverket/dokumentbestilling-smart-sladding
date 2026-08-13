"""
Count documents matching a given strategy configuration.

Run before training to verify how many documents and labels
are available for a given doc type / year range filter.
"""

import argparse
from pathlib import Path

import pandas as pd


def _has_doc_type(rettsstiftelsestyper: str, doc_type: str) -> bool:
    """Check if any of the comma-separated rettsstiftelsestyper matches doc_type."""
    return any(
        part.strip().split(" ", 1)[0] == doc_type
        for part in str(rettsstiftelsestyper).split(",")
    )


def count(metadata_csv: str, labels_csv: str | None, strategy: str,
          doc_type: str | None, year_from: int | None, year_to: int | None):
    meta = pd.read_csv(metadata_csv)
    print(f"Metadata: {len(meta)} documents total")

    # ── Filter by doc type ──────────────────────────────────────
    if strategy in ("doc_type", "year_and_doc_type") and doc_type:
        mask = meta["rettsstiftelsestyper"].apply(lambda s: _has_doc_type(s, doc_type))
        meta = meta[mask]
        print(f"  Filter doc_type={doc_type}: {len(meta)} documents match")

    # ── Filter by year range ────────────────────────────────────
    if strategy == "year_and_doc_type" and year_from is not None and year_to is not None:
        years = pd.to_numeric(meta["dokument_aar"], errors="coerce")
        meta = meta[(years >= year_from) & (years <= year_to)]
        print(f"  Filter years {year_from}–{year_to}: {len(meta)} documents match")
    elif strategy == "yearly":
        years = pd.to_numeric(meta["dokument_aar"], errors="coerce")
        meta = meta[years.notna()]
        print(f"  Yearly: {len(meta)} documents with valid year")

    # ── Summary ─────────────────────────────────────────────────
    matched_ids = set(meta["fil_revisjon_id"].astype(str))

    print(f"\n{'='*50}")
    print(f"Matching documents: {len(matched_ids)}")

    if "dokument_aar" in meta.columns:
        years = pd.to_numeric(meta["dokument_aar"], errors="coerce").dropna().astype(int)
        if not years.empty:
            year_counts = years.value_counts().sort_index()
            print(f"Year range: {years.min()}–{years.max()}")
            print(f"\nPer year:")
            for year, n in year_counts.items():
                print(f"  {year}: {n} documents")

    # ── Cross-reference with labels if provided ─────────────────
    if labels_csv and Path(labels_csv).exists():
        labels = pd.read_csv(labels_csv)
        labels["fil_revisjon_id"] = labels["fil_revisjon_id"].astype(str)
        matched_labels = labels[labels["fil_revisjon_id"].isin(matched_ids)]
        docs_with_labels = matched_labels["fil_revisjon_id"].nunique()
        total_boxes = len(matched_labels)
        total_pages = matched_labels[["fil_revisjon_id", "sidetall"]].drop_duplicates().shape[0]

        print(f"\nLabels ({Path(labels_csv).name}):")
        print(f"  Documents with annotations: {docs_with_labels} of {len(matched_ids)}")
        print(f"  Annotated pages: {total_pages}")
        print(f"  Total bounding boxes: {total_boxes}")
        docs_without = matched_ids - set(matched_labels["fil_revisjon_id"])
        if docs_without:
            print(f"  Documents without annotations: {len(docs_without)}")

    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count documents matching a training configuration")
    parser.add_argument("--metadata", required=True, help="Metadata CSV (with fil_revisjon_id, dokument_aar, rettsstiftelsestyper)")
    parser.add_argument("--labels", default="", help="Labels CSV (optional, for cross-referencing annotations)")
    parser.add_argument("--strategy", choices=["random", "yearly", "doc_type", "year_and_doc_type"], default="random")
    parser.add_argument("--doc-type", default="")
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    args = parser.parse_args()

    if args.strategy in ("doc_type", "year_and_doc_type") and not args.doc_type:
        print(f"ERROR: --doc-type is required for {args.strategy} strategy")
        raise SystemExit(1)

    if args.strategy == "year_and_doc_type" and (args.year_from is None or args.year_to is None):
        print("ERROR: --year-from and --year-to are required for year_and_doc_type strategy")
        raise SystemExit(1)

    count(args.metadata, args.labels or None, args.strategy,
          args.doc_type or None, args.year_from, args.year_to)

