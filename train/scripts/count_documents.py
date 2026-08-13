"""
Count documents matching a given strategy configuration.

Run before training to verify how many documents and labels
are available for a given doc type / year range filter.

No external dependencies — uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def _has_doc_type(rettsstiftelsestyper: str, doc_type: str) -> bool:
    """Check if any of the comma-separated rettsstiftelsestyper matches doc_type."""
    return any(
        part.strip().split(" ", 1)[0] == doc_type
        for part in str(rettsstiftelsestyper).split(",")
    )


def _safe_int(value: str) -> int | None:
    """Try to parse an int, return None on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count(metadata_csv: str, labels_csv: str | None, strategy: str,
          doc_type: str | None, year_from: int | None, year_to: int | None,
          list_files: bool = False, output_ids: bool = False):
    rows = _read_csv(metadata_csv)

    # ── Filter by doc type ──────────────────────────────────────
    if strategy in ("doc_type", "year_and_doc_type") and doc_type:
        rows = [r for r in rows if _has_doc_type(r.get("rettsstiftelsestyper", ""), doc_type)]

    # ── Filter by year range ────────────────────────────────────
    if strategy == "year_and_doc_type" and year_from is not None and year_to is not None:
        rows = [r for r in rows
                if (y := _safe_int(r.get("dokument_aar"))) is not None
                and year_from <= y <= year_to]
    elif strategy == "yearly":
        rows = [r for r in rows if _safe_int(r.get("dokument_aar")) is not None]

    matched_ids = {str(r["fil_revisjon_id"]) for r in rows}

    # ── If --output-ids: just print IDs and exit ────────────────
    if output_ids:
        for doc_id in sorted(matched_ids):
            print(doc_id)
        return

    # ── Normal summary output ───────────────────────────────────
    print(f"Metadata: {len(_read_csv(metadata_csv))} documents total")
    if strategy in ("doc_type", "year_and_doc_type") and doc_type:
        print(f"  Filter doc_type={doc_type}: {len(rows)} documents match")
    if strategy == "year_and_doc_type" and year_from is not None and year_to is not None:
        print(f"  Filter years {year_from}\u2013{year_to}: {len(rows)} documents match")
    elif strategy == "yearly":
        print(f"  Yearly: {len(rows)} documents with valid year")

    print(f"\n{'='*50}")
    print(f"Matching documents: {len(matched_ids)}")

    years = [_safe_int(r.get("dokument_aar")) for r in rows]
    years = [y for y in years if y is not None]
    if years:
        year_counts = Counter(years)
        print(f"Year range: {min(years)}\u2013{max(years)}")
        print(f"\nPer year:")

        if list_files:
            docs_by_year: dict[int, list[str]] = defaultdict(list)
            for r in rows:
                y = _safe_int(r.get("dokument_aar"))
                if y is not None:
                    docs_by_year[y].append(str(r["fil_revisjon_id"]))
            for year in sorted(year_counts):
                files = ", ".join(f"{d}.pdf" for d in sorted(docs_by_year[year]))
                print(f"  {year}: {year_counts[year]} documents — {files}")
        else:
            for year in sorted(year_counts):
                print(f"  {year}: {year_counts[year]} documents")

    # ── Cross-reference with labels if provided ─────────────────
    if labels_csv and Path(labels_csv).exists():
        label_rows = _read_csv(labels_csv)
        matched_labels = [r for r in label_rows if str(r["fil_revisjon_id"]) in matched_ids]
        docs_with_labels = {r["fil_revisjon_id"] for r in matched_labels}
        total_boxes = len(matched_labels)
        total_pages = len({(r["fil_revisjon_id"], r["sidetall"]) for r in matched_labels})

        print(f"\nLabels ({Path(labels_csv).name}):")
        print(f"  Documents with annotations: {len(docs_with_labels)} of {len(matched_ids)}")
        print(f"  Annotated pages: {total_pages}")
        print(f"  Total bounding boxes: {total_boxes}")
        docs_without = matched_ids - {str(d) for d in docs_with_labels}
        if docs_without:
            print(f"  Documents without annotations: {len(docs_without)}")
            if list_files:
                files = ", ".join(f"{d}.pdf" for d in sorted(docs_without))
                print(f"    {files}")

    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count documents matching a training configuration")
    parser.add_argument("--metadata", required=True, help="Metadata CSV (with fil_revisjon_id, dokument_aar, rettsstiftelsestyper)")
    parser.add_argument("--labels", default="", help="Labels CSV (optional, for cross-referencing annotations)")
    parser.add_argument("--strategy", choices=["random", "yearly", "doc_type", "year_and_doc_type"], default="random")
    parser.add_argument("--doc-type", default="")
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument("--list-files", action="store_true", help="Print PDF filenames per year")
    parser.add_argument("--output-ids", action="store_true", help="Print only matching fil_revisjon_ids (for piping)")
    args = parser.parse_args()

    if args.strategy in ("doc_type", "year_and_doc_type") and not args.doc_type:
        print(f"ERROR: --doc-type is required for {args.strategy} strategy")
        raise SystemExit(1)

    if args.strategy == "year_and_doc_type" and (args.year_from is None or args.year_to is None):
        print("ERROR: --year-from and --year-to are required for year_and_doc_type strategy")
        raise SystemExit(1)

    count(args.metadata, args.labels or None, args.strategy,
          args.doc_type or None, args.year_from, args.year_to,
          list_files=args.list_files, output_ids=args.output_ids)
          list_files=args.list_files)
