"""
Convert CSV with FNR bounding boxes (PDF points, top-left origin) to YOLO format.

Reads coordinates.csv, filters to ACCEPTED + ml_generated rows,
renders each PDF page at 300 DPI, and writes normalized YOLO labels.

Rendering is the whole cost here — pure CPU and disk, no GPU — so documents
are converted in parallel processes. Each document is independent: its own
PDF handle, its own output files.

Coordinate conventions (confirmed from codebase):
  - (x, y) is the TOP-LEFT corner of the bounding box
  - Y-axis origin is TOP-LEFT (image convention, no flip needed)
  - Values are in PDF points (1/72 inch)
  - Scale factor: DPI / 72
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz
import numpy as np
import pandas as pd


DPI = 300
SCALE = DPI / 72.0


def _default_workers():
    """One process per physical core, roughly, capped to keep RAM sane."""
    return min(max((os.cpu_count() or 4) // 2, 1), 32)


def _render_document(job):
    """Render one PDF to page images + YOLO labels. Runs in a worker process.

    `job["pages"]` maps page number (1-based) to a list of (x, y, w, h) boxes
    in PDF points. Pages not listed are rendered as negatives (empty labels).
    Returns counters for the parent to accumulate.
    """
    fil_id = job["fil_id"]
    images_dir = Path(job["images_dir"])
    labels_dir = Path(job["labels_dir"])
    out = {"fil_id": fil_id, "pages": 0, "boxes": 0, "negatives": 0,
           "skipped_pages": 0, "error": None}

    try:
        doc = fitz.open(job["pdf_path"])
    except Exception as e:
        out["error"] = repr(e)
        return out

    try:
        annotated = set()
        for page_no, bokser in sorted(job["pages"].items()):
            idx = int(page_no) - 1
            if idx < 0 or idx >= len(doc):
                out["skipped_pages"] += 1
                continue

            annotated.add(int(page_no))
            pix = doc[idx].get_pixmap(dpi=DPI)
            img_w, img_h = pix.width, pix.height
            stem = f"{fil_id}_p{page_no}"
            pix.save(str(images_dir / f"{stem}.png"))

            arr = np.asarray(bokser, dtype=float) * SCALE
            x_px, y_px, w_px, h_px = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            x_center = np.clip((x_px + w_px / 2) / img_w, 0.0, 1.0)
            y_center = np.clip((y_px + h_px / 2) / img_h, 0.0, 1.0)
            bw = np.clip(w_px / img_w, 0.0, 1.0)
            bh = np.clip(h_px / img_h, 0.0, 1.0)

            lines = [f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                     for xc, yc, w, h in zip(x_center, y_center, bw, bh)]
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines))
            out["boxes"] += len(lines)
            out["pages"] += 1

        # Unannotated pages become negatives (empty label files)
        for idx in range(len(doc)):
            page_no = idx + 1
            if page_no in annotated:
                continue
            pix = doc[idx].get_pixmap(dpi=DPI)
            stem = f"{fil_id}_p{page_no}"
            pix.save(str(images_dir / f"{stem}.png"))
            (labels_dir / f"{stem}.txt").write_text("")
            out["negatives"] += 1
    except Exception as e:
        out["error"] = repr(e)
    finally:
        doc.close()
    return out


def convert(csv_path: str, pdf_dir: str, output_dir: str, only_ids: set = None,
            workers: int = 0):
    csv_path = Path(csv_path)
    pdf_dir = Path(pdf_dir)
    output = Path(output_dir)
    images_dir = output / "images_all"
    labels_dir = output / "labels_all"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Total documents in CSV: {df['fil_revisjon_id'].nunique()}, total boxes: {len(df)}")

    if only_ids:
        df = df[df["fil_revisjon_id"].astype(str).isin(only_ids)]
        print(f"Filtered to {df['fil_revisjon_id'].nunique()} documents matching --ids")

    df["ml_status"] = df["ml_status"].astype(str).str.strip().str.upper()
    df["ml_generated"] = (
        df["ml_generated"].astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES"])
    )

    # Beholder kun dokumenter med: ml_generated=TRUE + ACCEPTED, OR ml_generated=FALSE (human-placed boxes)
    ml_accepted = (df["ml_generated"]) & (df["ml_status"] == "ACCEPTED")
    manual = ~df["ml_generated"]
    rejected = (df["ml_generated"]) & (df["ml_status"] == "REJECTED")
    df = df[ml_accepted | manual]
    print(f"Filtered to {len(df)} boxes ({ml_accepted.sum()} ML accepted, {manual.sum()} manual, {rejected.sum()} ML rejected excluded)")
    print(f"Unique documents after filtering: {df['fil_revisjon_id'].nunique()}, unique pages: {df[['fil_revisjon_id', 'sidetall']].drop_duplicates().shape[0]}")

    missing = set()
    skipped_docs = 0
    jobs = []

    def _legg_til(fil_id, pages):
        """Queue one document unless the PDF is gone or it is already done."""
        nonlocal skipped_docs
        pdf_path = pdf_dir / f"{fil_id}.pdf"
        if not pdf_path.exists():
            missing.add(str(fil_id))
            return
        if (images_dir / f"{fil_id}_p1.png").exists():   # cache
            skipped_docs += 1
            return
        jobs.append({"fil_id": fil_id, "pdf_path": str(pdf_path), "pages": pages,
                     "images_dir": str(images_dir), "labels_dir": str(labels_dir)})

    for fil_id, doc_group in df.groupby("fil_revisjon_id"):
        pages = {
            int(page_no): page_group[["x", "y", "width", "height"]].values.tolist()
            for page_no, page_group in doc_group.groupby("sidetall")
        }
        _legg_til(fil_id, pages)

    # Documents in the ID list but not in the CSV are pure negatives
    negative_docs = 0
    if only_ids:
        labeled_ids = set(df["fil_revisjon_id"].astype(str).unique())
        unlabeled_ids = only_ids - labeled_ids
        if unlabeled_ids:
            print(f"Rendering {len(unlabeled_ids)} unlabeled documents as negatives...")
        for fil_id in sorted(unlabeled_ids):
            før = len(jobs)
            _legg_til(fil_id, {})
            negative_docs += len(jobs) - før

    done = total_boxes = negatives = skipped_pages = 0
    failed = []
    n_workers = min(workers or _default_workers(), max(len(jobs), 1))
    print(f"Converting {len(jobs)} documents with {n_workers} processes...")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_render_document, j): j["fil_id"] for j in jobs}
        for n, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            if r["error"]:
                failed.append((r["fil_id"], r["error"]))
                print(f"  [{n}/{len(jobs)}] FAILED {r['fil_id']}: {r['error']}")
                continue
            done += r["pages"]
            total_boxes += r["boxes"]
            negatives += r["negatives"]
            skipped_pages += r["skipped_pages"]
            print(f"  [{n}/{len(jobs)}] Converted {r['fil_id']} "
                  f"({r['boxes']} boxes, {r['pages']} pages, {r['negatives']} negatives)")

    print(f"Wrote {done} page-images with {total_boxes} boxes total, {negatives} negative pages ({negative_docs} fully negative docs)")
    if skipped_docs:
        print(f"Skipped {skipped_docs} already converted documents")
    if missing:
        print(f"Missing PDFs: {len(missing)}")
    if skipped_pages:
        print(f"Skipped {skipped_pages} out-of-range pages")
    if failed:
        print(f"Failed: {len(failed)} documents")
        for fil_id, err in failed[:10]:
            print(f"  {fil_id}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV annotations to YOLO format")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("pdfs", help="Directory containing PDF files")
    parser.add_argument("--output", default="dataset", help="Output directory (default: dataset)")
    parser.add_argument("--ids", default=None, help="File with document IDs to convert (one per line). Converts all if not set.")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel rendering processes (0 = half the cores, max 32)")
    args = parser.parse_args()

    only_ids = None
    if args.ids:
        with open(args.ids) as f:
            only_ids = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(only_ids)} IDs from {args.ids}")

    convert(args.csv, args.pdfs, args.output, only_ids=only_ids, workers=args.workers)
