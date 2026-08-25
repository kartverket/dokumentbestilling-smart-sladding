"""Convert a labels CSV into rendered pages plus YOLO label files.

Rendering is the whole cost, pure CPU and disk, so documents run in parallel
processes: each one is independent, with its own PDF handle and output files.

CSV coordinates are PDF points with (x, y) at the TOP-LEFT corner and the
y-axis growing downwards, the same convention as the image, so no flip.

Run:
    python train/scripts/convert_csv_to_yolo.py labels.csv pdfs/ --output dataset
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz
import numpy as np
import pandas as pd

# ugyldige_labels.txt lives at the repo root; reuse the reader in utils/.
_UTILS = str(Path(__file__).resolve().parents[2] / "utils")
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)
from filter_common import read_invalid_label_ids


DPI = 300
SCALE = DPI / 72.0


def _default_workers():
    """One process per physical core, roughly, capped to keep RAM sane."""
    return min(max((os.cpu_count() or 4) // 2, 1), 32)


def _render_document(job):
    """Render one PDF to page images + YOLO labels, in a worker process.

    `job["pages"]` maps a 1-based page number to (x, y, w, h) boxes in PDF
    points. Pages not listed are rendered as negatives (empty labels).
    """
    file_id = job["fil_id"]
    images_dir = Path(job["images_dir"])
    labels_dir = Path(job["labels_dir"])
    out = {"fil_id": file_id, "pages": 0, "boxes": 0, "negatives": 0,
           "skipped_pages": 0, "error": None}

    try:
        doc = fitz.open(job["pdf_path"])
    except Exception as e:
        out["error"] = repr(e)
        return out

    try:
        annotated = set()
        for page_no, boxes in sorted(job["pages"].items()):
            idx = int(page_no) - 1
            if idx < 0 or idx >= len(doc):
                out["skipped_pages"] += 1
                continue

            annotated.add(int(page_no))
            pix = doc[idx].get_pixmap(dpi=DPI)
            img_w, img_h = pix.width, pix.height
            stem = f"{file_id}_p{page_no}"
            pix.save(str(images_dir / f"{stem}.png"))

            arr = np.asarray(boxes, dtype=float) * SCALE
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
            stem = f"{file_id}_p{page_no}"
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

    invalid = read_invalid_label_ids()
    if invalid and "id" in df.columns:
        listed = df["id"].astype(str).str.strip().isin(invalid)
        if listed.any():
            print(f"Excluded {int(listed.sum())} boxes listed in ugyldige_labels.txt")
            df = df[~listed]
    elif invalid:
        print(f"WARNING: ugyldige_labels.txt has {len(invalid)} ids, but the "
              f"labels CSV has no id column - nothing excluded.")

    df["ml_status"] = df["ml_status"].astype(str).str.strip().str.upper()
    df["ml_generated"] = (
        df["ml_generated"].astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES"])
    )

    # Keep ml_generated=TRUE + ACCEPTED, plus every hand-placed box.
    ml_accepted = (df["ml_generated"]) & (df["ml_status"] == "ACCEPTED")
    manual = ~df["ml_generated"]
    rejected = (df["ml_generated"]) & (df["ml_status"] == "REJECTED")
    df = df[ml_accepted | manual]
    print(f"Filtered to {len(df)} boxes ({ml_accepted.sum()} ML accepted, {manual.sum()} manual, {rejected.sum()} ML rejected excluded)")
    print(f"Unique documents after filtering: {df['fil_revisjon_id'].nunique()}, unique pages: {df[['fil_revisjon_id', 'sidetall']].drop_duplicates().shape[0]}")

    missing = set()
    skipped_docs = 0
    jobs = []

    def _add_to(file_id, pages):
        """Queue one document unless the PDF is gone or it is already done."""
        nonlocal skipped_docs
        pdf_path = pdf_dir / f"{file_id}.pdf"
        if not pdf_path.exists():
            missing.add(str(file_id))
            return
        if (images_dir / f"{file_id}_p1.png").exists():   # cache
            skipped_docs += 1
            return
        jobs.append({"fil_id": file_id, "pdf_path": str(pdf_path), "pages": pages,
                     "images_dir": str(images_dir), "labels_dir": str(labels_dir)})

    for file_id, doc_group in df.groupby("fil_revisjon_id"):
        pages = {
            int(page_no): page_group[["x", "y", "width", "height"]].values.tolist()
            for page_no, page_group in doc_group.groupby("sidetall")
        }
        _add_to(file_id, pages)

    # Documents in the ID list but not in the CSV are pure negatives
    negative_docs = 0
    if only_ids:
        labeled_ids = set(df["fil_revisjon_id"].astype(str).unique())
        unlabeled_ids = only_ids - labeled_ids
        if unlabeled_ids:
            print(f"Rendering {len(unlabeled_ids)} unlabeled documents as negatives...")
        for file_id in sorted(unlabeled_ids):
            before = len(jobs)
            _add_to(file_id, {})
            negative_docs += len(jobs) - before

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
        for file_id, err in failed[:10]:
            print(f"  {file_id}: {err}")


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
