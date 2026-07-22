"""
Convert CSV with FNR bounding boxes (PDF points, top-left origin) to YOLO format.

Reads coordinates.csv, filters to ACCEPTED + ml_generated rows,
renders each PDF page at 300 DPI, and writes normalized YOLO labels.

Coordinate conventions (confirmed from codebase):
  - (x, y) is the TOP-LEFT corner of the bounding box
  - Y-axis origin is TOP-LEFT (image convention, no flip needed)
  - Values are in PDF points (1/72 inch)
  - Scale factor: DPI / 72
"""

import argparse
import numpy as np
import pandas as pd
import fitz
from pathlib import Path


DPI = 300
SCALE = DPI / 72.0


def convert(csv_path: str, pdf_dir: str, output_dir: str):
    csv_path = Path(csv_path)
    pdf_dir = Path(pdf_dir)
    output = Path(output_dir)
    images_dir = output / "images_all"
    labels_dir = output / "labels_all"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Total documents in CSV: {df['fil_revisjon_id'].nunique()}, total boxes: {len(df)}")

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
    done = 0
    skipped_pages = 0
    total_boxes = 0
    negatives = 0

    for fil_id, doc_group in df.groupby("fil_revisjon_id"):
        pdf_path = pdf_dir / f"{fil_id}.pdf"
        if not pdf_path.exists():
            missing.add(str(fil_id))
            continue

        doc = fitz.open(pdf_path)
        annotated_pages = set()

        for page_no, page_group in doc_group.groupby("sidetall"):
            idx = int(page_no) - 1
            if idx < 0 or idx >= len(doc):
                skipped_pages += 1
                continue

            annotated_pages.add(int(page_no))
            page = doc[idx]
            pix = page.get_pixmap(dpi=DPI)
            img_w, img_h = pix.width, pix.height
            stem = f"{fil_id}_p{page_no}"
            pix.save(str(images_dir / f"{stem}.png"))

            x_px = page_group["x"].values * SCALE
            y_px = page_group["y"].values * SCALE
            w_px = page_group["width"].values * SCALE
            h_px = page_group["height"].values * SCALE

            x_center = np.clip((x_px + w_px / 2) / img_w, 0.0, 1.0)
            y_center = np.clip((y_px + h_px / 2) / img_h, 0.0, 1.0)
            bw = np.clip(w_px / img_w, 0.0, 1.0)
            bh = np.clip(h_px / img_h, 0.0, 1.0)

            lines = [
                f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                for xc, yc, w, h in zip(x_center, y_center, bw, bh)
            ]

            (labels_dir / f"{stem}.txt").write_text("\n".join(lines))
            total_boxes += len(lines)
            done += 1

        # Render unannotated pages as negatives (empty label files)
        for idx in range(len(doc)):
            page_no = idx + 1
            if page_no in annotated_pages:
                continue
            page = doc[idx]
            pix = page.get_pixmap(dpi=DPI)
            stem = f"{fil_id}_p{page_no}"
            pix.save(str(images_dir / f"{stem}.png"))
            (labels_dir / f"{stem}.txt").write_text("")
            negatives += 1

        n_negatives_in_doc = len(doc) - len(annotated_pages)
        doc.close()
        print(f"  Converted {fil_id} ({len(doc_group)} boxes, {doc_group['sidetall'].nunique()} pages, {n_negatives_in_doc} negatives)")

    print(f"Wrote {done} page-images with {total_boxes} boxes total, {negatives} negative pages")
    if missing:
        print(f"Missing PDFs: {len(missing)}")
    if skipped_pages:
        print(f"Skipped {skipped_pages} out-of-range pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV annotations to YOLO format")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("pdfs", help="Directory containing PDF files")
    parser.add_argument("--output", default="dataset", help="Output directory (default: dataset)")
    args = parser.parse_args()
    convert(args.csv, args.pdfs, args.output)
