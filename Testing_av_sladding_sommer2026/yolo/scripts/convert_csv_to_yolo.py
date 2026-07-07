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
import pandas as pd
import fitz
from pathlib import Path


DPI = 300
SCALE = DPI / 72.0


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def convert(csv_path: str, pdf_dir: str, output_dir: str):
    csv_path = Path(csv_path)
    pdf_dir = Path(pdf_dir)
    output = Path(output_dir)
    images_dir = output / "images_all"
    labels_dir = output / "labels_all"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    df["ml_status"] = df["ml_status"].astype(str).str.strip().str.upper()
    df["ml_generated"] = (
        df["ml_generated"].astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES"])
    )

    # Keep: ml_generated=TRUE + ACCEPTED, OR ml_generated=FALSE (human-placed boxes)
    ml_accepted = (df["ml_generated"]) & (df["ml_status"] == "ACCEPTED")
    manual = ~df["ml_generated"]
    rejected = (df["ml_generated"]) & (df["ml_status"] == "REJECTED")
    df = df[ml_accepted | manual]
    print(f"Filtered to {len(df)} boxes ({ml_accepted.sum()} ML accepted, {manual.sum()} manual, {rejected.sum()} ML rejected excluded)")

    missing = set()
    done = 0
    skipped_pages = 0
    total_boxes = 0

    for (fil_id, page_no), group in df.groupby(["fil_revisjon_id", "sidetall"]):
        pdf_path = pdf_dir / f"{fil_id}.pdf"
        if not pdf_path.exists():
            missing.add(str(fil_id))
            continue

        doc = fitz.open(pdf_path)
        idx = int(page_no) - 1
        if idx < 0 or idx >= len(doc):
            doc.close()
            skipped_pages += 1
            continue

        page = doc[idx]
        pix = page.get_pixmap(dpi=DPI)
        img_w, img_h = pix.width, pix.height
        stem = f"{fil_id}_p{page_no}"
        pix.save(str(images_dir / f"{stem}.png"))

        lines = []
        for _, r in group.iterrows():
            x_px = r["x"] * SCALE
            y_px = r["y"] * SCALE
            w_px = r["width"] * SCALE
            h_px = r["height"] * SCALE

            x_center = clamp((x_px + w_px / 2) / img_w)
            y_center = clamp((y_px + h_px / 2) / img_h)
            bw = clamp(w_px / img_w)
            bh = clamp(h_px / img_h)
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        (labels_dir / f"{stem}.txt").write_text("\n".join(lines))
        total_boxes += len(lines)
        done += 1
        doc.close()

    print(f"Wrote {done} page-images with {total_boxes} boxes total")
    if missing:
        print(f"Missing PDFs: {len(missing)}")
    if skipped_pages:
        print(f"Skipped {skipped_pages} out-of-range pages")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV annotations to YOLO format")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("pdfs", help="Directory containing PDF files")
    args = parser.parse_args()
    convert(args.csv, args.pdfs, "dataset")
