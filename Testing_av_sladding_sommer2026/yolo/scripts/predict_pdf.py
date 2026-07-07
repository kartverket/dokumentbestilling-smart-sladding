"""
Run FNR detection on a PDF using the trained YOLO model.
Renders each page, runs prediction, saves annotated images.
Only saves results for PDFs where detections were found.
Each run gets a timestamped output folder.
"""

import argparse
from datetime import datetime
import fitz
from ultralytics import YOLO
from pathlib import Path


DPI = 300
SCALE = DPI / 72.0


def predict_pdf(model, pdf_path: str, output_dir: Path, conf: float):
    doc = fitz.open(pdf_path)
    pdf_name = Path(pdf_path).stem

    found_any = False
    page_results = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=DPI)
        img_path = output_dir / f"{pdf_name}_p{page_idx + 1}.png"
        pix.save(str(img_path))

        results = model.predict(str(img_path), conf=conf, imgsz=1280, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            found_any = True
            page_results.append((page_idx, results[0], boxes))

        img_path.unlink()

    if found_any:
        print(f"  {pdf_name}: found FNRs on {len(page_results)} page(s)")
        for page_idx, result, boxes in page_results:
            print(f"    Page {page_idx + 1}: {len(boxes)} FNRs")
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                x_pt = x1 / SCALE
                y_pt = y1 / SCALE
                w_pt = (x2 - x1) / SCALE
                h_pt = (y2 - y1) / SCALE
                print(f"      conf={confidence:.2f}  x={x_pt:.1f} y={y_pt:.1f} w={w_pt:.1f} h={h_pt:.1f}")
            result.save(str(output_dir / f"{pdf_name}_p{page_idx + 1}_pred.jpg"))

    doc.close()
    return found_any


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FNR detection on a PDF or folder of PDFs")
    parser.add_argument("source", help="Path to PDF file or folder of PDFs")
    parser.add_argument("--model", default="runs/detect/train/weights/best.pt", help="Model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", default="predictions", help="Base output directory")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    source = Path(args.source)
    if source.is_dir():
        pdfs = sorted(source.glob("*.pdf"))
        print(f"Found {len(pdfs)} PDFs in {source}")
        hits = sum(1 for pdf in pdfs if predict_pdf(model, str(pdf), output_dir, args.conf))
        print(f"\nDone. {hits}/{len(pdfs)} PDFs had detections -> {output_dir}/")
    else:
        predict_pdf(model, args.source, output_dir, args.conf)
        print(f"\nResults -> {output_dir}/")
