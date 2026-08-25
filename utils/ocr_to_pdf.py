"""Run PaddleOCR on a PDF and write a new PDF with the text placed where OCR
found it, over a faded copy of the original, so the reading can be inspected.

Uses the same PaddleOCR models as the rest of the project.

Run:
    python utils/ocr_to_pdf.py dokument.pdf -o ut.pdf
    python utils/ocr_to_pdf.py dokument.pdf -o ut.pdf --opacity 0.3
    python utils/ocr_to_pdf.py document.pdf -o out.pdf --invisible
"""

import argparse
import os
import sys

import fitz  # PyMuPDF
import numpy as np
from paddleocr import PaddleOCR

_UTILS = os.path.dirname(os.path.abspath(__file__))
_APP = os.path.join(_UTILS, "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from config import PDF_DPI, PADDLE_MODEL_SET, DET_PAGE_LEN, REC_BATCH

# Model paths, the same as in paddle_ocr_model_fnr.py
_NAME = {
    "v5": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    "v6": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
}
DET_MODEL, REC_MODEL = _NAME[PADDLE_MODEL_SET]
_MODEL_DIR = os.path.abspath(_APP)
DET_MODEL_DIR = os.path.join(_MODEL_DIR, DET_MODEL + "_infer")
REC_MODEL_DIR = os.path.join(_MODEL_DIR, REC_MODEL + "_infer")


def _has_gpu():
    try:
        import paddle
        return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def _make_reader():
    """PaddleOCR reader with the same config as the rest of the project."""
    gpu = _has_gpu()
    print(f"GPU available: {gpu}")

    kwargs = dict(
        long="en",
        device="gpu" if gpu else "cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_type="max",
        text_det_limit_page_len=DET_PAGE_LEN,
        text_recognition_batch_size=REC_BATCH * 2 if gpu else REC_BATCH,
        text_detection_model_name=DET_MODEL,
        text_recognition_model_name=REC_MODEL,
        text_detection_model_dir=DET_MODEL_DIR,
        text_recognition_model_dir=REC_MODEL_DIR,
    )
    if gpu:
        kwargs["precision"] = "fp16"
    else:
        kwargs["enable_mkldnn"] = True

    return PaddleOCR(**kwargs)


def render_page(page, dpi=PDF_DPI):
    """Renders a fitz page to an RGB numpy array."""
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    return img.copy()


def _fetch_text_boxes(res):
    """Returns [(text, x0, y0, x1, y1), ...] in pixel coordinates."""
    boxes = []
    if not res:
        return boxes

    # Word level first, line level below as a fallback
    word_per_line = res.get("text_word")
    box_per_line = res.get("text_word_boxes")
    if word_per_line and box_per_line:
        for word_names, box_names in zip(word_per_line, box_per_line):
            for text, box in zip(word_names, box_names):
                if not text.strip():
                    continue
                pts = np.asarray(box, dtype=float).reshape(-1)
                x_coords = pts[0::2]
                y_coords = pts[1::2]
                boxes.append((text, float(x_coords.min()), float(y_coords.min()),
                               float(x_coords.max()), float(y_coords.max())))
        if boxes:
            return boxes

    texts = res.get("rec_texts") or []
    polys = res.get("rec_polys")
    if polys is None:
        polys = res.get("dt_polys") or []
    for text, poly in zip(texts, polys):
        if not text.strip():
            continue
        pts = np.asarray(poly, dtype=float)
        x0 = float(pts[:, 0].min())
        y0 = float(pts[:, 1].min())
        x1 = float(pts[:, 0].max())
        y1 = float(pts[:, 1].max())
        boxes.append((text, x0, y0, x1, y1))

    return boxes


def make_pdf_with_text(in_pdf_path, out_pdf_path, visible=True, background_opacity=0.15, dpi=PDF_DPI):
    """Writes a new PDF with the OCR text placed where OCR found it.

    With visible=False the text becomes an invisible, searchable layer.
    bakgrunn_opacity fades the original page underneath it.
    """
    print("Initialising PaddleOCR...")
    reader = _make_reader()

    in_doc = fitz.open(in_pdf_path)
    out_doc = fitz.open()

    for page_no in range(in_doc.page_count):
        page = in_doc[page_no]
        width_pt = page.rect.width
        height_pt = page.rect.height

        print(f"  Page {page_no + 1}/{in_doc.page_count} ({width_pt:.0f}×{height_pt:.0f} pt) ...", end=" ")

        image = render_page(page, dpi=dpi)
        image_h, image_w = image.shape[:2]

        # pixels -> PDF points
        scale_x = width_pt / image_w
        scale_y = height_pt / image_h

        # PaddleOCR wants BGR, and predict wants a *list* of images
        image_bgr = np.ascontiguousarray(image[:, :, ::-1])

        results = reader.predict([image_bgr], return_word_box=True) or []
        res = results[0] if results else None
        if res:
            print(f"    OCR result keys: {list(res.keys()) if hasattr(res, 'keys') else type(res)}")
        text_boxes = _fetch_text_boxes(res)
        print(f"{len(text_boxes)} text fragments found")

        new_page = out_doc.new_page(width=width_pt, height=height_pt)

        # The original goes in as a raster image, not a PDF XObject, so the OCR
        # text is guaranteed to land on top of it.
        if background_opacity < 1.0:
            faded = (image.astype(np.float32) * background_opacity
                     + 255.0 * (1.0 - background_opacity)).astype(np.uint8)
        else:
            faded = image
        import io
        from PIL import Image
        pil_img = Image.fromarray(faded)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        new_page.insert_image(new_page.rect, stream=buf.getvalue())

        for text, x0_px, y0_px, x1_px, y1_px in text_boxes:
            x0 = x0_px * scale_x
            y0 = y0_px * scale_y
            x1 = x1_px * scale_x
            y1 = y1_px * scale_y

            box_height_pt = y1 - y0
            box_width_pt = x1 - x0

            # Font size from both box height and text length. Helvetica averages
            # a character width of about 0.52 × fontsize.
            font_from_height = box_height_pt * 0.85
            if len(text) > 0:
                font_from_width = box_width_pt / (len(text) * 0.52)
            else:
                font_from_width = font_from_height
            font_str = max(4, min(font_from_height, font_from_width))

            # insert_text places text on the baseline, roughly 80 % down the box.
            point = fitz.Point(x0, y0 + box_height_pt * 0.8)
            render = 3 if not visible else 0  # 0=visible, 3=invisible

            new_page.insert_text(
                point,
                text,
                fontsize=font_str,
                fontname="helv",
                color=(0, 0, 0),
                render_mode=render,
            )

    out_doc.save(out_pdf_path)
    out_doc.close()
    in_doc.close()
    print(f"\nDone! Wrote {out_pdf_path}")


def main():
    p = argparse.ArgumentParser(
        description="Run PaddleOCR on a PDF and write a new PDF with the text in place."
    )
    p.add_argument("pdf", help="input PDF")
    p.add_argument("-o", "--output", default=None, help="output PDF (default: <input>_ocr.pdf)")
    p.add_argument("--visible", action="store_true", default=True,
                   help="make the OCR text visible (default: on)")
    p.add_argument("--invisible", action="store_true",
                   help="make the OCR text invisible (a searchable layer only)")
    p.add_argument("--dpi", type=int, default=PDF_DPI,
                   help=f"DPI for rendering the PDF pages (default: {PDF_DPI})")
    p.add_argument("--opacity", type=float, default=0.15,
                   help="opacity of the original page underneath (0.0-1.0, default: 0.15)")
    args = p.parse_args()

    if not args.output:
        args.output = args.pdf.rsplit(".", 1)[0] + "_ocr.pdf"

    visible = not args.invisible

    make_pdf_with_text(
        in_pdf_path=args.pdf,
        out_pdf_path=args.output,
        visible=visible,
        background_opacity=args.opacity if visible else 1.0,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()






