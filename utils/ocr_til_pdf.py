"""
Kjør PaddleOCR på et PDF-dokument og lag en ny PDF
med teksten plassert i riktige posisjoner.

Standard: synlig tekst over originaldokumentet med 50% opacity,
slik at man tydelig ser OCR-teksten oppå dokumentet.

Bruker samme PaddleOCR-modeller som resten av prosjektet
(PP-OCRv6_medium_det + PP-OCRv6_medium_rec).

Bruk:
    python utils/ocr_til_pdf.py dokument.pdf -o ut.pdf
    python utils/ocr_til_pdf.py dokument.pdf -o ut.pdf --opacity 0.3
    python utils/ocr_til_pdf.py dokument.pdf -o ut.pdf --usynlig
"""

import argparse
import os
import sys

import fitz  # PyMuPDF
import numpy as np
from paddleocr import PaddleOCR

# Legg app-mappen i path for å hente config
_UTILS = os.path.dirname(os.path.abspath(__file__))
_APP = os.path.join(_UTILS, "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from config import PDF_DPI, MODELL_SETT, DET_SIDE_LEN, REC_BATCH

# Modellstier (samme som paddle_ocr_model_fnr.py)
_NAVN = {
    "v5": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    "v6": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
}
DET_MODELL, REC_MODELL = _NAVN[MODELL_SETT]
_MODELL_MAPPE = os.path.abspath(_APP)
DET_MODELL_DIR = os.path.join(_MODELL_MAPPE, DET_MODELL + "_infer")
REC_MODELL_DIR = os.path.join(_MODELL_MAPPE, REC_MODELL + "_infer")


def _har_gpu():
    try:
        import paddle
        return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def _lag_reader():
    """Opprett PaddleOCR-reader med samme konfig som resten av prosjektet."""
    gpu = _har_gpu()
    print(f"GPU tilgjengelig: {gpu}")

    kwargs = dict(
        lang="en",
        device="gpu" if gpu else "cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_type="max",
        text_det_limit_side_len=DET_SIDE_LEN,
        text_recognition_batch_size=REC_BATCH * 2 if gpu else REC_BATCH,
        text_detection_model_name=DET_MODELL,
        text_recognition_model_name=REC_MODELL,
        text_detection_model_dir=DET_MODELL_DIR,
        text_recognition_model_dir=REC_MODELL_DIR,
    )
    if gpu:
        kwargs["precision"] = "fp16"
    else:
        kwargs["enable_mkldnn"] = True

    return PaddleOCR(**kwargs)


def render_side(side, dpi=PDF_DPI):
    """Render en fitz-side til numpy-array (RGB)."""
    pix = side.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    return img.copy()


def _hent_tekstbokser(res):
    """
    Hent tekstlinjer med polygon-koordinater fra PaddleOCR-resultat.
    Returnerer liste med (tekst, x0, y0, x1, y1) i pikselkoordinater.
    """
    bokser = []
    if not res:
        return bokser

    # Prøv ord-nivå først
    ord_per_linje = res.get("text_word")
    boks_per_linje = res.get("text_word_boxes")
    if ord_per_linje and boks_per_linje:
        for ord_liste, boks_liste in zip(ord_per_linje, boks_per_linje):
            for tekst, boks in zip(ord_liste, boks_liste):
                if not tekst.strip():
                    continue
                pts = np.asarray(boks, dtype=float).reshape(-1)
                x0, y0 = float(pts[0]), float(pts[1])
                x1, y1 = float(pts[2]), float(pts[3])
                # Sørg for min/max
                bokser.append((tekst, min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        if bokser:
            return bokser

    # Fallback: linjenivå (fire hjørnepunkter per boks)
    tekster = res.get("rec_texts") or []
    polys = res.get("rec_polys")
    if polys is None:
        polys = res.get("dt_polys") or []
    for tekst, poly in zip(tekster, polys):
        if not tekst.strip():
            continue
        pts = np.asarray(poly, dtype=float)
        x0 = float(pts[:, 0].min())
        y0 = float(pts[:, 1].min())
        x1 = float(pts[:, 0].max())
        y1 = float(pts[:, 1].max())
        bokser.append((tekst, x0, y0, x1, y1))

    return bokser


def lag_pdf_med_tekst(inn_pdf_sti, ut_pdf_sti, synlig=True, bakgrunn_opacity=0.5, dpi=PDF_DPI):
    """
    Les inn-PDF, kjør PaddleOCR per side, og skriv en ny PDF der teksten
    er plassert i posisjonene OCR fant den.

    Parametere:
        inn_pdf_sti:      sti til input-PDF
        ut_pdf_sti:       sti til output-PDF
        synlig:           om teksten skal være synlig (True) eller usynlig overlay (False)
        bakgrunn_opacity: opacity for originaldokumentet (0.0-1.0, default 0.5)
        dpi:              DPI for rendering av PDF-sider
    """
    print("Initialiserer PaddleOCR...")
    reader = _lag_reader()

    inn_dok = fitz.open(inn_pdf_sti)
    ut_dok = fitz.open()

    for side_nr in range(inn_dok.page_count):
        side = inn_dok[side_nr]
        bredde_pt = side.rect.width
        hoyde_pt = side.rect.height

        print(f"  Side {side_nr + 1}/{inn_dok.page_count} ({bredde_pt:.0f}×{hoyde_pt:.0f} pt) ...", end=" ")

        # Render siden til bilde for OCR
        bilde = render_side(side, dpi=dpi)
        bilde_h, bilde_w = bilde.shape[:2]

        # Skaleringsfaktorer: piksel -> PDF-punkt
        skala_x = bredde_pt / bilde_w
        skala_y = hoyde_pt / bilde_h

        # PaddleOCR forventer BGR
        bilde_bgr = np.ascontiguousarray(bilde[:, :, ::-1])

        # Kjør PaddleOCR
        resultater = reader.predict(bilde_bgr, return_word_box=True)
        res = resultater[0] if resultater else None
        tekstbokser = _hent_tekstbokser(res)
        print(f"{len(tekstbokser)} tekstfragmenter funnet")

        # Opprett ny side med samme dimensjoner
        ny_side = ut_dok.new_page(width=bredde_pt, height=hoyde_pt)

        # Sett inn originalsiden som bakgrunn med redusert opacity
        if bakgrunn_opacity < 1.0:
            ny_side.draw_rect(ny_side.rect, color=None, fill=(1, 1, 1))
            ny_side.show_pdf_page(ny_side.rect, inn_dok, side_nr)
            # Legg et semi-transparent hvitt rektangel over for å "fade" bakgrunnen
            fade_alpha = 1.0 - bakgrunn_opacity
            shape = ny_side.new_shape()
            shape.draw_rect(ny_side.rect)
            shape.finish(color=None, fill=(1, 1, 1), fill_opacity=fade_alpha)
            shape.commit()
        else:
            ny_side.show_pdf_page(ny_side.rect, inn_dok, side_nr)

        for tekst, x0_px, y0_px, x1_px, y1_px in tekstbokser:
            # Konverter fra piksel til PDF-punkt
            x0 = x0_px * skala_x
            y0 = y0_px * skala_y
            x1 = x1_px * skala_x
            y1 = y1_px * skala_y

            rect = fitz.Rect(x0, y0, x1, y1)
            boks_hoyde_pt = y1 - y0

            # Velg fontstørrelse som passer boksens høyde
            fontstr = max(4, boks_hoyde_pt * 0.85)

            if synlig:
                ny_side.insert_textbox(
                    rect,
                    tekst,
                    fontsize=fontstr,
                    fontname="helv",
                    color=(0, 0, 0),
                    align=fitz.TEXT_ALIGN_LEFT,
                )
            else:
                ny_side.insert_textbox(
                    rect,
                    tekst,
                    fontsize=fontstr,
                    fontname="helv",
                    color=(0, 0, 0),
                    align=fitz.TEXT_ALIGN_LEFT,
                    render_mode=3,  # usynlig
                )

    ut_dok.save(ut_pdf_sti)
    ut_dok.close()
    inn_dok.close()
    print(f"\nFerdig! Skrev {ut_pdf_sti}")


def main():
    p = argparse.ArgumentParser(
        description="Kjør PaddleOCR på en PDF og lag ny PDF med tekst plassert i riktige posisjoner."
    )
    p.add_argument("pdf", help="input-PDF")
    p.add_argument("-o", "--output", default=None, help="output-PDF (default: <input>_ocr.pdf)")
    p.add_argument("--synlig", action="store_true", default=True,
                   help="gjør OCR-teksten synlig (default: på)")
    p.add_argument("--usynlig", action="store_true",
                   help="gjør OCR-teksten usynlig (søkbart lag uten visuell tekst)")
    p.add_argument("--dpi", type=int, default=PDF_DPI,
                   help=f"DPI for rendering av PDF-sider (default: {PDF_DPI})")
    p.add_argument("--opacity", type=float, default=0.5,
                   help="opacity for originaldokumentet i bakgrunnen (0.0-1.0, default: 0.5)")
    args = p.parse_args()

    if not args.output:
        args.output = args.pdf.rsplit(".", 1)[0] + "_ocr.pdf"

    synlig = not args.usynlig

    lag_pdf_med_tekst(
        inn_pdf_sti=args.pdf,
        ut_pdf_sti=args.output,
        synlig=synlig,
        bakgrunn_opacity=args.opacity if synlig else 1.0,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()


