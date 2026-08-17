"""
Kjør EasyOCR på et PDF-dokument og lag en ny PDF
med teksten plassert i riktige posisjoner.

Standard: synlig tekst over originaldokumentet med 50% opacity,
slik at man tydelig ser OCR-teksten oppå dokumentet.

Bruk:
    python utils/easyocr_til_pdf.py dokument.pdf -o ut.pdf
    python utils/easyocr_til_pdf.py dokument.pdf -o ut.pdf --opacity 0.3
    python utils/easyocr_til_pdf.py dokument.pdf -o ut.pdf --usynlig
    python utils/easyocr_til_pdf.py dokument.pdf -o ut.pdf --spraak no en
"""

import argparse

import easyocr
import fitz  # PyMuPDF
import numpy as np

# DPI brukt for rendering – bestemmer oppløsningen OCR jobber på
RENDER_DPI = 300


def render_side(side, dpi=RENDER_DPI):
    """Render en fitz-side til numpy-array (RGB)."""
    pix = side.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    return img.copy()


def kjor_easyocr(reader, bilde):
    """Kjør EasyOCR på et bilde og returner liste med (bbox, tekst, konfidens)."""
    resultater = reader.readtext(bilde)
    return resultater


def lag_pdf_med_tekst(inn_pdf_sti, ut_pdf_sti, spraak, synlig=True, min_konf=0.1, bakgrunn_opacity=0.5, dpi=300):
    """
    Les inn-PDF, kjør EasyOCR per side, og skriv en ny PDF der teksten
    er plassert i posisjonene OCR fant den.

    Parametere:
        inn_pdf_sti:      sti til input-PDF
        ut_pdf_sti:       sti til output-PDF
        spraak:           liste med språkkoder for EasyOCR (f.eks. ['no', 'en'])
        synlig:           om teksten skal være synlig (True) eller usynlig overlay (False)
        min_konf:         minimum OCR-konfidens for å inkludere teksten
        bakgrunn_opacity: opacity for originaldokumentet (0.0-1.0, default 0.5)
    """
    print(f"Initialiserer EasyOCR med språk: {spraak}")
    reader = easyocr.Reader(spraak, gpu=True)

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

        # Kjør EasyOCR
        resultater = kjor_easyocr(reader, bilde)
        print(f"{len(resultater)} tekstfragmenter funnet")

        # Opprett ny side med samme dimensjoner
        ny_side = ut_dok.new_page(width=bredde_pt, height=hoyde_pt)

        # Sett inn originalsiden som bakgrunn med redusert opacity
        if bakgrunn_opacity < 1.0:
            # Tegn hvit bakgrunn først
            ny_side.draw_rect(ny_side.rect, color=None, fill=(1, 1, 1))
            ny_side.show_pdf_page(ny_side.rect, inn_dok, side_nr)
            # Legg et semi-transparent hvitt rektangel over for å "fade" bakgrunnen
            fade_alpha = 1.0 - bakgrunn_opacity
            ny_side.draw_rect(
                ny_side.rect,
                color=None,
                fill=(1, 1, 1),
                opacity=fade_alpha,
            )
        else:
            ny_side.show_pdf_page(ny_side.rect, inn_dok, side_nr)

        for bbox, tekst, konf in resultater:
            if konf < min_konf:
                continue

            # EasyOCR returnerer bbox som 4 hjørner: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x0_px = min(xs)
            y0_px = min(ys)
            x1_px = max(xs)
            y1_px = max(ys)

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
                # Synlig tekst (svart, tydelig over faded bakgrunn)
                ny_side.insert_textbox(
                    rect,
                    tekst,
                    fontsize=fontstr,
                    fontname="helv",
                    color=(0, 0, 0),
                    align=fitz.TEXT_ALIGN_LEFT,
                )
            else:
                # Usynlig tekst (søkbart lag, som i vanlig OCR-PDF)
                ny_side.insert_textbox(
                    rect,
                    tekst,
                    fontsize=fontstr,
                    fontname="helv",
                    color=(0, 0, 0),
                    align=fitz.TEXT_ALIGN_LEFT,
                    render_mode=3,  # 3 = usynlig (invisible)
                )

    ut_dok.save(ut_pdf_sti)
    ut_dok.close()
    inn_dok.close()
    print(f"\nFerdig! Skrev {ut_pdf_sti}")


def main():
    p = argparse.ArgumentParser(
        description="Kjør EasyOCR på en PDF og lag ny PDF med tekst plassert i riktige posisjoner."
    )
    p.add_argument("pdf", help="input-PDF")
    p.add_argument("-o", "--output", default=None, help="output-PDF (default: <input>_ocr.pdf)")
    p.add_argument("--spraak", nargs="+", default=["no", "en"],
                   help="språk for EasyOCR (default: no en)")
    p.add_argument("--synlig", action="store_true", default=True,
                   help="gjør OCR-teksten synlig (default: på)")
    p.add_argument("--usynlig", action="store_true",
                   help="gjør OCR-teksten usynlig (søkbart lag uten visuell tekst)")
    p.add_argument("--min-konf", type=float, default=0.1,
                   help="minimum konfidens for å inkludere tekst (0.0-1.0)")
    p.add_argument("--dpi", type=int, default=300,
                   help="DPI for rendering av PDF-sider (default: 300)")
    p.add_argument("--opacity", type=float, default=0.5,
                   help="opacity for originaldokumentet i bakgrunnen (0.0-1.0, default: 0.5)")
    args = p.parse_args()

    if not args.output:
        args.output = args.pdf.rsplit(".", 1)[0] + "_ocr.pdf"


    synlig = not args.usynlig

    lag_pdf_med_tekst(
        inn_pdf_sti=args.pdf,
        ut_pdf_sti=args.output,
        spraak=args.spraak,
        synlig=synlig,
        min_konf=args.min_konf,
        bakgrunn_opacity=args.opacity if synlig else 1.0,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()









