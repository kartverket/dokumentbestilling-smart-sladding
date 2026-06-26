import io

import fitz
from PIL import Image

PDF_DPI = 300


def _render(dokument):
    sider = []
    for side in dokument:
        png = side.get_pixmap(dpi=PDF_DPI).tobytes("png")
        sider.append(Image.open(io.BytesIO(png)).convert("RGB"))
    dokument.close()
    return sider


def les_sider(sti):
    return _render(fitz.open(sti))


def les_sider_fra_bytes(pdf_bytes):
    return _render(fitz.open(stream=pdf_bytes, filetype="pdf"))
