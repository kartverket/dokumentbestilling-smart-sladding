import fitz
import numpy as np

PDF_DPI = 300


def _render(dokument):
    sider = []
    for side in dokument:
        pix = side.get_pixmap(dpi=PDF_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:  # RGBA → RGB
            img = img[:, :, :3]
        sider.append(img.copy())  # copy: pixmap buffer is reused
    dokument.close()
    return sider


def les_sider(sti):
    return _render(fitz.open(sti))


def les_sider_fra_bytes(pdf_bytes):
    return _render(fitz.open(stream=pdf_bytes, filetype="pdf"))
