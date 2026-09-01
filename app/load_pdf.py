import fitz
import numpy as np

from config import PDF_DPI


def _render(document):
    pages = []
    for page in document:
        pix = page.get_pixmap(dpi=PDF_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:  
            img = img[:, :, :3]
        pages.append(img.copy())  
    document.close()
    return pages


def read_pages(path):
    return _render(fitz.open(path))


def read_pages_from_bytes(pdf_bytes):
    return _render(fitz.open(stream=pdf_bytes, filetype="pdf"))
