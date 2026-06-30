from load_pdf import les_sider_fra_bytes, PDF_DPI
from paddleocr_model_fnr import finn_bokser


def run_model_on_pdf_bytes(pdf_bytes):
    sider = []
    for si, bilde in enumerate(les_sider_fra_bytes(pdf_bytes), start=1):
        bokser = [
            {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            for (x0, y0, x1, y1), _mod11 in finn_bokser(bilde)
        ]
        sider.append({
            "side": si,
            "bilde_bredde": bilde.size[0],
            "bilde_hoyde": bilde.size[1],
            "bokser": bokser,
        })
    return {"dpi": PDF_DPI, "sider": sider}
