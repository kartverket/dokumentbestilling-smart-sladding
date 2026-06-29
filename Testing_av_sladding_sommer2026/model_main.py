from load_pdf import les_sider_fra_bytes, PDF_DPI
from ocr_model_fnr import les_tokens_batched, finn_bokser_fra_tokens

from collections import defaultdict


def run_model_on_pdf_bytes(pdf_bytes):
    bilder = list(les_sider_fra_bytes(pdf_bytes))
    tokens_per_side = les_tokens_batched(bilder)

    sider = []
    for si, (bilde, tokens) in enumerate(zip(bilder, tokens_per_side), start=1):
        bokser = [
            {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            for (x0, y0, x1, y1), _mod11 in finn_bokser_fra_tokens(tokens)
        ]
        sider.append({
            "side": si,
            "bilde_bredde": bilde.size[0],
            "bilde_hoyde": bilde.size[1],
            "bokser": bokser,
        })
    return {"dpi": PDF_DPI, "sider": sider}
