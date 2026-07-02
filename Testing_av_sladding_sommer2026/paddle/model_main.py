import time

import numpy as np

from load_pdf import les_sider_fra_bytes, PDF_DPI
from paddle_ocr_model_fnr import les_tokens_batched, finn_bokser_fra_tokens, ocr_linjer_fra_tokens
from orientering import finn_rotasjon, boks_tilbake


def run_model_on_pdf_bytes(pdf_bytes, skriv_tid=False, med_linjer=False, navn=None):
    t = {}

    tstart = time.perf_counter()
    bilder = list(les_sider_fra_bytes(pdf_bytes))
    t["render"] = time.perf_counter() - tstart

    tstart = time.perf_counter()
    rotasjoner = [finn_rotasjon(b) for b in bilder]
    bilder_ocr = [np.rot90(b, k) if k else b for b, k in zip(bilder, rotasjoner)]
    tokens_per_side = les_tokens_batched(bilder_ocr)
    t["ocr"] = time.perf_counter() - tstart

    sider = []

    tstart = time.perf_counter()
    for si, (bilde, tokens, k) in enumerate(
            zip(bilder, tokens_per_side, rotasjoner), start=1):
        treff = finn_bokser_fra_tokens(tokens)

        h, w = bilde.shape[:2]        # ORIGINALENS dimensjoner
        bokser = []
        for (boks, _mod11) in treff:
            x0, y0, x1, y1 = boks_tilbake(boks, k, w, h)
            bokser.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
        side_data = {
            "side": si,
            "bilde_bredde": w,
            "bilde_hoyde": h,
            "bokser": bokser,
        }
        if med_linjer:
            side_data["linjer"] = ocr_linjer_fra_tokens(tokens)
        sider.append(side_data)
    t["etterbehandling"] = time.perf_counter() - tstart

    if skriv_tid:
        _skriv_tid(t, len(sider), navn)

    return {"dpi": PDF_DPI, "sider": sider}


def _skriv_tid(t, n_sider, navn=None):
    total = t["render"] + t["ocr"] + t.get("etterbehandling", 0.0)

    def linje(navn, sek):
        pct = (sek / total * 100) if total else 0.0
        return f"  {navn:<18}{sek:9.3f} s{pct:7.1f}%"

    tittel = f"Timing [{navn}]:" if navn else "Timing:"
    print(tittel)
    print(linje("render",          t["render"]))
    print(linje("ocr (batched)",   t["ocr"]))
    print(linje("etterbehandling", t.get("etterbehandling", 0.0)))
    print(f"  {'Total':<18}{total:9.3f} s")
    print(f"  {'Sider totalt':<18}{n_sider:9d}")
    if n_sider:
        print(f"  {'Per side':<18}{total / n_sider:9.3f} s")