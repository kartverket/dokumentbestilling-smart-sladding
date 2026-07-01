import time

from load_pdf import les_sider_fra_bytes, PDF_DPI
from ocr_model_fnr import les_tokens_batched, finn_bokser_fra_tokens, ocr_linjer_fra_tokens


def run_model_on_pdf_bytes(pdf_bytes, skriv_tid=False, med_linjer=False):
    t = {}

    tstart = time.perf_counter()
    bilder = list(les_sider_fra_bytes(pdf_bytes))
    t["render"] = time.perf_counter() - tstart

    tstart = time.perf_counter()
    tokens_per_side = les_tokens_batched(bilder)
    t["ocr"] = time.perf_counter() - tstart

    sider = []

    tstart = time.perf_counter()
    for si, (bilde, tokens) in enumerate(zip(bilder, tokens_per_side), start=1):
        treff = finn_bokser_fra_tokens(tokens)

        bokser = [
            {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            for (x0, y0, x1, y1), _mod11 in treff
        ]
        h, w = bilde.shape[:2]
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
        _skriv_tid(t)

    return {"dpi": PDF_DPI, "sider": sider}


def _skriv_tid(t):
    total = t["render"] + t["ocr"] + t.get("etterbehandling", 0.0)

    def linje(navn, sek):
        pct = (sek / total * 100) if total else 0.0
        return f"  {navn:<18}{sek:9.3f} s{pct:7.1f}%"

    print("Timing:")
    print(linje("render",          t["render"]))
    print(linje("ocr (batched)",   t["ocr"]))
    print(linje("etterbehandling", t.get("etterbehandling", 0.0)))
    print(f"  {'Total':<18}{total:9.3f} s")