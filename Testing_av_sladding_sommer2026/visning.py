import csv
import glob
import os
import re

import fitz
from PIL import ImageDraw

from sladd_lib import (
    les_sider, _les_tokens, _grupper_til_linjer, _bygg_linjetekst,
    analyzer, gyldig_mod11, _dok_nr,
)


def _skriv_ocr_logg(logg, navn, si, bilde):
    linjer = _grupper_til_linjer(_les_tokens(bilde))
    logg.write(f"\n===== {navn} side {si} — {len(linjer)} tekstlinjer =====\n")
    for li, linje in enumerate(sorted(linjer, key=lambda l: min(t.y0 for t in l)), start=1):
        tekst, _kart = _bygg_linjetekst(linje)
        if not tekst.strip():
            continue
        treff = analyzer.analyze(tekst, entities=["NO_FNR"], language="en")
        if treff:
            merker = []
            for tr in treff:
                cifre = re.sub(r"\D", "", tekst[tr.start:tr.end])
                merker.append(f"{cifre} (mod11 {'OK' if gyldig_mod11(cifre) else 'FEIL'})")
            logg.write(f"  linje {li:>2}: {tekst.strip()!r}   <-- FNR-TREFF: {', '.join(merker)}\n")
        else:
            logg.write(f"  linje {li:>2}: {tekst.strip()!r}\n")


def tegn_og_lagre(sladd_bokser, fasit, mappe, ut_mappe,
                  y_origin="topp", skriv_logg=True, rydd=True):
    os.makedirs(ut_mappe, exist_ok=True)
    if rydd:
        for f in glob.glob(os.path.join(ut_mappe, "*.png")):
            os.remove(f)

    logg_sti = os.path.join(ut_mappe, "ocr_linjer.txt")
    csv_sti = os.path.join(ut_mappe, "bokser.csv")
    print(f"Tegner og lagrer bilder i «{ut_mappe}/» …")

    with open(logg_sti, "w", encoding="utf-8") as logg, \
         open(csv_sti, "w", newline="", encoding="utf-8") as csvfil:

        boks_csv = csv.writer(csvfil)
        boks_csv.writerow(["fil", "dok_nr", "side", "type", "height", "width", "x", "y"])

        behandlede = sorted({navn for (navn, _si) in sladd_bokser})
        for navn in behandlede:
            fil = os.path.join(mappe, navn)
            stem = os.path.splitext(navn)[0]
            nr = _dok_nr(navn)
            er_pdf = fil.lower().endswith(".pdf")
            doc = fitz.open(fil) if er_pdf else None

            for si, bilde in enumerate(les_sider(fil), start=1):
                iw, ih = bilde.size
                pw, ph = (doc[si - 1].rect.width, doc[si - 1].rect.height) if er_pdf else (iw, ih)

                if skriv_logg:
                    _skriv_ocr_logg(logg, navn, si, bilde)

                dr = ImageDraw.Draw(bilde)
                raw = sladd_bokser.get((navn, si), (iw, ih, []))[2]
                for (x0, y0, x1, y1) in raw:                       
                    dr.rectangle([x0, y0 - 2, x1, y1 + 2], fill=(0, 0, 0))
                    boks_csv.writerow([navn, nr, si, "sladd",
                                       round(y1 - y0, 1), round(x1 - x0, 1),
                                       round(x0, 1), round(y0, 1)])

                sx, sy = iw / pw, ih / ph
                fasit_paa_siden = fasit.get((nr, si), []) if fasit else []
                for (x, y, w, h, t) in fasit_paa_siden:            
                    X0, X1 = sorted((x, x + w))
                    Y0, Y1 = sorted((y, y + h))
                    if y_origin == "bunn":
                        Y0, Y1 = ph - Y1, ph - Y0
                    px0, py0, px1, py1 = X0 * sx, Y0 * sy, X1 * sx, Y1 * sy
                    dr.rectangle([px0, py0, px1, py1], outline=(0, 200, 0), width=4)
                    boks_csv.writerow([navn, nr, si, "fasit",
                                       round(py1 - py0, 1), round(px1 - px0, 1),
                                       round(px0, 1), round(py0, 1)])

                utfil = os.path.join(ut_mappe, f"{stem}_side{si}.png")
                bilde.save(utfil)
                logg.write(f"{navn} side {si}:  sladd={len(raw)} (svart)   "
                           f"fasit={len(fasit_paa_siden)} (grønn ramme)  ->  {utfil}\n")

            if doc is not None:
                doc.close()

    print(f"Lagret bilder + ocr_linjer.txt + bokser.csv for {len(behandlede)} fil(er) i «{ut_mappe}/».")
