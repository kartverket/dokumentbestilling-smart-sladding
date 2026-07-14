import glob
import os
import re

import fitz
from PIL import Image, ImageDraw, ImageFont

from load_pdf import PDF_DPI
from utils_config import (
    FUNNET_FARGE, PADDLE_FARGE, YOLO_FARGE, OVERSLADD_FARGE, FASIT_FARGE
)

SKALA = PDF_DPI / 72.0           # PDF-punkt -> piksel


def _dok_nr(navn):
    m = re.match(r"0*(\d+)", os.path.basename(navn))
    return int(m.group(1)) if m else None


def _fasit_piksler(boks, ph_px, y_origin):
    x, y, w, h = boks
    x0, x1 = sorted((x, x + w))
    y0, y1 = sorted((y, y + h))
    if y_origin == "bunn":
        ph_pt = ph_px / SKALA
        y0, y1 = ph_pt - max(y, y + h), ph_pt - min(y, y + h)
    return [x0 * SKALA, y0 * SKALA, x1 * SKALA, y1 * SKALA]


def _render_side(side):
    pix = side.get_pixmap(dpi=PDF_DPI)
    modus = "RGBA" if pix.n == 4 else "RGB"
    return Image.frombytes(modus, (pix.w, pix.h), pix.samples).convert("RGB")


def _sider_aa_tegne(sladd_bokser, ground_truth, mappe):
    per_fil = {}
    for (navn, si) in sladd_bokser:
        per_fil.setdefault(navn, set()).add(si)

    if ground_truth:
        nr_til_navn = {}
        for navn in per_fil:
            nr_til_navn.setdefault(_dok_nr(navn), navn)
        for (nr, si) in ground_truth:
            navn = nr_til_navn.get(nr)
            if navn:
                per_fil.setdefault(navn, set()).add(si)

    return {navn: sorted(sider) for navn, sider in per_fil.items()}


def tegn_og_lagre(sladd_bokser, ground_truth, mappe, ut_mappe, y_origin="topp",
                  skriv_logg=True, rydd=True, yolo_bokser=None, kilder=None,
                  oversladd_bokser=None, bom_indekser=None):
    os.makedirs(ut_mappe, exist_ok=True)
    if rydd:
        for png in glob.glob(os.path.join(ut_mappe, "*.png")):
            os.remove(png)

    per_fil = _sider_aa_tegne(sladd_bokser, ground_truth, mappe)

    for navn in sorted(per_fil):
        nr = _dok_nr(navn)
        try:
            d = fitz.open(os.path.join(mappe, navn))
        except Exception as e:
            print(f"   {navn}: kunne ikke aapnes ({e!r})")
            continue
        for si in per_fil[navn]:
            if not 1 <= si <= len(d):
                print(f"   {navn} side {si}: finnes ikke i PDF-en ({len(d)} sider)")
                continue
            bilde = _render_side(d[si - 1])
            tegner = ImageDraw.Draw(bilde)

            bw, bh, funnet = sladd_bokser.get((navn, si), (bilde.width, bilde.height, []))
            sx, sy = bilde.width / bw, bilde.height / bh

            # 1) svart fyll for all sladding
            for boks in funnet:
                x0, y0, x1, y1 = boks[:4]
                tegner.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], fill=FUNNET_FARGE)

            # 2) kilde-rammer oppaa sladden
            if kilder and (navn, si) in kilder:
                _, _, med_kilde = kilder[(navn, si)]
                for boks in med_kilde:
                    x0, y0, x1, y1 = boks[:4]
                    kilde = boks[4] if len(boks) > 4 else "paddle"
                    r = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
                    if kilde in ("paddle", "begge"):
                        tegner.rectangle(r, outline=PADDLE_FARGE, width=3)
                    if kilde in ("yolo", "begge"):
                        # "begge": roed indre ramme innenfor den blaa
                        indre = [r[0] + 4, r[1] + 4, r[2] - 4, r[3] - 4] if kilde == "begge" else r
                        tegner.rectangle(indre, outline=YOLO_FARGE, width=3)
            elif yolo_bokser and (navn, si) in yolo_bokser:
                # bakoverkompatibel modus: kun yolo-rammer, som foer
                _, _, yolo_f = yolo_bokser[(navn, si)]
                for boks in yolo_f:
                    x0, y0, x1, y1 = boks[:4]
                    conf = boks[4] if len(boks) > 4 else None
                    r = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
                    tegner.rectangle(r, outline=YOLO_FARGE, width=3)
                    if conf is not None:
                        font = ImageFont.load_default(size=28)
                        ty = max(r[1] + 2, 2)   # inni boksen øverst
                        tegner.text((r[0] + 2, ty), f"{conf:.2f}",
                                    fill=YOLO_FARGE, font=font)

            # 3) fasit oeverst
            if ground_truth:
                for fi, (x, y, w, h, _t) in enumerate(ground_truth.get((nr, si), [])):
                    tegner.rectangle(_fasit_piksler((x, y, w, h), bilde.height, y_origin),
                                     outline=FASIT_FARGE, width=3)

            # 4) over-sladding: oransje outline
            if oversladd_bokser and (navn, si) in oversladd_bokser:
                _, _, over_f = oversladd_bokser[(navn, si)]
                for boks in over_f:
                    x0, y0, x1, y1 = boks[:4]
                    r = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
                    tegner.rectangle(r, outline=OVERSLADD_FARGE, width=4)

            ut = os.path.join(ut_mappe, f"{os.path.splitext(navn)[0]}_side{si}.png")
            bilde.save(ut)
            if skriv_logg:
                n_fa = len(ground_truth.get((nr, si), [])) if ground_truth else 0
                print(f"   {navn} side {si}: {len(funnet)} funnet, {n_fa} ground_truth -> {ut}")
        d.close()