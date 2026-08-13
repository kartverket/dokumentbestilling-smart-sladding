import glob
import os
import re

import fitz
from PIL import Image, ImageDraw, ImageFont

from load_pdf import PDF_DPI
from utils_config import (
    BOM_FARGE, KORREKT_PADDLE_FARGE, KORREKT_YOLO_FARGE, KORREKT_BEGGE_FARGE,
    OVERSLADD_PADDLE_FARGE, OVERSLADD_YOLO_FARGE, OVERSLADD_BEGGE_FARGE, UKJENT_FARGE
)

SKALA = PDF_DPI / 72.0           # PDF-punkt -> piksel

_CONF_FONT = None


def _conf_font():
    global _CONF_FONT
    if _CONF_FONT is None:
        _CONF_FONT = ImageFont.load_default(size=28)
    return _CONF_FONT


def _tegn_conf(tegner, r, conf, farge):
    if conf is None:
        return
    tegner.text((r[0] + 2, max(r[1] + 2, 2)), f"{conf:.2f}",
                fill=farge, font=_conf_font())


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


def _er_oversladd(boks, oversladd_liste):
    """Sjekk om en boks finnes i over-sladding-lista (sammenlign koordinater)."""
    x0, y0, x1, y1 = boks[:4]
    for ob in oversladd_liste:
        ox0, oy0, ox1, oy1 = ob[:4]
        if abs(x0 - ox0) < 0.5 and abs(y0 - oy0) < 0.5 and \
           abs(x1 - ox1) < 0.5 and abs(y1 - oy1) < 0.5:
            return True
    return False


def _velg_farge(kilde, er_over):
    """Velg farge basert paa kilde og om boksen er over-sladding."""
    if kilde not in ("paddle", "yolo", "begge"):
        return UKJENT_FARGE
    if er_over:
        if kilde == "paddle":
            return OVERSLADD_PADDLE_FARGE
        if kilde == "yolo":
            return OVERSLADD_YOLO_FARGE
        return OVERSLADD_BEGGE_FARGE
    else:
        if kilde == "paddle":
            return KORREKT_PADDLE_FARGE
        if kilde == "yolo":
            return KORREKT_YOLO_FARGE
        return KORREKT_BEGGE_FARGE


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
            base = bilde.convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            tegner = ImageDraw.Draw(overlay)

            bw, bh, funnet = sladd_bokser.get((navn, si), (bilde.width, bilde.height, []))
            sx, sy = bilde.width / bw, bilde.height / bh

            # Hent over-sladding-liste for denne siden (om tilgjengelig)
            over_liste = []
            if oversladd_bokser and (navn, si) in oversladd_bokser:
                _, _, over_liste = oversladd_bokser[(navn, si)]

            # 1) Tegn prediksjons-bokser (kun outline, ingen fyll)
            if kilder and (navn, si) in kilder:
                _, _, med_kilde = kilder[(navn, si)]
                for boks in med_kilde:
                    x0, y0, x1, y1 = boks[:4]
                    kilde_val = boks[4] if len(boks) > 4 else "paddle"
                    conf = boks[5] if len(boks) > 5 else None
                    r = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]

                    er_over = _er_oversladd(boks, over_liste) if over_liste else False
                    farge = _velg_farge(kilde_val, er_over)
                    tegner.rectangle(r, outline=farge, width=3)
                    if kilde_val in ("yolo", "begge"):
                        _tegn_conf(tegner, r, conf, farge)

            # 2) YOLO kjort live (--yolo): egne rammer med conf
            if yolo_bokser and (navn, si) in yolo_bokser:
                yw, yh, yolo_f = yolo_bokser[(navn, si)]
                ysx, ysy = bilde.width / yw, bilde.height / yh
                for boks in yolo_f:
                    x0, y0, x1, y1 = boks[:4]
                    r = [x0 * ysx, y0 * ysy, x1 * ysx, y1 * ysy]
                    tegner.rectangle(r, outline=KORREKT_YOLO_FARGE, width=3)
                    _tegn_conf(tegner, r, boks[4] if len(boks) > 4 else None,
                               KORREKT_YOLO_FARGE)

            # 3) Fasit: vis kun bommede bokser (roed)
            if ground_truth:
                for fi, (x, y, w, h, _t) in enumerate(ground_truth.get((nr, si), [])):
                    if bom_indekser is None or (nr, si, fi) in bom_indekser:
                        tegner.rectangle(
                            _fasit_piksler((x, y, w, h), bilde.height, y_origin),
                            outline=BOM_FARGE, width=3)

            # Komponer overlay med 0.8 opacity paa base
            bilde = Image.alpha_composite(base, overlay).convert("RGB")

            ut = os.path.join(ut_mappe, f"{os.path.splitext(navn)[0]}_side{si}.png")
            bilde.save(ut)
            if skriv_logg:
                n_fa = len(ground_truth.get((nr, si), [])) if ground_truth else 0
                print(f"   {navn} side {si}: {len(funnet)} funnet, {n_fa} ground_truth -> {ut}")
        d.close()