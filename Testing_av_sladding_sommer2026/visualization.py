import glob
import os

from PIL import ImageDraw

from load_pdf import les_sider, PDF_DPI

FUNNET_FARGE = (0, 0, 0)        # svart fyll = faktisk sladding
FASIT_FARGE = (30, 160, 30)     # grønt = ground_truth
SKALA = PDF_DPI / 72.0          # PDF-punkt -> piksel


def _dok_nr(navn):
    import re
    m = re.match(r"0*(\d+)", os.path.basename(navn))
    return int(m.group(1)) if m else None


def _fasit_piksler(boks, ph_px, y_origin):
    "Fasit (x,y,w,h) i PDF-punkt -> pikselrektangel i bildet."
    x, y, w, h = boks
    x0, x1 = sorted((x, x + w))
    y0, y1 = sorted((y, y + h))
    if y_origin == "bunn":
        ph_pt = ph_px / SKALA
        y0, y1 = ph_pt - max(y, y + h), ph_pt - min(y, y + h)
    return [x0 * SKALA, y0 * SKALA, x1 * SKALA, y1 * SKALA]


def tegn_og_lagre(sladd_bokser, ground_truth, mappe, ut_mappe, y_origin="topp",
                  skriv_logg=True, rydd=True):
    "Sladder funne bokser (svart fyll) + tegner ground_truth (grønn ramme) på hver side og lagrer PNG."
    os.makedirs(ut_mappe, exist_ok=True)
    if rydd:
        for png in glob.glob(os.path.join(ut_mappe, "*.png")):
            os.remove(png)

    # grupper sider per fil så vi rendrer hver PDF bare én gang
    per_fil = {}
    for (navn, si) in sladd_bokser:
        per_fil.setdefault(navn, []).append(si)

    for navn in sorted(per_fil):
        nr = _dok_nr(navn)
        try:
            bilder = les_sider(os.path.join(mappe, navn))
        except Exception as e:
            print(f"   {navn}: kunne ikke rendres ({e!r})")
            continue
        for si in sorted(per_fil[navn]):
            bilde = bilder[si - 1].copy()
            tegner = ImageDraw.Draw(bilde)
            _bw, _bh, funnet = sladd_bokser[(navn, si)]
            for (x0, y0, x1, y1) in funnet:
                tegner.rectangle([x0, y0, x1, y1], fill=FUNNET_FARGE)
            if ground_truth:
                for (x, y, w, h, _t) in ground_truth.get((nr, si), []):
                    tegner.rectangle(_fasit_piksler((x, y, w, h), bilde.height, y_origin),
                                     outline=FASIT_FARGE, width=3)
            ut = os.path.join(ut_mappe, f"{os.path.splitext(navn)[0]}_side{si}.png")
            bilde.save(ut)
            if skriv_logg:
                n_fa = len(ground_truth.get((nr, si), [])) if ground_truth else 0
                print(f"   {navn} side {si}: {len(funnet)} funnet, {n_fa} ground_truth -> {ut}")