"""
Visuell gjennomgang av bokser som fjernes av filterkonfigurasjoner.

Matcher prediksjoner mot fasit (som filter_sweep), deretter tegner
bokser som ville blitt fjernet av angitt filter på original-PDF-ene.

Farger:
  RØD      = riktig prediksjon som fjernes (recall-tap — dårlig)
  GRØNN    = oversladding som fjernes (presisjonsgevinst — bra)
  GRÅ      = beholdte bokser (kontekst)
  BLÅ      = fasit-bokser

Kjør enkelt-konfigurasjon:
    python utils/filter_review.py \\
        --fasit-csv labels.csv \\
        --res-csv resultat.csv \\
        --mappe /sti/til/pdfer \\
        --elongation 1.5

Kjør med ulike parametre per kilde:
    python utils/filter_review.py \\
        --fasit-csv labels.csv \\
        --res-csv resultat.csv \\
        --mappe /sti/til/pdfer \\
        --per-kilde "yolo:e=2.5,h=40,b=80" "paddle:e=1.5,h=60,b=120"

Kjør alle konfigurasjoner (sweep):
    python utils/filter_review.py \\
        --fasit-csv labels.csv \\
        --res-csv resultat.csv \\
        --mappe /sti/til/pdfer \\
        --sweep
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

import fitz
from PIL import Image, ImageDraw, ImageFont

from load_pdf import PDF_DPI

SKALA = PDF_DPI / 72.0


# ── Farger ────────────────────────────────────────────────────
RIKTIG_FJERNET    = (220, 30, 30, 220)      # rød = riktig boks fjernet
OVERSLADD_FJERNET = (30, 180, 30, 220)      # grønn = oversladding fjernet
BEHOLDT           = (160, 160, 160, 100)    # grå = beholdt boks
FASIT             = (30, 80, 220, 140)      # blå = fasit-boks

_FONT = None
_FONT_LITEN = None


def _font():
    global _FONT
    if _FONT is None:
        _FONT = ImageFont.load_default(size=22)
    return _FONT


def _font_liten():
    global _FONT_LITEN
    if _FONT_LITEN is None:
        _FONT_LITEN = ImageFont.load_default(size=16)
    return _FONT_LITEN


# ── Hjelpefunksjoner ──────────────────────────────────────────

def _dok_nr(navn):
    m = re.match(r"0*(\d+)", os.path.basename(navn))
    return int(m.group(1)) if m else None


def _overlap(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return (ix1 - ix0) * (iy1 - iy0) if (ix1 > ix0 and iy1 > iy0) else 0.0


def _areal(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


# ── Datainnlesing ─────────────────────────────────────────────

def les_fasit(sti):
    """Leser fasit-labels (ACCEPTED + manuell, ekskluderer REJECTED)."""
    fasit = defaultdict(list)
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("ml_status") or "").strip().upper() == "REJECTED":
                continue
            try:
                nr = int(r["fil_revisjon_id"])
                side = int(r["sidetall"])
                x, y = float(r["x"]), float(r["y"])
                w, h = float(r["width"]), float(r["height"])
            except (TypeError, ValueError, KeyError):
                continue
            x0, x1 = sorted((x, x + w))
            y0, y1 = sorted((y, y + h))
            fasit[(nr, side)].append((x0, y0, x1, y1))
    return fasit


def les_prediksjoner(sti):
    """Leser resultat-CSV med pikselkoordinater.
    Lagrer både pikselkoordinater (for tegning) og pt-dimensjoner (for filtrering)."""
    pred = []
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                navn = r["navn"]
                side = int(r["side"])
                bw, bh = int(r["bilde_bredde"]), int(r["bilde_hoyde"])
                x0, y0 = float(r["x0"]), float(r["y0"])
                x1, y1 = float(r["x1"]), float(r["y1"])
                kilde = r.get("kilde", "ukjent")
                conf_s = r.get("conf", "")
                conf = float(conf_s) if conf_s else None
            except (TypeError, ValueError, KeyError):
                continue
            norm = (x0 / bw, y0 / bh, x1 / bw, y1 / bh)
            w_pt = abs(x1 - x0) / SKALA
            h_pt = abs(y1 - y0) / SKALA
            if w_pt <= 0 or h_pt <= 0:
                continue
            ratio = w_pt / h_pt
            pred.append({
                "navn": navn, "side": side, "dok_nr": _dok_nr(navn),
                "bw": bw, "bh": bh,
                "px": (x0, y0, x1, y1),
                "norm": norm,
                "w": w_pt, "h": h_pt,
                "ratio": ratio,
                "elongation": max(ratio, 1 / ratio) if ratio > 0 else 0,
                "areal": w_pt * h_pt,
                "kilde": kilde, "conf": conf,
            })
    return pred


# ── Matching ──────────────────────────────────────────────────

def match_prediksjoner(pred_liste, fasit, terskel=0.15):
    """Matcher prediksjoner mot fasit via normalisert overlapp."""
    side_str = {}
    for p in pred_liste:
        key = (p["dok_nr"], p["side"])
        if key not in side_str:
            side_str[key] = (p["bw"] / SKALA, p["bh"] / SKALA)

    fasit_norm = {}
    for (nr, si), bokser in fasit.items():
        pw, ph = side_str.get((nr, si), (595, 842))
        fasit_norm[(nr, si)] = [(x0 / pw, y0 / ph, x1 / pw, y1 / ph)
                                for (x0, y0, x1, y1) in bokser]

    n_riktig = n_oversladd = n_uten = 0
    for p in pred_liste:
        key = (p["dok_nr"], p["side"])
        fbokser = fasit_norm.get(key, [])
        if not fbokser:
            p["riktig"] = None
            n_uten += 1
            continue
        pn = p["norm"]
        best_dek = 0.0
        for fb in fbokser:
            ov = _overlap(pn, fb)
            fa = _areal(fb)
            dek = ov / fa if fa > 0 else 0.0
            if dek > best_dek:
                best_dek = dek
        p["riktig"] = best_dek >= terskel
        if p["riktig"]:
            n_riktig += 1
        else:
            n_oversladd += 1

    for p in pred_liste:
        if p["riktig"] is None:
            p["riktig"] = False
            n_oversladd += 1

    return n_riktig, n_oversladd, n_uten


# ── Filtrering ────────────────────────────────────────────────

def _filter_grunner(p, min_elongation=None, maks_hoyde=None,
                    maks_bredde=None, maks_areal=None, conf_terskel=None):
    """Returnerer liste av grunner til at boksen filtreres (tom = beholdt)."""
    # Høy confidence → stol på prediksjonen, ikke filtrer
    if conf_terskel is not None and p.get("conf") is not None and p["conf"] >= conf_terskel:
        return []
    grunner = []
    if min_elongation is not None and p["elongation"] < min_elongation:
        grunner.append(f"elong {p['elongation']:.1f} < {min_elongation:g}")
    if maks_hoyde is not None and p["h"] > maks_hoyde:
        grunner.append(f"høyde {p['h']:.0f} > {maks_hoyde:g}")
    if maks_bredde is not None and p["w"] > maks_bredde:
        grunner.append(f"bredde {p['w']:.0f} > {maks_bredde:g}")
    if maks_areal is not None and p["areal"] > maks_areal:
        grunner.append(f"areal {p['areal']:.0f} > {maks_areal:g}")
    return grunner


def _er_filtrert(p, **kwargs):
    return len(_filter_grunner(p, **kwargs)) > 0


def _parse_per_kilde(spec_list):
    """Parser per-kilde filterstrenger.

    Format: "kilde:e=V,h=V,b=V,a=V,c=V"
    Eksempel: "yolo:e=2.5,h=40,b=80,c=0.7"

    Returnerer dict: kilde -> {min_elongation, maks_hoyde, maks_bredde, maks_areal, conf_terskel}
    """
    resultat = {}
    param_map = {
        "e": "min_elongation",
        "h": "maks_hoyde",
        "b": "maks_bredde",
        "a": "maks_areal",
        "c": "conf_terskel",
    }
    for spec in spec_list:
        if ":" not in spec:
            raise ValueError(f"Ugyldig per-kilde-format: {spec!r} "
                             f"(forventet 'kilde:e=V,h=V,...')")
        kilde, param_str = spec.split(":", 1)
        kilde = kilde.strip().lower()
        kwargs = {}
        for del_ in param_str.split(","):
            del_ = del_.strip()
            if not del_:
                continue
            if "=" not in del_:
                raise ValueError(f"Ugyldig parameter: {del_!r} i {spec!r}")
            nøkkel, verdi = del_.split("=", 1)
            nøkkel = nøkkel.strip().lower()
            if nøkkel not in param_map:
                raise ValueError(f"Ukjent parameter {nøkkel!r} i {spec!r}. "
                                 f"Gyldige: {', '.join(param_map.keys())}")
            kwargs[param_map[nøkkel]] = float(verdi)
        resultat[kilde] = kwargs
    return resultat


def _filter_grunner_per_kilde(p, per_kilde_kwargs):
    """Filtrer en prediksjon basert på kilde-spesifikke parametre."""
    kilde = p.get("kilde", "ukjent").lower()
    kwargs = per_kilde_kwargs.get(kilde)
    if kwargs is None:
        # Ingen filter for denne kilden — behold
        return []
    return _filter_grunner(p, **kwargs)


def _filter_etikett_per_kilde(per_kilde_kwargs):
    """Lag lesbar etikett for per-kilde-konfigurasjon."""
    deler = []
    for kilde, kwargs in sorted(per_kilde_kwargs.items()):
        param_deler = []
        if kwargs.get("min_elongation") is not None:
            param_deler.append(f"e≥{kwargs['min_elongation']:g}")
        if kwargs.get("maks_hoyde") is not None:
            param_deler.append(f"h≤{kwargs['maks_hoyde']:g}")
        if kwargs.get("maks_bredde") is not None:
            param_deler.append(f"b≤{kwargs['maks_bredde']:g}")
        if kwargs.get("maks_areal") is not None:
            param_deler.append(f"a≤{kwargs['maks_areal']:g}")
        if kwargs.get("conf_terskel") is not None:
            param_deler.append(f"c≥{kwargs['conf_terskel']:g}→behold")
        deler.append(f"{kilde}({', '.join(param_deler)})")
    return " + ".join(deler)


def _filter_mappenavn_per_kilde(per_kilde_kwargs):
    """Lag filnavn-vennlig mappenavn for per-kilde-konfigurasjon."""
    deler = []
    for kilde, kwargs in sorted(per_kilde_kwargs.items()):
        param_deler = []
        if kwargs.get("min_elongation") is not None:
            param_deler.append(f"e{kwargs['min_elongation']:g}")
        if kwargs.get("maks_hoyde") is not None:
            param_deler.append(f"h{kwargs['maks_hoyde']:g}")
        if kwargs.get("maks_bredde") is not None:
            param_deler.append(f"b{kwargs['maks_bredde']:g}")
        if kwargs.get("maks_areal") is not None:
            param_deler.append(f"a{kwargs['maks_areal']:g}")
        if kwargs.get("conf_terskel") is not None:
            param_deler.append(f"c{kwargs['conf_terskel']:g}")
        deler.append(f"{kilde}_{'_'.join(param_deler)}")
    return "__".join(deler)


# ── Rendering ─────────────────────────────────────────────────

def _render_side(pdf_dok, si):
    pix = pdf_dok[si - 1].get_pixmap(dpi=PDF_DPI)
    modus = "RGBA" if pix.n == 4 else "RGB"
    return Image.frombytes(modus, (pix.w, pix.h), pix.samples).convert("RGB")


def _tegn_tekst(tegner, r, tekst, farge, over=True):
    """Tegner tekst over (over=True) eller under boksen."""
    if over:
        y = max(r[1] - 24, 2)
    else:
        y = r[3] + 4
    tegner.text((r[0] + 2, y), tekst, fill=farge, font=_font_liten())


def _filter_etikett(kwargs):
    """Lag en lesbar etikett for en filterkonfigurasjon."""
    deler = []
    if kwargs.get("min_elongation") is not None:
        deler.append(f"elong≥{kwargs['min_elongation']:g}")
    if kwargs.get("maks_hoyde") is not None:
        deler.append(f"h≤{kwargs['maks_hoyde']:g}")
    if kwargs.get("maks_bredde") is not None:
        deler.append(f"b≤{kwargs['maks_bredde']:g}")
    if kwargs.get("maks_areal") is not None:
        deler.append(f"a≤{kwargs['maks_areal']:g}")
    return " + ".join(deler) if deler else "ingen filter"


def _filter_mappenavn(kwargs):
    """Lag et filnavn-vennlig mappenavn fra filterkonfigurasjonen."""
    deler = []
    if kwargs.get("min_elongation") is not None:
        deler.append(f"elong_{kwargs['min_elongation']:g}")
    if kwargs.get("maks_hoyde") is not None:
        deler.append(f"hoyde_{kwargs['maks_hoyde']:g}")
    if kwargs.get("maks_bredde") is not None:
        deler.append(f"bredde_{kwargs['maks_bredde']:g}")
    if kwargs.get("maks_areal") is not None:
        deler.append(f"areal_{kwargs['maks_areal']:g}")
    return "_".join(deler) if deler else "ingen_filter"


# ── Hovedlogikk ───────────────────────────────────────────────

def generer_bilder(pred, fasit, mappe, ut_mappe, filter_kwargs=None,
                   per_kilde_kwargs=None, kun_riktige=False, velg=None):
    """Generer gjennomgangsbilder for bokser fjernet av angitt filter.

    Args:
        pred: Liste av prediksjon-dicts (med 'riktig'-felt fra matching).
        fasit: Dict (dok_nr, side) -> [(x0,y0,x1,y1), ...] i punkt.
        mappe: Sti til PDF-dokumentene.
        ut_mappe: Sti for PNG-output.
        filter_kwargs: Dict med filterparametre (min_elongation, maks_hoyde, ...).
        per_kilde_kwargs: Dict kilde -> {filterparametre} for uavhengige filtre per kilde.
        kun_riktige: Vis kun sider der riktige bokser fjernes.
        velg: Set med filnavn å begrense til (eller None for alle).
    """

    # Finn fjernede bokser med grunner
    if per_kilde_kwargs:
        etikett = _filter_etikett_per_kilde(per_kilde_kwargs)
        for p in pred:
            p["_grunner"] = _filter_grunner_per_kilde(p, per_kilde_kwargs)
    else:
        etikett = _filter_etikett(filter_kwargs or {})
        for p in pred:
            p["_grunner"] = _filter_grunner(p, **(filter_kwargs or {}))

    fjernet = [p for p in pred if p["_grunner"]]
    beholdt = [p for p in pred if not p["_grunner"]]

    rik_fj = [p for p in fjernet if p["riktig"]]
    ov_fj = [p for p in fjernet if not p["riktig"]]

    print(f"\nFilter: {etikett}")
    print(f"  Fjernet totalt:      {len(fjernet)}")
    print(f"    Riktige fjernet:   {len(rik_fj)} (recall-tap)")
    print(f"    Oversladd fjernet: {len(ov_fj)} (presisjonsgevinst)")
    print(f"  Beholdt:             {len(beholdt)}")

    if not fjernet:
        print("  Ingen bokser fjernet — ingenting å tegne.")
        return

    # Finn sider som trenger tegning
    sider_med_fjerning = set()
    for p in fjernet:
        if kun_riktige and not p["riktig"]:
            continue
        sider_med_fjerning.add((p["navn"], p["side"]))

    if not sider_med_fjerning:
        print("  Ingen sider å tegne (ingen riktige fjernet).")
        return

    # Filtrer på --velg
    if velg:
        velg_set = {os.path.basename(v) for v in velg}
        sider_med_fjerning = {
            (n, s) for (n, s) in sider_med_fjerning
            if os.path.basename(n) in velg_set
        }
        if not sider_med_fjerning:
            print("  Ingen sider matchet --velg.")
            return

    # Grupper etter fil
    per_fil = defaultdict(set)
    for (navn, si) in sider_med_fjerning:
        per_fil[navn].add(si)

    # Sidestørrelser for fasit-normalisering
    side_str = {}
    for p in pred:
        key = (p["dok_nr"], p["side"])
        if key not in side_str:
            side_str[key] = (p["bw"] / SKALA, p["bh"] / SKALA)

    # Opprett undermapper
    riktige_mappe = os.path.join(ut_mappe, "riktige_fjernet")
    oversladd_mappe = os.path.join(ut_mappe, "oversladd_fjernet")
    os.makedirs(riktige_mappe, exist_ok=True)
    os.makedirs(oversladd_mappe, exist_ok=True)

    n_tegnet = 0
    n_riktige_sider = 0
    n_oversladd_sider = 0

    for navn in sorted(per_fil):
        sti = os.path.join(mappe, navn)
        if not os.path.isfile(sti):
            print(f"  ⚠ Finner ikke {sti}, hopper over")
            continue
        try:
            dok = fitz.open(sti)
        except Exception as e:
            print(f"  ⚠ Kunne ikke åpne {navn}: {e!r}")
            continue

        for si in sorted(per_fil[navn]):
            if not 1 <= si <= len(dok):
                continue

            bilde = _render_side(dok, si)
            base = bilde.convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            tegner = ImageDraw.Draw(overlay)

            # Prediksjoner for denne siden
            side_pred = [p for p in pred if p["navn"] == navn and p["side"] == si]
            if not side_pred:
                continue
            bw, bh = side_pred[0]["bw"], side_pred[0]["bh"]
            sx, sy = bilde.width / bw, bilde.height / bh

            # 1) Fasit-bokser (blå, tynn)
            nr = _dok_nr(navn)
            for (fx0, fy0, fx1, fy1) in fasit.get((nr, si), []):
                r = [fx0 * SKALA, fy0 * SKALA, fx1 * SKALA, fy1 * SKALA]
                tegner.rectangle(r, outline=FASIT, width=2)

            # 2) Beholdte bokser (grå, tynn)
            for p in side_pred:
                if not p["_grunner"]:
                    px = p["px"]
                    r = [px[0] * sx, px[1] * sy, px[2] * sx, px[3] * sy]
                    tegner.rectangle(r, outline=BEHOLDT, width=1)

            # 3) Fjernede bokser (tykk, fargekodet)
            har_riktig_fjernet = False
            har_oversladd_fjernet = False
            for p in side_pred:
                if not p["_grunner"]:
                    continue
                px = p["px"]
                r = [px[0] * sx, px[1] * sy, px[2] * sx, px[3] * sy]
                grunn_tekst = "; ".join(p["_grunner"])

                if p["riktig"]:
                    farge = RIKTIG_FJERNET
                    kategori = "RIKTIG"
                    har_riktig_fjernet = True
                else:
                    farge = OVERSLADD_FJERNET
                    kategori = "OVERSLADD"
                    har_oversladd_fjernet = True

                tegner.rectangle(r, outline=farge, width=4)
                # Etikett over boksen: kategori + kilde
                _tegn_tekst(tegner, r, f"{kategori} [{p['kilde']}]", farge, over=True)
                # Grunn under boksen
                _tegn_tekst(tegner, r, grunn_tekst, farge, over=False)
                # Conf inni boksen
                if p["conf"] is not None:
                    tegner.text(
                        (r[0] + 2, max(r[1] + 2, 2)),
                        f"conf={p['conf']:.2f}", fill=farge, font=_font_liten())

            bilde = Image.alpha_composite(base, overlay).convert("RGB")
            filnavn = f"{os.path.splitext(navn)[0]}_side{si}.png"

            # Lagre i riktig undermappe(r)
            if har_riktig_fjernet:
                bilde.save(os.path.join(riktige_mappe, filnavn))
                n_riktige_sider += 1
            if har_oversladd_fjernet:
                bilde.save(os.path.join(oversladd_mappe, filnavn))
                n_oversladd_sider += 1

            n_tegnet += 1

        dok.close()

    # Rydd tomme mapper
    for m in (riktige_mappe, oversladd_mappe):
        if os.path.isdir(m) and not os.listdir(m):
            os.rmdir(m)

    print(f"  Tegnet {n_tegnet} sider til {ut_mappe}")
    if n_riktige_sider:
        print(f"    {n_riktige_sider} side(r) i riktige_fjernet/")
    if n_oversladd_sider:
        print(f"    {n_oversladd_sider} side(r) i oversladd_fjernet/")


# ── Sweep-konfigurasjoner ────────────────────────────────────

SWEEP_CONFIGS = [
    {"min_elongation": 1.5},
    {"min_elongation": 2.0},
    {"maks_hoyde": 40},
    {"maks_hoyde": 50},
    {"maks_hoyde": 60},
    {"maks_bredde": 80},
    {"maks_bredde": 100},
    {"maks_bredde": 120},
    {"min_elongation": 1.5, "maks_hoyde": 50, "maks_bredde": 100},
    {"min_elongation": 1.5, "maks_hoyde": 50, "maks_bredde": 120},
    {"min_elongation": 1.5, "maks_hoyde": 60, "maks_bredde": 100},
    {"min_elongation": 1.5, "maks_hoyde": 60, "maks_bredde": 120},
    {"min_elongation": 2.0, "maks_hoyde": 50, "maks_bredde": 100},
]


def main():
    p = argparse.ArgumentParser(
        description="Visuell gjennomgang av bokser som fjernes av filterkonfigurasjoner. "
                    "Tegner fjernede bokser på original-PDF-ene, fargekodet etter om de "
                    "er riktige (røde) eller oversladdinger (grønne).")
    p.add_argument("--fasit-csv", required=True,
                   help="Labels-CSV (fasit)")
    p.add_argument("--res-csv", required=True,
                   help="Resultat-CSV fra modellen (pikselkoordinater)")
    p.add_argument("--mappe", required=True,
                   help="Mappe med PDF-dokumentene")
    p.add_argument("--ut-mappe", default="filter_review",
                   help="Mappe for PNG-output (default: filter_review)")
    p.add_argument("--terskel", type=float, default=0.15,
                   help="Overlapp-terskel for matching (default: 0.15)")

    filt = p.add_argument_group("Filterparametre (oppgi minst ett, eller bruk --sweep)")
    filt.add_argument("--elongation", type=float, default=None,
                      help="MIN_ELONGATION: min max(w/h, h/w)")
    filt.add_argument("--maks-hoyde", type=float, default=None,
                      help="Maks bokshøyde i punkt")
    filt.add_argument("--maks-bredde", type=float, default=None,
                      help="Maks boksbredde i punkt")
    filt.add_argument("--maks-areal", type=float, default=None,
                      help="Maks boksareal i pt²")

    p.add_argument("--sweep", action="store_true",
                   help="Generer bilder for et sett forhåndsdefinerte konfigurasjoner")
    p.add_argument("--per-kilde", nargs="+", metavar="SPEC",
                   help="Uavhengige filtre per kilde. Format: "
                        "\"kilde:e=V,h=V,b=V,a=V\" "
                        "(e=elongation, h=maks_hoyde, b=maks_bredde, a=maks_areal). "
                        "Eksempel: --per-kilde \"yolo:e=2.5,h=40\" \"paddle:e=1.5,h=60,b=120\"")
    p.add_argument("--kun-riktige", action="store_true",
                   help="Vis kun sider der riktige bokser fjernes (fokus på recall-tap)")
    p.add_argument("--velg", nargs="+", metavar="PDF",
                   help="Begrens til disse PDF-filene")

    args = p.parse_args()

    # ── Last data ─────────────────────────────────────────────
    fasit = les_fasit(args.fasit_csv)
    pred = les_prediksjoner(args.res_csv)

    n_fasit = sum(len(v) for v in fasit.values())
    print(f"Fasit:        {n_fasit} bokser i {len(fasit)} (dok, side)-grupper")
    print(f"Prediksjoner: {len(pred)} bokser")

    pred_kilder = {}
    for b in pred:
        pred_kilder[b["kilde"]] = pred_kilder.get(b["kilde"], 0) + 1
    for k, n in sorted(pred_kilder.items()):
        print(f"  {k}: {n}")

    # ── Match ─────────────────────────────────────────────────
    print(f"\nMatcher med overlapp-terskel {args.terskel:.0%} ...")
    n_riktig, n_oversladd, n_uten = match_prediksjoner(pred, fasit, args.terskel)

    totalt = n_riktig + n_oversladd
    pres = n_riktig / totalt * 100 if totalt else 0
    print(f"  Riktige: {n_riktig}, Oversladdinger: {n_oversladd} "
          f"(herav {n_uten} på sider uten fasit), Presisjon: {pres:.1f}%")

    # ── Generer bilder ────────────────────────────────────────
    if args.per_kilde:
        per_kilde_kwargs = _parse_per_kilde(args.per_kilde)
        mappenavn = _filter_mappenavn_per_kilde(per_kilde_kwargs)
        ut = os.path.join(args.ut_mappe, mappenavn)
        generer_bilder(pred, fasit, args.mappe, ut,
                       per_kilde_kwargs=per_kilde_kwargs,
                       kun_riktige=args.kun_riktige, velg=args.velg)
    elif args.sweep:
        for kwargs in SWEEP_CONFIGS:
            mappenavn = _filter_mappenavn(kwargs)
            ut = os.path.join(args.ut_mappe, mappenavn)
            generer_bilder(pred, fasit, args.mappe, ut, filter_kwargs=kwargs,
                           kun_riktige=args.kun_riktige, velg=args.velg)
    else:
        kwargs = {}
        if args.elongation is not None:
            kwargs["min_elongation"] = args.elongation
        if args.maks_hoyde is not None:
            kwargs["maks_hoyde"] = args.maks_hoyde
        if args.maks_bredde is not None:
            kwargs["maks_bredde"] = args.maks_bredde
        if args.maks_areal is not None:
            kwargs["maks_areal"] = args.maks_areal

        if not kwargs:
            p.error("Oppgi minst ett filter (--elongation, --maks-hoyde, "
                    "--maks-bredde, --maks-areal), --per-kilde, eller --sweep")

        generer_bilder(pred, fasit, args.mappe, args.ut_mappe, filter_kwargs=kwargs,
                       kun_riktige=args.kun_riktige, velg=args.velg)

    print("\nFerdig!")


if __name__ == "__main__":
    main()

