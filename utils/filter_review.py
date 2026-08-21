"""
Visuell gjennomgang av bokser som fjernes av en filterkonfigurasjon.

Bruker samme fasit-sentriske måling som filter_sweep: en fjernet prediksjon som
dekker en fasit-boks er bare et reelt tap hvis ingen annen prediksjon dekker
samme boks. Sidene sorteres og mappelegges etter det skillet.

Farger:
  MAGENTA  fasit-boks som mistet ALL dekning        ← reelt recall-tap
  RØD      fjernet prediksjon som var eneste dekker av en tapt boks
  ORANSJE  fjernet prediksjon som dekket fasit, men boksen er fortsatt dekket
           (redundant — gratis gevinst)
  GRØNN    fjernet oversladding (BOM)               ← gevinst
  GRÅ      beholdt boks (kontekst)
  BLÅ      fasit-boks som fortsatt er dekket

Mapper under --ut-mappe:
  tapt/                sider der minst én fasit-boks mistet all dekning
  redundant_fjernet/   sider der dekkende bokser ble fjernet uten tap
  oversladd_fjernet/   sider der oversladdinger ble fjernet

Kjør:
    python utils/filter_review.py \\
        --fasit-csv labels.csv --res-csv resultat.csv --mappe /sti/til/pdfer \\
        --per-kilde "paddle:e=1.8,h=80,b=150" "yolo:e=1.5,h=40,b=150,c=0.5"
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz
from PIL import Image, ImageDraw, ImageFont

from filter_felles import (KRITERIER, KRITERIUM_FELT, KRITERIUM_LAV_ER_BRA,
                           PDF_DPI, SKALA, STD_KRITERIUM, STD_SLURV_FAKTOR,
                           STD_TERSKEL,
                           bygg_datasett, dok_nr, evaluer, filter_grunner,
                           match_metrikker, overlapp,
                           lag_filter, lag_filter_per_kilde, les_fasit, les_kjorte_dok,
                           les_fasit_rader, les_prediksjoner, parse_per_kilde,
                           skriv_oppsummering, OCR_PARAMETRE)

# ── Farger ────────────────────────────────────────────────────
TAPT_FASIT         = (230, 0, 200, 255)     # magenta = fasit-boks uten dekning
KRITISK_FJERNET    = (220, 30, 30, 230)     # rød     = eneste dekker, fjernet
REDUNDANT_FJERNET  = (235, 150, 20, 230)    # oransje = dekkende, men erstattet
OVERSLADD_FJERNET  = (30, 180, 30, 220)     # grønn   = oversladding fjernet
BOM                = (255, 90, 0, 235)      # oransje = treffer ingen fasit-boks
BEHOLDT            = (160, 160, 160, 100)   # grå     = beholdt
FASIT              = (30, 80, 220, 140)     # blå     = fasit, fortsatt dekket

_FONT_LITEN = None


def _font_liten():
    global _FONT_LITEN
    if _FONT_LITEN is None:
        _FONT_LITEN = ImageFont.load_default(size=16)
    return _FONT_LITEN


def _tegn_tekst(tegner, r, tekst, farge, over=True):
    y = max(r[1] - 24, 2) if over else r[3] + 4
    tegner.text((r[0] + 2, y), tekst, fill=farge, font=_font_liten())


# ── Etiketter ─────────────────────────────────────────────────

def _etikett(kwargs):
    deler = []
    for nøkkel, mal in (("min_elongation", "e≥{:g}"), ("maks_elongation", "e≤{:g}"),
                        ("maks_hoyde", "h≤{:g}"), ("min_hoyde", "h≥{:g}"),
                        ("maks_bredde", "b≤{:g}"), ("min_bredde", "b≥{:g}"),
                        ("min_kortside", "kort≥{:g}"), ("maks_kortside", "kort≤{:g}"),
                        ("min_langside", "lang≥{:g}"), ("maks_langside", "lang≤{:g}"),
                        ("maks_areal", "a≤{:g}"), ("min_areal_px", "apx≥{:g}"),
                        ("conf_terskel", "c≥{:g}→behold"),
                        ("min_siffer", "siffer≥{:g}"),
                        ("maks_bokstaver", "bokst≤{:g}"),
                        ("min_siffer_run", "løp≥{:g}"),
                        ("krev_fnr_kandidat", "fnr-kandidat"),
                        ("avvis_desimal", "ikke-desimal"),
                        ("rec_veto", "rec≥{:g}→gjelder"),
                        ("ocr_conf_fritak", "c≥{:g}→OCR-fritak"),
                        ("avvis_00_run", "ikke-00-løp"),
                        ("avvis_orgnr", "ikke-orgnr"),
                        ("avvis_org_ord", "ikke-org-ord({:g})"),
                        ("linje_veto", "linjerec≥{:g}→gjelder"),
                        ("avvis_run_6_10", "ikke-run-6-10"),
                        ("uten_tekst_conf", "utenTekst-c≥{:g}"),
                        ("maks_luke", "luke<{:g}"),
                        ("avvis_desimal_luke", "ikke-desimal-luke")):
        if kwargs.get(nøkkel) is not None:
            deler.append(mal.format(kwargs[nøkkel]))
    return ", ".join(deler) if deler else "ingen filter"


def _etikett_per_kilde(per_kilde):
    return " + ".join(f"{k}({_etikett(kw)})" for k, kw in sorted(per_kilde.items()))


def _mappenavn(kwargs):
    deler = []
    for nøkkel, kort in (("min_elongation", "e"), ("maks_elongation", "eM"),
                         ("maks_hoyde", "h"), ("min_hoyde", "hm"),
                         ("maks_bredde", "b"), ("min_bredde", "bm"),
                         ("min_kortside", "km"), ("maks_kortside", "kM"),
                         ("min_langside", "lm"), ("maks_langside", "lM"),
                         ("maks_areal", "a"), ("min_areal_px", "apx"),
                         ("conf_terskel", "c"),
                         ("min_siffer", "smin"), ("maks_bokstaver", "bmaks"),
                         ("min_siffer_run", "rmin"),
                         ("krev_fnr_kandidat", "fnr"),
                         ("avvis_desimal", "des"), ("rec_veto", "rveto"),
                         ("ocr_conf_fritak", "cfritak"),
                         ("avvis_00_run", "r00"), ("avvis_orgnr", "orgnr"),
                         ("avvis_org_ord", "orgord"),
                         ("linje_veto", "lveto"), ("avvis_run_6_10", "run610"),
                         ("uten_tekst_conf", "utconf"),
                         ("maks_luke", "luke"),
                         ("avvis_desimal_luke", "desluke")):
        if kwargs.get(nøkkel) is not None:
            deler.append(f"{kort}{kwargs[nøkkel]:g}")
    return "_".join(deler) if deler else "ingen_filter"


def _mappenavn_per_kilde(per_kilde):
    return "__".join(f"{k}_{_mappenavn(kw)}" for k, kw in sorted(per_kilde.items()))


# ── Rendering ─────────────────────────────────────────────────

def _render_side(dok, si):
    pix = dok[si - 1].get_pixmap(dpi=PDF_DPI)
    modus = "RGBA" if pix.n == 4 else "RGB"
    return Image.frombytes(modus, (pix.w, pix.h), pix.samples).convert("RGB")


def generer_bilder(ds, mappe, ut_mappe, filter_kwargs=None, per_kilde=None,
                   kun_tapt=False, velg=None, maks_sider=None,
                   utsnitt_margin=60.0):
    """Tegner fjernede bokser på original-PDF-ene, gruppert etter alvorlighet."""
    if per_kilde:
        etikett = _etikett_per_kilde(per_kilde)
        fjernes = lag_filter_per_kilde(per_kilde)
        grunner_for = lambda p: (filter_grunner(p, **per_kilde[p["kilde"].lower()])
                                 if p["kilde"].lower() in per_kilde else [])
    else:
        kw = filter_kwargs or {}
        etikett = _etikett(kw)
        fjernes = lag_filter(**kw)
        grunner_for = lambda p: filter_grunner(p, **kw)

    m = evaluer(ds, fjernes, samle_tapte=True)
    tapte = set(m.tapte_bokser or ())

    print(f"\nFilter: {etikett}")
    print(f"  Fasit-bokser tapt (mistet all dekning): {m.tapt}"
          f"   ({m.tapt_pst:.2f}% av {ds.dekket_foer} dekkede)")
    print(f"  Oversladdinger fjernet:                 {m.ov_fj}"
          f"   ({m.ov_pst:.1f}% av {ds.n_bom})")
    print(f"  Dekkende bokser fjernet uten tap:       {m.red_fj}   (gratis)")
    print(f"  Fjernet totalt:                         {m.n_fj}"
          f"   ({m.areal_fj:,.0f} pt²)".replace(",", " "))
    print(f"  Recall etter:  {m.recall_etter:.2f}%"
          f"   Presisjon etter: {m.pres_etter:.1f}%")

    if not m.n_fj:
        print("  Ingen bokser fjernet — ingenting å tegne.")
        return

    # Merk hver prediksjon med kategori
    for p in ds.pred:
        p["_grunner"] = grunner_for(p) if fjernes(p) else []
        if not p["_grunner"]:
            p["_kat"] = None
        elif not p["dekker"]:
            p["_kat"] = "oversladd"
        elif any(j in tapte for j in p["dekker"]):
            p["_kat"] = "kritisk"
        else:
            p["_kat"] = "redundant"

    # ── Manifest over hver tapt fasit-boks, for manuell gjennomgang ──
    fjernet_for = defaultdict(list)
    for p in ds.pred:
        if p["_kat"] in ("kritisk",):
            for j in p["dekker"]:
                if j in tapte:
                    fjernet_for[j].append(p)

    navn_per_dok = {}
    for p in ds.pred:
        navn_per_dok.setdefault(p["dok_nr"], p["navn"])

    manifest = []
    for j in sorted(tapte):
        fb = ds.fasit_bokser[j]
        fx0, fy0, fx1, fy1 = fb["boks"]
        for p in fjernet_for.get(j, [{}]):
            fa = fb["norm_areal"]
            dekning = (overlapp(p["norm"], fb["norm"]) / fa * 100
                       if p and fa > 0 else "")
            manifest.append({
                "dekning_pst": round(dekning, 1) if dekning != "" else "",
                "fil": navn_per_dok.get(fb["dok_nr"], f"{fb['dok_nr']}"),
                "side": fb["side"],
                "fasit_x0": round(fx0, 1), "fasit_y0": round(fy0, 1),
                "fasit_bredde_pt": round(fx1 - fx0, 1),
                "fasit_hoyde_pt": round(fy1 - fy0, 1),
                "dekkere_foer": ds.dekning_foer[j],
                "kilde": p.get("kilde", ""),
                "conf": p.get("conf") if p.get("conf") is not None else "",
                "pred_bredde_pt": round(p["w"], 1) if p else "",
                "pred_hoyde_pt": round(p["h"], 1) if p else "",
                "elongation": round(p["elongation"], 2) if p else "",
                "kortside_pt": round(p["kortside"], 1) if p else "",
                "langside_pt": round(p["langside"], 1) if p else "",
                "grunn": "; ".join(p.get("_grunner", ())),
                "_j": j,
                "vurdering": "",
            })
    # Grupper like årsaker sammen — gjør gjennomgangen raskere
    manifest.sort(key=lambda r: (r["grunn"], r["fil"], r["side"]))
    utsnitt_navn = {}
    for nr, rad in enumerate(manifest, 1):
        rad["nr"] = nr
        base = os.path.splitext(os.path.basename(rad["fil"]))[0]
        rad["utsnitt"] = f"{nr:04d}_{base}_side{rad['side']}.png"
        utsnitt_navn.setdefault(rad["_j"], []).append(rad["utsnitt"])

    if manifest:
        os.makedirs(ut_mappe, exist_ok=True)
        manifest_sti = os.path.join(ut_mappe, "tapt.csv")
        felt = ["nr", "fil", "side", "grunn", "dekning_pst", "kilde", "conf",
                "elongation", "kortside_pt", "langside_pt",
                "pred_bredde_pt", "pred_hoyde_pt", "fasit_bredde_pt",
                "fasit_hoyde_pt", "dekkere_foer", "fasit_x0", "fasit_y0",
                "utsnitt", "vurdering"]
        with open(manifest_sti, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=felt, extrasaction="ignore")
            w.writeheader()
            w.writerows(manifest)
        print(f"  Manifest over tapte bokser: {manifest_sti}")
        print(f"    {len(manifest)} rader for {len(tapte)} tapte bokser"
              + ("  (en boks kan ha flere fjernede dekkere)"
                 if len(manifest) != len(tapte) else ""))
        per_grunn = defaultdict(int)
        for rad in manifest:
            per_grunn[rad["grunn"]] += 1
        print("  Tap gruppert etter regel som utløste fjerningen:")
        for grunn, n in sorted(per_grunn.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>5}  {grunn}")

    # Alvorlighet per side: (antall tapte fasit-bokser, antall kritiske bokser)
    tapt_per_side = defaultdict(int)
    for j in tapte:
        fb = ds.fasit_bokser[j]
        tapt_per_side[(fb["dok_nr"], fb["side"])] += 1

    sider = defaultdict(lambda: {"kritisk": 0, "redundant": 0, "oversladd": 0})
    navn_for_dok = {}
    for p in ds.pred:
        navn_for_dok.setdefault(p["dok_nr"], p["navn"])
        if p["_kat"]:
            sider[(p["navn"], p["side"])][p["_kat"]] += 1

    # Sider der en fasit-boks gikk tapt, men ingen prediksjon på siden ble
    # merket kritisk, kan ikke forekomme — tapet skyldes alltid en fjerning.
    aktuelle = []
    for (navn, si), tell in sider.items():
        n_tapt = tapt_per_side.get((dok_nr(navn), si), 0)
        if kun_tapt and not n_tapt:
            continue
        aktuelle.append((n_tapt, tell["kritisk"], tell["oversladd"], navn, si))

    if velg:
        velg_sett = {os.path.basename(v) for v in velg}
        aktuelle = [a for a in aktuelle if os.path.basename(a[3]) in velg_sett]

    if not aktuelle:
        print("  Ingen sider å tegne.")
        return

    aktuelle.sort(key=lambda a: (-a[0], -a[1], -a[2], a[3], a[4]))  # verst først
    if maks_sider:
        aktuelle = aktuelle[:maks_sider]

    mapper = {k: os.path.join(ut_mappe, k) for k in
              ("tapt", "redundant_fjernet", "oversladd_fjernet")}
    for sti in mapper.values():
        os.makedirs(sti, exist_ok=True)

    # Prediksjoner gruppert per side, for rask oppslag
    per_side = defaultdict(list)
    for p in ds.pred:
        per_side[(p["navn"], p["side"])].append(p)
    fasit_indekser_per_side = defaultdict(list)
    for j, fb in enumerate(ds.fasit_bokser):
        fasit_indekser_per_side[(fb["dok_nr"], fb["side"])].append(j)
    fasit_per_side = fasit_indekser_per_side

    # Tegn i samme rekkefølge som rangeringen (verst først), så tapt-sidene
    # ligger klare tidlig i kjøringen — ikke alfabetisk etter filnavn.
    # Sidene grupperes fortsatt per fil, så hver PDF åpnes bare én gang.
    rang = {(navn, si): i for i, (_, _, _, navn, si) in enumerate(aktuelle)}
    per_fil = defaultdict(list)
    for (_, _, _, navn, si) in aktuelle:
        per_fil[navn].append(si)

    telling = defaultdict(int)
    n_tegnet = 0
    for navn in sorted(per_fil,
                       key=lambda n: min(rang[(n, s)] for s in per_fil[n])):
        sti = os.path.join(mappe, navn)
        if not os.path.isfile(sti):
            print(f"  ⚠ Finner ikke {sti}, hopper over")
            continue
        try:
            dok = fitz.open(sti)
        except Exception as e:
            print(f"  ⚠ Kunne ikke åpne {navn}: {e!r}")
            continue

        for si in sorted(per_fil[navn], key=lambda s: rang[(navn, s)]):
            if not 1 <= si <= len(dok):
                continue
            side_pred = per_side[(navn, si)]
            if not side_pred:
                continue

            bilde = _render_side(dok, si)
            base = bilde.convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            tegner = ImageDraw.Draw(overlay)
            sx = bilde.width / side_pred[0]["bw"]
            sy = bilde.height / side_pred[0]["bh"]

            # 1) Fasit — magenta og tykk hvis den mistet all dekning
            for j in fasit_per_side.get((dok_nr(navn), si), ()):
                fx0, fy0, fx1, fy1 = ds.fasit_bokser[j]["boks"]
                r = [fx0 * SKALA * sx, fy0 * SKALA * sy,
                     fx1 * SKALA * sx, fy1 * SKALA * sy]
                if j in tapte:
                    # Utvid litt, ellers skjules fasit-boksen av prediksjonen
                    ytre = [r[0] - 6, r[1] - 6, r[2] + 6, r[3] + 6]
                    tegner.rectangle(ytre, outline=TAPT_FASIT, width=5)
                    tegner.text((ytre[0] + 2, max(ytre[1] - 44, 2)),
                                "TAPT FASIT", fill=TAPT_FASIT,
                                font=_font_liten())
                else:
                    tegner.rectangle(r, outline=FASIT, width=2)

            # 2) Beholdte bokser
            for p in side_pred:
                if p["_kat"] is None:
                    px = p["px"]
                    tegner.rectangle([px[0] * sx, px[1] * sy,
                                      px[2] * sx, px[3] * sy],
                                     outline=BEHOLDT, width=1)

            # 3) Fjernede bokser
            farger = {"kritisk": KRITISK_FJERNET,
                      "redundant": REDUNDANT_FJERNET,
                      "oversladd": OVERSLADD_FJERNET}
            har = set()
            for p in side_pred:
                if p["_kat"] is None:
                    continue
                px = p["px"]
                r = [px[0] * sx, px[1] * sy, px[2] * sx, px[3] * sy]
                farge = farger[p["_kat"]]
                har.add(p["_kat"])
                tegner.rectangle(r, outline=farge, width=4)
                merke = p["_kat"].upper()
                if p["klasse"] == "SLURV":
                    merke += "/SLURV"
                _tegn_tekst(tegner, r, f"{merke} [{p['kilde']}]", farge, over=True)
                _tegn_tekst(tegner, r, "; ".join(p["_grunner"]), farge, over=False)
                if p["conf"] is not None:
                    tegner.text((r[0] + 2, max(r[1] + 2, 2)),
                                f"conf={p['conf']:.2f}", fill=farge,
                                font=_font_liten())

            bilde = Image.alpha_composite(base, overlay).convert("RGB")

            # Utsnitt rundt hver tapt fasit-boks på denne siden
            if utsnitt_margin:
                mrg = utsnitt_margin * SKALA
                for j in fasit_indekser_per_side.get((dok_nr(navn), si), ()):
                    if j not in tapte:
                        continue
                    fx0, fy0, fx1, fy1 = ds.fasit_bokser[j]["boks"]
                    boks = (max(0, int(fx0 * SKALA * sx - mrg)),
                            max(0, int(fy0 * SKALA * sy - mrg)),
                            min(bilde.width, int(fx1 * SKALA * sx + mrg)),
                            min(bilde.height, int(fy1 * SKALA * sy + mrg)))
                    if boks[2] <= boks[0] or boks[3] <= boks[1]:
                        continue
                    ut = os.path.join(mapper["tapt"], "utsnitt")
                    os.makedirs(ut, exist_ok=True)
                    for filnavn_u in utsnitt_navn.get(j, ()):
                        bilde.crop(boks).save(os.path.join(ut, filnavn_u))
                        telling["utsnitt"] += 1

            filnavn = f"{os.path.splitext(navn)[0]}_side{si}.png"
            if "kritisk" in har:
                bilde.save(os.path.join(mapper["tapt"], filnavn))
                telling["tapt"] += 1
            if "redundant" in har:
                bilde.save(os.path.join(mapper["redundant_fjernet"], filnavn))
                telling["redundant_fjernet"] += 1
            if "oversladd" in har:
                bilde.save(os.path.join(mapper["oversladd_fjernet"], filnavn))
                telling["oversladd_fjernet"] += 1
            n_tegnet += 1

        dok.close()

    for navn, sti in mapper.items():
        if os.path.isdir(sti) and not os.listdir(sti):
            os.rmdir(sti)

    print(f"  Tegnet {n_tegnet} sider til {ut_mappe}")
    for navn in ("tapt", "redundant_fjernet", "oversladd_fjernet"):
        if telling[navn]:
            print(f"    {telling[navn]:>5} side(r) i {navn}/")
    if telling["utsnitt"]:
        print(f"    {telling['utsnitt']:>5} utsnitt i tapt/utsnitt/ "
              f"— én per tapt boks, samme rekkefølge som tapt.csv")


# ── OCR-tekstlag for triage ───────────────────────────────────
# «Hva leste Paddle her?» kan ikke besvares fra originalsiden alene. Med
# --ocr-tekst falmes originalen og pipelinens CACHEDE tokens tegnes oppå i
# sine posisjoner — nøyaktig det OCR-en faktisk så, med rec-score som farge.
# Tokens ligger i det ROTERTE bildets pikselrom (siden pipelinen OCR-er den
# roterte siden); prediksjoner og fasit ligger i sidens uroterte rom og
# transformeres frem med inversen av orientering.boks_tilbake.

OCR_REC_HOY = (0, 115, 0)        # grønn = rec >= 0.98
OCR_REC_MID = (25, 45, 170)      # blå   = 0.90 <= rec < 0.98
OCR_REC_LAV = (200, 30, 30)      # rød   = rec < 0.90
OCR_REC_UKJ = (130, 130, 130)    # grå   = uten rec-score

_FONTER = {}
_OCR_LES_CACHE = None


def _font_str(px):
    px = max(9, min(int(px), 44))
    if px not in _FONTER:
        _FONTER[px] = ImageFont.load_default(size=px)
    return _FONTER[px]


def _rec_farge(rec):
    if rec is None:
        return OCR_REC_UKJ
    if rec >= 0.98:
        return OCR_REC_HOY
    if rec >= 0.90:
        return OCR_REC_MID
    return OCR_REC_LAV


def _rekt_frem(r, k, w0, h0):
    """Urotert pikselrekt → rotert rom (invers av orientering.boks_tilbake)."""
    if not k:
        return list(r)
    x0, y0, x1, y1 = r
    if k == 1:
        pts = [(y0, w0 - x0), (y1, w0 - x1)]
    elif k == 2:
        pts = [(w0 - x0, h0 - y0), (w0 - x1, h0 - y1)]
    else:
        pts = [(h0 - y0, x0), (h0 - y1, x1)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _ocr_les_cache(ocr_mappe, navn):
    """Les (rotasjoner, tokens_per_side) fra pipelinens OCR-cache."""
    global _OCR_LES_CACHE
    if _OCR_LES_CACHE is None:
        app = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "app"))
        if app not in sys.path:
            sys.path.insert(0, app)
        from ocr_cache import les_cache
        _OCR_LES_CACHE = les_cache
    return _OCR_LES_CACHE(ocr_mappe, navn)


def _tegn_tokens(tegner, tokens, srx, sry):
    """Tegner hvert token i sin boks, skalert til boksen, farget etter rec."""
    for t in tokens:
        if not t.tekst.strip():
            continue
        r = [t.x0 * srx, t.y0 * sry, t.x1 * srx, t.y1 * sry]
        farge = _rec_farge(t.rec_score)
        tegner.rectangle(r, outline=farge + (70,), width=1)
        str_h = (r[3] - r[1]) * 0.8
        str_b = (r[2] - r[0]) / (max(len(t.tekst), 1) * 0.55)
        tegner.text((r[0] + 1, r[1]), t.tekst, fill=farge + (255,),
                    font=_font_str(min(str_h, str_b)))


def triage_bom(ds, mappe, ut_mappe, velg=None, maks_sider=None, kilde=None,
               ocr_mappe=None, ocr_opacity=0.15):
    """Tegner ALLE BOM-prediksjoner, uavhengig av filter.

    En BOM-boks treffer ingen fasit-boks, men fasit er menneskeskapt: den kan
    like godt være et fødselsnummer saksbehandleren bommet på som en reell
    oversladding. Det skillet kan ikke leses ut av geometrien — det må ses.

    Sidene sorteres slik at 'begge'-bokser kommer først: der begge modellene
    er enige om at det står et fødselsnummer, er sannsynligheten for at fasit
    er feil størst.
    """
    bom = [p for p in ds.pred if p["klasse"] == "BOM"]
    if kilde:
        bom = [p for p in bom if p["kilde"].lower() == kilde.lower()]
    print(f"\nTRIAGE av BOM-bokser (treffer ingen fasit-boks)"
          + (f" — kun kilde «{kilde}»" if kilde else ""))
    print(f"  Totalt: {len(bom)}")
    per_kilde = defaultdict(int)
    for p in bom:
        per_kilde[p["kilde"]] += 1
    for k in sorted(per_kilde):
        print(f"    {k:>8}: {per_kilde[k]:>5}")
    if not bom:
        return

    prioritet = {"begge": 0, "yolo": 1, "paddle": 2}
    sider = defaultdict(list)
    for p in bom:
        sider[(p["navn"], p["side"])].append(p)

    if velg:
        velg_sett = {os.path.basename(v) for v in velg}
        sider = {k: v for k, v in sider.items()
                 if os.path.basename(k[0]) in velg_sett}

    rangert = sorted(
        sider.items(),
        key=lambda kv: (min(prioritet.get(p["kilde"], 9) for p in kv[1]),
                        -len(kv[1]), kv[0]))
    if maks_sider:
        rangert = rangert[:maks_sider]

    per_side = defaultdict(list)
    for p in ds.pred:
        per_side[(p["navn"], p["side"])].append(p)
    fasit_per_side = defaultdict(list)
    for fb in ds.fasit_bokser:
        fasit_per_side[(fb["dok_nr"], fb["side"])].append(fb)

    # Tegn i rangert rekkefølge (begge-sidene først), gruppert per fil så
    # hver PDF fortsatt bare åpnes én gang.
    rang = {nokkel: i for i, (nokkel, _) in enumerate(rangert)}
    per_fil = defaultdict(list)
    for (navn, si), _ in rangert:
        per_fil[navn].append(si)

    telling = defaultdict(int)
    n_tegnet = 0
    for navn in sorted(per_fil,
                       key=lambda n: min(rang[(n, s)] for s in per_fil[n])):
        sti = os.path.join(mappe, navn)
        if not os.path.isfile(sti):
            print(f"  ⚠ Finner ikke {sti}, hopper over")
            continue
        try:
            dok = fitz.open(sti)
        except Exception as e:
            print(f"  ⚠ Kunne ikke åpne {navn}: {e!r}")
            continue

        cache = _ocr_les_cache(ocr_mappe, navn) if ocr_mappe else None
        if ocr_mappe and cache is None:
            print(f"  ⚠ Ingen OCR-cache for {navn} — tegner uten tekstlag")

        for si in sorted(per_fil[navn], key=lambda s: rang[(navn, s)]):
            if not 1 <= si <= len(dok):
                continue
            side_pred = per_side[(navn, si)]
            bilde = _render_side(dok, si)
            w0, h0 = bilde.width, bilde.height
            k, tokens = 0, []
            if cache:
                rotasjoner, tokens_per_side = cache
                if si <= len(rotasjoner):
                    k = rotasjoner[si - 1] or 0
                    tokens = tokens_per_side[si - 1]
                if k:
                    # Samme rotasjon som pipelinen OCR-et med (np.rot90 = CCW)
                    bilde = bilde.rotate(90 * k, expand=True)
                bilde = Image.blend(
                    Image.new("RGB", bilde.size, (255, 255, 255)),
                    bilde, ocr_opacity)
            base = bilde.convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            tegner = ImageDraw.Draw(overlay)
            sx = w0 / side_pred[0]["bw"]
            sy = h0 / side_pred[0]["bh"]
            if cache and tokens:
                bw, bh = side_pred[0]["bw"], side_pred[0]["bh"]
                rot_w, rot_h = (bh, bw) if k % 2 else (bw, bh)
                _tegn_tokens(tegner, tokens,
                             base.width / rot_w, base.height / rot_h)

            for fb in fasit_per_side.get((dok_nr(navn), si), ()):
                fx0, fy0, fx1, fy1 = fb["boks"]
                r = _rekt_frem([fx0 * SKALA * sx, fy0 * SKALA * sy,
                                fx1 * SKALA * sx, fy1 * SKALA * sy], k, w0, h0)
                tegner.rectangle(r, outline=FASIT, width=2)

            kilder_her = set()
            for p in side_pred:
                px = p["px"]
                r = _rekt_frem([px[0] * sx, px[1] * sy, px[2] * sx, px[3] * sy],
                               k, w0, h0)
                if p["klasse"] == "BOM":
                    kilder_her.add(p["kilde"])
                    tegner.rectangle(r, outline=BOM, width=4)
                    merke = f"BOM [{p['kilde']}]"
                    if p["conf"] is not None:
                        merke += f" conf={p['conf']:.2f}"
                    _tegn_tekst(tegner, r, merke, BOM, over=True)
                    _tegn_tekst(tegner, r,
                                f"{p['w']:.0f}x{p['h']:.0f}pt e={p['elongation']:.1f}",
                                BOM, over=False)
                else:
                    tegner.rectangle(r, outline=BEHOLDT, width=1)

            bilde = Image.alpha_composite(base, overlay).convert("RGB")
            filnavn = f"{os.path.splitext(navn)[0]}_side{si}.png"
            for k in kilder_her:
                undermappe = os.path.join(ut_mappe, "bom", k)
                os.makedirs(undermappe, exist_ok=True)
                bilde.save(os.path.join(undermappe, filnavn))
                telling[k] += 1
            n_tegnet += 1

        dok.close()

    print(f"  Tegnet {n_tegnet} sider til {os.path.join(ut_mappe, 'bom')}")
    for k in sorted(telling):
        print(f"    {telling[k]:>5} side(r) i bom/{k}/")
    print("\n  Blå = fasit (saksbehandlerens sladding), oransje = BOM, grå = treff.")
    print("  Spørsmålet per oransje boks: står det et fødselsnummer der?")
    print("    ja  → saksbehandleren bommet, modellen har rett (ikke oversladding)")
    print("    nei → reell oversladding")
    if ocr_mappe:
        print("  Tekstlag: Paddles cachede tokens oppå falmet original —")
        print("    grønn = rec ≥ 0.98, blå = 0.90–0.98, rød = < 0.90, "
              "grå = uten score.")


BAND_FASIT = (30, 120, 255, 255)     # blå   = fasit (saksbehandlerens sladding)
BAND_PRED  = (230, 30, 30, 255)      # rød   = prediksjonen i båndet
BAND_ANNEN = (150, 150, 150, 140)    # grå   = andre bokser på siden


def band_review(ds, mappe, ut_mappe, kriterium, lo, hi, maks=None,
                utsnitt_margin=25.0, ut_csv=None):
    """Tegner UTSNITT av hvert (prediksjon, fasit)-par der målet ligger i [lo, hi).

    Dette er gråsonen terskelen faktisk avgjør. Under lo forkastes paret uansett
    valg, over hi godtas det uansett — det er kun båndet som skifter side når
    terskelen flyttes fra lo til hi. Derfor er det bare disse som må ses.

    Fasit er menneskeskapt og kan være feil: en fasit-boks kan være slurvete
    tegnet, dekke to felt, eller sitte på noe som ikke er et fødselsnummer.
    Utsnittet viser derfor selve dokumentinnholdet, ikke bare rammene, slik at
    spørsmålet «er fasit riktig her?» kan besvares.
    """
    felt = KRITERIUM_FELT[kriterium]
    lav_er_bra = kriterium in KRITERIUM_LAV_ER_BRA

    fasit_per_side = defaultdict(list)
    for j, fb in enumerate(ds.fasit_bokser):
        fasit_per_side[(fb["dok_nr"], fb["side"])].append((j, fb))
    pred_per_side = defaultdict(list)
    for p in ds.pred:
        pred_per_side[(p["dok_nr"], p["side"])].append(p)

    # Alle par med overlapp, uansett verdi — så fordelingen kan rapporteres
    par = []
    for nøkkel, fasit_her in fasit_per_side.items():
        for p in pred_per_side.get(nøkkel, ()):
            for j, fb in fasit_her:
                m = match_metrikker(p["norm"], fb["norm"], fb["horisontal"])
                if m is None:
                    continue
                par.append((m[felt], m, p, j, fb))

    i_band = [t for t in par if lo <= t[0] < hi]
    i_band.sort(key=lambda t: t[0], reverse=lav_er_bra)

    print(f"\nBÅNDGJENNOMGANG — kriterium «{kriterium}» ({felt}) i [{lo:.0%}, {hi:.0%})")
    print(f"  Overlappende par totalt:      {len(par)}")
    print(f"  Par i båndet:                 {len(i_band)}")
    print(f"  Berørte fasit-bokser:         {len({t[3] for t in i_band})}")
    print(f"  Berørte prediksjoner:         {len({id(t[2]) for t in i_band})}")
    if not i_band:
        print("  Ingen par i båndet — terskelvalget mellom disse to verdiene "
              "endrer ingenting.")
        return

    # Hvor mange fasit-bokser SKIFTER side? Bare de som ikke har en annen
    # prediksjon som klarer den høye terskelen uansett.
    klarer_hi = defaultdict(bool)
    for verdi, _m, _p, j, _fb in par:
        if (verdi < hi) if lav_er_bra else (verdi >= hi):
            klarer_hi[j] = True
    vipper = {t[3] for t in i_band if not klarer_hi[t[3]]}
    print(f"  Fasit-bokser som faktisk vipper: {len(vipper)}  "
          f"(resten er dekket av en annen prediksjon uansett terskel)")

    per_kilde = defaultdict(int)
    for t in i_band:
        per_kilde[t[2]["kilde"]] += 1
    print("  Per kilde: " + ", ".join(f"{k}={v}" for k, v in sorted(per_kilde.items())))

    valgte = i_band[:maks] if maks else i_band
    if maks and len(i_band) > maks:
        print(f"  ⚠ Tegner kun de {maks} første av {len(i_band)} "
              f"(--maks-sider styrer dette)")

    mappenavn = os.path.join(
        ut_mappe, f"band_{kriterium}_{lo:.2f}-{hi:.2f}".replace(".", ""))
    os.makedirs(mappenavn, exist_ok=True)

    if ut_csv:
        import csv as _csv
        with open(ut_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            # Kolonnen heter "verdi", ikke felt-navnet: for areal-kriteriet er
            # felt == "dek_f", og da ville overskriften stått to ganger.
            w.writerow(["utsnitt", "fil", "side", "fasit_idx", "vipper", "kilde",
                        "conf", "verdi", "dek_f", "dek_kort", "dek_lang", "iou",
                        "senter_kort", "pred_wpt", "pred_hpt"])
            for n, (verdi, m, p, j, fb) in enumerate(valgte, 1):
                w.writerow([f"{n:04d}", p["navn"], p["side"], j,
                            "ja" if j in vipper else "nei", p["kilde"],
                            f"{p['conf']:.3f}" if p["conf"] is not None else "",
                            f"{verdi:.4f}",
                            *[f"{m[k]:.4f}" for k in ("dek_f", "dek_kort",
                                                      "dek_lang", "iou",
                                                      "senter_kort")],
                            f"{p['w']:.1f}", f"{p['h']:.1f}"])
        print(f"  Måltall skrevet til {ut_csv}")

    # ── Tegn utsnittene, gruppert per fil for å åpne hver PDF én gang ──
    per_fil = defaultdict(list)
    for n, t in enumerate(valgte, 1):
        per_fil[t[2]["navn"]].append((n, t))

    n_tegnet = 0
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
        # Render hver side maks én gang, selv med flere par på samme side
        sider_cache = {}
        for n, (verdi, m, p, j, fb) in sorted(per_fil[navn]):
            si = p["side"]
            if not 1 <= si <= len(dok):
                continue
            if si not in sider_cache:
                sider_cache[si] = _render_side(dok, si)
            bilde = sider_cache[si]
            sx = bilde.width / p["bw"]
            sy = bilde.height / p["bh"]

            base = bilde.convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            tegner = ImageDraw.Draw(overlay)

            # Andre bokser på siden, som kontekst
            for annen in pred_per_side.get((p["dok_nr"], si), ()):
                if annen is p:
                    continue
                a = annen["px"]
                tegner.rectangle([a[0] * sx, a[1] * sy, a[2] * sx, a[3] * sy],
                                 outline=BAND_ANNEN, width=2)
            for jj, andre_fb in fasit_per_side.get((p["dok_nr"], si), ()):
                if jj == j:
                    continue
                b = andre_fb["boks"]
                tegner.rectangle([b[0] * SKALA * sx, b[1] * SKALA * sy,
                                  b[2] * SKALA * sx, b[3] * SKALA * sy],
                                 outline=BAND_ANNEN, width=2)

            fx = [fb["boks"][0] * SKALA * sx, fb["boks"][1] * SKALA * sy,
                  fb["boks"][2] * SKALA * sx, fb["boks"][3] * SKALA * sy]
            px = [p["px"][0] * sx, p["px"][1] * sy,
                  p["px"][2] * sx, p["px"][3] * sy]
            tegner.rectangle(fx, outline=BAND_FASIT, width=3)
            tegner.rectangle(px, outline=BAND_PRED, width=3)

            flat = Image.alpha_composite(base, overlay).convert("RGB")

            # Utsnitt rundt unionen av de to boksene
            marg = utsnitt_margin * SKALA
            u = [min(fx[0], px[0]) - marg * sx, min(fx[1], px[1]) - marg * sy,
                 max(fx[2], px[2]) + marg * sx, max(fx[3], px[3]) + marg * sy]
            u = [max(0, u[0]), max(0, u[1]),
                 min(flat.width, u[2]), min(flat.height, u[3])]
            utsnitt = flat.crop([int(v) for v in u])

            # Måltall brennes inn under utsnittet, så bildet står alene
            tekst = (f"{kriterium} {verdi:.3f}  |  dek_f={m['dek_f']:.2f} "
                     f"kort={m['dek_kort']:.2f} lang={m['dek_lang']:.2f} "
                     f"iou={m['iou']:.2f} senter={m['senter_kort']:.2f}")
            tekst2 = (f"blaa=fasit roed=pred   {navn} s{si}  {p['kilde']}"
                      + (f" conf={p['conf']:.2f}" if p["conf"] is not None else "")
                      + f"  pred {p['w']:.0f}x{p['h']:.0f}pt"
                      + f"  fasit {fb['boks'][2]-fb['boks'][0]:.0f}"
                      + f"x{fb['boks'][3]-fb['boks'][1]:.0f}pt"
                      + ("   VIPPER" if j in vipper else ""))
            # Bred nok at bildeteksten ikke klippes
            bunn = Image.new("RGB", (max(utsnitt.width, 780), utsnitt.height + 48),
                             (255, 255, 255))
            bunn.paste(utsnitt, (0, 0))
            d = ImageDraw.Draw(bunn)
            d.text((4, utsnitt.height + 4), tekst, fill=(0, 0, 0), font=_font_liten())
            d.text((4, utsnitt.height + 26), tekst2, fill=(90, 90, 90),
                   font=_font_liten())

            vipp = "vipper_" if j in vipper else ""
            filnavn = (f"{n:04d}_{verdi:.3f}_{vipp}"
                       f"{os.path.splitext(navn)[0]}_s{si}_f{j}.png")
            bunn.save(os.path.join(mappenavn, filnavn))
            n_tegnet += 1
        dok.close()

    print(f"\n  Tegnet {n_tegnet} utsnitt til {mappenavn}/")
    print(f"  Filnavnene starter med løpenummer og {felt}-verdien, så de "
          f"sorterer stigende.")
    print("  Blå = fasit, rød = prediksjonen i båndet, grå = andre bokser.")
    print("  Spørsmål per utsnitt:")
    print("    1. Er fasit-boksen riktig? (feiltegnet fasit skal ikke telle mot oss)")
    print("    2. Peker prediksjonen på SAMME felt som fasit — eller nabolinjen?")
    print("    3. Dekker prediksjonen nok av sifrene til å være en reell sladding?")
    print("  Er svaret ja/ja/ja bør paret godtas — da er terskelen satt for høyt.")


def test_mot_fasit(fasit_csv, mappe, ut_mappe, filter_kwargs, maks_sider=None,
                   utsnitt_margin=60.0, velg=None, ds=None):
    """Anvender filteret DIREKTE på saksbehandlernes sladdinger.

    Hver fasit-boks er en sladding et menneske faktisk gjorde, altså per
    definisjon riktig. Blir den forkastet av filteret, er det en form filteret
    ikke godtar — og det gjelder uansett hva modellen predikerte, så alle
    labels kan vurderes, ikke bare de dokumentene modellen kjørte på.
    """
    rader, ekskludert, kolonner = les_fasit_rader(fasit_csv)
    n_eks = sum(ekskludert.values())

    print(f"FASIT-TEST — filteret anvendt direkte på sladdingene")
    print(f"  Kolonner i labels-CSV: {', '.join(kolonner)}")
    print(f"  Labels lest:      {len(rader)}  (riktige sladdinger)")
    if ekskludert:
        print(f"  Ekskludert:       {n_eks}  "
              + ", ".join(f"{k}: {v}" for k, v in sorted(ekskludert.items())))

    for felt in ("ml_status", "type"):
        fordeling = defaultdict(int)
        for r in rader:
            fordeling[r[felt]] += 1
        if len(fordeling) > 1 or felt == "ml_status":
            print(f"  {felt}: "
                  + ", ".join(f"{k}={v}" for k, v in
                              sorted(fordeling.items(), key=lambda kv: -kv[1])))

    # Formfordelingen til ALLE riktige sladdinger — viser hvor mye margin
    # terskelen har mot virkelige data, ikke bare mot modellens bokser.
    def _pst(sortert, p):
        if not sortert:
            return 0.0
        i = (len(sortert) - 1) * p / 100.0
        lav, hoy = int(i), min(int(i) + 1, len(sortert) - 1)
        return sortert[lav] + (sortert[hoy] - sortert[lav]) * (i - lav)

    PST = (0.01, 0.1, 1, 50, 99, 99.9, 99.99)
    print("")
    print("  Form på alle riktige sladdinger "
          "(her ligger grensene dine, mål mot halene):")
    hode = f"    {'mål':<14}" + "".join(f" {('p' + format(p, 'g')):>9}" for p in PST)
    print(hode)
    print(f"    {'-' * (len(hode) - 4)}")
    for nokkel, navn, des in (("elongation", "elongation", 2),
                              ("kortside", "kortside (pt)", 1),
                              ("langside", "langside (pt)", 1),
                              ("areal_px", "areal (px²)", 0)):
        sortert = sorted(r[nokkel] for r in rader)
        print(f"    {navn:<14}"
              + "".join(f" {_pst(sortert, p):>9.{des}f}" for p in PST))

    etikett = _etikett(filter_kwargs)
    print(f"\n  Filter: {etikett}")
    print(f"  Merk: fasit-bokser har ingen conf, så conf-porten slår ikke inn "
          f"— testen viser hva geometrien alene forkaster.")

    forkastet = []
    for r in rader:
        grunner = filter_grunner(r, **filter_kwargs)
        if grunner:
            r["_grunner"] = grunner
            forkastet.append(r)

    andel = len(forkastet) / len(rader) * 100 if rader else 0
    print(f"\n  Sladdinger filteret ville forkastet: {len(forkastet)} "
          f"({andel:.3f}% av {len(rader)})")
    if not forkastet:
        print("  Ingen — filteret forkaster ingen av saksbehandlernes sladdinger.")
        return

    for felt, tittel in (("_grunn", "regel (en boks kan bryte flere)"),
                         ("ml_status", "ml_status"), ("type", "type")):
        fordeling, nevner = defaultdict(int), defaultdict(int)
        for r in forkastet:
            if felt == "_grunn":
                for g in r["_grunner"]:
                    fordeling[re.sub(r"[\d.]+", "N", g, count=1)] += 1
            else:
                fordeling[r[felt]] += 1
        if felt != "_grunn":
            for r in rader:
                nevner[r[felt]] += 1
        if len(fordeling) > 1 or felt == "_grunn":
            print(f"\n  Gruppert etter {tittel}:")
            for k, v in sorted(fordeling.items(), key=lambda kv: -kv[1]):
                tot = nevner.get(k)
                andel = f"  av {tot} = {v / tot * 100:.3f}%" if tot else ""
                print(f"    {v:>6}  {k}{andel}")

    # ── Krysstabell: forkastet form vs. faktisk mistet dekning ──
    # En fasit-boks med ulovlig form betyr ingenting hvis modellens EGEN boks
    # for samme felt har lovlig form og overlever filteret. Det er forskjellen
    # mellom «mennesket tegnet stygt» og «vi mister sladdingen».
    if ds is not None:
        m = evaluer(ds, lag_filter(**filter_kwargs), samle_tapte=True)
        tapte = set(m.tapte_bokser or ())
        i_scope = {}
        for j, fb in enumerate(ds.fasit_bokser):
            i_scope[(fb["dok_nr"], fb["side"],
                     round(fb["boks"][0], 1), round(fb["boks"][1], 1))] = j
        ute = form_og_tapt = form_men_dekket = 0
        for r in forkastet:
            j = i_scope.get((r["dok_nr"], r["side"],
                             round(r["boks"][0], 1), round(r["boks"][1], 1)))
            if j is None:
                r["_status"] = "utenfor_scope"
                ute += 1
            elif j in tapte:
                r["_status"] = "MISTET_DEKNING"
                form_og_tapt += 1
            else:
                r["_status"] = "fortsatt_dekket"
                form_men_dekket += 1
        i_scope_forkastet = form_og_tapt + form_men_dekket
        print("")
        print("  Kryssjekk mot modellens egne bokser "
              f"({len(ds.fasit_bokser)} labels på kjørte dokumenter):")
        print(f"    Ulovlig form OG mistet dekning:  {form_og_tapt:>5}"
              "   ← reell risiko")
        print(f"    Ulovlig form, men fortsatt dekket: {form_men_dekket:>3}"
              "   ← artefakt: modellens boks har lovlig form")
        if ute:
            print(f"    Utenfor scope (dokument ikke kjørt):{ute:>5}"
                  "   ← kan ikke vurderes")
        if i_scope_forkastet:
            print(f"    Andel artefakt av vurderbare: "
                  f"{form_men_dekket / i_scope_forkastet * 100:.0f}%")
        print(f"    Totalt mistet dekning under samme filter: {m.tapt}")

    # ── Manifest ──
    os.makedirs(ut_mappe, exist_ok=True)
    rang = {"MISTET_DEKNING": 0, "": 1, "utenfor_scope": 2, "fortsatt_dekket": 3}
    forkastet.sort(key=lambda r: (rang.get(r.get("_status", ""), 1),
                                  "; ".join(r["_grunner"]),
                                  r["dok_nr"], r["side"]))
    manifest = []
    for nr, r in enumerate(forkastet, 1):
        x0, y0, x1, y1 = r["boks"]
        manifest.append({
            "nr": nr, "fil_revisjon_id": r["dok_nr"], "side": r["side"],
            "grunn": "; ".join(r["_grunner"]),
            "ml_status": r["ml_status"], "type": r["type"],
            "elongation": round(r["elongation"], 2),
            "kortside_pt": round(r["kortside"], 1),
            "langside_pt": round(r["langside"], 1),
            "bredde_pt": round(r["w"], 1), "hoyde_pt": round(r["h"], 1),
            "areal_px": round(r["areal_px"]),
            "status": r.get("_status", ""),
            "x0": round(x0, 1), "y0": round(y0, 1),
            "utsnitt": f"{nr:04d}_{r['dok_nr']}_side{r['side']}.png",
            "vurdering": "",
        })
        r["_utsnitt"] = manifest[-1]["utsnitt"]
    manifest_sti = os.path.join(ut_mappe, "forkastede_sladdinger.csv")
    with open(manifest_sti, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    print(f"\n  Manifest: {manifest_sti}")

    # ── Utsnitt ──
    if not mappe:
        return
    if velg:
        velg_sett = {os.path.basename(v) for v in velg}
    per_dok = defaultdict(list)
    for r in (forkastet[:maks_sider] if maks_sider else forkastet):
        per_dok[r["dok_nr"]].append(r)

    # Finn PDF-en for hvert dokumentnummer
    filer = {}
    for navn in os.listdir(mappe):
        if navn.lower().endswith(".pdf"):
            n = dok_nr(navn)
            if n is not None:
                filer.setdefault(n, navn)

    ut = os.path.join(ut_mappe, "utsnitt")
    os.makedirs(ut, exist_ok=True)
    n_tegnet = mangler = 0
    for nr in sorted(per_dok):
        navn = filer.get(nr)
        if navn is None or (velg and os.path.basename(navn) not in velg_sett):
            mangler += len(per_dok[nr])
            continue
        try:
            dok = fitz.open(os.path.join(mappe, navn))
        except Exception as e:
            print(f"  ⚠ Kunne ikke åpne {navn}: {e!r}")
            mangler += len(per_dok[nr])
            continue
        for r in per_dok[nr]:
            si = r["side"]
            if not 1 <= si <= len(dok):
                mangler += 1
                continue
            bilde = _render_side(dok, si)
            overlay = Image.new("RGBA", bilde.size, (0, 0, 0, 0))
            tegner = ImageDraw.Draw(overlay)
            x0, y0, x1, y1 = r["boks"]
            rr = [x0 * SKALA, y0 * SKALA, x1 * SKALA, y1 * SKALA]
            tegner.rectangle(rr, outline=KRITISK_FJERNET, width=4)
            _tegn_tekst(tegner, rr, "FORKASTET AV FILTER", KRITISK_FJERNET, True)
            _tegn_tekst(tegner, rr, "; ".join(r["_grunner"]),
                        KRITISK_FJERNET, False)
            ferdig = Image.alpha_composite(bilde.convert("RGBA"),
                                           overlay).convert("RGB")
            mrg = utsnitt_margin * SKALA
            boks = (max(0, int(rr[0] - mrg)), max(0, int(rr[1] - mrg)),
                    min(ferdig.width, int(rr[2] + mrg)),
                    min(ferdig.height, int(rr[3] + mrg)))
            if boks[2] > boks[0] and boks[3] > boks[1]:
                ferdig.crop(boks).save(os.path.join(ut, r["_utsnitt"]))
                n_tegnet += 1
        dok.close()

    print(f"  Tegnet {n_tegnet} utsnitt til {ut}")
    if mangler:
        print(f"  {mangler} uten utsnitt (PDF mangler i {mappe})")


SWEEP_KONFIGER = [
    {"min_elongation": 1.5},
    {"min_elongation": 2.0},
    {"maks_hoyde": 40},
    {"maks_hoyde": 50},
    {"maks_bredde": 100},
    {"maks_bredde": 120},
    {"min_elongation": 1.5, "maks_hoyde": 50, "maks_bredde": 100},
    {"min_elongation": 1.5, "maks_hoyde": 50, "maks_bredde": 120},
    {"min_elongation": 1.5, "maks_hoyde": 60, "maks_bredde": 120},
    {"min_elongation": 2.0, "maks_hoyde": 50, "maks_bredde": 100},
]



# ── Udekket-gjennomgang ───────────────────────────────────────
# Motstykket til tapt-gjennomgangen: ikke «hva fjerner filteret», men «hva
# fant modellen aldri, eller bokset for dårlig». Fasit splittes på
# ml_generated: ML-aksepterte bokser er modellens egne godkjente forslag og
# gjenfinnes nesten alltid av en re-kjøring (sirkulært) — reell
# deteksjonsevne måles på de manuelt tegnede boksene. Manuelle sladdinger
# kan selv være slurvete tegnet, så utsnittene viser både fasit-boksen og
# modellens beste forslag: da kan gjennomgangen skille «modell-bom»,
# «dårlig boksing» og «fasit-slurv» i vurdering-kolonnen.

def _er_ml_generert(rad):
    return (rad.get("ml_generated") or "").strip().lower() in ("true", "t", "1")


def _p10_p50_p90(verdier):
    v = sorted(verdier)
    n = len(v)
    return v[int(0.10 * (n - 1))], v[int(0.50 * (n - 1))], v[int(0.90 * (n - 1))]


def gjennomgang_udekket(fasit_csv, res_csv, mappe, ut_mappe,
                        kriterium=STD_KRITERIUM, terskel=STD_TERSKEL,
                        god_dekning=0.90, kjorte_dok=None, ogsaa_ml=False,
                        maks_utsnitt=None, utsnitt_margin=60.0):
    """Katalogiserer og tegner fasit-bokser uten (god nok) dekning."""
    if kriterium in KRITERIUM_LAV_ER_BRA:
        raise SystemExit(f"--udekket støtter ikke kriterium {kriterium!r} "
                         f"(der er lav verdi bra — bruk areal/kortside/iou)")
    felt = KRITERIUM_FELT[kriterium]

    fasit_rader, _forkastet, kolonner = les_fasit_rader(fasit_csv)
    if "ml_generated" not in kolonner:
        print("⚠ Fasit-CSV-en mangler ml_generated — alle bokser behandles "
              "som manuelle. Ta kolonnen med i neste eksport.")
    pred = les_prediksjoner(res_csv)

    kjorte = (set(kjorte_dok) if kjorte_dok is not None
              else {p["dok_nr"] for p in pred})
    scope = {r["dok_nr"] for r in fasit_rader} & kjorte

    side_str = {}
    per_side_pred = defaultdict(list)
    navn_per_dok = {}
    for p in pred:
        key = (p["dok_nr"], p["side"])
        side_str.setdefault(key, (p["bw"] / SKALA, p["bh"] / SKALA))
        per_side_pred[key].append(p)
        navn_per_dok.setdefault(p["dok_nr"], p["navn"])

    grupper = {"udekket": [], "daarlig_dekket": [], "ok": []}
    for r in fasit_rader:
        if r["dok_nr"] not in scope:
            continue
        x0, y0, x1, y1 = r["boks"]
        pw, ph = side_str.get((r["dok_nr"], r["side"]), (595.0, 842.0))
        fn = (x0 / pw, y0 / ph, x1 / pw, y1 / ph)
        horisontal = (x1 - x0) >= (y1 - y0)
        beste_v, beste_p = 0.0, None
        for p in per_side_pred.get((r["dok_nr"], r["side"]), ()):
            m = match_metrikker(p["norm"], fn, horisontal)
            if m is not None and m[felt] > beste_v:
                beste_v, beste_p = m[felt], p
        r["_dekning"] = beste_v
        r["_beste_p"] = beste_p
        r["_ml"] = _er_ml_generert(r["rad"])
        gruppe = ("udekket" if beste_v < terskel
                  else "daarlig_dekket" if beste_v < god_dekning else "ok")
        grupper[gruppe].append(r)

    n_scope = sum(len(g) for g in grupper.values())
    print(f"\nUdekket-gjennomgang  (kriterium «{kriterium}», terskel "
          f"{terskel:g}, god dekning ≥ {god_dekning:g})")
    print(f"  Scope: {len(scope)} dokumenter, {n_scope} fasit-bokser")
    for ml, navn in ((False, "manuelt tegnet"), (True, "ML-akseptert ")):
        tell = {k: sum(1 for r in g if r["_ml"] == ml)
                for k, g in grupper.items()}
        n = sum(tell.values())
        if n:
            print(f"  {navn} ({n:>5} bokser): "
                  f"udekket {tell['udekket']:>4} ({100*tell['udekket']/n:4.1f}%)  "
                  f"dårlig {tell['daarlig_dekket']:>4} "
                  f"({100*tell['daarlig_dekket']/n:4.1f}%)  "
                  f"god {tell['ok']:>5} ({100*tell['ok']/n:4.1f}%)")

    utvalg = [r for r in grupper["udekket"] + grupper["daarlig_dekket"]
              if ogsaa_ml or not r["_ml"]]
    if not utvalg:
        print("  Ingenting å tegne.")
        return

    # ── Karakteristikk: hva har de udekkede boksene til felles? ──
    staaende = sum(1 for r in utvalg if r["h"] > r["w"])
    print(f"\n  Utvalg til gjennomgang: {len(utvalg)} bokser "
          f"({'manuelle + ML' if ogsaa_ml else 'kun manuelt tegnede'})")
    print(f"    stående (h > b):   {staaende} ({100*staaende/len(utvalg):.1f}%)")
    for maal in ("kortside", "langside"):
        p10, p50, p90 = _p10_p50_p90([r[maal] for r in utvalg])
        print(f"    {maal:>8} (pt):    p10 {p10:5.1f}   p50 {p50:5.1f}   "
              f"p90 {p90:5.1f}")
    per_type = defaultdict(int)
    for r in utvalg:
        per_type[r["type"]] += 1
    print("    per type:          "
          + "  ".join(f"{t}={n}" for t, n in
                      sorted(per_type.items(), key=lambda kv: -kv[1])))
    per_dok = defaultdict(int)
    for r in utvalg:
        per_dok[r["dok_nr"]] += 1
    n_en = sum(1 for n in per_dok.values() if n == 1)
    print(f"    fordelt på {len(per_dok)} dokumenter "
          f"({n_en} med bare én boks); verstinger:")
    for dnr, n in sorted(per_dok.items(), key=lambda kv: -kv[1])[:15]:
        print(f"      {n:>4}  {navn_per_dok.get(dnr, f'{dnr}.pdf')}")

    # ── Manifest ──
    utvalg.sort(key=lambda r: (r["_dekning"] >= terskel, navn_per_dok.get(
        r["dok_nr"], str(r["dok_nr"])), r["side"], r["boks"][1]))
    if maks_utsnitt:
        utvalg = utvalg[:maks_utsnitt]

    manifest = []
    for nr, r in enumerate(utvalg, 1):
        bp = r["_beste_p"]
        fil = navn_per_dok.get(r["dok_nr"], f"{r['dok_nr']}.pdf")
        base = os.path.splitext(os.path.basename(fil))[0]
        manifest.append({
            "nr": nr, "fil": fil, "side": r["side"],
            "gruppe": ("udekket" if r["_dekning"] < terskel
                       else "daarlig_dekket"),
            "dekning_pst": round(100 * r["_dekning"], 1),
            "ml_generated": int(r["_ml"]),
            "ml_status": r["ml_status"], "type": r["type"],
            "staaende": int(r["h"] > r["w"]),
            "fasit_bredde_pt": round(r["w"], 1),
            "fasit_hoyde_pt": round(r["h"], 1),
            "beste_kilde": bp["kilde"] if bp else "",
            "beste_conf": (bp["conf"] if bp and bp["conf"] is not None
                           else ""),
            "beste_bredde_pt": round(bp["w"], 1) if bp else "",
            "beste_hoyde_pt": round(bp["h"], 1) if bp else "",
            "fasit_x0": round(r["boks"][0], 1),
            "fasit_y0": round(r["boks"][1], 1),
            "utsnitt": f"{nr:04d}_{base}_side{r['side']}.png",
            "vurdering": "",
            "_r": r,
        })

    os.makedirs(ut_mappe, exist_ok=True)
    manifest_sti = os.path.join(ut_mappe, "udekket.csv")
    felt_csv = ["nr", "fil", "side", "gruppe", "dekning_pst", "ml_generated",
                "ml_status", "type", "staaende", "fasit_bredde_pt",
                "fasit_hoyde_pt", "beste_kilde", "beste_conf",
                "beste_bredde_pt", "beste_hoyde_pt", "fasit_x0", "fasit_y0",
                "utsnitt", "vurdering"]
    with open(manifest_sti, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=felt_csv, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest)
    print(f"\n  Manifest: {manifest_sti}  ({len(manifest)} rader)")

    # ── Tegning ──
    side_mappe = os.path.join(ut_mappe, "sider")
    utsnitt_mappe = os.path.join(ut_mappe, "utsnitt")
    os.makedirs(side_mappe, exist_ok=True)
    os.makedirs(utsnitt_mappe, exist_ok=True)

    per_fil = defaultdict(lambda: defaultdict(list))
    for rad in manifest:
        per_fil[rad["fil"]][rad["side"]].append(rad)

    n_sider = n_utsnitt = 0
    for fil in sorted(per_fil):
        sti = os.path.join(mappe, fil)
        if not os.path.isfile(sti):
            print(f"  ⚠ Finner ikke {sti}, hopper over")
            continue
        try:
            dok = fitz.open(sti)
        except Exception as e:
            print(f"  ⚠ Kunne ikke åpne {fil}: {e!r}")
            continue
        for si, rader in sorted(per_fil[fil].items()):
            if not 1 <= si <= len(dok):
                continue
            bilde = _render_side(dok, si)
            rekt = dok[si - 1].rect
            skx = bilde.width / rekt.width if rekt.width else SKALA
            sky = bilde.height / rekt.height if rekt.height else SKALA
            base = bilde.convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            tegner = ImageDraw.Draw(overlay)
            dnr = dok_nr(fil)
            valgt = {id(rad["_r"]) for rad in rader}

            # Alle prediksjoner på siden, tynt grått
            side_pred = per_side_pred.get((dnr, si), ())
            for p2 in side_pred:
                px = p2["px"]
                sx2 = bilde.width / p2["bw"]
                sy2 = bilde.height / p2["bh"]
                tegner.rectangle([px[0] * sx2, px[1] * sy2,
                                  px[2] * sx2, px[3] * sy2],
                                 outline=BEHOLDT, width=1)

            # Øvrige fasit-bokser på siden, tynt blått
            for r2 in grupper["udekket"] + grupper["daarlig_dekket"] + grupper["ok"]:
                if (r2["dok_nr"], r2["side"]) != (dnr, si) or id(r2) in valgt:
                    continue
                fx0, fy0, fx1, fy1 = r2["boks"]
                tegner.rectangle([fx0 * skx, fy0 * sky, fx1 * skx, fy1 * sky],
                                 outline=FASIT, width=2)

            # De utvalgte: fasit magenta + beste forslag oransje
            for rad in rader:
                r2 = rad["_r"]
                fx0, fy0, fx1, fy1 = r2["boks"]
                rr = [fx0 * skx - 6, fy0 * sky - 6,
                      fx1 * skx + 6, fy1 * sky + 6]
                tegner.rectangle(rr, outline=TAPT_FASIT, width=5)
                merke = (f"{rad['gruppe'].upper()} {rad['dekning_pst']:g}% "
                         f"[{'ML' if r2['_ml'] else 'manuell'}]")
                _tegn_tekst(tegner, rr, merke, TAPT_FASIT, over=True)
                bp = r2["_beste_p"]
                if bp is not None:
                    px = bp["px"]
                    sx2 = bilde.width / bp["bw"]
                    sy2 = bilde.height / bp["bh"]
                    rp = [px[0] * sx2, px[1] * sy2, px[2] * sx2, px[3] * sy2]
                    tegner.rectangle(rp, outline=BOM, width=3)
                    tekst = f"BESTE FORSLAG [{bp['kilde']}]"
                    if bp["conf"] is not None:
                        tekst += f" conf={bp['conf']:.2f}"
                    _tegn_tekst(tegner, rp, tekst, BOM, over=False)

            ferdig = Image.alpha_composite(base, overlay).convert("RGB")
            ferdig.save(os.path.join(
                side_mappe, f"{os.path.splitext(fil)[0]}_side{si}.png"))
            n_sider += 1

            mrg = utsnitt_margin * skx
            for rad in rader:
                fx0, fy0, fx1, fy1 = rad["_r"]["boks"]
                boks = (max(0, int(fx0 * skx - mrg)),
                        max(0, int(fy0 * sky - mrg)),
                        min(ferdig.width, int(fx1 * skx + mrg)),
                        min(ferdig.height, int(fy1 * sky + mrg)))
                if boks[2] > boks[0] and boks[3] > boks[1]:
                    ferdig.crop(boks).save(
                        os.path.join(utsnitt_mappe, rad["utsnitt"]))
                    n_utsnitt += 1
        dok.close()

    print(f"  Tegnet {n_sider} sider til {side_mappe}")
    print(f"  {n_utsnitt} utsnitt i {utsnitt_mappe} — samme rekkefølge som "
          f"udekket.csv, klare for vurdering-kolonnen")


def main():
    p = argparse.ArgumentParser(
        description="Visuell gjennomgang av bokser som fjernes av en "
                    "filterkonfigurasjon, gruppert etter om fjerningen faktisk "
                    "koster recall.")
    p.add_argument("--fasit-csv", required=True, help="Labels-CSV (fasit)")
    p.add_argument("--res-csv", default=None,
                   help="Resultat-CSV fra modellen (ikke nødvendig med --mot-fasit)")
    p.add_argument("--mappe", default=None, help="Mappe med PDF-dokumentene")
    p.add_argument("--ut-mappe", default="filter_review",
                   help="Mappe for PNG-output (default: filter_review)")
    p.add_argument("--kriterium", default=STD_KRITERIUM,
                   choices=sorted(KRITERIER),
                   help=f"Matcheregel for dekning (default: {STD_KRITERIUM})")
    p.add_argument("--terskel", type=float, default=STD_TERSKEL,
                   help=f"Overlapp-terskel for dekning (default: {STD_TERSKEL})")
    p.add_argument("--slurv-faktor", type=float, default=STD_SLURV_FAKTOR,
                   help=f"SLURV-grense (default: {STD_SLURV_FAKTOR})")
    p.add_argument("--inkluder-ulabelte", action="store_true", default=True,
                   help="(default på) Ta med kjørte dokumenter uten rader i "
                        "fasit-CSV-en — labels-filen dekker hele uttrekket, "
                        "så de er gjennomgått med null fnr og prediksjoner "
                        "der er ekte oversladdinger")
    p.add_argument("--ekskluder-ulabelte", dest="inkluder_ulabelte",
                   action="store_false",
                   help="Gammel oppførsel: hold dokumenter uten fasit-rader "
                        "utenfor scope (for eldre labels-filer som ikke "
                        "dekket hele uttrekket)")

    p.add_argument("--kjorte-liste", default=None, metavar="FIL",
                   help="Fil med dokumentene modellen har kjørt på (ett navn "
                        "eller nummer per linje). Uten den antas dokumentene "
                        "i resultat-CSV-en, og et dokument der modellen ikke "
                        "fant noe regnes som ukjørt.")
    filt = p.add_argument_group("Filterparametre (oppgi minst ett, "
                                "eller bruk --per-kilde/--sweep)")
    filt.add_argument("--elongation", type=float, default=None,
                      dest="min_elongation", help="MIN_ELONGATION")
    filt.add_argument("--maks-hoyde", type=float, default=None,
                      dest="maks_hoyde", help="Maks bokshøyde i punkt")
    filt.add_argument("--maks-bredde", type=float, default=None,
                      dest="maks_bredde", help="Maks boksbredde i punkt")
    filt.add_argument("--maks-areal", type=float, default=None,
                      dest="maks_areal", help="Maks boksareal i pt²")
    filt.add_argument("--maks-elongation", type=float, default=None,
                      dest="maks_elongation",
                      help="Maks elongation — fjerner tynne, lange streker")
    filt.add_argument("--min-hoyde", type=float, default=None, dest="min_hoyde",
                      help="Min bokshøyde i punkt")
    filt.add_argument("--min-bredde", type=float, default=None, dest="min_bredde",
                      help="Min boksbredde i punkt")
    filt.add_argument("--min-kortside", type=float, default=None,
                      dest="min_kortside",
                      help="Min korteste side i punkt (orienteringsuavhengig — "
                           "rammer ikke stående bokser slik --min-bredde gjør)")
    filt.add_argument("--maks-kortside", type=float, default=None,
                      dest="maks_kortside", help="Maks korteste side i punkt")
    filt.add_argument("--min-langside", type=float, default=None,
                      dest="min_langside",
                      help="Min lengste side i punkt — for kort til 5 sifre")
    filt.add_argument("--maks-langside", type=float, default=None,
                      dest="maks_langside", help="Maks lengste side i punkt")
    filt.add_argument("--min-areal-px", type=float, default=None,
                      dest="min_areal_px",
                      help="Min boksareal i PIKSEL² (som MIN_BOKS_AREAL)")
    filt.add_argument("--conf", type=float, default=None, dest="conf_terskel",
                      help="conf ≥ denne verdien beholdes uansett geometri")

    ocr = p.add_argument_group(
        "OCR-trekk (strengere snill_sjekk; treffer kun kilde «yolo» med "
        "tekst i boksen — se _ocr_grunn i filter_felles.py)")
    ocr.add_argument("--min-siffer", type=float, default=None,
                     dest="min_siffer",
                     help="Krev minst N siffer i boksen (prod i dag: 1)")
    ocr.add_argument("--maks-bokstaver", type=float, default=None,
                     dest="maks_bokstaver",
                     help="Tillat høyst N bokstaver i boksen (prod i dag: 1)")
    ocr.add_argument("--min-siffer-run", type=float, default=None,
                     dest="min_siffer_run",
                     help="Krev at lengste sifferløp over boksen er minst N")
    ocr.add_argument("--krev-fnr-kandidat", action="store_const", const=1,
                     default=None, dest="krev_fnr_kandidat",
                     help="Krev et 11-sifret løp med gyldig fnr-form på linjen "
                          "(uten mod11)")
    ocr.add_argument("--avvis-desimal", action="store_const", const=1,
                     default=None, dest="avvis_desimal",
                     help="Forkast bokser med desimalskille i tallet")
    ocr.add_argument("--rec-veto", type=float, default=None, dest="rec_veto",
                     help="Slå OCR-reglene over på først når rec_min ≥ V. "
                          "Under V leste Paddle dårlig, og fraværet av et fnr "
                          "er ikke bevis for noe.")
    ocr.add_argument("--ocr-conf-fritak", type=float, default=None,
                     dest="ocr_conf_fritak",
                     help="OCR-reglene viker for bokser med deteksjons-conf "
                          "≥ V — sikker YOLO-deteksjon overstyrer tekstbevis.")
    ocr.add_argument("--avvis-00-run", action="store_const", const=1,
                     default=None, dest="avvis_00_run",
                     help="Forkast bokser der et 10-12-sifret løp starter "
                          "med 00 — orgnr paddet til fnr-bredde; dag 00 er "
                          "ugyldig i et fnr")
    ocr.add_argument("--avvis-orgnr", action="store_const", const=1,
                     default=None, dest="avvis_orgnr",
                     help="Forkast bokser med gyldig orgnr-mod11 (9 siffer "
                          "som starter på 8/9, evt. 00-paddet)")
    ocr.add_argument("--avvis-org-ord", type=float, default=None,
                     dest="avvis_org_ord", metavar="{1,2}",
                     help="Forkast bokser med selskapsform-ord nær seg "
                          "(AS, Borettslag, Org.nr, …). 1=alltid, 2=kun når "
                          "boksen også mangler fnr-kandidat")
    ocr.add_argument("--linje-veto", type=float, default=None,
                     dest="linje_veto",
                     help="Slå OCR-reglene på først når rec_min_linje ≥ V — "
                          "fnr-kandidat og løpelengde avhenger av at HELE "
                          "linjen er riktig lest, ikke bare boksen")
    ocr.add_argument("--avvis-run-6-10", type=float, nargs="?", const=1,
                     default=None, dest="avvis_run_6_10", metavar="MAKS",
                     help="Forkast bokser over sifferløp på 6..MAKS (med "
                          "luker); uten verdi = 6..10. Bruk 9: 10-løp er "
                          "ofte fnr med ensifret dag/måned eller mistet tegn")
    ocr.add_argument("--uten-tekst-conf", type=float, default=None,
                     dest="uten_tekst_conf",
                     help="Bokser UTEN tekst (har_tokens=0) krever conf ≥ V "
                          "— strengere enn prods YOLO_CONF_UTEN_TEKST (0.40)")

    ocr.add_argument("--maks-luke", type=float, default=None,
                     dest="maks_luke", metavar="BREDDER",
                     help="Forkast paddle/begge-bokser der største fysiske "
                          "luke i 11-siffer-vinduet er ≥ V sifferbredder — "
                          "vinduer sydd på tvers av kolonnegap")
    ocr.add_argument("--avvis-desimal-luke", action="store_const", const=1,
                     default=None, dest="avvis_desimal_luke",
                     help="Forkast paddle/begge-bokser der 11-vinduet er "
                          "sydd over et desimalskille (. eller ,)")
    p.add_argument("--per-kilde", nargs="+", metavar="SPEC",
                   help='Uavhengige filtre per kilde: "kilde:e=V,h=V,b=V,a=V,c=V"')
    p.add_argument("--mot-fasit", action="store_true", dest="mot_fasit",
                   help="Anvend filteret DIREKTE på saksbehandlernes sladdinger "
                        "(alle labels, ikke bare dokumenter modellen kjørte på). "
                        "Svarer på hvor mange riktige sladdinger filteret ville "
                        "forkastet. Krever ikke --res-csv.")
    p.add_argument("--bom", nargs="?", const="alle", default=None,
                   metavar="KILDE",
                   help="Triage-modus: tegn ALLE BOM-bokser (treffer ingen "
                        "fasit-boks) uavhengig av filter, 'begge'-kilden "
                        "først. Med verdi (begge/paddle/yolo) tegnes kun den "
                        "kilden. Svarer på om de er oversladding eller "
                        "fødselsnumre saksbehandleren bommet på.")
    p.add_argument("--ocr-tekst", action="store_true", dest="ocr_tekst",
                   help="--bom: falm originalen og tegn Paddles cachede "
                        "OCR-tokens oppå, farget etter rec-score (grønn "
                        "≥0.98, blå ≥0.90, rød <0.90). Viser hva OCR "
                        "faktisk leste der boksen ble satt.")
    p.add_argument("--ocr-cache", default=None, metavar="STI",
                   dest="ocr_cache",
                   help="OCR-cache-mappe for --ocr-tekst "
                        "(default: $SLADD_CACHE/<mappenavn>/ocr)")
    p.add_argument("--ocr-opacity", type=float, default=0.15,
                   dest="ocr_opacity", metavar="ANDEL",
                   help="Opacity for originalen bak tekstlaget "
                        "(default 0.15)")
    p.add_argument("--udekket", action="store_true",
                   help="Gjennomgangs-modus: katalogiser og tegn fasit-bokser "
                        "modellen ikke dekker (godt nok). Splitter på "
                        "ml_generated — reell deteksjonsevne måles på "
                        "manuelt tegnede bokser. Trenger ingen filterflagg.")
    p.add_argument("--god-dekning", type=float, default=0.90,
                   dest="god_dekning", metavar="ANDEL",
                   help="--udekket: dekning under dette regnes som «dårlig "
                        "dekket» selv om terskelen er nådd (default 0.90)")
    p.add_argument("--udekket-ogsaa-ml", action="store_true",
                   dest="udekket_ogsaa_ml",
                   help="--udekket: tegn også ML-aksepterte bokser (ellers "
                        "kun manuelt tegnede — ML-boksene er sirkulære)")
    p.add_argument("--maks-utsnitt", type=int, default=None,
                   dest="maks_utsnitt", metavar="N",
                   help="--udekket: maks antall bokser å tegne "
                        "(dårligst dekning først)")
    p.add_argument("--band", nargs=3, default=None,
                   metavar=("KRITERIUM", "LO", "HI"),
                   help="Tegn utsnitt av hvert (prediksjon, fasit)-par der "
                        "målet ligger i [LO, HI) — gråsonen terskelvalget "
                        "faktisk avgjør. F.eks. «--band areal 0.40 0.45». "
                        f"Kriterier: {', '.join(sorted(KRITERIER))}")
    p.add_argument("--band-csv", default=None, metavar="FIL",
                   help="Skriv måltallene for båndet til CSV")
    p.add_argument("--sweep", action="store_true",
                   help="Kjør et sett forhåndsdefinerte konfigurasjoner")
    p.add_argument("--kun-tapt", "--kun-riktige", action="store_true",
                   dest="kun_tapt",
                   help="Tegn kun sider der en fasit-boks mistet all dekning")
    p.add_argument("--utsnitt-margin", type=float, default=60.0, metavar="PT",
                   help="Margin rundt utsnittene av tapte bokser, i punkt. "
                        "0 slår av utsnitt (default: 60)")
    p.add_argument("--maks-sider", type=int, default=None,
                   help="Maks antall sider å tegne (verst først)")
    p.add_argument("--velg", nargs="+", metavar="PDF",
                   help="Begrens til disse PDF-filene")
    args = p.parse_args()

    kw_alle = {n: getattr(args, n) for n in
               ("min_elongation", "maks_elongation", "maks_hoyde", "min_hoyde",
                "maks_bredde", "min_bredde", "min_kortside", "maks_kortside",
                "min_langside", "maks_langside", "maks_areal", "min_areal_px",
                "conf_terskel") + OCR_PARAMETRE
               if getattr(args, n) is not None}

    if args.udekket:
        if not args.res_csv or not args.mappe:
            p.error("--udekket krever --res-csv og --mappe")
        kjorte = les_kjorte_dok(args.kjorte_liste) if args.kjorte_liste else None
        gjennomgang_udekket(args.fasit_csv, args.res_csv, args.mappe,
                            args.ut_mappe, kriterium=args.kriterium,
                            terskel=args.terskel,
                            god_dekning=args.god_dekning, kjorte_dok=kjorte,
                            ogsaa_ml=args.udekket_ogsaa_ml,
                            maks_utsnitt=args.maks_utsnitt,
                            utsnitt_margin=args.utsnitt_margin)
        print("\nFerdig!")
        return

    if args.mot_fasit:
        if not kw_alle:
            p.error("--mot-fasit krever minst ett filter "
                    "(--elongation, --maks-elongation, --min-kortside, ...)")
        ds_kryss = None
        if args.res_csv:
            kjorte = (les_kjorte_dok(args.kjorte_liste)
                      if args.kjorte_liste else None)
            ds_kryss = bygg_datasett(
                les_fasit(args.fasit_csv), les_prediksjoner(args.res_csv),
                terskel=args.terskel, slurv_faktor=args.slurv_faktor,
                inkluder_ulabelte=args.inkluder_ulabelte, kjorte_dok=kjorte,
                kriterium=args.kriterium)
        test_mot_fasit(args.fasit_csv, args.mappe, args.ut_mappe, kw_alle,
                       maks_sider=args.maks_sider,
                       utsnitt_margin=args.utsnitt_margin, velg=args.velg,
                       ds=ds_kryss)
        print("\nFerdig!")
        return

    if not args.res_csv:
        p.error("--res-csv er påkrevd (unntatt med --mot-fasit)")
    if not args.mappe:
        p.error("--mappe er påkrevd")

    kjorte = les_kjorte_dok(args.kjorte_liste) if args.kjorte_liste else None
    ds = bygg_datasett(les_fasit(args.fasit_csv),
                       les_prediksjoner(args.res_csv),
                       terskel=args.terskel, slurv_faktor=args.slurv_faktor,
                       inkluder_ulabelte=args.inkluder_ulabelte,
                       kjorte_dok=kjorte, kriterium=args.kriterium)
    skriv_oppsummering(ds)

    felles = dict(kun_tapt=args.kun_tapt, velg=args.velg,
                  maks_sider=args.maks_sider,
                  utsnitt_margin=args.utsnitt_margin)

    if args.band:
        kriterium, lo, hi = args.band[0], float(args.band[1]), float(args.band[2])
        if kriterium not in KRITERIER:
            p.error(f"ukjent kriterium {kriterium!r} — "
                    f"gyldige: {', '.join(sorted(KRITERIER))}")
        if not lo < hi:
            p.error(f"LO må være mindre enn HI (fikk {lo} og {hi})")
        band_review(ds, args.mappe, args.ut_mappe, kriterium, lo, hi,
                    maks=args.maks_sider,
                    utsnitt_margin=(args.utsnitt_margin
                                    if args.utsnitt_margin != 60.0 else 25.0),
                    ut_csv=args.band_csv)
    elif args.bom:
        ocr_mappe = None
        if args.ocr_tekst:
            ocr_mappe = args.ocr_cache
            if not ocr_mappe:
                base = os.environ.get("SLADD_CACHE")
                if not base:
                    p.error("--ocr-tekst: oppgi --ocr-cache, eller sett "
                            "$SLADD_CACHE (source activate.sh)")
                ocr_mappe = os.path.join(
                    base, os.path.basename(os.path.normpath(args.mappe)),
                    "ocr")
            if not os.path.isdir(ocr_mappe):
                p.error(f"--ocr-tekst: finner ikke cache-mappen {ocr_mappe}")
        triage_bom(ds, args.mappe, args.ut_mappe, velg=args.velg,
                   maks_sider=args.maks_sider,
                   kilde=None if args.bom == "alle" else args.bom,
                   ocr_mappe=ocr_mappe, ocr_opacity=args.ocr_opacity)
    elif args.per_kilde:
        per_kilde = parse_per_kilde(args.per_kilde)
        ukjente = set(per_kilde) - {k.lower() for k in ds.kilder()}
        if ukjente:
            print(f"  ⚠ Ingen prediksjoner fra kilde(r): {', '.join(sorted(ukjente))}")
        ut = os.path.join(args.ut_mappe, _mappenavn_per_kilde(per_kilde))
        generer_bilder(ds, args.mappe, ut, per_kilde=per_kilde, **felles)
    elif args.sweep:
        for kw in SWEEP_KONFIGER:
            ut = os.path.join(args.ut_mappe, _mappenavn(kw))
            generer_bilder(ds, args.mappe, ut, filter_kwargs=kw, **felles)
    else:
        kw = kw_alle
        if not kw:
            p.error("Oppgi minst ett filter (--elongation, --maks-hoyde, "
                    "--maks-bredde, --maks-areal, --conf), --per-kilde, "
                    "--bom eller --sweep")
        generer_bilder(ds, args.mappe, args.ut_mappe, filter_kwargs=kw, **felles)

    print("\nFerdig!")


if __name__ == "__main__":
    main()
