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
                        ("ocr_conf_fritak", "c≥{:g}→OCR-fritak")):
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
                         ("ocr_conf_fritak", "cfritak")):
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

    per_fil = defaultdict(list)
    for (_, _, _, navn, si) in aktuelle:
        per_fil[navn].append(si)

    telling = defaultdict(int)
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

        for si in sorted(per_fil[navn]):
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


def triage_bom(ds, mappe, ut_mappe, velg=None, maks_sider=None):
    """Tegner ALLE BOM-prediksjoner, uavhengig av filter.

    En BOM-boks treffer ingen fasit-boks, men fasit er menneskeskapt: den kan
    like godt være et fødselsnummer saksbehandleren bommet på som en reell
    oversladding. Det skillet kan ikke leses ut av geometrien — det må ses.

    Sidene sorteres slik at 'begge'-bokser kommer først: der begge modellene
    er enige om at det står et fødselsnummer, er sannsynligheten for at fasit
    er feil størst.
    """
    bom = [p for p in ds.pred if p["klasse"] == "BOM"]
    print(f"\nTRIAGE av BOM-bokser (treffer ingen fasit-boks)")
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

    per_fil = defaultdict(list)
    for (navn, si), _ in rangert:
        per_fil[navn].append(si)

    telling = defaultdict(int)
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

        for si in sorted(per_fil[navn]):
            if not 1 <= si <= len(dok):
                continue
            side_pred = per_side[(navn, si)]
            bilde = _render_side(dok, si)
            base = bilde.convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            tegner = ImageDraw.Draw(overlay)
            sx = bilde.width / side_pred[0]["bw"]
            sy = bilde.height / side_pred[0]["bh"]

            for fb in fasit_per_side.get((dok_nr(navn), si), ()):
                fx0, fy0, fx1, fy1 = fb["boks"]
                tegner.rectangle([fx0 * SKALA * sx, fy0 * SKALA * sy,
                                  fx1 * SKALA * sx, fy1 * SKALA * sy],
                                 outline=FASIT, width=2)

            kilder_her = set()
            for p in side_pred:
                px = p["px"]
                r = [px[0] * sx, px[1] * sy, px[2] * sx, px[3] * sy]
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
    p.add_argument("--inkluder-ulabelte", action="store_true",
                   help="Ta med dokumenter som ikke finnes i fasit-CSV-en")

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

    p.add_argument("--per-kilde", nargs="+", metavar="SPEC",
                   help='Uavhengige filtre per kilde: "kilde:e=V,h=V,b=V,a=V,c=V"')
    p.add_argument("--mot-fasit", action="store_true", dest="mot_fasit",
                   help="Anvend filteret DIREKTE på saksbehandlernes sladdinger "
                        "(alle labels, ikke bare dokumenter modellen kjørte på). "
                        "Svarer på hvor mange riktige sladdinger filteret ville "
                        "forkastet. Krever ikke --res-csv.")
    p.add_argument("--bom", action="store_true",
                   help="Triage-modus: tegn ALLE BOM-bokser (treffer ingen "
                        "fasit-boks) uavhengig av filter, 'begge'-kilden "
                        "først. Svarer på om de er oversladding eller "
                        "fødselsnumre saksbehandleren bommet på.")
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
        triage_bom(ds, args.mappe, args.ut_mappe, velg=args.velg,
                   maks_sider=args.maks_sider)
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
