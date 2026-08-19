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
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz
from PIL import Image, ImageDraw, ImageFont

from filter_felles import (PDF_DPI, SKALA, STD_SLURV_FAKTOR, STD_TERSKEL,
                           bygg_datasett, dok_nr, evaluer, filter_grunner,
                           lag_filter, lag_filter_per_kilde, les_fasit, les_kjorte_dok,
                           les_prediksjoner, parse_per_kilde,
                           skriv_oppsummering)

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
                        ("maks_areal", "a≤{:g}"), ("min_areal_px", "apx≥{:g}"),
                        ("conf_terskel", "c≥{:g}→behold")):
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
                         ("maks_areal", "a"), ("min_areal_px", "apx"),
                         ("conf_terskel", "c")):
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
            manifest.append({
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
        felt = ["nr", "fil", "side", "grunn", "kilde", "conf", "elongation",
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
    p.add_argument("--res-csv", required=True, help="Resultat-CSV fra modellen")
    p.add_argument("--mappe", required=True, help="Mappe med PDF-dokumentene")
    p.add_argument("--ut-mappe", default="filter_review",
                   help="Mappe for PNG-output (default: filter_review)")
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
    filt.add_argument("--min-areal-px", type=float, default=None,
                      dest="min_areal_px",
                      help="Min boksareal i PIKSEL² (som MIN_BOKS_AREAL)")
    filt.add_argument("--conf", type=float, default=None, dest="conf_terskel",
                      help="conf ≥ denne verdien beholdes uansett geometri")

    p.add_argument("--per-kilde", nargs="+", metavar="SPEC",
                   help='Uavhengige filtre per kilde: "kilde:e=V,h=V,b=V,a=V,c=V"')
    p.add_argument("--bom", action="store_true",
                   help="Triage-modus: tegn ALLE BOM-bokser (treffer ingen "
                        "fasit-boks) uavhengig av filter, 'begge'-kilden "
                        "først. Svarer på om de er oversladding eller "
                        "fødselsnumre saksbehandleren bommet på.")
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

    kjorte = les_kjorte_dok(args.kjorte_liste) if args.kjorte_liste else None
    ds = bygg_datasett(les_fasit(args.fasit_csv),
                       les_prediksjoner(args.res_csv),
                       terskel=args.terskel, slurv_faktor=args.slurv_faktor,
                       inkluder_ulabelte=args.inkluder_ulabelte,
                       kjorte_dok=kjorte)
    skriv_oppsummering(ds)

    felles = dict(kun_tapt=args.kun_tapt, velg=args.velg,
                  maks_sider=args.maks_sider,
                  utsnitt_margin=args.utsnitt_margin)

    if args.bom:
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
        kw = {n: getattr(args, n) for n in
              ("min_elongation", "maks_elongation", "maks_hoyde", "min_hoyde",
               "maks_bredde", "min_bredde", "maks_areal", "min_areal_px",
               "conf_terskel") if getattr(args, n) is not None}
        if not kw:
            p.error("Oppgi minst ett filter (--elongation, --maks-hoyde, "
                    "--maks-bredde, --maks-areal, --conf), --per-kilde, "
                    "--bom eller --sweep")
        generer_bilder(ds, args.mappe, args.ut_mappe, filter_kwargs=kw, **felles)

    print("\nFerdig!")


if __name__ == "__main__":
    main()
