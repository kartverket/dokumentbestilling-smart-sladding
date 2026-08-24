"""Eksporterer bilde-utsnitt av foreslåtte sladdebokser for VLM-dømming.

Steg 1 av VLM-verifikator-piloten. Spørsmålet piloten skal svare på er om en
lokal vision-LLM kan skille ekte fødselsnumre fra oversladdinger (koordinater,
kontonumre, tabellceller) der regelverket ikke kan — se docs/ og
utils/filter_sweep.py for regelsiden.

Dette scriptet gjør ingenting nytt med klassifiseringen: det gjenbruker
bygg_datasett fra filter_felles (samme scope-, terskel- og klasseregler som
filter_review/filter_sweep) og croppe-maskineriet fra filter_review, og skriver
ut ett PNG-utsnitt per prediksjon pluss et manifest.

Utvalget er ASYMMETRISK med vilje:
  * ALLE BOM-bokser (oversladdingene) — det er dem gevinsten skal hentes fra
  * et TILFELDIG utvalg dekkende bokser (TREFF/SLURV) — de måler tapsrisikoen

Fordi TREFF er et utvalg og BOM er totalen, skrives utvalgsfaktorene til
utvalg.json ved siden av manifestet. vlm_evaluer bruker dem til å skalere
tapsestimatet opp til fullt uttrekk — uten dem blir ov/tapt-forholdet
meningsløst optimistisk.

Utsnittet tegnes med en rød ramme rundt boksen som skal dømmes: uten den vet
ikke modellen hvilket av tallene i utsnittet spørsmålet gjelder.

Eksempel:
    python utils/vlm_eksport.py \
        --res-csv  $SLADD_VALIDERING/uttrekk6_frreg/resultat.csv \
        --fasit-csv $SLADD_LABELS/uttrekk_6.csv \
        --mappe    $SLADD_UTTREKK/uttrekk_6 \
        --ut-mappe /data2/vlm/uttrekk6_kalibrering \
        --kjorte-liste /data2/tmp/rs_lister/rs_TR_MAS.txt \
        --treff-utvalg 0 \
        --ocr-cache $SLADD_CACHE/uttrekk_6/ocr
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz
from PIL import Image, ImageDraw

from filter_felles import (KRITERIER, PDF_DPI, SKALA, STD_KRITERIUM,
                           STD_SLURV_FAKTOR, STD_TERSKEL, bygg_datasett,
                           dok_nr, les_fasit, les_fasit_rader,
                           les_kjorte_dok, les_prediksjoner,
                           skriv_oppsummering)
from filter_review import _ocr_les_cache, _rekt_frem, _render_side

MARKOR = (230, 20, 20)        # rød ramme rundt boksen som skal dømmes
MARKOR_BREDDE = 3
MARKOR_LUFT = 3               # piksler utenfor boksen, så sifrene ikke dekkes

STD_MARGIN = 60.0             # kontekstmargin i PDF-punkt
STD_MAKS_PX = 1400            # utsnitt bredere enn dette skaleres ned
STD_JOBBER = max(1, min(8, (os.cpu_count() or 2) - 1))

MANIFEST_FELT = [
    "nr", "utsnitt", "fil", "dok_nr", "side", "klasse", "kilde", "conf",
    "paddle_rec", "label_id", "dekkere",
    "x0", "y0", "x1", "y1", "bredde_pt", "hoyde_pt", "elongation",
    "kortside_pt", "langside_pt",
    "har_tokens", "n_siffer", "siffer_run", "har_fnr_kandidat",
    "har_desimal_naer", "rec_min", "rec_min_linje",
    "ocr_tekst", "ocr_linje", "ocr_blokk",
    "utsnitt_bredde", "utsnitt_hoyde", "m_x0", "m_y0", "m_x1", "m_y1",
    "vurdering",
]


# ── Utvalg ────────────────────────────────────────────────────

def velg_bokser(ds, treff_utvalg, seed=42, maks_bom=None, kilder=None):
    """Alle BOM + et deterministisk utvalg dekkende bokser.

    Returnerer (valgte, statistikk). «Dekkende» er TREFF og SLURV under ett:
    begge dekker en fasit-boks, og begge koster recall om VLM-en sier nei.
    SLURV skilles ikke ut her — klassen følger med i manifestet.
    """
    kilder = {k.lower() for k in kilder} if kilder else None
    i_scope = [p for p in ds.pred
               if kilder is None or p["kilde"].lower() in kilder]

    bom = [p for p in i_scope if p["klasse"] == "BOM"]
    dekkende = [p for p in i_scope if p["klasse"] != "BOM"]

    # Sorter før trekning: utvalget skal være identisk mellom kjøringer
    # uansett radrekkefølge i resultat-CSV-en.
    nokkel = lambda p: (p["navn"], p["side"], p["px"][1], p["px"][0], p["kilde"])
    bom.sort(key=nokkel)
    dekkende.sort(key=nokkel)

    rng = random.Random(seed)
    valgt_bom = bom
    if maks_bom is not None and maks_bom < len(bom):
        valgt_bom = sorted(rng.sample(bom, maks_bom), key=nokkel)

    valgt_dekkende = dekkende
    if treff_utvalg is not None and treff_utvalg < len(dekkende):
        valgt_dekkende = sorted(rng.sample(dekkende, treff_utvalg), key=nokkel)

    stat = {
        "n_bom_total": len(bom),
        "n_bom_eksportert": len(valgt_bom),
        "n_dekkende_total": len(dekkende),
        "n_dekkende_eksportert": len(valgt_dekkende),
        "bom_faktor": (len(bom) / len(valgt_bom)) if valgt_bom else 0.0,
        "treff_faktor": ((len(dekkende) / len(valgt_dekkende))
                         if valgt_dekkende else 0.0),
        "seed": seed,
        "kilder": sorted(kilder) if kilder else "alle",
    }
    return valgt_bom + valgt_dekkende, stat


# ── OCR-kontekst ──────────────────────────────────────────────

def _linjer(tokens):
    """Grupperer tokens i tekstlinjer, ovenfra og ned.

    OCR-en gir løse ord. En linje er tokens som overlapper hverandre
    vertikalt — samme forestilling som _ocr_grunn i filter_felles bygger på,
    men her materialisert som lesbar tekst.
    """
    rest = sorted((t for t in tokens if t.tekst.strip()), key=lambda t: t.y0)
    linjer = []
    for t in rest:
        for lin in linjer:
            h = min(lin["y1"] - lin["y0"], t.y1 - t.y0)
            if min(lin["y1"], t.y1) - max(lin["y0"], t.y0) > 0.3 * max(h, 1.0):
                lin["tokens"].append(t)
                lin["y0"] = min(lin["y0"], t.y0)
                lin["y1"] = max(lin["y1"], t.y1)
                break
        else:
            linjer.append({"y0": t.y0, "y1": t.y1, "tokens": [t]})
    for lin in linjer:
        lin["tokens"].sort(key=lambda t: t.x0)
        lin["tekst"] = " ".join(t.tekst for t in lin["tokens"])
    linjer.sort(key=lambda l: l["y0"])
    return linjer


def _ocr_kontekst(tokens, rekt, n_linjer=0):
    """(tekst i boksen, tekst på linjen, blokk av nabolinjer).

    rekt er boksen i det ROTERTE pikselrommet — samme rom tokens ligger i.
    Linjen er tokens som overlapper boksens vertikale bånd; blokken er
    n_linjer over og under i tillegg. Ett fødselsnummer kan være brutt over
    et linjeskift, og i tabeller står ledeteksten en linje eller to over —
    linjen alene er ikke alltid nok kontekst.
    """
    x0, y0, x1, y1 = rekt
    hoyde = max(y1 - y0, 1.0)
    i_boks, i_linje = [], []
    for t in tokens:
        if not t.tekst.strip():
            continue
        # Vertikal overlapp med boksen, målt som andel av boksens høyde
        v = min(y1, t.y1) - max(y0, t.y0)
        if v <= 0.3 * min(hoyde, max(t.y1 - t.y0, 1.0)):
            continue
        i_linje.append(t)
        if min(x1, t.x1) - max(x0, t.x0) > 0:
            i_boks.append(t)
    i_boks.sort(key=lambda t: t.x0)
    i_linje.sort(key=lambda t: t.x0)
    boks_tekst = " ".join(t.tekst for t in i_boks)
    linje_tekst = " ".join(t.tekst for t in i_linje)

    blokk = ""
    if n_linjer > 0 and tokens:
        linjer = _linjer(tokens)
        midt = max(range(len(linjer)),
                   key=lambda i: min(y1, linjer[i]["y1"]) - max(y0, linjer[i]["y0"]),
                   default=None) if linjer else None
        if midt is not None:
            lo = max(0, midt - n_linjer)
            blokk = "\n".join(l["tekst"] for l in linjer[lo:midt + n_linjer + 1])
    return boks_tekst, linje_tekst, blokk


# ── Eksport ───────────────────────────────────────────────────

def _jobb_for_fil(oppgave):
    """Renderer, cropper og lagrer alle boksene i ÉN PDF.

    Kjøres i egen prosess, så den tar bare enkle verdier inn og gir enkle
    verdier ut — ingen Datasett, ingen delt tilstand. label_id og dekkere er
    slått opp i foreldreprosessen og ligger ferdig på hver prediksjon.
    Advarsler samles og skrives av forelderen, ellers ville de flettet seg
    inn i hverandre på skjermen.
    """
    (navn, preds, mappe, utsnitt_mappe, margin_x, margin_y, full_bredde,
     fra_toppen, maks_px, ocr_mappe, roter, n_linjer) = oppgave

    sti = os.path.join(mappe, navn)
    if not os.path.isfile(sti):
        return [], [f"Finner ikke {sti}, hopper over"], len(preds)
    try:
        dok = fitz.open(sti)
    except Exception as e:
        return [], [f"Kunne ikke åpne {navn}: {e!r}"], len(preds)

    cache = _ocr_les_cache(ocr_mappe, navn) if ocr_mappe else None
    per_side = defaultdict(list)
    for p in preds:
        per_side[p["side"]].append(p)

    rader, advarsler, droppet = [], [], 0
    for si in sorted(per_side):
        side_pred = per_side[si]
        if not 1 <= si <= len(dok):
            droppet += len(side_pred)
            continue
        bilde = _render_side(dok, si)
        w0, h0 = bilde.width, bilde.height
        # bw/bh = 0 er mot-fasit-sentinellen: koordinatene er allerede i
        # render-rommet (pt × SKALA), så skalering og rotasjon regnes mot
        # den faktiske renderingen.
        bw = side_pred[0]["bw"] or w0
        bh = side_pred[0]["bh"] or h0
        sx, sy = w0 / bw, h0 / bh

        # Samme rotasjon som pipelinen OCR-et med: uten den står teksten
        # på tvers i utsnittet av en liggende skanning, og både VLM-en og
        # et menneske leser den dårlig.
        k, tokens = 0, []
        if cache:
            rotasjoner, tokens_per_side = cache
            if si <= len(rotasjoner):
                k = (rotasjoner[si - 1] or 0) if roter else 0
                tokens = tokens_per_side[si - 1]
        if k:
            bilde = bilde.rotate(90 * k, expand=True)

        mx = margin_x * SKALA
        m_opp, m_ned = margin_y[0] * SKALA, margin_y[1] * SKALA
        for p in side_pred:
            px = p["px"]
            r = _rekt_frem([px[0] * sx, px[1] * sy, px[2] * sx, px[3] * sy],
                           k, w0, h0)
            # Bokser helt utenfor den rendrede siden (en label tegnet forbi
            # sidekanten, avvikende CropBox e.l.) hoppes over — én rar rad
            # skal ikke velte en eksport på titusener av bokser.
            if (r[2] <= 0 or r[3] <= 0
                    or r[0] >= bilde.width or r[1] >= bilde.height):
                droppet += 1
                advarsler.append(
                    f"{navn} s{si}: boks utenfor siden "
                    f"({r[0]:.0f},{r[1]:.0f},{r[2]:.0f},{r[3]:.0f} "
                    f"i {bilde.width}x{bilde.height}) — hoppet over")
                continue
            venstre = 0 if full_bredde else max(0, int(r[0] - mx))
            hoyre = (bilde.width if full_bredde
                     else min(bilde.width, int(r[2] + mx)))
            topp = 0 if fra_toppen else max(0, int(r[1] - m_opp))
            boks = (venstre, topp, hoyre,
                    min(bilde.height, int(r[3] + m_ned)))
            if boks[2] <= boks[0] or boks[3] <= boks[1]:
                droppet += 1
                continue
            ut = bilde.crop(boks).convert("RGB")

            m = [r[0] - boks[0], r[1] - boks[1],
                 r[2] - boks[0], r[3] - boks[1]]
            if maks_px and ut.width > maks_px:
                f = maks_px / ut.width
                ut = ut.resize((maks_px, max(1, int(ut.height * f))),
                               Image.LANCZOS)
                m = [v * f for v in m]
            # Rammen skaleres med bildet. En 3-pikslers strek forsvinner i
            # et helsides utsnitt, og da vet ikke modellen hva spørsmålet
            # gjelder — luften vokser med streken så sifrene ikke dekkes.
            strek = max(MARKOR_BREDDE, round(ut.width / 400))
            luft = strek + 2
            m = [max(0, m[0] - luft), max(0, m[1] - luft),
                 min(ut.width - 1, m[2] + luft),
                 min(ut.height - 1, m[3] + luft)]
            if m[2] <= m[0] or m[3] <= m[1]:
                droppet += 1
                advarsler.append(f"{navn} s{si}: markør uten areal etter "
                                 f"klipping — hoppet over")
                continue
            ImageDraw.Draw(ut).rectangle(m, outline=MARKOR, width=strek)

            base = os.path.splitext(os.path.basename(navn))[0]
            filnavn = f"{p['_nr']:05d}_{p['klasse']}_{base}_s{si}.png"
            ut.save(os.path.join(utsnitt_mappe, filnavn))

            i_boks = i_linje = i_blokk = ""
            if tokens:
                # Tokens ligger i resultat-CSV-ens pikselrom, rotert.
                # _rekt_frem er lineær, så rotasjonen kan gjøres rett på
                # CSV-koordinatene i stedet for å skalere frem og tilbake
                # via renderingen.
                i_boks, i_linje, i_blokk = _ocr_kontekst(
                    tokens, _rekt_frem(list(px), k, bw, bh), n_linjer)

            rader.append({
                "nr": p["_nr"], "utsnitt": filnavn, "fil": navn,
                "dok_nr": p["dok_nr"], "side": si, "klasse": p["klasse"],
                "kilde": p["kilde"],
                "conf": p["conf"] if p["conf"] is not None else "",
                "paddle_rec": (p["paddle_rec"]
                               if p["paddle_rec"] is not None else ""),
                "label_id": p["_label_id"], "dekkere": p["_dekkere"],
                "x0": round(px[0], 1), "y0": round(px[1], 1),
                "x1": round(px[2], 1), "y1": round(px[3], 1),
                "bredde_pt": round(p["w"], 1), "hoyde_pt": round(p["h"], 1),
                "elongation": round(p["elongation"], 2),
                "kortside_pt": round(p["kortside"], 1),
                "langside_pt": round(p["langside"], 1),
                "har_tokens": _tall(p.get("har_tokens")),
                "n_siffer": _tall(p.get("n_siffer")),
                "siffer_run": _tall(p.get("siffer_run")),
                "har_fnr_kandidat": _tall(p.get("har_fnr_kandidat")),
                "har_desimal_naer": _tall(p.get("har_desimal_naer")),
                "rec_min": _tall(p.get("rec_min")),
                "rec_min_linje": _tall(p.get("rec_min_linje")),
                "ocr_tekst": i_boks, "ocr_linje": i_linje,
                "ocr_blokk": i_blokk,
                "utsnitt_bredde": ut.width, "utsnitt_hoyde": ut.height,
                "m_x0": round(m[0], 1), "m_y0": round(m[1], 1),
                "m_x1": round(m[2], 1), "m_y1": round(m[3], 1),
                "vurdering": "",
            })
    dok.close()
    return rader, advarsler, droppet


def eksporter(ds, valgte, mappe, ut_mappe, margin_x=STD_MARGIN,
              margin_y=STD_MARGIN, full_bredde=False, fra_toppen=False,
              maks_px=STD_MAKS_PX, ocr_mappe=None, roter=True, jobber=1,
              n_linjer=0):
    """Renderer, cropper og skriver manifest. Returnerer manifest-radene.

    Margenen er asymmetrisk fordi konteksten er det: det som avgjør om et
    tall er et fnr eller en koordinat, står nesten alltid som en ledetekst
    ELLER PÅ SAMME LINJE («Koordinat N», «Konto», «gnr … bnr …»), ikke over
    eller under. Bredde kjøper altså mer enn høyde per piksel. Med
    full_bredde tas hele sidebredden, og margin_x ignoreres.

    fra_toppen tar alt fra sidens overkant og ned til boksen. I skjemaer og
    tabeller står ledeteksten som avgjør hva kolonnen inneholder helt oppe,
    langt utenfor enhver rimelig margin — den kan ikke nås med margin_y uten
    å ta like mye søppel under boksen på kjøpet.

    margin_y kan være ett tall (symmetrisk) eller et par (opp, ned) —
    i tabeller står den avslørende ledeteksten over boksen, ikke under,
    så oppover-margin er mer verdt enn nedover.

    jobber deler arbeidet per DOKUMENT over flere prosesser. Rendering i 300
    dpi er ren CPU, og hvert dokument åpnes uansett bare én gang, så det er
    den naturlige delelinjen — og den holder én PDF i én prosess.
    """
    try:
        margin_y = (float(margin_y[0]), float(margin_y[1]))
    except TypeError:
        margin_y = (float(margin_y), float(margin_y))
    utsnitt_mappe = os.path.join(ut_mappe, "utsnitt")
    os.makedirs(utsnitt_mappe, exist_ok=True)

    # Manifestrekkefølge: samme side samlet, så en menneskelig gjennomgang av
    # utsnittene leser dokumentet ovenfra og ned. Nummereringen settes her,
    # før arbeidet fordeles, så den er uavhengig av hvem som blir ferdig først.
    valgte = sorted(valgte, key=lambda p: (p["navn"], p["side"],
                                           p["px"][1], p["px"][0], p["kilde"]))
    per_fil = defaultdict(list)
    for nr, p in enumerate(valgte, 1):
        p["_nr"] = nr
        if "_label_id" not in p:
            p["_label_id"] = ";".join(
                i for i in (ds.fasit_bokser[j]["label_id"]
                            for j in p["dekker"]) if i)
            p["_dekkere"] = (min(ds.dekning_foer[j] for j in p["dekker"])
                             if p["dekker"] else 0)
        per_fil[p["navn"]].append(p)

    oppgaver = [(navn, per_fil[navn], mappe, utsnitt_mappe, margin_x, margin_y,
                 full_bredde, fra_toppen, maks_px, ocr_mappe, roter,
                 n_linjer)
                for navn in sorted(per_fil)]
    print(f"\n  {len(valgte)} bokser i {len(oppgaver)} dokumenter"
          f"  ({jobber} parallelle prosesser)")

    rader, advarsler, droppet = [], [], 0
    t0 = time.monotonic()
    n_fil = 0

    def fremdrift(nye):
        nonlocal n_fil, droppet
        r, a, d = nye
        rader.extend(r)
        advarsler.extend(a)
        droppet += d
        n_fil += 1
        if n_fil % 25 == 0 or n_fil == len(oppgaver):
            gaatt = time.monotonic() - t0
            fart = n_fil / gaatt if gaatt else 0
            igjen = (len(oppgaver) - n_fil) / fart if fart else 0
            print(f"    {n_fil:>6}/{len(oppgaver)} dok  {gaatt:6.0f}s  "
                  f"{fart:5.1f} dok/s  {len(rader):>6} utsnitt  "
                  f"ETA {igjen:5.0f}s", flush=True)

    if jobber <= 1:
        for oppgave in oppgaver:
            fremdrift(_jobb_for_fil(oppgave))
    else:
        with ProcessPoolExecutor(max_workers=jobber) as pool:
            for nye in pool.map(_jobb_for_fil, oppgaver, chunksize=1):
                fremdrift(nye)

    for melding in advarsler:
        print(f"  ⚠ {melding}")
    if droppet:
        print(f"  ⚠ {droppet} bokser droppet — manglende PDF, ugyldig "
              f"sidetall eller tomt utsnitt")

    rader.sort(key=lambda r: r["nr"])
    manifest_sti = os.path.join(ut_mappe, "manifest.csv")
    with open(manifest_sti, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FELT, extrasaction="ignore")
        w.writeheader()
        w.writerows(rader)
    print(f"\n  {len(rader)} utsnitt i {utsnitt_mappe}")
    print(f"  Manifest: {manifest_sti}  ({len(rader)} rader)")
    return rader


def _tall(v):
    return "" if v is None else v


# ── CLI ───────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Eksporterer PNG-utsnitt av foreslåtte sladdebokser + "
                    "manifest, som inngang til VLM-dømming (vlm_dommer.py).")
    p.add_argument("--res-csv", default=None,
                   help="Resultat-CSV fra modellen (ikke nødvendig med "
                        "--mot-fasit)")
    p.add_argument("--mot-fasit", action="store_true", dest="mot_fasit",
                   help="Eksporter FASIT-boksene (labels) i stedet for "
                        "modellens prediksjoner — VLM-en som fasit-revisor. "
                        "Alle labels tas med (ingen utvalgsfaktorer); "
                        "--kjorte-liste begrenser til en dokumentliste. "
                        "NB: rotasjon og OCR-kontekst krever --ocr-cache "
                        "med cache for dokumentene.")
    p.add_argument("--fasit-csv", required=True, help="Labels-CSV (fasit)")
    p.add_argument("--mappe", required=True, help="Mappe med PDF-dokumentene")
    p.add_argument("--ut-mappe", required=True, help="Mappe for utsnitt+manifest")

    p.add_argument("--treff-utvalg", type=int, default=1000, metavar="N",
                   help="Antall dekkende bokser (TREFF/SLURV) å trekke. "
                        "0 = ingen (ren BOM-kalibrering), -1 = alle. "
                        "Default 1000.")
    p.add_argument("--maks-bom", type=int, default=None, metavar="N",
                   help="Ta bare N tilfeldige BOM-bokser (default: alle)")
    p.add_argument("--seed", type=int, default=42, help="Trekningsseed (42)")
    p.add_argument("--kilde", nargs="+", default=None, metavar="KILDE",
                   help="Begrens til disse kildene (yolo/paddle/begge)")

    p.add_argument("--margin", type=float, default=STD_MARGIN, metavar="PT",
                   help=f"Kontekstmargin rundt boksen i punkt "
                        f"(default {STD_MARGIN:g}). Overstyres per akse av "
                        f"--margin-x/--margin-y.")
    p.add_argument("--margin-x", type=float, default=None, metavar="PT",
                   help="Margin til side. Ledeteksten som avslører hva tallet "
                        "er («Koordinat N», «Konto», «gnr»), står som regel "
                        "på samme linje — bredde er derfor mer verdt enn "
                        "høyde. 200-250 dekker typisk hele tekstlinjen.")
    p.add_argument("--margin-y", type=float, default=None, metavar="PT",
                   help="Margin opp/ned. 90 gir ~2 tekstlinjer over og under, "
                        "nok til å se en kolonneoverskrift.")
    p.add_argument("--margin-opp", type=float, default=None, metavar="PT",
                   help="Margin kun oppover — overstyrer --margin-y/--margin "
                        "over boksen. Ledetekster og kolonneoverskrifter står "
                        "OVER tallet; mer margin opp enn ned gir konteksten "
                        "uten like mye søppel under.")
    p.add_argument("--margin-ned", type=float, default=None, metavar="PT",
                   help="Margin kun nedover — overstyrer --margin-y/--margin "
                        "under boksen.")
    p.add_argument("--fra-toppen", action="store_true",
                   help="Ta med alt fra sidens overkant og ned til boksen. "
                        "Ledetekster og kolonneoverskrifter i skjemaer står "
                        "for høyt til å nås med --margin-y. NB: gir høye "
                        "utsnitt og mange bildetokens — hev "
                        "OLLAMA_CONTEXT_LENGTH tilsvarende.")
    p.add_argument("--full-bredde", action="store_true",
                   help="Ta hele sidebredden i stedet for --margin-x. Gir "
                        "alltid hele linjen med, på bekostning av oppløsning "
                        "etter nedskalering — vurder å heve --maks-px.")
    p.add_argument("--maks-px", type=int, default=STD_MAKS_PX, metavar="PX",
                   help=f"Skaler ned utsnitt bredere enn dette "
                        f"(default {STD_MAKS_PX}). 0 = ingen skalering.")
    p.add_argument("--ocr-cache", default=None, metavar="STI",
                   help="OCR-cache-mappe ($SLADD_CACHE/uttrekk_N/ocr). Gir "
                        "ocr_tekst/ocr_linje i manifestet (inngang til "
                        "tekst-armen i vlm_dommer) og retter opp sider som "
                        "er skannet på tvers.")
    p.add_argument("--jobber", type=int, default=STD_JOBBER, metavar="N",
                   help=f"Parallelle prosesser, én PDF om gangen per prosess "
                        f"(default {STD_JOBBER} på denne maskinen). "
                        f"Rendering i 300 dpi er CPU-bundet, så dette er den "
                        f"eneste spaken som betyr noe for eksporttiden.")
    p.add_argument("--ocr-linjer", type=int, default=0, metavar="N",
                   help="Ta med N OCR-linjer over og under boksens linje i "
                        "kolonnen ocr_blokk. Et fødselsnummer kan være brutt "
                        "over et linjeskift, og i tabeller står ledeteksten "
                        "en linje eller to over. 2 er et fornuftig sted å "
                        "begynne. Krever --ocr-cache.")
    p.add_argument("--ikke-roter", dest="roter", action="store_false",
                   help="Ikke rett opp siden selv om OCR-cachen har rotasjon")

    p.add_argument("--kjorte-liste", default=None, metavar="FIL",
                   help="Fil med dokumentene modellen har kjørt på — samme "
                        "semantikk som i filter_review/filter_sweep. Bruk "
                        "rs_<KODE>.txt for én rettsstiftelsestype.")
    p.add_argument("--kriterium", default=STD_KRITERIUM, choices=sorted(KRITERIER),
                   help=f"Matcheregel for dekning (default {STD_KRITERIUM})")
    p.add_argument("--terskel", type=float, default=STD_TERSKEL,
                   help=f"Overlapp-terskel (default {STD_TERSKEL})")
    p.add_argument("--slurv-faktor", type=float, default=STD_SLURV_FAKTOR,
                   help=f"SLURV-grense (default {STD_SLURV_FAKTOR})")
    p.add_argument("--inkluder-ulabelte", action="store_true", default=True,
                   help="(default på) Kjørte dokumenter uten fasit-rader er "
                        "gjennomgått med null fnr — prediksjoner der er BOM")
    p.add_argument("--ekskluder-ulabelte", dest="inkluder_ulabelte",
                   action="store_false",
                   help="Gammel oppførsel: bare dokumenter med fasit-rader")
    a = p.parse_args()

    kjorte = les_kjorte_dok(a.kjorte_liste) if a.kjorte_liste else None
    if a.mot_fasit:
        # VLM-en som fasit-revisor: hver label er en sladd et menneske har
        # gjort. Dommen «nei» er da en påstand om label-støy — utfallet
        # leses i gjennomgang_label.md (❌-radene bærer label_id).
        ds = None
        rader_f, ekskludert, _kol = les_fasit_rader(a.fasit_csv)
        navn_for = {}
        for fn in sorted(os.listdir(a.mappe)):
            if fn.lower().endswith(".pdf"):
                nr = dok_nr(fn)
                if nr is not None:
                    navn_for.setdefault(nr, fn)
        valgte, mangler = [], 0
        for r in rader_f:
            if kjorte is not None and r["dok_nr"] not in kjorte:
                continue
            navn = navn_for.get(r["dok_nr"])
            if navn is None:
                mangler += 1
                continue
            x0, y0, x1, y1 = r["boks"]
            valgte.append({
                "navn": navn, "dok_nr": r["dok_nr"], "side": r["side"],
                "px": [x0 * SKALA, y0 * SKALA, x1 * SKALA, y1 * SKALA],
                "bw": 0, "bh": 0, "klasse": "FASIT", "kilde": "fasit",
                "conf": None, "paddle_rec": None,
                "w": r["w"], "h": r["h"], "kortside": r["kortside"],
                "langside": r["langside"], "elongation": r["elongation"],
                "_label_id": (r["rad"].get("id") or "").strip(),
                "_dekkere": "",
            })
        stat = {"mot_fasit": True,
                "n_bom_total": 0, "n_bom_eksportert": 0, "bom_faktor": 0.0,
                "n_dekkende_total": len(valgte),
                "n_dekkende_eksportert": len(valgte), "treff_faktor": 1.0,
                "seed": a.seed, "kilder": "fasit"}
        print(f"MOT-FASIT — eksporterer labels, ikke prediksjoner")
        print(f"  Labels lest:    {len(rader_f)}"
              + (f"  (ekskludert: "
                 + ", ".join(f"{k}={v}" for k, v in sorted(ekskludert.items()))
                 + ")" if ekskludert else ""))
        if kjorte is not None:
            print(f"  --kjorte-liste: {len(valgte) + mangler} innenfor listen")
        if mangler:
            print(f"  Uten PDF i {a.mappe}: {mangler} — hoppet over")
        print(f"  Eksporteres:    {len(valgte)}")
    else:
        if not a.res_csv:
            p.error("--res-csv er påkrevd (unntatt med --mot-fasit)")
        fasit = les_fasit(a.fasit_csv)
        pred = les_prediksjoner(a.res_csv)
        ds = bygg_datasett(fasit, pred, terskel=a.terskel,
                           slurv_faktor=a.slurv_faktor,
                           inkluder_ulabelte=a.inkluder_ulabelte,
                           kjorte_dok=kjorte, kriterium=a.kriterium)
        skriv_oppsummering(ds)

        treff_utvalg = None if a.treff_utvalg < 0 else a.treff_utvalg
        valgte, stat = velg_bokser(ds, treff_utvalg, seed=a.seed,
                                   maks_bom=a.maks_bom, kilder=a.kilde)
        print(f"\nUtvalg for VLM-dømming:")
        print(f"  BOM:      {stat['n_bom_eksportert']:>6} av "
              f"{stat['n_bom_total']}   (faktor {stat['bom_faktor']:.2f})")
        print(f"  Dekkende: {stat['n_dekkende_eksportert']:>6} av "
              f"{stat['n_dekkende_total']}   (faktor {stat['treff_faktor']:.2f})")
    if not valgte:
        print("  Ingenting å eksportere.")
        return

    os.makedirs(a.ut_mappe, exist_ok=True)
    margin_x = a.margin if a.margin_x is None else a.margin_x
    margin_y = a.margin if a.margin_y is None else a.margin_y
    margin_opp = margin_y if a.margin_opp is None else a.margin_opp
    margin_ned = margin_y if a.margin_ned is None else a.margin_ned
    rader = eksporter(ds, valgte, a.mappe, a.ut_mappe, margin_x=margin_x,
                      margin_y=(margin_opp, margin_ned),
                      full_bredde=a.full_bredde,
                      fra_toppen=a.fra_toppen,
                      maks_px=a.maks_px or 0, ocr_mappe=a.ocr_cache,
                      roter=a.roter, jobber=a.jobber,
                      n_linjer=a.ocr_linjer)

    # Faktorene telles på nytt fra det som FAKTISK ble skrevet: bokser i
    # dokumenter uten PDF eller med feil sidetall faller ut over, og da ville
    # de opprinnelige faktorene skalert tapsestimatet feil.
    skrevet = defaultdict(int)
    for r in rader:
        skrevet["bom" if r["klasse"] == "BOM" else "dekkende"] += 1
    stat["n_bom_eksportert"] = skrevet["bom"]
    stat["n_dekkende_eksportert"] = skrevet["dekkende"]
    stat["bom_faktor"] = (stat["n_bom_total"] / skrevet["bom"]
                          if skrevet["bom"] else 0.0)
    stat["treff_faktor"] = (stat["n_dekkende_total"] / skrevet["dekkende"]
                            if skrevet["dekkende"] else 0.0)
    stat.update({
        "margin_x_pt": margin_x, "margin_y_pt": [margin_opp, margin_ned],
        "full_bredde": a.full_bredde, "fra_toppen": a.fra_toppen,
        "maks_px": a.maks_px,
        "res_csv": os.path.abspath(a.res_csv) if a.res_csv else None,
        "fasit_csv": os.path.abspath(a.fasit_csv),
        "mappe": os.path.abspath(a.mappe),
        "kjorte_liste": (os.path.abspath(a.kjorte_liste)
                         if a.kjorte_liste else None),
        "kriterium": a.kriterium, "terskel": a.terskel,
        "slurv_faktor": a.slurv_faktor,
        "inkluder_ulabelte": a.inkluder_ulabelte,
        "n_fasit_i_scope": ds.n_fasit if ds else len(valgte),
        "n_dekket_foer": ds.dekket_foer if ds else None,
        "n_dok_i_scope": (len(ds.scope_dok) if ds
                          else len({p["dok_nr"] for p in valgte})),
    })
    with open(os.path.join(a.ut_mappe, "utvalg.json"), "w",
              encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)
    print(f"  Utvalgsfaktorer: {os.path.join(a.ut_mappe, 'utvalg.json')}")
    print("\n  Neste steg:  python utils/vlm_dommer.py --manifest "
          f"{os.path.join(a.ut_mappe, 'manifest.csv')} --url ... --modell ...")


if __name__ == "__main__":
    main()
