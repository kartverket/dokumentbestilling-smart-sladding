"""Evaluerer VLM-dommene mot fasit-klassene og gir kost/nytte-svaret.

Steg 3 av VLM-verifikator-piloten. Joiner dommer_<modus>.csv mot manifest.csv
og svarer på det ene tallet beslutningen henger på:

    hvor mange BOM-bokser får «nei» (gevinst: oversladdinger vi kan fjerne)
    mot hvor mange dekkende bokser får «nei» (tapsrisiko: ekte fnr)

Beslutningsregelen er den samme som for alle filtre i dette repoet: én tapt
fasit-boks må kjøpe minst 20 fjernede oversladdinger.

To ting gjør regnestykket mindre trivielt enn det ser ut:

1. UTVALGSSKJEVHET. Eksporten tar ALLE BOM, men bare et utvalg av de dekkende
   boksene. Rått sammenlignet ville tapssiden vært systematisk undervurdert.
   Faktorene i utvalg.json skalerer tapet opp til fullt uttrekk.

2. FÅ HENDELSER. Ser vi 0 eller 1 tap i utvalget, er punktestimatet «uendelig
   god» eller «veldig god» — og verdiløst. Derfor rapporteres også en øvre
   Poisson-grense (95 %) for tapsraten, og ov/tapt regnet mot DEN. Det er
   tallet som skal måles mot kostnad 20 før noe settes i produksjon.

Alle dekkende bokser som fikk «nei» skrives til tapt.csv med label_id, samme
format som filter_review — fasit kan være støy, og id-ene går rett inn i
ugyldige_labels.txt hvis dommen viser seg å ha rett.

Eksempel:
    python utils/vlm_evaluer.py \
        --manifest /data2/vlm/uttrekk6_kalibrering/manifest.csv \
        --dommer   /data2/vlm/uttrekk6_kalibrering/dommer_bilde.csv
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "app")))

# Samme sifferforveksling som pipelinen selv bruker når den leter etter
# fnr-kandidater — importert, ikke kopiert, så en endring i prod ikke
# etterlater en stille avvikende kopi her.
from paddle_ocr_model_fnr import finn_fnr

SVAR = ("ja", "nei", "usikker")
STD_KOSTNAD = 20.0


# ── Statistikk ────────────────────────────────────────────────

def poisson_ovre(k, tillit=0.95):
    """Øvre (95 %) grense for forventet antall når k er observert.

    Løser P(X <= k | lambda) = 1 - tillit ved halvering. For k = 0 gir dette
    ~3.0 — «tre-regelen»: ser du null hendelser i n forsøk, kan raten fortsatt
    være opptil 3/n. Uten dette leser man 0 tap i utvalget som «null tap», og
    det er nettopp den feilen et pilotresultat ikke tåler.
    """
    if k < 0:
        return 0.0
    maal = 1.0 - tillit

    def cdf(lam):
        if lam <= 0:
            return 1.0
        s, ledd = 0.0, math.exp(-lam)
        for i in range(k + 1):
            if i:
                ledd *= lam / i
            s += ledd
        return s

    lo, hi = 0.0, max(10.0, k + 10.0)
    while cdf(hi) > maal:
        hi *= 2
        if hi > 1e6:
            return hi
    for _ in range(200):
        midt = (lo + hi) / 2
        if cdf(midt) > maal:
            lo = midt
        else:
            hi = midt
    return (lo + hi) / 2


# ── Innlesing ─────────────────────────────────────────────────

# ── Regelbasislinje ───────────────────────────────────────────
# En dom fra en VLM er bare interessant hvis den slår regelen den skulle
# erstattet. Derfor kan --dommer peke på «regel:<navn>» i stedet for en fil:
# da syntetiseres dommene fra manifestets egen OCR-linje, og de går gjennom
# nøyaktig samme regnskap. Er tallene like, er modellen overflødig.

_FNR_ORD = re.compile(
    r"(f\s*[øo]dsels\s*n|pers\s*[.\s]*n|p\s*\.\s*nr|f\s*\.\s*nr|fnr|personnummer)",
    re.I)
_FEM_LOP = re.compile(r"(?<!\d)\d{5}(?!\d)")


def _fnr_kandidat(linje):
    """Har linja et 11-sifret løp med gyldig fødselsnummer-form?

    Bruker pipelinens egen finn_fnr(krev_mod11=False) — den arbeider på
    sifferposisjoner med lukekontroll, ikke på en streng der skilletegnene er
    fjernet. Forskjellen er ikke akademisk: i «030392S0000 Iflg fullmakt» blir
    S til 5 og løpet 03039250000 gyldig, men fjerner man mellomrommene limes
    det sammen med «1f1g» og grensesjekken feiler. Den boksen var et ekte
    fødselsnummer vi mistet.
    """
    return bool(finn_fnr(linje or "", krev_mod11=False))


def _fnr_ledetekst(linje):
    """Ledetekst for fødselsnummer + et femsifret løp på samme linje.

    Fanger dokumentene der fødselsdatoen er skrevet i en annen form enn
    DDMMÅÅ — «født: 1.2.1950 Personnummer: 00000», «f. 12/3-1950, pers.nr.
    00000». Da finnes det ingen ellevesifret rekke å finne, men ordet
    «personnummer» ved siden av fem sifre er et sterkt nok tegn i seg selv.
    """
    tekst = linje or ""
    return bool(_FNR_ORD.search(tekst)) and bool(_FEM_LOP.search(tekst))


def _har_desimal(tekst):
    return bool(re.search(r"\d[.,]\d", tekst or ""))


REGLER = {
    # Behold boksen hvis linja har et fnr-kandidat, ellers fjern den.
    "fnr-kandidat": lambda r: "ja" if _fnr_kandidat(r.get("ocr_linje")) else "nei",
    # Fjern boksen hvis tallet i den har desimalskille — desimalregelen.
    "desimal": lambda r: "nei" if _har_desimal(r.get("ocr_tekst")) else "ja",
    # Begge: fjern bare når linja mangler fnr-kandidat OG tallet har desimal.
    "fnr-kandidat+desimal": lambda r: (
        "nei" if not _fnr_kandidat(r.get("ocr_linje"))
        and _har_desimal(r.get("ocr_tekst")) else "ja"),
}


def dommer_fra_regel(manifest, navn):
    try:
        regel = REGLER[navn]
    except KeyError:
        raise SystemExit(f"ukjent regel {navn!r} — gyldige: "
                         + ", ".join(sorted(REGLER)))
    return [{"nr": r["nr"], "svar": regel(r), "tall": r.get("ocr_tekst", ""),
             "begrunnelse": f"regel:{navn}", "feil": "", "sekunder": ""}
            for r in manifest]


def les_csv(sti):
    with open(sti, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def join(manifest, dommer):
    """manifest-rader + dom, nøklet på nr. Returnerer (rader, udømte)."""
    dom_per_nr = {}
    for d in dommer:
        # Ved --gjenoppta kan samme nr stå flere ganger (et feilet forsøk
        # etterfulgt av et vellykket). Siste rad uten feil vinner.
        nr = d["nr"]
        if nr in dom_per_nr and dom_per_nr[nr].get("feil") and not d.get("feil"):
            dom_per_nr[nr] = d
        elif nr not in dom_per_nr:
            dom_per_nr[nr] = d
    rader, udomte = [], 0
    for m in manifest:
        d = dom_per_nr.get(m["nr"])
        if d is None:
            udomte += 1
            continue
        rad = dict(m)
        rad["svar"] = (d.get("svar") or "usikker").strip().lower()
        if rad["svar"] not in SVAR:
            rad["svar"] = "usikker"
        rad["sikkerhet"] = d.get("sikkerhet", "")
        rad["tall"] = d.get("tall", "")
        rad["begrunnelse"] = d.get("begrunnelse", "")
        rad["feil"] = d.get("feil", "")
        rad["sekunder"] = d.get("sekunder", "")
        # Sjekklisten fra vlm_dommer — modellens egen avlesning. Uten disse
        # har --fnr-overstyring ingenting å arbeide på.
        for felt in ("linjen", "sifre_paa_linjen", "dato_gyldig", "holdepunkt"):
            if felt in d:
                rad[felt] = d[felt]
        rader.append(rad)
    return rader, udomte


# ── Delmengder ────────────────────────────────────────────────

_OPS = (("!=", lambda a, b: a != b), (">=", lambda a, b: a >= b),
        ("<=", lambda a, b: a <= b), ("=", lambda a, b: a == b),
        (">", lambda a, b: a > b), ("<", lambda a, b: a < b))


def parse_vilkaar(spec):
    """«har_fnr_kandidat=1» eller «conf>=0.7» -> predikat på en manifestrad.

    Tall sammenlignes som tall når begge sider lar seg tolke som det; ellers
    som tekst. Tomme felt faller alltid ut — «ikke beregnet» er ikke det samme
    som null, og et OCR-trekk som mangler skal ikke gi utslag.
    """
    for tegn, op in _OPS:
        if tegn in spec:
            kol, _, verdi = spec.partition(tegn)
            kol, verdi = kol.strip(), verdi.strip()

            def test(rad, kol=kol, verdi=verdi, op=op):
                raa = (rad.get(kol) or "").strip()
                if raa == "":
                    return False
                try:
                    return op(float(raa), float(verdi))
                except ValueError:
                    return op(raa, verdi)
            return test
    raise ValueError(f"ugyldig vilkår {spec!r} — bruk f.eks. har_fnr_kandidat=1")


def skriv_deling(rader, kolonne, skriv=print):
    """Forvirringsmatrise per verdi av en manifestkolonne.

    Poenget er å finne DELMENGDER der modellen er trygg. En regel som er
    farlig globalt kan være gratis i én familie — nøyaktig samme mønster som
    rettsstiftelsesprofilene i filter_sweep.
    """
    tab = defaultdict(lambda: defaultdict(int))
    for r in rader:
        verdi = (r.get(kolonne) or "").strip() or "(tom)"
        tab[verdi][(_gruppe(r), r["svar"])] += 1
    skriv(f"\nDELT ETTER {kolonne}")
    skriv(f"  {kolonne:>16} {'BOM nei':>9} {'BOM tot':>9} "
          f"{'DEK nei':>9} {'DEK tot':>9} {'DEK nei%':>9}")
    for verdi in sorted(tab):
        t = tab[verdi]
        bn = t[("BOM", "nei")]
        bt = sum(t[("BOM", s)] for s in SVAR)
        dn = t[("DEKKENDE", "nei")]
        dt = sum(t[("DEKKENDE", s)] for s in SVAR)
        skriv(f"  {verdi:>16} {bn:>9} {bt:>9} {dn:>9} {dt:>9} "
              f"{(dn / dt * 100 if dt else 0):>8.1f}%")
    skriv("  Let etter en rad med høy «BOM nei» og «DEK nei%» nær null — "
          "det er en profil.")


def advar_om_skjevt_utvalg(manifest, rader, skriv=print):
    """Sier fra når de dømte radene ikke ser ut som et tilfeldig utsnitt.

    Utvalgsfaktorene i utvalg.json forutsetter at de dekkende boksene som er
    dømt er et TILFELDIG utsnitt av populasjonen sin. Kjører man med
    --nr-fil på en liste over rader en tidligere prompt bommet på, er den
    forutsetningen brutt — settet er motstander-valgt, og oppskaleringen blir
    tull. Det er ikke synlig i tallene, så det må sies høyt.
    """
    if not rader or len(rader) >= 0.9 * len(manifest):
        return False
    def andel(rows):
        n_bom = sum(1 for r in rows if r["klasse"] == "BOM")
        return n_bom / len(rows) if rows else 0.0
    ventet, faktisk = andel(manifest), andel(rader)
    if ventet <= 0 or abs(faktisk - ventet) <= 0.15:
        return False
    skriv("")
    skriv("  " + "!" * 66)
    skriv(f"  ADVARSEL: de {len(rader)} dømte radene er {faktisk*100:.0f} % BOM, "
          f"mens manifestet er {ventet*100:.0f} %.")
    skriv("  Utvalget ser ikke tilfeldig ut — brukte du --nr-fil eller en "
          "hard-liste?")
    skriv("  Da gjelder IKKE utvalgsfaktorene, og gevinst, tap og ov/tapt "
          "under er meningsløse.")
    skriv("  Forvirringsmatrisen er fortsatt lesbar som «hva gjorde modellen "
          "med akkurat disse».")
    skriv("  " + "!" * 66)
    return True


def _gruppe(rad):
    """BOM = oversladding, DEKKENDE = TREFF eller SLURV (dekker en fasit-boks)."""
    return "BOM" if rad["klasse"] == "BOM" else "DEKKENDE"


# ── Rapport ───────────────────────────────────────────────────

def skriv_matrise(rader, skriv=print):
    tab = defaultdict(lambda: defaultdict(int))
    for r in rader:
        tab[r["klasse"]][r["svar"]] += 1
    skriv("\nFORVIRRINGSMATRISE  (rad = fasit-klasse, kolonne = VLM-dom)")
    skriv(f"  {'klasse':>10} {'ja':>8} {'nei':>8} {'usikker':>9} {'sum':>8}")
    for klasse in sorted(tab):
        rad = tab[klasse]
        n = sum(rad.values())
        skriv(f"  {klasse:>10} {rad['ja']:>8} {rad['nei']:>8} "
              f"{rad['usikker']:>9} {n:>8}")
    skriv(f"  {'SUM':>10} "
          + " ".join(f"{sum(tab[k][s] for k in tab):>8}" for s in SVAR)
          + f" {len(rader):>8}")


def skriv_per_kilde(rader, skriv=print):
    tab = defaultdict(lambda: defaultdict(int))
    for r in rader:
        tab[r["kilde"]][(_gruppe(r), r["svar"])] += 1
    skriv("\nPER KILDE")
    skriv(f"  {'kilde':>10} {'BOM nei':>9} {'BOM tot':>9} {'andel':>7}"
          f" {'DEK nei':>9} {'DEK tot':>9} {'andel':>7}")
    for kilde in sorted(tab):
        t = tab[kilde]
        bn = t[("BOM", "nei")]
        bt = sum(t[("BOM", s)] for s in SVAR)
        dn = t[("DEKKENDE", "nei")]
        dt = sum(t[("DEKKENDE", s)] for s in SVAR)
        skriv(f"  {kilde:>10} {bn:>9} {bt:>9} "
              f"{(bn / bt * 100 if bt else 0):>6.1f}% "
              f"{dn:>9} {dt:>9} {(dn / dt * 100 if dt else 0):>6.1f}%")


def _faktorer(utvalg, n_bom_domt, n_dek_domt):
    """Oppskaleringsfaktorer fra DØMTE rader til hele uttrekket.

    Uten totaler i utvalg.json faller vi tilbake til 1.0, altså «det du ser
    er alt som finnes» — det er den eneste antakelsen som ikke lyver når
    grunnlaget mangler.
    """
    bom_tot = float(utvalg.get("n_bom_total") or 0)
    dek_tot = float(utvalg.get("n_dekkende_total") or 0)
    bom = bom_tot / n_bom_domt if bom_tot and n_bom_domt else 1.0
    dek = dek_tot / n_dek_domt if dek_tot and n_dek_domt else 1.0
    return bom, dek


def skriv_sikkerhetskurve(rader, utvalg, kostnad=STD_KOSTNAD, skriv=print):
    """Gevinst og tap som funksjon av hvor sikker modellen må være.

    En fast feilrate er ikke en skjebne hvis modellen vet når den er i tvil.
    Kurven svarer på om det finnes et driftspunkt: en terskel der «nei» er
    sjelden nok på dekkende bokser til å komme over kostnadskravet, uten at
    gevinsten forsvinner. Finnes ingen slik rad, er sikkerheten ukalibrert —
    og da er den heller ikke verdt å bruke.
    """
    har = [r for r in rader if str(r.get("sikkerhet", "")).strip() != ""]
    if not har:
        skriv("\nSIKKERHETSKURVE: dommene mangler «sikkerhet» — kjør på nytt "
              "med en vlm_dommer som spør om det.")
        return
    bom_faktor, treff_faktor = _faktorer(
        utvalg, sum(1 for r in har if _gruppe(r) == "BOM"),
        sum(1 for r in har if _gruppe(r) == "DEKKENDE"))

    def tall(r):
        try:
            return float(r["sikkerhet"])
        except (TypeError, ValueError):
            return -1.0

    skriv(f"\nSIKKERHETSKURVE  ({len(har)} av {len(rader)} dommer har "
          f"sikkerhet)")
    skriv(f"  {'terskel':>8} {'BOM nei':>9} {'gevinst':>9} {'DEK nei':>9} "
          f"{'tap':>9} {'ov/tapt':>9}")
    for terskel in (0, 50, 60, 70, 80, 90, 95, 99, 100):
        bn = sum(1 for r in har if _gruppe(r) == "BOM"
                 and r["svar"] == "nei" and tall(r) >= terskel)
        dn = sum(1 for r in har if _gruppe(r) == "DEKKENDE"
                 and r["svar"] == "nei" and tall(r) >= terskel)
        gevinst = bn * bom_faktor
        tap_ovre = poisson_ovre(dn) * treff_faktor
        forhold = gevinst / tap_ovre if tap_ovre > 0 else float("inf")
        merke = "  ← over kostnadskravet" if forhold >= kostnad and bn else ""
        skriv(f"  {terskel:>8} {bn:>9} {gevinst:>9.0f} {dn:>9} "
              f"{tap_ovre:>9.0f} {forhold:>9.1f}{merke}")
    skriv("  «tap» er den konservative øvre grensen, samme som i regnskapet.")


def regnskap(rader, utvalg, kostnad=STD_KOSTNAD, usikker_fjerner=False,
             skriv=print):
    """Gevinst mot tapsrisiko, skalert opp til fullt uttrekk."""
    fjern = {"nei", "usikker"} if usikker_fjerner else {"nei"}

    bom = [r for r in rader if _gruppe(r) == "BOM"]
    dek = [r for r in rader if _gruppe(r) == "DEKKENDE"]
    bom_fjern = [r for r in bom if r["svar"] in fjern]
    dek_fjern = [r for r in dek if r["svar"] in fjern]

    # Faktorene regnes mot antall DØMTE, ikke antall eksporterte. Ved en
    # delvis kjøring er bare en del av manifestet dømt, og bruker man
    # eksport-faktorene blir tapet skalert opp mens gevinsten ikke blir det —
    # ov/tapt ser da katastrofalt ut av rene bokføringsgrunner.
    bom_faktor, treff_faktor = _faktorer(utvalg, len(bom), len(dek))

    gevinst = len(bom_fjern) * bom_faktor
    tap_pkt = len(dek_fjern) * treff_faktor
    tap_ovre = poisson_ovre(len(dek_fjern)) * treff_faktor

    skriv("\n" + "=" * 68)
    skriv(f"REGNSKAP   (usikker teller som "
          f"{'FJERNET' if usikker_fjerner else 'BEHOLDT'}, kostnad {kostnad:g})")
    skriv("=" * 68)
    skriv(f"  Dømte bokser:            {len(rader)}   "
          f"({len(bom)} BOM, {len(dek)} dekkende)")
    skriv(f"  Oppskalering BOM:        {bom_faktor:>6.2f}   "
          f"({len(bom)} dømt av {utvalg.get('n_bom_total', '?')} i uttrekket)")
    skriv(f"  Oppskalering dekkende:   {treff_faktor:>6.2f}   "
          f"({len(dek)} dømt av {utvalg.get('n_dekkende_total', '?')})")
    skriv("")
    skriv(f"  GEVINST  BOM med «nei»:       {len(bom_fjern):>6} i utvalget"
          f"  →  {gevinst:>8.0f} i uttrekket")
    skriv(f"           andel av BOM:        "
          f"{(len(bom_fjern) / len(bom) * 100 if bom else 0):>6.1f}%")
    skriv(f"  TAP      dekkende med «nei»:  {len(dek_fjern):>6} i utvalget"
          f"  →  {tap_pkt:>8.1f} i uttrekket (punktestimat)")
    skriv(f"           95 % øvre grense:    "
          f"{poisson_ovre(len(dek_fjern)):>6.1f} i utvalget"
          f"  →  {tap_ovre:>8.1f} i uttrekket")

    flere_dekkere = sum(1 for r in dek_fjern
                        if _int(r.get("dekkere")) > 1)
    if flere_dekkere:
        skriv(f"           NB: {flere_dekkere} av dem dekker en fasit-boks med "
              f"flere dekkere — de er ikke")
        skriv(f"           nødvendigvis tap (en annen boks kan dekke videre). "
              f"Tapstallet er en øvre grense.")

    skriv("")
    if tap_pkt > 0:
        skriv(f"  ov/tapt (punktestimat):  {gevinst / tap_pkt:>8.1f}")
    else:
        skriv(f"  ov/tapt (punktestimat):  {'∞':>8}   (null tap observert)")
    nedre = gevinst / tap_ovre if tap_ovre > 0 else float("inf")
    skriv(f"  ov/tapt (konservativt):  {nedre:>8.1f}   "
          f"← dette tallet måles mot kostnad {kostnad:g}")
    skriv("")
    if not dek:
        skriv("  DOM: kan ikke avgjøres — ingen dekkende bokser i utvalget.")
        skriv("       Dette er en ren kalibreringskjøring (prompt + hastighet).")
    elif nedre >= kostnad:
        skriv(f"  DOM: BESTÅTT — selv i det konservative tilfellet kjøper hvert "
              f"tapte fnr {nedre:.0f} oversladdinger.")
    elif gevinst / max(tap_pkt, 1e-9) >= kostnad:
        skriv(f"  DOM: USIKKER — punktestimatet holder, men den øvre "
              f"tapsgrensen gjør det ikke.")
        skriv(f"       Døm flere dekkende bokser (større --treff-utvalg) før "
              f"dette avgjøres.")
    else:
        skriv(f"  DOM: IKKE BESTÅTT — for dyrt ved kostnad {kostnad:g}.")
    skriv("")
    skriv("  Alle dekkende «nei» skal dømmes manuelt før tallet tros: fasit "
          "kan være støy.")
    skriv("  Se tapt.csv.")
    return {"gevinst": gevinst, "tap_pkt": tap_pkt, "tap_ovre": tap_ovre,
            "ov_per_tapt": nedre, "n_bom_nei": len(bom_fjern),
            "n_dek_nei": len(dek_fjern), "n_bom": len(bom), "n_dek": len(dek)}


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


# ── Manifester for manuell gjennomgang ────────────────────────

TAPT_FELT = ["nr", "fil", "side", "label_id", "grunn", "kilde", "conf",
             "klasse", "dekkere_foer", "vlm_tall", "ocr_linje",
             "elongation", "kortside_pt", "langside_pt",
             "pred_bredde_pt", "pred_hoyde_pt", "utsnitt", "vurdering"]


def skriv_manifest(rader, sti, felt=TAPT_FELT):
    with open(sti, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=felt, extrasaction="ignore")
        w.writeheader()
        for r in rader:
            w.writerow({
                "nr": r["nr"], "fil": r["fil"], "side": r["side"],
                "label_id": r.get("label_id", ""),
                "grunn": r.get("begrunnelse", ""),
                "kilde": r.get("kilde", ""), "conf": r.get("conf", ""),
                "klasse": r.get("klasse", ""),
                "dekkere_foer": r.get("dekkere", ""),
                "vlm_tall": r.get("tall", ""),
                "ocr_linje": r.get("ocr_linje", ""),
                "elongation": r.get("elongation", ""),
                "kortside_pt": r.get("kortside_pt", ""),
                "langside_pt": r.get("langside_pt", ""),
                "pred_bredde_pt": r.get("bredde_pt", ""),
                "pred_hoyde_pt": r.get("hoyde_pt", ""),
                "utsnitt": r.get("utsnitt", ""),
                "vurdering": "",
            })


def main():
    p = argparse.ArgumentParser(
        description="Joiner VLM-dommer mot fasit-klasser og regner "
                    "gevinst mot tapsrisiko ved kostnad 20.")
    p.add_argument("--manifest", required=True, help="manifest.csv fra vlm_eksport")
    p.add_argument("--dommer", required=True,
                   help="dommer_*.csv fra vlm_dommer, ELLER «regel:<navn>» for "
                        "å måle en ren regel på manifestets OCR-linje i "
                        "stedet. Gyldige: "
                        + ", ".join(f"regel:{n}" for n in sorted(REGLER))
                        + ". Bruk den til å se om modellen slår regelen den "
                          "skulle erstattet.")
    p.add_argument("--utvalg", default=None,
                   help="utvalg.json (default: ved siden av manifestet)")
    p.add_argument("--ut-mappe", default=None,
                   help="Mappe for tapt.csv/gevinst.csv (default: ved dommene)")
    p.add_argument("--kostnad", type=float, default=STD_KOSTNAD,
                   help=f"Oversladdinger per tapt fnr (default {STD_KOSTNAD:g})")
    p.add_argument("--fnr-overstyring", action="store_true",
                   help="Overstyr dommen til «ja» når modellens egen "
                        "avskrift ELLER PaddleOCRs linje fra manifestet "
                        "inneholder et gyldig 11-sifret fødselsnummer-løp. "
                        "Modellen leser bedre enn den slutter — dette lar "
                        "koden ta slutningen, på alt vi vet om linjen.")
    p.add_argument("--uten-ledetekst", action="store_true",
                   help="Skru av ledetekst-vernet i --fnr-overstyring, så "
                        "bare ekte 11-sifrede løp verner boksen. Bruk den "
                        "til å måle hva ledetekst-delen koster i gevinst.")
    p.add_argument("--min-sikkerhet", type=float, default=None, metavar="V",
                   help="Godta bare «nei» der modellen oppga sikkerhet ≥ V. "
                        "Resten nedgraderes til «usikker», altså behold. "
                        "Bruk sikkerhetskurven til å velge V.")
    p.add_argument("--del-etter", default=None, metavar="KOLONNE",
                   help="Vis forvirringsmatrisen per verdi av en "
                        "manifestkolonne (f.eks. har_fnr_kandidat, kilde, "
                        "har_desimal_naer). Leter etter delmengder der "
                        "modellen er trygg.")
    p.add_argument("--bare", nargs="+", default=None, metavar="VILKÅR",
                   help="Regn bare på rader som oppfyller alle vilkårene, "
                        "f.eks. --bare har_fnr_kandidat=0 conf<0.7. "
                        "Utvalgsfaktorene gjelder fortsatt: utvalget av "
                        "dekkende bokser var tilfeldig og uavhengig av "
                        "trekkene, så en delmengde av det er fortsatt et "
                        "tilfeldig utvalg av sin egen delpopulasjon.")
    p.add_argument("--usikker-fjerner", action="store_true",
                   help="Regn «usikker» som «nei». Default er å beholde — "
                        "en usikker dom er ikke bevis for oversladding.")
    a = p.parse_args()

    manifest = les_csv(a.manifest)
    if a.dommer.startswith("regel:"):
        dommer = dommer_fra_regel(manifest, a.dommer.split(":", 1)[1])
        print(f"Basislinje «{a.dommer}» — ingen modell involvert")
    else:
        dommer = les_csv(a.dommer)
    utvalg_sti = a.utvalg or os.path.join(
        os.path.dirname(os.path.abspath(a.manifest)), "utvalg.json")
    utvalg = {}
    if os.path.isfile(utvalg_sti):
        with open(utvalg_sti, encoding="utf-8") as f:
            utvalg = json.load(f)
    else:
        print(f"  ⚠ Fant ikke {utvalg_sti} — regner med faktor 1.0, "
              f"altså som om HELE uttrekket er dømt. Tapsestimatet blir "
              f"for lavt hvis dekkende bokser er et utvalg.")

    rader, udomte = join(manifest, dommer)
    print(f"Dømte {len(rader)} av {len(manifest)} manifestrader"
          + (f"  ({udomte} mangler dom)" if udomte else ""))
    n_feil = sum(1 for r in rader if r.get("feil"))
    if n_feil:
        print(f"  ⚠ {n_feil} rader hadde feil/uparsbart svar — talt som usikker")

    if a.fnr_overstyring:
        # Arbeidsdelingen piloten har avdekket: modellen LESER svært godt og
        # SLUTTER dårlig. Den skrev av «Kari Nordmann 010190 00000», fant
        # elleve sifre og gyldig dato — og svarte likevel nei fordi tallet
        # «lignet et organisasjonsnummer». Regelen er triviell å anvende i
        # kode, så vi anvender den på modellens egen avskrift i stedet for
        # å be den om å huske den.
        # Fire uavhengige lesninger av samme linje: modellens sifferrekke,
        # modellens avskrift, og PaddleOCRs egen linje og blokk fra
        # manifestet. Ett gyldig 11-sifret løp i ÉN av dem er nok til å verne
        # boksen. Paddle har ofte lest datohalvdelen som VLM-en ikke fikk med
        # seg i utsnittet, og motsatt — de feiler sjelden på samme sted.
        n, mangler = 0, 0
        ledetekst = not a.uten_ledetekst
        for r in rader:
            kilder = [r.get("sifre_paa_linjen"), r.get("linjen"),
                      r.get("ocr_linje"), r.get("ocr_blokk")]
            kilder = [k for k in (t.strip() for t in kilder if t) if k]
            if not kilder:
                mangler += 1
                continue
            verner = (any(_fnr_kandidat(k) for k in kilder)
                      or (ledetekst and any(_fnr_ledetekst(k) for k in kilder)))
            if r["svar"] != "ja" and verner:
                r["svar"] = "ja"
                n += 1
        print(f"  --fnr-overstyring: {n} dommer endret til «ja» fordi "
              f"modellens egen avskrift har et gyldig 11-sifret fnr-løp")
        if mangler:
            print(f"    ({mangler} rader har verken sjekklistefelt eller "
                  f"OCR-tekst — de er urørt)")

    if a.min_sikkerhet is not None:
        n = 0
        for r in rader:
            if r["svar"] != "nei":
                continue
            try:
                sikker = float(r.get("sikkerhet", ""))
            except (TypeError, ValueError):
                sikker = -1.0
            if sikker < a.min_sikkerhet:
                r["svar"] = "usikker"
                n += 1
        print(f"  --min-sikkerhet {a.min_sikkerhet:g}: {n} «nei»-dommer "
              f"nedgradert til «usikker» (beholdes)")

    if a.bare:
        vilkaar = [parse_vilkaar(v) for v in a.bare]
        foer = len(rader)
        rader = [r for r in rader if all(t(r) for t in vilkaar)]
        print(f"  Delmengde «{' '.join(a.bare)}»: {len(rader)} av {foer} rader")
        if not rader:
            print("  Ingen rader igjen — sjekk kolonnenavn og verdier.")
            return

    skjevt = advar_om_skjevt_utvalg(manifest, rader)
    skriv_matrise(rader)
    skriv_per_kilde(rader)
    if a.del_etter:
        skriv_deling(rader, a.del_etter)
    skriv_sikkerhetskurve(rader, utvalg, kostnad=a.kostnad)
    res = regnskap(rader, utvalg, kostnad=a.kostnad,
                   usikker_fjerner=a.usikker_fjerner)
    if skjevt:
        print("  NB: regnskapet over hviler på et utvalg som ikke ser "
              "tilfeldig ut — se advarselen øverst.")

    ut_mappe = a.ut_mappe or (
        os.path.dirname(os.path.abspath(a.manifest)) if a.dommer.startswith("regel:")
        else os.path.dirname(os.path.abspath(a.dommer)))
    os.makedirs(ut_mappe, exist_ok=True)
    fjern = {"nei", "usikker"} if a.usikker_fjerner else {"nei"}
    tapt = [r for r in rader if _gruppe(r) == "DEKKENDE" and r["svar"] in fjern]
    gevinst = [r for r in rader if _gruppe(r) == "BOM" and r["svar"] in fjern]
    tapt.sort(key=lambda r: (r["fil"], int(r["side"]), int(r["nr"])))
    gevinst.sort(key=lambda r: (r["fil"], int(r["side"]), int(r["nr"])))

    tapt_sti = os.path.join(ut_mappe, "tapt.csv")
    skriv_manifest(tapt, tapt_sti)
    gevinst_sti = os.path.join(ut_mappe, "gevinst.csv")
    skriv_manifest(gevinst, gevinst_sti)
    print(f"\n  {len(tapt):>6} rader i {tapt_sti}"
          f"   ← disse skal dømmes manuelt (label_id følger med)")
    print(f"  {len(gevinst):>6} rader i {gevinst_sti}"
          f"   ← stikkprøve på gevinsten")

    with open(os.path.join(ut_mappe, "oppsummering.json"), "w",
              encoding="utf-8") as f:
        json.dump({**res, "kostnad": a.kostnad,
                   "usikker_fjerner": a.usikker_fjerner,
                   "n_domt": len(rader), "n_manifest": len(manifest),
                   "n_feil": n_feil}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
