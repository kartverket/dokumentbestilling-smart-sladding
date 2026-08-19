"""
Felles innlesing, matching, filtrering og måling for filter_sweep og filter_review.

Kjernen er **fasit-sentrisk måling**: en filterkonfigurasjon vurderes ikke etter
hvor mange prediksjoner den fjerner, men etter hvor mange *fasit-bokser som
mister all dekning*. En riktig prediksjon som fjernes mens en annen prediksjon
fortsatt dekker samme fasit-boks koster ingenting — feltet er fremdeles sladdet.

Sentrale begreper:

  TREFF     prediksjon som dekker ≥ terskel av en fasit-boks, og som er
            noenlunde på størrelse med den
  SLURV     dekker en fasit-boks, men er mye større enn nødvendig
            (areal > slurv_faktor × dekket fasit-areal) — sladder for mye
  BOM       treffer ingen fasit-boks = ren oversladding

  tapt      fasit-bokser som mister all dekning etter filtrering  ← recall-tap
  ov.fj     BOM-prediksjoner fjernet                              ← gevinst
  red.fj    dekkende prediksjoner fjernet uten at noen fasit-boks gikk tapt
            (redundant dekning — gratis gevinst)

Dokument-scope: prediksjoner på dokumenter som ikke finnes i fasit-CSV-en er
per default *utenfor scope*. Uten det blir alle oversladdingstall oppblåst av
dokumenter som aldri ble labelt.
"""

import csv
import os
import re
import sys
from collections import defaultdict

_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

try:
    from config import PDF_DPI
except ImportError:  # kjøring utenfor repoet
    PDF_DPI = 300

SKALA = PDF_DPI / 72.0   # PDF-punkt → piksel

STD_TERSKEL = 0.15       # min dekning av fasit-boks for at prediksjonen "treffer"
STD_SLURV_FAKTOR = 3.0   # pred-areal > faktor × dekket fasit-areal ⇒ SLURV


# ── Geometri ─────────────────────────────────────────────────

def dok_nr(navn):
    m = re.match(r"0*(\d+)", os.path.basename(navn))
    return int(m.group(1)) if m else None


def overlapp(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return (ix1 - ix0) * (iy1 - iy0) if (ix1 > ix0 and iy1 > iy0) else 0.0


def areal(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


# ── Innlesing ────────────────────────────────────────────────

def les_fasit(sti):
    """Leser fasit-labels (ACCEPTED + manuell, ekskluderer REJECTED).

    Returnerer dict: (dok_nr, side) -> [(x0, y0, x1, y1), ...] i PDF-punkt.
    """
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

    Hver prediksjon får både pikselkoordinater (tegning), normaliserte
    koordinater (matching) og dimensjoner i PDF-punkt (filtrering).
    """
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
            w_pt = abs(x1 - x0) / SKALA
            h_pt = abs(y1 - y0) / SKALA
            if w_pt <= 0 or h_pt <= 0 or bw <= 0 or bh <= 0:
                continue
            ratio = w_pt / h_pt
            norm = (x0 / bw, y0 / bh, x1 / bw, y1 / bh)
            pred.append({
                "navn": navn, "side": side, "dok_nr": dok_nr(navn),
                "bw": bw, "bh": bh,
                "px": (x0, y0, x1, y1),
                "norm": norm,
                "norm_areal": areal(norm),
                "w": w_pt, "h": h_pt,
                "ratio": ratio,
                "elongation": max(ratio, 1 / ratio),
                "kortside": min(w_pt, h_pt),
                "langside": max(w_pt, h_pt),
                "areal": w_pt * h_pt,
                "areal_px": abs(x1 - x0) * abs(y1 - y0),
                "kilde": kilde, "conf": conf,
            })
    return pred


def les_kjorte_dok(sti):
    """Leser en liste over dokumenter modellen har kjørt på.

    Én oppføring per linje; både filnavn (00123.pdf) og rene tall godtas.
    Tomme linjer og linjer som starter med # hoppes over.
    """
    dokumenter = set()
    with open(sti, encoding="utf-8-sig") as f:
        for linje in f:
            linje = linje.strip()
            if not linje or linje.startswith("#"):
                continue
            nr = dok_nr(linje)
            if nr is not None:
                dokumenter.add(nr)
    return dokumenter


def les_fasit_rader(sti, ekskluder=("REJECTED",)):
    """Leser fasit-labels som fulle rader, med geometri og metadata.

    I motsetning til les_fasit beholdes alle kolonner, slik at fordelingen av
    ml_status/type kan rapporteres. Brukes til å teste et filter DIREKTE mot
    saksbehandlernes sladdinger, uavhengig av modellens prediksjoner — da
    trengs ingen resultat-CSV, og alle labels kan vurderes.

    Geometrien regnes i PDF-punkt (som filtrene), pluss areal i piksel².
    Fasit-bokser har ingen conf, så conf-porten slår aldri inn: testen viser
    hva geometrireglene alene ville forkastet.
    """
    rader, forkastet = [], defaultdict(int)
    ekskluder = {e.strip().upper() for e in ekskluder}
    with open(sti, newline="", encoding="utf-8-sig") as f:
        leser = csv.DictReader(f)
        kolonner = leser.fieldnames or []
        for r in leser:
            status = (r.get("ml_status") or "").strip().upper()
            if status in ekskluder:
                forkastet[status or "(tom)"] += 1
                continue
            try:
                nr = int(r["fil_revisjon_id"])
                side = int(r["sidetall"])
                x, y = float(r["x"]), float(r["y"])
                w, h = float(r["width"]), float(r["height"])
            except (TypeError, ValueError, KeyError):
                forkastet["(ugyldig rad)"] += 1
                continue
            x0, x1 = sorted((x, x + w))
            y0, y1 = sorted((y, y + h))
            bw, bh = x1 - x0, y1 - y0
            if bw <= 0 or bh <= 0:
                forkastet["(null areal)"] += 1
                continue
            ratio = bw / bh
            rader.append({
                "dok_nr": nr, "side": side,
                "boks": (x0, y0, x1, y1),
                "w": bw, "h": bh,
                "kortside": min(bw, bh), "langside": max(bw, bh),
                "elongation": max(ratio, 1 / ratio),
                "areal": bw * bh,
                "areal_px": bw * bh * SKALA * SKALA,
                "conf": None,
                "ml_status": status or "(tom)",
                "type": (r.get("type") or "").strip() or "(tom)",
                "rad": r,
            })
    return rader, dict(forkastet), kolonner


# ── Datasett med fasit-sentrisk indeks ───────────────────────

class Datasett:
    """Prediksjoner og fasit-bokser koblet sammen via dekning.

    Felter:
      pred            prediksjoner innenfor scope (labelte dokumenter)
      utenfor         prediksjoner på dokumenter uten fasit (ekskludert)
      fasit_bokser    flat liste av dicts: {dok_nr, side, boks, norm_areal}
      dekning_foer    antall prediksjoner som dekker fasit_bokser[j]
      dekket_foer     antall fasit-bokser med minst én dekker
      n_bom           antall BOM-prediksjoner (= oversladdinger)
    """

    def __init__(self, pred, utenfor, fasit_bokser, dekning_foer,
                 terskel, slurv_faktor, n_fasit=None, navn="alle",
                 scope_dok=None, n_fasit_ukjort=0, n_dok_ukjort=0):
        self.pred = pred
        self.utenfor = utenfor
        self.fasit_bokser = fasit_bokser
        self.dekning_foer = dekning_foer
        self.terskel = terskel
        self.slurv_faktor = slurv_faktor
        self.navn = navn
        self.n_fasit_ukjort = n_fasit_ukjort
        self.n_dok_ukjort = n_dok_ukjort
        self.scope_dok = (scope_dok if scope_dok is not None
                          else {p["dok_nr"] for p in pred})
        # Ved dokument-splitt peker dekker-indeksene fortsatt inn i den
        # globale fasit-listen, men bare en delmengde er i scope.
        self.n_fasit = len(fasit_bokser) if n_fasit is None else n_fasit
        self.dekket_foer = sum(1 for d in dekning_foer if d > 0)
        self.n_bom = sum(1 for p in pred if p["klasse"] == "BOM")
        self.n_treff = sum(1 for p in pred if p["klasse"] == "TREFF")
        self.n_slurv = sum(1 for p in pred if p["klasse"] == "SLURV")
        self.n_dekkende = self.n_treff + self.n_slurv
        self.per_kilde = defaultdict(list)
        for p in pred:
            self.per_kilde[p["kilde"]].append(p)

    def kilder(self):
        return sorted({p["kilde"] for p in self.pred})


def bygg_datasett(fasit, pred, terskel=STD_TERSKEL,
                  slurv_faktor=STD_SLURV_FAKTOR, inkluder_ulabelte=False,
                  kjorte_dok=None):
    """Kobler prediksjoner mot fasit-bokser og klassifiserer hver prediksjon.

    Setter på hver prediksjon:
      p["dekker"]  liste av fasit-boks-indekser den dekker ≥ terskel
      p["klasse"]  "TREFF" | "SLURV" | "BOM"
      p["riktig"]  bakoverkompatibel bool (dekker noe = True)

    Scope er skjæringen mellom dokumenter som er LABELT og dokumenter som er
    KJØRT. Fasit på dokumenter modellen aldri kjørte på holdes utenfor — de er
    ikke bom, de er umålte, og tas de med blir recall kunstig lav.

    kjorte_dok: eksplisitt sett med kjørte dokumentnumre. Utelates det, antas
    dokumentene som forekommer i resultat-CSV-en. Da regnes et dokument der
    modellen ikke fant noe som helst som ukjørt, så recall blir marginalt
    optimistisk — oppgi listen for å fjerne den tvetydigheten.
    """
    labelte_dok = {nr for (nr, _side) in fasit}
    kjorte = (set(kjorte_dok) if kjorte_dok is not None
              else {p["dok_nr"] for p in pred})
    scope_dok = labelte_dok & kjorte

    # Sidestørrelse i punkt, utledet fra pikselstørrelsen i resultat-CSV-en
    side_str = {}
    for p in pred:
        key = (p["dok_nr"], p["side"])
        if key not in side_str:
            side_str[key] = (p["bw"] / SKALA, p["bh"] / SKALA)

    # Flat, normalisert fasit-liste + oppslag (dok, side) -> [indekser]
    fasit_bokser = []
    per_side = defaultdict(list)
    n_fasit_ukjort = 0
    for (nr, si), bokser in sorted(fasit.items()):
        if nr not in scope_dok:
            n_fasit_ukjort += len(bokser)
            continue
        pw, ph = side_str.get((nr, si), (595, 842))   # fallback A4
        for (x0, y0, x1, y1) in bokser:
            n = (x0 / pw, y0 / ph, x1 / pw, y1 / ph)
            per_side[(nr, si)].append(len(fasit_bokser))
            fasit_bokser.append({
                "dok_nr": nr, "side": si,
                "boks": (x0, y0, x1, y1),
                "norm": n,
                "norm_areal": areal(n),
            })

    innenfor, utenfor = [], []
    for p in pred:
        if p["dok_nr"] in labelte_dok or inkluder_ulabelte:
            innenfor.append(p)
        else:
            utenfor.append(p)

    dekning_foer = [0] * len(fasit_bokser)
    for p in innenfor:
        pn = p["norm"]
        dekker = []
        dekket_areal = 0.0
        for j in per_side.get((p["dok_nr"], p["side"]), ()):
            fb = fasit_bokser[j]
            fa = fb["norm_areal"]
            if fa <= 0:
                continue
            if overlapp(pn, fb["norm"]) / fa >= terskel:
                dekker.append(j)
                dekket_areal += fa
                dekning_foer[j] += 1
        p["dekker"] = dekker
        if not dekker:
            p["klasse"] = "BOM"
        elif p["norm_areal"] > slurv_faktor * dekket_areal:
            p["klasse"] = "SLURV"
        else:
            p["klasse"] = "TREFF"
        p["riktig"] = bool(dekker)

    for p in utenfor:
        p["dekker"] = []
        p["klasse"] = "BOM"
        p["riktig"] = False

    return Datasett(innenfor, utenfor, fasit_bokser, dekning_foer,
                    terskel, slurv_faktor, scope_dok=scope_dok,
                    n_fasit_ukjort=n_fasit_ukjort,
                    n_dok_ukjort=len(labelte_dok - kjorte))


def del_datasett(ds, dok_sett, navn):
    """Ny Datasett begrenset til dokumentene i dok_sett.

    Fasit-indeksene beholdes globale (p["dekker"] peker fortsatt riktig),
    men dekningstellingen bygges på nytt fra kun delmengdens prediksjoner.
    """
    pred = [p for p in ds.pred if p["dok_nr"] in dok_sett]
    n_fasit = sum(1 for fb in ds.fasit_bokser if fb["dok_nr"] in dok_sett)
    dekning = [0] * len(ds.fasit_bokser)
    for p in pred:
        for j in p["dekker"]:
            dekning[j] += 1
    return Datasett(pred, [], ds.fasit_bokser, dekning, ds.terskel,
                    ds.slurv_faktor, n_fasit=n_fasit, navn=navn,
                    scope_dok=dok_sett & ds.scope_dok)


def splitt_dokumenter(ds, andel, seed=42):
    """Deler dokumentene i (trening, holdout). Splitten er på DOKUMENT, ikke
    prediksjon — ellers lekker samme side inn i begge settene."""
    import random
    # Del over HELE scope, ikke bare dokumenter med prediksjoner — ellers
    # faller dokumenter der modellen ikke fant noe ut av begge settene.
    dokumenter = sorted(ds.scope_dok)
    stokket = list(dokumenter)
    random.Random(seed).shuffle(stokket)
    n_test = max(1, round(len(stokket) * andel))
    test = set(stokket[:n_test])
    trening = set(stokket[n_test:])
    return (del_datasett(ds, trening, "trening"),
            del_datasett(ds, test, "holdout"))


def pareto_front(rader, maal=lambda r: (r.m.tapt, r.m.ov_fj)):
    """Ikke-dominerte rader: for hvert nivå av `tapt`, høyest `ov.fj`.

    En rad er dominert hvis en annen taper like lite eller mindre OG fjerner
    minst like mange oversladdinger.
    """
    front, beste = [], -1
    for r in sorted(rader, key=lambda r: (maal(r)[0], -maal(r)[1])):
        _tapt, ov = maal(r)
        if ov > beste:
            front.append(r)
            beste = ov
    return front


# ── Filtrering ───────────────────────────────────────────────

FILTER_PARAMETRE = ("min_elongation", "maks_elongation",
                    "maks_hoyde", "min_hoyde", "maks_bredde", "min_bredde",
                    "min_kortside", "maks_kortside",
                    "min_langside", "maks_langside",
                    "maks_areal", "min_areal_px", "conf_terskel")


def filter_grunner(p, min_elongation=None, maks_elongation=None,
                   maks_hoyde=None, min_hoyde=None,
                   maks_bredde=None, min_bredde=None,
                   min_kortside=None, maks_kortside=None,
                   min_langside=None, maks_langside=None,
                   maks_areal=None, min_areal_px=None, conf_terskel=None):
    """Grunner til at boksen filtreres bort (tom liste = beholdes).

    min_areal_px er i PIKSEL² for å matche MIN_BOKS_AREAL i config.py, og
    sjekkes FØR conf-porten: i prod gjelder støygrensen alle bokser, også
    høy-confidence og «begge».
    """
    grunner = []
    if min_areal_px is not None and p["areal_px"] < min_areal_px:
        grunner.append(f"areal {p['areal_px']:.0f}px² < {min_areal_px:g}")
        return grunner
    # Høy confidence → stol på prediksjonen, hopp over resten av geometrien
    if conf_terskel is not None and p.get("conf") is not None \
            and p["conf"] >= conf_terskel:
        return []
    if min_elongation is not None and p["elongation"] < min_elongation:
        grunner.append(f"elong {p['elongation']:.1f} < {min_elongation:g}")
    if maks_elongation is not None and p["elongation"] > maks_elongation:
        grunner.append(f"elong {p['elongation']:.1f} > {maks_elongation:g}")
    if maks_hoyde is not None and p["h"] > maks_hoyde:
        grunner.append(f"høyde {p['h']:.0f} > {maks_hoyde:g}")
    if min_hoyde is not None and p["h"] < min_hoyde:
        grunner.append(f"høyde {p['h']:.1f} < {min_hoyde:g}")
    if maks_bredde is not None and p["w"] > maks_bredde:
        grunner.append(f"bredde {p['w']:.0f} > {maks_bredde:g}")
    if min_bredde is not None and p["w"] < min_bredde:
        grunner.append(f"bredde {p['w']:.1f} < {min_bredde:g}")
    if min_kortside is not None and p["kortside"] < min_kortside:
        grunner.append(f"kortside {p['kortside']:.1f} < {min_kortside:g}")
    if maks_kortside is not None and p["kortside"] > maks_kortside:
        grunner.append(f"kortside {p['kortside']:.1f} > {maks_kortside:g}")
    if min_langside is not None and p["langside"] < min_langside:
        grunner.append(f"langside {p['langside']:.1f} < {min_langside:g}")
    if maks_langside is not None and p["langside"] > maks_langside:
        grunner.append(f"langside {p['langside']:.1f} > {maks_langside:g}")
    if maks_areal is not None and p["areal"] > maks_areal:
        grunner.append(f"areal {p['areal']:.0f} > {maks_areal:g}")
    return grunner


def er_filtrert(p, min_elongation=None, maks_elongation=None,
                maks_hoyde=None, min_hoyde=None,
                maks_bredde=None, min_bredde=None,
                min_kortside=None, maks_kortside=None,
                min_langside=None, maks_langside=None,
                maks_areal=None, min_areal_px=None, conf_terskel=None):
    """Rask variant av filter_grunner som ikke bygger tekst."""
    if min_areal_px is not None and p["areal_px"] < min_areal_px:
        return True
    if conf_terskel is not None and p["conf"] is not None \
            and p["conf"] >= conf_terskel:
        return False
    if min_elongation is not None and p["elongation"] < min_elongation:
        return True
    if maks_elongation is not None and p["elongation"] > maks_elongation:
        return True
    if maks_hoyde is not None and p["h"] > maks_hoyde:
        return True
    if min_hoyde is not None and p["h"] < min_hoyde:
        return True
    if maks_bredde is not None and p["w"] > maks_bredde:
        return True
    if min_bredde is not None and p["w"] < min_bredde:
        return True
    if min_kortside is not None and p["kortside"] < min_kortside:
        return True
    if maks_kortside is not None and p["kortside"] > maks_kortside:
        return True
    if min_langside is not None and p["langside"] < min_langside:
        return True
    if maks_langside is not None and p["langside"] > maks_langside:
        return True
    if maks_areal is not None and p["areal"] > maks_areal:
        return True
    return False


def lag_filter(**kwargs):
    """Predikat p -> bool for én felles filterkonfigurasjon."""
    return lambda p: er_filtrert(p, **kwargs)


def lag_filter_per_kilde(per_kilde, kun_kilde=None):
    """Predikat p -> bool med egne parametre per kilde.

    per_kilde:  {kilde: {filterparametre}}. Kilder som mangler filtreres ikke.
    kun_kilde:  begrens filtreringen til denne kilden (andre beholdes urørt).
    """
    def _fjernes(p):
        kilde = p["kilde"].lower()
        if kun_kilde is not None and kilde != kun_kilde.lower():
            return False
        kw = per_kilde.get(kilde)
        return er_filtrert(p, **kw) if kw else False
    return _fjernes


def parse_per_kilde(spec_liste):
    """Parser "kilde:e=V,h=V,b=V,c=V,..." → {kilde: {parametre}}.

    e/emaks = min/maks elongation, h/hmin = maks/min høyde (pt),
    b/bmin = maks/min bredde (pt), a = maks areal (pt²),
    amin = min areal (px², som MIN_BOKS_AREAL), c = conf-terskel.
    """
    param_map = {
        "e": "min_elongation",      "emaks": "maks_elongation",
        "h": "maks_hoyde",          "hmin": "min_hoyde",
        "b": "maks_bredde",         "bmin": "min_bredde",
        "kmin": "min_kortside",     "kmaks": "maks_kortside",
        "lmin": "min_langside",     "lmaks": "maks_langside",
        "a": "maks_areal",          "amin": "min_areal_px",
        "c": "conf_terskel",
    }
    resultat = {}
    for spec in spec_liste:
        if ":" not in spec:
            raise ValueError(f"Ugyldig per-kilde-format: {spec!r} "
                             f"(forventet 'kilde:e=V,h=V,...')")
        kilde, param_str = spec.split(":", 1)
        kwargs = {}
        for bit in param_str.split(","):
            bit = bit.strip()
            if not bit:
                continue
            if "=" not in bit:
                raise ValueError(f"Ugyldig parameter: {bit!r} i {spec!r}")
            nøkkel, verdi = bit.split("=", 1)
            nøkkel = nøkkel.strip().lower()
            if nøkkel not in param_map:
                raise ValueError(f"Ukjent parameter {nøkkel!r} i {spec!r}. "
                                 f"Gyldige: {', '.join(param_map)}")
            kwargs[param_map[nøkkel]] = float(verdi)
        resultat[kilde.strip().lower()] = kwargs
    return resultat


# ── Måling ───────────────────────────────────────────────────

class Maaling:
    """Resultatet av å anvende én filterkonfigurasjon på et datasett."""

    __slots__ = ("tapt", "tapt_pst", "ov_fj", "ov_pst", "red_fj", "kritisk_fj",
                 "slurv_fj", "n_fj", "areal_fj", "ov_areal_fj", "dekket_etter",
                 "recall_etter", "pres_etter", "rik_etter", "ov_etter",
                 "netto", "ov_per_tapt", "tapte_bokser")

    def __init__(self, **kw):
        for navn in self.__slots__:
            setattr(self, navn, kw.get(navn))


def evaluer(ds, fjernes, kostnad=1.0, samle_tapte=False, kandidater=None):
    """Måler én filterkonfigurasjon fasit-sentrisk.

    Args:
        ds:       Datasett fra bygg_datasett.
        fjernes:  predikat p -> bool (True = boksen filtreres bort).
        kostnad:  hvor mange fjernede oversladdinger én tapt fasit-boks er
                  verdt. netto = ov.fj − kostnad × tapt.
        samle_tapte: ta vare på hvilke fasit-bokser som gikk tapt.
        kandidater: begrens predikatet til denne delmengden av ds.pred
                    (resten beholdes urørt). Gir raskere per-kilde-sweep.
    """
    tap_teller = defaultdict(int)
    fjernet_dekkende = []          # fjernede prediksjoner som dekker fasit
    ov_fj = slurv_fj = n_fj = 0
    areal_fj = ov_areal_fj = 0.0

    for p in (ds.pred if kandidater is None else kandidater):
        if not fjernes(p):
            continue
        n_fj += 1
        areal_fj += p["areal"]
        if p["dekker"]:
            fjernet_dekkende.append(p)
            if p["klasse"] == "SLURV":
                slurv_fj += 1
            for j in p["dekker"]:
                tap_teller[j] += 1
        else:
            ov_fj += 1
            ov_areal_fj += p["areal"]

    tapte = {j for j, c in tap_teller.items() if c == ds.dekning_foer[j]}
    kritisk_fj = sum(1 for p in fjernet_dekkende
                     if any(j in tapte for j in p["dekker"]))

    tapt = len(tapte)
    dekket_etter = ds.dekket_foer - tapt
    behold_tot = len(ds.pred) - n_fj
    rik_etter = ds.n_dekkende - len(fjernet_dekkende)

    return Maaling(
        tapt=tapt,
        tapt_pst=tapt / ds.dekket_foer * 100 if ds.dekket_foer else 0.0,
        ov_fj=ov_fj,
        ov_pst=ov_fj / ds.n_bom * 100 if ds.n_bom else 0.0,
        red_fj=len(fjernet_dekkende) - kritisk_fj,
        kritisk_fj=kritisk_fj,
        slurv_fj=slurv_fj,
        n_fj=n_fj,
        areal_fj=areal_fj,
        ov_areal_fj=ov_areal_fj,
        dekket_etter=dekket_etter,
        recall_etter=dekket_etter / ds.n_fasit * 100 if ds.n_fasit else 0.0,
        pres_etter=rik_etter / behold_tot * 100 if behold_tot else 0.0,
        rik_etter=rik_etter,
        ov_etter=behold_tot - rik_etter,
        netto=ov_fj - kostnad * tapt,
        ov_per_tapt=(ov_fj / tapt) if tapt else float("inf") if ov_fj else 0.0,
        tapte_bokser=sorted(tapte) if samle_tapte else None,
    )


def baseline(ds):
    """Måling uten filtrering — utgangspunktet."""
    return evaluer(ds, lambda p: False)


# ── Oppsummering ─────────────────────────────────────────────

def skriv_oppsummering(ds, skriv=print):
    """Skriver utgangspunktet: fasit-dekning, klassefordeling, scope."""
    b = baseline(ds)
    skriv(f"Fasit i scope: {ds.n_fasit} bokser "
          f"i {len({(f['dok_nr'], f['side']) for f in ds.fasit_bokser})} "
          f"(dok, side)-grupper")
    skriv(f"Prediksjoner: {len(ds.pred)} i scope"
          + (f"  ({len(ds.utenfor)} ekskludert — dokument mangler i fasit)"
             if ds.utenfor else ""))
    skriv(f"Scope:        {len(ds.scope_dok)} dokumenter "
          f"(labelt OG kjørt)")
    if ds.n_dok_ukjort:
        skriv(f"              {ds.n_dok_ukjort} labelte dokumenter er ikke "
              f"kjørt — {ds.n_fasit_ukjort} fasit-bokser holdt utenfor")

    per_kilde = defaultdict(lambda: [0, 0, 0, 0])
    for p in ds.pred:
        rad = per_kilde[p["kilde"]]
        rad[0] += 1
        rad[{"TREFF": 1, "SLURV": 2, "BOM": 3}[p["klasse"]]] += 1

    skriv("")
    skriv(f"  {'kilde':>10} {'antall':>8} {'TREFF':>8} {'SLURV':>8} "
          f"{'BOM':>8} {'BOM%':>7}")
    for k in sorted(per_kilde):
        n, t, s, bo = per_kilde[k]
        skriv(f"  {k:>10} {n:>8} {t:>8} {s:>8} {bo:>8} {bo/n*100:>6.1f}%")
    skriv(f"  {'SUM':>10} {len(ds.pred):>8} {ds.n_treff:>8} {ds.n_slurv:>8} "
          f"{ds.n_bom:>8} "
          f"{ds.n_bom/len(ds.pred)*100 if ds.pred else 0:>6.1f}%")

    udekket = ds.n_fasit - ds.dekket_foer
    skriv("")
    skriv(f"  Fasit-bokser dekket av minst én prediksjon: "
          f"{ds.dekket_foer} / {ds.n_fasit} "
          f"({b.recall_etter:.1f}% recall før filtrering)")
    skriv(f"  Aldri dekket (modellen bommet):             {udekket}")
    if ds.dekket_foer:
        skriv(f"  Snitt antall dekkere per dekket fasit-boks: "
              f"{sum(ds.dekning_foer) / ds.dekket_foer:.2f}"
              f"   ← duplikater; jo høyere, jo mer 'riktig fjernet' er gratis")
    skriv(f"  Presisjon (prediksjoner som treffer fasit): {b.pres_etter:.1f}%")
    return b
