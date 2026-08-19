"""
Fasit-sentrisk evaluering av filterkonfigurasjoner.

Måler hver konfigurasjon på det som faktisk betyr noe:

    tapt    = fasit-bokser som mister ALL dekning etter filtrering
              (én prediksjon fjernet mens en annen fortsatt dekker samme
              fasit-boks koster ingenting — feltet er fremdeles sladdet)
    ov.fj   = rene oversladdinger (BOM) fjernet
    red.fj  = dekkende prediksjoner fjernet uten tap — gratis gevinst
    recall% = andel fasit-bokser fortsatt dekket

Pareto-fronten koker de hundrevis av kombinasjonene ned til de få som ikke er
dominert av en annen: for hvert nivå av `tapt`, konfigurasjonen som fjerner
flest oversladdinger. Det er hele avveiningskurven, uten støyen.

Med --holdout velges konfigurasjonen på ett sett dokumenter og måles på et
annet. Uten det er «beste av 500 konfigurasjoner» stort sett overtilpasning.

Kjør:
    python utils/filter_sweep.py \\
        --fasit-csv /path/to/labels.csv \\
        --res-csv /path/to/resultat.csv \\
        --kostnad 20 --holdout 0.3 --ut /tmp/sweep.txt
"""

import argparse
import os
import sys
from collections import namedtuple
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from filter_felles import (STD_SLURV_FAKTOR, STD_TERSKEL, baseline,
                           bygg_datasett, evaluer, lag_filter,
                           lag_filter_per_kilde, les_fasit, les_kjorte_dok, les_prediksjoner,
                           pareto_front, skriv_oppsummering, splitt_dokumenter)

# En sweep-rad: måling, kort etikett, og spesifikasjonen som gjenskaper
# filteret. spec = {None: kwargs} for et globalt filter, ellers {kilde: kwargs}.
Rad = namedtuple("Rad", "m etikett spec")


def lag_predikat(spec):
    if None in spec:
        return lag_filter(**spec[None])
    return lag_filter_per_kilde(spec)


# ── Sortering ────────────────────────────────────────────────

SORT_FNS = {
    "netto":   lambda m: (-m.netto, m.tapt),
    "ov.fj":   lambda m: (-m.ov_fj, m.tapt),
    "tapt":    lambda m: (m.tapt, -m.ov_fj),
    "recall":  lambda m: (-m.recall_etter, -m.ov_fj),
    "pres":    lambda m: (-m.pres_etter, -m.netto),
    "ov/tapt": lambda m: (-m.ov_per_tapt, m.tapt),
}
SORT_ALIAS = {"rik.fj": "tapt", "ov/rik": "ov/tapt"}


def _sort_fn(navn):
    return SORT_FNS[SORT_ALIAS.get(navn, navn)]


# ── Tabellformat ─────────────────────────────────────────────

HODE_MAAL = (f" {'tapt':>6} {'tapt%':>7} │ {'ov.fj':>7} {'ov%':>6} │"
             f" {'red.fj':>7} │ {'netto':>9} {'ov/tapt':>8} │"
             f" {'recall%':>8} {'pres%':>7}")


def _maal_celler(m):
    ov_tapt = f"{m.ov_per_tapt:.1f}" if m.tapt else ("∞" if m.ov_fj else "–")
    return (f" {m.tapt:>6} {m.tapt_pst:>6.2f}% │ {m.ov_fj:>7} {m.ov_pst:>5.1f}% │"
            f" {m.red_fj:>7} │ {m.netto:>+9.0f} {ov_tapt:>8} │"
            f" {m.recall_etter:>7.2f}% {m.pres_etter:>6.1f}%")


def _g(v):
    return f"{v:g}" if v is not None else "av"


def _skjules(m, maks_tapt, maks_tapt_pst, min_ov_tapt):
    return ((maks_tapt is not None and m.tapt > maks_tapt)
            or (maks_tapt_pst is not None and m.tapt_pst > maks_tapt_pst)
            or (min_ov_tapt is not None and m.ov_per_tapt <= min_ov_tapt))


def _skjult_tekst(n, maks_tapt, maks_tapt_pst, min_ov_tapt):
    krav = []
    if maks_tapt is not None:
        krav.append(f"tapt > {maks_tapt:g}")
    if maks_tapt_pst is not None:
        krav.append(f"tapt > {maks_tapt_pst:g}%")
    if min_ov_tapt is not None:
        krav.append(f"ov/tapt ≤ {min_ov_tapt:g}")
    return f"  ({n} rader skjult: {' eller '.join(krav)})"


# (kort per-kilde-kode, langt CLI-flagg) per filterparameter
PARAM_KODER = (
    ("min_elongation", "e", "--elongation"),
    ("maks_elongation", "emaks", "--maks-elongation"),
    ("maks_hoyde", "h", "--maks-hoyde"),
    ("min_hoyde", "hmin", "--min-hoyde"),
    ("maks_bredde", "b", "--maks-bredde"),
    ("min_bredde", "bmin", "--min-bredde"),
    ("min_kortside", "kmin", "--min-kortside"),
    ("maks_kortside", "kmaks", "--maks-kortside"),
    ("min_langside", "lmin", "--min-langside"),
    ("maks_langside", "lmaks", "--maks-langside"),
    ("maks_areal", "a", "--maks-areal"),
    ("min_areal_px", "amin", "--min-areal-px"),
    ("conf_terskel", "c", "--conf"),
)


def review_kommando(spec):
    """Gjenskaper filteret som argumenter til filter_review.py."""
    def _par(kw):
        return ",".join(f"{kort}={kw[navn]:g}"
                        for navn, kort, _flagg in PARAM_KODER
                        if kw.get(navn) is not None)
    if None in spec:
        kw = spec[None]
        biter = [f"{flagg} {kw[navn]:g}" for navn, _kort, flagg in PARAM_KODER
                 if kw.get(navn) is not None]
        return " ".join(biter) or "(ingen filter)"
    biter = [f'"{k}:{_par(kw)}"' for k, kw in sorted(spec.items()) if _par(kw)]
    return ("--per-kilde " + " ".join(biter)) if biter else "(ingen filter)"


# ── Sweeps ───────────────────────────────────────────────────

def _sweep_en_param(ds, navn, verdier, filter_fn, kostnad):
    print(f"\n{'─' * 118}")
    print(f"Sweep: {navn}")
    print(f"{'─' * 118}")
    print(f"  {'Verdi':>8} │{HODE_MAAL}")
    print(f"  {'─' * 8}─┼{'─' * 106}")
    for v in verdier:
        m = evaluer(ds, lag_filter(**filter_fn(v)), kostnad=kostnad)
        print(f"  {_g(v):>8} │{_maal_celler(m)}")


STD_FELT = ("min_elongation", "maks_hoyde", "maks_bredde", "conf_terskel")
STD_HODER = ("elong", "hoyde", "bredde", "conf≥")


def _sweep_kombinasjoner(ds, elong_v, hoyde_v, bredde_v, conf_v, kostnad,
                         sort_key, felt=STD_FELT, hoder=STD_HODER,
                         tittel=None, kun_kilde=None, bare_front=True,
                         maks_tapt=None, maks_tapt_pst=None, min_ov_tapt=None,
                         csv_rader=None, maks_rader=None):
    """Sweeper alle kombinasjoner. kun_kilde: filtrer bare den kilden,
    men mål effekten globalt (dekning kommer fra alle kilder samlet)."""
    kandidater = ds.per_kilde[kun_kilde] if kun_kilde else None

    filter_info = ""
    if maks_tapt is not None:
        filter_info += f"  [tapt ≤ {maks_tapt:g}]"
    if maks_tapt_pst is not None:
        filter_info += f"  [tapt ≤ {maks_tapt_pst:g}%]"
    if min_ov_tapt is not None:
        filter_info += f"  [ov/tapt > {min_ov_tapt:g}]"

    har_conf = any(c is not None for c in conf_v)
    h0, h1, h2, h3 = hoder
    param_hode = (f"  {h0:>6} {h1:>6} {h2:>7} {h3:>6} │"
                  if har_conf else f"  {h0:>6} {h1:>6} {h2:>7} │")

    rader = []
    for min_e, maks_h, maks_b, c_t in product(elong_v, hoyde_v, bredde_v, conf_v):
        kw = dict(zip(felt, (min_e, maks_h, maks_b, c_t)))
        m = evaluer(ds, lag_filter(**kw), kostnad=kostnad, kandidater=kandidater)
        etikett = (f"{_g(min_e)}/{_g(maks_h)}/{_g(maks_b)}/{_g(c_t)}"
                   + (f" [{kun_kilde}]" if kun_kilde else ""))
        rader.append(Rad(m, etikett, {kun_kilde: kw} if kun_kilde else {None: kw}))
        if csv_rader is not None:
            rad = {"omfang": kun_kilde or "alle"}
            for navn, _kort, _flagg in PARAM_KODER:
                rad[navn] = kw.get(navn)
            csv_rader.append({
                **rad,
                "tapt": m.tapt, "tapt_pst": round(m.tapt_pst, 4),
                "ov_fj": m.ov_fj, "ov_pst": round(m.ov_pst, 3),
                "red_fj": m.red_fj, "slurv_fj": m.slurv_fj,
                "kritisk_fj": m.kritisk_fj, "n_fj": m.n_fj,
                "ov_areal_fj_pt2": round(m.ov_areal_fj),
                "netto": round(m.netto, 2),
                "recall_etter": round(m.recall_etter, 4),
                "pres_etter": round(m.pres_etter, 3),
            })

    aktuelle = [r for r in rader
                if not _skjules(r.m, maks_tapt, maks_tapt_pst, min_ov_tapt)]
    n_skjult = len(rader) - len(aktuelle)

    if bare_front:
        vis = pareto_front(aktuelle)
        note = (f"Pareto-front: {len(vis)} av {len(rader)} konfigurasjoner "
                f"— resten er dominert eller likeverdig")
    else:
        vis = sorted(aktuelle, key=lambda r: _sort_fn(sort_key)(r.m))
        note = f"alle {len(vis)} konfigurasjoner, sortert: {sort_key}"
    if maks_rader is not None:
        vis = vis[:maks_rader]

    print(f"\n{'═' * 145}")
    print(f"{tittel or 'KOMBINASJONS-SWEEP'}"
          f"   ({ds.dekket_foer} dekkede fasit-bokser, {ds.n_bom} oversladdinger"
          + (f", filter kun på '{kun_kilde}' ({len(kandidater)} pred), "
             f"øvrige urørt" if kun_kilde else "") + ")")
    print(f"  {note}   [kostnad {kostnad:g}]{filter_info}")
    print(f"{'═' * 145}")
    print(param_hode + HODE_MAAL)
    print(f"  {'─' * (len(param_hode) - 4)}┼{'─' * 106}")

    for rad in vis:
        e, h, b, c = (rad.spec[kun_kilde if kun_kilde else None][n]
                      for n in felt)
        params = (f"  {_g(e):>6} {_g(h):>6} {_g(b):>7} {_g(c):>6} │"
                  if har_conf else f"  {_g(e):>6} {_g(h):>6} {_g(b):>7} │")
        print(params + _maal_celler(rad.m))

    if n_skjult:
        print(_skjult_tekst(n_skjult, maks_tapt, maks_tapt_pst, min_ov_tapt))
    return rader


def _sweep_kryss_kilder(ds, per_kilde_rader, kostnad, sort_key, maks_kand=8,
                        felt=STD_FELT,
                        maks_tapt=None, maks_tapt_pst=None, min_ov_tapt=None):
    """Kombinerer de beste kandidatene per kilde og måler globalt.

    Kandidatene beskjæres med SAMME objektiv som sluttabellen sorteres på —
    ellers kan optimum være beskåret bort før kryssproduktet.
    """
    kilder = sorted(per_kilde_rader)
    if len(kilder) < 2:
        return []

    sort_fn = _sort_fn(sort_key)
    kandidater = []
    for k in kilder:
        beste = sorted(per_kilde_rader[k], key=lambda r: sort_fn(r.m))[:maks_kand]
        kandidater.append([(k, r.spec[k]) for r in beste])

    rader = []
    for kombo in product(*kandidater):
        spec = {k: kw for (k, kw) in kombo}
        m = evaluer(ds, lag_filter_per_kilde(spec), kostnad=kostnad)
        etikett = "  ".join(
            f"{k} " + "/".join(_g(kw.get(n)) for n in felt)
            for k, kw in sorted(spec.items()))
        rader.append(Rad(m, etikett, spec))

    print(f"\n{'═' * 145}")
    print("KRYSS-KILDE SWEEP  (uavhengige parametre per kilde, målt globalt)")
    print(f"  Pareto-fronten av {len(rader)} kombinasjoner "
          f"[kostnad {kostnad:g}, topp {maks_kand} kandidater per kilde]")
    print(f"{'═' * 145}")

    akse_navn = "/".join(n.split("_")[-1][:4] for n in felt)
    kolonne = max(24, max(len(k) for k in kilder) + len(akse_navn) + 6)
    hode = "  " + "  │  ".join(f"{k + f' ({akse_navn})':>{kolonne}}"
                               for k in kilder)
    print(hode + "  │" + HODE_MAAL)
    print(f"  {'─' * (len(hode) + 106)}")

    aktuelle = [r for r in rader
                if not _skjules(r.m, maks_tapt, maks_tapt_pst, min_ov_tapt)]
    for rad in pareto_front(aktuelle)[:15]:
        celler = "  │  ".join(
            f"{f'{k} ' + '/'.join(_g(kw.get(n)) for n in felt):>{kolonne}}"
            for k, kw in sorted(rad.spec.items()))
        print("  " + celler + "  │" + _maal_celler(rad.m))
    n_skjult = len(rader) - len(aktuelle)
    if n_skjult:
        print(_skjult_tekst(n_skjult, maks_tapt, maks_tapt_pst, min_ov_tapt))
    return rader


def _sweep_terskel(fasit, pred, terskler, valgt, slurv_faktor,
                   inkluder_ulabelte, kjorte):
    """Viser hvordan utgangspunktet endrer seg med overlapp-terskelen."""
    print(f"\n{'─' * 118}")
    print("Sweep: OVERLAPP-TERSKEL (utgangspunkt uten geometrifiltre)")
    print(f"{'─' * 118}")
    print(f"  {'Terskel':>8} │ {'TREFF':>8} {'SLURV':>8} {'BOM':>8} │"
          f" {'dekket':>8} {'udekket':>8} {'recall%':>8} {'pres%':>7} "
          f"{'dekkere/boks':>13}")
    print(f"  {'─' * 8}─┼─{'─' * 88}")
    for t in terskler:
        d = bygg_datasett(fasit, pred, terskel=t, slurv_faktor=slurv_faktor,
                          inkluder_ulabelte=inkluder_ulabelte,
                          kjorte_dok=kjorte)
        b = baseline(d)
        snitt = sum(d.dekning_foer) / d.dekket_foer if d.dekket_foer else 0
        markør = " ◀" if abs(t - valgt) < 1e-9 else ""
        print(f"  {t:>8.2f} │ {d.n_treff:>8} {d.n_slurv:>8} {d.n_bom:>8} │"
              f" {d.dekket_foer:>8} {d.n_fasit - d.dekket_foer:>8}"
              f" {b.recall_etter:>7.2f}% {b.pres_etter:>6.1f}% {snitt:>13.2f}"
              f"{markør}")


# ── Formanalyse ──────────────────────────────────────────────

MAAL = (("elongation", "elongation", 2), ("kortside", "kortside (pt)", 1),
        ("langside", "langside (pt)", 1), ("areal_px", "areal (px²)", 0))

PERSENTILER = (0.1, 1, 50, 99, 99.9)


def _persentil(sortert, pst):
    if not sortert:
        return 0.0
    i = (len(sortert) - 1) * pst / 100.0
    lav, høy = int(i), min(int(i) + 1, len(sortert) - 1)
    return sortert[lav] + (sortert[høy] - sortert[lav]) * (i - lav)


def _sweep_fordeling(ds):
    """Persentiler for form, per kilde og klasse.

    Sladdeboksen dekker de 5 siste sifrene i et fødselsnummer, så formen er
    fysisk begrenset. Ligger en BOM-boks utenfor det TREFF-boksene noen gang
    har vært, er formen umulig — ikke bare uvanlig.
    """
    print(f"\n{'═' * 145}")
    print("FORM-FORDELING  (hva en 5-sifret sladding faktisk ser ut som)")
    print(f"{'═' * 145}")
    hode = f"  {'kilde':>8} {'klasse':>7} {'n':>7} │ {'mål':<12}"
    for pst in PERSENTILER:
        hode += f" {('p' + format(pst, 'g')):>9}"
    print(hode)
    print(f"  {'─' * (len(hode) - 2)}")

    for kilde in ds.kilder():
        for klasse in ("TREFF", "SLURV", "BOM"):
            gruppe = [p for p in ds.per_kilde[kilde] if p["klasse"] == klasse]
            if len(gruppe) < 20:      # for få til å si noe om haler
                continue
            for nr, (nøkkel, navn, des) in enumerate(MAAL):
                sortert = sorted(p[nøkkel] for p in gruppe)
                venstre = (f"  {kilde:>8} {klasse:>7} {len(gruppe):>7} │"
                           if nr == 0 else f"  {'':>8} {'':>7} {'':>7} │")
                rad = venstre + f" {navn:<12}"
                for pst in PERSENTILER:
                    rad += f" {_persentil(sortert, pst):>9.{des}f}"
                print(rad)
            print(f"  {'·' * 100}")


def _avled_grenser(ds, pst, bruk_conf=None):
    """Grenser per kilde utledet fra TREFF-fordelingen, ikke fra netto.

    Nedre grense = TREFF-persentil `pst` fra bunnen, øvre = fra toppen.
    Ingen tilpasning mot ov.fj — grensene beskriver bare hvilke former
    korrekte sladdinger har hatt.
    """
    spec = {}
    for kilde in ds.kilder():
        treff = [p for p in ds.per_kilde[kilde]
                 if p["klasse"] in ("TREFF", "SLURV")]
        if len(treff) < 100:          # for få til å estimere haler
            continue
        kw = {}
        for nøkkel, felt_min, felt_maks in (
                ("elongation", "min_elongation", "maks_elongation"),
                ("kortside", "min_kortside", "maks_kortside"),
                ("langside", "min_langside", "maks_langside")):
            sortert = sorted(p[nøkkel] for p in treff)
            kw[felt_min] = round(_persentil(sortert, pst), 2)
            kw[felt_maks] = round(_persentil(sortert, 100 - pst), 2)
        areal = sorted(p["areal_px"] for p in treff)
        kw["min_areal_px"] = round(_persentil(areal, pst))
        if bruk_conf is not None:
            kw["conf_terskel"] = bruk_conf
        spec[kilde] = kw
    return spec


def _rapport_grenser(ds, ds_test, pst, kostnad):
    """Måler den utledede form-grensen — trening og holdout."""
    print(f"\n{'═' * 145}")
    print(f"FORM-GRENSE UTLEDET FRA TREFF  (nedre = p{pst:g}, øvre = p{100 - pst:g} "
          f"av korrekte bokser per kilde)")
    print("  Grensene er IKKE tilpasset ov.fj — de beskriver bare hvilke former")
    print("  korrekte 5-siffer-sladdinger har hatt. Alt utenfor har umulig form.")
    print(f"{'═' * 145}")

    for merke, conf in (("uten conf-port", None), ("med conf≥0.5-port", 0.5)):
        spec = _avled_grenser(ds, pst, bruk_conf=conf)
        if not spec:
            print("  (for få TREFF-bokser per kilde til å estimere haler)")
            return
        print(f"\n  {merke}:")
        for kilde, kw in sorted(spec.items()):
            print(f"    {kilde:>8}  elong [{kw['min_elongation']:g}, "
                  f"{kw['maks_elongation']:g}]  kortside [{kw['min_kortside']:g}, "
                  f"{kw['maks_kortside']:g}]  langside [{kw['min_langside']:g}, "
                  f"{kw['maks_langside']:g}]  areal ≥ {kw['min_areal_px']:g}px²")
        m = evaluer(ds, lag_filter_per_kilde(spec), kostnad=kostnad)
        print(f"    trening: {_maal_celler(m)}")
        if ds_test is not None:
            t = evaluer(ds_test, lag_filter_per_kilde(spec), kostnad=kostnad)
            print(f"    holdout: {_maal_celler(t)}")
        print(f"    → filter_review.py {review_kommando(spec)}")


# ── Pareto-front ─────────────────────────────────────────────

def _pareto_tabell(rader, kostnad, ds_test=None, tittel="PARETO-FRONT",
                   maks_tapt=None, maks_tapt_pst=None, min_ov_tapt=None):
    """Viser de ikke-dominerte konfigurasjonene: for hvert nivå av `tapt`,
    den som fjerner flest oversladdinger. Med ds_test måles hver av dem også
    på holdout-settet, slik at overtilpasning blir synlig."""
    front = pareto_front(rader, maal=lambda r: (r.m.tapt, r.m.ov_fj))
    front = [r for r in front
             if not _skjules(r.m, maks_tapt, maks_tapt_pst, min_ov_tapt)]
    if not front:
        return []

    bredde = max(28, max(len(r.etikett) for r in front) + 2)
    print(f"\n{'═' * 145}")
    print(f"{tittel}   ({len(front)} ikke-dominerte av {len(rader)} "
          f"konfigurasjoner)   [kostnad {kostnad:g}]")
    if ds_test is not None:
        print("  Venstre blokk = trening (der konfigurasjonen ble valgt), "
              "høyre = holdout (uavhengige dokumenter).")
        print("  Δ er holdout minus trening i prosentpoeng — store negative "
              "Δov% eller positive Δtapt% betyr overtilpasning.")
    print(f"{'═' * 145}")

    if ds_test is None:
        print(f"  {'konfigurasjon':<{bredde}}│{HODE_MAAL}")
        print(f"  {'─' * bredde}┼{'─' * 106}")
        for r in front:
            print(f"  {r.etikett:<{bredde}}│{_maal_celler(r.m)}")
        return front

    print(f"  {'konfigurasjon':<{bredde}}│"
          f" {'tapt':>5} {'tapt%':>7} {'ov.fj':>7} {'ov%':>6} {'netto':>8} │"
          f" {'tapt':>5} {'tapt%':>7} {'ov.fj':>7} {'ov%':>6} {'netto':>8} │"
          f" {'Δtapt%':>7} {'Δov%':>7}")
    print(f"  {' ' * bredde}│{'  trening'.ljust(38)} │"
          f"{'  holdout'.ljust(38)} │")
    print(f"  {'─' * bredde}┼{'─' * 39}┼{'─' * 39}┼{'─' * 17}")

    resultat = []
    for r in front:
        t = evaluer(ds_test, lag_predikat(r.spec), kostnad=kostnad)
        print(f"  {r.etikett:<{bredde}}│"
              f" {r.m.tapt:>5} {r.m.tapt_pst:>6.2f}% {r.m.ov_fj:>7} "
              f"{r.m.ov_pst:>5.1f}% {r.m.netto:>+8.0f} │"
              f" {t.tapt:>5} {t.tapt_pst:>6.2f}% {t.ov_fj:>7} "
              f"{t.ov_pst:>5.1f}% {t.netto:>+8.0f} │"
              f" {t.tapt_pst - r.m.tapt_pst:>+6.2f}p {t.ov_pst - r.m.ov_pst:>+6.1f}p")
        resultat.append((r, t))
    return resultat


def _anbefaling(front, kostnad, ds_test=None):
    """Peker ut konfigurasjonen med best netto på fronten."""
    if not front:
        return
    if ds_test is not None:
        beste = max(front, key=lambda rt: rt[1].netto)   # velg på holdout
        r, t = beste
        print(f"\n  Beste netto på HOLDOUT (kostnad {kostnad:g}): {r.etikett}")
        print(f"    trening: tapt {r.m.tapt} ({r.m.tapt_pst:.2f}%), "
              f"ov.fj {r.m.ov_fj} ({r.m.ov_pst:.1f}%), recall {r.m.recall_etter:.2f}%")
        print(f"    holdout: tapt {t.tapt} ({t.tapt_pst:.2f}%), "
              f"ov.fj {t.ov_fj} ({t.ov_pst:.1f}%), recall {t.recall_etter:.2f}%")
        spec = r.spec
    else:
        r = max(front, key=lambda r: r.m.netto)
        print(f"\n  Beste netto på fronten (kostnad {kostnad:g}): {r.etikett}")
        print(f"    tapt {r.m.tapt} ({r.m.tapt_pst:.2f}%), "
              f"ov.fj {r.m.ov_fj} ({r.m.ov_pst:.1f}%), "
              f"red.fj {r.m.red_fj}, recall {r.m.recall_etter:.2f}%")
        spec = r.spec
    print(f"    filter_review.py ... {review_kommando(spec)}")


# ── Hovedprogram ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Fasit-sentrisk evaluering av filterkonfigurasjoner")
    p.add_argument("--fasit-csv", required=True,
                   help="Labels-CSV (ACCEPTED + manuell = fasit, REJECTED ekskluderes)")
    p.add_argument("--res-csv", required=True,
                   help="Resultat-CSV fra modellen (pikselkoordinater)")
    p.add_argument("--terskel", type=float, default=STD_TERSKEL,
                   help=f"Overlapp-terskel for dekning (default: {STD_TERSKEL})")
    p.add_argument("--slurv-faktor", type=float, default=STD_SLURV_FAKTOR,
                   help="Pred-areal > faktor × dekket fasit-areal ⇒ SLURV "
                        f"(default: {STD_SLURV_FAKTOR})")
    p.add_argument("--inkluder-ulabelte", action="store_true",
                   help="Ta med prediksjoner på dokumenter som ikke finnes i "
                        "fasit-CSV-en (default: ekskluderes, siden de ellers "
                        "blåser opp oversladdingstallene)")
    p.add_argument("--form-pst", type=float, default=0.1, metavar="P",
                   help="Persentil for form-grensen utledet fra TREFF-bokser: "
                        "nedre grense = pP, øvre = p(100-P). Lavere = mer "
                        "konservativt (default: 0.1)")
    p.add_argument("--kjorte-liste", default=None, metavar="FIL",
                   help="Fil med dokumentene modellen har kjørt på (ett navn "
                        "eller nummer per linje). Uten den antas dokumentene "
                        "i resultat-CSV-en, og et dokument der modellen ikke "
                        "fant noe regnes som ukjørt.")
    p.add_argument("--kostnad", type=float, default=1.0,
                   help="Hvor mange fjernede oversladdinger én tapt fasit-boks "
                        "er verdt. netto = ov.fj − kostnad × tapt (default: 1)")
    p.add_argument("--holdout", type=float, default=None, metavar="ANDEL",
                   help="Hold av denne andelen av DOKUMENTENE til uavhengig "
                        "måling (f.eks. 0.3). Sweepen kjøres på resten, og "
                        "Pareto-fronten måles på begge.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for holdout-splitten (default: 42)")
    p.add_argument("--sort", default="netto",
                   choices=sorted(set(SORT_FNS) | set(SORT_ALIAS)),
                   help="Sorteringskolonne (default: netto)")
    p.add_argument("--maks-tapt", type=float, default=None,
                   help="Skjul rader der flere enn N fasit-bokser går tapt")
    p.add_argument("--maks-tapt-pst", "--maks-rik-pst", type=float, default=None,
                   dest="maks_tapt_pst",
                   help="Skjul rader der mer enn denne %% av dekkede fasit-bokser tapes")
    p.add_argument("--min-ov-tapt", "--min-ov-rik", type=float, default=None,
                   dest="min_ov_tapt",
                   help="Vis kun rader der ov.fj/tapt > denne verdien")
    p.add_argument("--alle-rader", action="store_true",
                   help="Skriv alle konfigurasjoner i kombinasjons-tabellene, "
                        "ikke bare Pareto-fronten. Gir en mye større fil.")
    p.add_argument("--maks-rader", type=int, default=None,
                   help="Maks antall rader per tabell")
    p.add_argument("--ut", default=None, metavar="FIL",
                   help="Skriv rapport til fil (default: auto-generert filnavn)")
    p.add_argument("--ut-csv", default=None, metavar="FIL",
                   help="Skriv alle sweep-rader til CSV for videre analyse")
    args = p.parse_args()

    if args.holdout is not None and not 0 < args.holdout < 1:
        p.error("--holdout må være mellom 0 og 1 (f.eks. 0.3)")

    ut_fil = args.ut or (
        f"filter_sweep_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.txt")

    class _Tee:
        """Skriver alt til fil, men kun utvalgte deler til terminalen."""

        def __init__(self, filobj, terminal):
            self.fil, self.terminal = filobj, terminal
            self.til_terminal = True

        def write(self, tekst):
            self.fil.write(tekst)
            if self.til_terminal:
                self.terminal.write(tekst)

        def flush(self):
            self.fil.flush()
            self.terminal.flush()

    fil = open(ut_fil, "w", encoding="utf-8")
    tee = _Tee(fil, sys.stdout)
    sys.stdout = tee
    try:
        fasit = les_fasit(args.fasit_csv)
        pred = les_prediksjoner(args.res_csv)
        kjorte = les_kjorte_dok(args.kjorte_liste) if args.kjorte_liste else None
        ds_full = bygg_datasett(fasit, pred, terskel=args.terskel,
                                slurv_faktor=args.slurv_faktor,
                                inkluder_ulabelte=args.inkluder_ulabelte,
                                kjorte_dok=kjorte)

        print(f"Overlapp-terskel {args.terskel:.0%}, "
              f"slurv-faktor {args.slurv_faktor:g}, "
              f"kostnad {args.kostnad:g}\n")
        skriv_oppsummering(ds_full)

        ds_test = None
        if args.holdout is not None:
            ds, ds_test = splitt_dokumenter(ds_full, args.holdout, args.seed)
            n_dok_tren = len({p["dok_nr"] for p in ds.pred})
            n_dok_test = len({p["dok_nr"] for p in ds_test.pred})
            print(f"\n  Holdout-splitt (seed {args.seed}, andel "
                  f"{args.holdout:g}, delt på dokument):")
            print(f"    trening: {n_dok_tren:>6} dok, {len(ds.pred):>7} pred, "
                  f"{ds.dekket_foer:>6} dekkede fasit-bokser, "
                  f"{ds.n_bom:>6} oversladdinger")
            print(f"    holdout: {n_dok_test:>6} dok, {len(ds_test.pred):>7} pred, "
                  f"{ds_test.dekket_foer:>6} dekkede fasit-bokser, "
                  f"{ds_test.n_bom:>6} oversladdinger")
        else:
            ds = ds_full
            print("\n  (ingen holdout — bruk --holdout 0.3 for å se om den "
                  "valgte konfigurasjonen holder på uavhengige dokumenter)")

        tee.til_terminal = False

        _sweep_terskel(fasit, pred, [0.15, 0.25, 0.35, 0.5, 0.7, 0.9], args.terskel,
                       args.slurv_faktor, args.inkluder_ulabelte, kjorte)
        # bygg_datasett muterer prediksjonene — bygg opp igjen med valgt terskel
        ds_full = bygg_datasett(fasit, pred, terskel=args.terskel,
                                slurv_faktor=args.slurv_faktor,
                                inkluder_ulabelte=args.inkluder_ulabelte,
                                kjorte_dok=kjorte)
        if args.holdout is not None:
            ds, ds_test = splitt_dokumenter(ds_full, args.holdout, args.seed)
        else:
            ds = ds_full

        _sweep_en_param(ds, "MIN_ELONGATION max(w/h, h/w)",
                        [1.1, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0],
                        lambda v: {"min_elongation": v}, args.kostnad)
        _sweep_en_param(ds, "MAKS_BOKS_HOYDE_PT",
                        [25, 30, 35, 40, 45, 50, 60, 80, 100],
                        lambda v: {"maks_hoyde": v}, args.kostnad)
        _sweep_en_param(ds, "MAKS_BOKS_BREDDE_PT",
                        [60, 80, 100, 120, 150, 200, 250],
                        lambda v: {"maks_bredde": v}, args.kostnad)

        har_conf = any(x["conf"] is not None for x in ds.pred)
        if har_conf:
            _sweep_en_param(
                ds, "CONF_TERSKEL (conf≥V beholdes uansett geometri; "
                    "kombinert med e=1.5/h=50/b=120)",
                [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                lambda v: {"min_elongation": 1.5, "maks_hoyde": 50,
                           "maks_bredde": 120, "conf_terskel": v},
                args.kostnad)

        _sweep_en_param(ds, "MIN_BOKS_AREAL (px²) — bittesmå bokser",
                        [500, 700, 965, 1200, 1600, 2200, 3000],
                        lambda v: {"min_areal_px": v}, args.kostnad)
        _sweep_en_param(ds, "MIN_KORTSIDE_PT — for tynne til å være tekst "
                            "(orienteringsuavhengig: stående bokser rammes ikke)",
                        [3, 4, 5, 6, 7, 8, 10],
                        lambda v: {"min_kortside": v}, args.kostnad)
        _sweep_en_param(ds, "MIN_LANGSIDE_PT — for korte til å romme 5 sifre",
                        [10, 15, 20, 25, 30, 40, 50],
                        lambda v: {"min_langside": v}, args.kostnad)
        _sweep_en_param(ds, "MAKS_ELONGATION — tynne, lange streker",
                        [6, 8, 10, 12, 15, 20, 30, 50],
                        lambda v: {"maks_elongation": v}, args.kostnad)

        _sweep_fordeling(ds)
        _rapport_grenser(ds, ds_test, args.form_pst, args.kostnad)

        elong_v = [None, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0]
        hoyde_v = [None, 40, 50, 60, 80]
        bredde_v = [None, 80, 100, 120, 150]
        conf_v = [None, 0.5] if har_conf else [None]

        csv_rader = [] if args.ut_csv else None
        grenser = dict(maks_tapt=args.maks_tapt,
                       maks_tapt_pst=args.maks_tapt_pst,
                       min_ov_tapt=args.min_ov_tapt)
        felles = dict(kostnad=args.kostnad, sort_key=args.sort,
                      csv_rader=csv_rader, maks_rader=args.maks_rader,
                      bare_front=not args.alle_rader, **grenser)

        alle_rader = _sweep_kombinasjoner(ds, elong_v, hoyde_v, bredde_v,
                                          conf_v, **felles)

        STOY_FELT_G = ("min_kortside", "min_langside", "maks_elongation",
                       "conf_terskel")
        stoy_rader = _sweep_kombinasjoner(
            ds, [None, 4, 5, 6, 7], [None, 15, 20, 25, 30],
            [None, 6, 8, 10, 12, 15], conf_v,
            felt=STOY_FELT_G, hoder=("k.min", "l.min", "e.maks", "conf≥"),
            tittel="STØYFILTRE — for små eller for tynne til å være 5 sifre",
            **felles)

        STOY_FELT = ("min_kortside", "min_langside", "maks_elongation",
                     "conf_terskel")
        STOY_HODER = ("k.min", "l.min", "e.maks", "conf≥")
        STOY_AKSER = ([None, 4, 5, 6, 7], [None, 15, 20, 25, 30],
                      [None, 6, 8, 10, 12, 15])

        kilder = ds.kilder()
        # Per kilde på støy-aksene: paddle-bokser er tette 5-siffer-bokser med
        # smalt formområde, yolo-bokser er rå deteksjoner. Terskler som er
        # gratis for én kilde kan koste for en annen.
        stoy_per_kilde = {}
        for kilde in kilder:
            k_conf = ([None, 0.5]
                      if any(x["conf"] is not None for x in ds.per_kilde[kilde])
                      else [None])
            stoy_per_kilde[kilde] = _sweep_kombinasjoner(
                ds, *STOY_AKSER, k_conf, felt=STOY_FELT, hoder=STOY_HODER,
                tittel=f"STØYFILTRE PER KILDE: {kilde.upper()}",
                kun_kilde=kilde, **felles)
            alle_rader += stoy_per_kilde[kilde]

        kryss_rader = []
        if len(kilder) > 1:
            per_kilde_rader = {}
            for kilde in kilder:
                k_conf = ([None, 0.5]
                          if any(x["conf"] is not None for x in ds.per_kilde[kilde])
                          else [None])
                per_kilde_rader[kilde] = _sweep_kombinasjoner(
                    ds, elong_v, hoyde_v, bredde_v, k_conf,
                    tittel=f"PER KILDE: {kilde.upper()}", kun_kilde=kilde,
                    **felles)
                alle_rader += per_kilde_rader[kilde]
            kryss_rader = _sweep_kryss_kilder(
                ds, per_kilde_rader, args.kostnad, args.sort, **grenser)
            kryss_rader += _sweep_kryss_kilder(
                ds, stoy_per_kilde, args.kostnad, args.sort,
                felt=STOY_FELT, **grenser)

        # ── Pareto-fronter: det eneste avsnittet som også går til terminalen
        tee.til_terminal = True
        front = _pareto_tabell(
            alle_rader + stoy_rader + kryss_rader, args.kostnad, ds_test=ds_test,
            tittel="PARETO-FRONT — alle konfigurasjoner (felles, per kilde "
                   "og kryss-kilde)", **grenser)
        _anbefaling(front, args.kostnad, ds_test=ds_test)
        tee.til_terminal = False

        if args.ut_csv and csv_rader:
            import csv as _csv
            felt_navn = list(csv_rader[0])
            for r in csv_rader:
                for k in r:
                    if k not in felt_navn:
                        felt_navn.append(k)
            with open(args.ut_csv, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=felt_navn)
                w.writeheader()
                w.writerows(csv_rader)
    finally:
        sys.stdout = tee.terminal
        fil.close()

    print(f"\n✓ Rapport skrevet til: {ut_fil} "
          f"({os.path.getsize(ut_fil) // 1024} KB)")
    if args.ut_csv:
        print(f"✓ Sweep-rader skrevet til: {args.ut_csv}")


if __name__ == "__main__":
    main()
