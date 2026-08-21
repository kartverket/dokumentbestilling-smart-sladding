"""
Trekk som beskriver hva Paddle leste i og rundt en YOLO-boks.

Formålet er å kunne teste strengere varianter av snill_sjekk UTEN å kjøre
pipelinen på nytt for hver terskel: trekkene skrives til resultat-CSV-en, og
utils/filter_sweep.py feier over kombinasjoner rent i CPU.

Hvorfor det er gyldig å feie i CSV-en i stedet for i pipelinen:

  * _godta_yolo_boks kjøres BARE for YOLO-bokser uten overlappende
    Paddle-boks — overlappende blir «begge» og går aldri innom regelen. En
    strengere regel kan derfor bare treffe kilde «yolo».
  * MIN_SIFFER=1 / MAKS_BOKSTAVER=1 er den løsest mulige innstillingen. Alt en
    strengere variant ville beholdt ligger allerede i CSV-en.

Derfor beregnes trekkene KUN for kilde «yolo». Paddle- og «begge»-bokser har
et mod11-validert nummer bak seg og styres ikke av denne regelen; de får tomme
trekk, slik at et globalt filter i sweepen aldri kan treffe dem ved uhell.
Det samme gjelder «yolo_vertikal»: den grenen leser ikke tokens i det hele
tatt, og linjelogikken under forutsetter vannrett tekst.

Trekk per boks:

    har_tokens        1 hvis Paddle leste noe i boksen. Bokser uten tekst
                      styres av YOLO_CONF_UTEN_TEKST, ikke av snill_sjekk, og
                      må holdes utenfor siffer-reglene.
    n_siffer          siffer i boksen        (samme telling som snill_sjekk)
    n_bokstaver       bokstaver i boksen     (samme telling som snill_sjekk)
    rec_min           laveste rec_score blant tokens i boksen
    rec_median        median rec_score blant tokens i boksen
    rec_min_linje     laveste rec_score på hele linjen boksen ligger på
    n_siffer_linje    siffer på hele linjen
    siffer_run        lengste sammenhengende sifferløp som overlapper boksen
    har_fnr_kandidat  1 hvis et 11-sifret løp med gyldig fnr-FORM (uten
                      mod11) overlapper boksen
    har_desimal_naer  1 hvis et desimalskille (siffer , eller . siffer)
                      overlapper boksen — koordinater har det, fnr aldri

MERK om har_fnr_kandidat: finn_fnr godtar «.» og «,» som luke mellom
sifferbiter, fordi OCR deler opp et fnr på den måten. På en koordinatlinje syr
den samme regelen sammen to nabotall: «370600.83 -56912.29» inneholder løpet
60083569122, som har gyldig fnr-form (dag 60 = d-nummer 20, måned 08). Trekket
er altså ikke i seg selv nok til å skille koordinater fra fnr — det er
har_desimal_naer som gjør det. Bruk dem sammen.
"""

import re
import statistics

from config import TREKK_FELT                                   # noqa: F401 — re-eksport
from paddle_ocr_model_fnr import finn_fnr
from yolo_fnr import tell_siffer_bokstaver, tokens_i_boks

_DESIMAL = re.compile(r"\d[.,]\d")


def _overlapp_areal(t, boks):
    ix0, iy0 = max(t.x0, boks[0]), max(t.y0, boks[1])
    ix1, iy1 = min(t.x1, boks[2]), min(t.y1, boks[3])
    return (ix1 - ix0) * (iy1 - iy0) if (ix1 > ix0 and iy1 > iy0) else 0.0


def _velg_linje(linjer, boks):
    """Linjen boksen ligger på: den med størst samlet tokenoverlapp."""
    beste, beste_ov = None, 0.0
    for post in linjer:
        ov = sum(_overlapp_areal(t, boks) for t in post[0])
        if ov > beste_ov:
            beste, beste_ov = post, ov
    return beste


def _spenner_over(kart, start, slutt, boks):
    """Overlapper sifferboksene i [start, slutt) boksen vannrett?"""
    sifre = [kart[i] for i in range(start, slutt) if i < len(kart) and kart[i] is not None]
    if not sifre:
        return False
    venstre = min(s.venstre for s in sifre)
    hoyre = max(s.hoyre for s in sifre)
    return hoyre > boks[0] and venstre < boks[2]


def _siffer_run(tekst, kart, boks):
    """Lengste sammenhengende sifferløp i teksten som overlapper boksen.

    Et fnr sladdes på de siste 5 av 11 siffer, så et løp på 5-7 siffer uten
    noe mer rundt seg — en koordinat — skiller seg fra et ekte treff.
    """
    lengste = 0
    for m in re.finditer(r"\d+", tekst):
        if _spenner_over(kart, m.start(), m.end(), boks):
            lengste = max(lengste, m.end() - m.start())
    return lengste


def _har_desimal(tekst, kart, boks):
    for m in _DESIMAL.finditer(tekst):
        if _spenner_over(kart, m.start(), m.end(), boks):
            return 1
    return 0


def _rec_verdier(tokens):
    return [t.rec_score for t in tokens if t.rec_score is not None]


def trekk_for_boks(tokens, linjer, boks):
    """Trekkene for én YOLO-boks. `linjer` kommer fra bygg_linjer(tokens)."""
    i_boks = tokens_i_boks(tokens, boks)
    n_siffer, n_bokstaver = tell_siffer_bokstaver(i_boks)
    rec = _rec_verdier(i_boks)

    trekk = {
        "har_tokens": 1 if i_boks else 0,
        "n_siffer": n_siffer,
        "n_bokstaver": n_bokstaver,
        "rec_min": round(min(rec), 3) if rec else None,
        "rec_median": round(statistics.median(rec), 3) if rec else None,
        "rec_min_linje": None,
        "n_siffer_linje": 0,
        "siffer_run": 0,
        "har_fnr_kandidat": 0,
        "har_desimal_naer": 0,
    }

    post = _velg_linje(linjer, boks)
    if post is None:
        return trekk
    linje_tokens, tekst, kart = post

    rec_linje = _rec_verdier(linje_tokens)
    trekk["rec_min_linje"] = round(min(rec_linje), 3) if rec_linje else None
    trekk["n_siffer_linje"] = sum(ch.isdigit() for ch in tekst)
    trekk["siffer_run"] = _siffer_run(tekst, kart, boks)
    trekk["har_desimal_naer"] = _har_desimal(tekst, kart, boks)
    trekk["har_fnr_kandidat"] = 1 if any(
        _spenner_over(kart, tr.start, tr.end, boks)
        for tr in finn_fnr(tekst, krev_mod11=False)) else 0
    return trekk
