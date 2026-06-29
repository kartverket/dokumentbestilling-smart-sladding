import re
import statistics
from collections import namedtuple

import numpy as np
import easyocr
import torch
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerResult


SLADDE_SIFFER = 5
EXTRAMARGIN = 0.0            # ekstra bredde i median sifferbredder

VEKTER_KONTROLL_1 = [3, 7, 6, 1, 8, 9, 4, 5, 2]
VEKTER_KONTROLL_2 = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]

Token = namedtuple("Token", ["tekst", "x0", "y0", "x1", "y1"])
SifferBoks = namedtuple("SifferBoks", ["venstre", "hoyre", "topp", "bunn"])


reader = None

def _hent_reader():
    global reader
    if reader is None:
        gpu = torch.cuda.is_available()
        print(f"GPU tilgjengelig: {gpu}")

        reader = easyocr.Reader(["no", "en"], gpu=gpu, verbose=False)
    return reader


def _kontrollsiffer(vekter, siffer):
    rest = sum(vekt * tall for vekt, tall in zip(vekter, siffer)) % 11
    return (11 - rest) % 11


def gyldig_mod11(nummer):
    if len(nummer) != 11 or not nummer.isdigit():
        return False
    siffer = [int(tegn) for tegn in nummer]
    kontroll_1 = _kontrollsiffer(VEKTER_KONTROLL_1, siffer[:9])
    kontroll_2 = _kontrollsiffer(VEKTER_KONTROLL_2, siffer[:10])
    return kontroll_1 == siffer[9] and kontroll_2 == siffer[10]


def er_fnr_form(nummer):
    if len(nummer) != 11 or not nummer.isdigit():
        return False
    dag = int(nummer[0:2])
    maaned = int(nummer[2:4])
    return 1 <= dag <= 31 and 1 <= maaned <= 12


class FnrRecognizer(EntityRecognizer):
    def __init__(self):
        super().__init__(supported_entities=["NO_FNR"], supported_language="en")
    def load(self):
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        if "NO_FNR" not in entities:
            return []
        pos = [i for i, ch in enumerate(text) if ch.isdigit()]   # indeks til hvert siffer
        treff, i = [], 0
        while i + 11 <= len(pos):
            start, slutt = pos[i], pos[i + 10] + 1
            mellom = text[start:slutt]
            cifre = re.sub(r"\D", "", mellom)
            luker = re.findall(r"\D+", mellom)                   # sammenhengende ikke-siffer
            ok = (
                len(luker) <= 3                                  # OCR kan ha splittet fnr-et i biter
                and all(set(g) <= set(" .-") for g in luker)
                and all(len(g) <= 2 for g in luker)              # korte luker, ikke ny kolonne
                and er_fnr_form(cifre)
                and gyldig_mod11(cifre)
            )
            if ok:
                treff.append(RecognizerResult("NO_FNR", start, slutt, score=1.0))
                i += 11                                          # hopp forbi hele fnr-et
            else:
                i += 1                                           # gli ett siffer videre
        return treff


analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(FnrRecognizer())


def _les_tokens(bilde):
    ocr_treff = _hent_reader().readtext(np.array(bilde), allowlist="0123456789 .-", batch_size=16)
    tokens = []
    for poly, tekst, _ in ocr_treff:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        tokens.append(Token(tekst, min(xs), min(ys), max(xs), max(ys)))
    return tokens


def _grupper_til_linjer(tokens):
    linjer = []
    for token in sorted(tokens, key=lambda t: ((t.y0 + t.y1) / 2, t.x0)):  # middelhøyde, så start
        senter_y = (token.y0 + token.y1) / 2
        plassert = False
        for linje in linjer:
            if min(t.y0 for t in linje) <= senter_y <= max(t.y1 for t in linje):
                linje.append(token)
                plassert = True
                break
        if not plassert:
            linjer.append([token])
    return linjer


def _bygg_linjetekst(linje):
    tegn, kart = [], []
    for token_nr, token in enumerate(sorted(linje, key=lambda t: t.x0)):
        if token_nr > 0:
            tegn.append(" ")                          # mellomrom mellom to tokens
            kart.append(None)
        bredde = token.x1 - token.x0
        antall = len(token.tekst)
        for posisjon, ch in enumerate(token.tekst):
            tegn.append(ch)
            if ch.isdigit():
                venstre = token.x0 + bredde * posisjon / antall
                hoyre = token.x0 + bredde * (posisjon + 1) / antall
                kart.append(SifferBoks(venstre, hoyre, token.y0, token.y1))
            else:
                kart.append(None)
    return "".join(tegn), kart


def _sladdeboks(sifferbokser):
    if len(sifferbokser) <= SLADDE_SIFFER:
        return None
    siste = sifferbokser[-SLADDE_SIFFER:]
    anker = sifferbokser[-SLADDE_SIFFER - 1]
    median_bredde = statistics.median(boks.hoyre - boks.venstre for boks in siste)
    venstre = anker.hoyre
    hoyre = max(boks.hoyre for boks in siste) + EXTRAMARGIN * median_bredde
    topp = min(boks.topp for boks in siste)
    bunn = max(boks.bunn for boks in siste)
    return (round(venstre), round(topp), round(hoyre), round(bunn))


def finn_bokser(bilde):
    tokens = _les_tokens(bilde)
    linjer = _grupper_til_linjer(tokens)
    bokser = []
    for linje in linjer:
        tekst, kart = _bygg_linjetekst(linje)
        for treff in analyzer.analyze(tekst, entities=["NO_FNR"], language="en"):
            sifferbokser = [kart[i] for i in range(treff.start, treff.end) if kart[i] is not None]
            boks = _sladdeboks(sifferbokser)
            if boks is None:
                continue
            cifre = re.sub(r"\D", "", tekst[treff.start:treff.end])
            bokser.append((boks, gyldig_mod11(cifre)))
    return bokser
