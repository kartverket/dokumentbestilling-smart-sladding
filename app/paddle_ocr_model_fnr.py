import math
import os
import re
import statistics
from collections import namedtuple

import numpy as np

from config import SLADDE_SIFFER, LUFT_X, LUFT_Y, MAKS_HOYDE_FAKTOR, MODELL_SETT, DET_SIDE_LEN, REC_BATCH, SIDER_PER_OCR_BATCH, PDF_DPI

VEKTER_KONTROLL_1 = [3, 7, 6, 1, 8, 9, 4, 5, 2]
VEKTER_KONTROLL_2 = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]

_OCR_SIFFER_MAP = str.maketrans("oOsSlIbB", "00551166")


def _normaliser_ocr(tekst):
    return tekst.translate(_OCR_SIFFER_MAP)

Token = namedtuple("Token", ["tekst", "x0", "y0", "x1", "y1", "rec_score"])
SifferBoks = namedtuple("SifferBoks", ["venstre", "hoyre", "topp", "bunn", "rec_score"])
Treff = namedtuple("Treff", ["start", "end"])


_NAVN = {
    "v5": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    "v6": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
}
DET_MODELL, REC_MODELL = _NAVN[MODELL_SETT]
_MODELL_MAPPE = os.path.dirname(os.path.abspath(__file__))
DET_MODELL_DIR = os.path.join(_MODELL_MAPPE, DET_MODELL + "_infer")
REC_MODELL_DIR = os.path.join(_MODELL_MAPPE, REC_MODELL + "_infer")


reader = None


def _har_gpu():
    try:
        import paddle
        return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def _hent_reader():
    global reader
    if reader is None:
        # Importen ligger her, ikke på toppen: cache-lesere (run.py-arbeidere,
        # boks_trekk, filtersweep) bruker bare de rene tekstfunksjonene og
        # skal slippe å betale paddleocr-importen.
        from paddleocr import PaddleOCR

        gpu = _har_gpu()
        print(f"GPU tilgjengelig: {gpu}")

        kwargs = dict(
            lang="en",
            device="gpu" if gpu else "cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_type="max",
            text_det_limit_side_len=DET_SIDE_LEN,
            # paa GPU taaler vi stoerre rec-batch -> bedre gjennomstroemning.
            # SLADD_REC_BATCH kan senke den naar flere prosesser deler kortet;
            # batchstoerrelsen paavirker minnebruk og fart, ikke resultatet.
            text_recognition_batch_size=int(
                os.environ.get("SLADD_REC_BATCH")
                or (REC_BATCH * 2 if gpu else REC_BATCH)),
        )
        kwargs["text_detection_model_name"] = DET_MODELL
        kwargs["text_recognition_model_name"] = REC_MODELL
        kwargs["text_detection_model_dir"] = DET_MODELL_DIR
        kwargs["text_recognition_model_dir"] = REC_MODELL_DIR
        if gpu:
            kwargs["precision"] = "fp16"   
        else:
            kwargs["enable_mkldnn"] = True 
        if os.environ.get("SLADD_HPI") == "1":
            kwargs["enable_hpi"] = True

        reader = PaddleOCR(**kwargs)
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
    "Godtar baade fnr (dag 01-31) og d-nummer (dag 41-71)."
    if len(nummer) != 11 or not nummer.isdigit():
        return False
    dag = int(nummer[0:2])
    maaned = int(nummer[2:4])
    if 41 <= dag <= 71:          # d-nummer: 40 lagt til dagen
        dag -= 40
    return 1 <= dag <= 31 and 1 <= maaned <= 12


def finn_fnr(tekst, krev_mod11=True):
    """11-sifrede løp i teksten som ser ut som fnr/d-nummer.

    krev_mod11=False slår av kontrollsiffer-sjekken og gir *kandidater*: løp
    som har riktig form (11 siffer, gyldig dag og måned, korte luker), men som
    ikke nødvendigvis er et gyldig nummer. Brukes av boks_trekk til å svare på
    «finnes det i det hele tatt et 11-sifret nummer her?» — et enkelt lesefeil
    i et siffer velter mod11, så form alene er det riktige spørsmålet når
    poenget er å avvise bokser som ikke KAN være fnr.
    """
    norm = _normaliser_ocr(tekst)
    pos = [i for i, ch in enumerate(norm) if ch.isdigit()]   # indeks til hvert siffer
    treff, i = [], 0
    while i + 11 <= len(pos):
        start, slutt = pos[i], pos[i + 10] + 1
        mellom = norm[start:slutt]
        cifre = re.sub(r"\D", "", mellom)
        luker = re.findall(r"\D+", mellom)                    # sammenhengende ikke-siffer
        ok = (
            len(luker) <= 3                                   # OCR kan ha splittet fnr-et i biter
            and all(set(g) <= set(" .-,_") for g in luker)
            and all(len(g) <= 2 for g in luker)
            and er_fnr_form(cifre)
            and (not krev_mod11 or gyldig_mod11(cifre))
        )
        if ok:
            treff.append(Treff(start, slutt))
            i += 11                                           # hopp forbi hele fnr-et
        else:
            i += 1                                            # gli ett siffer videre
    return treff


def _les_tokens(res):
    tokens = []
    if not res:
        return tokens

    ord_per_linje = res.get("text_word")
    boks_per_linje = res.get("text_word_boxes")
    if ord_per_linje and boks_per_linje:
        scores_per_linje = res.get("text_word_scores") or []
        # Fallback: PaddleOCR 3.7+ gir ikke alltid per-ord-score, men har
        # linje-nivå rec_scores. Bruk linjescoren for alle ord i linjen.
        linje_rec_scores = res.get("rec_scores") or []
        for linje_idx, (ord_liste, boks_liste) in enumerate(zip(ord_per_linje, boks_per_linje)):
            # Per-ord-scores hvis tilgjengelig, ellers tom
            linje_scores = scores_per_linje[linje_idx] if linje_idx < len(scores_per_linje) else []
            # Fallback til linje-nivå rec_score
            fallback_rec = float(linje_rec_scores[linje_idx]) if linje_idx < len(linje_rec_scores) else None
            for ord_idx, (tekst, boks) in enumerate(zip(ord_liste, boks_liste)):
                if not tekst.strip():
                    continue
                if ord_idx < len(linje_scores):
                    rec_score = float(linje_scores[ord_idx])
                else:
                    rec_score = fallback_rec
                x0, y0, x1, y1 = (float(v) for v in np.asarray(boks).reshape(-1)[:4])
                tokens.append(Token(tekst, min(x0, x1), min(y0, y1),
                                    max(x0, x1), max(y0, y1), rec_score))
        if tokens:
            return tokens
    # Fallback: linjenivaa (fire hjornepunkter per boks).
    tekster = res.get("rec_texts") or []
    polys = res.get("rec_polys")
    if polys is None:
        polys = res.get("dt_polys") or []
    scores = res.get("rec_scores") or []
    for idx, (tekst, poly) in enumerate(zip(tekster, polys)):
        rec_score = float(scores[idx]) if idx < len(scores) else None
        pts = np.asarray(poly, dtype=float)
        tokens.append(Token(tekst, float(pts[:, 0].min()), float(pts[:, 1].min()),
                            float(pts[:, 0].max()), float(pts[:, 1].max()), rec_score))
    return tokens


def _grupper_til_linjer(tokens):
    # Linjens min-y0/maks-y1 holdes løpende i stedet for å regnes over alle
    # tokens per medlemskapstest — det var kvadratisk på token-tunge sider.
    linjer = []                 # [tokens, min_y0, maks_y1] per linje
    for token in sorted(tokens, key=lambda t: ((t.y0 + t.y1) / 2, t.x0)):  # middelhøyde, så start
        senter_y = (token.y0 + token.y1) / 2
        plassert = False
        for linje in linjer:
            if linje[1] <= senter_y <= linje[2]:
                linje[0].append(token)
                if token.y0 < linje[1]:
                    linje[1] = token.y0
                if token.y1 > linje[2]:
                    linje[2] = token.y1
                plassert = True
                break
        if not plassert:
            linjer.append([[token], token.y0, token.y1])
    return [linje[0] for linje in linjer]


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
            if _normaliser_ocr(ch).isdigit():
                venstre = token.x0 + bredde * posisjon / antall
                hoyre = token.x0 + bredde * (posisjon + 1) / antall
                kart.append(SifferBoks(venstre, hoyre, token.y0, token.y1, token.rec_score))
            else:
                kart.append(None)
    return "".join(tegn), kart


def bygg_linjer(tokens):
    """Tekstlinjene på én side: [(tokens, tekst, kart)].

    Grupperingen er den samme som finn_bokser_fra_tokens bruker. Skilt ut
    fordi boks_trekk trenger de samme linjene per YOLO-boks, og gruppering
    per boks ville kostet O(bokser x tokens) i stedet for én gang per side.
    """
    ut = []
    for linje in _grupper_til_linjer(tokens):
        tekst, kart = _bygg_linjetekst(linje)
        ut.append((linje, tekst, kart))
    return ut


def _sladdeboks(sifferbokser):
    if len(sifferbokser) <= SLADDE_SIFFER:
        return None
    siste = sifferbokser[-SLADDE_SIFFER:]            # de 5 som skal dekkes
    anker = sifferbokser[-SLADDE_SIFFER - 1]         # sifferet rett før (skal IKKE dekkes)

    median_bredde = statistics.median(b.hoyre - b.venstre for b in siste)

    topp = statistics.median(b.topp for b in sifferbokser)
    bunn = statistics.median(b.bunn for b in sifferbokser)

    mx = LUFT_X * median_bredde
    my = LUFT_Y * (bunn - topp)

    grense = (anker.hoyre + siste[0].venstre) / 2
    venstre = max(grense - mx, (anker.venstre + anker.hoyre) / 2)

    hoyre = max(b.hoyre for b in siste) + mx
    topp -= my
    bunn += my

    tak = MAKS_HOYDE_FAKTOR * median_bredde
    if (bunn - topp) > tak:
        senter = (topp + bunn) / 2
        topp, bunn = senter - tak / 2, senter + tak / 2

    return (math.floor(venstre), math.floor(topp), math.ceil(hoyre), math.ceil(bunn))



# Ekte fnr skrives med skilletegn på FASTE plasser i 11-sifferet: dato-
# punktumene etter siffer 2 og 4 («01.01.50») og skilletegnet/feltskillet
# etter siffer 6 («010150 12345», eget skjemafelt for personnummer-delen).
# Luker der beviser ingenting. Koordinat-søm legger lukene sine på
# vilkårlige plasser i vinduet — det er DE som flagges.
# (Manuell gjennomgang uttrekk 6: alle 4 tapene til den posisjonsblinde
# varianten satt nettopp på 2/4/6 — 3 datoformat-fnr og ett feltskille.)
_LOVLIGE_LUKE_POS = frozenset((2, 4, 6))


def _vindu_trekk(vindu, sifferbokser):
    """Trekk ved 11-siffer-vinduet en sladdeboks ble bygget fra.

    Skrives til resultat-CSV-en (TREKK_FELT) så etterfiltre kan feies uten ny
    kjøring. Koordinat- og målekolonner («6626630.58 549810.29») syr sammen
    vinduer på tvers av desimalpunktum og kolonnegap — lukereglene tillater
    det (≤2 tegn av « .-,_»), og linjeteksten har alltid nøyaktig ETT
    mellomrom mellom tokens uansett fysisk avstand, så selv tall i hver sin
    ende av en skisse sys sammen. Begge trekkene ser bort fra de lovlige
    posisjonene i _LOVLIGE_LUKE_POS:

      maks_luke        største horisontale avstand mellom to nabosiffer
                       UTENFOR lovlig posisjon, i median sifferbredde
      har_desimal_luke 1 hvis en luke med «.» eller «,» står utenfor
                       lovlig posisjon
    """
    har_desimal = 0
    pos = 0                      # antall siffer lest før tegnet
    for ch in vindu:
        if ch.isdigit():
            pos += 1
        elif ch in ".," and pos not in _LOVLIGE_LUKE_POS:
            har_desimal = 1
    bredder = sorted(b.hoyre - b.venstre for b in sifferbokser)
    median = bredder[len(bredder) // 2] or 1.0
    gap = max((sifferbokser[j + 1].venstre - sifferbokser[j].hoyre
               for j in range(len(sifferbokser) - 1)
               if (j + 1) not in _LOVLIGE_LUKE_POS), default=0.0)
    return {"maks_luke": round(max(gap, 0.0) / median, 2),
            "har_desimal_luke": har_desimal}


def finn_bokser_fra_tokens(tokens, linjer=None):
    """Sladdebokser fra OCR-tokens.

    `linjer` kan gis inn fra bygg_linjer når kalleren alt har gruppert siden,
    så grupperingen ikke gjøres to ganger per side.
    """
    bokser = []
    for _linje, tekst, kart in (linjer if linjer is not None else bygg_linjer(tokens)):
        for treff in finn_fnr(tekst):
            sifferbokser = [kart[i] for i in range(treff.start, treff.end) if kart[i] is not None]
            boks = _sladdeboks(sifferbokser)
            if boks is None:
                continue
            vindu = _normaliser_ocr(tekst[treff.start:treff.end])
            cifre = re.sub(r"\D", "", vindu)
            # Beregn paddle rec_score: minimum rec_score blant involverte tokens
            rec_scores = [sb.rec_score for sb in sifferbokser if sb.rec_score is not None]
            rec_score = round(min(rec_scores), 3) if rec_scores else None
            bokser.append((boks, gyldig_mod11(cifre), rec_score,
                           _vindu_trekk(vindu, sifferbokser)))
    return bokser


def ocr_linjer_fra_tokens(tokens):
    linjer_ut = []
    linjer = _grupper_til_linjer(tokens)
    for linje in sorted(linjer, key=lambda l: min(t.y0 for t in l)):
        tekst, _kart = _bygg_linjetekst(linje)
        tekst = tekst.strip()
        if not tekst:
            continue
        merker = []
        for tr in finn_fnr(tekst):
            cifre = re.sub(r"\D", "", _normaliser_ocr(tekst[tr.start:tr.end]))
            merker.append((cifre, gyldig_mod11(cifre)))
        linjer_ut.append((tekst, merker))
    return linjer_ut


def les_tokens_batched(bilder, batch_size=None):
    reader = _hent_reader()
    chunk_size = batch_size or SIDER_PER_OCR_BATCH

    tokens_per_side = []
    for start in range(0, len(bilder), chunk_size):
        chunk = bilder[start:start + chunk_size]
        bgr_chunk = [np.ascontiguousarray(b[:, :, ::-1]) for b in chunk]
        resultater = reader.predict(bgr_chunk, return_word_box=True) or []
        for res in resultater:
            tokens_per_side.append(_les_tokens(res))
        while len(tokens_per_side) < start + len(chunk):
            tokens_per_side.append([])

    return tokens_per_side