import math
import os
import re
import statistics
from collections import namedtuple

import numpy as np
from paddleocr import PaddleOCR


SLADDE_SIFFER = 5
LUFT_X = 0.35              
LUFT_Y = 0.0             
MIN_MARG_PX = 4
MAKS_HOYDE_FAKTOR = 3.0     # sladdehoyde maks N x median sifferbredde
MAKS_BREDDE_FAKTOR = 10     # 5 siffer + luker skal ikke spenne mer enn ~10 sifferbredder             

VEKTER_KONTROLL_1 = [3, 7, 6, 1, 8, 9, 4, 5, 2]
VEKTER_KONTROLL_2 = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]

_OCR_SIFFER_MAP = str.maketrans("oOsSlIbB", "00551166")


def _normaliser_ocr(tekst):
    return tekst.translate(_OCR_SIFFER_MAP)

Token = namedtuple("Token", ["tekst", "x0", "y0", "x1", "y1"])
SifferBoks = namedtuple("SifferBoks", ["venstre", "hoyre", "topp", "bunn"])
Treff = namedtuple("Treff", ["start", "end"])


DET_SIDE_LEN = 2048
REC_BATCH = 64                # tekstlinjer per gjenkjennings-batch (fart)
SIDER_PER_OCR_BATCH = 8       # sider matet inn i ETT predict-kall (GPU-utnyttelse)

DET_MODELL = "PP-OCRv5_server_det"
REC_MODELL = "PP-OCRv5_server_rec"
_MODELL_MAPPE = os.path.dirname(os.path.abspath(__file__))
DET_MODELL_DIR = os.path.join(_MODELL_MAPPE, "PP-OCRv5_server_det_infer")
REC_MODELL_DIR = os.path.join(_MODELL_MAPPE, "PP-OCRv5_server_rec_infer")


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
            # paa GPU taaler vi stoerre rec-batch -> bedre gjennomstroemning
            text_recognition_batch_size=REC_BATCH * 2 if gpu else REC_BATCH,
        )
        kwargs["text_detection_model_name"] = DET_MODELL
        kwargs["text_recognition_model_name"] = REC_MODELL
        kwargs["text_detection_model_dir"] = DET_MODELL_DIR
        kwargs["text_recognition_model_dir"] = REC_MODELL_DIR
        if gpu:
            kwargs["precision"] = "fp16"   
        else:
            kwargs["enable_mkldnn"] = True 

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


def finn_fnr(tekst):
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
            and all(len(g) <= 2 for g in luker)               # korte luker, ikke ny kolonne
            and er_fnr_form(cifre)
            and gyldig_mod11(cifre)
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
        for ord_liste, boks_liste in zip(ord_per_linje, boks_per_linje):
            for tekst, boks in zip(ord_liste, boks_liste):
                if not tekst.strip():                 # hopp over rene mellomrom
                    continue
                x0, y0, x1, y1 = (float(v) for v in np.asarray(boks).reshape(-1)[:4])
                tokens.append(Token(tekst, min(x0, x1), min(y0, y1),
                                    max(x0, x1), max(y0, y1)))
        if tokens:
            return tokens

    # Fallback: linjenivaa (fire hjornepunkter per boks).
    tekster = res.get("rec_texts") or []
    polys = res.get("rec_polys")
    if polys is None:
        polys = res.get("dt_polys") or []
    for tekst, poly in zip(tekster, polys):
        pts = np.asarray(poly, dtype=float)
        tokens.append(Token(tekst, float(pts[:, 0].min()), float(pts[:, 1].min()),
                            float(pts[:, 0].max()), float(pts[:, 1].max())))
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
            if _normaliser_ocr(ch).isdigit():
                venstre = token.x0 + bredde * posisjon / antall
                hoyre = token.x0 + bredde * (posisjon + 1) / antall
                kart.append(SifferBoks(venstre, hoyre, token.y0, token.y1))
            else:
                kart.append(None)
    return "".join(tegn), kart


def _sladdeboks(sifferbokser):
    if len(sifferbokser) <= SLADDE_SIFFER:
        return None
    siste = sifferbokser[-SLADDE_SIFFER:]            # de 5 som skal dekkes
    anker = sifferbokser[-SLADDE_SIFFER - 1]         # sifferet rett foer (skal IKKE dekkes)

    median_bredde = statistics.median(b.hoyre - b.venstre for b in siste)

    spenn = max(b.hoyre for b in siste) - min(b.venstre for b in siste)
    if spenn > MAKS_BREDDE_FAKTOR * median_bredde:
        return None           

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


def finn_bokser_fra_tokens(tokens):
    linjer = _grupper_til_linjer(tokens)
    bokser = []
    for linje in linjer:
        tekst, kart = _bygg_linjetekst(linje)
        for treff in finn_fnr(tekst):
            sifferbokser = [kart[i] for i in range(treff.start, treff.end) if kart[i] is not None]
            boks = _sladdeboks(sifferbokser)
            if boks is None:
                continue
            cifre = re.sub(r"\D", "", _normaliser_ocr(tekst[treff.start:treff.end]))
            bokser.append((boks, gyldig_mod11(cifre)))
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


def les_tokens_batched(bilder):
    reader = _hent_reader()

    tokens_per_side = []
    for start in range(0, len(bilder), SIDER_PER_OCR_BATCH):
        chunk = bilder[start:start + SIDER_PER_OCR_BATCH]
        bgr_chunk = [np.ascontiguousarray(b[:, :, ::-1]) for b in chunk]
        resultater = reader.predict(bgr_chunk, return_word_box=True) or []
        for res in resultater:
            tokens_per_side.append(_les_tokens(res))
        while len(tokens_per_side) < start + len(chunk):
            tokens_per_side.append([])

    return tokens_per_side