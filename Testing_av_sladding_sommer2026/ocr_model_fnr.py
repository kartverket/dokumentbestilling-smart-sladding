import math
import os
import re
import statistics
from collections import namedtuple

import numpy as np
from paddleocr import PaddleOCR


SLADDE_SIFFER = 5
LUFT_X = 0.20               # horisontal margin, andel av median sifferbredde
LUFT_Y = 0.12               # vertikal margin, andel av sifferhoyde

VEKTER_KONTROLL_1 = [3, 7, 6, 1, 8, 9, 4, 5, 2]
VEKTER_KONTROLL_2 = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]

Token = namedtuple("Token", ["tekst", "x0", "y0", "x1", "y1"])
SifferBoks = namedtuple("SifferBoks", ["venstre", "hoyre", "topp", "bunn"])
Treff = namedtuple("Treff", ["start", "end"])


DET_SIDE_LEN = 2048
REC_BATCH = 64                # tekstlinjer per gjenkjennings-batch (fart)
SIDER_PER_OCR_BATCH_GPU = 24  # V100 32GB: ~6-8 GB VRAM per 16 sider, 24 er trygt
SIDER_PER_OCR_BATCH_CPU = 4

DET_MODELL = "PP-OCRv5_server_det"
REC_MODELL = "PP-OCRv5_server_rec"
DET_MODELL_DIR = "PP-OCRv5_server_det_infer"
REC_MODELL_DIR = "PP-OCRv5_server_rec_infer"


reader = None
gpu_tilgjengelig = None

IKKE_SIFFER = re.compile(r"\D")
IKKE_SIFFER_GRUPPER = re.compile(r"\D+")


def _har_gpu():
    global gpu_tilgjengelig
    if gpu_tilgjengelig is not None:
        return gpu_tilgjengelig
    try:
        import paddle
        gpu_tilgjengelig = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        gpu_tilgjengelig = False
    return gpu_tilgjengelig


def _hent_side_batch_size(gpu):
    standard = SIDER_PER_OCR_BATCH_GPU if gpu else SIDER_PER_OCR_BATCH_CPU
    verdi = os.getenv("OCR_PAGES_PER_BATCH")
    if verdi is None:
        return standard
    try:
        return max(1, int(verdi))
    except ValueError:
        return standard


def _hent_rec_batch_size(gpu):
    standard = REC_BATCH * 3 if gpu else REC_BATCH
    verdi = os.getenv("OCR_RECOGNITION_BATCH_SIZE")
    if verdi is None:
        return standard
    try:
        return max(1, int(verdi))
    except ValueError:
        return standard


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
            text_recognition_batch_size=_hent_rec_batch_size(gpu),
        )
        kwargs["text_detection_model_name"] = DET_MODELL
        kwargs["text_recognition_model_name"] = REC_MODELL
        kwargs["text_detection_model_dir"] = DET_MODELL_DIR
        kwargs["text_recognition_model_dir"] = REC_MODELL_DIR
        if gpu:
            kwargs["precision"] = "fp16"
            kwargs["enable_tensorrt_engine"] = True   # JIT-kompilerer til TensorRT ved foerste kjoering
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
    if len(nummer) != 11 or not nummer.isdigit():
        return False
    dag = int(nummer[0:2])
    maaned = int(nummer[2:4])
    return 1 <= dag <= 31 and 1 <= maaned <= 12


def finn_fnr(tekst):
    pos = [i for i, ch in enumerate(tekst) if ch.isdigit()]   # indeks til hvert siffer
    treff, i = [], 0
    while i + 11 <= len(pos):
        start, slutt = pos[i], pos[i + 10] + 1
        mellom = tekst[start:slutt]
        cifre = IKKE_SIFFER.sub("", mellom)
        luker = IKKE_SIFFER_GRUPPER.findall(mellom)            # sammenhengende ikke-siffer
        ok = (
            len(luker) <= 3                                   # OCR kan ha splittet fnr-et i biter
            and all(set(g) <= set(" .-") for g in luker)
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
            if linje[0] <= senter_y <= linje[1]:
                if token.y0 < linje[0]:
                    linje[0] = token.y0
                if token.y1 > linje[1]:
                    linje[1] = token.y1
                linje[2].append(token)
                plassert = True
                break
        if not plassert:
            linjer.append([token.y0, token.y1, [token]])
    return [linje[2] for linje in linjer]


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
    siste = sifferbokser[-SLADDE_SIFFER:]            # de 5 som skal dekkes
    anker = sifferbokser[-SLADDE_SIFFER - 1]         # sifferet rett foer (skal IKKE dekkes)

    median_bredde = statistics.median(b.hoyre - b.venstre for b in siste)
    hoyde = max(b.bunn for b in siste) - min(b.topp for b in siste)
    mx = LUFT_X * median_bredde
    my = LUFT_Y * hoyde

    grense = (anker.hoyre + siste[0].venstre) / 2
    venstre = max(grense - mx, (anker.venstre + anker.hoyre) / 2)

    hoyre = max(b.hoyre for b in siste) + mx
    topp = min(b.topp for b in siste) - my
    bunn = max(b.bunn for b in siste) + my

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
            cifre = IKKE_SIFFER.sub("", tekst[treff.start:treff.end])
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
            cifre = IKKE_SIFFER.sub("", tekst[tr.start:tr.end])
            merker.append((cifre, gyldig_mod11(cifre)))
        linjer_ut.append((tekst, merker))
    return linjer_ut


def les_tokens_batched(bilder):
    reader = _hent_reader()
    side_batch_size = _hent_side_batch_size(_har_gpu())

    tokens_per_side = []
    for start in range(0, len(bilder), side_batch_size):
        chunk = bilder[start:start + side_batch_size]
        bgr_chunk = [np.ascontiguousarray(b[:, :, ::-1]) for b in chunk]
        resultater = reader.predict(bgr_chunk, return_word_box=True) or []
        for res in resultater:
            tokens_per_side.append(_les_tokens(res))
        while len(tokens_per_side) < start + len(chunk):
            tokens_per_side.append([])

    return tokens_per_side