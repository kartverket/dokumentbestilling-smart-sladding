import time
from contextlib import contextmanager

import fitz
import numpy as np

from config import DEDUP_OVERLAPP, YOLO_CONF_UTEN_TEKST, YOLO_CONF_VERTIKAL, YOLO_CONF_GEOMETRI_TERSKEL
from load_pdf import les_sider_fra_bytes
from paddle_ocr_model_fnr import les_tokens_batched, finn_bokser_fra_tokens, ocr_linjer_fra_tokens
from orientering import finn_rotasjon, boks_tilbake
from yolo_fnr import finn_yolo_bokser, snill_sjekk, tokens_i_boks, overlapp_andel_boks, er_vertikal, er_for_liten, har_feil_ratio, er_for_hoy, er_for_bred
from ocr_cache import les_cache, skriv_cache


@contextmanager
def _ta_tid(t, post):
    start = time.perf_counter()
    yield
    t[post] = t.get(post, 0.0) + (time.perf_counter() - start)


def _hopp_over_geometrifilter(kilde, conf):
    """«begge»-bokser bekreftet av begge modeller → ingen geometrifilter.
    Høy-konfidens YOLO-bokser → stol på modellen, ikke filtrer."""
    if kilde == "begge":
        return True
    return conf is not None and conf >= YOLO_CONF_GEOMETRI_TERSKEL


def _finn_bokser_kun_yolo(bilde_ocr):
    bokser = []
    for (x0, y0, x1, y1, conf) in finn_yolo_bokser(bilde_ocr):
        yb = (x0, y0, x1, y1)
        if (not er_for_liten(yb) and not har_feil_ratio(yb)
                and not er_for_hoy(yb) and not er_for_bred(yb)):
            bokser.append([(x0, y0, x1, y1), "yolo", round(conf, 3)])
    return [(tuple(boks), kilde, conf) for boks, kilde, conf in bokser]


def _finn_bokser_med_kilde(tokens, bilde_ocr, elektronisk_tinglyst=False):
    bokser = [[boks, "paddle", None] for (boks, _mod11) in finn_bokser_fra_tokens(tokens)]

    if not elektronisk_tinglyst:
        for (x0, y0, x1, y1, conf) in finn_yolo_bokser(bilde_ocr):
            yb = (x0, y0, x1, y1)
            dekket = [par for par in bokser if overlapp_andel_boks(yb, par[0]) > DEDUP_OVERLAPP]
            if dekket:
                for par in dekket:
                    par[1] = "begge"
                    par[2] = round(conf, 3)           # conf fra YOLO
            elif kilde := _godta_yolo_boks(tokens, yb, conf):
                bokser.append([yb, kilde, round(conf, 3)])

    # ── Dimensjonsfiltre (kildebevisste) ────────────────────────
    # «begge» og høy-konfidens YOLO fritas — kun minimumsstørrelse gjelder alltid
    bokser = [par for par in bokser
              if not er_for_liten(par[0]) and (
                  _hopp_over_geometrifilter(par[1], par[2])
                  or (not har_feil_ratio(par[0])
                      and not er_for_hoy(par[0])
                      and not er_for_bred(par[0], elektronisk=elektronisk_tinglyst))
              )]

    return [(tuple(boks), kilde, conf) for boks, kilde, conf in bokser]


def _godta_yolo_boks(tokens, boks, conf):
    if er_vertikal(boks):
        return "yolo_vertikal" if conf >= YOLO_CONF_VERTIKAL else None
    if tokens_i_boks(tokens, boks):
        return "yolo" if snill_sjekk(tokens, boks) else None
    return "yolo" if conf >= YOLO_CONF_UTEN_TEKST else None


def _bygg_side(si, bilde, tokens, bokser_med_kilde, k, med_linjer):
    h, w = bilde.shape[:2]
    bokser = []
    for boks, kilde, conf in bokser_med_kilde:
        x0, y0, x1, y1 = boks_tilbake(boks, k, w, h)
        b = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "kilde": kilde}
        if conf is not None:
            b["conf"] = conf
        bokser.append(b)
    side = {"side": si, "bilde_bredde": w, "bilde_hoyde": h, "bokser": bokser}
    if med_linjer:
        side["linjer"] = ocr_linjer_fra_tokens(tokens)
    return side


def _til_flat(sider, pdf_bytes):
    ut = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as dok:
        for side in sider:
            n = side["side"]
            bb, bh = side.get("bilde_bredde"), side.get("bilde_hoyde")
            if not bb or not bh or not (1 <= n <= dok.page_count):
                continue
            rect = dok[n - 1].rect                
            sx = rect.width / bb
            sy = rect.height / bh
            for b in side.get("bokser", []):
                x0, x1 = sorted((b["x0"] * sx, b["x1"] * sx))
                y0, y1 = sorted((b["y0"] * sy, b["y1"] * sy))
                d = {"page": n, "x": x0, "y": y0,
                     "width": x1 - x0, "height": y1 - y0,
                     "kilde": b.get("kilde")}
                if "conf" in b:
                    d["conf"] = b["conf"]
                ut.append(d)
    return ut


def run_model_on_pdf_bytes(pdf_bytes, skriv_tid=False, med_linjer=False, navn=None,
                           elektronisk_tinglyst=False, kun_yolo=False,
                           cache_mappe=None):
    t = {}

    with _ta_tid(t, "render"):
        bilder = list(les_sider_fra_bytes(pdf_bytes))

    # ── Forsøk å laste OCR-resultater fra cache ─────────────────
    cache_treff = False
    if cache_mappe and not kun_yolo and navn:
        cachet = les_cache(cache_mappe, navn)
        if cachet is not None:
            rotasjoner, tokens_per_side = cachet
            if len(rotasjoner) == len(bilder):
                bilder_ocr = [np.rot90(b, k) if k else b for b, k in zip(bilder, rotasjoner)]
                cache_treff = True

    if not cache_treff:
        with _ta_tid(t, "orientering"):
            rotasjoner = [finn_rotasjon(b) for b in bilder]
            bilder_ocr = [np.rot90(b, k) if k else b for b, k in zip(bilder, rotasjoner)]

        if not kun_yolo:
            with _ta_tid(t, "ocr"):
                tokens_per_side = les_tokens_batched(bilder_ocr)
            # Lagre til cache for fremtidige kjøringer
            if cache_mappe and navn:
                skriv_cache(cache_mappe, navn, rotasjoner, tokens_per_side)
        else:
            tokens_per_side = [[] for _ in bilder_ocr]
    else:
        if kun_yolo:
            tokens_per_side = [[] for _ in bilder_ocr]

    sider = []
    for si, (bilde, bilde_ocr, tokens, k) in enumerate(
            zip(bilder, bilder_ocr, tokens_per_side, rotasjoner), start=1):
        with _ta_tid(t, "yolo+match"):
            if kun_yolo:
                bokser_med_kilde = _finn_bokser_kun_yolo(bilde_ocr)
            else:
                bokser_med_kilde = _finn_bokser_med_kilde(tokens, bilde, elektronisk_tinglyst=elektronisk_tinglyst)
        with _ta_tid(t, "etterbehandling"):
            sider.append(_bygg_side(si, bilde, tokens, bokser_med_kilde, k, med_linjer))

    if skriv_tid:
        _skriv_tid(t, len(sider), navn, cache_treff)

    return _til_flat(sider, pdf_bytes)


def _skriv_tid(t, n_sider, navn=None, cache_treff=False):
    poster = ["render", "orientering", "ocr", "yolo+match", "etterbehandling"]
    total = sum(t.get(p, 0.0) for p in poster)

    label = f"Timing [{navn}]:" if navn else "Timing:"
    if cache_treff:
        label += " (OCR fra cache)"
    print(label)
    for post in poster:
        sek = t.get(post, 0.0)
        pct = (sek / total * 100) if total else 0.0
        print(f"  {post:<18}{sek:9.3f} s{pct:7.1f}%")
    print(f"  {'Total':<18}{total:9.3f} s")
    print(f"  {'Sider totalt':<18}{n_sider:9d}")
    if n_sider:
        print(f"  {'Per side':<18}{total / n_sider:9.3f} s")