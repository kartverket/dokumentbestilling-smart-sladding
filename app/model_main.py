import time
from contextlib import contextmanager

import fitz
import numpy as np

from config import (DEDUP_OVERLAPP, PDF_DPI, YOLO_CACHE_CONF_GULV, YOLO_CONF,
                    YOLO_CONF_UTEN_TEKST, YOLO_CONF_VERTIKAL, YOLO_CONF_GEOMETRI_TERSKEL,
                    AVVIS_DESIMAL_REC_VETO, AVVIS_DESIMAL_CONF_FRITAK,
                    AVVIS_DESIMAL_REC_VETO_LAV, AVVIS_DESIMAL_CONF_TAK_LAV,
                    LINJEBEVIS_LINJE_VETO, LINJEBEVIS_CONF_FRITAK,
                    LINJEBEVIS_RUN_MAKS)
from load_pdf import les_sider_fra_bytes
from paddle_ocr_model_fnr import (les_tokens_batched, finn_bokser_fra_tokens,
                                  ocr_linjer_fra_tokens, bygg_linjer)
from orientering import finn_rotasjoner_batch, boks_tilbake
from yolo_fnr import (finn_yolo_bokser, snill_sjekk, tokens_i_boks, overlapp_andel_boks,
                      er_vertikal, er_for_liten, har_feil_ratio, er_for_tynn,
                      er_for_smal_yolo, er_for_kort_yolo, har_paddle_stoyform)
from boks_trekk import trekk_for_boks
from ocr_cache import les_cache as les_ocr_cache, skriv_cache as skriv_ocr_cache
from yolo_cache import les_cache as les_yolo_cache, skriv_cache as skriv_yolo_cache


@contextmanager
def _ta_tid(t, post):
    start = time.perf_counter()
    yield
    t[post] = t.get(post, 0.0) + (time.perf_counter() - start)


def _hopp_over_geometrifilter(conf, kilde):
    """Høy konfidens → stol på modellen, hopp over geometrifiltrene.

    Tar bare konfidens, ikke kilde: grensene er de samme for yolo og
    begge. «begge»-bokser var tidligere fritatt uansett konfidens, men
    gjennomgangen av uttrekk 4 viste at lav-konfidens «begge» står for en reell
    del av oversladdingen (4 av 7 tap hadde conf 0.17-0.31), så de går nå
    gjennom samme port som resten. Paddle-bokser fritas aldri — OCR-konfidens
    indikerer gjenkjenningskvalitet, ikke deteksjonssikkerhet.
    """
    if kilde == "paddle":
        return False
    return conf is not None and conf >= YOLO_CONF_GEOMETRI_TERSKEL


def _desimalregel_forkaster(trekk, conf):
    """Desimalskille i sikkert lest tekst → koordinat/beløp, ikke fnr.

    Speiler _ocr_grunn i utils/filter_felles.py med des=1,
    rveto=AVVIS_DESIMAL_REC_VETO, cfritak=AVVIS_DESIMAL_CONF_FRITAK.
    Kalles på ENDELIG kilde etter dedup: en boks som ble «begge» underveis
    har et Paddle-funn bak seg og rammes ikke. Se config for tallgrunnlaget.
    """
    if not trekk or not trekk.get("har_tokens"):
        return False
    rec = trekk.get("rec_min")
    if rec is None or not trekk.get("har_desimal_naer"):
        return False
    if rec >= AVVIS_DESIMAL_REC_VETO \
            and (conf is None or conf < AVVIS_DESIMAL_CONF_FRITAK):
        return True
    # Lav-tier: litt svakere lesing holder når deteksjonen selv er svak.
    return (rec >= AVVIS_DESIMAL_REC_VETO_LAV
            and (conf is None or conf < AVVIS_DESIMAL_CONF_TAK_LAV))


def _linjebevis_forkaster(trekk, conf):
    """Sikkert lest linje beviser at tallet ikke kan være et fnr.

    Speiler _ocr_grunn i utils/filter_felles.py med avvis_run_6_10=RUN_MAKS,
    avvis_orgnr=1, linje_veto og ocr_conf_fritak. Kalles på ENDELIG kilde
    etter dedup, som desimalregelen. Se config for tallgrunnlaget.
    """
    if not trekk or not trekk.get("har_tokens"):
        return False
    if conf is not None and conf >= LINJEBEVIS_CONF_FRITAK:
        return False
    linje = trekk.get("rec_min_linje")
    if linje is None or linje < LINJEBEVIS_LINJE_VETO:
        return False
    lang = trekk.get("lang_run")
    if lang is not None and 6 <= lang <= LINJEBEVIS_RUN_MAKS:
        return True
    return bool(trekk.get("har_orgnr"))


def _finn_bokser_kun_yolo(yolo_bokser):
    bokser = []
    for (x0, y0, x1, y1, conf) in yolo_bokser:
        yb = (x0, y0, x1, y1)
        if not er_for_liten(yb) and not er_for_smal_yolo(yb) and (
                _hopp_over_geometrifilter(round(conf, 3), "yolo")
                or (not har_feil_ratio(yb) and not er_for_tynn(yb)
                    and not er_for_kort_yolo(yb))):
            bokser.append([(x0, y0, x1, y1), "yolo", round(conf, 3), None, None])
    return [tuple(par) for par in bokser]


def _finn_bokser_med_kilde(tokens, yolo_bokser):
    """Slå sammen Paddle- og YOLO-bokser.

    En tom `yolo_bokser` gir ren Paddle-deteksjon — det er slik
    elektronisk tinglyste dokumenter behandles.

    Intern struktur per boks: [boks, kilde, yolo_conf, paddle_rec_score, trekk]
    """
    # Linjegrupperingen er den dyre delen. Den gjøres én gang per side og
    # deles av både fnr-søket og trekk-beregningen for hver YOLO-boks.
    linjer = bygg_linjer(tokens) if tokens else []

    bokser = [[boks, "paddle", None, rec_score, vindu_trekk]
              for (boks, _mod11, rec_score, vindu_trekk)
              in finn_bokser_fra_tokens(tokens, linjer)]

    for (x0, y0, x1, y1, conf) in yolo_bokser:
        yb = (x0, y0, x1, y1)
        dekket = [par for par in bokser if overlapp_andel_boks(yb, par[0]) > DEDUP_OVERLAPP]
        # Bare Paddle-funn kan bekreftes til «begge». Tidligere ble også
        # tidligere YOLO-bokser i listen omdøpt og fikk denne boksens
        # konfidens — det kontaminerte «begge»-bøtta med rene YOLO-funn og
        # skjulte dem for OCR-reglene (som kun gjelder kilde «yolo»).
        paddle_dekket = [par for par in dekket if par[1] in ("paddle", "begge")]
        if paddle_dekket:
            for par in paddle_dekket:
                par[1] = "begge"
                par[2] = round(conf, 3)           # yolo_conf
        elif dekket:
            # Overlapper kun en tidligere YOLO-boks: duplikat — behold den
            # første, uten omdøping.
            pass
        elif kilde := _godta_yolo_boks(tokens, yb, conf):
            # Trekkene beskriver hva snill_sjekk hadde å gå på, og skrives til
            # resultat-CSV-en så strengere varianter kan feies uten ny kjøring.
            # Se boks_trekk: «yolo» får tekst-trekkene — «yolo_vertikal»
            # leser ikke tokens. Paddle/begge bærer i stedet VINDU-trekkene
            # (maks_luke/har_desimal_luke) fra finn_bokser_fra_tokens.
            trekk = trekk_for_boks(tokens, linjer, yb) if kilde == "yolo" else None
            bokser.append([yb, kilde, round(conf, 3), None, trekk])

    # ── Desimal- og linjebevis-reglene ──────────────────────────────────────────
    # Etter dedup-løkken med vilje: kilden er endelig, så bokser som ble
    # «begge» underveis beholder sladdingen sin.
    bokser = [par for par in bokser
              if not (par[1] == "yolo"
                      and (_desimalregel_forkaster(par[4], par[2])
                           or _linjebevis_forkaster(par[4], par[2])))]

    # ── Dimensjonsfiltre ────────────────────────────────────────
    # Universelle grenser for alle kilder; kun høy yolo-konfidens fritar.
    # I tillegg: kortsidekrav for «yolo» og full støyform for «paddle», som
    # begge gjelder uansett konfidens, og langsidekrav for «yolo» bak
    # conf-porten.
    bokser = [par for par in bokser
              if not er_for_liten(par[0])
              and not (par[1] == "yolo" and er_for_smal_yolo(par[0]))
              and not (par[1] == "paddle" and har_paddle_stoyform(par[0]))
              and (
                  _hopp_over_geometrifilter(par[2], par[1])
                  or (not har_feil_ratio(par[0]) and not er_for_tynn(par[0])
                      and not (par[1] == "yolo" and er_for_kort_yolo(par[0])))
              )]

    return [tuple(par) for par in bokser]


def _godta_yolo_boks(tokens, boks, conf):
    if er_vertikal(boks):
        return "yolo_vertikal" if conf >= YOLO_CONF_VERTIKAL else None
    if tokens_i_boks(tokens, boks):
        return "yolo" if snill_sjekk(tokens, boks) else None
    return "yolo" if conf >= YOLO_CONF_UTEN_TEKST else None


def _bygg_side(si, storrelse, tokens, bokser_med_kilde, k, med_linjer):
    w, h = storrelse
    bokser = []
    for boks, kilde, yolo_conf, paddle_rec_score, trekk in bokser_med_kilde:
        x0, y0, x1, y1 = boks_tilbake(boks, k, w, h)
        b = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "kilde": kilde}
        if yolo_conf is not None:
            b["yolo_conf"] = yolo_conf
        if paddle_rec_score is not None:
            b["paddle_rec_score"] = paddle_rec_score
        if trekk is not None:
            b["trekk"] = trekk
        bokser.append(b)
    side = {"side": si, "bilde_bredde": w, "bilde_hoyde": h, "bokser": bokser}
    if med_linjer:
        side["linjer"] = ocr_linjer_fra_tokens(tokens)
    return side


def _sidegeometri(pdf_bytes):
    """Sidestørrelser uten å rasterisere: (piksler ved PDF_DPI, punkt).

    Pikselmålene må bli identiske med det get_pixmap(dpi=PDF_DPI) ville gitt,
    siden de brukes som bilde_bredde/bilde_hoyde når renderingen hoppes over.
    get_pixmap tar irect-en av siden skalert med zoom, så vi gjør det samme.
    """
    zoom = PDF_DPI / 72.0
    m = fitz.Matrix(zoom, zoom)
    piksler, punkt = [], []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as dok:
        for side in dok:
            r = side.rect
            ir = (r * m).irect
            piksler.append((ir.width, ir.height))
            punkt.append((r.width, r.height))
    return piksler, punkt


def _til_flat(sider, sidefelt):
    ut = []
    for side in sider:
        n = side["side"]
        bb, bh = side.get("bilde_bredde"), side.get("bilde_hoyde")
        if not bb or not bh or not (1 <= n <= len(sidefelt)):
            continue
        pt_bredde, pt_hoyde = sidefelt[n - 1]
        sx = pt_bredde / bb
        sy = pt_hoyde / bh
        for b in side.get("bokser", []):
            x0, x1 = sorted((b["x0"] * sx, b["x1"] * sx))
            y0, y1 = sorted((b["y0"] * sy, b["y1"] * sy))
            d = {"page": n, "x": x0, "y": y0,
                 "width": x1 - x0, "height": y1 - y0,
                 "kilde": b.get("kilde")}
            if "yolo_conf" in b:
                d["yolo_conf"] = b["yolo_conf"]
            if "paddle_rec_score" in b:
                d["paddle_rec_score"] = b["paddle_rec_score"]
            if "trekk" in b:
                d["trekk"] = b["trekk"]
            ut.append(d)
    return ut


def run_model_on_pdf_bytes(pdf_bytes, skriv_tid=False, med_linjer=False, navn=None,
                           elektronisk_tinglyst=False, kun_yolo=False,
                           cache_mappe=None, yolo_cache_mappe=None):
    t = {}
    bruker_yolo = kun_yolo or not elektronisk_tinglyst
    yolo_cache = bool(yolo_cache_mappe and navn and bruker_yolo)

    with _ta_tid(t, "render"):
        sidemaal, sidefelt = _sidegeometri(pdf_bytes)
    n_sider = len(sidemaal)

    # ── Forsøk å laste OCR-resultater fra cache ─────────────────
    # Rotasjonene er verdt å hente selv med --kun-yolo: de kommer fra
    # orienteringsmodellen, ikke fra tekstgjenkjenningen, og YOLO kjører på
    # det orienteringskorrigerte bildet. Uten dem måtte --kun-yolo rendre og
    # orientere hvert dokument på nytt selv med full YOLO-cache.
    rotasjoner, tokens_per_side = None, None
    ocr_treff = rot_treff = False
    if cache_mappe and navn:
        cachet = les_ocr_cache(cache_mappe, navn)
        if cachet is not None and len(cachet[0]) == n_sider:
            rotasjoner = cachet[0]
            rot_treff = True
            if not kun_yolo:
                tokens_per_side = cachet[1]
                ocr_treff = True

    # ── Forsøk å laste YOLO-bokser fra cache ────────────────────
    # YOLO kjører på det orienteringskorrigerte bildet, så oppslaget krever
    # rotasjonene. Har vi dem fra cache før render, kan rasteriseringen
    # hoppes over helt; ellers må orienteringssteget kjøre først.
    yolo_bokser_per_side = None
    if yolo_cache and rot_treff:
        yolo_bokser_per_side = les_yolo_cache(yolo_cache_mappe, navn, rotasjoner)

    # ── Rendre bare når noe faktisk trenger piksler ─────────────
    # Treff i begge cachene betyr at ingenting leser bildene: sidemålene over
    # dekker det _bygg_side trenger. Med --kun-yolo trengs ikke tokens, så
    # rotasjoner + YOLO fra cache er nok.
    mangler_ocr = not ocr_treff and not kun_yolo
    trenger_piksler = (mangler_ocr or not rot_treff
                       or (bruker_yolo and yolo_bokser_per_side is None))
    bilder = bilder_ocr = None

    if trenger_piksler:
        with _ta_tid(t, "render"):
            bilder = list(les_sider_fra_bytes(pdf_bytes))

        if not rot_treff:
            with _ta_tid(t, "orientering"):
                # Batch: ett modellkall for hele dokumentet. Per side ble
                # GPU-en startet og stoppet én gang per side.
                rotasjoner = finn_rotasjoner_batch(bilder)

        bilder_ocr = [np.rot90(b, k) if k else b for b, k in zip(bilder, rotasjoner)]

        if not ocr_treff and not kun_yolo:
            with _ta_tid(t, "ocr"):
                tokens_per_side = les_tokens_batched(bilder_ocr)
            # Lagre til cache for fremtidige kjøringer
            if cache_mappe and navn:
                skriv_ocr_cache(cache_mappe, navn, rotasjoner, tokens_per_side)

    if tokens_per_side is None:
        tokens_per_side = [[] for _ in range(n_sider)]

    # Uten treff ble rotasjonene kjent først nå, etter orienteringssteget
    if yolo_cache and not rot_treff:
        yolo_bokser_per_side = les_yolo_cache(yolo_cache_mappe, navn, rotasjoner)

    yolo_treff = yolo_bokser_per_side is not None
    # Rå bokser å skrive til cache etterpå (kun når vi faktisk kjørte modellen)
    nye_yolo_bokser = [] if (yolo_cache and not yolo_treff) else None

    sider = []
    for si in range(n_sider):
        k = rotasjoner[si]
        tokens = tokens_per_side[si]

        with _ta_tid(t, "yolo+match"):
            if not bruker_yolo:
                yolo_bokser = []
            elif yolo_treff:
                yolo_bokser = yolo_bokser_per_side[si]
            else:
                bilde_yolo = bilder_ocr[si]
                # Med cache predikerer vi ned til gulvet og filtrerer selv, slik
                # at cachen dekker senere endringer i YOLO_CONF. Boksene som
                # overlever filteret er de samme som en predict på YOLO_CONF gir.
                raa = finn_yolo_bokser(
                    bilde_yolo, conf=YOLO_CACHE_CONF_GULV if yolo_cache else None)
                if nye_yolo_bokser is not None:
                    nye_yolo_bokser.append(raa)
                yolo_bokser = [b for b in raa if b[4] >= YOLO_CONF] if yolo_cache else raa

            if kun_yolo:
                bokser_med_kilde = _finn_bokser_kun_yolo(yolo_bokser)
            else:
                bokser_med_kilde = _finn_bokser_med_kilde(tokens, yolo_bokser)

        with _ta_tid(t, "etterbehandling"):
            sider.append(_bygg_side(si + 1, sidemaal[si], tokens, bokser_med_kilde,
                                    k, med_linjer))

    if nye_yolo_bokser is not None and len(nye_yolo_bokser) == n_sider:
        skriv_yolo_cache(yolo_cache_mappe, navn, rotasjoner, nye_yolo_bokser)

    if skriv_tid:
        _skriv_tid(t, len(sider), navn, ocr_treff, yolo_treff)

    return _til_flat(sider, sidefelt)


def _skriv_tid(t, n_sider, navn=None, ocr_treff=False, yolo_treff=False):
    poster = ["render", "orientering", "ocr", "yolo+match", "etterbehandling"]
    total = sum(t.get(p, 0.0) for p in poster)

    label = f"Timing [{navn}]:" if navn else "Timing:"
    fra_cache = [n for n, treff in (("OCR", ocr_treff), ("YOLO", yolo_treff)) if treff]
    if fra_cache:
        label += f" ({' + '.join(fra_cache)} fra cache)"
    print(label)
    for post in poster:
        sek = t.get(post, 0.0)
        pct = (sek / total * 100) if total else 0.0
        print(f"  {post:<18}{sek:9.3f} s{pct:7.1f}%")
    print(f"  {'Total':<18}{total:9.3f} s")
    print(f"  {'Sider totalt':<18}{n_sider:9d}")
    if n_sider:
        print(f"  {'Per side':<18}{total / n_sider:9.3f} s")
