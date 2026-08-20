"""Forhåndsfyll OCR- og YOLO-cachen for et sett med dokumenter.

Rendrer hvert dokument én gang og fyller begge cachene i samme pass. Per
dokument gjøres bare det som mangler, så en ny modell koster kun
YOLO-delen — rotasjonene kommer gratis fra OCR-cachen, og YOLO-cachen
krever dem (den invalideres hvis rotasjonen per side ikke stemmer).

Med begge cachene varme hopper valideringen (utils/run.py) over både OCR,
YOLO og PDF-rendering, så dette er stedet å bruke maskinen — ikke i selve
valideringsløpet.

Parallellkjøringen ligger i parallell_pipeline.py; her er bare det som er
spesifikt for OCR og YOLO.

Bruk:
    python precache.py --mappe /sti/til/pdfer
    python precache.py --mappe /sti/til/pdfer --kun yolo --yolo-vekter runs/x/weights/best.pt
    python precache.py --mappe /sti/til/pdfer --gpu-prosesser 4 --start-batch 8 --profil
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*ccache.*")
os.environ["GLOG_minloglevel"] = "3"
os.environ["GLOG_v"] = "0"
os.environ["FLAGS_call_stack_level"] = "0"

_UTILS = os.path.dirname(os.path.abspath(__file__))
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)
_APP = os.path.join(_UTILS, "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

import numpy as np

import parallell_pipeline as pp
from config import SIDER_PER_OCR_BATCH
from file_selection import velg_filer
from ocr_cache import les_cache as les_ocr_cache, skriv_cache as skriv_ocr_cache
from yolo_cache import (les_cache as les_yolo_cache, skriv_cache as skriv_yolo_cache,
                        cache_mappe_for_vekter)

STANDARD_VEKTER = os.path.join(_APP, "weights", "best.pt")


# ── Behandler (kjører i arbeidsprosessen) ────────────────────────

def lag_behandler(oppgave):
    """Bygg funksjonen som gjør OCR + YOLO for en gruppe dokumenter.

    Kalles i arbeidsprosessen, etter fork og etter at minneflaggene er
    satt — derfor ligger de tunge importene her.
    """
    e = oppgave["ekstra"]
    tider = oppgave["tider"]
    ocr_mappe, yolo_mappe = e["ocr_mappe"], e["yolo_mappe"]
    ocr_batch = e["ocr_batch"]
    force = e["force"]

    from orientering import finn_rotasjoner_batch

    les_tokens_batched = None
    if ocr_mappe:
        from paddle_ocr_model_fnr import les_tokens_batched

    finn_yolo_bokser = None
    if yolo_mappe:
        import yolo_fnr
        yolo_fnr.sett_vekter(e["vekter"])
        finn_yolo_bokser = yolo_fnr.finn_yolo_bokser

    from config import YOLO_CACHE_CONF_GULV

    def _tid(post, t0):
        tider[post] = tider.get(post, 0.0) + (time.perf_counter() - t0)

    # Varm opp modellene — engangskostnad som ikke skal med i estimatene
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    finn_rotasjoner_batch([dummy])
    if les_tokens_batched:
        les_tokens_batched([dummy])
    if finn_yolo_bokser:
        finn_yolo_bokser(dummy, conf=YOLO_CACHE_CONF_GULV)

    def behandle(gruppe, sider_om_gangen=None):
        # ── 1. Hva mangler per dokument? ──────────────────────────
        # Cachet OCR gir rotasjonene gratis, og YOLO-cachen trenger dem.
        jobber = []
        for navn, bilder in gruppe:
            cachet = None if force or not ocr_mappe else les_ocr_cache(ocr_mappe, navn)
            rotasjoner = list(cachet[0]) if cachet and len(cachet[0]) == len(bilder) else None
            trenger_yolo = bool(yolo_mappe) and (
                force or rotasjoner is None
                or les_yolo_cache(yolo_mappe, navn, rotasjoner) is None)
            jobber.append({
                "navn": navn, "bilder": bilder, "rotasjoner": rotasjoner,
                "tokens": cachet[1] if cachet else None,
                "trenger_ocr": bool(ocr_mappe) and cachet is None,
                "trenger_yolo": trenger_yolo,
            })

        # ── 2. Orientering for de som mangler rotasjoner ──────────
        mangler_rot = [j for j in jobber if j["rotasjoner"] is None]
        if mangler_rot:
            t0 = time.perf_counter()
            sider = [b for j in mangler_rot for b in j["bilder"]]
            steg = sider_om_gangen or max(4 * ocr_batch, 8)
            alle = []
            for i in range(0, len(sider), steg):
                alle.extend(finn_rotasjoner_batch(sider[i:i + steg]))
            _tid("orientering", t0)
            i = 0
            for j in mangler_rot:
                j["rotasjoner"] = alle[i:i + len(j["bilder"])]
                i += len(j["bilder"])

        # ── 3. Roter én gang — både OCR og YOLO bruker samme bilde ─
        t0 = time.perf_counter()
        for j in jobber:
            if j["trenger_ocr"] or j["trenger_yolo"]:
                j["rotert"] = [np.rot90(b, k) if k else b
                               for b, k in zip(j["bilder"], j["rotasjoner"])]
        _tid("rotasjon", t0)

        # ── 4. OCR i én batch over alle dokumentene som mangler ───
        ocr_jobber = [j for j in jobber if j["trenger_ocr"]]
        if ocr_jobber:
            t0 = time.perf_counter()
            sider = [b for j in ocr_jobber for b in j["rotert"]]
            tokens = les_tokens_batched(sider, batch_size=sider_om_gangen or ocr_batch)
            _tid("ocr", t0)
            i = 0
            for j in ocr_jobber:
                j["tokens"] = tokens[i:i + len(j["bilder"])]
                i += len(j["bilder"])

        # ── 5. YOLO, én side om gangen (ultralytics-API-et her) ───
        for j in jobber:
            if not j["trenger_yolo"]:
                continue
            t0 = time.perf_counter()
            j["yolo"] = [finn_yolo_bokser(b, conf=YOLO_CACHE_CONF_GULV)
                         for b in j["rotert"]]
            _tid("yolo", t0)

        # ── 6. Skriv cachene ─────────────────────────────────────
        t0 = time.perf_counter()
        resultater = []
        for j in jobber:
            if j["trenger_ocr"]:
                skriv_ocr_cache(ocr_mappe, j["navn"], j["rotasjoner"], j["tokens"])
            if j["trenger_yolo"]:
                skriv_yolo_cache(yolo_mappe, j["navn"], j["rotasjoner"], j["yolo"])

            deler = []
            if j["tokens"] is not None:
                deler.append(f"{sum(len(t) for t in j['tokens'])} tokens"
                             + ("" if j["trenger_ocr"] else " (cachet)"))
            if j.get("yolo") is not None:
                deler.append(f"{sum(len(b) for b in j['yolo'])} yolo-bokser")
            elif j["trenger_yolo"] is False and yolo_mappe:
                deler.append("yolo cachet")
            if any(k != 0 for k in j["rotasjoner"]):
                deler.append("rot=[" + ",".join(f"{k * 90}°" for k in j["rotasjoner"]) + "]")
            resultater.append({"navn": j["navn"], "sider": len(j["bilder"]),
                               "tekst": ", ".join(deler)})
        _tid("cache", t0)
        return resultater

    def nullstill():
        """Riv ned modellene så allokatoren gir minnet tilbake.

        Siste utvei når prosessen går tom for minne innenfor sitt eget tak:
        da er det dens egen cache som er full, og empty_cache() alene får
        den ikke ned. Modellene er lazy og bygges opp igjen ved neste kall
        (~10 s), mot at dokumentet faktisk blir gjort.
        """
        import orientering
        orientering._orient = None
        if ocr_mappe:
            import paddle_ocr_model_fnr
            paddle_ocr_model_fnr.reader = None
        if yolo_mappe:
            import yolo_fnr as yf
            yf._modell = None
        pp.frigjør_gpu_cache()

    behandle.nullstill = nullstill
    return behandle


# ── Filvalg ──────────────────────────────────────────────────────

def _utled_cache_base(args):
    """Basemappe for cachene: --cache, ellers $SLADD_CACHE/<uttrekk>/."""
    if args.cache:
        return args.cache
    base = os.environ.get("SLADD_CACHE")
    if base:
        return os.path.join(base, os.path.basename(os.path.normpath(args.mappe)))
    return None


def _mangler_cache(filer, ocr_mappe, yolo_mappe):
    """Behold filer der minst én av de aktive cachene mangler.

    YOLO-cachen valideres mot rotasjonene, så uten OCR-cache regnes YOLO
    som manglende — den kan uansett ikke leses uten dem.
    """
    ut = []
    for f in filer:
        navn = os.path.basename(f)
        cachet = les_ocr_cache(ocr_mappe, navn) if ocr_mappe else None
        if ocr_mappe and cachet is None:
            ut.append(f)
            continue
        if yolo_mappe:
            rotasjoner = list(cachet[0]) if cachet else None
            if rotasjoner is None or les_yolo_cache(yolo_mappe, navn, rotasjoner) is None:
                ut.append(f)
    return ut


def main():
    pp.sett_opp_loggfil("precache")

    p = argparse.ArgumentParser(
        description="Forhåndsfyll OCR- og YOLO-cachen. Rendrer hvert dokument "
                    "én gang og gjør bare det som mangler.")
    p.add_argument("--mappe", required=True, help="mappe med PDF-filer")
    p.add_argument("--cache", default=None,
                   help="basemappe for cachene (default: $SLADD_CACHE/<uttrekk>/)")
    p.add_argument("--kun", choices=("ocr", "yolo", "begge"), default="begge",
                   help="hvilke cacher som skal fylles (default: begge)")
    p.add_argument("--yolo-vekter", default=STANDARD_VEKTER,
                   help="vektfil for YOLO. Cachen er per vektfil")
    p.add_argument("--velg", nargs="*", default=[],
                   help="spesifikke filer (filnavn/delstreng)")
    p.add_argument("--velg-fra-fil", default=None,
                   help="les fil-IDer fra en tekstfil (én per linje)")
    p.add_argument("--antall", default="alle",
                   help="antall filer når --velg er tom (tall, eller 'alle')")
    p.add_argument("--ocr-batch", type=int, default=None,
                   help=f"sider per OCR-batch (default: {SIDER_PER_OCR_BATCH} fra "
                        "config, delt på antall prosesser). Største driver av "
                        "GPU-minnetoppen — deteksjonsmodellen holder "
                        "aktiveringer for hele sidebatchen samtidig")
    p.add_argument("--rec-batch", type=int, default=0,
                   help="tekstlinjer per gjenkjennings-batch (0=auto ut fra "
                        "minnetaket per prosess). Påvirker kun minne og "
                        "hastighet, ikke resultatet")
    p.add_argument("--hpi", action="store_true",
                   help="aktiver High Performance Inference (TensorRT)")
    p.add_argument("--force", action="store_true",
                   help="kjør på nytt selv om dokumentet allerede er cachet")
    pp.legg_til_argumenter(p)
    args = p.parse_args()

    if not os.path.isdir(args.mappe):
        print(f"FEIL: --mappe finnes ikke: {args.mappe}")
        return 1
    if args.velg_fra_fil and not os.path.isfile(args.velg_fra_fil):
        print(f"FEIL: --velg-fra-fil finnes ikke: {args.velg_fra_fil}")
        return 1

    cache_base = _utled_cache_base(args)
    if not cache_base:
        print("FEIL: Ingen cache-mappe angitt. Bruk --cache eller sett $SLADD_CACHE "
              "(server-standard: /data2/cache).")
        return 1

    ocr_mappe = os.path.join(cache_base, "ocr") if args.kun in ("ocr", "begge") else None
    yolo_mappe = None
    if args.kun in ("yolo", "begge"):
        if not os.path.isfile(args.yolo_vekter):
            print(f"FEIL: Finner ikke YOLO-vekter: {args.yolo_vekter}")
            return 1
        yolo_mappe = cache_mappe_for_vekter(os.path.join(cache_base, "yolo"),
                                            args.yolo_vekter)
    for m in (ocr_mappe, yolo_mappe):
        if m:
            os.makedirs(m, exist_ok=True)

    # ── Filliste ─────────────────────────────────────────────────
    velg, fra_fil = args.velg, False
    if args.velg_fra_fil:
        with open(args.velg_fra_fil, encoding="utf-8") as f:
            velg = [linje.strip() for linje in f if linje.strip()]
        fra_fil = True
        print(f"Leste {len(velg)} IDer fra {args.velg_fra_fil}")

    filer = velg_filer(args.mappe, velg, args.antall, eksakt=fra_fil)
    if not filer:
        print("Ingen filer å behandle — sjekk --mappe / --velg / --antall.")
        return 1
    print(f"Filer funnet: {len(filer)}")
    if ocr_mappe:
        print(f"OCR-cache:   {ocr_mappe}")
    if yolo_mappe:
        print(f"YOLO-cache:  {yolo_mappe}")
        print(f"YOLO-vekter: {args.yolo_vekter}")

    if not args.force:
        opprinnelig = len(filer)
        filer = _mangler_cache(filer, ocr_mappe, yolo_mappe)
        if opprinnelig - len(filer):
            print(f"Hopper over: {opprinnelig - len(filer)} allerede cachet, "
                  f"{len(filer)} gjenstår")
    if not filer:
        print("Alt er allerede cachet!")
        return 0

    # ── Oppsett ──────────────────────────────────────────────────
    if args.hpi:
        os.environ["SLADD_HPI"] = "1"
    opts = pp.oppsett(args)
    n = opts["n_prosesser"]

    # Gjenkjenningsmodellen kjører 128 linjer per batch som default, og én
    # slik batch ba om 1,4 GB i målingene. Med et tak på ~6,5 GB per prosess
    # sprekker den så snart allokatoren har vokst litt, og siden minnet er
    # prosessens eget hjelper det ikke å vente. 32 linjer holdt kortet på
    # 69 % uten en eneste minnefeil. Leses av app/paddle_ocr_model_fnr.py.
    rec_batch = args.rec_batch
    if not rec_batch and opts["gpu_mb"]:
        rec_batch = 32 if opts["gpu_mb"] < 10000 else 64
    if rec_batch:
        os.environ["SLADD_REC_BATCH"] = str(rec_batch)
        print(f"  Rec-batch:  {rec_batch} tekstlinjer"
              + ("" if args.rec_batch else " (auto ut fra minnetaket)"))
    if args.ocr_batch:
        ocr_batch = args.ocr_batch
    else:
        gpu = pp.gpu_minne_info()
        grunn = 32 if gpu and gpu[1] >= 24000 else 16 if gpu and gpu[1] >= 16000 else SIDER_PER_OCR_BATCH
        ocr_batch = max(grunn // n, 4)
    opts["ekstra"] = dict(ocr_mappe=ocr_mappe, yolo_mappe=yolo_mappe,
                          vekter=args.yolo_vekter, ocr_batch=ocr_batch,
                          force=args.force)
    print(f"  OCR-batch:  {ocr_batch} sider  |  filene deles dynamisk mellom "
          f"{n} prosess(er)")
    pp.skriv_ressursstatus()

    ferdig, _, feilet = pp.kjør(filer, lag_behandler, opts)
    if ocr_mappe:
        print(f"  OCR-cache:  {ocr_mappe}")
    if yolo_mappe:
        print(f"  YOLO-cache: {yolo_mappe}")
    # Ufullført teller som feil: krasjer alle prosessene, står det ingen
    # feilede dokumenter i lista, men jobben er like fullt ikke gjort.
    if feilet or ferdig < len(filer):
        print(f"  Ikke ferdig: {len(filer) - ferdig} dokumenter gjenstår")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
