import argparse
import io
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout

# Demp irrelevante advarsler (PaddlePaddle ccache etc.)
warnings.filterwarnings("ignore", message=".*ccache.*")
os.environ["GLOG_minloglevel"] = "2"    # demp PaddlePaddle C++ logging

import fitz

_UTILS = os.path.dirname(os.path.abspath(__file__))
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)

_APP = os.path.join(_UTILS, "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from file_selection import velg_filer
from model_main import run_model_on_pdf_bytes
from csv_export import initialiser_csv, append_csv
from evaluation import mal_overlapp, les_fasit, _dok_nr
from visualization import tegn_og_lagre
from redaction import sladd_alle
from yolo_fnr import sett_vekter
from load_pdf import PDF_DPI
import traceback
from save_result import lagre_resultat

import time
import csv as csv_modul

from utils_config import (
    MAPPE, ANTALL, FASIT_CSV, CSV_UT, OCR_LOGG_FIL,
    PNG_MAPPE, SLADD_MAPPE, Y_ORIGIN, TERSKEL
)

SKALA = PDF_DPI / 72.0                     # PDF-punkt -> piksel


def _sider_fra_resultat(resultat, pdf_bytes):
    if isinstance(resultat, dict):
        return resultat.get("sider", [])

    per_side = defaultdict(list)
    for b in resultat:
        per_side[b["page"]].append(b)

    sider = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as dok:
        for n in range(1, dok.page_count + 1):
            rect = dok[n - 1].rect
            bw = int(round(rect.width * SKALA))
            bh = int(round(rect.height * SKALA))
            bokser = []
            for b in per_side.get(n, []):
                boks = {
                    "x0": b["x"] * SKALA,
                    "y0": b["y"] * SKALA,
                    "x1": (b["x"] + b["width"]) * SKALA,
                    "y1": (b["y"] + b["height"]) * SKALA,
                    "kilde": b.get("kilde", "paddle"),
                }
                if b.get("conf") is not None:
                    boks["conf"] = b["conf"]
                bokser.append(boks)
            sider.append({"side": n, "bilde_bredde": bw, "bilde_hoyde": bh,
                          "bokser": bokser})
    return sider


def _skriv_ocr_logg(ocr_linjer, sti):
    n_sider = 0
    with open(sti, "w", encoding="utf-8") as logg:
        for (navn, si) in sorted(ocr_linjer):
            linjer = ocr_linjer[(navn, si)]
            logg.write(f"\n===== {navn} side {si} - {len(linjer)} tekstlinjer =====\n")
            for li, (tekst, merker) in enumerate(linjer, start=1):
                if merker:
                    merk = ", ".join(
                        f"{cifre} (mod11 {'OK' if ok else 'FEIL'})" for cifre, ok in merker)
                    logg.write(f"  linje {li:>2}: {tekst!r}   <-- FNR-TREFF: {merk}\n")
                else:
                    logg.write(f"  linje {li:>2}: {tekst!r}\n")
            n_sider += 1
    return n_sider


def _les_ferdige_fra_csv(csv_sti):
    """Les allerede prosesserte filnavn fra en eksisterende CSV (for --fortsett)."""
    ferdige = set()
    if os.path.isfile(csv_sti) and os.path.getsize(csv_sti) > 0:
        with open(csv_sti, newline="", encoding="utf-8") as f:
            for rad in csv_modul.DictReader(f):
                ferdige.add(rad["navn"])
    return ferdige


def _les_filer_fra_fil(sti):
    """Les en liste med filnavn/IDer fra en tekstfil (én per linje)."""
    with open(sti, encoding="utf-8") as f:
        return [linje.strip() for linje in f if linje.strip()]


def _tegn_fortlopende(navn, sider, mappe, png_mappe, fasit, y_origin, csv_bokser_dok, sladd_bokser_dok):
    """Tegn PNG for ett dokument umiddelbart etter inferens."""
    tegn_og_lagre(sladd_bokser_dok, fasit, mappe, png_mappe,
                  y_origin=y_origin, skriv_logg=True, rydd=False, kilder=csv_bokser_dok)


def _evaluer_og_tegn_feil(sladd_dok, csv_dok, fasit, mappe, png_mappe,
                           terskel, y_origin):
    """Evaluer ett dokument mot fasit og tegn feilbilder hvis det finnes feil.

    Lagrer i undermapper: bom/ (manglende deteksjoner) og oversladd/ (over-sladding).
    En side kan havne i begge mapper hvis den har begge typer feil.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        dok_eval = mal_overlapp(sladd_dok, fasit, mappe, terskel=terskel,
                                y_origin=y_origin, kilder=csv_dok)
    if not dok_eval:
        return

    har_bom = bool(dok_eval.get("bom_filer"))
    har_over = bool(dok_eval.get("overflod_filer"))
    if not har_bom and not har_over:
        return

    bom_indekser = {
        (_dok_nr(d["fil"]), d["side"], d["fasit_nr"] - 1)
        for d in dok_eval.get("detaljer", [])
        if d["resultat"] == "MANGLER"
    }
    oversladd = dok_eval.get("oversladd_bokser", None)

    bom_sider = {(bf["fil"], bf["side"]) for bf in dok_eval.get("bom_filer", [])}
    over_sider = {(of["fil"], of["side"]) for of in dok_eval.get("overflod_filer", [])}

    if bom_sider:
        bom_mappe = os.path.join(png_mappe, "bom")
        os.makedirs(bom_mappe, exist_ok=True)
        sladd_b = {k: v for k, v in sladd_dok.items() if k in bom_sider}
        csv_b = {k: v for k, v in csv_dok.items() if k in bom_sider}
        # Filtrer fasit til kun bom-sider så _sider_aa_tegne ikke legger til
        # sider uten feil (som ville blitt tomme PNG-er)
        bom_nr_sider = {(_dok_nr(navn), si) for (navn, si) in bom_sider}
        fasit_bom = {k: v for k, v in fasit.items() if k in bom_nr_sider} if fasit else None
        tegn_og_lagre(sladd_b, fasit_bom, mappe, bom_mappe,
                      y_origin=y_origin, skriv_logg=False, rydd=False, kilder=csv_b,
                      oversladd_bokser=oversladd, bom_indekser=bom_indekser)

    if over_sider:
        over_mappe = os.path.join(png_mappe, "oversladd")
        os.makedirs(over_mappe, exist_ok=True)
        sladd_o = {k: v for k, v in sladd_dok.items() if k in over_sider}
        csv_o = {k: v for k, v in csv_dok.items() if k in over_sider}
        # Filtrer fasit til kun oversladd-sider
        over_nr_sider = {(_dok_nr(navn), si) for (navn, si) in over_sider}
        fasit_over = {k: v for k, v in fasit.items() if k in over_nr_sider} if fasit else None
        tegn_og_lagre(sladd_o, fasit_over, mappe, over_mappe,
                      y_origin=y_origin, skriv_logg=False, rydd=False, kilder=csv_o,
                      oversladd_bokser=oversladd, bom_indekser=bom_indekser)


def main():
    p = argparse.ArgumentParser(
        description="Kjor modellen lokalt som om filene var POST-er: "
                    "bytes -> run_model_on_pdf_bytes. Flagg legger til CSV/PNG/fasit/sladding.")
    p.add_argument("--mappe", default=MAPPE, help="mappe med PDF-er (spiller rollen som POST-body)")
    p.add_argument("--velg", nargs="*", default=[], help="spesifikke filer (filnavn/delstreng)")
    p.add_argument("--velg-fra-fil", default=None,
                   help="les fil-IDer fra en tekstfil (én per linje), brukes som --velg")
    p.add_argument("--antall", default=ANTALL, help="antall filer naar --velg er tom (tall, eller 'alle')")
    p.add_argument("--csv", action="store_true", help="skriv alle funne bokser til CSV")
    p.add_argument("--png", action="store_true", help="tegn funne + fasit-bokser til PNG")
    p.add_argument("--fasit", action="store_true", help="maal recall mot fasit")
    p.add_argument("--sladd", action="store_true", help="lag faktisk sladdede PDF-er")
    p.add_argument("--ocr-logg", action="store_true", help="skriv OCR-teksten linje for linje til fil")
    p.add_argument("--fasit-csv", default=FASIT_CSV, help="fasit-CSV")
    p.add_argument("--csv-ut", default=CSV_UT, help="hvor boks-CSV-en skrives")
    p.add_argument("--ocr-logg-fil", default=OCR_LOGG_FIL, help="hvor OCR-loggen skrives")
    p.add_argument("--png-mappe", default=PNG_MAPPE, help="hvor PNG-ene lagres")
    p.add_argument("--sladd-mappe", default=SLADD_MAPPE, help="hvor sladdede PDF-er lagres")
    p.add_argument("--terskel", type=float, default=TERSKEL, help="andel fasit-areal for TRUFFET")
    p.add_argument("--y-origin", choices=["topp", "bunn"], default=Y_ORIGIN, help="CSV y-origo")
    p.add_argument("--elektronisk-tinglyst", action="store_true",
                   help="behandle som elektronisk tinglyst: uten YOLO, med bredde-filter")
    p.add_argument("--kun-yolo", action="store_true",
                   help="kjør kun YOLO uten Paddle OCR")
    p.add_argument("--kun-feil", action="store_true",
                   help="generer kun PNG for sider med bom eller over-sladding (krever fasit)")
    p.add_argument("--tid", action="store_true", help="skriv timing (render/ocr/etterbehandling) per dokument")
    p.add_argument("--beskrivelse", default=None, help="valgfritt suffiks i mappenavnet for resultatet")
    p.add_argument("--yolo-vekter", default=None,
                   help="path til YOLO-vektfil (best.pt); default er weights/weights/best.pt i app-mappen")
    p.add_argument("--fortsett", action="store_true",
                   help="fortsett fra der forrige kjøring stoppet (hopper over filer allerede i CSV)")
    p.add_argument("--resultat-mappe", default=".",
                   help="mappe der result-* undermappen opprettes (default: gjeldende mappe)")
    p.add_argument("--overskriv", action="store_true",
                   help="overskriv eksisterende CSV uten å spørre")
    p.add_argument("--ocr-cache", default=None,
                   help="mappe for per-dokument OCR-cache (tokens + orientering). "
                        "Standard: $SLADD_CACHE/<uttrekk-navn>/ocr/ utledet fra --mappe. "
                        "Sett til eksplisitt sti for å overstyre.")
    p.add_argument("--no-ocr-cache", action="store_true",
                   help="deaktiver OCR-cache helt")
    args = p.parse_args()

    # ── Utled OCR-cache-sti ──────────────────────────────────────
    if args.no_ocr_cache:
        args.ocr_cache = None
    elif args.ocr_cache is None:
        # Auto-utled fra SLADD_CACHE + mappenavnet til uttrekket
        cache_base = os.environ.get("SLADD_CACHE")
        if cache_base:
            uttrekk_navn = os.path.basename(os.path.normpath(args.mappe))
            args.ocr_cache = os.path.join(cache_base, uttrekk_navn, "ocr")

    # ── Tidlig validering av inputfiler ─────────────────────────
    if args.velg_fra_fil and not os.path.isfile(args.velg_fra_fil):
        print(f"FEIL: --velg-fra-fil ikke funnet: {args.velg_fra_fil}")
        return

    if (args.fasit or args.kun_feil) and not os.path.isfile(args.fasit_csv):
        print(f"FEIL: --fasit-csv ikke funnet: {args.fasit_csv}")
        print(f"      Spesifiser riktig sti med --fasit-csv /sti/til/labels.csv")
        return

    if args.yolo_vekter and not os.path.isfile(args.yolo_vekter):
        print(f"FEIL: --yolo-vekter ikke funnet: {args.yolo_vekter}")
        return

    if not os.path.isdir(args.mappe):
        print(f"FEIL: --mappe ikke funnet: {args.mappe}")
        return

    sett_vekter(args.yolo_vekter)

    # ── Opprett utdata-mapper automatisk ─────────────────────────
    utdata_mapper = [m for m in [
        os.path.dirname(args.csv_ut) if args.csv else None,
        args.png_mappe if (args.png or args.kun_feil) else None,
        args.sladd_mappe if args.sladd else None,
        args.resultat_mappe if args.fasit else None,
    ] if m]

    if not args.fortsett and not args.overskriv:
        eksisterende = [m for m in utdata_mapper if os.path.isdir(m)]
        if eksisterende:
            print("FEIL: Utdata-mappe(r) finnes allerede:")
            for m in eksisterende:
                print(f"      {m}")
            print("      Bruk --fortsett for å gjenoppta, eller --overskriv for å starte på nytt.")
            return

    for mappe in utdata_mapper:
        os.makedirs(mappe, exist_ok=True)

    # Bygg fillisten
    velg = args.velg
    fra_fil = False
    if args.velg_fra_fil:
        velg = _les_filer_fra_fil(args.velg_fra_fil)
        fra_fil = True
        print(f"Leste {len(velg)} IDer fra {args.velg_fra_fil}")

    filer = velg_filer(args.mappe, velg, args.antall, eksakt=fra_fil)

    if not filer:
        print("Ingen filer aa behandle - sjekk --mappe / --velg / --antall.")
        return

    if args.ocr_cache:
        os.makedirs(args.ocr_cache, exist_ok=True)
        n_cachet = sum(1 for f in filer
                       if os.path.isfile(os.path.join(args.ocr_cache,
                           os.path.splitext(os.path.basename(f))[0] + ".json")))
        print(f"OCR-cache: {args.ocr_cache} ({n_cachet}/{len(filer)} dokumenter cachet)")

    # Resume: hopp over allerede prosesserte filer
    hoppet_over = 0
    if args.fortsett and args.csv and os.path.isfile(args.csv_ut):
        ferdige = _les_ferdige_fra_csv(args.csv_ut)
        opprinnelig = len(filer)
        filer = [f for f in filer if os.path.basename(f) not in ferdige]
        hoppet_over = opprinnelig - len(filer)
        if hoppet_over:
            print(f"--fortsett: hopper over {hoppet_over} allerede prosesserte, {len(filer)} gjenstår")
    elif args.csv and not args.fortsett:
        if os.path.isfile(args.csv_ut) and os.path.getsize(args.csv_ut) > 0 and not args.overskriv:
            n_eksisterende = len(_les_ferdige_fra_csv(args.csv_ut))
            if n_eksisterende:
                print(f"FEIL: {args.csv_ut} inneholder allerede {n_eksisterende} dokumenter.")
                print(f"      Bruk --fortsett for å fortsette, eller --overskriv for å starte på nytt.")
                return
        initialiser_csv(args.csv_ut)
        print(f"Starter kontinuerlig skriving til {args.csv_ut}")

    total_tid = 0
    totalt_antall = len(filer)

    vil_ha_artefakt = args.csv or args.png or args.fasit or args.sladd or args.kun_feil

    # Forhåndslast fasit
    fasit = None
    if args.png or args.kun_feil or args.fasit:
        fasit = les_fasit(args.fasit_csv) if os.path.isfile(args.fasit_csv) else None
    if args.png or args.kun_feil:
        os.makedirs(args.png_mappe, exist_ok=True)

    sladd_bokser, yolo_bokser, csv_bokser, feilet = {}, {}, {}, []
    tider = {}
    ocr_linjer = {}                          # (navn, side) -> liste av (tekst, merker)
    advart_om_linjer = False

    # Bakgrunnstråder for PNG-generering (CPU) mens GPU jobber videre
    png_executor = ThreadPoolExecutor(max_workers=2) if (args.png and not args.kun_feil) else None
    png_futures = []
    feil_executor = ThreadPoolExecutor(max_workers=2) if args.kun_feil else None
    feil_futures = []

    # Pre-les neste fil mens GPU jobber
    neste_bytes = None
    if filer:
        with open(filer[0], "rb") as f:
            neste_bytes = f.read()

    for i, fil in enumerate(filer, start=1):
        start = time.perf_counter()

        navn = os.path.basename(fil)
        print(f"\n[{i}/{totalt_antall}] → {navn}")

        pdf_bytes = neste_bytes

        # Pre-les neste fil i parallell med inferens
        if i < totalt_antall:
            # Les neste fil nå (rask I/O, ferdig før GPU er done)
            with open(filer[i], "rb") as f:
                neste_bytes = f.read()

        try:
            resultat = run_model_on_pdf_bytes(pdf_bytes, skriv_tid=args.tid, med_linjer=args.ocr_logg, navn=navn,
                                              elektronisk_tinglyst=args.elektronisk_tinglyst,
                                              kun_yolo=args.kun_yolo,
                                              cache_mappe=args.ocr_cache)
        except Exception as e:
            feilet.append((navn, repr(e)))
            traceback.print_exc()
            continue

        sider = _sider_fra_resultat(resultat, pdf_bytes)

        tid_brukt = time.perf_counter() - start
        total_tid += tid_brukt
        tider[navn] = tid_brukt

        if args.ocr_logg:
            if isinstance(resultat, list) and not advart_om_linjer:
                print("  !! --ocr-logg: modellen returnerer flatt format uten 'linjer' - loggen blir tom.")
                advart_om_linjer = True
            for side in sider:
                ocr_linjer[(navn, side["side"])] = side.get("linjer", [])

        if not vil_ha_artefakt:
            n = sum(len(s["bokser"]) for s in sider)
            snitt = total_tid / i
            gjenstaar = snitt * (totalt_antall - i)
            print(f"  {n} boks(er), {len(sider)} side(r) — {tid_brukt:.2f}s (est. gjenstår: {gjenstaar:.0f}s)")
            continue

        sladd_dok = {}
        csv_dok = {}
        for side in sider:
            bokser     = [(b["x0"], b["y0"], b["x1"], b["y1"]) for b in side["bokser"]]
            med_kilde  = [(b["x0"], b["y0"], b["x1"], b["y1"], b.get("kilde", "paddle"), b.get("conf"))
                          for b in side["bokser"]]
            yolo_bare  = [(b["x0"], b["y0"], b["x1"], b["y1"]) for b in side["bokser"]
                          if b.get("kilde") in ("yolo", "begge")]
            sladd_bokser[(navn, side["side"])] = (
                side["bilde_bredde"], side["bilde_hoyde"], bokser)
            sladd_dok[(navn, side["side"])] = (
                side["bilde_bredde"], side["bilde_hoyde"], bokser)
            csv_bokser[(navn, side["side"])] = (
                side["bilde_bredde"], side["bilde_hoyde"], med_kilde)
            csv_dok[(navn, side["side"])] = (
                side["bilde_bredde"], side["bilde_hoyde"], med_kilde)
            if yolo_bare:
                yolo_bokser[(navn, side["side"])] = (
                    side["bilde_bredde"], side["bilde_hoyde"], yolo_bare)

        # Fortløpende CSV
        if args.csv:
            append_csv(csv_dok, args.csv_ut)

        # Fortløpende PNG i bakgrunnen (ikke ved --kun-feil, de trenger eval)
        if png_executor:
            fut = png_executor.submit(
                _tegn_fortlopende, navn, sider, args.mappe, args.png_mappe,
                fasit, args.y_origin, csv_dok, sladd_dok)
            png_futures.append(fut)

        # Fortløpende eval + feilbilder i bakgrunnen
        if feil_executor and fasit:
            fut = feil_executor.submit(
                _evaluer_og_tegn_feil, sladd_dok, csv_dok,
                fasit, args.mappe, args.png_mappe, args.terskel, args.y_origin)
            feil_futures.append(fut)

        n = sum(len(s["bokser"]) for s in sider)
        snitt = total_tid / i
        gjenstaar = snitt * (totalt_antall - i)
        print(f"  {n} boks(er), {len(sider)} side(r) — {tid_brukt:.2f}s (est. gjenstår: {gjenstaar:.0f}s)")

    # Vent på at alle bakgrunns-PNG-er er ferdige
    if png_futures:
        print(f"\nVenter på {len(png_futures)} PNG-jobber i bakgrunnen...")
        for fut in png_futures:
            try:
                fut.result()
            except Exception as e:
                print(f"  PNG-feil: {e!r}")
        png_executor.shutdown(wait=False)

    if feil_futures:
        print(f"\nVenter på {len(feil_futures)} feilbilde-jobber i bakgrunnen...")
        for fut in feil_futures:
            try:
                fut.result()
            except Exception as e:
                print(f"  Feilbilde-feil: {e!r}")
        feil_executor.shutdown(wait=False)

    print(f"\nFerdig! {totalt_antall} dokumenter på {total_tid:.1f}s ({total_tid/max(totalt_antall,1):.2f}s/dok)")

    if feilet:
        print(f"Feilet ({len(feilet)}):", feilet[:5])

    if args.ocr_logg:
        n = _skriv_ocr_logg(ocr_linjer, args.ocr_logg_fil)
        print(f"Skrev OCR-linjer for {n} side(r) til {args.ocr_logg_fil}")

    if not vil_ha_artefakt and not args.kun_feil:
        return

    # Kjør evaluering (for sammendrag + lagre_resultat)
    eval_resultat = None
    if args.fasit or args.kun_feil:
        buf = io.StringIO()
        with redirect_stdout(buf):
            eval_resultat = mal_overlapp(sladd_bokser, fasit, args.mappe, terskel=args.terskel,
                                         y_origin=args.y_origin, kilder=csv_bokser)
        logg = buf.getvalue()
        print(logg, end="")  # vis fortsatt i terminalen
        if args.fasit:
            tid_linjer = "".join(f"  {n}: {t:.2f}s\n" for n, t in sorted(tider.items()))
            header = (
                f"Mappe:     {os.path.abspath(args.mappe)}\n"
                f"Fasit-CSV: {os.path.abspath(args.fasit_csv)}\n"
                f"Total tid: {total_tid:.2f}s\n"
                f"Tid per dokument:\n{tid_linjer}\n"
            )
            lagre_resultat(eval_resultat, mappe=args.resultat_mappe, beskrivelse=args.beskrivelse, logg=header + logg)

    # --kun-feil uten feil_executor (fallback, bør ikke skje)
    if args.kun_feil and eval_resultat and not feil_futures:
        bom_indekser = {
            (_dok_nr(d["fil"]), d["side"], d["fasit_nr"] - 1)
            for d in eval_resultat.get("detaljer", [])
            if d["resultat"] == "MANGLER"
        }
        oversladd = eval_resultat.get("oversladd_bokser", None)
        bom_sider = {(bf["fil"], bf["side"]) for bf in eval_resultat.get("bom_filer", [])}
        over_sider = {(of["fil"], of["side"]) for of in eval_resultat.get("overflod_filer", [])}
        alle_feil_sider = bom_sider | over_sider
        print(f"\n--kun-feil: tegner {len(alle_feil_sider)} side(r) med feil (bom/ og oversladd/)")

        if bom_sider:
            bom_mappe = os.path.join(args.png_mappe, "bom")
            os.makedirs(bom_mappe, exist_ok=True)
            sladd_b = {k: v for k, v in sladd_bokser.items() if k in bom_sider}
            csv_b = {k: v for k, v in csv_bokser.items() if k in bom_sider}
            bom_nr_sider = {(_dok_nr(navn), si) for (navn, si) in bom_sider}
            fasit_bom = {k: v for k, v in fasit.items() if k in bom_nr_sider} if fasit else None
            tegn_og_lagre(sladd_b, fasit_bom, args.mappe, bom_mappe,
                          y_origin=args.y_origin, kilder=csv_b,
                          oversladd_bokser=oversladd, bom_indekser=bom_indekser)

        if over_sider:
            over_mappe = os.path.join(args.png_mappe, "oversladd")
            os.makedirs(over_mappe, exist_ok=True)
            sladd_o = {k: v for k, v in sladd_bokser.items() if k in over_sider}
            csv_o = {k: v for k, v in csv_bokser.items() if k in over_sider}
            over_nr_sider = {(_dok_nr(navn), si) for (navn, si) in over_sider}
            fasit_over = {k: v for k, v in fasit.items() if k in over_nr_sider} if fasit else None
            tegn_og_lagre(sladd_o, fasit_over, args.mappe, over_mappe,
                          y_origin=args.y_origin, kilder=csv_o,
                          oversladd_bokser=oversladd, bom_indekser=bom_indekser)

    if args.sladd:
        sladd_alle(sladd_bokser, args.mappe, args.sladd_mappe)


if __name__ == "__main__":
    main()