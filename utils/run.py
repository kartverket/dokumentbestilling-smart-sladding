import argparse
import io
import os
import sys
from collections import defaultdict
from contextlib import redirect_stdout

import fitz

_APP = os.path.join(os.path.dirname(__file__), "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from file_selection import velg_filer
from model_main import run_model_on_pdf_bytes
from csv_export import initialiser_csv, append_csv
from evaluation import mal_overlapp, les_fasit
from visualization import tegn_og_lagre
from redaction import sladd_alle
from yolo_fnr import sett_vekter
from load_pdf import PDF_DPI
import traceback
from save_result import lagre_resultat

import time

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


def main():
    p = argparse.ArgumentParser(
        description="Kjor modellen lokalt som om filene var POST-er: "
                    "bytes -> run_model_on_pdf_bytes. Flagg legger til CSV/PNG/fasit/sladding.")
    p.add_argument("--mappe", default=MAPPE, help="mappe med PDF-er (spiller rollen som POST-body)")
    p.add_argument("--velg", nargs="*", default=[], help="spesifikke filer (filnavn/delstreng)")
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
    args = p.parse_args()

    sett_vekter(args.yolo_vekter)

    filer = velg_filer(args.mappe, args.velg, args.antall)

    total_tid = 0

    if not filer:
        print("Ingen filer aa behandle - sjekk --mappe / --velg / --antall.")
        return

    vil_ha_artefakt = args.csv or args.png or args.fasit or args.sladd or args.kun_feil

    if args.csv:
        initialiser_csv(args.csv_ut)
        print(f"Starter kontinuerlig skriving til {args.csv_ut}")

    sladd_bokser, yolo_bokser, csv_bokser, feilet = {}, {}, {}, []
    tider = {}
    ocr_linjer = {}                          # (navn, side) -> liste av (tekst, merker)
    advart_om_linjer = False
    for fil in filer:
        start = time.perf_counter()

        navn = os.path.basename(fil)
        print(f"\n→ Starter: {navn}")
        try:
            with open(fil, "rb") as f:
                t0 = time.perf_counter()
                pdf_bytes = f.read()
                print(f"  lest fil: {time.perf_counter()-t0:.2f}s")
                resultat = run_model_on_pdf_bytes(pdf_bytes, skriv_tid=args.tid, med_linjer=args.ocr_logg, navn=navn,
                                                  elektronisk_tinglyst=args.elektronisk_tinglyst,
                                                  kun_yolo=args.kun_yolo)  # akkurat som POST-endepunktet
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
            print(f"{navn}: {n} boks(er) over {len(sider)} side(r)")
            print(f"    Tid brukt: {tid_brukt:.6f} sekunder")

            continue

        for side in sider:
            bokser     = [(b["x0"], b["y0"], b["x1"], b["y1"]) for b in side["bokser"]]
            med_kilde  = [(b["x0"], b["y0"], b["x1"], b["y1"], b.get("kilde", "paddle"), b.get("conf"))
                          for b in side["bokser"]]
            yolo_bare  = [(b["x0"], b["y0"], b["x1"], b["y1"]) for b in side["bokser"]
                          if b.get("kilde") in ("yolo", "begge")]
            sladd_bokser[(navn, side["side"])] = (
                side["bilde_bredde"], side["bilde_hoyde"], bokser)
            csv_bokser[(navn, side["side"])] = (
                side["bilde_bredde"], side["bilde_hoyde"], med_kilde)
            if yolo_bare:
                yolo_bokser[(navn, side["side"])] = (
                    side["bilde_bredde"], side["bilde_hoyde"], yolo_bare)

        if args.csv:
            dok_bokser = {k: v for k, v in csv_bokser.items() if k[0] == navn}
            append_csv(dok_bokser, args.csv_ut)

    print(f"\nTotal tid brukt: {total_tid:.6f} sekunder")

    if feilet:
        print("Feilet:", feilet[:5])

    if args.ocr_logg:
        n = _skriv_ocr_logg(ocr_linjer, args.ocr_logg_fil)
        print(f"Skrev OCR-linjer for {n} side(r) til {args.ocr_logg_fil}")

    if not vil_ha_artefakt and not args.kun_feil:
        return

    fasit = les_fasit(args.fasit_csv) if (args.png or args.fasit or args.kun_feil) else None

    # Kjør evaluering (trenger den alltid for --kun-feil, og for --fasit)
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
            lagre_resultat(eval_resultat, beskrivelse=args.beskrivelse, logg=header + logg)

    # PNG: tegn enten alle sider, eller bare feil-sider
    # Beregn bom_indekser fra eval-resultat for korrekt fargekoding
    bom_indekser = None
    oversladd = None
    if eval_resultat:
        from visualization import _dok_nr as _vnr
        bom_indekser = {
            (_vnr(d["fil"]), d["side"], d["fasit_nr"] - 1)
            for d in eval_resultat.get("detaljer", [])
            if d["resultat"] == "MANGLER"
        }
        oversladd = eval_resultat.get("oversladd_bokser", None)

    if args.png or args.kun_feil:
        if args.kun_feil and eval_resultat:
            # Filtrer til bare sider med bom eller over-sladding
            feil_sider = set()
            for bf in eval_resultat.get("bom_filer", []):
                feil_sider.add((bf["fil"], bf["side"]))
            for of in eval_resultat.get("overflod_filer", []):
                feil_sider.add((of["fil"], of["side"]))
            sladd_filtrert = {k: v for k, v in sladd_bokser.items() if k in feil_sider}
            csv_filtrert = {k: v for k, v in csv_bokser.items() if k in feil_sider}
            print(f"\n--kun-feil: tegner {len(sladd_filtrert)} side(r) med feil")
            tegn_og_lagre(sladd_filtrert, fasit, args.mappe, args.png_mappe,
                          y_origin=args.y_origin, kilder=csv_filtrert,
                          oversladd_bokser=oversladd, bom_indekser=bom_indekser)
        else:
            tegn_og_lagre(sladd_bokser, fasit, args.mappe, args.png_mappe,
                          y_origin=args.y_origin, kilder=csv_bokser,
                          oversladd_bokser=oversladd, bom_indekser=bom_indekser)

    if args.sladd:
        sladd_alle(sladd_bokser, args.mappe, args.sladd_mappe)


if __name__ == "__main__":
    main()