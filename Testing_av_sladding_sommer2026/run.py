import argparse
import os

from file_selection import velg_filer
from model_main import run_model_on_pdf_bytes
from csv_export import skriv_csv
from evaluation import mal_overlapp, les_fasit
from visualization import tegn_og_lagre
from redaction import sladd_alle
import traceback

import time


def main():
    p = argparse.ArgumentParser(
        description="Kjør modellen lokalt som om filene var POST-er: "
                    "bytes -> run_model_on_pdf_bytes. Flagg legger til CSV/PNG/fasit/sladding.")
    p.add_argument("--mappe", default="panteboksdokumenter", help="mappe med PDF-er (spiller rollen som POST-body)")
    p.add_argument("--velg", nargs="*", default=[], help="spesifikke filer (filnavn/delstreng)")
    p.add_argument("--antall", default="20", help="antall filer når --velg er tom (tall, eller 'alle')")
    p.add_argument("--csv", action="store_true", help="skriv alle funne bokser til CSV")
    p.add_argument("--png", action="store_true", help="tegn funne + fasit-bokser til PNG")
    p.add_argument("--fasit", action="store_true", help="mål recall mot fasit")
    p.add_argument("--sladd", action="store_true", help="lag faktisk sladdede PDF-er")
    p.add_argument("--fasit-csv", default="smartsladding_uttrekk_labels_1_22_06_26.csv", help="fasit-CSV")
    p.add_argument("--csv-ut", default="sladd_koordinater.csv", help="hvor boks-CSV-en skrives")
    p.add_argument("--png-mappe", default="visning", help="hvor PNG-ene lagres")
    p.add_argument("--sladd-mappe", default="sladdet", help="hvor sladdede PDF-er lagres")
    p.add_argument("--terskel", type=float, default=0.40, help="andel fasit-areal for TRUFFET")
    p.add_argument("--y-origin", choices=["topp", "bunn"], default="topp", help="CSV y-origo")
    p.add_argument("--tid", action="store_true", help="viser tiden de ulike stegene i pipelinen tar")
    args = p.parse_args()

    filer = velg_filer(args.mappe, args.velg, args.antall)

    if not filer:
        print("Ingen filer å behandle — sjekk --mappe / --velg / --antall.")
        return

    vil_ha_artefakt = args.csv or args.png or args.fasit or args.sladd
    total_tid = 0

    sladd_bokser, feilet = {}, []
    for fil in filer:
        start = time.perf_counter()

        navn = os.path.basename(fil)
        print(f"\nBehandler: {navn}")

        try:
            with open(fil, "rb") as f:
                resultat = run_model_on_pdf_bytes(f.read(), args.tid)     # akkurat som POST-endepunktet
        except Exception as e:
            feilet.append((navn, repr(e)))
            traceback.print_exc() 
            continue

        tid_brukt = time.perf_counter() - start
        total_tid += tid_brukt

        if vil_ha_artefakt:
            n = sum(len(s["bokser"]) for s in resultat["sider"])
            print(f"Fant {n} boks(er) over {len(resultat['sider'])} side(r)")

        for side in resultat["sider"]:
            bokser = [(b["x0"], b["y0"], b["x1"], b["y1"]) for b in side["bokser"]]
            sladd_bokser[(navn, side["side"])] = (
                side["bilde_bredde"], side["bilde_hoyde"], bokser)

    print(f"\nTotal tid brukt for alle dokumenter: {total_tid:.6f} sekunder")

    if feilet:
        print("Feilet:", feilet[:5])
    if not vil_ha_artefakt:
        return

    fasit = les_fasit(args.fasit_csv) if (args.png or args.fasit) else None
    if args.csv:
        n = skriv_csv(sladd_bokser, args.csv_ut)
        print(f"Skrev {n} boks(er) til {args.csv_ut}")
    if args.png:
        tegn_og_lagre(sladd_bokser, fasit, args.mappe, args.png_mappe, y_origin=args.y_origin)
    if args.fasit:
        mal_overlapp(sladd_bokser, fasit, args.mappe, terskel=args.terskel, y_origin=args.y_origin)
    if args.sladd:
        sladd_alle(sladd_bokser, args.mappe, args.sladd_mappe)


if __name__ == "__main__":
    main()