import argparse

import sladd_lib
from filvalg import velg_filer
from fasit import les_fasit
from visning import tegn_og_lagre
from evaluering import mal_overlapp


def main():
    p = argparse.ArgumentParser(description="OCR + sladd av fnr, med fasit-sammenligning.")
    p.add_argument("--mappe", default="panteboksdokumenter", help="mappe med PDF-er")
    p.add_argument("--ut-mappe", default="visning_bom2", help="hvor PNG/csv/logg lagres")
    p.add_argument("--csv", default="smartsladding_uttrekk_labels_1_22_06_26.csv", help="fasit-CSV")
    p.add_argument("--velg", nargs="*", default=[],
                   help="spesifikke filer (filnavn/delstreng). Vinner over --antall hvis satt.")
    p.add_argument("--antall", default="20",
                   help="antall filer når --velg er tom (tall, eller 'alle')")
    p.add_argument("--terskel", type=float, default=0.40,
                   help="andel av fasit-areal som må dekkes for TRUFFET")
    p.add_argument("--y-origin", choices=["topp", "bunn"], default="topp", help="CSV y-origo")
    p.add_argument("--ingen-logg", action="store_true",
                   help="hopp over per-linje-OCR i loggen (mye raskere ved mange filer)")
    p.add_argument("--ingen-rydd", action="store_true",
                   help="ikke slett gamle PNG-er i ut-mappe før kjøring")
    args = p.parse_args()

    # 1) velg filer
    filer = velg_filer(args.mappe, args.velg, args.antall)
    if not filer:
        print("Ingen filer å behandle — sjekk --mappe / --velg / --antall.")
        return

    print(f"\nBehandler {len(filer)} fil(er) med OCR + sladd …")
    sladd_bokser, feilet = sladd_lib.behandle_alle(filer)
    ant_sladd = sum(len(v[2]) for v in sladd_bokser.values())
    print(f"Behandlet: {len(sladd_bokser)} side(r), {ant_sladd} sladde-boks(er), "
          f"{len(feilet)} fil(er) feilet.")
    if feilet:
        print("Feilet:", feilet[:5])

    fasit = les_fasit(args.csv)

    print()
    tegn_og_lagre(sladd_bokser, fasit, args.mappe, args.ut_mappe,
                  y_origin=args.y_origin,
                  skriv_logg=not args.ingen_logg,
                  rydd=not args.ingen_rydd)

    mal_overlapp(sladd_bokser, fasit, args.mappe,
                 terskel=args.terskel, y_origin=args.y_origin)


if __name__ == "__main__":
    main()
