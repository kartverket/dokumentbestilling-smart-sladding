import argparse
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np

_APP = os.path.join(os.path.dirname(__file__), "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from csv_export import les_csv
from evaluation import les_fasit, mal_overlapp
from load_pdf import les_sider
from visualization import tegn_og_lagre
from yolo_fnr import finn_yolo_bokser


def _kjor_yolo_paa_mappe(sladd_bokser, mappe):
    yolo_bokser = {}
    per_fil = {}
    for (navn, si) in sladd_bokser:
        per_fil.setdefault(navn, set()).add(si)

    for navn, sider in sorted(per_fil.items()):
        sti = os.path.join(mappe, navn)
        try:
            bilder = les_sider(sti)
        except Exception as e:
            print(f"   YOLO: kunne ikke lese {navn}: {e!r}")
            continue
        for si in sider:
            if not 1 <= si <= len(bilder):
                continue
            bilde = bilder[si - 1]
            h, w = bilde.shape[:2]
            treff = finn_yolo_bokser(bilde)
            if treff:
                bokser = [(x0, y0, x1, y1, conf) for x0, y0, x1, y1, conf in treff]
                yolo_bokser[(navn, si)] = (w, h, bokser)
    return yolo_bokser


def main():
    p = argparse.ArgumentParser(
        description="Tegn bokser fra en ferdig CSV rett til PNG - uten aa kjore modellen. "
                    "CSV maa ha header: navn,side,bilde_bredde,bilde_hoyde,x0,y0,x1,y1")
    p.add_argument("--csv", default="res.csv",
                   help="CSV-en med boksene (pek paa fila i fasit-mappa di)")
    p.add_argument("--mappe", default="../uttrekk_3",
                   help="mappe med PDF-ene som CSV-en viser til")
    p.add_argument("--png-mappe", default="visning", help="hvor PNG-ene lagres")
    p.add_argument("--fasit-csv", default="smartsladding_uttrekk_labels_3_07_07_26.csv",
                   help="fasit-CSV; tegnes som groenne rammer")
    p.add_argument("--y-origin", choices=["topp", "bunn"], default="topp", help="fasit y-origo")
    p.add_argument("--velg", nargs="+", metavar="PDF",
                   help="begrens til disse PDF-filene (filnavn, kan inkludere .pdf)")
    p.add_argument("--yolo", action="store_true",
                   help="kjor YOLO og vis treff som roede rammer")
    p.add_argument("--fasit", action="store_true",
                   help="maal recall mot fasit og skriv ut i terminal")
    p.add_argument("--terskel", type=float, default=0.15,
                   help="overlapp-terskel for TRUFFET (default 0.15)")
    p.add_argument("--kun-oversladd", action="store_true",
                   help="tegn kun sider med over-sladding (boks uten fasit-treff), lagres i visning/kun_oversladd/")
    p.add_argument("--kun-bom", action="store_true",
                   help="tegn kun sider med minst én MANGLER (krever --fasit)")
    args = p.parse_args()

    sladd_bokser = les_csv(args.csv)
    if args.velg:
        velg_sett = {os.path.basename(f) for f in args.velg}
        sladd_bokser = {k: v for k, v in sladd_bokser.items()
                        if os.path.basename(k[0]) in velg_sett}
    n_bokser = sum(len(v[2]) for v in sladd_bokser.values())
    print(f"Leste {n_bokser} boks(er) over {len(sladd_bokser)} (fil, side)-grupper fra {args.csv}")

    yolo_bokser = None
    if args.yolo:
        print("Kjorer YOLO...")
        yolo_bokser = _kjor_yolo_paa_mappe(sladd_bokser, args.mappe)
        n_yolo = sum(len(v[2]) for v in yolo_bokser.values())
        print(f"YOLO fant {n_yolo} boks(er)")

    fasit = les_fasit(args.fasit_csv)

    if args.kun_oversladd:
        # Kjør eval stille for å finne over-sladding-sider
        import io as _io
        from contextlib import redirect_stdout as _rs
        with _io.StringIO() as _buf, _rs(_buf):
            eval_res = mal_overlapp(sladd_bokser, fasit, args.mappe,
                                    terskel=args.terskel, y_origin=args.y_origin)
        oversladd_sider = {(r["fil"], r["side"]) for r in (eval_res or {}).get("overflod_filer", [])}
        if not oversladd_sider:
            print("Ingen over-sladding funnet.")
            return
        ov_bokser = (eval_res or {}).get("oversladd_bokser", {})
        sladd_bokser = {k: v for k, v in sladd_bokser.items() if k in oversladd_sider}
        print(f"Tegner {len(oversladd_sider)} side(r) med over-sladding...")
        tegn_og_lagre(sladd_bokser, fasit, args.mappe, "visning/kun_oversladd",
                      y_origin=args.y_origin, kilder=sladd_bokser,
                      oversladd_bokser=ov_bokser)
    else:
        bom_indekser = None
        if args.fasit:
            eval_res = mal_overlapp(sladd_bokser, fasit, args.mappe,
                                    terskel=args.terskel, y_origin=args.y_origin,
                                    kilder=sladd_bokser)
            from visualization import _dok_nr as _vnr
            bom_indekser = {
                (_vnr(d["fil"]), d["side"], d["fasit_nr"] - 1)
                for d in eval_res.get("detaljer", [])
                if d["resultat"] == "MANGLER"
            }
            if args.kun_bom:
                bom_sider = {(d["fil"], d["side"]) for d in eval_res.get("detaljer", []) if d["resultat"] == "MANGLER"}
                sladd_bokser = {k: v for k, v in sladd_bokser.items() if k in bom_sider}
                print(f"Filtrert til {len(bom_sider)} side(r) med bom.")
        tegn_og_lagre(sladd_bokser, fasit, args.mappe, args.png_mappe,
                      y_origin=args.y_origin, yolo_bokser=yolo_bokser,
                      kilder=sladd_bokser, bom_indekser=bom_indekser)


if __name__ == "__main__":
    main()