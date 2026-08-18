import argparse
import csv
import io
import os
import sys
from collections import defaultdict
from contextlib import redirect_stdout

import fitz
import numpy as np

_APP = os.path.join(os.path.dirname(__file__), "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from config import MAKS_BREDDE_ELEKTRONISK_PT
from csv_export import les_csv
from evaluation import les_fasit, mal_overlapp
from load_pdf import PDF_DPI, les_sider
from visualization import tegn_og_lagre
from yolo_fnr import finn_yolo_bokser

SKALA = PDF_DPI / 72.0                     # PDF-punkt -> piksel


def _filtrer_for_brede(sladd_bokser, maks_pt):
    ut = {}
    fjernet = 0
    for (navn, si), (bw, bh, bokser) in sladd_bokser.items():
        beholdt = [b for b in bokser if (b[2] - b[0]) / SKALA <= maks_pt]
        fjernet += len(bokser) - len(beholdt)
        ut[(navn, si)] = (bw, bh, beholdt)
    return ut, fjernet


def _filtrer_lav_conf(sladd_bokser, min_conf):
    ut = {}
    fjernet = 0
    for (navn, si), (bw, bh, bokser) in sladd_bokser.items():
        beholdt = []
        for b in bokser:
            kilde = b[4] if len(b) > 4 else "paddle"
            conf = b[5] if len(b) > 5 else None
            if kilde == "yolo" and conf is not None and conf < min_conf:
                fjernet += 1
                continue
            beholdt.append(b)
        ut[(navn, si)] = (bw, bh, beholdt)
    return ut, fjernet


def _les_prod_csv(sti, mappe):
    per_fil = defaultdict(list)
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            per_fil[r["navn"]].append((int(r["side"]), float(r["x"]), float(r["y"]),
                                       float(r["width"]), float(r["height"])))

    sladd_bokser = {}
    for navn in sorted(per_fil):
        try:
            d = fitz.open(os.path.join(mappe, navn))
        except Exception as e:
            print(f"   {navn}: kunne ikke aapnes ({e!r})")
            continue
        for (si, x, y, w, h) in sorted(per_fil[navn]):
            if not 1 <= si <= len(d):
                continue
            rect = d[si - 1].rect
            bw = int(round(rect.width * SKALA))
            bh = int(round(rect.height * SKALA))
            x0, x1 = sorted((x, x + w))
            y0, y1 = sorted((y, y + h))
            # prod-CSV har verken kilde eller conf -> None = ukjent (graa ramme)
            boks = (x0 * SKALA, y0 * SKALA, x1 * SKALA, y1 * SKALA, None, None)
            sladd_bokser.setdefault((navn, si), (bw, bh, []))[2].append(boks)
        d.close()
    return sladd_bokser


def _les_boks_csv(sti, mappe, prod=False):
    if prod:
        return _les_prod_csv(sti, mappe)
    with open(sti, newline="", encoding="utf-8-sig") as f:
        felt = csv.DictReader(f).fieldnames or []
    if "bilde_bredde" in felt:
        return les_csv(sti)
    return _les_prod_csv(sti, mappe)


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


def _stille_eval(sladd_bokser, fasit, mappe, terskel, y_origin):
    with io.StringIO() as buf, redirect_stdout(buf):
        return mal_overlapp(sladd_bokser, fasit, mappe,
                            terskel=terskel, y_origin=y_origin,
                            kilder=sladd_bokser)


def _skriv_foer_etter(foer, etter, filter_navn):
    if not foer or not etter:
        return
    print("\n" + "=" * 64)
    print(f"Foer/etter filter ({', '.join(filter_navn)}):")
    rader = [
        ("Recall",         f"{foer['truffet']}/{foer['fasit']} = {foer['recall']:.0%}",
                           f"{etter['truffet']}/{etter['fasit']} = {etter['recall']:.0%}",
                           etter["truffet"] - foer["truffet"]),
        ("Sladde-bokser",  str(foer["pred"]), str(etter["pred"]),
                           etter["pred"] - foer["pred"]),
        ("Over-sladding",  str(foer["overflod"]), str(etter["overflod"]),
                           etter["overflod"] - foer["overflod"]),
        ("Samlet overlapp", f"{foer['samlet_overlapp']:.0%}",
                            f"{etter['samlet_overlapp']:.0%}", None),
    ]
    print(f"   {'':<18}{'foer':>16}{'etter':>16}{'endring':>12}")
    for navn, a, b, delta in rader:
        d = "" if delta is None else f"{delta:+d}"
        print(f"   {navn:<18}{a:>16}{b:>16}{d:>12}")


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
    p.add_argument("--prod", action="store_true",
                   help="les CSV-en som prod-format (punkt, uten kilde/conf) - alle bokser "
                        "faar graa ramme siden kilden er ukjent")
    p.add_argument("--yolo", action="store_true",
                   help="kjor YOLO og vis treff som roede rammer med conf")
    p.add_argument("--fasit", action="store_true",
                   help="maal recall mot fasit og skriv ut i terminal")
    p.add_argument("--terskel", type=float, default=0.15,
                   help="overlapp-terskel for TRUFFET (default 0.15)")
    p.add_argument("--kun-oversladd", action="store_true",
                   help="tegn kun sider med over-sladding (boks uten fasit-treff), "
                        "lagres i undermappa kun_oversladd/ under --png-mappe")
    p.add_argument("--kun-bom", action="store_true",
                   help="tegn kun sider med minst én MANGLER (krever --fasit)")
    p.add_argument("--min-conf", type=float, default=None, metavar="CONF",
                   help="kast bokser som kun YOLO fant og som har conf under CONF "
                        "(paddle- og begge-bokser beholdes uansett)")
    p.add_argument("--maks-bredde", type=float, nargs="?", const=MAKS_BREDDE_ELEKTRONISK_PT, default=None,
                   metavar="PT",
                   help=f"kast bokser bredere enn PT punkter (uten verdi: {MAKS_BREDDE_ELEKTRONISK_PT} fra config)")
    args = p.parse_args()

    sladd_bokser = _les_boks_csv(args.csv, args.mappe, prod=args.prod)
    if args.velg:
        velg_sett = {os.path.basename(f) for f in args.velg}
        sladd_bokser = {k: v for k, v in sladd_bokser.items()
                        if os.path.basename(k[0]) in velg_sett}
    n_bokser = sum(len(v[2]) for v in sladd_bokser.values())
    print(f"Leste {n_bokser} boks(er) over {len(sladd_bokser)} (fil, side)-grupper fra {args.csv}")

    ufiltrert = sladd_bokser          # kopi foer filtrene, til foer/etter-maaling
    filter_navn = []

    if args.min_conf is not None:
        sladd_bokser, fjernet = _filtrer_lav_conf(sladd_bokser, args.min_conf)
        n_bokser -= fjernet
        print(f"Conf-filter: kastet {fjernet} kun-YOLO-boks(er) med conf under "
              f"{args.min_conf:g} ({n_bokser} igjen)")
        filter_navn.append(f"conf>={args.min_conf:g}")

    if args.maks_bredde is not None:
        sladd_bokser, fjernet = _filtrer_for_brede(sladd_bokser, args.maks_bredde)
        print(f"Bredde-filter: kastet {fjernet} boks(er) bredere enn {args.maks_bredde:g} pt "
              f"({n_bokser - fjernet} igjen)")
        filter_navn.append(f"bredde<={args.maks_bredde:g}pt")

    yolo_bokser = None
    if args.yolo:
        print("Kjorer YOLO...")
        yolo_bokser = _kjor_yolo_paa_mappe(sladd_bokser, args.mappe)
        n_yolo = sum(len(v[2]) for v in yolo_bokser.values())
        print(f"YOLO fant {n_yolo} boks(er)")
        if args.min_conf is not None:
            yolo_bokser = {k: (w, h, [b for b in bs if b[4] >= args.min_conf])
                           for k, (w, h, bs) in yolo_bokser.items()}
            n_igjen = sum(len(v[2]) for v in yolo_bokser.values())
            print(f"Conf-filter: kastet {n_yolo - n_igjen} YOLO-boks(er) med conf under "
                  f"{args.min_conf:g} ({n_igjen} igjen)")

    fasit = les_fasit(args.fasit_csv)

    # Maal ufiltrert saett naar det finnes filtre aa sammenligne mot
    foer_res = None
    if filter_navn and fasit:
        foer_res = _stille_eval(ufiltrert, fasit, args.mappe, args.terskel, args.y_origin)

    if args.kun_oversladd:
        eval_res = _stille_eval(sladd_bokser, fasit, args.mappe, args.terskel, args.y_origin)
        _skriv_foer_etter(foer_res, eval_res, filter_navn)
        oversladd_sider = {(r["fil"], r["side"]) for r in (eval_res or {}).get("overflod_filer", [])}
        if not oversladd_sider:
            print("Ingen over-sladding funnet.")
            return
        ov_bokser = (eval_res or {}).get("oversladd_bokser", {})
        sladd_bokser = {k: v for k, v in sladd_bokser.items() if k in oversladd_sider}
        ut_mappe = os.path.join(args.png_mappe, "kun_oversladd")
        print(f"Tegner {len(oversladd_sider)} side(r) med over-sladding...")
        tegn_og_lagre(sladd_bokser, fasit, args.mappe, ut_mappe,
                      y_origin=args.y_origin, kilder=sladd_bokser,
                      oversladd_bokser=ov_bokser)
    else:
        bom_indekser = None
        oversladd = None
        if args.fasit:
            eval_res = mal_overlapp(sladd_bokser, fasit, args.mappe,
                                    terskel=args.terskel, y_origin=args.y_origin,
                                    kilder=sladd_bokser)
            _skriv_foer_etter(foer_res, eval_res, filter_navn)
            from visualization import _dok_nr as _vnr
            bom_indekser = {
                (_vnr(d["fil"]), d["side"], d["fasit_nr"] - 1)
                for d in eval_res.get("detaljer", [])
                if d["resultat"] == "MANGLER"
            }
            oversladd = eval_res.get("oversladd_bokser", None)
            if args.kun_bom:
                bom_sider = {(d["fil"], d["side"]) for d in eval_res.get("detaljer", []) if d["resultat"] == "MANGLER"}
                sladd_bokser = {k: v for k, v in sladd_bokser.items() if k in bom_sider}
                print(f"Filtrert til {len(bom_sider)} side(r) med bom.")
        tegn_og_lagre(sladd_bokser, fasit, args.mappe, args.png_mappe,
                      y_origin=args.y_origin, yolo_bokser=yolo_bokser,
                      kilder=sladd_bokser, bom_indekser=bom_indekser,
                      oversladd_bokser=oversladd)


if __name__ == "__main__":
    main()