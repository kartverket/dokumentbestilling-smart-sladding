"""
Fnr- og oversladdingsstatistikk per rettsstiftelsestype.

Joiner metadata-CSV-en (rettsstiftelsestyper per dokument) mot fasit og
resultat-CSV, og viser per XX_YYY-kode: hvor fnr-tette dokumentene er og
hvor oversladdingene (BOM) kommer fra. Svarer på hvilke rettsstiftelser som
er verdt egne filterprofiler — koordinat-tunge typer (målebrev o.l.) burde
ha høy BOM-tetthet, hjemmelsdokumenter høy fnr-tetthet.

Et dokument kan ha flere koder og telles da under hver av dem — kolonnene
summerer derfor til MER enn totalene nederst.

Med --skriv-lister DIR skrives én ID-liste per kode (rs_<KODE>.txt), som kan
brukes direkte i eksisterende verktøy for å måle en regel på én type:

    python utils/filter_sweep.py ... --kjorte-liste DIR/rs_SR_ERK.txt
    python utils/filter_review.py ... --kjorte-liste DIR/rs_SR_ERK.txt

Kjør:
    python utils/rettsstiftelse_stat.py \
        --metadata-csv $SLADD_METADATA/uttrekk_6.csv \
        --fasit-csv $SLADD_LABELS/uttrekk_6.csv \
        --res-csv $SLADD_VALIDERING/<run>/resultat.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from filter_felles import (STD_KRITERIUM, STD_SLURV_FAKTOR, STD_TERSKEL,
                           bygg_datasett, les_fasit, les_prediksjoner)


def les_metadata(sti):
    """fil_revisjon_id -> (koder, elektronisk); kode -> beskrivelse."""
    per_dok, beskrivelse = {}, {}
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                dok = int(r["fil_revisjon_id"])
            except (KeyError, TypeError, ValueError):
                continue
            koder = []
            for del_ in (r.get("rettsstiftelsestyper") or "").split(","):
                del_ = del_.strip()
                if not del_:
                    continue
                kode, _, tekst = del_.partition(" - ")
                kode = kode.strip()
                if kode:
                    koder.append(kode)
                    if tekst.strip():
                        beskrivelse.setdefault(kode, tekst.strip())
            elektronisk = ((r.get("er_elektronisk_tinglyst") or "")
                           .strip().lower() in ("true", "t", "1"))
            per_dok[dok] = (koder, elektronisk)
    return per_dok, beskrivelse


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--fasit-csv", required=True)
    p.add_argument("--res-csv", required=True)
    p.add_argument("--terskel", type=float, default=STD_TERSKEL)
    p.add_argument("--kriterium", default=STD_KRITERIUM)
    p.add_argument("--slurv-faktor", type=float, default=STD_SLURV_FAKTOR)
    p.add_argument("--ekskluder-ulabelte", dest="inkluder_ulabelte",
                   action="store_false", default=True,
                   help="Gammel oppførsel: kun dokumenter med fasit-rader")
    p.add_argument("--topp", type=int, default=None, metavar="N",
                   help="Vis kun de N kodene med flest BOM")
    p.add_argument("--min-dok", type=int, default=1, metavar="N",
                   help="Skjul koder med færre enn N kjørte dokumenter")
    p.add_argument("--skriv-lister", default=None, metavar="DIR",
                   help="Skriv én ID-liste per kode (rs_<KODE>.txt) for bruk "
                        "med --kjorte-liste i sweep/review")
    args = p.parse_args()

    meta, beskrivelse = les_metadata(args.metadata_csv)
    ds = bygg_datasett(les_fasit(args.fasit_csv),
                       les_prediksjoner(args.res_csv),
                       terskel=args.terskel, slurv_faktor=args.slurv_faktor,
                       inkluder_ulabelte=args.inkluder_ulabelte,
                       kriterium=args.kriterium)

    fasit_per_dok = defaultdict(int)
    for fb in ds.fasit_bokser:
        fasit_per_dok[fb["dok_nr"]] += 1

    pred_per_dok = defaultdict(list)
    for pr in ds.pred:
        pred_per_dok[pr["dok_nr"]].append(pr)

    KILDER = ("begge", "yolo", "paddle")

    def _ny():
        return {"dok_meta": 0, "dok_kjort": 0, "fasit": 0, "treff": 0,
                "bom": 0, **{f"bom_{k}": 0 for k in KILDER}}

    per_kode = defaultdict(_ny)
    elektronisk = {False: _ny(), True: _ny()}
    uten_meta = _ny()

    def _tell(rad, dok, kjort):
        rad["dok_meta"] += 1
        if not kjort:
            return
        rad["dok_kjort"] += 1
        rad["fasit"] += fasit_per_dok.get(dok, 0)
        for pr in pred_per_dok.get(dok, ()):
            if pr["klasse"] == "BOM":
                rad["bom"] += 1
                nokkel = f"bom_{pr['kilde'].lower()}"
                if nokkel in rad:
                    rad[nokkel] += 1
            elif pr["klasse"] == "TREFF":
                rad["treff"] += 1

    for dok, (koder, el) in meta.items():
        kjort = dok in ds.scope_dok
        _tell(elektronisk[el], dok, kjort)
        for kode in set(koder):
            _tell(per_kode[kode], dok, kjort)

    for dok in ds.scope_dok:
        if dok not in meta:
            _tell(uten_meta, dok, True)
    uten_meta["dok_meta"] = uten_meta["dok_kjort"]

    total_bom = ds.n_bom or 1

    print(f"Metadata: {len(meta)} dokumenter, {len(per_kode)} unike koder")
    print(f"Scope:    {len(ds.scope_dok)} kjørte dokumenter, "
          f"{len(ds.fasit_bokser)} fasit-bokser, {ds.n_bom} BOM totalt")
    print("Et dokument telles under HVER av kodene sine — kolonnene "
          "summerer til mer enn totalen.\n")

    hode = (f"  {'kode':<8} {'dok':>6} {'kjørt':>6} {'fasit':>6} "
            f"{'fnr/dok':>8} {'treff':>6} {'bom':>6} {'bom/dok':>8} "
            f"{'bom%':>6}  {'b/y/p':>13}  beskrivelse")
    print(hode)
    print(f"  {'─' * (len(hode) + 8)}")

    def _rad(navn, r, tekst=""):
        kj = r["dok_kjort"]
        print(f"  {navn:<8} {r['dok_meta']:>6} {kj:>6} {r['fasit']:>6} "
              f"{(r['fasit'] / kj if kj else 0):>8.2f} "
              f"{r['treff']:>6} {r['bom']:>6} "
              f"{(r['bom'] / kj if kj else 0):>8.2f} "
              f"{r['bom'] / total_bom * 100:>5.1f}%  "
              f"{r['bom_begge']:>4}/{r['bom_yolo']:>4}/{r['bom_paddle']:>3}  "
              f"{tekst[:44]}")

    rader = [(k, r) for k, r in per_kode.items()
             if r["dok_kjort"] >= args.min_dok]
    rader.sort(key=lambda kr: -kr[1]["bom"])
    skjult = len(per_kode) - len(rader)
    if args.topp:
        skjult += max(0, len(rader) - args.topp)
        rader = rader[:args.topp]

    for kode, r in rader:
        _rad(kode, r, beskrivelse.get(kode, ""))
    if uten_meta["dok_kjort"]:
        _rad("(uten)", uten_meta, "kjørt, men mangler metadata-rad")
    if skjult:
        print(f"  ({skjult} koder skjult — --topp/--min-dok)")

    print("\n  Elektronisk tinglyst vs. skannet:")
    for el, navn in ((False, "skannet"), (True, "elektr.")):
        _rad(navn[:8], elektronisk[el])

    if args.skriv_lister:
        os.makedirs(args.skriv_lister, exist_ok=True)
        n = 0
        for kode, r in per_kode.items():
            if r["dok_kjort"] < args.min_dok:
                continue
            dok_ids = sorted(d for d, (koder, _el) in meta.items()
                             if kode in koder and d in ds.scope_dok)
            sti = os.path.join(args.skriv_lister, f"rs_{kode}.txt")
            with open(sti, "w", encoding="utf-8") as f:
                f.write("\n".join(str(d) for d in dok_ids) + "\n")
            n += 1
        print(f"\n  {n} ID-lister skrevet til {args.skriv_lister}/rs_<KODE>.txt")
        print("  Bruk: filter_sweep/filter_review ... --kjorte-liste "
              f"{args.skriv_lister}/rs_<KODE>.txt")


if __name__ == "__main__":
    main()
