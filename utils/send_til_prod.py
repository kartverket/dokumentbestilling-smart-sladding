import argparse
import json
import os
import sys
import time

import requests

_APP = os.path.join(os.path.dirname(__file__), "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from file_selection import velg_filer
from csv_export import initialiser_csv, append_csv, les_csv
from evaluation import les_fasit, _norm_csv, _overlap, _areal, _dok_nr

try:
    from config import PDF_DPI
except Exception:
    PDF_DPI = 300


def _boks_tuple(b):
    # samme format som run.py bygger for append_csv: (x0,y0,x1,y1,kilde,conf)
    return (b["x0"], b["y0"], b["x1"], b["y1"], b.get("kilde", "paddle"), b.get("conf"))


def _send_en(sess, url, pdf_bytes, timeout):
    r = sess.post(url, data=pdf_bytes,
                  headers={"Content-Type": "application/pdf"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _evaluer(sladd, fasit, dpi, overlapp_terskel):
    """Recall + oversladding for alle bokser i CSV-en (samme matching som konfidens_sveip)."""
    truffet = fasit_tot = oversladd = 0
    for (navn, si) in sladd:
        nr = _dok_nr(navn)
        iw, ih, raw = sladd[(navn, si)]
        pw, ph = iw * 72.0 / dpi, ih * 72.0 / dpi
        pred = [(b[0] / iw, (b[1] - 2) / ih, b[2] / iw, (b[3] + 2) / ih) for b in raw]
        fb = [(_norm_csv(x, y, w, h, pw, ph, "topp"), t)
              for (x, y, w, h, t) in fasit.get((nr, si), [])]
        fasit_tot += len(fb)
        truffet_pred = set()
        for (f, _t) in fb:
            fa = _areal(f)
            best, bpi = 0.0, -1
            for pi, pb in enumerate(pred):
                ov = _overlap(f, pb)
                if ov > best:
                    best, bpi = ov, pi
            if fa and best / fa >= overlapp_terskel:
                truffet += 1
                truffet_pred.add(bpi)
        oversladd += len(pred) - len(truffet_pred)
    return truffet, fasit_tot, oversladd


def main():
    p = argparse.ArgumentParser(description="Send PDF-er til prod-API-et og skriv bokser til CSV.")
    p.add_argument("--mappe", default="/data2/smartsladding-uttrekk/uttrekk_4/",
                   help="mappe med PDF-er")
    p.add_argument("--antall", default="1000", help="antall filer (tall, eller 'alle') - ignoreres naar --velg er satt")
    p.add_argument("--velg", nargs="*", default=[],
                   help="spesifikke filer, f.eks. --velg $(tr '\\n' ' ' < ids_server.txt)")
    p.add_argument("--url", default="http://localhost:5071/model", help="prod-API-endepunkt")
    p.add_argument("--csv-ut", default="res_prod.csv", help="hvor boks-CSV-en skrives")
    p.add_argument("--json-ut", default="res_prod.jsonl",
                   help="raa API-svar (ett JSON-objekt per linje) - kan re-evalueres uten aa kjoere API-et paa nytt")
    p.add_argument("--elektronisk-tinglyst", action="store_true",
                   help="sett ?elektronisk_tinglyst=true (skrur av YOLO)")
    p.add_argument("--timeout", type=float, default=120, help="timeout per PDF (sekunder)")
    p.add_argument("--forsett-ved-feil", action="store_true", default=True,
                   help="hopp over dokumenter som feiler (default paa)")
    p.add_argument("--fasit-csv", default="/home/smartsladding/smartsladding-uttrekk-labels/uttrekk_4.csv",
                   help="fasit-CSV for automatisk evaluering til slutt")
    p.add_argument("--rapport-ut", default="res_prod_fasit.txt",
                   help="txt-rapport med treff/oversladding/timing")
    p.add_argument("--terskel", type=float, default=0.15,
                   help="overlapp-terskel for TRUFFET (default 0.15)")
    p.add_argument("--dpi", type=int, default=PDF_DPI, help=f"rasterings-DPI (default {PDF_DPI})")
    p.add_argument("--ingen-fasit", action="store_true", help="hopp over automatisk fasit-evaluering")
    args = p.parse_args()

    url = args.url
    if args.elektronisk_tinglyst:
        url += ("&" if "?" in url else "?") + "elektronisk_tinglyst=true"

    filer = velg_filer(args.mappe, args.velg, args.antall)
    if not filer:
        print("Ingen filer aa behandle - sjekk --mappe / --velg / --antall.")
        return

    initialiser_csv(args.csv_ut)
    print(f"Sender {len(filer)} PDF-er til {url}")
    print(f"Skriver bokser til {args.csv_ut} og raa svar til {args.json_ut}\n")

    sess = requests.Session()
    n_bokser = n_ok = 0
    feilet = []
    api_tid = 0.0                         # sum av ren API-svartid (uten fil-lesing)
    start = time.perf_counter()           # vegg-klokke for hele kjoeringen

    with open(args.json_ut, "w", encoding="utf-8") as jf:
        for i, fil in enumerate(filer, start=1):
            navn = os.path.basename(fil)
            try:
                with open(fil, "rb") as f:
                    pdf_bytes = f.read()
                t0 = time.perf_counter()
                resultat = _send_en(sess, url, pdf_bytes, args.timeout)
                dt = time.perf_counter() - t0
                api_tid += dt
            except Exception as e:
                feilet.append((navn, repr(e)))
                print(f"[{i}/{len(filer)}] FEIL {navn}: {e!r}")
                if not args.forsett_ved_feil:
                    break
                continue

            # raa svar: ett JSON-objekt per linje, flushes med en gang
            jf.write(json.dumps({"navn": navn, "sekunder": round(dt, 3),
                                 "resultat": resultat}, ensure_ascii=False) + "\n")
            jf.flush()

            grupper = {}
            dok_bokser = 0
            for side in resultat.get("sider", []):
                si = side["side"]
                bokser = [_boks_tuple(b) for b in side.get("bokser", [])]
                grupper[(navn, si)] = (side["bilde_bredde"], side["bilde_hoyde"], bokser)
                dok_bokser += len(bokser)

            append_csv(grupper, args.csv_ut)
            n_bokser += dok_bokser
            n_ok += 1
            snitt = api_tid / n_ok
            gjenst = snitt * (len(filer) - i)
            print(f"[{i}/{len(filer)}] {navn}: {dok_bokser} boks(er), "
                  f"{len(resultat.get('sider', []))} side(r), {dt:.1f}s "
                  f"(snitt {snitt:.1f}s, ETA ~{gjenst/60:.0f} min)")

    total = time.perf_counter() - start

    # --- bygg rapport-linjer (skrives baade til skjerm og txt) ---
    L = []
    L.append("=" * 60)
    L.append("SEND TIL PROD - RESULTAT")
    L.append("=" * 60)
    L.append(f"Tidspunkt:          {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"URL:                {url}")
    L.append(f"Mappe:              {args.mappe}")
    L.append(f"Dokumenter OK:      {n_ok}/{len(filer)}")
    L.append(f"Feilet:             {len(feilet)}")
    L.append(f"Bokser totalt:      {n_bokser}")
    L.append("")
    L.append("--- TID ---")
    L.append(f"Vegg-klokke total:  {total:.1f}s ({total/60:.1f} min)")
    L.append(f"Ren API-tid:        {api_tid:.1f}s ({api_tid/60:.1f} min)")
    if n_ok:
        L.append(f"Snitt per dokument: {api_tid/n_ok:.2f}s")

    # --- automatisk fasit-evaluering ---
    if not args.ingen_fasit and n_ok:
        fasit = les_fasit(args.fasit_csv)
        if fasit is None:
            L.append("")
            L.append(f"!! Fant ikke fasit-CSV: {args.fasit_csv} - hoppet over evaluering.")
        else:
            sladd = les_csv(args.csv_ut)
            tr, fasit_tot, ov = _evaluer(sladd, fasit, args.dpi, args.terskel)
            rec = tr / fasit_tot if fasit_tot else 0.0
            L.append("")
            L.append("--- FASIT (overlapp-terskel "
                     f"{args.terskel:.2f}) ---")
            L.append(f"Fasit-bokser:       {fasit_tot}")
            L.append(f"Treff:              {tr}")
            L.append(f"Recall:             {rec:.2%}")
            L.append(f"Bommet (mistet):    {fasit_tot - tr}")
            L.append(f"Oversladding:       {ov}")

    if feilet:
        L.append("")
        L.append("--- FEILEDE DOKUMENTER ---")
        for navn, e in feilet:
            L.append(f"  {navn}: {e}")
    L.append("=" * 60)

    rapport = "\n".join(L)
    print("\n" + rapport)
    with open(args.rapport_ut, "w", encoding="utf-8") as f:
        f.write(rapport + "\n")

    print(f"\nCSV (res-format): {os.path.abspath(args.csv_ut)}")
    print(f"Raa svar (JSONL): {os.path.abspath(args.json_ut)}")
    print(f"Rapport (txt):    {os.path.abspath(args.rapport_ut)}")


if __name__ == "__main__":
    main()
