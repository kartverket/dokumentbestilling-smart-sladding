import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import requests

_APP = os.path.join(os.path.dirname(__file__), "..", "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from evaluation import les_fasit, _overlap, _areal, _dok_nr

FELT = ["navn", "side", "x", "y", "width", "height"]


def _til_xyxy(x, y, w, h):
    """(x, y, w, h) -> (x0, y0, x1, y1), robust mot negativ bredde/hoyde."""
    x0, x1 = sorted((x, x + w))
    y0, y1 = sorted((y, y + h))
    return (x0, y0, x1, y1)


def _velg_eksakt(mappe, ids):
    """Eksakt match paa filnavn (stamme), ikke delstreng. Returnerer (filer, mangler)."""
    idset = {os.path.splitext(s.strip())[0] for s in ids if s.strip()}
    valgt = []
    for fn in sorted(os.listdir(mappe)):
        if os.path.splitext(fn)[0] in idset:
            valgt.append(os.path.join(mappe, fn))
    funnet = {os.path.splitext(os.path.basename(f))[0] for f in valgt}
    mangler = sorted(idset - funnet)
    return valgt, mangler


def _send_en(sess, url, pdf_bytes, timeout):
    r = sess.post(url, data=pdf_bytes,
                  headers={"Content-Type": "application/pdf"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _evaluer(pred_bokser, sendt_docnr, fasit, terskel):
    """Treff + oversladding i punkt-rom, kun for dokumentene vi faktisk sendte.

    pred_bokser: {(docnr, side): [(x0,y0,x1,y1), ...]}  (prod-bokser i punkter)
    sendt_docnr: set med docnr vi sendte OK
    fasit:       {(fil_revisjon_id, side): [(x,y,w,h,type), ...]}  fra les_fasit
    """
    truffet = fasit_tot = oversladd = 0
    matchet = defaultdict(set)

    # recall: gaa gjennom alle fasit-bokser for de sendte dokumentene
    for (nr, side), fbokser in fasit.items():
        if nr not in sendt_docnr:
            continue
        preds = pred_bokser.get((nr, side), [])
        for (fx, fy, fw, fh, _t) in fbokser:
            f = _til_xyxy(fx, fy, fw, fh)
            fa = _areal(f)
            best, bpi = 0.0, -1
            for pi, pb in enumerate(preds):
                ov = _overlap(f, pb)
                if ov > best:
                    best, bpi = ov, pi
            fasit_tot += 1
            if fa and best / fa >= terskel:
                truffet += 1
                matchet[(nr, side)].add(bpi)

    # oversladding: prod-bokser som ikke traff noen fasit-boks
    for key, preds in pred_bokser.items():
        oversladd += len(preds) - len(matchet.get(key, set()))
    return truffet, fasit_tot, oversladd


def main():
    p = argparse.ArgumentParser(description="Send PDF-er til prod-API-et, lagre svar og mal treff mot fasit.")
    p.add_argument("--mappe", default="/data2/smartsladding-uttrekk/uttrekk_4/",
                   help="mappe med PDF-er")
    p.add_argument("--ids-fil", default="ids_server.txt",
                   help="fil med ett dokument-navn/-id per linje (eksakt match)")
    p.add_argument("--url", default="http://localhost:5071/model", help="prod-API-endepunkt")
    p.add_argument("--csv-ut", default="res_prod.csv", help="prod-bokser i punkter (navn,side,x,y,width,height)")
    p.add_argument("--json-ut", default="res_prod.jsonl",
                   help="raa API-svar, ett JSON-objekt per linje")
    p.add_argument("--elektronisk-tinglyst", action="store_true",
                   help="sett ?elektronisk_tinglyst=true (skrur av YOLO)")
    p.add_argument("--timeout", type=float, default=120, help="timeout per PDF (sekunder)")
    p.add_argument("--fasit-csv", default="/home/smartsladding/smartsladding-uttrekk-labels/uttrekk_4.csv",
                   help="fasit-CSV for automatisk evaluering til slutt")
    p.add_argument("--rapport-ut", default="res_prod_fasit.txt",
                   help="txt-rapport med treff/oversladding/timing")
    p.add_argument("--terskel", type=float, default=0.15,
                   help="andel fasit-areal som kreves for TRUFFET (default 0.15)")
    p.add_argument("--ingen-fasit", action="store_true", help="hopp over automatisk fasit-evaluering")
    args = p.parse_args()

    url = args.url
    if args.elektronisk_tinglyst:
        url += ("&" if "?" in url else "?") + "elektronisk_tinglyst=true"

    # --- velg filer (eksakt match mot id-fila) ---
    try:
        with open(args.ids_fil, encoding="utf-8") as f:
            ids = f.read().split()
    except FileNotFoundError:
        print(f"Fant ikke id-fil: {args.ids_fil}")
        return
    filer, mangler = _velg_eksakt(args.mappe, ids)
    print(f"Id-fil: {len(set(ids))} unike id-er -> {len(filer)} filer funnet i {args.mappe}")
    if mangler:
        print(f"  {len(mangler)} id-er uten matchende fil. Foerste 5: {mangler[:5]}")
    if not filer:
        print("Ingen filer aa behandle.")
        return

    with open(args.csv_ut, "w", newline="", encoding="utf-8") as cf:
        csv.writer(cf).writerow(FELT)
    print(f"Sender {len(filer)} PDF-er til {url}")
    print(f"Bokser -> {args.csv_ut} | raa svar -> {args.json_ut}\n")

    sess = requests.Session()
    n_bokser = n_ok = 0
    feilet = []
    pred_bokser = defaultdict(list)
    sendt_docnr = set()
    api_tid = 0.0
    start = time.perf_counter()

    with open(args.json_ut, "w", encoding="utf-8") as jf, \
         open(args.csv_ut, "a", newline="", encoding="utf-8") as cf:
        skriv = csv.writer(cf)
        for i, fil in enumerate(filer, start=1):
            navn = os.path.basename(fil)
            try:
                with open(fil, "rb") as f:
                    pdf_bytes = f.read()
                t0 = time.perf_counter()
                resultat = _send_en(sess, url, pdf_bytes, args.timeout)
                dt = time.perf_counter() - t0
                api_tid += dt

                if not isinstance(resultat, list):
                    raise ValueError(f"forventet liste, fikk {type(resultat).__name__}: {str(resultat)[:120]}")

                jf.write(json.dumps({"navn": navn, "sekunder": round(dt, 3),
                                     "resultat": resultat}, ensure_ascii=False) + "\n")
                jf.flush()

                nr = _dok_nr(navn)
                dok_bokser = 0
                for b in resultat:
                    side = b["page"]
                    x, y, w, h = b["x"], b["y"], b["width"], b["height"]
                    skriv.writerow([navn, side, x, y, w, h])
                    pred_bokser[(nr, side)].append(_til_xyxy(x, y, w, h))
                    dok_bokser += 1
                cf.flush()
            except Exception as e:
                feilet.append((navn, repr(e)))
                print(f"[{i}/{len(filer)}] FEIL {navn}: {e!r}")
                continue

            sendt_docnr.add(nr)
            n_bokser += dok_bokser
            n_ok += 1
            snitt = api_tid / n_ok
            eta = snitt * (len(filer) - i)
            print(f"[{i}/{len(filer)}] {navn}: {dok_bokser} boks(er), {dt:.1f}s "
                  f"(snitt {snitt:.1f}s, ETA ~{eta/60:.0f} min)")

    total = time.perf_counter() - start

    # --- rapport ---
    L = []
    L.append("=" * 60)
    L.append("SEND TIL PROD - RESULTAT")
    L.append("=" * 60)
    L.append(f"Tidspunkt:          {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"URL:                {url}")
    L.append(f"Dokumenter OK:      {n_ok}/{len(filer)}")
    L.append(f"Feilet:             {len(feilet)}")
    L.append(f"Bokser totalt:      {n_bokser}")
    L.append("")
    L.append("--- TID ---")
    L.append(f"Vegg-klokke total:  {total:.1f}s ({total/60:.1f} min)")
    L.append(f"Ren API-tid:        {api_tid:.1f}s ({api_tid/60:.1f} min)")
    if n_ok:
        L.append(f"Snitt per dokument: {api_tid/n_ok:.2f}s")

    if not args.ingen_fasit and n_ok:
        fasit = les_fasit(args.fasit_csv)
        if fasit is None:
            L.append("")
            L.append(f"!! Fant ikke fasit-CSV: {args.fasit_csv} - hoppet over evaluering.")
        else:
            tr, fasit_tot, ov = _evaluer(pred_bokser, sendt_docnr, fasit, args.terskel)
            rec = tr / fasit_tot if fasit_tot else 0.0
            L.append("")
            L.append(f"--- FASIT (overlapp-terskel {args.terskel:.2f}) ---")
            L.append(f"Fasit-bokser:       {fasit_tot}")
            L.append(f"Treff:              {tr}")
            L.append(f"Recall:             {rec:.2%}")
            L.append(f"Bommet (mistet):    {fasit_tot - tr}")
            L.append(f"Oversladding:       {ov}")

    if feilet:
        L.append("")
        L.append("--- FEILEDE DOKUMENTER (foerste 20) ---")
        for navn, e in feilet[:20]:
            L.append(f"  {navn}: {e}")
    L.append("=" * 60)

    rapport = "\n".join(L)
    print("\n" + rapport)
    with open(args.rapport_ut, "w", encoding="utf-8") as f:
        f.write(rapport + "\n")
    print(f"\nCSV:     {os.path.abspath(args.csv_ut)}")
    print(f"JSONL:   {os.path.abspath(args.json_ut)}")
    print(f"Rapport: {os.path.abspath(args.rapport_ut)}")


if __name__ == "__main__":
    main()
