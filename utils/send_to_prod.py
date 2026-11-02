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

from evaluation import read_truth_xywh, _doc_no
from geometry import intersection_area, area
from utils_config import HIT_THRESHOLD

FIELD = ["navn", "side", "x", "y", "width", "height"]


def _to_xyxy(x, y, w, h):
    """(x, y, w, h) -> (x0, y0, x1, y1), robust against negative width/height."""
    x0, x1 = sorted((x, x + w))
    y0, y1 = sorted((y, y + h))
    return (x0, y0, x1, y1)


def _select_exact(folder, ids):
    """Exact match on the filename stem, not substring. Returns (files, missing)."""
    idset = {os.path.splitext(s.strip())[0] for s in ids if s.strip()}
    chosen = []
    for fn in sorted(os.listdir(folder)):
        if os.path.splitext(fn)[0] in idset:
            chosen.append(os.path.join(folder, fn))
    found = {os.path.splitext(os.path.basename(f))[0] for f in chosen}
    missing = sorted(idset - found)
    return chosen, missing


def _send_one(sess, url, pdf_bytes, timeout):
    r = sess.post(url, data=pdf_bytes,
                  headers={"Content-Type": "application/pdf"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _evaluate(pred_boxes, sent_doc_no, truth, threshold):
    """Hits + oversladding in point space, only for the documents we sent.

    pred_boxes:  {(docnr, page): [(x0,y0,x1,y1), ...]}  prod boxes in points
    sent_doc_no: docnrs that were sent OK
    truth:       {(fil_revisjon_id, page): [(x,y,w,h,type), ...]}
    """
    hit = truth_tot = oversladd = 0
    matched = defaultdict(set)

    # recall: walk every truth box for the documents we sent
    for (nr, page), filtered_boxes in truth.items():
        if nr not in sent_doc_no:
            continue
        preds = pred_boxes.get((nr, page), [])
        for (fx, fy, fw, fh, _t) in filtered_boxes:
            f = _to_xyxy(fx, fy, fw, fh)
            fa = area(f)
            best, bpi = 0.0, -1
            for pi, pb in enumerate(preds):
                ov = intersection_area(f, pb)
                if ov > best:
                    best, bpi = ov, pi
            truth_tot += 1
            if fa and best / fa >= threshold:
                hit += 1
                matched[(nr, page)].add(bpi)

    # oversladding: prod boxes that hit no truth box
    for key, preds in pred_boxes.items():
        oversladd += len(preds) - len(matched.get(key, set()))
    return hit, truth_tot, oversladd


def main():
    p = argparse.ArgumentParser(description="Send PDFs to the prod API, store the responses and measure hits against truth.")
    p.add_argument("--folder", default="/data2/smartsladding-uttrekk/uttrekk_4/",
                   help="folder of PDFs")
    p.add_argument("--ids-file", default="ids_server.txt",
                   help="file with one document name/id per line (exact match)")
    p.add_argument("--url", default="http://localhost:5071/model", help="prod API endpoint")
    p.add_argument("--csv-out", default="res_prod.csv", help="prod boxes in points (navn,side,x,y,width,height)")
    p.add_argument("--json-out", default="res_prod.jsonl",
                   help="raw API responses, one JSON object per line")
    p.add_argument("--elektronisk-tinglyst", action="store_true",
                   help="set ?elektronisk_tinglyst=true (turns YOLO off)")
    p.add_argument("--timeout", type=float, default=120, help="timeout per PDF (seconds)")
    p.add_argument("--truth-csv", default="/home/smartsladding/smartsladding-uttrekk-labels/uttrekk_4.csv",
                   help="truth CSV for the automatic evaluation at the end")
    p.add_argument("--report-out", default="res_prod_fasit.txt",
                   help="txt report with hits/oversladding/timing")
    p.add_argument("--threshold", type=float, default=HIT_THRESHOLD,
                   help=f"fraction of truth area required for a hit (default {HIT_THRESHOLD})")
    p.add_argument("--no-truth", action="store_true", help="skip the automatic evaluation against truth")
    args = p.parse_args()

    url = args.url
    if args.elektronisk_tinglyst:
        url += ("&" if "?" in url else "?") + "elektronisk_tinglyst=true"

    # --- select files (exact match against the id file) ---
    try:
        with open(args.ids_file, encoding="utf-8") as f:
            ids = f.read().split()
    except FileNotFoundError:
        print(f"id file not found: {args.ids_file}")
        return
    files, missing = _select_exact(args.folder, ids)
    print(f"id file: {len(set(ids))} unique ids -> {len(files)} files found in {args.folder}")
    if missing:
        print(f"  {len(missing)} ids with no matching file. First 5: {missing[:5]}")
    if not files:
        print("No files to process.")
        return

    with open(args.csv_out, "w", newline="", encoding="utf-8") as cf:
        csv.writer(cf).writerow(FIELD)
    print(f"Sending {len(files)} PDFs to {url}")
    print(f"Boxes -> {args.csv_out} | raw responses -> {args.json_out}\n")

    sess = requests.Session()
    n_boxes = n_ok = 0
    failed = []
    pred_boxes = defaultdict(list)
    sent_doc_no = set()
    api_time = 0.0
    start = time.perf_counter()

    with open(args.json_out, "w", encoding="utf-8") as jf, \
         open(args.csv_out, "a", newline="", encoding="utf-8") as cf:
        write = csv.writer(cf)
        for i, file in enumerate(files, start=1):
            name = os.path.basename(file)
            try:
                with open(file, "rb") as f:
                    pdf_bytes = f.read()
                t0 = time.perf_counter()
                result = _send_one(sess, url, pdf_bytes, args.timeout)
                dt = time.perf_counter() - t0
                api_time += dt

                if not isinstance(result, list):
                    raise ValueError(f"expected a list, got {type(result).__name__}: {str(result)[:120]}")

                jf.write(json.dumps({"navn": name, "sekunder": round(dt, 3),
                                     "result": result}, ensure_ascii=False) + "\n")
                jf.flush()

                nr = _doc_no(name)
                doc_boxes = 0
                for b in result:
                    page = b["page"]
                    x, y, w, h = b["x"], b["y"], b["width"], b["height"]
                    write.writerow([name, page, x, y, w, h])
                    pred_boxes[(nr, page)].append(_to_xyxy(x, y, w, h))
                    doc_boxes += 1
                cf.flush()
            except Exception as e:
                failed.append((name, repr(e)))
                print(f"[{i}/{len(files)}] ERROR {name}: {e!r}")
                continue

            sent_doc_no.add(nr)
            n_boxes += doc_boxes
            n_ok += 1
            mean = api_time / n_ok
            eta = mean * (len(files) - i)
            print(f"[{i}/{len(files)}] {name}: {doc_boxes} box(es), {dt:.1f}s "
                  f"(avg {mean:.1f}s, ETA ~{eta/60:.0f} min)")

    total = time.perf_counter() - start

    # --- report ---
    L = []
    L.append("=" * 60)
    L.append("SEND TO PROD - RESULT")
    L.append("=" * 60)
    L.append(f"Time:               {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"URL:                {url}")
    L.append(f"Documents OK:       {n_ok}/{len(files)}")
    L.append(f"Failed:             {len(failed)}")
    L.append(f"Boxes in total:     {n_boxes}")
    L.append("")
    L.append("--- TIME ---")
    L.append(f"Wall clock total:   {total:.1f}s ({total/60:.1f} min)")
    L.append(f"Pure API time:      {api_time:.1f}s ({api_time/60:.1f} min)")
    if n_ok:
        L.append(f"Avg per document:   {api_time/n_ok:.2f}s")

    if not args.no_truth and n_ok:
        truth = read_truth_xywh(args.truth_csv)
        if truth is None:
            L.append("")
            L.append(f"!! Truth CSV not found: {args.truth_csv} - evaluation skipped.")
        else:
            tr, truth_tot, ov = _evaluate(pred_boxes, sent_doc_no, truth, args.threshold)
            rec = tr / truth_tot if truth_tot else 0.0
            L.append("")
            L.append(f"--- TRUTH (overlap threshold {args.threshold:.2f}) ---")
            L.append(f"Truth boxes:        {truth_tot}")
            L.append(f"Hits:               {tr}")
            L.append(f"Recall:             {rec:.2%}")
            L.append(f"Missed:             {truth_tot - tr}")
            L.append(f"Oversladding:       {ov}")

    if failed:
        L.append("")
        L.append("--- FAILED DOCUMENTS (first 20) ---")
        for name, e in failed[:20]:
            L.append(f"  {name}: {e}")
    L.append("=" * 60)

    report = "\n".join(L)
    print("\n" + report)
    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nCSV:     {os.path.abspath(args.csv_out)}")
    print(f"JSONL:   {os.path.abspath(args.json_out)}")
    print(f"Report:  {os.path.abspath(args.report_out)}")


if __name__ == "__main__":
    main()
