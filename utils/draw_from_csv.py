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

from config import MAX_WIDTH_ELECTRONIC_PT
from csv_export import read_result_csv
from evaluation import read_truth_xywh, evaluate_against_truth
from load_pdf import PDF_DPI, read_pages
from visualization import draw_and_save
from yolo_fnr import find_yolo_boxes

SCALE = PDF_DPI / 72.0                     # PDF points -> pixels


def _filter_for_wide(sladd_boxes, max_pt):
    ut = {}
    removed = 0
    for (name, si), (bw, bh, boxes) in sladd_boxes.items():
        kept = [b for b in boxes if (b[2] - b[0]) / SCALE <= max_pt]
        removed += len(boxes) - len(kept)
        ut[(name, si)] = (bw, bh, kept)
    return ut, removed


def _filter_low_conf(sladd_boxes, min_conf):
    ut = {}
    removed = 0
    for (name, si), (bw, bh, boxes) in sladd_boxes.items():
        kept = []
        for b in boxes:
            source = b[4] if len(b) > 4 else "paddle"
            conf = b[5] if len(b) > 5 else None
            if source == "yolo" and conf is not None and conf < min_conf:
                removed += 1
                continue
            kept.append(b)
        ut[(name, si)] = (bw, bh, kept)
    return ut, removed


def _read_prod_csv(path, folder):
    per_file = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            per_file[r["navn"]].append((int(r["side"]), float(r["x"]), float(r["y"]),
                                       float(r["width"]), float(r["height"])))

    sladd_boxes = {}
    for name in sorted(per_file):
        try:
            d = fitz.open(os.path.join(folder, name))
        except Exception as e:
            print(f"   {name}: could not be opened ({e!r})")
            continue
        for (si, x, y, w, h) in sorted(per_file[name]):
            if not 1 <= si <= len(d):
                continue
            rect = d[si - 1].rect
            bw = int(round(rect.width * SCALE))
            bh = int(round(rect.height * SCALE))
            x0, x1 = sorted((x, x + w))
            y0, y1 = sorted((y, y + h))
            # A prod CSV carries neither source nor conf -> None = unknown (gray frame)
            box = (x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE, None, None)
            sladd_boxes.setdefault((name, si), (bw, bh, []))[2].append(box)
        d.close()
    return sladd_boxes


def _read_box_csv(path, folder, prod=False):
    if prod:
        return _read_prod_csv(path, folder)
    with open(path, newline="", encoding="utf-8-sig") as f:
        field = csv.DictReader(f).fieldnames or []
    if "bilde_bredde" in field:
        return read_result_csv(path)
    return _read_prod_csv(path, folder)


def _run_yolo_on_dir(sladd_boxes, folder):
    yolo_boxes = {}
    per_file = {}
    for (name, si) in sladd_boxes:
        per_file.setdefault(name, set()).add(si)

    for name, pages in sorted(per_file.items()):
        path = os.path.join(folder, name)
        try:
            images = read_pages(path)
        except Exception as e:
            print(f"   YOLO: could not read {name}: {e!r}")
            continue
        for si in pages:
            if not 1 <= si <= len(images):
                continue
            image = images[si - 1]
            h, w = image.shape[:2]
            hit = find_yolo_boxes(image)
            if hit:
                boxes = [(x0, y0, x1, y1, conf) for x0, y0, x1, y1, conf in hit]
                yolo_boxes[(name, si)] = (w, h, boxes)
    return yolo_boxes


def _quiet_eval(sladd_boxes, truth, folder, threshold, y_origin):
    with io.StringIO() as buf, redirect_stdout(buf):
        return evaluate_against_truth(sladd_boxes, truth, folder,
                            threshold=threshold, y_origin=y_origin,
                            sources=sladd_boxes)


def _write_before_after(before, after, filter_name):
    if not before or not after:
        return
    print("\n" + "=" * 64)
    print(f"Before/after filter ({', '.join(filter_name)}):")
    rows = [
        ("Recall",         f"{before['hit']}/{before['fasit']} = {before['recall']:.0%}",
                           f"{after['hit']}/{after['fasit']} = {after['recall']:.0%}",
                           after["hit"] - before["hit"]),
        ("Sladd boxes",    str(before["pred"]), str(after["pred"]),
                           after["pred"] - before["pred"]),
        ("Over-sladding",  str(before["oversladd"]), str(after["oversladd"]),
                           after["oversladd"] - before["oversladd"]),
        ("Total overlap",  f"{before['total_overlap']:.0%}",
                            f"{after['total_overlap']:.0%}", None),
    ]
    print(f"   {'':<18}{'before':>16}{'after':>16}{'change':>12}")
    for name, a, b, delta in rows:
        d = "" if delta is None else f"{delta:+d}"
        print(f"   {name:<18}{a:>16}{b:>16}{d:>12}")


def main():
    p = argparse.ArgumentParser(
        description="Draw boxes from a finished CSV straight to PNG, without running the "
                    "model. Header must be: navn,side,bilde_bredde,bilde_hoyde,x0,y0,x1,y1")
    p.add_argument("--csv", default="res.csv",
                   help="the CSV holding the boxes")
    p.add_argument("--folder", default="../uttrekk_3",
                   help="directory of the PDFs the CSV refers to")
    p.add_argument("--png-dir", default="visning", help="where the PNGs are written")
    p.add_argument("--truth-csv", default="smartsladding_uttrekk_labels_3_07_07_26.csv",
                   help="truth CSV; drawn as green frames")
    p.add_argument("--y-origin", choices=["top", "bottom"], default="top", help="y origin of the truth boxes")
    p.add_argument("--select", nargs="+", metavar="PDF",
                   help="limit to these PDF files (filename, .pdf optional)")
    p.add_argument("--prod", action="store_true",
                   help="read the CSV as prod format (points, no source/conf) - every box "
                        "gets a gray frame because the source is unknown")
    p.add_argument("--yolo", action="store_true",
                   help="run YOLO and show hits as red frames with conf")
    p.add_argument("--truth", action="store_true",
                   help="measure recall against the truth and print it")
    p.add_argument("--threshold", type=float, default=0.32,
                   help="overlap threshold for a hit (default 0.32)")
    p.add_argument("--only-oversladd", action="store_true",
                   help="draw only pages with over-sladding (a box with no truth hit), "
                        "written to kun_oversladd/ under --png-dir")
    p.add_argument("--only-miss", action="store_true",
                   help="draw only pages with at least one miss (needs --truth)")
    p.add_argument("--min-conf", type=float, default=None, metavar="CONF",
                   help="drop YOLO-only boxes with conf below CONF (paddle and both-source "
                        "boxes are kept regardless)")
    p.add_argument("--max-width", type=float, nargs="?", const=MAX_WIDTH_ELECTRONIC_PT, default=None,
                   metavar="PT",
                   help=f"drop boxes wider than PT points (bare flag: {MAX_WIDTH_ELECTRONIC_PT} from config)")
    args = p.parse_args()

    sladd_boxes = _read_box_csv(args.csv, args.folder, prod=args.prod)
    if args.select:
        select_apply = {os.path.basename(f) for f in args.select}
        sladd_boxes = {k: v for k, v in sladd_boxes.items()
                        if os.path.basename(k[0]) in select_apply}
    n_boxes = sum(len(v[2]) for v in sladd_boxes.values())
    print(f"Read {n_boxes} box(es) across {len(sladd_boxes)} (file, page) groups from {args.csv}")

    unfiltered = sladd_boxes          # kept for the before/after measurement
    filter_name = []

    if args.min_conf is not None:
        sladd_boxes, removed = _filter_low_conf(sladd_boxes, args.min_conf)
        n_boxes -= removed
        print(f"Conf filter: dropped {removed} YOLO-only box(es) with conf below "
              f"{args.min_conf:g} ({n_boxes} left)")
        filter_name.append(f"conf>={args.min_conf:g}")

    if args.max_width is not None:
        sladd_boxes, removed = _filter_for_wide(sladd_boxes, args.max_width)
        print(f"Width filter: dropped {removed} box(es) wider than {args.max_width:g} pt "
              f"({n_boxes - removed} left)")
        filter_name.append(f"width<={args.max_width:g}pt")

    yolo_boxes = None
    if args.yolo:
        print("Running YOLO...")
        yolo_boxes = _run_yolo_on_dir(sladd_boxes, args.folder)
        n_yolo = sum(len(v[2]) for v in yolo_boxes.values())
        print(f"YOLO found {n_yolo} box(es)")
        if args.min_conf is not None:
            yolo_boxes = {k: (w, h, [b for b in bs if b[4] >= args.min_conf])
                           for k, (w, h, bs) in yolo_boxes.items()}
            n_left = sum(len(v[2]) for v in yolo_boxes.values())
            print(f"Conf filter: dropped {n_yolo - n_left} YOLO box(es) with conf below "
                  f"{args.min_conf:g} ({n_left} left)")

    truth = read_truth_xywh(args.truth_csv)

    # Only worth measuring the unfiltered set when there is a filter to compare against
    before_res = None
    if filter_name and truth:
        before_res = _quiet_eval(unfiltered, truth, args.folder, args.threshold, args.y_origin)

    if args.only_oversladd:
        eval_res = _quiet_eval(sladd_boxes, truth, args.folder, args.threshold, args.y_origin)
        _write_before_after(before_res, eval_res, filter_name)
        oversladd_pages = {(r["fil"], r["side"]) for r in (eval_res or {}).get("oversladd_files", [])}
        if not oversladd_pages:
            print("No over-sladding found.")
            return
        ov_boxes = (eval_res or {}).get("oversladd_boxes", {})
        sladd_boxes = {k: v for k, v in sladd_boxes.items() if k in oversladd_pages}
        out_dir = os.path.join(args.png_dir, "kun_oversladd")
        print(f"Drawing {len(oversladd_pages)} page(s) with over-sladding...")
        draw_and_save(sladd_boxes, truth, args.folder, out_dir,
                      y_origin=args.y_origin, sources=sladd_boxes,
                      oversladd_boxes=ov_boxes)
    else:
        miss_indices = None
        oversladd = None
        if args.truth:
            eval_res = evaluate_against_truth(sladd_boxes, truth, args.folder,
                                    threshold=args.threshold, y_origin=args.y_origin,
                                    sources=sladd_boxes)
            _write_before_after(before_res, eval_res, filter_name)
            from visualization import _doc_no as _vnr
            miss_indices = {
                (_vnr(d["fil"]), d["side"], d["fasit_nr"] - 1)
                for d in eval_res.get("details", [])
                if d["result"] == "MISSING"
            }
            oversladd = eval_res.get("oversladd_boxes", None)
            if args.only_miss:
                miss_pages = {(d["fil"], d["side"]) for d in eval_res.get("details", []) if d["result"] == "MISSING"}
                sladd_boxes = {k: v for k, v in sladd_boxes.items() if k in miss_pages}
                print(f"Filtered to {len(miss_pages)} page(s) with a miss.")
        draw_and_save(sladd_boxes, truth, args.folder, args.png_dir,
                      y_origin=args.y_origin, yolo_boxes=yolo_boxes,
                      sources=sladd_boxes, miss_indices=miss_indices,
                      oversladd_boxes=oversladd)


if __name__ == "__main__":
    main()