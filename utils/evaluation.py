import os
import re
import sys
from collections import defaultdict

import fitz

from save_result import write_result_files


def _doc_no(name):
    m = re.match(r"0*(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


def _norm_csv(x, y, w, h, pw, ph, y_origin):
    x0, x1 = sorted((x, x + w))
    y0, y1 = sorted((y, y + h))
    if y_origin == "bottom":
        y0, y1 = ph - y1, ph - y0
    return (x0 / pw, y0 / ph, x1 / pw, y1 / ph)


def _overlap(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return (ix1 - ix0) * (iy1 - iy0) if (ix1 > ix0 and iy1 > iy0) else 0.0


def _area(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _page_str(name, si, folder, sladd_boxes):
    file = os.path.join(folder, name)
    if file.lower().endswith(".pdf"):
        try:
            d = fitz.open(file)
            r = d[si - 1].rect
            d.close()
            return r.width, r.height
        except Exception:
            pass
    iw, ih, _ = sladd_boxes[(name, si)]
    return iw, ih

def evaluate_against_truth(sladd_boxes, truth, folder, threshold=0.32, y_origin="top", sources=None,
                 yolo_boxes=None, write=None, diagnostics=True):
    """Measures sladd boxes against the truth labels.

    write        receives the report lines (same signature as print). Threads can
                   call concurrently by each passing its own collector;
                   redirect_stdout would swap the global sys.stdout.
    diagnostics  prints the ID diagnostics when no truth boxes were found. Only
                   meaningful for a whole run: for a single document "no truth"
                   usually just means the document is unlabelled.
    """
    write = write or print

    if truth is None:
        write("No truth labels, skipping measurement.")
        return None

    total_truth = total_hit = total_pred = total_surplus = 0
    total_ov_area = total_fa_area = 0.0
    pr_type = defaultdict(lambda: [0, 0])
    miss_files = defaultdict(lambda: [0, 0])
    surplus_files = defaultdict(int)
    oversladd_boxes = {}   # (navn, si) -> (iw, ih, [(x0,y0,x1,y1)])
    details = []

    for (name, si) in sorted(sladd_boxes):
        nr = _doc_no(name)
        iw, ih, raw = sladd_boxes[(name, si)]
        yolo_coords = set()
        if yolo_boxes and (name, si) in yolo_boxes:
            _, _, yolo_raw = yolo_boxes[(name, si)]
            yolo_coords = set(yolo_raw)
        source_names = []
        conf_names = []
        if sources and (name, si) in sources:
            _, _, with_source = sources[(name, si)]
            source_names = [b[4] if len(b) > 4 else "paddle" for b in with_source]
            conf_names  = [b[5] if len(b) > 5 else None for b in with_source]
        pw, ph = _page_str(name, si, folder, sladd_boxes)
        pred = [(b[0] / iw, (b[1] - 2) / ih, b[2] / iw, (b[3] + 2) / ih) for b in raw]
        filtered_boxes = [(_norm_csv(x, y, w, h, pw, ph, y_origin), t)
                   for (x, y, w, h, t) in truth.get((nr, si), [])]

        total_pred += len(pred)
        total_truth += len(filtered_boxes)
        hit_pred = set()
        if filtered_boxes:
            write(f"\n{name}  (doc_no={nr}, page {si})")
        for fi, (fb, t) in enumerate(filtered_boxes):
            fa = _area(fb)
            best_cov = best_iou = best_ov = 0.0
            best_pi = -1
            for pi, pb in enumerate(pred):
                ov = _overlap(fb, pb)
                if ov > best_ov:
                    best_ov = ov
                    best_cov = ov / fa if fa else 0.0
                    best_iou = ov / (fa + _area(pb) - ov)
                    best_pi = pi
            hit = best_cov >= threshold
            if source_names:
                source = source_names[best_pi] if (0 <= best_pi < len(source_names)) else ""
                source = source or "unknown"   # prod CSV has no kilde column
                conf  = conf_names[best_pi]  if (0 <= best_pi < len(conf_names)) else None
            else:
                source = "yolo" if (best_pi >= 0 and raw[best_pi] in yolo_coords) else "paddle"
                conf  = None
            pr_type[t][1] += 1
            total_fa_area += fa
            total_ov_area += best_ov
            miss_files[(name, si)][1] += 1
            details.append({
                "fil": name, "side": si, "fasit_nr": fi + 1, "type": t,
                "coverage_pct": round(best_cov * 100, 1),
                "result": "HIT" if hit else "MISSING",
                "kilde": source if hit else "",
                "conf": round(conf, 3) if (hit and conf is not None) else "",
                "fasit_x0": round(fb[0], 6),
                "fasit_y0": round(fb[1], 6),
                "fasit_x1": round(fb[2], 6),
                "fasit_y1": round(fb[3], 6),
            })
            if hit:
                total_hit += 1
                pr_type[t][0] += 1
                hit_pred.add(best_pi)
            else:
                miss_files[(name, si)][0] += 1
            write(f"   truth#{fi + 1} {t:<22} coverage={best_cov:5.0%}  IoU={best_iou:5.0%}  "
                  f"-> {'HIT' if hit else 'MISSING'}")
        n_surplus = len(pred) - len(hit_pred)
        total_surplus += n_surplus
        if n_surplus > 0:
            surplus_files[(name, si)] += n_surplus
            over_boxes = [raw[i] for i in range(len(raw)) if i not in hit_pred]
            oversladd_boxes[(name, si)] = (iw, ih, over_boxes)

    write("\n" + "=" * 64)
    rec = total_hit / total_truth if total_truth else 0.0
    write(f"Recall (hits / truth):        {total_hit}/{total_truth} = {rec:.0%}")
    write(f"Total overlap (area):         {(total_ov_area / total_fa_area if total_fa_area else 0):.0%}")
    write(f"Sladd boxes in total:         {total_pred}")
    write(f"Oversladding (no truth hit):  {total_surplus}")
    write(f"Hit threshold:                {threshold:.0%} of the truth box area")

    eval_doc_nos = {_doc_no(n) for (n, _) in sladd_boxes}
    eval_doc_nos.discard(None)
    truth_doc_nos = {nr for (nr, _) in truth}
    common = eval_doc_nos & truth_doc_nos
    n_truth_total = sum(len(v) for v in truth.values())

    if diagnostics and total_truth == 0 and n_truth_total > 0 and eval_doc_nos:
        write(f"\n!! WARNING: none of the {len(eval_doc_nos)} evaluated documents "
              f"match the {len(truth_doc_nos)} documents in the truth labels.")
        write(f"   Evaluated (doc_no from filename), 5 lowest: {sorted(eval_doc_nos)[:5]}")
        write(f"   Truth (fil_revisjon_id), 5 lowest:          {sorted(truth_doc_nos)[:5]}")
        eval_name_examples = sorted({n for (n, _) in sladd_boxes})[:3]
        write(f"   Filename examples:                          {eval_name_examples}")
        write(f"   Check that the filenames in --folder match fil_revisjon_id in --truth-csv.")
    elif diagnostics and common and len(common) < len(eval_doc_nos):
        n_without = len(eval_doc_nos) - len(common)
        write(f"\n   Info: {len(common)}/{len(eval_doc_nos)} evaluated documents have "
              f"truth labels ({n_without} without truth boxes)")

    write("Recall per type:")
    for t, (tr, tot) in sorted(pr_type.items()):
        write(f"   {t or '(empty)':<22} {tr}/{tot} = {tr / tot:.0%}")

    error = sorted((k for k, (b, _tot) in miss_files.items() if b > 0))
    write("\n" + "=" * 64)
    if total_truth == 0:
        if n_truth_total > 0:
            write(f"No truth box matched the evaluated documents "
                  f"({n_truth_total} truth boxes exist, but for other documents).")
        else:
            write("No truth boxes loaded, cannot measure recall.")
    elif error:
        write(f"Files with bom ({len(error)} page(s) with at least one MISSING):")
        for (name, si) in error:
            miss, tot = miss_files[(name, si)]
            write(f"   {name}  page {si}:  {miss}/{tot} truth boxes missed")
    else:
        write("No bom, every truth box was hit. 🎉")


    return {
        "recall": rec, "hit": total_hit, "fasit": total_truth,
        "pred": total_pred, "surplus": total_surplus,
        "total_overlap": total_ov_area / total_fa_area if total_fa_area else 0.0,
        "threshold": threshold,
        "pr_type": {t: tuple(v) for t, v in pr_type.items()},
        "details": details,
        "miss_files": [
            {"fil": name, "side": si, "bom": b, "truth_total": tot}
            for (name, si), (b, tot) in sorted(miss_files.items()) if b > 0
        ],
        "oversladd_boxes": oversladd_boxes,
        "surplus_files": [
            {"fil": name, "side": si, "oversladd": n}
            for (name, si), n in sorted(surplus_files.items())
        ],
    }

import csv
from collections import defaultdict


def read_truth_xywh(csv_path):
    from filter_common import iter_label_rows
    truth = defaultdict(list)
    try:
        for r in iter_label_rows(csv_path):
            try:
                nr = int(r["fil_revisjon_id"])
                page = int(r["sidetall"])
                x, y = float(r["x"]), float(r["y"])
                w, h = float(r["width"]), float(r["height"])
            except (TypeError, ValueError, KeyError):
                continue
            truth[(nr, page)].append((x, y, w, h, (r.get("type") or "").strip()))
    except FileNotFoundError:
        print(f"!! CSV not found: {csv_path}, truth is not drawn and hits are not measured.")
        return None

    print(f"Truth: {sum(len(v) for v in truth.values())} box(es) in "
          f"{len(truth)} (doc_no, page) groups from {csv_path}.")
    return truth
