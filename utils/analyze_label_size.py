"""Box dimensions from a labels CSV, to find usable filter thresholds.

Run:
    python utils/analyze_label_size.py --csv smartsladding_uttrekk_labels_5_29_07_26.csv
    python utils/analyze_label_size.py --res-csv utils/magnusruler.csv
"""

import argparse
import csv
import statistics

SCALE = 300 / 72.0  # PDF points -> pixels at 300 DPI


def read_labels(path):
    boxes = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                w = float(r["width"])
                h = float(r["height"])
                ml = (r.get("ml_generated") or "").strip().lower() == "true"
                status = (r.get("ml_status") or "").strip().upper()
            except (TypeError, ValueError, KeyError):
                continue
            boxes.append({"w": w, "h": h, "ml": ml, "status": status,
                           "area": abs(w * h), "ratio": w / h if h != 0 else 0,
                           "kilde": "ml" if ml else "manual"})
    return boxes


def read_result(path):
    """Reads a result CSV, whose coordinates are pixels rather than PDF points."""
    boxes = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                x0, y0 = float(r["x0"]), float(r["y0"])
                x1, y1 = float(r["x1"]), float(r["y1"])
                source = r.get("kilde", "unknown")
                # yolo_conf in the current format, conf in older result CSVs
                raw_conf = r.get("yolo_conf") or r.get("conf")
                conf = float(raw_conf) if raw_conf else None
            except (TypeError, ValueError, KeyError):
                continue
            w_px = abs(x1 - x0)
            h_px = abs(y1 - y0)
            w_pt = w_px / SCALE
            h_pt = h_px / SCALE
            boxes.append({"w": w_pt, "h": h_pt, "w_px": w_px, "h_px": h_px,
                           "area": w_pt * h_pt, "kilde": source, "conf": conf,
                           "ratio": w_pt / h_pt if h_pt > 0 else 0,
                           "ml": True, "status": "PRED"})
    return boxes


def percentiles(values, name):
    values = sorted(values)
    n = len(values)
    ps = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  {name} (n={n}):")
    print(f"    min={values[0]:.2f}  max={values[-1]:.2f}  "
          f"mean={statistics.mean(values):.2f}  std={statistics.stdev(values):.2f}")
    for p in ps:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        print(f"    P{p:02d} = {values[idx]:.2f}")


def _analyse_group(boxes, title):
    if not boxes:
        print(f"\n  (no boxes in '{title}')")
        return
    print(f"\n{'=' * 60}")
    print(f"{title} (n={len(boxes)}):")
    percentiles([b["w"] for b in boxes], "Width (pt)")
    percentiles([b["h"] for b in boxes], "Height (pt)")
    percentiles([b["area"] for b in boxes], "Area (pt²)")
    percentiles([b["ratio"] for b in boxes], "Ratio (w/h)")


def _analyse_per_source(boxes):
    sources = sorted(set(b["kilde"] for b in boxes))
    for source in sources:
        group = [b for b in boxes if b["kilde"] == source]
        _analyse_group(group, f"Source: {source}")


def main():
    p = argparse.ArgumentParser(description="Analyse box dimensions in a labels CSV and/or a result CSV")
    p.add_argument("--csv", default=None,
                   help="labels CSV (truth, width/height in PDF points)")
    p.add_argument("--res-csv", default=None,
                   help="result CSV from the model (pixel coordinates: x0,y0,x1,y1)")
    args = p.parse_args()

    if not args.csv and not args.res_csv:
        args.csv = "smartsladding_uttrekk_labels_5_29_07_26.csv"

    if args.csv:
        all_of = read_labels(args.csv)
        print(f"Read {len(all_of)} boxes from {args.csv}")

        accepted = [b for b in all_of if b["ml"] and b["status"] == "ACCEPTED"]
        rejected = [b for b in all_of if b["ml"] and b["status"] == "REJECTED"]
        manual = [b for b in all_of if not b["ml"]]

        neg = [b for b in all_of if b["w"] <= 0 or b["h"] <= 0]
        print(f"\nBoxes with negative width or height: {len(neg)}")
        if neg:
            for b in neg[:5]:
                print(f"  w={b['w']:.1f} h={b['h']:.1f} ml={b['ml']} status={b['status']}")
            if len(neg) > 5:
                print(f"  ... and {len(neg) - 5} more")

        pos = [b for b in all_of if b["w"] > 0 and b["h"] > 0]
        pos_accepted = [b for b in accepted if b["w"] > 0 and b["h"] > 0]

        _analyse_group(pos, "ALL positive boxes")
        _analyse_group(pos_accepted, "ACCEPTED only (ml_generated + accepted)")

        print(f"\n{'=' * 60}")
        print(f"ACCEPTED in pixels (300 DPI, scale={SCALE:.3f}):")
        percentiles([b["w"] * SCALE for b in pos_accepted], "Width (px)")
        percentiles([b["h"] * SCALE for b in pos_accepted], "Height (px)")
        percentiles([b["w"] * b["h"] * SCALE**2 for b in pos_accepted], "Area (px²)")

        min_area_px = 965
        max_width_pt = 50

        print(f"\n{'=' * 60}")
        print("Compared with the current filters:")
        print(f"  MIN_BOX_AREA = {min_area_px} px²")
        under_min = sum(1 for b in pos_accepted if b["w"] * b["h"] * SCALE**2 < min_area_px)
        print(f"    ACCEPTED boxes that would be filtered out: {under_min}/{len(pos_accepted)}")

        print(f"  MAKS_BREDDE_PT = {max_width_pt} pt ({max_width_pt * SCALE:.0f} px)")
        over_max = sum(1 for b in pos_accepted if b["w"] > max_width_pt)
        print(f"    ACCEPTED boxes that would be filtered out: {over_max}/{len(pos_accepted)}")

        print(f"\n{'=' * 60}")
        print("Possible new filters (from the truth data):")
        heights = sorted(b["h"] for b in pos_accepted)
        p99_h = heights[int(len(heights) * 0.99)]
        print(f"  Max height: P99={p99_h:.1f} pt -> suggestion: {p99_h * 1.5:.0f} pt")

        widths = sorted(b["w"] for b in pos_accepted)
        p01_w = widths[int(len(widths) * 0.01)]
        print(f"  Min width: P01={p01_w:.1f} pt -> suggestion: {p01_w * 0.7:.0f} pt")

        ratioer = sorted(b["ratio"] for b in pos_accepted)
        p01_r = ratioer[int(len(ratioer) * 0.01)]
        p99_r = ratioer[int(len(ratioer) * 0.99)]
        print(f"  Ratio (w/h): P01={p01_r:.2f}  P99={p99_r:.2f}")
        print(f"    -> suggested min ratio: {p01_r * 0.7:.1f} (an fnr is always wider than it is tall)")

        areas = sorted(b["area"] for b in pos_accepted)
        p99_a = areas[int(len(areas) * 0.99)]
        print(f"  Max area: P99={p99_a:.0f} pt² -> suggestion: {p99_a * 1.5:.0f} pt²")

    if args.res_csv:
        pred = read_result(args.res_csv)
        print(f"\n\n{'#' * 60}")
        print(f"RESULT CSV: {args.res_csv}")
        print(f"{len(pred)} predicted boxes")

        pos_pred = [b for b in pred if b["w"] > 0 and b["h"] > 0]
        _analyse_group(pos_pred, "All predictions (pt)")

        _analyse_per_source(pos_pred)

        print(f"\n{'=' * 60}")
        print("Predictions in pixels:")
        percentiles([b["w_px"] for b in pos_pred], "Width (px)")
        percentiles([b["h_px"] for b in pos_pred], "Height (px)")
        percentiles([b["w_px"] * b["h_px"] for b in pos_pred], "Area (px²)")

        min_area_px = 965
        max_width_pt = 50
        print(f"\n{'=' * 60}")
        print("Current filters against the predictions:")
        print(f"  MIN_BOX_AREA = {min_area_px} px²")
        under_min = sum(1 for b in pos_pred if b["w_px"] * b["h_px"] < min_area_px)
        print(f"    Predictions that WOULD be filtered: {under_min}/{len(pos_pred)}")

        print(f"  MAKS_BREDDE_PT = {max_width_pt} pt (electronic only)")
        over_max = sum(1 for b in pos_pred if b["w"] > max_width_pt)
        print(f"    Predictions wider than {max_width_pt} pt: {over_max}/{len(pos_pred)}")

        print(f"\n{'=' * 60}")
        print("Possible outliers among the predictions:")
        outliers = [b for b in pos_pred if b["ratio"] < 1.2 or b["area"] > 3300 or b["h"] > 50]
        print(f"  Boxes with ratio < 1.2 OR area > 3300 pt² OR height > 50 pt: {len(outliers)}/{len(pos_pred)}")
        for b in sorted(outliers, key=lambda x: -x["area"])[:15]:
            print(f"    w={b['w']:.1f}pt h={b['h']:.1f}pt area={b['area']:.0f}pt² "
                  f"ratio={b['ratio']:.2f} kilde={b['kilde']} conf={b['conf']}")

        if args.csv:
            print(f"\n{'=' * 60}")
            print("COMPARISON truth vs. predictions (median):")
            print(f"  {'':20} {'Truth (ACCEPTED)':>20} {'Predictions':>20}")
            if pos_accepted and pos_pred:
                fw = statistics.median(b["w"] for b in pos_accepted)
                pw = statistics.median(b["w"] for b in pos_pred)
                print(f"  {'Width (pt)':<20} {fw:>20.1f} {pw:>20.1f}")
                fh = statistics.median(b["h"] for b in pos_accepted)
                ph = statistics.median(b["h"] for b in pos_pred)
                print(f"  {'Height (pt)':<20} {fh:>20.1f} {ph:>20.1f}")
                fa = statistics.median(b["area"] for b in pos_accepted)
                pa = statistics.median(b["area"] for b in pos_pred)
                print(f"  {'Area (pt²)':<20} {fa:>20.0f} {pa:>20.0f}")
                fr = statistics.median(b["ratio"] for b in pos_accepted)
                pr = statistics.median(b["ratio"] for b in pos_pred)
                print(f"  {'Ratio (w/h)':<20} {fr:>20.2f} {pr:>20.2f}")


if __name__ == "__main__":
    main()



