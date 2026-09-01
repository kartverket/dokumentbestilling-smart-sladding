import argparse
import csv
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from filter_common import iter_label_rows



def _doc_no(name):
    m = re.match(r"0*(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


def read_details(folder):
    path = folder / "detaljer.csv"
    if not path.exists():
        sys.exit(f"No {path}, is this a result directory from run.py?")
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["side"] = int(r["side"])
        r["coverage_pct"] = float(r["coverage_pct"])
        r["doc_no"] = _doc_no(r["fil"])
    return rows


def read_summary(folder):
    path = folder / "sammendrag.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    for i, row in enumerate(rows):
        if row and row[0] == "## Overall" and i + 2 < len(rows):
            return dict(zip(rows[i + 1], [float(v) for v in rows[i + 2]]))
    return {}


def read_log_info(folder):
    info = {"folder": "", "truth_csv": ""}
    log = folder / "logg.txt"
    if not log.exists():
        return info
    # Older result dirs carry the Norwegian prefixes; both spellings have to resolve.
    for line in log.read_text(encoding="utf-8").splitlines():
        for prefix, key in (("Folder:", "folder"), ("Mappe:", "folder"),
                            ("Truth CSV:", "truth_csv"), ("Fasit-CSV:", "truth_csv")):
            if line.startswith(prefix) and not info[key]:
                info[key] = line.split(":", 1)[1].strip()
    return info


def find_labels_csv(log_info):
    path = log_info["truth_csv"]
    if not path:
        return None
    if Path(path).exists():
        return Path(path)
    here = Path(__file__).parent
    local = here / Path(path).name
    if local.exists():
        return local
    m = re.search(r"uttrekk[_-]?(\d+)", path + " " + log_info["folder"])
    if m:
        candidates = sorted(here.glob(f"*uttrekk_labels_{m.group(1)}_*.csv"))
        if candidates:
            return candidates[-1]
    return None


def read_labels(path):
    info = {}
    rows = []
    for r in iter_label_rows(path, exclude_status=(), info=info):
        try:
            doc_no = int(r["fil_revisjon_id"])
        except (TypeError, ValueError, KeyError):
            continue
        rows.append({
            "doc_no": doc_no,
            "type": (r.get("type") or "").strip(),
            "ml": (r.get("ml_generated") or "").strip().lower() == "true",
            "status": (r.get("ml_status") or "").strip().upper(),
        })
    skipped = info["discarded"]["(ugyldig-listet)"]
    if skipped:
        print(f"  {skipped} boxes in ugyldige_labels.txt excluded")
    return rows

def paddle_stats(details, summary):
    s = {}
    s["fasit"] = len(details)
    s["hit"] = sum(1 for r in details if r["result"] == "HIT")
    s["missing"] = s["fasit"] - s["hit"]
    s["recall"] = s["hit"] / s["fasit"] if s["fasit"] else 0.0
    s["files"] = len({r["fil"] for r in details})
    s["sider"] = len({(r["fil"], r["side"]) for r in details})

    s["pred"] = int(summary.get("pred", 0))
    # older result dirs name this field "surplus" or "overflod"
    s["oversladd"] = int(summary.get("oversladd")
                         or summary.get("surplus")
                         or summary.get("overflod") or 0)
    s["oversladding"] = s["oversladd"] / s["pred"] if s["pred"] else 0.0
    s["precision"] = 1.0 - s["oversladding"] if s["pred"] else 0.0
    s["total_overlap"] = summary.get("total_overlap_pct", 0.0) / 100
    s["threshold"] = summary.get("threshold_pct", 0.0) / 100

    coverages = [r["coverage_pct"] for r in details]
    hit_cov = [r["coverage_pct"] for r in details if r["result"] == "HIT"]
    s["dekning_snitt"] = statistics.mean(coverages) if coverages else 0.0
    s["dekning_median"] = statistics.median(coverages) if coverages else 0.0
    s["dekning_snitt_truffet"] = statistics.mean(hit_cov) if hit_cov else 0.0
    s["coverages"] = coverages

    pr_type = defaultdict(lambda: [0, 0])         
    for r in details:
        pr_type[r["type"]][1] += 1
        if r["result"] == "HIT":
            pr_type[r["type"]][0] += 1
    s["pr_type"] = dict(pr_type)

    pr_file = defaultdict(lambda: [0, 0])           
    for r in details:
        pr_file[r["fil"]][1] += 1
        if r["result"] == "MISSING":
            pr_file[r["fil"]][0] += 1
    s["pr_fil"] = dict(pr_file)
    return s


def labels_stats(labels, doc_nos=None):
    s = {"tp": 0, "fp": 0, "fn": 0, "unresolved": 0, "pr_type": defaultdict(lambda: [0, 0, 0])}
    for r in labels:
        if doc_nos is not None and r["doc_no"] not in doc_nos:
            continue
        t = s["pr_type"][r["type"]]
        if r["ml"] and r["status"] == "ACCEPTED":
            s["tp"] += 1
            t[0] += 1
        elif r["ml"] and r["status"] == "REJECTED":
            s["fp"] += 1
            t[1] += 1
        elif not r["ml"]:
            s["fn"] += 1
            t[2] += 1
        else:
            s["unresolved"] += 1
    s["fasit"] = s["tp"] + s["fn"]                       
    s["recall"] = s["tp"] / s["fasit"] if s["fasit"] else 0.0
    s["pred"] = s["tp"] + s["fp"]                        
    s["oversladding"] = s["fp"] / s["pred"] if s["pred"] else 0.0
    s["precision"] = 1.0 - s["oversladding"] if s["pred"] else 0.0
    s["manuell_andel"] = s["fn"] / s["fasit"] if s["fasit"] else 0.0
    s["pr_type"] = dict(s["pr_type"])
    return s


# ---------------------------------------------------------------- report

def pct(x):
    return f"{x * 100:5.1f} %"


def make_report(folder, p, n, labels_path, log_info):
    L = []
    stroke = "=" * 66

    L.append(stroke)
    L.append(f"STATISTICS: {folder.name}")
    if log_info["folder"]:
        L.append(f"Data:      {os.path.basename(log_info['folder'])}  ({log_info['folder']})")
    if labels_path is not None:
        L.append(f"Truth from: {labels_path.name}")
        if str(labels_path) != log_info["truth_csv"]:
            L.append(f"           (the run used {log_info['truth_csv']}, matched locally on uttrekk number)")
    L.append(stroke)

    L.append("")
    L.append("--- This run (PaddleOCR) " + "-" * 40)
    L.append(f"Files / pages with truth:     {p['files']} / {p['sider']}")
    L.append(f"Truth boxes:                  {p['fasit']}")
    L.append(f"Hit / missed:                 {p['hit']} / {p['missing']}")
    L.append(f"Recall:                       {pct(p['recall'])}")
    L.append(f"Sladd boxes drawn:            {p['pred']}")
    L.append(f"OVER-SLADDING:                {pct(p['oversladding'])}   "
             f"({p['oversladd']} of {p['pred']} boxes with no truth hit)")
    L.append(f"Precision:                    {pct(p['precision'])}")
    L.append(f"Total overlap (area):         {pct(p['total_overlap'])}")
    L.append(f"Coverage mean / median:       {p['dekning_snitt']:.1f} % / {p['dekning_median']:.1f} %")
    L.append(f"Coverage mean (hits only):    {p['dekning_snitt_truffet']:.1f} %")
    L.append(f"Hit threshold:                {pct(p['threshold'])}")

    L.append("")
    L.append("Recall per type:")
    for t, (tr, tot) in sorted(p["pr_type"].items()):
        L.append(f"   {t or '(empty)':<22} {tr}/{tot} = {pct(tr / tot if tot else 0)}")

    miss = sorted(((m, tot, file) for file, (m, tot) in p["pr_fil"].items() if m), reverse=True)
    if miss:
        L.append("")
        L.append(f"Files with misses ({len(miss)}, worst first):")
        for m, tot, file in miss[:15]:
            L.append(f"   {file:<28} {m}/{tot} missed")
        if len(miss) > 15:
            L.append(f"   ... and {len(miss) - 15} more")
    else:
        L.append("")
        L.append("No files with misses. Every truth box was hit.")

    if n is None:
        L.append("")
        L.append("!! No labels CSV found. Comparison with the current solution skipped.")
        L.append("   Point at it with --labels <file>.")
        return "\n".join(L) + "\n"

    L.append("")
    L.append("--- Current solution (same documents) " + "-" * 27)
    L.append(f"Found by itself (ml+ACCEPTED):{n['tp']}")
    L.append(f"Case worker added (FN):       {n['fn']}")
    L.append(f"Case worker rejected (FP):    {n['fp']}")
    L.append(f"Recall:                       {pct(n['recall'])}")
    L.append(f"OVER-SLADDING:                {pct(n['oversladding'])}   "
             f"({n['fp']} of {n['pred']} boxes rejected)")
    L.append(f"Case worker share:            {pct(n['manuell_andel'])}   "
             f"({n['fn']} of {n['fasit']} boxes added by hand)")
    if n["fasit"] != p["fasit"]:
        L.append(f"NB: truth in the labels CSV ({n['fasit']}) != truth in the run "
                 f"({p['fasit']}), the numbers are not 1:1 comparable.")
    L.append("")
    L.append("Current solution per type (same documents):")
    for t, (tp, fp, fn) in sorted(n["pr_type"].items()):
        rec = tp / (tp + fn) if tp + fn else 0.0
        L.append(f"   {t or '(empty)':<22} TP {tp:>4}  FP {fp:>4}  FN {fn:>4}  recall {pct(rec)}")

    L.append("")
    L.append("--- Comparison " + "-" * 50)
    diff_pp = (p["recall"] - n["recall"]) * 100
    L.append(f"Recall  paddle vs current:    {pct(p['recall'])} vs {pct(n['recall'])}"
             f"   ({diff_pp:+.1f} percentage points)")
    if n["tp"]:
        more = (p["hit"] - n["tp"]) / n["tp"] * 100
        L.append(f"Hits    paddle vs current:    {p['hit']} vs {n['tp']}"
                 f"   ({more:+.1f} % {'more' if more >= 0 else 'fewer'} hits)")
    L.append(f"Over-sladding paddle vs cur.: {pct(p['oversladding'])} vs {pct(n['oversladding'])}")

    L.append(stroke)
    return "\n".join(L) + "\n"


def make_graphs(folder, p, n, log_info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uttrekk = os.path.basename(log_info["folder"]) or folder.name
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Smartsladding, {uttrekk}  ({p['files']} files, {p['fasit']} truth boxes)"
                 f"\n{folder.name}", fontsize=13)
    green, gray, roed, blue = "#2e7d32", "#9e9e9e", "#c62828", "#1565c0"

    ax = axes[0][0]
    if n is not None:
        name = ["PaddleOCR", "Current solution"]
        values = [p["recall"], n["recall"]]
        colors = [green, gray]
    else:
        name, values, colors = ["PaddleOCR"], [p["recall"]], [green]
    bars = ax.bar(name, [v * 100 for v in values], color=colors)
    ax.bar_label(bars, fmt="%.1f %%")
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title("Recall (share of truth boxes hit)")

    ax = axes[0][1]
    if n is not None:
        name = ["PaddleOCR", "Current solution"]
        values = [p["oversladding"], n["oversladding"]]
    else:
        name, values = ["PaddleOCR"], [p["oversladding"]]
    bars = ax.bar(name, [v * 100 for v in values], color=roed)
    ax.bar_label(bars, fmt="%.1f %%")
    ax.set_ylim(0, max(10, max(v * 100 for v in values) * 1.3))
    ax.set_ylabel("%")
    ax.set_title("Over-sladding (share of boxes with no truth hit)")

    ax = axes[1][0]
    if n is not None and n["fasit"]:
        ax.pie([n["tp"], n["fn"]],
               labels=[f"Model found itself\n{n['tp']}", f"Case worker added\n{n['fn']}"],
               colors=[blue, roed], autopct="%.1f %%", startangle=90)
        ax.set_title("Truth boxes today: model vs case worker")
    else:
        ax.axis("off")
        ax.set_title("No labels CSV")

    ax = axes[1][1]
    ax.hist(p["coverages"], bins=20, range=(0, 100), color=blue, edgecolor="white")
    ax.axvline(p["threshold"] * 100, color=roed, linestyle="--",
               label=f"threshold {p['threshold'] * 100:.0f} %")
    ax.set_xlabel("coverage of the truth box (%)")
    ax.set_ylabel("truth boxes")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("Coverage distribution (this run)")
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    ut = folder / "statistikk.png"
    fig.savefig(ut, dpi=150)
    plt.close(fig)
    return ut


def _write_labels_report(n, labels_path):
    stroke = "=" * 66
    pct_fn = lambda x: f"{x * 100:5.1f} %"
    L = [stroke, f"CURRENT SOLUTION: {labels_path.name}", stroke, ""]
    L.append(f"Truth (TP + FN):               {n['fasit']}")
    L.append(f"Found by itself (ml+ACCEPTED): {n['tp']}")
    L.append(f"Case worker added (FN):        {n['fn']}")
    L.append(f"Case worker rejected (FP):     {n['fp']}")
    L.append(f"Recall:                        {pct_fn(n['recall'])}")
    L.append(f"Over-sladding:                 {pct_fn(n['oversladding'])}   ({n['fp']} of {n['pred']} boxes rejected)")
    L.append(f"Case worker share:             {pct_fn(n['manuell_andel'])}")
    L.append("")
    L.append("Per type:")
    for t, (tp, fp, fn) in sorted(n["pr_type"].items()):
        rec = tp / (tp + fn) if tp + fn else 0.0
        L.append(f"   {t or '(empty)':<22} TP {tp:>4}  FP {fp:>4}  FN {fn:>4}  recall {pct_fn(rec)}")
    L.append(stroke)
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Combined statistics for a result directory from run.py.")
    ap.add_argument("folder", nargs="?", default=None, help="result directory (e.g. result-2026-07-06T12-35-21)")
    ap.add_argument("--labels", default=None,
                    help="labels CSV (default: taken from logg.txt in the result directory)")
    ap.add_argument("--no-graph", action="store_true", help="skip statistikk.png")
    args = ap.parse_args()

    # --- labels-only mode ---
    if args.folder is None:
        if not args.labels:
            ap.error("Give either a result directory or --labels <file>")
        labels_path = Path(args.labels)
        if not labels_path.exists():
            sys.exit(f"No such labels CSV: {args.labels}")
        labels = read_labels(labels_path)
        n = labels_stats(labels)
        print(_write_labels_report(n, labels_path), end="")
        return

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"No such directory: {folder}")

    details = read_details(folder)
    if not details:
        sys.exit("detaljer.csv is empty, nothing to compute.")
    summary = read_summary(folder)
    p = paddle_stats(details, summary)

    log_info = read_log_info(folder)
    labels_path = Path(args.labels) if args.labels else find_labels_csv(log_info)
    n = None
    if labels_path and labels_path.exists():
        labels = read_labels(labels_path)
        doc_nos = {r["doc_no"] for r in details}
        n = labels_stats(labels, doc_nos)
    elif args.labels:
        sys.exit(f"No such labels CSV: {args.labels}")

    report = make_report(folder, p, n, labels_path, log_info)
    print(report, end="")

    (folder / "statistikk.txt").write_text(report, encoding="utf-8")
    print(f"Report:   {folder / 'statistikk.txt'}")

    if not args.no_graph:
        try:
            ut = make_graphs(folder, p, n, log_info)
            print(f"Charts:   {ut}")
        except ImportError:
            print("matplotlib missing, charts skipped (pip install matplotlib).")


if __name__ == "__main__":
    main()
