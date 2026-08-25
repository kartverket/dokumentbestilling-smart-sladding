"""Figures and a summary of a labels CSV: boxes per page, per document, per
type, where on the page they sit, and how the current solution scores.

Run:
    python utils/stat_utrekk.py --csv smartsladding_uttrekk_labels_2_29_06_26.csv \
        --out-dir stats_uttrekk2
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

import matplotlib
import matplotlib.pyplot as plt

COLOR = "#7396bf"              
EDGE = "#2b2b2b"


def read_labels(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "dok": int(r["fil_revisjon_id"]),
                    "side": int(r["sidetall"]),
                    "type": (r.get("type") or "").strip() or "(empty)",
                    "w": float(r["width"]), "h": float(r["height"]),
                    "x": float(r["x"]), "y": float(r["y"]),
                    "ml": (r.get("ml_generated") or "").strip().lower() == "true",
                    "status": (r.get("ml_status") or "").strip().upper(),
                })
            except (TypeError, ValueError, KeyError):
                continue
    return rows


def _style(ax, title, xlab, ylab):
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)


def fig_per_sidetall(rows, path):
    counter = Counter(r["side"] for r in rows)
    max_items = max(counter)
    pages = list(range(1, max_items + 1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(pages, [counter.get(s, 0) for s in pages], color=COLOR, edgecolor=EDGE)
    _style(ax, "Personnumre per page", "Page", "Boxes")
    if max_items > 20:
        ax.set_xlim(0.5, 20.5)     
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_per_document(rows, path):
    per_doc = Counter(r["dok"] for r in rows)
    distribution = Counter(per_doc.values())
    n_max = max(distribution)
    xs = list(range(1, min(n_max, 25) + 1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(xs, [distribution.get(x, 0) for x in xs], color=COLOR, edgecolor=EDGE)
    _style(ax, "fnr boxes per document", "Boxes in the document", "Documents")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_per_type(rows, path):
    counter = Counter(r["type"] for r in rows)
    typer = [t for t, _ in counter.most_common()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(typer[::-1], [counter[t] for t in typer][::-1], color=COLOR, edgecolor=EDGE)
    _style(ax, "Boxes per type", "Boxes", "")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_position(rows, path, page_width=595.0, page_height=842.0):
    xs = [(r["x"] + r["w"] / 2) / page_width for r in rows]
    ys = [(r["y"] + r["h"] / 2) / page_height for r in rows]
    fig, ax = plt.subplots(figsize=(5, 6.5))
    hb = ax.hexbin(xs, ys, gridsize=30, cmap="Blues", extent=(0, 1, 0, 1))
    ax.invert_yaxis()              # (0,0) top left, as on paper
    _style(ax, "Where on the page the boxes sit", "x (share of width)", "y (share of height)")
    fig.colorbar(hb, ax=ax, label="boxes")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_box_size(rows, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    a1.hist([r["w"] for r in rows], bins=40, color=COLOR, edgecolor=EDGE)
    _style(a1, "Box width", "points", "count")
    a2.hist([r["h"] for r in rows], bins=40, color=COLOR, edgecolor=EDGE)
    _style(a2, "Box height", "points", "count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_per_year(year_csv, rows, path):
    doc_to_year = {}
    with open(year_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                doc_to_year[int(r["fil_revisjon_id"])] = int(r["aar"])
            except (TypeError, ValueError, KeyError):
                continue
    counter = Counter(doc_to_year[r["dok"]] for r in rows if r["dok"] in doc_to_year)
    if not counter:
        print("!! --year-csv matched no labels - skipping fig0")
        return
    year = sorted(counter)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(year, [counter[a] for a in year], color=COLOR, edgecolor=EDGE)
    _style(ax, "Personnumre per year", "Year", "Boxes")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_ml_vs_manual(all_rows, path):
    ml_acc  = sum(1 for r in all_rows if r["ml"] and r["status"] == "ACCEPTED")
    ml_rej  = sum(1 for r in all_rows if r["ml"] and r["status"] == "REJECTED")
    manual = sum(1 for r in all_rows if not r["ml"])
    total  = ml_acc + ml_rej + manual

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    categories = ["ML found\n(accepted)", "ML found\n(rejected)", "Case worker\nadded"]
    values    = [ml_acc, ml_rej, manual]
    colors     = ["#5a9e6f", "#c0504d", "#f0a830"]
    ax1.bar(categories, values, color=colors, edgecolor=EDGE)
    for i, v in enumerate(values):
        ax1.text(i, v + total * 0.005, str(v), ha="center", va="bottom", fontsize=9)
    _style(ax1, "ML vs by hand (boxes)", "", "Boxes")

    recall = ml_acc / (ml_acc + manual) if (ml_acc + manual) > 0 else 0
    oversladd = ml_rej / (ml_acc + ml_rej) if (ml_acc + ml_rej) > 0 else 0
    ax2.barh(["Recall", "Over-sladding"], [recall * 100, oversladd * 100],
             color=["#5a9e6f", "#c0504d"], edgecolor=EDGE)
    ax2.set_xlim(0, 100)
    ax2.axvline(100, color="#aaa", linestyle="--", linewidth=0.8)
    for i, v in enumerate([recall * 100, oversladd * 100]):
        ax2.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    _style(ax2, "Recall and over-sladding (current solution)", "Percent", "")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(rows, all_rows, path):
    per_doc = Counter(r["dok"] for r in rows)
    per_page = Counter(r["side"] for r in rows)
    per_type = Counter(r["type"] for r in rows)
    status = Counter((("ml" if r["ml"] else "manual"), r["status"] or "(empty)") for r in rows)
    n = len(rows)
    side1 = per_page.get(1, 0)

    with open(path, "w", encoding="utf-8") as f:
        w = f.write
        w("=== Summary ===\n")
        w(f"Boxes in total:           {n}\n")
        w(f"Documents with boxes:     {len(per_doc)}\n")
        w(f"Mean boxes/document:      {n / len(per_doc):.2f}\n")
        w(f"Median boxes/document:    {sorted(per_doc.values())[len(per_doc) // 2]}\n")
        w(f"Most boxes in one doc:    {max(per_doc.values())} (doc {per_doc.most_common(1)[0][0]})\n")
        w(f"Share of boxes on page 1: {side1 / n:.1%}\n")
        w(f"Share on pages 1-3:       {sum(per_page.get(s, 0) for s in (1, 2, 3)) / n:.1%}\n")
        w(f"Highest page with a box:  {max(per_page)}\n")
        w("\n=== Per type ===\n")
        for t, c in per_type.most_common():
            w(f"  {t:<24} {c:>6}  ({c / n:.1%})\n")
        w("\n=== ml_generated x status ===\n")
        for (ml, st), c in sorted(status.items()):
            w(f"  {ml:<8} {st:<10} {c:>6}\n")

        ml_acc  = sum(1 for r in all_rows if r["ml"] and r["status"] == "ACCEPTED")
        ml_rej  = sum(1 for r in all_rows if r["ml"] and r["status"] == "REJECTED")
        manual = sum(1 for r in all_rows if not r["ml"])
        recall  = ml_acc / (ml_acc + manual) if (ml_acc + manual) > 0 else 0
        oversladd = ml_rej / (ml_acc + ml_rej) if (ml_acc + ml_rej) > 0 else 0
        w("\n=== Current solution (recall estimate) ===\n")
        w(f"  ML found + accepted:     {ml_acc:>6}\n")
        w(f"  ML found + rejected:     {ml_rej:>6}  (over-sladding)\n")
        w(f"  Case worker added:       {manual:>6}  (model miss)\n")
        w(f"  Recall estimate:         {recall:.1%}\n")
        w(f"  Over-sladding rate:      {oversladd:.1%}\n")
        w("\n=== Pages (top 10) ===\n")
        for s, c in per_page.most_common(10):
            w(f"  page {s:<4} {c:>6}\n")


def main():
    p = argparse.ArgumentParser(description="Statistics and figures from a labels CSV.")
    p.add_argument("--csv", default="smartsladding_uttrekk_labels_2_29_06_26.csv")
    p.add_argument("--out-dir", default="stats_uttrekk2")
    p.add_argument("--year-csv", default=None,
                   help="optional CSV with columns fil_revisjon_id,aar for the per-year figure")
    p.add_argument("--with-rejected", action="store_true",
                   help="include REJECTED boxes (default: real boxes only, like read_truth)")
    args = p.parse_args()

    rows_every = read_labels(args.csv)
    rows = rows_every if args.with_rejected else [r for r in rows_every if r["status"] != "REJECTED"]
    if not rows:
        print("No rows read - check --csv.")
        return
    print(f"Read {len(rows)} boxes from {args.csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    fig_per_sidetall(rows, os.path.join(args.out_dir, "fig1_bokser_per_sidetall.png"))
    fig_per_document(rows, os.path.join(args.out_dir, "fig2_bokser_per_dokument.png"))
    fig_per_type(rows, os.path.join(args.out_dir, "fig3_per_type.png"))
    fig_position(rows, os.path.join(args.out_dir, "fig4_posisjon_heatmap.png"))
    fig_box_size(rows, os.path.join(args.out_dir, "fig5_boksstorrelse.png"))
    fig_ml_vs_manual(rows_every, os.path.join(args.out_dir, "fig6_ml_vs_manuell.png"))
    if args.year_csv:
        fig_per_year(args.year_csv, rows, os.path.join(args.out_dir, "fig0_per_aar.png"))
    write_summary(rows, rows_every, os.path.join(args.out_dir, "oppsummering.txt"))

    print(f"Figures and summary written to {args.out_dir}/")


if __name__ == "__main__":
    main()