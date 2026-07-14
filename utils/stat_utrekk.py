import argparse
import csv
import os
from collections import Counter, defaultdict

import matplotlib
import matplotlib.pyplot as plt

FARGE = "#7396bf"              
KANT = "#2b2b2b"


def les_labels(sti):
    rader = []
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                rader.append({
                    "dok": int(r["fil_revisjon_id"]),
                    "side": int(r["sidetall"]),
                    "type": (r.get("type") or "").strip() or "(tom)",
                    "w": float(r["width"]), "h": float(r["height"]),
                    "x": float(r["x"]), "y": float(r["y"]),
                    "ml": (r.get("ml_generated") or "").strip().lower() == "true",
                    "status": (r.get("ml_status") or "").strip().upper(),
                })
            except (TypeError, ValueError, KeyError):
                continue
    return rader


def _stil(ax, tittel, xlab, ylab):
    ax.set_title(tittel)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)


def fig_per_sidetall(rader, sti):
    teller = Counter(r["side"] for r in rader)
    maks = max(teller)
    sider = list(range(1, maks + 1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(sider, [teller.get(s, 0) for s in sider], color=FARGE, edgecolor=KANT)
    _stil(ax, "Histogram av antall personnummer per sidetall", "Sidetall", "Antall bokser")
    if maks > 20:
        ax.set_xlim(0.5, 20.5)     
    fig.tight_layout()
    fig.savefig(sti, dpi=150)
    plt.close(fig)


def fig_per_dokument(rader, sti):
    per_dok = Counter(r["dok"] for r in rader)
    fordeling = Counter(per_dok.values())
    n_maks = max(fordeling)
    xs = list(range(1, min(n_maks, 25) + 1))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(xs, [fordeling.get(x, 0) for x in xs], color=FARGE, edgecolor=KANT)
    _stil(ax, "Antall fnr-bokser per dokument", "Bokser i dokumentet", "Antall dokumenter")
    fig.tight_layout()
    fig.savefig(sti, dpi=150)
    plt.close(fig)


def fig_per_type(rader, sti):
    teller = Counter(r["type"] for r in rader)
    typer = [t for t, _ in teller.most_common()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(typer[::-1], [teller[t] for t in typer][::-1], color=FARGE, edgecolor=KANT)
    _stil(ax, "Antall bokser per type", "Antall bokser", "")
    fig.tight_layout()
    fig.savefig(sti, dpi=150)
    plt.close(fig)


def fig_posisjon(rader, sti, side_bredde=595.0, side_hoyde=842.0):
    xs = [(r["x"] + r["w"] / 2) / side_bredde for r in rader]
    ys = [(r["y"] + r["h"] / 2) / side_hoyde for r in rader]
    fig, ax = plt.subplots(figsize=(5, 6.5))
    hb = ax.hexbin(xs, ys, gridsize=30, cmap="Blues", extent=(0, 1, 0, 1))
    ax.invert_yaxis()              # (0,0) oppe til venstre, som paa papiret
    _stil(ax, "Hvor paa sida ligger boksene", "x (andel av bredde)", "y (andel av hoyde)")
    fig.colorbar(hb, ax=ax, label="antall bokser")
    fig.tight_layout()
    fig.savefig(sti, dpi=150)
    plt.close(fig)


def fig_boksstorrelse(rader, sti):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    a1.hist([r["w"] for r in rader], bins=40, color=FARGE, edgecolor=KANT)
    _stil(a1, "Boksbredde", "punkter", "antall")
    a2.hist([r["h"] for r in rader], bins=40, color=FARGE, edgecolor=KANT)
    _stil(a2, "Bokshoyde", "punkter", "antall")
    fig.tight_layout()
    fig.savefig(sti, dpi=150)
    plt.close(fig)


def fig_per_aar(aar_csv, rader, sti):
    dok_til_aar = {}
    with open(aar_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                dok_til_aar[int(r["fil_revisjon_id"])] = int(r["aar"])
            except (TypeError, ValueError, KeyError):
                continue
    teller = Counter(dok_til_aar[r["dok"]] for r in rader if r["dok"] in dok_til_aar)
    if not teller:
        print("!! --aar-csv ga ingen treff mot labels - hopper over fig0")
        return
    aar = sorted(teller)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(aar, [teller[a] for a in aar], color=FARGE, edgecolor=KANT)
    _stil(ax, "Histogram av antall personnummer per aar", "Aar", "Antall bokser")
    fig.tight_layout()
    fig.savefig(sti, dpi=150)
    plt.close(fig)


def fig_ml_vs_manuell(alle_rader, sti):
    ml_acc  = sum(1 for r in alle_rader if r["ml"] and r["status"] == "ACCEPTED")
    ml_rej  = sum(1 for r in alle_rader if r["ml"] and r["status"] == "REJECTED")
    manuell = sum(1 for r in alle_rader if not r["ml"])
    totalt  = ml_acc + ml_rej + manuell

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Venstre: absolutt
    kategorier = ["ML funnet\n(godkjent)", "ML funnet\n(avvist)", "Saksbehandler\nla til"]
    verdier    = [ml_acc, ml_rej, manuell]
    farger     = ["#5a9e6f", "#c0504d", "#f0a830"]
    ax1.bar(kategorier, verdier, color=farger, edgecolor=KANT)
    for i, v in enumerate(verdier):
        ax1.text(i, v + totalt * 0.005, str(v), ha="center", va="bottom", fontsize=9)
    _stil(ax1, "ML vs manuell (antall bokser)", "", "Antall bokser")

    # Høyre: recall-diagram
    recall = ml_acc / (ml_acc + manuell) if (ml_acc + manuell) > 0 else 0
    oversladd = ml_rej / (ml_acc + ml_rej) if (ml_acc + ml_rej) > 0 else 0
    ax2.barh(["Recall", "Oversladding"], [recall * 100, oversladd * 100],
             color=["#5a9e6f", "#c0504d"], edgecolor=KANT)
    ax2.set_xlim(0, 100)
    ax2.axvline(100, color="#aaa", linestyle="--", linewidth=0.8)
    for i, v in enumerate([recall * 100, oversladd * 100]):
        ax2.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    _stil(ax2, "Recall og oversladding (nåværende løsning)", "Prosent", "")

    fig.tight_layout()
    fig.savefig(sti, dpi=150)
    plt.close(fig)


def skriv_oppsummering(rader, alle_rader, sti):
    per_dok = Counter(r["dok"] for r in rader)
    per_side = Counter(r["side"] for r in rader)
    per_type = Counter(r["type"] for r in rader)
    status = Counter((("ml" if r["ml"] else "manuell"), r["status"] or "(tom)") for r in rader)
    n = len(rader)
    side1 = per_side.get(1, 0)

    with open(sti, "w", encoding="utf-8") as f:
        w = f.write
        w("=== Oppsummering ===\n")
        w(f"Bokser totalt:            {n}\n")
        w(f"Dokumenter med bokser:    {len(per_dok)}\n")
        w(f"Snitt bokser/dokument:    {n / len(per_dok):.2f}\n")
        w(f"Median bokser/dokument:   {sorted(per_dok.values())[len(per_dok) // 2]}\n")
        w(f"Maks bokser i ett dok:    {max(per_dok.values())} (dok {per_dok.most_common(1)[0][0]})\n")
        w(f"Andel bokser paa side 1:  {side1 / n:.1%}\n")
        w(f"Andel paa side 1-3:       {sum(per_side.get(s, 0) for s in (1, 2, 3)) / n:.1%}\n")
        w(f"Hoyeste sidetall m/boks:  {max(per_side)}\n")
        w("\n=== Per type ===\n")
        for t, c in per_type.most_common():
            w(f"  {t:<24} {c:>6}  ({c / n:.1%})\n")
        w("\n=== ml_generated x status ===\n")
        for (ml, st), c in sorted(status.items()):
            w(f"  {ml:<8} {st:<10} {c:>6}\n")

        ml_acc  = sum(1 for r in alle_rader if r["ml"] and r["status"] == "ACCEPTED")
        ml_rej  = sum(1 for r in alle_rader if r["ml"] and r["status"] == "REJECTED")
        manuell = sum(1 for r in alle_rader if not r["ml"])
        recall  = ml_acc / (ml_acc + manuell) if (ml_acc + manuell) > 0 else 0
        oversladd = ml_rej / (ml_acc + ml_rej) if (ml_acc + ml_rej) > 0 else 0
        w("\n=== Nåværende løsning (recall-estimat) ===\n")
        w(f"  ML funnet + godkjent:    {ml_acc:>6}\n")
        w(f"  ML funnet + avvist:      {ml_rej:>6}  (oversladding)\n")
        w(f"  Saksbehandler la til:    {manuell:>6}  (model bom)\n")
        w(f"  Recall-estimat:          {recall:.1%}\n")
        w(f"  Oversladding-rate:       {oversladd:.1%}\n")
        w("\n=== Sidetall (topp 10) ===\n")
        for s, c in per_side.most_common(10):
            w(f"  side {s:<4} {c:>6}\n")


def main():
    p = argparse.ArgumentParser(description="Statistikk og figurer fra labels-CSV.")
    p.add_argument("--csv", default="smartsladding_uttrekk_labels_2_29_06_26.csv")
    p.add_argument("--ut-mappe", default="stats_uttrekk2")
    p.add_argument("--aar-csv", default=None,
                   help="valgfri CSV med kolonner fil_revisjon_id,aar for per-aar-figur")
    p.add_argument("--med-rejected", action="store_true",
                   help="ta med REJECTED-bokser (default: kun ekte bokser, som les_fasit)")
    args = p.parse_args()

    rader_alle = les_labels(args.csv)
    rader = rader_alle if args.med_rejected else [r for r in rader_alle if r["status"] != "REJECTED"]
    if not rader:
        print("Ingen rader lest - sjekk --csv.")
        return
    print(f"Leste {len(rader)} bokser fra {args.csv}")

    os.makedirs(args.ut_mappe, exist_ok=True)
    fig_per_sidetall(rader, os.path.join(args.ut_mappe, "fig1_bokser_per_sidetall.png"))
    fig_per_dokument(rader, os.path.join(args.ut_mappe, "fig2_bokser_per_dokument.png"))
    fig_per_type(rader, os.path.join(args.ut_mappe, "fig3_per_type.png"))
    fig_posisjon(rader, os.path.join(args.ut_mappe, "fig4_posisjon_heatmap.png"))
    fig_boksstorrelse(rader, os.path.join(args.ut_mappe, "fig5_boksstorrelse.png"))
    fig_ml_vs_manuell(rader_alle, os.path.join(args.ut_mappe, "fig6_ml_vs_manuell.png"))
    if args.aar_csv:
        fig_per_aar(args.aar_csv, rader, os.path.join(args.ut_mappe, "fig0_per_aar.png"))
    skriv_oppsummering(rader, rader_alle, os.path.join(args.ut_mappe, "oppsummering.txt"))

    print(f"Figurer og oppsummering skrevet til {args.ut_mappe}/")


if __name__ == "__main__":
    main()