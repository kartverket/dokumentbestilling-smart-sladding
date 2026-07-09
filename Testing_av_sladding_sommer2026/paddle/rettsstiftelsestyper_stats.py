import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt

DETALJER = "detaljer.csv"
LABELS   = "labels.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--type-per-aar", metavar="TYPE",
                    help="f.eks. HJ_HJG")
args = parser.parse_args()

# Hjelpefunksjon for å hente rettsstiftelsestyper fra en streng
def _hent_typer(rettsstiftelsestyper_str):
    typer = []
    for del_ in rettsstiftelsestyper_str.split(","):
        del_ = del_.strip()
        if del_:
            typer.append(del_.split(" - ")[0].strip())
    return typer


id_til_typer = {}
id_til_aar   = {}
with open(LABELS, encoding="utf-8") as f:
    for rad in csv.DictReader(f):
        fid = rad["fil_revisjon_id"].strip()
        id_til_typer[fid] = _hent_typer(rad["rettsstiftelsestyper"])
        id_til_aar[fid]   = rad["dokument_aar"].strip()


totalt   = defaultdict(int)
truffet  = defaultdict(int)

type_per_aar_totalt  = defaultdict(int)
type_per_aar_truffet = defaultdict(int)

with open(DETALJER, encoding="utf-8") as f:
    for rad in csv.DictReader(f):
        fid = rad["fil"].replace(".pdf", "").strip()
        resultat = rad["resultat"].strip()
        typer = id_til_typer.get(fid, ["UKJENT"])
        aar   = id_til_aar.get(fid, "ukjent")

        for t in typer:
            totalt[t] += 1
            if resultat == "TRUFFET":
                truffet[t] += 1
            if args.type_per_aar and t == args.type_per_aar:
                type_per_aar_totalt[aar] += 1
                if resultat == "TRUFFET":
                    type_per_aar_truffet[aar] += 1


# Utskrift og plotting av statistikk for rettsstiftelsestyper
print(f"{'Type':<40} {'Totalt':>8} {'Truffet':>8} {'Treff-%':>8}")
print("-" * 68)
for t in sorted(totalt, key=lambda t: totalt[t], reverse=True):
    tot = totalt[t]
    tr  = truffet[t]
    pst = 100 * tr / tot if tot else 0
    print(f"{t:<40} {tot:>8} {tr:>8} {pst:>7.1f}%")

print("-" * 68)
total_bokser  = sum(totalt.values())
total_truffet = sum(truffet.values())
total_pst     = 100 * total_truffet / total_bokser if total_bokser else 0
print(f"{'TOTALT':<40} {total_bokser:>8} {total_truffet:>8} {total_pst:>7.1f}%")

# Bar plot med alle rettsstiftelsestyper
typer_sortert = sorted(totalt)
prosenter = [100 * truffet[t] / totalt[t] if totalt[t] else 0 for t in typer_sortert]
farger = ["steelblue" if p == 100 else "salmon" if p == 0 else "cornflowerblue" for p in prosenter]

fig, ax = plt.subplots(figsize=(max(10, len(typer_sortert) * 0.4), 6))
bars = ax.bar(typer_sortert, prosenter, color=farger)
ax.axhline(total_pst, color="red", linestyle="--", linewidth=1, label=f"Snitt {total_pst:.1f}%")
ax.set_ylim(0, 110)
ax.set_ylabel("Treff-%")
ax.set_title("Recall per rettsstiftelsestype")
ax.set_xticks(range(len(typer_sortert)))
ax.set_xticklabels(typer_sortert, rotation=90, fontsize=8)
ax.legend()

for bar, pst in zip(bars, prosenter):
    if totalt[typer_sortert[bars.index(bar)]] > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pst:.0f}%", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig("stats_barplot.png", dpi=150)
plt.show()
print("Barplot lagret som stats_barplot.png")

# Type per år plot
if args.type_per_aar:
    valgt = args.type_per_aar
    if not type_per_aar_totalt:
        print(f"Ingen treff for type '{valgt}' i resultatsettet.")
    else:
        aar_sortert = sorted(type_per_aar_totalt)
        pst_per_aar = [100 * type_per_aar_truffet[a] / type_per_aar_totalt[a]
                       if type_per_aar_totalt[a] else 0 for a in aar_sortert]

        fig2, ax2 = plt.subplots(figsize=(max(10, len(aar_sortert) * 0.4), 5))
        ax2.bar(aar_sortert, pst_per_aar, color="steelblue")
        ax2.set_ylim(0, 110)
        ax2.set_ylabel("Treff-%")
        ax2.set_title(f"{valgt} — treffrate per år")
        ax2.set_xticks(range(len(aar_sortert)))
        ax2.set_xticklabels(aar_sortert, rotation=90, fontsize=8)

        for x, (pst, a) in enumerate(zip(pst_per_aar, aar_sortert)):
            ax2.text(x, pst + 1, f"{pst:.0f}%\n(n={type_per_aar_totalt[a]})",
                     ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        filnavn = f"stats_{valgt}_per_aar.png"
        plt.savefig(filnavn, dpi=150)
        plt.show()
        print(f"Plot lagret som {filnavn}")