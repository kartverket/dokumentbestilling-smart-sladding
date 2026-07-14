import argparse
import csv
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path



def _dok_nr(navn):
    m = re.match(r"0*(\d+)", os.path.basename(navn))
    return int(m.group(1)) if m else None


def les_detaljer(mappe):
    sti = mappe / "detaljer.csv"
    if not sti.exists():
        sys.exit(f"Fant ikke {sti} — er dette en result-mappe fra run.py?")
    with open(sti, newline="", encoding="utf-8-sig") as f:
        rader = list(csv.DictReader(f))
    for r in rader:
        r["side"] = int(r["side"])
        r["dekning_pst"] = float(r["dekning_pst"])
        r["dok_nr"] = _dok_nr(r["fil"])
    return rader


def les_sammendrag(mappe):
    sti = mappe / "sammendrag.csv"
    if not sti.exists():
        return {}
    with open(sti, newline="", encoding="utf-8-sig") as f:
        rader = list(csv.reader(f))
    for i, rad in enumerate(rader):
        if rad and rad[0] == "## Overordnet" and i + 2 < len(rader):
            return dict(zip(rader[i + 1], [float(v) for v in rader[i + 2]]))
    return {}


def les_logg_info(mappe):
    info = {"mappe": "", "fasit_csv": ""}
    logg = mappe / "logg.txt"
    if not logg.exists():
        return info
    for linje in logg.read_text(encoding="utf-8").splitlines():
        if linje.startswith("Mappe:"):
            info["mappe"] = linje.split(":", 1)[1].strip()
        elif linje.startswith("Fasit-CSV:"):
            info["fasit_csv"] = linje.split(":", 1)[1].strip()
    return info


def finn_labels_csv(logg_info):
    sti = logg_info["fasit_csv"]
    if not sti:
        return None
    if Path(sti).exists():
        return Path(sti)
    her = Path(__file__).parent
    lokal = her / Path(sti).name
    if lokal.exists():
        return lokal
    m = re.search(r"uttrekk[_-]?(\d+)", sti + " " + logg_info["mappe"])
    if m:
        kandidater = sorted(her.glob(f"*uttrekk_labels_{m.group(1)}_*.csv"))
        if kandidater:
            return kandidater[-1]
    return None


def les_labels(sti):
    rader = []
    with open(sti, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                dok_nr = int(r["fil_revisjon_id"])
            except (TypeError, ValueError, KeyError):
                continue
            rader.append({
                "dok_nr": dok_nr,
                "type": (r.get("type") or "").strip(),
                "ml": (r.get("ml_generated") or "").strip().lower() == "true",
                "status": (r.get("ml_status") or "").strip().upper(),
            })
    return rader

def paddle_stats(detaljer, sammendrag):
    s = {}
    s["fasit"] = len(detaljer)
    s["truffet"] = sum(1 for r in detaljer if r["resultat"] == "TRUFFET")
    s["mangler"] = s["fasit"] - s["truffet"]
    s["recall"] = s["truffet"] / s["fasit"] if s["fasit"] else 0.0
    s["filer"] = len({r["fil"] for r in detaljer})
    s["sider"] = len({(r["fil"], r["side"]) for r in detaljer})

    s["pred"] = int(sammendrag.get("pred", 0))
    s["overflod"] = int(sammendrag.get("overflod", 0))
    s["oversladding"] = s["overflod"] / s["pred"] if s["pred"] else 0.0
    s["presisjon"] = 1.0 - s["oversladding"] if s["pred"] else 0.0
    s["samlet_overlapp"] = sammendrag.get("samlet_overlapp_pst", 0.0) / 100
    s["terskel"] = sammendrag.get("terskel_pst", 0.0) / 100

    dekninger = [r["dekning_pst"] for r in detaljer]
    truffet_dek = [r["dekning_pst"] for r in detaljer if r["resultat"] == "TRUFFET"]
    s["dekning_snitt"] = statistics.mean(dekninger) if dekninger else 0.0
    s["dekning_median"] = statistics.median(dekninger) if dekninger else 0.0
    s["dekning_snitt_truffet"] = statistics.mean(truffet_dek) if truffet_dek else 0.0
    s["dekninger"] = dekninger

    pr_type = defaultdict(lambda: [0, 0])         
    for r in detaljer:
        pr_type[r["type"]][1] += 1
        if r["resultat"] == "TRUFFET":
            pr_type[r["type"]][0] += 1
    s["pr_type"] = dict(pr_type)

    pr_fil = defaultdict(lambda: [0, 0])           
    for r in detaljer:
        pr_fil[r["fil"]][1] += 1
        if r["resultat"] == "MANGLER":
            pr_fil[r["fil"]][0] += 1
    s["pr_fil"] = dict(pr_fil)
    return s


def labels_stats(labels, dok_nrs=None):
    s = {"tp": 0, "fp": 0, "fn": 0, "uavklart": 0, "pr_type": defaultdict(lambda: [0, 0, 0])}
    for r in labels:
        if dok_nrs is not None and r["dok_nr"] not in dok_nrs:
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
            s["uavklart"] += 1
    s["fasit"] = s["tp"] + s["fn"]                       
    s["recall"] = s["tp"] / s["fasit"] if s["fasit"] else 0.0
    s["pred"] = s["tp"] + s["fp"]                        
    s["oversladding"] = s["fp"] / s["pred"] if s["pred"] else 0.0
    s["presisjon"] = 1.0 - s["oversladding"] if s["pred"] else 0.0
    s["manuell_andel"] = s["fn"] / s["fasit"] if s["fasit"] else 0.0
    s["pr_type"] = dict(s["pr_type"])
    return s


# ---------------------------------------------------------------- rapport

def pst(x):
    return f"{x * 100:5.1f} %"


def lag_rapport(mappe, p, n, labels_sti, logg_info):
    L = []
    strek = "=" * 66

    L.append(strek)
    L.append(f"STATISTIKK — {mappe.name}")
    if logg_info["mappe"]:
        L.append(f"Data:      {os.path.basename(logg_info['mappe'])}  ({logg_info['mappe']})")
    if labels_sti is not None:
        L.append(f"Fasit fra: {labels_sti.name}")
        if str(labels_sti) != logg_info["fasit_csv"]:
            L.append(f"           (kjøringen brukte {logg_info['fasit_csv']} — matchet lokalt på uttrekk-nummer)")
    L.append(strek)

    L.append("")
    L.append("--- Denne kjøringen (PaddleOCR) " + "-" * 33)
    L.append(f"Filer / sider med fasit:      {p['filer']} / {p['sider']}")
    L.append(f"Fasit-bokser:                 {p['fasit']}")
    L.append(f"Truffet / mangler:            {p['truffet']} / {p['mangler']}")
    L.append(f"Recall:                       {pst(p['recall'])}")
    L.append(f"Sladde-bokser tegnet:         {p['pred']}")
    L.append(f"OVERSLADDING:                 {pst(p['oversladding'])}   "
             f"({p['overflod']} av {p['pred']} bokser uten fasit-treff)")
    L.append(f"Presisjon:                    {pst(p['presisjon'])}")
    L.append(f"Samlet overlapp (areal):      {pst(p['samlet_overlapp'])}")
    L.append(f"Dekning snitt / median:       {p['dekning_snitt']:.1f} % / {p['dekning_median']:.1f} %")
    L.append(f"Dekning snitt (kun truffet):  {p['dekning_snitt_truffet']:.1f} %")
    L.append(f"Terskel for treff:            {pst(p['terskel'])}")

    L.append("")
    L.append("Recall per type:")
    for t, (tr, tot) in sorted(p["pr_type"].items()):
        L.append(f"   {t or '(tom)':<22} {tr}/{tot} = {pst(tr / tot if tot else 0)}")

    bom = sorted(((m, tot, fil) for fil, (m, tot) in p["pr_fil"].items() if m), reverse=True)
    if bom:
        L.append("")
        L.append(f"Filer med bom ({len(bom)} stk, verste først):")
        for m, tot, fil in bom[:15]:
            L.append(f"   {fil:<28} {m}/{tot} bommet")
        if len(bom) > 15:
            L.append(f"   ... og {len(bom) - 15} til")
    else:
        L.append("")
        L.append("Ingen filer med bom — alle fasit-bokser truffet.")

    if n is None:
        L.append("")
        L.append("!! Fant ingen labels-CSV — sammenligning med nåværende løsning droppet.")
        L.append("   Angi den med --labels <fil>.")
        return "\n".join(L) + "\n"

    L.append("")
    L.append("--- Nåværende løsning (samme dokumenter) " + "-" * 24)
    L.append(f"Fant selv (ml + ACCEPTED):    {n['tp']}")
    L.append(f"Saksbehandler la til (FN):    {n['fn']}")
    L.append(f"Avvist av saksbehandler (FP): {n['fp']}")
    L.append(f"Recall:                       {pst(n['recall'])}")
    L.append(f"OVERSLADDING:                 {pst(n['oversladding'])}   "
             f"({n['fp']} av {n['pred']} bokser avvist)")
    L.append(f"Saksbehandler-andel:          {pst(n['manuell_andel'])}   "
             f"({n['fn']} av {n['fasit']} bokser lagt til manuelt)")
    if n["fasit"] != p["fasit"]:
        L.append(f"NB: fasit i labels-CSV ({n['fasit']}) != fasit i kjøringen "
                 f"({p['fasit']}) — tallene er ikke 1:1 sammenlignbare.")
    L.append("")
    L.append("Nåværende løsning per type (samme dokumenter):")
    for t, (tp, fp, fn) in sorted(n["pr_type"].items()):
        rec = tp / (tp + fn) if tp + fn else 0.0
        L.append(f"   {t or '(tom)':<22} TP {tp:>4}  FP {fp:>4}  FN {fn:>4}  recall {pst(rec)}")

    L.append("")
    L.append("--- Sammenligning " + "-" * 47)
    diff_pp = (p["recall"] - n["recall"]) * 100
    L.append(f"Recall  paddle vs nåværende:  {pst(p['recall'])} vs {pst(n['recall'])}"
             f"   ({diff_pp:+.1f} prosentpoeng)")
    if n["tp"]:
        flere = (p["truffet"] - n["tp"]) / n["tp"] * 100
        L.append(f"Treff   paddle vs nåværende:  {p['truffet']} vs {n['tp']}"
                 f"   ({flere:+.1f} % {'flere' if flere >= 0 else 'færre'} treff)")
    L.append(f"Oversladding paddle vs nåv.:  {pst(p['oversladding'])} vs {pst(n['oversladding'])}")

    L.append(strek)
    return "\n".join(L) + "\n"


def lag_grafer(mappe, p, n, logg_info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uttrekk = os.path.basename(logg_info["mappe"]) or mappe.name
    fig, aksene = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Smartsladding — {uttrekk}  ({p['filer']} filer, {p['fasit']} fasit-bokser)"
                 f"\n{mappe.name}", fontsize=13)
    gronn, graa, roed, blaa = "#2e7d32", "#9e9e9e", "#c62828", "#1565c0"

    ax = aksene[0][0]
    if n is not None:
        navn = ["PaddleOCR", "Nåværende løsning"]
        verdier = [p["recall"], n["recall"]]
        farger = [gronn, graa]
    else:
        navn, verdier, farger = ["PaddleOCR"], [p["recall"]], [gronn]
    stolper = ax.bar(navn, [v * 100 for v in verdier], color=farger)
    ax.bar_label(stolper, fmt="%.1f %%")
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title("Recall (andel fasit-bokser truffet)")

    ax = aksene[0][1]
    if n is not None:
        navn = ["PaddleOCR", "Nåværende løsning"]
        verdier = [p["oversladding"], n["oversladding"]]
    else:
        navn, verdier = ["PaddleOCR"], [p["oversladding"]]
    stolper = ax.bar(navn, [v * 100 for v in verdier], color=roed)
    ax.bar_label(stolper, fmt="%.1f %%")
    ax.set_ylim(0, max(10, max(v * 100 for v in verdier) * 1.3))
    ax.set_ylabel("%")
    ax.set_title("Oversladding (andel bokser uten fasit-treff)")

    ax = aksene[1][0]
    if n is not None and n["fasit"]:
        ax.pie([n["tp"], n["fn"]],
               labels=[f"Modell fant selv\n{n['tp']}", f"Saksbehandler la til\n{n['fn']}"],
               colors=[blaa, roed], autopct="%.1f %%", startangle=90)
        ax.set_title("Fasit-bokser i dag: modell vs saksbehandler")
    else:
        ax.axis("off")
        ax.set_title("Ingen labels-CSV")

    ax = aksene[1][1]
    ax.hist(p["dekninger"], bins=20, range=(0, 100), color=blaa, edgecolor="white")
    ax.axvline(p["terskel"] * 100, color=roed, linestyle="--",
               label=f"terskel {p['terskel'] * 100:.0f} %")
    ax.set_xlabel("dekning av fasit-boks (%)")
    ax.set_ylabel("antall fasit-bokser")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("Dekningsfordeling (denne kjøringen)")
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    ut = mappe / "statistikk.png"
    fig.savefig(ut, dpi=150)
    plt.close(fig)
    return ut


def _skriv_labels_rapport(n, labels_sti):
    strek = "=" * 66
    pst_fn = lambda x: f"{x * 100:5.1f} %"
    L = [strek, f"NÅVÆRENDE LØSNING — {labels_sti.name}", strek, ""]
    L.append(f"Fasit (TP + FN):               {n['fasit']}")
    L.append(f"Fant selv (ml + ACCEPTED):     {n['tp']}")
    L.append(f"Saksbehandler la til (FN):     {n['fn']}")
    L.append(f"Avvist av saksbehandler (FP):  {n['fp']}")
    L.append(f"Recall:                        {pst_fn(n['recall'])}")
    L.append(f"Oversladding:                  {pst_fn(n['oversladding'])}   ({n['fp']} av {n['pred']} bokser avvist)")
    L.append(f"Saksbehandler-andel:           {pst_fn(n['manuell_andel'])}")
    L.append("")
    L.append("Per type:")
    for t, (tp, fp, fn) in sorted(n["pr_type"].items()):
        rec = tp / (tp + fn) if tp + fn else 0.0
        L.append(f"   {t or '(tom)':<22} TP {tp:>4}  FP {fp:>4}  FN {fn:>4}  recall {pst_fn(rec)}")
    L.append(strek)
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Samlet statistikk for en result-mappe fra run.py.")
    ap.add_argument("mappe", nargs="?", default=None, help="result-mappe (f.eks. result-2026-07-06T12-35-21)")
    ap.add_argument("--labels", default=None,
                    help="labels-CSV (default: hentes fra logg.txt i resultatmappa)")
    ap.add_argument("--ingen-graf", action="store_true", help="dropp statistikk.png")
    args = ap.parse_args()

    # --- Bare labels-modus ---
    if args.mappe is None:
        if not args.labels:
            ap.error("Angi enten mappe eller --labels <fil>")
        labels_sti = Path(args.labels)
        if not labels_sti.exists():
            sys.exit(f"Fant ikke labels-CSV: {args.labels}")
        labels = les_labels(labels_sti)
        n = labels_stats(labels)
        print(_skriv_labels_rapport(n, labels_sti), end="")
        return

    mappe = Path(args.mappe)
    if not mappe.is_dir():
        sys.exit(f"Fant ikke mappa {mappe}")

    detaljer = les_detaljer(mappe)
    if not detaljer:
        sys.exit("detaljer.csv er tom — ingenting å regne på.")
    sammendrag = les_sammendrag(mappe)
    p = paddle_stats(detaljer, sammendrag)

    logg_info = les_logg_info(mappe)
    labels_sti = Path(args.labels) if args.labels else finn_labels_csv(logg_info)
    n = None
    if labels_sti and labels_sti.exists():
        labels = les_labels(labels_sti)
        dok_nrs = {r["dok_nr"] for r in detaljer}
        n = labels_stats(labels, dok_nrs)
    elif args.labels:
        sys.exit(f"Fant ikke labels-CSV: {args.labels}")

    rapport = lag_rapport(mappe, p, n, labels_sti, logg_info)
    print(rapport, end="")

    (mappe / "statistikk.txt").write_text(rapport, encoding="utf-8")
    print(f"Rapport:  {mappe / 'statistikk.txt'}")

    if not args.ingen_graf:
        try:
            ut = lag_grafer(mappe, p, n, logg_info)
            print(f"Grafer:   {ut}")
        except ImportError:
            print("matplotlib mangler — hoppet over grafene (pip install matplotlib).")


if __name__ == "__main__":
    main()
