import csv

FELT = ["navn", "side", "bilde_bredde", "bilde_hoyde", "x0", "y0", "x1", "y1"]


def skriv_csv(sladd_bokser, sti):
    n = 0
    with open(sti, "w", newline="", encoding="utf-8") as f:
        skriv = csv.writer(f)
        skriv.writerow(FELT)
        for (navn, si) in sorted(sladd_bokser):
            bw, bh, bokser = sladd_bokser[(navn, si)]
            for (x0, y0, x1, y1) in bokser:
                skriv.writerow([navn, si, bw, bh, x0, y0, x1, y1])
                n += 1
    return n


def les_csv(sti):
    sladd_bokser = {}
    with open(sti, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            navn, si = r["navn"], int(r["side"])
            bw, bh = int(r["bilde_bredde"]), int(r["bilde_hoyde"])
            boks = (float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"]))
            sladd_bokser.setdefault((navn, si), (bw, bh, []))[2].append(boks)
    return sladd_bokser
