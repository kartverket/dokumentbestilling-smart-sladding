import csv

FELT = ["navn", "side", "bilde_bredde", "bilde_hoyde", "x0", "y0", "x1", "y1", "kilde", "conf"]


def initialiser_csv(sti):
    with open(sti, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(FELT)


def append_csv(grupper, sti):
    n = 0
    with open(sti, "a", newline="", encoding="utf-8") as f:
        skriv = csv.writer(f)
        for (navn, si) in sorted(grupper):
            bw, bh, bokser = grupper[(navn, si)]
            for boks in bokser:
                x0, y0, x1, y1 = boks[:4]
                kilde = boks[4] if len(boks) > 4 else "paddle"
                conf  = boks[5] if len(boks) > 5 else ""
                skriv.writerow([navn, si, bw, bh, x0, y0, x1, y1, kilde, conf])
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