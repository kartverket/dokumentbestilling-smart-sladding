import csv
from collections import defaultdict


def les_fasit(csv_sti):
    fasit = defaultdict(list)
    try:
        with open(csv_sti, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if (r.get("ml_status") or "").strip().upper() == "REJECTED":
                    continue
                try:
                    nr = int(r["fil_revisjon_id"])
                    side = int(r["sidetall"])
                    x, y = float(r["x"]), float(r["y"])
                    w, h = float(r["width"]), float(r["height"])
                except (TypeError, ValueError, KeyError):
                    continue
                fasit[(nr, side)].append((x, y, w, h, (r.get("type") or "").strip()))
    except FileNotFoundError:
        print(f"!! Fant ikke CSV: {csv_sti} — fasit tegnes ikke, og treff måles ikke.")
        return None

    print(f"Fasit: {sum(len(v) for v in fasit.values())} boks(er) i "
          f"{len(fasit)} (dok_nr, side)-grupper fra {csv_sti}.")
    return fasit
