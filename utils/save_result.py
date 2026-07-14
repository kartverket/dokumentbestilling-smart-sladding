import csv
from datetime import datetime
from pathlib import Path


def lagre_resultat(resultat, mappe=".", beskrivelse=None, logg=None):
    if resultat is None:
        return None

    tidsstempel = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    mappenavn = f"result-{tidsstempel}"
    if beskrivelse:
        mappenavn += f"-{beskrivelse}"

    kjoring_mappe = Path(mappe) / mappenavn
    kjoring_mappe.mkdir(parents=True, exist_ok=True)

    sammendrag_fil = kjoring_mappe / "sammendrag.csv"
    detaljer_fil = kjoring_mappe / "detaljer.csv"

    # --- sammendrag ---
    with open(sammendrag_fil, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        w.writerow(["## Overordnet"])
        w.writerow(["recall_pst", "truffet", "fasit", "pred",
                    "overflod", "samlet_overlapp_pst", "terskel_pst"])
        w.writerow([
            round(resultat["recall"] * 100, 1),
            resultat["truffet"],
            resultat["fasit"],
            resultat["pred"],
            resultat["overflod"],
            round(resultat.get("samlet_overlapp", 0.0) * 100, 1),
            round(resultat.get("terskel", 0.0) * 100, 1),
        ])

        w.writerow([])
        w.writerow(["## Recall per type"])
        w.writerow(["type", "truffet", "fasit_totalt", "recall_pst"])
        for t, (tr, tot) in sorted(resultat["pr_type"].items()):
            w.writerow([t or "(tom)", tr, tot, round(tr / tot * 100, 1) if tot else 0])

        bom = resultat.get("bom_filer", [])
        if bom:
            w.writerow([])
            w.writerow(["## Filer med bom"])
            w.writerow(["fil", "side", "bom", "fasit_totalt"])
            for rad in bom:
                w.writerow([rad["fil"], rad["side"], rad["bom"], rad["fasit_totalt"]])

    # --- detaljer per fasit-boks ---
    with open(detaljer_fil, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "fil", "side", "fasit_nr", "type",
            "dekning_pst", "resultat", "kilde", "conf",
            "fasit_x0", "fasit_y0", "fasit_x1", "fasit_y1",
        ])
        w.writeheader()
        w.writerows(resultat.get("detaljer", []))

    if logg is not None:
        logg_fil = kjoring_mappe / "logg.txt"
        logg_fil.write_text(logg, encoding="utf-8")
        print(f"Logg:       {logg_fil}")

    print(f"Sammendrag: {sammendrag_fil}")
    print(f"Detaljer:   {detaljer_fil}")
    return kjoring_mappe
