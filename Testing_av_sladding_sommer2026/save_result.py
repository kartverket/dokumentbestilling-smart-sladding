import json
from datetime import datetime
from pathlib import Path


def lagre_resultat(resultat, mappe="."):
    """Lagrer resultatet fra mal_overlapp til en JSON-fil.

    Filnavnet får formatet result-<tidsstempel>.json,
    f.eks. result-2026-07-02T14-32-05.json
    """
    if resultat is None:
        return None

    tidsstempel = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filnavn = Path(mappe) / f"result-{tidsstempel}.json"

    with open(filnavn, "w", encoding="utf-8") as f:
        json.dump(resultat, f, ensure_ascii=False, indent=2)

    print(f"Resultat lagret: {filnavn}")
    return filnavn
