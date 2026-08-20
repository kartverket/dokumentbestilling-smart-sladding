"""
Cache for PaddleOCR-tokens og orienteringsresultater per dokument.

Lagrer OCR-resultater per dokument slik at dyre GPU-operasjoner kan
hoppes over når samme dokument dukker opp i en ny kjøring.

Cache-nøkkel: dokumentnavn (fil_revisjon_id).
Struktur: én cache-mappe per uttrekk, utledet automatisk fra --mappe:
    /data2/cache/uttrekk_4/ocr/123456789.json
    /data2/cache/uttrekk_5/ocr/345678901.json

Invalidering: OCR-modellversjon + DPI lagres i hver fil og sjekkes
              ved oppslag. Ved endring misser alle oppslag automatisk.

Filformat per dokument:
    {cache_mappe}/{doc_id}.json
    {
      "versjon": 2,
      "ocr_modell": "v6",
      "dpi": 300,
      "sider": [
        {
          "side": 1,
          "rotasjon": 0,
          "tokens": [
            {"tekst": "ord", "x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0, "rec_score": 0.95},
            ...
          ]
        }
      ]
    }
"""

import json
import os
from collections import namedtuple

from config import MODELL_SETT, PDF_DPI

# Identisk med Token i paddle_ocr_model_fnr.py — definert her separat
# for å unngå å importere PaddleOCR bare for å lese cache.
Token = namedtuple("Token", ["tekst", "x0", "y0", "x1", "y1", "rec_score"])

CACHE_VERSJON = 2


def _cache_sti(cache_mappe, doc_navn):
    doc_id = os.path.splitext(os.path.basename(doc_navn))[0]
    return os.path.join(cache_mappe, f"{doc_id}.json")


def les_cache(cache_mappe, doc_navn):
    """Les cachet OCR-resultat for et dokument.

    Returnerer (rotasjoner, tokens_per_side) hvis cachen finnes og er gyldig,
    ellers None.
    """
    sti = _cache_sti(cache_mappe, doc_navn)
    if not os.path.isfile(sti):
        return None

    try:
        with open(sti, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Sjekk at forutsetningene stemmer
    if data.get("versjon") != CACHE_VERSJON:
        return None
    if data.get("ocr_modell") != MODELL_SETT:
        return None
    if data.get("dpi") != PDF_DPI:
        return None

    rotasjoner = []
    tokens_per_side = []
    for side in data["sider"]:
        rotasjoner.append(side["rotasjon"])
        tokens = [
            Token(t["tekst"], t["x0"], t["y0"], t["x1"], t["y1"], t.get("rec_score"))
            for t in side["tokens"]
        ]
        tokens_per_side.append(tokens)

    return rotasjoner, tokens_per_side


def skriv_cache(cache_mappe, doc_navn, rotasjoner, tokens_per_side):
    """Lagre OCR-resultat for et dokument til cache."""
    os.makedirs(cache_mappe, exist_ok=True)
    sti = _cache_sti(cache_mappe, doc_navn)

    sider = []
    for si, (rot, tokens) in enumerate(zip(rotasjoner, tokens_per_side), start=1):
        sider.append({
            "side": si,
            "rotasjon": rot,
            "tokens": [
                {"tekst": t.tekst, "x0": t.x0, "y0": t.y0, "x1": t.x1, "y1": t.y1,
                 "rec_score": t.rec_score}
                for t in tokens
            ],
        })

    data = {
        "versjon": CACHE_VERSJON,
        "ocr_modell": MODELL_SETT,
        "dpi": PDF_DPI,
        "sider": sider,
    }

    # Skriv til temp-fil først for å unngå korrupte filer ved avbrudd
    tmp = sti + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, sti)




