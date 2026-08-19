"""
Cache for YOLO-deteksjoner per dokument og per modell.

Lagrer rå YOLO-bokser slik at en ny kjøring med samme vektfil kan hoppe over
GPU-inferensen — og, når OCR-cachen også treffer, hele PDF-renderingen.
Se ocr_cache.py for tokens og orientering.

Cache-nøkkel: vektfilens sha256 (i mappenavnet) + dokumentnavn (fil_revisjon_id).
Struktur: én mappe per uttrekk og modell, utledet automatisk fra --mappe:
    /data2/cache/uttrekk_5/yolo/a1b2c3d4e5f6a7b8/1000039999.json
    /data2/cache/uttrekk_5/yolo/9f8e7d6c5b4a3928/1000039999.json

Invalidering: imgsz, DPI, konfidens-gulv og rotasjon per side lagres i hver fil
              og sjekkes ved oppslag. Vektene ligger i mappenavnet, så en ny
              modell får automatisk sin egen cache uten å skygge for den gamle.

Bokser lagres ned til YOLO_CACHE_CONF_GULV og filtreres mot YOLO_CONF ved
lesing. Å endre YOLO_CONF — eller geometrifiltrene, matchingen eller
evalueringsterskelen — gir dermed fortsatt treff, så lenge den nye terskelen
ikke er lavere enn gulvet cachen ble skrevet med.

«rotasjon» er rotasjonen på bildet YOLO fikk, altså resultatet av
orienteringssteget. Full pipeline og --kun-yolo mater YOLO med det samme
orienteringskorrigerte bildet, så entryene er de samme og valider_full.sh og
valider_yolo.sh deler cache fullt ut. Feltet er likevel en del av nøkkelen:
skulle orienteringsmodellen endre seg, gir det miss i stedet for bokser i feil
koordinatrom.

Filformat per dokument:
    {cache_mappe}/{doc_id}.json
    {
      "versjon": 1,
      "imgsz": 1280,
      "dpi": 300,
      "conf_gulv": 0.05,
      "sider": [
        {
          "side": 1,
          "rotasjon": 0,
          "bokser": [[x0, y0, x1, y1, conf], ...]
        }
      ]
    }
"""

import hashlib
import json
import os

from config import PDF_DPI, YOLO_CACHE_CONF_GULV, YOLO_CONF, YOLO_IMGSZ

CACHE_VERSJON = 1

# Antall hex-siffer av vekt-hashen som brukes i mappenavnet. 16 hex = 64 bit,
# rikelig mot kollisjon mellom et håndterlig antall modeller.
HASH_LENGDE = 16

# (sti, mtime, størrelse) -> hash. Vektfilen er stor nok (~50 MB) at vi ikke vil
# lese den om igjen for hvert dokument.
_hash_cache = {}


def vekter_hash(vekter_sti):
    """sha256-prefiks for en vektfil, memoisert på mtime + størrelse."""
    st = os.stat(vekter_sti)
    nokkel = (os.path.abspath(vekter_sti), st.st_mtime_ns, st.st_size)
    if nokkel not in _hash_cache:
        h = hashlib.sha256()
        with open(vekter_sti, "rb") as f:
            for blokk in iter(lambda: f.read(1 << 20), b""):
                h.update(blokk)
        _hash_cache[nokkel] = h.hexdigest()[:HASH_LENGDE]
    return _hash_cache[nokkel]


def cache_mappe_for_vekter(base_mappe, vekter_sti):
    """Utled modell-spesifikk cache-mappe: {base}/{vekt-hash}."""
    return os.path.join(base_mappe, vekter_hash(vekter_sti))


def _cache_sti(cache_mappe, doc_navn):
    doc_id = os.path.splitext(os.path.basename(doc_navn))[0]
    return os.path.join(cache_mappe, f"{doc_id}.json")


def les_cache(cache_mappe, doc_navn, rotasjoner):
    """Les cachede YOLO-bokser for et dokument.

    `rotasjoner` er rotasjonen på bildet YOLO skal kjøre på, én per side, og
    må stemme med det som ble lagret.

    Returnerer en liste med én liste av (x0, y0, x1, y1, conf) per side —
    filtrert mot YOLO_CONF — hvis cachen finnes og er gyldig, ellers None.
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
    if data.get("imgsz") != YOLO_IMGSZ:
        return None
    if data.get("dpi") != PDF_DPI:
        return None
    # Gulvet må ligge på eller under dagens terskel, ellers mangler cachen
    # bokser vi nå vil ha med.
    gulv = data.get("conf_gulv")
    if gulv is None or gulv > YOLO_CONF:
        return None

    sider = data["sider"]
    if len(sider) != len(rotasjoner):
        return None
    if any(side["rotasjon"] != k for side, k in zip(sider, rotasjoner)):
        return None

    return [
        [tuple(b) for b in side["bokser"] if b[4] >= YOLO_CONF]
        for side in sider
    ]


def skriv_cache(cache_mappe, doc_navn, rotasjoner, bokser_per_side):
    """Lagre rå YOLO-bokser for et dokument til cache.

    `bokser_per_side` må komme fra en predict på YOLO_CACHE_CONF_GULV — lagrer
    vi et strengere utvalg, blir gulvet i filen en løgn og senere kjøringer med
    lavere YOLO_CONF får treff på en ufullstendig cache.
    """
    os.makedirs(cache_mappe, exist_ok=True)
    sti = _cache_sti(cache_mappe, doc_navn)

    sider = []
    for si, (k, bokser) in enumerate(zip(rotasjoner, bokser_per_side), start=1):
        sider.append({
            "side": si,
            "rotasjon": k,
            # Lagres uavrundet: json skriver korteste streng som round-tripper
            # eksakt, så en cachet kjøring gir bit-identiske bokser med en
            # ucachet. Avrunding her ville kunne vippe en boks over YOLO_CONF.
            "bokser": [list(boks) for boks in bokser],
        })

    data = {
        "versjon": CACHE_VERSJON,
        "imgsz": YOLO_IMGSZ,
        "dpi": PDF_DPI,
        "conf_gulv": YOLO_CACHE_CONF_GULV,
        "sider": sider,
    }

    # Skriv til temp-fil først for å unngå korrupte filer ved avbrudd
    tmp = sti + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, sti)
