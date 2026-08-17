# Treningsserver: Oppsett og bruk

## Første gang: Sett opp miljøet

```bash
tmux new -s trening
source /home/smartsladding/dokumentbestilling-smart-sladding_test/dokumentbestilling-smart-sladding/activate.sh
```

Én linje aktiverer både venv og alle `SLADD_`-variabler.

> **Tips:** Legg gjerne `source`-linjen i `~/.bashrc` så slipper du å gjenta den.

## Tilgjengelige variabler

| Variabel            | Verdi                                          | Beskrivelse                     |
|--------------------|--------------------------------------------------|---------------------------------|
| `SLADD_REPO`       | `.../dokumentbestilling-smart-sladding`         | Repo-rot                        |
| `SLADD_UTTREKK`    | `/data2/smartsladding-uttrekk`                  | PDFer per uttrekk               |
| `SLADD_LABELS`     | `.../smartsladding-uttrekk-labels`              | Labels-CSVer                    |
| `SLADD_METADATA`   | `.../smartsladding-uttrekk-metadata`            | Metadata-CSVer                  |
| `SLADD_RUNS`       | `/data2/runs`                                   | Treningskjøringer               |
| `SLADD_VALIDERING` | `/data2/validering`                             | Valideringsresultater           |
| `SLADD_LISTER`     | `/data2/validering/lister`                      | Fil-ID-lister                   |
| `SLADD_PRODVEKTER` | `.../yolo-yearly-10000-docs.pt`                 | Nåværende prod-modell           |
| `SLADD_CACHE`      | `/data2/cache`                                  | Alt derivert, per uttrekk       |
| `SLADD_RUN`        | `.../utils/run.py`                              | Validerings-script              |
| `SLADD_TRAIN`      | `.../train`                                     | Trenings-mappe                  |

---

## Trening

### Standard trening (fra scratch)

```bash
make -C $SLADD_TRAIN \
  PDFS=$SLADD_UTTREKK/uttrekk_4/ \
  CSV=$SLADD_LABELS/uttrekk_4.csv \
  DATASET=$SLADD_CACHE/uttrekk_4/dataset \
  STRATEGY=doc_type \
  METADATA=$SLADD_METADATA/uttrekk_4.csv \
  DOC_TYPE=SR_JOU \
  DEVICE=cuda \
  NAME=uttrekk_4_jou_med_negative \
  PROJECT=$SLADD_RUNS
```

### Trening basert på en eksisterende modell

```bash
make -C $SLADD_TRAIN \
  PDFS=$SLADD_UTTREKK/uttrekk_4/ \
  CSV=$SLADD_LABELS/uttrekk_4.csv \
  DATASET=$SLADD_CACHE/uttrekk_4/dataset \
  STRATEGY=doc_type \
  METADATA=$SLADD_METADATA/uttrekk_4.csv \
  DOC_TYPE=SR_JOU \
  DEVICE=cuda \
  NAME=uttrekk_4_jou_based_pat20 \
  PROJECT=$SLADD_RUNS \
  PATIENCE=20 \
  MODEL=$SLADD_PRODVEKTER
```

---

## Validering

### Steg 1: Lag en dokumentliste

```bash
./lag_liste.sh uttrekk=5 docs=SR_JOU name=jou
```

Flere dokumenttyper og årsfilter:

```bash
./lag_liste.sh uttrekk=5 docs=SR_JOU,FR_REG years=2020-2026 name=jou_reg
./lag_liste.sh uttrekk=5 years=2024,2025 name=nyere
./lag_liste.sh uttrekk=4 docs=OB_MOB name=mob
```

### Steg 2: Kjør validering med wrapper-skriptet

```bash
# Valider prod-modellen på uttrekk 5, JOU-dokumenter
./valider.sh modell=$SLADD_PRODVEKTER uttrekk=5 liste=jou

# Valider en trent modell
./valider.sh modell=$SLADD_RUNS/uttrekk_4_jou_med_negative/weights/best.pt uttrekk=5 liste=jou

# Valider en annen modell på et annet uttrekk/doctype
./valider.sh modell=$SLADD_RUNS/uttrekk_4_jou_based_pat20/weights/best.pt uttrekk=4 liste=mob

# Egendefinert navn på utmappen
./valider.sh modell=$SLADD_PRODVEKTER uttrekk=5 liste=jou navn=prod_test
```

Det er alt! Skriptet:
- Bygger alle stier automatisk fra modellnavn, uttrekk-nr og doc_type
- Sjekker at alle filer/mapper finnes før det starter
- Viser en oppsummering av hva som kjøres
- Navngir utmappen etter konvensjonen `{modell}_validert_pa_uttrekk_{nr}_{type}`

### Full pipeline (YOLO + PaddleOCR)

For å kjøre full pipeline (ikke bare YOLO), bruk `python -u $SLADD_RUN` direkte uten `--kun-yolo`:

```bash
python -u $SLADD_RUN \
  --mappe $SLADD_UTTREKK/uttrekk_5/ \
  --velg-fra-fil $SLADD_LISTER/uttrekk_5_jou.txt \
  --yolo-vekter $SLADD_PRODVEKTER \
  --csv --fasit --kun-feil \
  --fasit-csv $SLADD_LABELS/uttrekk_5.csv \
  --csv-ut $SLADD_VALIDERING/full_pipeline_uttrekk_5_jou/resultat.csv \
  --png-mappe $SLADD_VALIDERING/full_pipeline_uttrekk_5_jou/feilbilder \
  --resultat-mappe $SLADD_VALIDERING/full_pipeline_uttrekk_5_jou
```

OCR-cachen aktiveres automatisk. Første kjøring prosesserer alle dokumenter; påfølgende kjøringer med samme uttrekk hopper over PaddleOCR.

---

## OCR-cache

PaddleOCR er den tyngste operasjonen per dokument. Resultatet er deterministisk for en gitt PDF, så det caches per dokument og gjenbrukes på tvers av kjøringer.

### Cache-struktur (alt per uttrekk)

```
/data2/cache/
  uttrekk_4/
    ocr/                  ← PaddleOCR-tokens + orientering
      123456789.json
      234567890.json
    dataset/              ← Konverterte bilder + YOLO-labels (trening)
      images_all/
      labels_all/
  uttrekk_5/
    ocr/
      345678901.json
    dataset/
      ...
```

Alt derivert for ett uttrekk samlet i én mappe. Slett alt cachet for ett uttrekk med:

```bash
rm -rf $SLADD_CACHE/uttrekk_5
```

### Hvordan det aktiveres

Cachen er **på som standard** når `SLADD_CACHE` er satt (via `server.env`). OCR-cache-stien utledes automatisk:

```
--mappe $SLADD_UTTREKK/uttrekk_5/  →  cache: $SLADD_CACHE/uttrekk_5/ocr/
--mappe $SLADD_UTTREKK/uttrekk_4/  →  cache: $SLADD_CACHE/uttrekk_4/ocr/
```

Eksplisitt overstyring:

```bash
# Spesifiser cache-sti manuelt
python -u $SLADD_RUN --ocr-cache /data2/cache/uttrekk_5/ocr ...

# Deaktiver cache helt
python -u $SLADD_RUN --no-ocr-cache ...
```

### Flyt

```
Første kjøring (cache-miss):
  render PDF → orientering → PaddleOCR → lagre til cache → YOLO → resultat

Neste kjøring, samme uttrekk (cache-treff):
  render PDF → last fra cache → YOLO → resultat
```

Ved `--kun-yolo`-kjøringer brukes ikke cachen (ingen OCR å cache).

### Invalidering

Hver cache-fil inneholder forutsetningene (OCR-modellversjon, DPI). Ved oppslag sjekkes disse automatisk. Hvis du bytter OCR-modell (f.eks. v6 → v7), vil alle oppslag misse og dokumentene prosesseres på nytt.

For å tvinge full reprosessering av ett uttrekk:

```bash
rm -rf $SLADD_CACHE/uttrekk_5/ocr
```

### Hva caches?

| Operasjon | Caches? | Begrunnelse |
|-----------|---------|-------------|
| PaddleOCR-tokens | ✅ | Tung GPU, deterministisk per dokument |
| Orientering | ✅ | Lagres sammen med tokens |
| YOLO-inferens | ❌ | Avhenger av modellvekter som endres ofte |
| PDF-rendering | ❌ | For store filer (~25 MB/side) |

---

## Filstruktur

```
activate.sh      ← Source denne: aktiverer venv + laster variabler
server.env       ← Globale stier (lastes av activate.sh)
valider.sh       ← Snarvei for kun-YOLO-validering
lag_liste.sh     ← Generer dokument-ID-lister fra metadata
app/ocr_cache.py ← Per-dokument OCR-cache (les/skriv)
train/Makefile   ← Treningspipeline
```




