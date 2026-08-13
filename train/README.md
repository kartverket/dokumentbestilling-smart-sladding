# Treningspipeline for YOLO

Denne mappen trener en YOLO-modell til å detektere fødselsnummer (FNR) i skannede PDF-dokumenter.
Pipelinen inneholder disse delene:

1. **Konvertering:** CSV-annotasjoner + PDFer -> PNG-bilder + YOLO-labels
2. **Split:** Fordeler data i train/val/test (tilfeldig, per år, per dokumenttype, eller kombinasjon)
3. **Trening:** Trener en YOLO-modell på det splittede datasettet


## Forutsetninger
- Python venv med `ultralytics`, `pymupdf`, `pandas` installert
- Aktiver venv før du kjører `make`

```bash
pip install ultralytics pymupdf pandas numpy
```

- En YOLO-modell, eks. yolo26x eller yolo26l. Modellen man velger å bruke vil bli lastet ned automatisk.

## Full kjøring

For å kjøre pipelinen er det inkludert en `Makefile` for å lettere kjøre treningen. 

Eksempel kjøring

```bash
make \
PDFS=/sti/til/pdf-mappe \
CSV=/sti/til/labels.csv
```

Dette kjører hele pipelinen: konverterer CSV -> YOLO-format, splitter til train/val/test, og starter trening.

Outputen fra pipelinen vil være en `runs/detect/train/` mappe som inneholder statistikk for trening, i tillegg til selve vektene (`best.pt`), som brukes senere i inferens.

Det er derimot mulig å kjøre hvert steg separat hvis man for eksempel allerede har et eksisterende dataset man ønsker å trene på. Hvert steg har en egen make-kommando man kan kjøre:

### Steg 1: Konvertering (`make convert`)

`scripts/convert_csv_to_yolo.py` leser CSV-filen med bounding-box-annotasjoner, rendrer hver PDF-side til PNG, og skriver normaliserte YOLO-labels.

Eksempel kjøring:

```bash
make convert \
PDFS=/sti/til/pdf-mappe \
CSV=/sti/til/labels.csv \
OUTPUT_DIR=/datasets
```

Outputtet fra dette vil ligge i dataset-mappen spesifisert med `OUTPUT_DIR`. Her vil de konverte filene ligge i `images_all`.

### Steg 2: Split (`make split`)

`scripts/split_train_val.py` fordeler bilder og labels tilfeldig i train/val/test (70/15/15 som standard). Logger også trening i en `split_log.txt`.

Eksempel kjøring:

```bash
# Tilfeldig split (standard)
make split \
  PDFS=/sti/til/pdf-mappe \
  CSV=/sti/til/labels.csv \
  OUTPUT_DIR=/datasets \
  TRAIN_RATIO=0.7 \
  VAL_RATIO=0.15

# Yearly split — tar maks 100 bilder per år
make split \
  PDFS=/sti/til/pdf-mappe \
  CSV=/sti/til/labels.csv \
  OUTPUT_DIR=/datasets \
  STRATEGY=yearly \
  METADATA=/sti/til/metadata.csv \
  PER_YEAR=100

# Doc type split — kun én dokumenttype
make split \
  PDFS=/sti/til/pdf-mappe \
  CSV=/sti/til/labels.csv \
  OUTPUT_DIR=/datasets \
  STRATEGY=doc_type \
  METADATA=/sti/til/metadata.csv \
  DOC_TYPE=Pantedokument

# Year + doc type — én dokumenttype innenfor et årsintervall
make split \
  PDFS=/sti/til/pdf-mappe \
  CSV=/sti/til/labels.csv \
  OUTPUT_DIR=/datasets \
  STRATEGY=year_and_doc_type \
  METADATA=/sti/til/metadata.csv \
  DOC_TYPE=Pantedokument \
  YEAR_FROM=1970 \
  YEAR_TO=1978
```

Strategier:
- **random** — tilfeldig shuffle og split etter ratio
- **yearly** — grupperer per år, velger opptil `PER_YEAR` per år, splitter innenfor hver gruppe
- **doc_type** — filtrerer på `rettsstiftelsestyper`-kolonnen, splitter tilfeldig
- **year_and_doc_type** — filtrerer på dokumenttype og årsintervall (`YEAR_FROM`–`YEAR_TO`), splitter tilfeldig

Yearly-strategien krever en metadata-CSV med kolonnene `fil_revisjon_id` og `dokument_aar`. Doc type-strategiene krever i tillegg `rettsstiftelsestyper`.

### Steg 3: Trening (`make train`)

Genererer `data.yaml` i datasett-mappen og starter YOLO-trening. Resultater havner i `runs/detect/train/` (Ultralytics sin standard).

Eksempel kjøring:

```bash
# Trene på et eksisterende datasett:
make train DATASET=/sti/til/dataset

# Trene med justerte hyperparametre:
make train \
DATASET=/sti/til/dataset \
EPOCHS=50 \
PATIENCE=10
```

### Forhåndssjekk: Tell dokumenter (`make count`)

Før du starter en treningskjøring kan du sjekke hvor mange dokumenter og annotasjoner som matcher filteret ditt. Dette krever ikke PDF-ene — kun metadata- og labels-CSV.

```bash
# Tell HJG-dokumenter fra 1990–2006
make count \
  METADATA=/sti/til/metadata.csv \
  CSV=/sti/til/labels.csv \
  STRATEGY=year_and_doc_type \
  DOC_TYPE=HJ_HJG \
  YEAR_FROM=1990 \
  YEAR_TO=2006

# Tell alle Pantedokument-dokumenter
make count \
  METADATA=/sti/til/metadata.csv \
  STRATEGY=doc_type \
  DOC_TYPE=OB_PAN
```

Outputen viser antall matchende dokumenter, fordeling per år, og (om `CSV` er satt) hvor mange av dem som har annotasjoner.

## Make-targets

| Target     | Beskrivelse                                        |
|------------|----------------------------------------------------|
| `all`      | Kjører `split` + `train` (default)                 |
| `convert`  | Kun konvertering fra CSV/PDF til YOLO-format        |
| `split`    | Konvertering + train/val/test-split                 |
| `train`    | Full pipeline inkl. YOLO-trening                    |
| `verify`   | Tegner labels på bilder for visuell sjekk           |
| `coverage` | Finner sider uten labels                            |
| `smoke`    | 3-epoch smoketest for å sjekke at alt fungerer       |
| `count`    | Teller matchende dokumenter før trening               |
| `help`     | Viser tilgjengelige targets og variabler             |

## Konfigurerbare variabler
Man kan endre ulike variabler ved kjøring. Disse kan man legge ved med `make VARIABEL=verdi`.

### Input / output

| Variabel      | Standard                              | Beskrivelse                        |
|---------------|---------------------------------------|------------------------------------|
| `PDFS`        | `pdfs`                                | Filbane til mappe med PDFer              |
| `CSV`         | `labels.csv`                          | CSV med annotasjoner               |
| `OUTPUT_DIR`  | `.`                                   | Mappen for datasett (generert automatisk )             |
| `DATASET`     | `OUTPUT_DIR/dataset_<timestamp>`      | Utmappe (generert automatisk)      |

### Split

| Variabel      | Standard                              | Beskrivelse                        |
|---------------|---------------------------------------|------------------------------------|
| `TRAIN_RATIO` | `0.7`                                 | Andel treningsdata                 |
| `VAL_RATIO`   | `0.15`                                | Andel valideringsdata              |
| `SEED`        | `42`                                  | Random seed for reproduserbar split|
| `STRATEGY`    | `random`                              | Split-strategi: `random` eller `yearly` |
| `METADATA`    | *(tom)*                               | CSV med `dokument_aar` (påkrevd for `yearly`) |
| `PER_YEAR`    | `100`                                 | Maks bilder per år for `yearly`    |

### Trening

| Variabel      | Standard                              | Beskrivelse                        |
|---------------|---------------------------------------|------------------------------------|
| `MODEL`       | `yolo26x.pt`                          | Pretrent modell å fintune fra      |
| `EPOCHS`      | `200`                                 | Antall epoker                      |
| `IMGSZ`       | `1280`                                | Bildestørrelse under trening       |
| `BATCH`       | `4`                                   | Batch-størrelse                    |
| `DEVICE`      | `cuda`                                | Enhet (`cuda`/`cpu`/`mps`)         |
| `PATIENCE`    | `20`                                  | Early stopping (epoker uten gain)  |

