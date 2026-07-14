# Treningspipeline for YOLO

Denne mappen trener en YOLO-modell til å detektere fødselsnummer (FNR) i skannede PDF-dokumenter.

## Forutsetninger

- Python venv med `ultralytics`, `pymupdf`, `pandas` installert
- Aktiver venv før du kjører `make`

```bash
pip install ultralytics pymupdf pandas
```

## Full kjøring

For å kjøre pipelinen er det inkludert en `Makefile` for å lettere kjøre trening. 

Eksempel kjøring

```bash
make \
  PDFS=/sti/til/pdf-mappe \
  CSV=/sti/til/labels.csv
```

Dette kjører hele pipelinen: konverterer CSV → YOLO-format, splitter til train/val/test, og starter trening.

Det er derimot mulig å kjøre hvert steg separat hvis man for eksempel allerede har et eksisterende dataset man ønsker å trene på. Hvert steg har en egen make kommando man kjøre:

### Steg 1: Konvertering (`make convert`)

`scripts/convert_csv_to_yolo.py` leser CSV-filen med bounding-box-annotasjoner, rendrer hver PDF-side til PNG, og skriver normaliserte YOLO-labels.

Eksempel kjøring:

```bash
make convert \
  PDFS=/sti/til/pdf-mappe \
  CSV=/sti/til/labels.csv \
  OUTPUT_DIR=/datasets
```

### Steg 2: Split (`make split`)

`scripts/split_train_val.py` fordeler bilder og labels tilfeldig i train/val/test (70/15/15 som standard). Logger også trening i en `split_log.txt`.

Eksempel kjøring:

```bash
make split \
  PDFS=/sti/til/pdf-mappe \
  CSV=/sti/til/labels.csv \
  OUTPUT_DIR=/datasets \
  TRAIN_RATIO=0.8 \
  VAL_RATIO=0.2
```

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

## Make-targets

| Target     | Beskrivelse                                        |
|------------|----------------------------------------------------|
| `all`      | Kjører `split` + `train` (default)                 |
| `convert`  | Kun konvertering fra CSV/PDF til YOLO-format        |
| `split`    | Konvertering + train/val/test-split                 |
| `train`    | Full pipeline inkl. YOLO-trening                    |
| `verify`   | Tegner labels på bilder for visuell sjekk           |
| `coverage` | Finner sider uten labels                            |
| `smoke`    | 3-epoch røyktest for å sjekke at alt fungerer       |
| `help`     | Viser tilgjengelige targets og variabler             |

## Konfigurerbare variabler
Man kan endre ulike variabler ved kjøring. Disse kan man legge ved med `make VARIABEL=verdi`.

| Variabel      | Standard                              | Beskrivelse                        |
|---------------|---------------------------------------|------------------------------------|
| `PDFS`        | `pdfs`                                | Mappe med kilde-PDFer              |
| `CSV`         | `labels.csv`                          | CSV med annotasjoner               |
| `DATASET`     | `dataset_<timestamp>`                 | Utmappe (ny per kjøring)           |
| `TRAIN_RATIO` | `0.7`                                 | Andel treningsdata                 |
| `VAL_RATIO`   | `0.15`                                | Andel valideringsdata              |
| `MODEL`       | `yolo26x.pt`                          | Pretrent modell å fintune fra      |
| `EPOCHS`      | `100`                                 | Antall epoker                      |
| `IMGSZ`       | `1280`                                | Bildestørrelse under trening       |
| `BATCH`       | `4`                                   | Batch-størrelse                    |
| `DEVICE`      | `cuda`                                | Enhet (cuda/cpu/mps)               |
| `PATIENCE`    | `20`                                  | Early stopping (epoker uten gain)  |

## Trene på eksisterende datasett

Hvis du allerede har konvertert et datasett og vil trene på nytt (eller med andre hyperparametre):

```bash
make train DATASET=/Users/william/Downloads/dataset_2026-07-13T12-27-05
```

Konverteringen hoppes over automatisk hvis `images_all/` allerede finnes.