# YOLO FNR Detection – Treningspipeline

Trener en YOLO-modell til å detektere fødselsnummer (FNR) i skannede PDF-dokumenter.

## Forutsetninger

- Python venv med `ultralytics`, `pymupdf`, `pandas` installert
- Aktiver venv før du kjører `make`
- Kjør alltid `make` fra `yolo/`-mappen

```bash
pip install ultralytics pymupdf pandas
```

## Rask start

```bash
make \
  PDFS=/sti/til/pdf-mappe \
  CSV=/sti/til/labels.csv
```

Dette kjører hele pipelinen: konverterer CSV → YOLO-format, splitter til train/val/test, og starter trening.

## Pipeline-steg

```
CSV + PDFs
    │
    ▼  (convert_csv_to_yolo.py)
dataset_<timestamp>/
├── images_all/          ← alle sider rendret som PNG (300 DPI)
├── labels_all/          ← YOLO-labels (.txt) per bilde
│
    │
    ▼  (split_train_val.py)
├── images/
│   ├── train/           ← 70%
│   ├── val/             ← 15%
│   └── test/            ← 15%
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml            ← generert automatisk
├── .converted           ← stamp-fil (make-intern)
└── .split               ← stamp-fil (make-intern)
```

### Steg 1: Konvertering (`make convert`)

`scripts/convert_csv_to_yolo.py` leser CSV-filen med bounding-box-annotasjoner, rendrer hver PDF-side til PNG, og skriver normaliserte YOLO-labels.

- Beholder kun: manuelle bokser + ML-genererte som er ACCEPTED
- Koordinater konverteres fra PDF-punkter → piksel → normalisert [0,1]

### Steg 2: Split (`make split`)

`scripts/split_train_val.py` fordeler bilder og labels tilfeldig i train/val/test (70/15/15 som standard).

### Steg 3: Trening (`make train`)

Genererer `data.yaml` i datasett-mappen og starter YOLO-trening. Resultater havner i `runs/detect/train/` (Ultralytics sin standard).

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

## Mappestruktur

```
yolo/
├── Makefile
├── README.md
├── data.yaml              ← statisk referanse (ikke brukt ved trening)
├── scripts/
│   ├── convert_csv_to_yolo.py
│   ├── split_train_val.py
│   ├── verify_boxes.py
│   └── check_coverage.py
├── dataset_<timestamp>/   ← generert per kjøring
└── runs/                  ← YOLO-treningsresultater (vekter, metrikker)
```
