# YOLO training pipeline

Trains a YOLO model to detect fnr (fødselsnummer) in scanned PDF documents.

1. **Convert:** CSV annotations + PDFs -> PNG images + YOLO labels
2. **Split:** train/val/test (random, per year, per document type, or combined)
3. **Train:** train a YOLO model on the split dataset
4. **Publish:** put the finished model in the weights store with its metadata

## Prerequisites

A Python venv with `ultralytics`, `pymupdf` and `pandas`, activated before you run `make`:

```bash
pip install ultralytics pymupdf pandas numpy
```

The base model (for example yolo26x or yolo26l) downloads automatically.

## Full run

```bash
make PDFS=/path/to/pdf-folder CSV=/path/to/labels.csv
```

This runs the whole pipeline and produces two things:

1. **The training run** in `$(PROJECT)/$(NAME)/` (`$SLADD_RUNS/<NAME>/` on the server):
   statistics, plots, checkpoints and `weights/best.pt`. This is a working directory.
2. **The published model** in `$SLADD_WEIGHTS/<name>/`: the weights named after the
   model, next to the metadata saying what it was trained on.

```
$SLADD_WEIGHTS/uttrekk_4_jou/
  uttrekk_4_jou.pt     ← the weights (copy of best.pt, named)
  modell.json          ← dataset, split strategy, hyperparameters, metrics, git sha
  training/            ← results.csv, args.yaml, data.yaml, split_log.txt
```

Validation (`valider_yolo.sh`) and deploy (`./deploy.sh build weights=…`) point at the
published model, not the run. Every training run produces a file called `best.pt`, and in
a directory with ten of them there is no way to tell which model is which, or what any of
them trained on. A published model carries its name in the filename and its history in
`modell.json`.

Each step can also be run on its own, which helps when you already have a dataset.

### Step 1: Convert (`make convert`)

`scripts/convert_csv_to_yolo.py` reads the CSV of bounding-box annotations, renders each
PDF page to PNG, and writes normalised YOLO labels into `images_all` under `OUTPUT_DIR`.

Rows go through `filter_common.iter_label_rows`, so training and evaluation see the same
fasit: REJECTED and `ugyldige_labels.txt` rows are excluded, `manglende_labels.csv` rows
are included. On a re-run, labels for already-rendered documents in the run's scope are
rewritten from the current CSV without re-rendering, so a dataset directory can be reused
across label washes. Documents from another CSV sharing the directory keep their labels.

```bash
make convert \
  PDFS=/path/to/pdf-folder \
  CSV=/path/to/labels.csv \
  OUTPUT_DIR=/datasets
```

### Step 2: Split (`make split`)

`scripts/split_train_val.py` distributes images and labels into train/val/test (70/15/15
by default) and logs the split to `split_log.txt`.

The split is per document: pages of the same document share handwriting, stamps and
layout, so splitting per page leaks train into val/test and inflates both early stopping
and model selection. Ratios are filled by image count, and negatives are whole zero-fnr
documents added until each subset holds about 10% negative images per positive image.

```bash
# Random split (default)
make split PDFS=/path/to/pdf-folder CSV=/path/to/labels.csv OUTPUT_DIR=/datasets \
  TRAIN_RATIO=0.7 VAL_RATIO=0.15

# Yearly: at most 100 images per year
make split PDFS=/path/to/pdf-folder CSV=/path/to/labels.csv OUTPUT_DIR=/datasets \
  STRATEGY=yearly METADATA=/path/to/metadata.csv PER_YEAR=100

# Doc type: one document type only
make split PDFS=/path/to/pdf-folder CSV=/path/to/labels.csv OUTPUT_DIR=/datasets \
  STRATEGY=doc_type METADATA=/path/to/metadata.csv DOC_TYPE=Pantedokument

# Year + doc type: one document type within a year range
make split PDFS=/path/to/pdf-folder CSV=/path/to/labels.csv OUTPUT_DIR=/datasets \
  STRATEGY=year_and_doc_type METADATA=/path/to/metadata.csv \
  DOC_TYPE=Pantedokument YEAR_FROM=1970 YEAR_TO=1978
```

Strategies:

- `random` shuffles and splits by ratio
- `yearly` groups by year, takes up to `PER_YEAR` per year, and splits within each group
- `doc_type` filters on the `rettsstiftelsestyper` column, then splits randomly
- `year_and_doc_type` filters on document type and year range, then splits randomly

`yearly` needs a metadata CSV with the columns `fil_revisjon_id` and `dokument_aar`. The
doc type strategies additionally need `rettsstiftelsestyper`.

### Step 3: Train (`make train`)

Generates `data.yaml` in the dataset directory and starts YOLO training. Results land in
`$(PROJECT)/$(NAME)/`. `NAME` is timestamped unless you set it, so two runs never shadow
each other.

```bash
make train DATASET=/path/to/dataset
make train DATASET=/path/to/dataset EPOCHS=50 PATIENCE=10
```

### Step 4: Publish (`make publiser`)

`scripts/publish_model.py` copies `best.pt` from the training run into the weights store
under the model's own name and writes `modell.json` beside it. The metadata is read out of
the run itself (`args.yaml`, `results.csv`, `data.yaml`) and out of git, so it describes
what actually ran rather than what someone remembered to write down:

```json
{
  "name": "uttrekk_4_jou",
  "published": "2026-08-20T14:12:03+02:00",
  "weights": { "file": "uttrekk_4_jou.pt", "sha256": "4927f577…", "checkpoint": "best" },
  "trained": { "date": "2026-08-19T22:41:07+02:00", "base_model": "yolo26x.pt",
               "epochs": 200, "imgsz": 1280, "batch": 4, "patience": 20 },
  "dataset": { "path": "…/uttrekk_4/dataset", "classes": {"0": "fnr"},
               "n_images": {"train": 812, "val": 174, "test": 175},
               "strategy": "doc_type", "doc_type": "SR_JOU", "labels_csv": "…/uttrekk_4.csv" },
  "results": { "epochs_run": 168, "epoch": 143, "metrics/mAP50(B)": 0.88, "metrics/mAP50-95(B)": 0.62 },
  "code": { "git_sha": "eb6f64dd…", "git_clean_tree": true },
  "env": { "python": "3.12.3", "ultralytics": "8.4.90", "torch": "2.12.1" }
}
```

`make` runs this at the end by itself. You only need `make publiser` alone when training
was run separately:

```bash
make publiser \
  NAME=uttrekk_4_jou \
  DATASET=/path/to/dataset \
  STRATEGY=doc_type \
  DOC_TYPE=SR_JOU \
  CSV=/path/to/labels.csv
```

The variables you pass end up in `modell.json`; leave them out and they stay empty. The
`sha256` lets the same model be republished without writing anything: identical weights
just report back, differing weights are refused unless you set `OVERWRITE=1`.

The model can then be validated and built into an image:

```bash
./valider_yolo.sh model=$SLADD_WEIGHTS/uttrekk_4_jou/uttrekk_4_jou.pt uttrekk=5
./deploy.sh build weights=$SLADD_WEIGHTS/uttrekk_4_jou/uttrekk_4_jou.pt
```

### Pre-check: count documents (`make count`)

Before starting a run you can check how many documents and annotations match your filter.
This needs only the metadata and labels CSVs, not the PDFs.

```bash
# Count HJG documents from 1990-2006
make count METADATA=/path/to/metadata.csv CSV=/path/to/labels.csv \
  STRATEGY=year_and_doc_type DOC_TYPE=HJ_HJG YEAR_FROM=1990 YEAR_TO=2006

# Count all Pantedokument documents
make count METADATA=/path/to/metadata.csv STRATEGY=doc_type DOC_TYPE=OB_PAN
```

The output shows the number of matching documents, the distribution per year, and (if
`CSV` is set) how many of them have annotations.

## Make targets

| Target     | Description                                              |
|------------|----------------------------------------------------------|
| `all`      | `split` + `train` + `publiser` (default)                 |
| `convert`  | CSV/PDF to YOLO format only                              |
| `split`    | Convert + train/val/test split                           |
| `train`    | Full pipeline including YOLO training                    |
| `publiser` | Put the finished model in `$SLADD_WEIGHTS` with metadata  |
| `verify`   | Draw labels on images for a visual check                 |
| `coverage` | Find pages without labels                                |
| `smoke`    | 3-epoch smoke test                                       |
| `count`    | Count matching documents before training                 |
| `help`     | Show available targets and variables                     |

## Variables

Pass these as `make VARIABLE=value`.

### Input / output

| Variable      | Default                          | Description                    |
|---------------|----------------------------------|--------------------------------|
| `PDFS`        | `pdfs`                           | Folder of PDFs                 |
| `CSV`         | `labels.csv`                     | CSV of annotations             |
| `OUTPUT_DIR`  | `.`                              | Base directory for datasets    |
| `DATASET`     | `OUTPUT_DIR/dataset_<timestamp>` | Dataset directory (generated)  |

### Split

| Variable      | Default   | Description                                              |
|---------------|-----------|----------------------------------------------------------|
| `TRAIN_RATIO` | `0.7`     | Share of training data                                   |
| `VAL_RATIO`   | `0.15`    | Share of validation data                                 |
| `SEED`        | `42`      | Random seed for a reproducible split                     |
| `STRATEGY`    | `random`  | `random`, `yearly`, `doc_type` or `year_and_doc_type`    |
| `METADATA`    | *(empty)* | CSV with `dokument_aar` (required for non-random)        |
| `PER_YEAR`    | `100`     | Max images per year for `yearly`                         |
| `DOC_TYPE`    | *(empty)* | `rettsstiftelsestyper` value for the doc type strategies |
| `YEAR_FROM`   | *(empty)* | First year for `year_and_doc_type`                       |
| `YEAR_TO`     | *(empty)* | Last year for `year_and_doc_type`                        |

### Training

| Variable   | Default                            | Description                        |
|------------|------------------------------------|------------------------------------|
| `MODEL`    | `yolo26x.pt`                       | Pretrained model to finetune from  |
| `EPOCHS`   | `200`                              | Number of epochs                   |
| `IMGSZ`    | `1280`                             | Image size during training         |
| `BATCH`    | `4`                                | Batch size                         |
| `DEVICE`   | `cuda`                             | Device (`cuda`/`cpu`/`mps`)        |
| `PATIENCE` | `20`                               | Early stopping (epochs without gain) |
| `NAME`     | `trening_<timestamp>`              | Name of the run (and of the model) |
| `PROJECT`  | `$SLADD_RUNS` else `runs/detect`   | Where the run is written           |
| `RESUME`   | *(empty)*                          | `1` resumes from the previous `last.pt` |

### Publishing

| Variable          | Default                        | Description                                 |
|-------------------|--------------------------------|---------------------------------------------|
| `WEIGHTS`          | `$SLADD_WEIGHTS` else `vekter` | Weights store the model is published to     |
| `PUBLISH_NAME`   | `$(NAME)`                      | Name of the published model                 |
| `PUBLISH_WEIGHTS` | `best`                         | `best` or `last` checkpoint                 |
| `OVERWRITE`       | *(empty)*                      | `1` replaces a model with the same name     |
