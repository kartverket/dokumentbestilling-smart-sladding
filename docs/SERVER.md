# Training server: setup and use

## First time: set up the environment

```bash
tmux new -s trening
source /home/smartsladding/dokumentbestilling-smart-sladding_test/dokumentbestilling-smart-sladding/activate.sh
```

One line activates both the venv and every `SLADD_` variable. Put the `source`
line in `~/.bashrc` so you do not have to repeat it.

## Variables

Set by `server.env`, which `activate.sh` sources.

| Variable | Value | What it is |
|---|---|---|
| `SLADD_REPO` | `.../dokumentbestilling-smart-sladding` | Repo root |
| `SLADD_UTTREKK` | `/data2/smartsladding-uttrekk` | PDFs per uttrekk |
| `SLADD_LABELS` | `.../smartsladding-uttrekk-labels` | Label CSVs (ground truth) |
| `SLADD_METADATA` | `.../smartsladding-uttrekk-metadata` | Metadata CSVs |
| `SLADD_RUNS` | `/data2/runs` | Raw training runs |
| `SLADD_WEIGHTS` | `/data2/vekter` | Published models |
| `SLADD_PRODWEIGHTS` | `$SLADD_WEIGHTS/yolo-yearly-10000-docs/yolo-yearly-10000-docs.pt` | Default model |
| `SLADD_VALIDATION` | `/data2/validering` | Validation results |
| `SLADD_LISTS` | `/data2/validering/lister` | Document ID lists |
| `SLADD_CACHE` | `/data2/cache` | Everything derived, per uttrekk |
| `SLADD_RUN` | `.../utils/run.py` | Validation script |
| `SLADD_PRECACHE` | `.../utils/precache.py` | Cache filler |
| `SLADD_TRAIN` | `.../train` | Training directory |
| `SLADD_LOGS` | `/data/docker` | Container log root on the host |
| `SLADD_LOG_DAYS` | `30` | Days of log history per log file |
| `SLADD_VLM` | empty | `1` turns the VLM verifier on |
| `SLADD_VLM_URL` | `http://127.0.0.1:8080/v1` | llama-server endpoint, as the host sees it |
| `SLADD_VLM_URL_DOCKER` | `http://host.docker.internal:8080/v1` | Same endpoint, as a container sees it |
| `SLADD_VLM_MODEL` | `qwen3.8:27b` | Label only, llama-server serves one model |
| `SLADD_VLM_TIMEOUT` | `20` | Seconds per box, then the box is kept |
| `SLADD_VLM_CONCURRENT` | `4` | Boxes in flight per page |
| `SLADD_VENV` | `.../venv/bin/activate` | Venv |

`SLADD_LOGS` and `SLADD_LOG_DAYS` belong to deploy, not to training:
`deploy.sh` reads them and passes them to compose as `LOG_ROOT` and
`LOG_BACKUP_DAYS`.

The `SLADD_VLM` variables belong to both. `deploy.sh` sources this file and
compose passes them to the containers, and `utils/run.py` reads them as the
defaults for `--vlm-url` and `--vlm-model`. `SLADD_VLM` plus a URL and a model
are needed before anything happens. The container gets
`SLADD_VLM_URL_DOCKER` where the host-side tools get `SLADD_VLM_URL`, because
loopback inside a container is the container. See the VLM verifier section in
the README.

An empty value after login means a stale shell, not a broken install. Run
`source activate.sh` again in that pane.

---

## Training

```bash
make -C $SLADD_TRAIN \
  PDFS=$SLADD_UTTREKK/uttrekk_4/ \
  CSV=$SLADD_LABELS/uttrekk_4.csv \
  DATASET=$SLADD_CACHE/uttrekk_4/dataset \
  STRATEGY=doc_type \
  METADATA=$SLADD_METADATA/uttrekk_4.csv \
  DOC_TYPE=SR_JOU \
  DEVICE=cuda \
  NAME=uttrekk_4_jou_med_negative
```

Add `MODEL=$SLADD_PRODWEIGHTS` to train on top of an existing model, and
`PATIENCE=N` to change early stopping. The full flag list is in
[train/README.md](../train/README.md).

The run lands in `$SLADD_RUNS/<NAME>/` and is published as a finished model in
`$SLADD_WEIGHTS/<NAME>/` with `<NAME>.pt` and `modell.json`. Validation and
`deploy.sh` point at the published model, never at `best.pt` inside the run.

To publish a run that was trained without publishing:

```bash
make -C $SLADD_TRAIN publiser \
  NAME=uttrekk_4_jou_med_negative \
  DATASET=$SLADD_CACHE/uttrekk_4/dataset
```

`modell.json` is only as complete as the variables you pass. Run `publiser`
with the same `PDFS`/`CSV`/`STRATEGY` values the training used, or they stay
empty in the metadata.

---

## Validation

### Step 1: build a document list

```bash
./lag_liste.sh uttrekk=5 docs=SR_JOU name=jou
./lag_liste.sh uttrekk=5 docs=SR_JOU,FR_REG years=2020-2026 name=jou_reg
./lag_liste.sh uttrekk=5 years=2024,2025 name=nyere
```

At least one of `docs=` and `years=` is required. The list is written to
`$SLADD_LISTS/uttrekk_<n>_<name>.txt`.

### Step 2: run the validation

| Script | What it runs | Use for |
|---|---|---|
| `valider_yolo.sh` | YOLO only, no OCR | Quick test of YOLO weights |
| `valider_full.sh` | OCR + YOLO + matching (production logic) | The full pipeline |

```bash
./valider_yolo.sh model=$SLADD_PRODWEIGHTS uttrekk=5 list=jou
./valider_full.sh model=$SLADD_PRODWEIGHTS uttrekk=5 list=jou
```

`model=` and `uttrekk=` are required on both. Without `list=`, every document
in the uttrekk is run. `name=` overrides the output directory name, which is
otherwise derived from the model and the uttrekk. Results land in
`$SLADD_VALIDATION/<name>/`.

Both scripts build every path from the uttrekk number and the list name, check
that the files exist before starting, print a summary of what is about to run,
and pass `--csv --truth --only-error` (CSV result, evaluation against ground
truth, images for errors only).

`valider_full.sh` takes a few more parameters: `rules=no` skips every
post-filter (raw detection, for measuring what the rules contribute),
`metadata=yes` sends rettsstiftelse types from `$SLADD_METADATA/uttrekk_N.csv`
so the rule profiles match prod, `images=N` caps how many error images are
drawn, `processes=N` sets the worker count. `precache=no` skips filling the
cache first. `live=N` prints a running count of documents, boxes and kilder
every N seconds; `live=no` turns it off, and the default is every 60 seconds.

Anything starting with a dash is handed straight to `run.py`, so
`--vlm --vlm-concurrent 1` works without the script knowing those flags.

`deploy=test` replaces `model=` and sends every PDF over HTTP to a running
container instead. The two cannot be combined, because the image carries its
own weights and rules. See "Validating a deployment" in the README.

---

## Cache

PaddleOCR and YOLO are the heavy operations per document. Both are
deterministic for a given input, so both are cached per document and reused
across runs. The cache is **on by default** when `SLADD_CACHE` is set.

```
/data2/cache/uttrekk_5/
  ocr/       ← PaddleOCR tokens + orientation
  yolo/      ← raw boxes, one subdirectory per weights hash
  dataset/   ← converted images + YOLO labels (training)
/data2/cache/vlm/
  <fingerprint>/  ← VLM verdicts, one directory per prompt and model
```

The paths are derived from `--folder`, so `$SLADD_UTTREKK/uttrekk_5/` gives
`$SLADD_CACHE/uttrekk_5/`. The YOLO cache is per weights file: each model gets
its own subdirectory, and a new model never reads another model's boxes. On a
hit in both caches the PDF rendering is skipped too, so re-running the same
model over the same uttrekk costs almost nothing.

The VLM cache sits beside the uttrekk directories rather than inside one,
because the key is the crop: the same box judged from another uttrekk gives the
same answer. Its subdirectory name is a fingerprint of the prompt and the model
settings, so editing the prompt starts a fresh directory and rejudges
everything. Only `utils/run.py` writes it; the containers run without a
judgement cache.

Override or disable:

```bash
python -u $SLADD_RUN --ocr-cache /data2/cache/uttrekk_5/ocr ...
python -u $SLADD_RUN --no-ocr-cache --no-yolo-cache ...
```

Every cache file records its own assumptions (OCR model version, DPI, weights),
and they are checked on lookup. Change the OCR model and every lookup misses,
and the documents are reprocessed. To force that for one uttrekk:

```bash
rm -rf $SLADD_CACHE/uttrekk_5/ocr      # or the whole uttrekk_5/
```

PDF rendering is not cached: the files are too large, roughly 25 MB per page.

---

## The model store

`$SLADD_WEIGHTS` is where finished models live, one directory per model:

```
/data2/vekter/
  yolo-yearly-10000-docs/
    yolo-yearly-10000-docs.pt    ← the weights
    modell.json                  ← dataset, hyperparameters, metrics, git sha
    training/                    ← results.csv, args.yaml, data.yaml, split_log.txt
```

`$SLADD_RUNS` holds working directories instead: checkpoints, plots and
`weights/best.pt` from each run. Every model is called the same thing there, so
nothing outside training should point into it. `make publiser` is the bridge
between the two.

`$SLADD_PRODWEIGHTS` points at one of the published models in the store, the
weights file itself and not the directory. That is the model `run.py` uses when
`--yolo-weights` is omitted and the one `./deploy.sh build` bakes in without a
`weights=` argument.

## File map

```
activate.sh        ← source this: venv + variables
server.env         ← the SLADD_ variables (loaded by activate.sh)
lag_liste.sh       ← generate document ID lists from metadata
valider_yolo.sh    ← validation, YOLO only
valider_full.sh    ← validation, full production logic (OCR + YOLO)
deploy.sh          ← build, test and promote the container images
app/ocr_cache.py   ← per-document OCR cache
app/yolo_cache.py  ← per-document YOLO cache
app/vlm_cache.py   ← per-crop cache of the verifier's answers
train/Makefile     ← training pipeline
train/scripts/publish_model.py  ← run → finished model in $SLADD_WEIGHTS
```
