# dokumentbestilling-smart-sladding

Automatic sladding of fødselsnummer and d-nummer in ordered tinglysing
documents. A PDF goes in, a list of boxes to cover comes out.

> **Working at Kartverket?** Read the routines before you start:
> [Confluence](https://kartverket.atlassian.net/wiki/x/F4Dwn)

See [docs/TEKNISK.md](docs/TEKNISK.md) for how the detection actually works.

## Repo layout

```
app/      the model API (Flask + PaddleOCR + YOLO); all the detection logic
job/      the batch job that drives production; calls the API, no ML of its own
train/    training pipeline for the YOLO model            (train/README.md)
utils/    analysis and test tooling: run, draw, statistics
config/   gunicorn config for the container
docs/     technical description, server notes, diagrams   (docs/SERVER.md)
```

## Install

Python 3.12.

```sh
git clone https://github.com/kartverket/dokumentbestilling-smart-sladding.git
cd dokumentbestilling-smart-sladding
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins `paddlepaddle==3.3.1`. Which build you get depends on
the index you install from: plain PyPI gives the CPU wheel (Mac, laptops),
while the `Dockerfile` adds `--extra-index-url
https://www.paddlepaddle.org.cn/packages/stable/cu126/` to get the CUDA build
for the GPU server. Do not mix the two in one environment.

On the server, `source activate.sh` activates the venv, loads the `SLADD_*`
paths from `server.env` and enables the repo git hooks in one go.

## Models

**YOLO weights are not in the repo.** They live in the model store on the
server, `$SLADD_WEIGHTS` (see `server.env`), one directory per published model:

```
$SLADD_WEIGHTS/<name>/
  <name>.pt        the weights, named after the model
  modell.json      what it was trained on, with which parameters
  training/        results.csv, args.yaml, data.yaml, split_log.txt
```

`make -C $SLADD_TRAIN publiser` creates them after a training run. See
[train/README.md](train/README.md). `$SLADD_PRODWEIGHTS` points at the default
model: the one `./deploy.sh build` bakes in, and the one `run.py` uses when
`--yolo-weights` is omitted.

**PaddleOCR models** are pretrained weights from PaddlePaddle and are
downloaded by hand into `app/` (the code looks for them next to the source):

```sh
cd app
for m in PP-OCRv6_medium_det_infer PP-OCRv6_medium_rec_infer PP-LCNet_x1_0_doc_ori_infer; do
  curl -L -O "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/${m}.tar"
  tar -xvf "${m}.tar"
done
```

`PADDLE_MODEL_SET` in `app/config.py` selects the set (`"v6"` or `"v5"`). For
`"v5"` download `PP-OCRv5_server_det_infer` and `PP-OCRv5_server_rec_infer`
instead. The `Dockerfile` hardcodes the v6 downloads, so switching the config
without editing the Dockerfile builds an image without the models the code
asks for.

## Run the API locally

```sh
cd app
mkdir -p logs
ML_LOG_DIR=logs python app.py
```

`ML_LOG_DIR` matters: without it `app.py` falls back to the container path
`/data/ml_logs` and fails to start. Then, in another terminal:

```sh
curl http://localhost:5070/health

curl -X POST http://localhost:5070/model \
  -H "Content-Type: application/pdf" \
  --data-binary "@/path/to/document.pdf"
```

A test document ships with the repo. Use it to check the setup end to end:

```sh
cd utils
python run.py --folder . --select testdokument.pdf --csv --csv-out test_ut.csv \
  --png --png-dir visning_test --time
```

The PNGs land in `utils/visning_test/`. Measuring recall (`--truth`) needs a
labels CSV, which is not in the repo.

## API contract

### `GET /health`

Returns `{"health": "healthy"}` with status 200. The models load lazily on the
first `/model` call, so `/health` answers long before the service is warm.

### `POST /model`

Takes a PDF as raw bytes in the request body (`Content-Type: application/pdf`)
and returns the sladd boxes as JSON.

| Status | Body |
|--------|------|
| 200 | the list described below |
| 400 | empty request body |
| 500 | `{"error": "<description>"}` |

Optional query parameters: `elektronisk_tinglyst=true` skips YOLO entirely,
`rettsstiftelsestyper=SR_JOU,SE_SEK` (comma-separated grunnbok codes) enables
the per-document-type rule profiles, and `vlm=false` turns the VLM verifier off
for one request when the container runs with it on. `filrevisjonid=883421`
names the document in the application log; the batch job sends it with every
call.

#### Response format

A **flat list of boxes**, not grouped by page. Pages with no findings simply
contribute no entries, and an empty document returns `[]`.

```json
[
  {
    "page": 1,
    "x": 205.61, "y": 288.74, "width": 34.12, "height": 8.93,
    "kilde": "begge",
    "yolo_conf": 0.871,
    "paddle_rec_score": 0.99412
  },
  {
    "page": 3,
    "x": 118.2, "y": 512.44, "width": 31.7, "height": 9.41,
    "kilde": "yolo",
    "yolo_conf": 0.53,
    "trekk": { "har_tokens": 1, "n_siffer": 11, "n_bokstaver": 0 }
  }
]
```

| Field | Description |
|-------|-------------|
| `page` | page number, 1-based |
| `x`, `y` | top-left corner of the box, in PDF points, origin top-left |
| `width`, `height` | box size in PDF points |
| `kilde` | `paddle`, `yolo`, `begge` or `yolo_vertikal` |
| `yolo_conf` | YOLO detection confidence (0-1). Only when YOLO was involved |
| `paddle_rec_score` | how confidently the OCR read the characters. Only when Paddle was involved |
| `trekk` | what the OCR saw in and around the box. Only for `kilde: "yolo"` |

`yolo_conf` and `paddle_rec_score` are deliberately two fields. They measure
different things: a confident *read* says nothing about detection certainty,
and merging them would let a well-read Paddle box skip the geometry filters.
`trekk` does not affect the sladding; it exists so stricter filter variants can
be swept offline. See `app/box_features.py` for the full field list.

Coordinates refer to the page's original orientation; any rotation applied
during analysis has already been undone.

## Test tooling

All three run from `utils/`. Only the flags you need day to day are listed.
`--help` has the rest.

### `run.py`: run the model over a folder of PDFs

Same code path as a POST: bytes in, `run_model_on_pdf_bytes` out.

| Flag | Default | Description |
|------|---------|-------------|
| `--folder PATH` | `../uttrekk_3` | folder of PDFs |
| `--select FILE [FILE ...]` | none | only these files (filename/substring) |
| `--select-from-file FILE` | none | read IDs from a text file, one per line |
| `--count N` | `20` | number of files when `--select` is empty (`alle` = all) |
| `--yolo-weights FILE` | `$SLADD_PRODWEIGHTS` | weights to test |
| `--csv` / `--csv-out FILE` | off / `sladd_koordinater.csv` | write found boxes to CSV |
| `--truth` / `--truth-csv FILE` | off | measure recall against a labels CSV |
| `--png` / `--png-dir PATH` | off / `visning` | draw found + truth boxes to PNG |
| `--only-error` | off | PNGs only for pages with a miss or oversladding |
| `--result-dir PATH` | `.` | where the `result-*` directory is created |
| `--metadata-csv FILE` | none | rettsstiftelse types per document; enables rule profiles |
| `--without-postfilter` | off | skip all postfilters; baseline of what the rules contribute |
| `--vlm` | off | let a vision model re-read the boxes and remove the ones it rejects |
| `--vlm-model NAME` | `$SLADD_VLM_MODEL` | model name at the endpoint |
| `--vlm-url URL` | `$SLADD_VLM_URL` | OpenAI-compatible `/v1`, comma-separated for several backends |
| `--time` | off | timing per document |

Output directories that already exist abort the run; use `--proceed` to resume
or `--overwrite` to start over. OCR and YOLO results are cached per document
under `$SLADD_CACHE`, so rerunning the same model is nearly free.

```sh
python run.py --select 10000676.pdf --csv --truth --time
python run.py --count alle --truth --csv --png
```

On the server, `valider_full.sh` and `valider_yolo.sh` wrap this with the
standard paths. Start there rather than assembling flags by hand.

### `draw_from_csv.py`: visualise boxes as PNG

Draws the boxes from a finished CSV onto the PDF pages, without running the
model. Use it to flip through what was found. With `--truth` the labels are
drawn as green frames; `--only-oversladd` and `--only-miss` narrow it down to
the pages worth looking at.

```sh
python draw_from_csv.py --csv sladd_koordinater.csv --png-dir visning
python draw_from_csv.py --csv sladd_koordinater.csv --truth --only-miss
```

### `run_stats.py`: combined report for one run

```sh
python run_stats.py result-2026-07-14T08-15-20 --labels labels.csv
```

Writes `statistikk.txt` and `statistikk.png` into the result directory.

## VLM verifier

A vision model can re-read every proposed sladdeboks and remove the ones it is
sure hold no fødselsnummer. It is off unless you turn it on, and when it is on
it can only remove boxes. It never adds one and never moves one.

Three limits bound what a «nei» from the model can do.

- Only boxes with kilde `yolo` are judged, and only in documents that get no
  rule profile. Measured on uttrekk4, the removable boxes were 806 of 1027 for
  `yolo`, 0 of 203 for `begge` and 1 of 129 for `paddle`, so judging the other
  kilder costs GPU time and returns nothing.
- Before a «nei» is acted on, PaddleOCR's line and the model's own
  transcription go back through `find_fnr`. A valid eleven-digit run overrules
  the verdict, and so does a fnr ledetekst next to a five-digit run. The model
  reads better than it infers, so the code does the inferring.
- Anything that fails keeps the box: a timeout, an HTTP error, an answer that
  does not parse, an endpoint that is not running.

The endpoint is OpenAI-compatible `/v1/chat/completions`, so llama-server,
vLLM and LM Studio all work. `llama-server` is what runs on the GPU host,
because it has no registry client and no cloud backend to switch off. Each
judged box is one call to the model. `run.py --time` reports what it cost on
its own `vlm` line.

The crops hold real fødselsnumre, so what the model server may reach matters as
much as what it answers. `docs/VLM-ISOLATION.md` covers the sandbox and how to
check that it holds.

### Turning it on in prod

Set these in `server.env` and deploy again. `deploy.sh` passes them to the
container.

```sh
export SLADD_VLM=1
export SLADD_VLM_URL=http://<host>:8080/v1
export SLADD_VLM_MODEL=qwen3.8:27b
```

All three are needed. Without a URL and a model there is nothing to call, and
the pipeline runs as it did before. The container reaches the URL from the
inside, so `localhost` there is the container itself and not the host.

The container also runs with `HTTP_PROXY` set for traffic to the outside. VLM
calls skip it, since the endpoint is on the inside. Set `SLADD_VLM_PROXY` if
yours really does sit behind the proxy.

### Turning it on in run.py

```sh
python run.py --count alle --truth --csv --vlm --vlm-model qwen3.8:27b
```

The crops come from the page image, so `--vlm` renders every document even
when OCR and YOLO both come from cache. Judgements are cached under
`$SLADD_CACHE/vlm` in a directory named after the prompt version, so a rerun
with the same prompt and model reuses them. Edit the prompt and you get a new
directory and a full rejudge.

## Configuration

| File | Contents |
|------|----------|
| `app/config.py` | PDF DPI, YOLO thresholds, OCR parameters, filter rules, orientation |
| `utils/utils_config.py` | paths, evaluation threshold, visualisation colours |
| `server.env` | `SLADD_*` paths on the GPU server, and the VLM verifier settings |
| `.env` | deploy state for one machine, which tag runs where (see `.env.example`) |

## CSV formats

### Box CSV (`run.py --csv`, read back by `draw_from_csv.py`)

`navn, side, bilde_bredde, bilde_hoyde, x0, y0, x1, y1, kilde, yolo_conf,
paddle_rec_score` followed by one column per feature in `FEATURE_FIELDS`
(`app/config.py`). Coordinates are pixels here, not points. Feature columns are
empty for every `kilde` other than `yolo`.

### Labels CSV (fasit from the existing solution)

Column names come from the database export and are not ours to rename:
`fil_revisjon_id` (the number in the PDF filename), `sidetall`, `x`, `y`,
`width`, `height` (PDF points), `type`, `ml_generated`, `ml_status`
(`ACCEPTED`/`REJECTED`; rejected rows are skipped).

A labels file covers a whole uttrekk, so a document with no rows has been
reviewed and found to contain no fnr. Predictions on it are real oversladdinger.

### Details CSV (`run.py --truth`, read by `run_stats.py`)

One row per truth box, written to `detaljer.csv` in the result directory:
`fil, side, fasit_nr, type, dekning_pst, resultat` (`TRUFFET`/`MANGLER`),
`kilde, conf, fasit_x0/y0/x1/y1`. Alongside it, `sammendrag.csv` holds the
overall recall and the per-type breakdown.

## Production (Docker)

Production is a Docker container on the GPU server. **Port 5071 is production**,
and nothing else is.

An image is built once and gets an immutable tag
(`<date>-<commit>-<model>`, e.g. `20260820-6d7e6820-yolo-yearly-10000-docs`).
After that only *which* tag runs where moves. Prod never builds, so rollback is
pointing back at a tag that has already run.

Images are stored **only locally on the server**. There is no registry. So
`docker image prune -a` destroys the ability to roll back, and a new machine
has to rebuild everything.

| Role | Port | Container | Tag from |
|------|------|-----------|----------|
| Prod | 5071 | `smsl-prod` | `PROD_TAG` |
| Test | 5072 | `smsl-test` | `TEST_TAG` |

`deploy.sh` does all the work. First time on a new machine:

```sh
cp .env.example .env
source activate.sh          # loads server.env, which points at the model store
ls $SLADD_WEIGHTS           # models available to build in
```

### build → test → promote

```sh
./deploy.sh build                                            # model from $SLADD_PRODWEIGHTS
./deploy.sh build weights=$SLADD_WEIGHTS/uttrekk_4_jou/uttrekk_4_jou.pt   # another model

./deploy.sh test 20260820-6d7e6820-yolo-yearly-10000-docs
curl http://localhost:5072/health

./deploy.sh promote 20260820-6d7e6820-yolo-yearly-10000-docs # same bits that were tested
./deploy.sh stop test                                        # release the GPU memory
```

`promote` requires the tag explicitly, asks for confirmation, and **rolls back
automatically** if `/health` does not answer. A tag built on uncommitted changes
gets a `-dirty` suffix and is refused.

### Other commands

```sh
./deploy.sh status            # what runs where, and is it healthy
./deploy.sh versions          # locally built tags with model, newest first
./deploy.sh rollback          # back to the previous prod tag
./deploy.sh start|stop prod|test
./deploy.sh logs prod|test
./deploy.sh prune             # delete old images, cannot be undone
```

`start` and `stop` never change version; they bring the tag already in `.env`
down and up. Deploy history lives in `.deploy-historikk` and is what `rollback`
reads.

### Worth knowing

- Prod and test share the GPU. Run `./deploy.sh stop test` when you are done.
- Which model is baked in is always an explicit choice. Without one, `build`
  refuses. An image without weights starts fine and fails at the first
  `/model` call.
- The weights live outside the repo, so the commit alone does not say what an
  image contains. That is why the model name is in the tag, and model name +
  sha256 are labels on the image.
- `deploy.sh build` stages the chosen model in `.byggvekter/` because Docker
  can only copy from the build context. `docker build .` directly no longer
  works; go through `deploy.sh`.
- Confirm the GPU is actually in use:

```sh
./deploy.sh logs prod | grep -i "GPU available"   # -> True on gpu
```

### Logs

Three streams through the same rotating handler: zipped at midnight, oldest zip
dropped when the history is full.

| Log | File | Source |
|-----|------|--------|
| Application | `app.log` | `app/app.py` |
| Access | `gunicorn_access_prod.log` | `config/gunicorn_config_prod.py` |
| Gunicorn errors | `gunicorn_error_prod.log` | same |

`server.env` sets `SLADD_LOGS` (log root on the host) and `SLADD_LOG_DAYS`
(days of history); `deploy.sh` passes them to compose as `LOG_ROOT` and
`LOG_BACKUP_DAYS`. Under the log root: `gunicorn_logs/` and `ml_logs/` for
prod, `gunicorn_logs_test/` and `ml_logs_test/` for test. Test writes to its own
directories on purpose. Sharing them would have two rotation handlers fighting
over the same file at midnight. The `_prod` suffix in the filenames is
historical; it is kept so the existing log history on the server does not split
into two series.

## License

[MIT](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report vulnerabilities via
[SECURITY.md](.github/SECURITY.md), not in a public issue.
