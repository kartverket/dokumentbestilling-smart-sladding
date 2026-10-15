# The batch job

`skip_job.py` is what drives production. It walks the unprocessed documents,
sends each PDF to the model API and writes the proposed sladdinger back to the
database. It contains no machine learning of its own, so it needs no GPU and no
model weights.

## Dependencies

The job installs `requests` and nothing else. That is deliberate: it runs
somewhere other than the model server, and pulling in PaddleOCR or ultralytics
to make an HTTP call would be a large install for no gain.

```sh
cd job
pip3 install -r requirements-skip-job.txt
python3 skip_job.py
```

## Files

`skip_job.py` is the job itself. `pdf_utils.py` downloads a PDF and recognises
a skjermet document, which is skipped rather than treated as an error.
`url_utils.py` turns the three environment variables below into base URLs.

## Environment variables

| Variable | Default | Points at |
|---|---|---|
| `DATABASE_URL` | `http://localhost:8000/` | The document queue and the labels |
| `API_URL` | `http://localhost:8080/` | The document API that serves the PDFs |
| `MODEL_URL` | `http://localhost:5070/` | The model API |

The defaults are for local development. Port 5070 is `app.py` run by hand;
production is the container on port 5071.

## What one run does

The job reads the queue once, then handles each document on its own: read the
document status, download the PDF, post it to the model, store the result. The
status is read again even though the queue just returned the document, because
a person can pick it up in between. Anything other than `KLAR_FOR_BEHANDLING`
is skipped, and so is a skjermet document.

Nothing is retried. A failed document is logged and left unprocessed, so the
next run picks it up again. The model call has a 600 second timeout, which is
generous because large scanned documents are slow.

## What the job sends to the model

The PDF goes in the request body as raw bytes. Three query parameters go with
it, and each one changes what comes back.

`elektronisk_tinglyst=true` turns YOLO off for that document, leaving the OCR
track to work alone.

`rettsstiftelsestyper` is the document's grunnbok codes, comma separated. They
enable the per-document-type rule profiles described in
[docs/TEKNISK.md](../docs/TEKNISK.md). A document that arrives without codes
gets the global behaviour, so missing metadata costs precision and never
recall. The job logs a warning when a document has none.

`filrevisjonid` names the document in the model's application log. Without it a
log line cannot be traced back to a document.

## What comes back

A flat list of boxes in PDF points, described under "API contract" in the
[README](../README.md). The job rewrites each one into a label row with
`mlGenerated: true` and stores the lot with `mlFerdigBehandlet=true`.

`mlGenerated` is what lets a person tell a machine suggestion from a manual
sladding and override it. `mlFerdigBehandlet` says the model has run on the
document. It says nothing about whether a person has reviewed the result.
