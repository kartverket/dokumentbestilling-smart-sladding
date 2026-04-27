# Job - Lightweight Document Processing Job

This folder contains a lightweight job (`skip_job.py`) that processes documents by calling the external model API.

## Purpose

The skip_job fetches unprocessed documents from the database, sends them to the model API for processing, and stores the results back in the database.

## Dependencies

This job has **minimal dependencies**:
- `requests` - For HTTP calls to APIs

See `../requirements-skip-job.txt` for the exact versions.

```sh
cd job
pip3 install -r requirements-skip-job.txt
python3 skip_job.py
```
## Files

- `skip_job.py` - Main job script that processes documents
- `pdf_utils.py` - Lightweight utilities for downloading PDFs
- `url_utils.py` - URL configuration helpers

## Environment Variables

- `API_URL` - Base URL for the document API
- `DATABASE_URL` - Base URL for the database API
- `MODEL_URL` - Base URL for the model server
