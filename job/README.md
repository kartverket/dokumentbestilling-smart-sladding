# Job - Lightweight Document Processing Job

This folder contains a lightweight job (`skip_job.py`) that processes documents by calling the external model API.

## Purpose

The skip_job fetches unprocessed documents from the database, sends them to the model API for processing, and stores the results back in the database.

## Dependencies

This job has **minimal dependencies**:
- `requests` - For HTTP calls to APIs

See `../requirements-skip-job.txt` for the exact versions.

## Files

- `skip_job.py` - Main job script that processes documents
- `pdf_utils.py` - Lightweight utilities for downloading PDFs
- `url_utils.py` - URL configuration helpers

## Docker

Build the lightweight container:
```bash
docker build -f Dockerfile.skip-job -t skip-job:latest .
```

## Why a Separate Folder?

The skip_job doesn't need any of the heavy ML/OCR dependencies (torch, easyocr, opencv, etc.). 

By separating it from the `app` folder, we:
1. **Reduce container size**: ~150MB vs ~3GB
2. **Faster builds**: No need to install heavy dependencies
3. **Faster startup**: Less code to load
4. **Better security**: Smaller attack surface
5. **Clearer separation**: Job vs model server

## Environment Variables

- `API_URL` - Base URL for the document API
- `DATABASE_URL` - Base URL for the database API
- `MODEL_URL` - Base URL for the model server
