# Lightweight Dockerfile for skip_job.py
# This job only calls APIs, no ML/OCR processing needed
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /job

# Install only what we need for skip_job
COPY requirements-skip-job.txt /job/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-skip-job.txt

# Copy only the files needed for skip_job
COPY skip_job.py url_utils.py pdf_utils.py /job/

# Create non-root user
RUN useradd -r -u 1000 appuser
USER appuser

CMD ["python", "-u", "/job/skip_job.py"]
