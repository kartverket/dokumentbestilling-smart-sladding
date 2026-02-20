# Multi-stage build for smaller final image
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /smart_sladding_ml

# Install build dependencies in builder stage
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libtesseract-dev \
        libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /smart_sladding_ml/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Final stage - runtime only
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set environment variables to use /tmp for caches (K8s restricted environment)
ENV HOME=/tmp
ENV TORCH_HOME=/tmp/.torch
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/.torch/inductor_cache
ENV TRANSFORMERS_CACHE=/tmp/.cache/huggingface
ENV HF_HOME=/tmp/.cache/huggingface
ENV EASYOCR_MODULE_PATH=/tmp/.EasyOCR
ENV MPLCONFIGDIR=/tmp/.matplotlib
ENV USER=appuser

WORKDIR /smart_sladding_ml

# Install only runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with specific UID to avoid getpwuid errors
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    mkdir -p /tmp/.torch /tmp/.cache /tmp/.EasyOCR /tmp/.matplotlib && \
    chown -R appuser:appuser /tmp

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY /app /smart_sladding_ml

# Switch to non-root user
USER appuser

CMD ["python", "-u", "/smart_sladding_ml/skip_job.py"]
