FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies including Tesseract, Poppler, and Norwegian language package
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        libleptonica-dev \
        poppler-utils \
        build-essential \
        pkg-config \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /smart_sladding_ml

# Copy your application code into the container
COPY /app /smart_sladding_ml

COPY requirements.txt /smart_sladding_ml/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "/smart_sladding_ml/skip_job.py"]
