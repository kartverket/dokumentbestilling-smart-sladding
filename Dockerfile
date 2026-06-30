FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    poppler-utils tesseract-ocr \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && apt-get remove -y python3-blinker \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
COPY config/ config/

EXPOSE 8080
CMD gunicorn --config config/gunicorn_config_${MODE}.py -b 0.0.0.0:8080 app:app
