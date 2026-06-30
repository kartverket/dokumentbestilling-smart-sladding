FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.14 python3.14-venv \
    poppler-utils tesseract-ocr \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && apt-get remove -y python3-blinker \
    && rm -rf /var/lib/apt/lists/*

RUN python3.14 -m ensurepip && \
    ln -sf /usr/bin/python3.14 /usr/bin/python && \
    ln -sf /usr/bin/python3.14 /usr/bin/python3 && \
    python3.14 -m pip install --upgrade pip

WORKDIR /app

RUN python3.14 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu132

COPY requirements.txt .
RUN python3.14 -m pip install --no-cache-dir -r requirements.txt

COPY app/ .
COPY config/ config/

#RUN useradd -r -s /bin/false appuser && chown -R appuser:appuser /app
#USER appuser

EXPOSE 8080
CMD gunicorn --config config/gunicorn_config_${MODE}.py -b 0.0.0.0:8080 app:app
