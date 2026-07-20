FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    poppler-utils tesseract-ocr \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    curl \
    && apt-get remove -y python3-blinker \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app

ARG HTTP_PROXY=http://159.162.48.7:3128

RUN set -eux; \
    for model in \
        PP-LCNet_x1_0_doc_ori_infer \
        PP-OCRv6_medium_det_infer \
        PP-OCRv6_medium_rec_infer \
    ; do \
        curl -x "$HTTP_PROXY" -L -O "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/${model}.tar"; \
        tar -xvf "${model}.tar"; \
        rm "${model}.tar"; \
    done

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/

COPY app/ .
COPY config/ config/

RUN mkdir -p /data/ml_logs

EXPOSE 8080
CMD gunicorn --config config/gunicorn_config_${MODE}.py -b 0.0.0.0:8080 app:app