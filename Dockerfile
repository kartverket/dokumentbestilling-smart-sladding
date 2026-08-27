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

# Before requirements.txt, which names the same torch version. PyPI's default
# wheel is a CUDA 13 build, and CUDA 13 dropped sm_70, which is this card.
# Installed first, the cu126 build satisfies that pin and pip leaves it alone.
RUN pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 \
    --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
RUN pip install --no-cache-dir --force-reinstall --no-deps nvidia-cudnn-cu12==9.5.1.17

# paddlepaddle-gpu pins its own nvidia-* wheels, and the NCCL it leaves behind is
# missing symbols libtorch_cuda.so needs. Paddle uses NCCL only for distributed
# training, so torch's pin wins.
RUN NCCL=$(python3 -c "import importlib.metadata as m; print(next(r.split(';')[0].strip() for r in m.requires('torch') if r.startswith('nvidia-nccl')))") && \
    echo "pinning $NCCL" && \
    pip install --no-cache-dir --force-reinstall --no-deps "$NCCL"

# The weights live outside the repo (see SLADD_WEIGHTS in server.env) and
# docker only copies from the build context, so ./deploy.sh stages the chosen
# model in .byggvekter/ (modell.pt + modell.json) before the build and cleans
# up afterwards. Without that directory the build fails here, on purpose: an
# image without weights starts fine and fails first at /model.
#
# Own layer, before the code, on purpose. The weights are 51 MB and change
# rarely, the code is 64 kB and changes constantly, so every build reuses the
# weight layer instead of storing 51 MB per version.
COPY .byggvekter/ ./weights/

COPY config/ config/
COPY app/*.py .

RUN mkdir -p /data/ml_logs

EXPOSE 8080
CMD ["gunicorn", "--config", "config/gunicorn_config_prod.py", "-b", "0.0.0.0:8080", "app:app"]