# CUDA 12.x, not 13: CUDA 13 dropped Volta (sm_70), which the server runs.
# Ubuntu 24.04 for Python 3.12; Pillow 13 needs 3.11 or newer.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    python3 python3-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 refuses pip installs into the system Python (PEP 668).
RUN python3 -m venv /opt/venv && pip install --no-cache-dir --upgrade pip

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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/

# We have to use 2.12.1+cu126 for it to work on the server with sm-70
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.12.1+cu126 torchvision==0.27.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir --force-reinstall --no-deps nvidia-cudnn-cu12==9.5.1.17

# paddlepaddle-gpu pins its own nvidia-* wheels, and the NCCL it leaves behind is
# missing symbols libtorch_cuda.so needs. Paddle uses NCCL only for distributed
# training, so torch's pin wins.
RUN NCCL=$(python3 -c "import importlib.metadata as m; print(next(r.split(';')[0].strip() for r in m.requires('torch') if r.startswith('nvidia-nccl')))") && \
    echo "pinning $NCCL" && \
    pip install --no-cache-dir --force-reinstall --no-deps "$NCCL"

# The torch step above has already replaced numpy with what the PyTorch index
# has. This sets the version we chose and tested.
COPY requirements-override.txt .
RUN pip install --no-cache-dir --no-deps -r requirements-override.txt

# opencv-contrib-python (paddlex) and opencv-python (ultralytics) both write
# cv2/, and the last one installed wins. Reinstall the pinned one last.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    "$(grep -o '^opencv-contrib-python==[^ #]*' requirements.txt)"

# The torch step reinstalls torch's dependencies from the PyTorch index, which
# can replace a pinned package without a word. Stop the build if it did. pip's
# version of an opencv package does not show which one owns cv2/, so cv2 is
# checked on import.
RUN python3 - <<'EOF'
import re
from importlib.metadata import PackageNotFoundError, version

import cv2

pins = {}
for path in ("requirements.txt", "requirements-override.txt"):
    for line in open(path):
        m = re.match(r"([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if m:
            pins[m.group(1)] = m.group(2)
wrong = []
for name, pinned in pins.items():
    try:
        installed = version(name)
    except PackageNotFoundError:
        installed = "missing"
    if installed.split("+")[0] != pinned:
        wrong.append(f"  {name}: pinned {pinned}, installed {installed}")
# The wheel's fourth number is the build, which cv2.__version__ leaves out.
cv2_pinned = ".".join(pins["opencv-contrib-python"].split(".")[:3])
if cv2.__version__ != cv2_pinned:
    wrong.append(f"  cv2: pinned {cv2_pinned} (opencv-contrib-python), imports {cv2.__version__}")
if wrong:
    raise SystemExit("Installed packages differ from their pins:\n" + "\n".join(wrong))
print(f"All {len(pins)} pins match")
EOF

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