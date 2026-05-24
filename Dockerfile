# CUDA 12.1 runtime for non-Blackwell GPUs (RTX 4090 / 3090 / A5000 / L4)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    PORT=8000 \
    MODEL_NAME=Marqo/marqo-fashionSigLIP \
    MIN_CONFIDENCE=0.38 \
    MIN_MARGIN=0.07

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app.py /app/app.py

EXPOSE 8000

# Health endpoint: GET /ping
CMD ["python", "-u", "/app/app.py"]
