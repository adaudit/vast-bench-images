# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/models \
    HUGGINGFACE_HUB_CACHE=/workspace/models/hub \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl ffmpeg libsndfile1 python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.5.1+cu124 torchaudio==2.5.1+cu124 && \
    python3 -m pip install \
      'nemo_toolkit[asr]==2.2.1' \
      huggingface_hub==0.27.1 \
      runpod==1.11.0 && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY asr/worker.py asr/rp_handler.py ./
COPY asr/bake_models.py ./
RUN python3 bake_models.py && \
    curl -fsSL --retry 3 -o /tmp/fixture.mp3 \
      https://archive.org/download/world_set_free_0907_librivox/the_world_set_free_00_wells.mp3 && \
    ffmpeg -y -i /tmp/fixture.mp3 -t 3000 -ac 1 -ar 16000 -c:a pcm_s16le /opt/fixture-50min.wav && \
    rm /tmp/fixture.mp3 && \
    test -f model-pins.json && test -f .models_ready && test -d models/parakeet && \
    test -s /opt/fixture-50min.wav && \
    python3 -c 'import torch; from nemo.collections.asr.models import ASRModel; print(torch.__version__)'

ENV WORKSPACE_ROOT=/workspace \
    PARAKEET_INSTANCES=3
CMD ["python3", "rp_handler.py"]
