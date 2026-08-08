# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/models \
    HUGGINGFACE_HUB_CACHE=/workspace/models/hub \
    NEMO_CACHE_DIR=/workspace/models/nemo \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ffmpeg libsndfile1 python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.1+cu128 torchaudio==2.7.1+cu128 && \
    python3 -m pip install 'nemo_toolkit[asr]==2.5.3' runpod==1.11.0

WORKDIR /workspace
COPY diar/requirements.txt diar/constraints.txt ./
RUN python3 -m pip install --no-deps -r requirements.txt && \
    python3 -m pip install --no-deps -r constraints.txt
COPY diar/worker.py diar/rp_handler.py diar/bake_models.py ./
RUN --mount=type=secret,id=hf_token,required=true \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python3 bake_models.py && \
    curl -fsSL --retry 3 \
      -o /tmp/fixture.mp3 \
      https://archive.org/download/world_set_free_0907_librivox/the_world_set_free_00_wells.mp3 && \
    ffmpeg -y -i /tmp/fixture.mp3 -t 3000 -ac 1 -ar 16000 -c:a pcm_s16le /opt/fixture-50min.wav && \
    rm /tmp/fixture.mp3 && \
    test -s /opt/fixture-50min.wav && \
    python3 -c 'import torch; from nemo.collections.asr.models import SortformerEncLabelModel; from pyannote.audio import Pipeline; print(torch.__version__)'

ENV WORKSPACE_ROOT=/workspace \
    DIAR_CONCURRENCY=8
CMD ["python3", "rp_handler.py"]
