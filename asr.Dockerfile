# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/models \
    HUGGINGFACE_HUB_CACHE=/workspace/models/hub \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY asr/constraints.txt /tmp/constraints.txt
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ffmpeg libsndfile1 pybind11-dev && \
    python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 -c /tmp/constraints.txt \
      'nemo_toolkit[asr]==2.2.1' whisperx==3.3.1 faster-whisper==1.1.0 huggingface_hub==0.27.1 runpod==1.11.0 && \
    apt-get purge -y --auto-remove build-essential pybind11-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY asr/worker.py asr/rp_handler.py ./
COPY asr/bake_models.py ./
# One model per layer: a single 6.7 GB layer stalls docker pulls on hosts with
# flaky links to GHCR; per-model layers let pulls checkpoint progress.
RUN --mount=type=secret,id=hf_token,required=true \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python3 bake_models.py parakeet
# Whisper fallback weights are NOT baked: they serve only rare low-confidence
# retries, and ~4 GB off the image cuts every cold pull. The worker lazy-downloads
# them from HF on first fallback and caches on the container disk.
RUN python3 bake_models.py finalize && \
    curl -fsSL --retry 3 -o /tmp/fixture.mp3 \
      https://archive.org/download/world_set_free_0907_librivox/the_world_set_free_00_wells.mp3 && \
    ffmpeg -y -i /tmp/fixture.mp3 -t 3000 -ac 1 -ar 16000 -c:a pcm_s16le /opt/fixture-50min.wav && \
    rm /tmp/fixture.mp3 && \
    test -f model-pins.json && test -f .models_ready && \
    ls models/parakeet/*.nemo && \
    test -s /opt/fixture-50min.wav && \
    python3 -c 'import torch; from nemo.collections.asr.models import ASRModel; print(torch.__version__)'

ENV WORKSPACE_ROOT=/workspace \
    PARAKEET_INSTANCES=3
CMD ["python3", "rp_handler.py"]
