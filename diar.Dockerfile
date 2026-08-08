# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/nemo:25.09.02

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/models \
    HUGGINGFACE_HUB_CACHE=/workspace/models/hub \
    NEMO_CACHE_DIR=/workspace/models/nemo

WORKDIR /workspace
COPY diar/requirements.txt diar/constraints.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends curl ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir "$(grep '^runpod' requirements.txt)" && \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps \
        'https://github.com/pytorch/audio/archive/refs/tags/v2.8.0.tar.gz' && \
    python3 -m pip install --no-cache-dir --no-deps "$(grep '^pyannote' requirements.txt)" && \
    python3 -m pip install --no-cache-dir --no-deps -r constraints.txt && \
    python3 -c 'import numpy, scipy, torch, torchaudio, torchmetrics, lightning, nemo, runpod; from nemo.collections.asr.models import SortformerEncLabelModel; from pyannote.audio import Pipeline; assert numpy.__version__.startswith("1."); print("dependency imports OK")'
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
