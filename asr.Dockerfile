# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/models \
    HUGGINGFACE_HUB_CACHE=/workspace/models/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

COPY asr/constraints.txt /tmp/constraints.txt
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ffmpeg libsndfile1 pybind11-dev && \
    python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 -c /tmp/constraints.txt \
      'nemo_toolkit[asr]==2.2.1' whisperx==3.3.1 faster-whisper==1.1.0 huggingface_hub==0.27.1 hf_transfer==0.1.9 runpod==1.11.0 'cuda-python>=12.3' 'nvidia-cudnn-cu12==8.9.7.29' && \
    ln -s /opt/conda/lib/python3*/site-packages/nvidia/cudnn/lib/libcudnn*.so.8 /usr/lib/ 2>/dev/null; \
    python3 -c "import ctypes; ctypes.CDLL('libcudnn_ops_infer.so.8')" && \
    apt-get purge -y --auto-remove build-essential pybind11-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY asr/worker.py asr/rp_handler.py ./
COPY asr/bake_models.py ./
# No model weights in the image: docker pulls of multi-GB layers are the slow,
# flaky part of a cold start on marketplace hosts. The worker fetches Parakeet
# from HF's CDN at startup via hf_transfer (~1 min, persisted on host disk);
# whisper fallback lanes lazy-download only when they actually fire.
RUN python3 bake_models.py finalize && \
    curl -fsSL --retry 3 -o /tmp/fixture.mp3 \
      https://archive.org/download/world_set_free_0907_librivox/the_world_set_free_00_wells.mp3 && \
    ffmpeg -y -i /tmp/fixture.mp3 -t 3000 -ac 1 -ar 16000 -c:a pcm_s16le /opt/fixture-50min.wav && \
    rm /tmp/fixture.mp3 && \
    test -f model-pins.json && test -f .models_ready && \
    test -s /opt/fixture-50min.wav && \
    python3 -c 'import torch; from nemo.collections.asr.models import ASRModel; print(torch.__version__)'

ENV WORKSPACE_ROOT=/workspace \
    PARAKEET_INSTANCES=3
CMD ["python3", "rp_handler.py"]
