# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 python:3.11-slim-bookworm@sha256:2fc9207f64226cb05ac317cee0bab6fa55a9ea311ce5a086baddd4b4a83c2d3c AS deps

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1
RUN python -m venv "$VIRTUAL_ENV"
COPY locks/parakeet-v3.requirements.txt /locks/parakeet-v3.requirements.txt
COPY locks/parakeet-v3.bootstrap.requirements.txt /locks/parakeet-v3.bootstrap.requirements.txt
COPY vendor/wheelhouses/parakeet-v3/ /wheels/
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1+cu124 torchaudio==2.5.1+cu124 \
 && pip install --no-deps --no-index --find-links /wheels --require-hashes -r /locks/parakeet-v3.requirements.txt \
 && pip install --no-deps --no-index --find-links /wheels --require-hashes -r /locks/parakeet-v3.bootstrap.requirements.txt \
 && PIP_NO_INDEX=0 pip install --no-cache-dir vastai-sdk==1.5.5 uv==0.8.15 \
 && pip check \
 && python -I -c "import vastai"

FROM deps AS model
WORKDIR /workspace
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo /workspace/models/parakeet-tdt-0.6b-v3.nemo
COPY asr/extract_model.py asr/offline_entrypoint.py /workspace/
ARG ASR_VERIFY_NEMO_CONNECTOR=1
# Verify the original model hash, extract exactly once, hash every extracted
# file, and assert the pinned NeMo connector supports model_extracted_dir.
RUN ASR_VERIFY_NEMO_CONNECTOR="$ASR_VERIFY_NEMO_CONNECTOR" python /workspace/extract_model.py

FROM --platform=linux/amd64 python:3.11-slim-bookworm@sha256:2fc9207f64226cb05ac317cee0bab6fa55a9ea311ce5a086baddd4b4a83c2d3c
LABEL io.adaudit.asr.model="nvidia/parakeet-tdt-0.6b-v3" \
      io.adaudit.asr.model-revision="541d1f99c6b0c3cd0b11a95167540bb8edefd82b" \
      io.adaudit.asr.cuda="cu124"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_INDEX=1 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PARAKEET_INSTANCES=3
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 libsndfile1 openssl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY --from=deps /opt/venv /opt/venv
COPY --from=model /workspace/models/parakeet-v3 /workspace/models/parakeet-v3
COPY asr/offline_entrypoint.py asr/vast_adapter.py asr/server.py asr/parakeet_pool.py /workspace/
COPY asr/startup.py asr/batch_runtime.py /workspace/
COPY asr/inference_runtime.py /workspace/
COPY asr/vast-pyworker.py /workspace/vast-pyworker/worker.py
COPY asr/production_vast_batch.py asr/drain_worker.py asr/vast_adapter.py asr/offline_entrypoint.py asr/vast_failure_guard.py /workspace/vast-pyworker/
COPY asr/fixtures/audio/2086-149220-0033.wav /workspace/fixtures/2086-149220-0033.wav
COPY --chmod=0755 asr/vast-entrypoint.sh /workspace/vast-entrypoint.sh
USER root
ENTRYPOINT ["/workspace/vast-entrypoint.sh"]
