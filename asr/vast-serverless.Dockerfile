# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 python:3.11-slim-bookworm@sha256:2fc9207f64226cb05ac317cee0bab6fa55a9ea311ce5a086baddd4b4a83c2d3c AS deps

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1
RUN python -m venv "$VIRTUAL_ENV"
COPY locks/parakeet-v3.requirements.txt /locks/parakeet-v3.requirements.txt
COPY vendor/wheelhouses/parakeet-v3/ /wheels/
RUN pip install --no-deps --no-index --find-links /wheels --require-hashes -r /locks/parakeet-v3.requirements.txt \
 && pip check \
 && python -I -c "import nemo, torch, vastai"

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
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo /workspace/models/parakeet-tdt-0.6b-v3.nemo
COPY asr/offline_entrypoint.py asr/vast_adapter.py asr/server.py asr/parakeet_pool.py /workspace/
COPY asr/vast-pyworker.py /workspace/vast-pyworker/worker.py
COPY asr/production_vast_batch.py asr/drain_worker.py asr/vast_adapter.py asr/offline_entrypoint.py asr/vast_failure_guard.py /workspace/vast-pyworker/
COPY asr/fixtures/audio/2086-149220-0033.wav /workspace/fixtures/2086-149220-0033.wav
COPY --chmod=0755 asr/vast-entrypoint.sh /workspace/vast-entrypoint.sh
RUN python -c "import hashlib, pathlib; p=pathlib.Path('/workspace/models/parakeet-tdt-0.6b-v3.nemo'); assert p.stat().st_size == 2509332480; assert hashlib.sha256(p.read_bytes()).hexdigest() == '3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d'"
USER root
ENTRYPOINT ["/workspace/vast-entrypoint.sh"]
