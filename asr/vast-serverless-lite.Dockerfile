# syntax=docker/dockerfile:1.7
# The cu124 PyTorch wheels carry the exact CUDA/cuDNN user-space runtime; this
# avoids inheriting the 8.94 GB general-purpose PyTorch image.
FROM python:3.11-slim-bookworm@sha256:2fc9207f64226cb05ac317cee0bab6fa55a9ea311ce5a086baddd4b4a83c2d3c AS deps

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1
RUN python -m venv "$VIRTUAL_ENV"
COPY locks/parakeet-v3.requirements.txt /locks/parakeet-v3.requirements.txt
COPY locks/parakeet-v3.bootstrap.requirements.txt /locks/parakeet-v3.bootstrap.requirements.txt
COPY vendor/wheelhouses/parakeet-v3/ /wheels/
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1+cu124 torchaudio==2.5.1+cu124 \
 && pip install --no-deps --no-index --find-links /wheels --require-hashes -r /locks/parakeet-v3.requirements.txt \
 && pip install --no-deps --no-index --find-links /wheels --require-hashes -r /locks/parakeet-v3.bootstrap.requirements.txt \
 && pip check

FROM python:3.11-slim-bookworm@sha256:2fc9207f64226cb05ac317cee0bab6fa55a9ea311ce5a086baddd4b4a83c2d3c
LABEL io.adaudit.asr.model="nvidia/parakeet-tdt-0.6b-v3" \
      io.adaudit.asr.model-revision="541d1f99c6b0c3cd0b11a95167540bb8edefd82b" \
      io.adaudit.asr.cuda="cu124"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_INDEX=1 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libsndfile1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY --from=deps /opt/venv /opt/venv
COPY vendor/models/parakeet-tdt-0.6b-v3.nemo /workspace/models/parakeet-tdt-0.6b-v3.nemo
COPY asr/offline_entrypoint.py asr/vast_adapter.py asr/server.py /workspace/
COPY asr/vast-pyworker.py /workspace/vast-pyworker/worker.py
COPY asr/vast-entrypoint.sh /workspace/vast-entrypoint.sh
RUN python -c "import hashlib, pathlib; p=pathlib.Path('/workspace/models/parakeet-tdt-0.6b-v3.nemo'); assert p.stat().st_size == 2509332480; assert hashlib.sha256(p.read_bytes()).hexdigest() == '3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d'" \
 && chmod 0444 /workspace/models/parakeet-tdt-0.6b-v3.nemo \
 && chmod +x /workspace/vast-entrypoint.sh
ENTRYPOINT ["/workspace/vast-entrypoint.sh"]
