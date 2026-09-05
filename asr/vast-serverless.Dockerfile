# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 python:3.11-slim-bookworm@sha256:2fc9207f64226cb05ac317cee0bab6fa55a9ea311ce5a086baddd4b4a83c2d3c AS deps

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1
RUN python -m venv "$VIRTUAL_ENV"
COPY locks/parakeet-v3.requirements.txt /locks/parakeet-v3.requirements.txt
COPY vendor/wheelhouses/parakeet-v3/ /wheels/
COPY asr/split_venv_layers.py /split_venv_layers.py
RUN pip install --no-deps --no-index --find-links /wheels --require-hashes -r /locks/parakeet-v3.requirements.txt \
 && python -I -c 'import nemo, torch, vastai, cuda.bindings; from nemo.collections.asr.models import ASRModel' \
 && du -sm /opt/venv/lib/python3.11/site-packages/* | sort -rn | head -40 \
 && python /split_venv_layers.py /opt/venv /layers 320 24

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
COPY --from=deps --link /layers/venv-00/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-01/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-02/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-03/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-04/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-05/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-06/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-07/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-08/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-09/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-10/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-11/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-12/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-13/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-14/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-15/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-16/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-17/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-18/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-19/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-20/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-21/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-22/opt/venv/ /opt/venv/
COPY --from=deps --link /layers/venv-23/opt/venv/ /opt/venv/
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo.part.00 /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.00
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo.part.01 /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.01
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo.part.02 /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.02
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo.part.03 /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.03
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo.part.04 /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.04
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo.part.05 /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.05
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo.part.06 /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.06
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo.part.07 /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.07
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo.part.08 /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.08
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo.part.09 /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.09
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo.part.10 /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.10
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo.part.11 /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.11
COPY --chmod=0444 vendor/models/parakeet-tdt-0.6b-v3.nemo.part.12 /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.12
COPY asr/offline_entrypoint.py asr/vast_adapter.py asr/server.py asr/parakeet_pool.py /workspace/
COPY asr/vast-pyworker.py /workspace/vast-pyworker/worker.py
COPY asr/production_vast_batch.py asr/drain_worker.py asr/vast_adapter.py asr/offline_entrypoint.py asr/vast_failure_guard.py /workspace/vast-pyworker/
COPY asr/fixtures/audio/2086-149220-0033.wav /workspace/fixtures/2086-149220-0033.wav
COPY --chmod=0755 asr/vast-entrypoint.sh /workspace/vast-entrypoint.sh
RUN test "$(cat /workspace/models/parts/parakeet-tdt-0.6b-v3.nemo.part.* | sha256sum | awk '{print $1}')" = 3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d
USER root
ENTRYPOINT ["/workspace/vast-entrypoint.sh"]
