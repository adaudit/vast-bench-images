# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 python:3.11-slim-bookworm@sha256:2fc9207f64226cb05ac317cee0bab6fa55a9ea311ce5a086baddd4b4a83c2d3c AS deps

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1
RUN python -m venv "$VIRTUAL_ENV"
COPY locks/diar-v1.requirements.txt /locks/diar-v1.requirements.txt
COPY vendor/wheelhouses/diar-v1/ /wheels/
RUN pip install --no-deps --no-index --find-links /wheels --require-hashes -r /locks/diar-v1.requirements.txt \
 && python -I -c 'import nemo, torch, vastai, cuda.bindings; from nemo.collections.asr.models import SortformerEncLabelModel'

FROM --platform=linux/amd64 python:3.11-slim-bookworm@sha256:2fc9207f64226cb05ac317cee0bab6fa55a9ea311ce5a086baddd4b4a83c2d3c
LABEL io.adaudit.diar.model="nvidia/diar_streaming_sortformer_4spk-v2.1" \
      io.adaudit.diar.model-revision="fafaab5faa1617a0ca52d38dd3dc4bd636800d3d" \
      io.adaudit.diar.cuda="cu124"
ENV DIAR_CONCURRENCY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_INDEX=1 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libgomp1 libsndfile1 openssl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY --from=deps /opt/venv /opt/venv
COPY --chmod=0444 vendor/models/diar_streaming_sortformer_4spk-v2.1.nemo /workspace/models/diar_streaming_sortformer_4spk-v2.1.nemo
COPY diar/server.py diar/bake_models.py /workspace/
COPY diar/vast-pyworker.py /workspace/vast-pyworker/worker.py
COPY asr/fixtures/audio/2086-149220-0033.wav /workspace/fixtures/2086-149220-0033.wav
COPY --chmod=0755 diar/vast-entrypoint.sh /workspace/vast-entrypoint.sh
RUN python /workspace/bake_models.py /workspace/models/diar_streaming_sortformer_4spk-v2.1.nemo
USER root
ENTRYPOINT ["/workspace/vast-entrypoint.sh"]
