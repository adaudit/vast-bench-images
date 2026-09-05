# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 python:3.11-slim-bookworm@sha256:2fc9207f64226cb05ac317cee0bab6fa55a9ea311ce5a086baddd4b4a83c2d3c AS deps

ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1
RUN python -m venv "$VIRTUAL_ENV"
COPY locks/diar-v1.requirements.txt /locks/diar-v1.requirements.txt
COPY vendor/wheelhouses/diar-v1/ /wheels/
COPY asr/split_venv_layers.py /split_venv_layers.py
RUN pip install --no-deps --no-index --find-links /wheels --require-hashes -r /locks/diar-v1.requirements.txt \
 && python -I -c 'import nemo, torch, vastai, cuda.bindings; from nemo.collections.asr.models import SortformerEncLabelModel' \
 && du -sm /opt/venv/lib/python3.11/site-packages/* | sort -rn | head -40 \
 && python /split_venv_layers.py /opt/venv /layers 320 24

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
COPY --chmod=0444 vendor/models/diar_streaming_sortformer_4spk-v2.1.nemo.part.00 /workspace/models/parts/diar_streaming_sortformer_4spk-v2.1.nemo.part.00
COPY --chmod=0444 vendor/models/diar_streaming_sortformer_4spk-v2.1.nemo.part.01 /workspace/models/parts/diar_streaming_sortformer_4spk-v2.1.nemo.part.01
COPY --chmod=0444 vendor/models/diar_streaming_sortformer_4spk-v2.1.nemo.part.02 /workspace/models/parts/diar_streaming_sortformer_4spk-v2.1.nemo.part.02
COPY diar/server.py /workspace/
COPY diar/vast-pyworker.py /workspace/vast-pyworker/worker.py
COPY asr/fixtures/audio/2086-149220-0033.wav /workspace/fixtures/2086-149220-0033.wav
COPY --chmod=0755 diar/vast-entrypoint.sh /workspace/vast-entrypoint.sh
RUN test "$(cat /workspace/models/parts/diar_streaming_sortformer_4spk-v2.1.nemo.part.* | sha256sum | awk '{print $1}')" = 8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8
USER root
ENTRYPOINT ["/workspace/vast-entrypoint.sh"]
