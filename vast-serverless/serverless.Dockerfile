# Test A: Vast serverless image. Extends the proven ASR worker image (all v10-v12
# Parakeet fixes baked in) with an HTTP model server + Vast PyWorker proxy.
# Base already ships fastapi/uvicorn/pydantic (conda py3.11) and has no git, so
# we vendor the pyworker scaffold instead of cloning it.
# syntax=docker/dockerfile:1.7
FROM ghcr.io/adaudit/vast-bench-asr:v12

# Vendored Vast pyworker scaffold: start_server.sh does the TLS cert signing +
# run.vast.ai autoscaler registration, then runs $SERVER_DIR/worker.py (ours).
RUN mkdir -p /workspace/vast-pyworker
COPY vast-serverless/start_server.sh /workspace/vast-pyworker/start_server.sh
COPY vast-serverless/pyworker-requirements.txt /workspace/vast-pyworker/requirements.txt
COPY vast-serverless/pyworker.py /workspace/vast-pyworker/worker.py

COPY vast-serverless/server.py /workspace/server.py
COPY vast-serverless/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh /workspace/vast-pyworker/start_server.sh

ENV MODEL_SERVER_PORT=18100 \
    MODEL_LOG=/workspace/model-server.log \
    WORKSPACE_DIR=/workspace

CMD ["bash", "/workspace/entrypoint.sh"]
