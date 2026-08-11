# Test A: Vast serverless image. Extends the proven ASR worker image (all v10-v12
# Parakeet fixes baked in) with an HTTP model server + Vast PyWorker proxy.
# syntax=docker/dockerfile:1.7
FROM ghcr.io/adaudit/vast-bench-asr:v12

# HTTP model-server deps in the image python (same env as worker.py / NeMo).
RUN python3 -m pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29" "pydantic>=2"

# Pre-clone the Vast pyworker scaffold so boot doesn't pay a git clone, and drop
# our PyWorker config in as its top-level worker.py (start_server.sh runs that first).
RUN git clone --depth 1 https://github.com/vast-ai/pyworker /workspace/vast-pyworker

COPY vast-serverless/server.py /workspace/server.py
COPY vast-serverless/pyworker.py /workspace/vast-pyworker/worker.py
COPY vast-serverless/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENV MODEL_SERVER_PORT=18100 \
    MODEL_LOG=/workspace/model-server.log \
    WORKSPACE_DIR=/workspace

CMD ["bash", "/workspace/entrypoint.sh"]
