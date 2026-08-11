#!/bin/bash
# Test A serverless entrypoint: run our ASR model server in the background,
# then hand off to Vast's pyworker bootstrap (start_server.sh) which does the
# TLS cert signing + run.vast.ai autoscaler registration and launches our
# PyWorker proxy (vast-pyworker/worker.py -> our pyworker.py config).
set -e -o pipefail

export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
export MODEL_SERVER_PORT="${MODEL_SERVER_PORT:-18100}"
export MODEL_LOG="${MODEL_LOG:-/workspace/model-server.log}"

: > "$MODEL_LOG"
echo "starting ASR model server on 127.0.0.1:${MODEL_SERVER_PORT} (log: ${MODEL_LOG})"
cd /workspace
python3 -m uvicorn server:app --host 127.0.0.1 --port "$MODEL_SERVER_PORT" >>"$MODEL_LOG" 2>&1 &

echo "handing off to Vast pyworker bootstrap"
exec bash "$WORKSPACE_DIR/vast-pyworker/start_server.sh"
