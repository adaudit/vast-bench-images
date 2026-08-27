#!/bin/sh
set -eu

log=/workspace/parakeet-server.log
deadline=$(( $(date +%s) + "${STARTUP_TIMEOUT_SECONDS:-300}" ))
: > "$log"
python3 /workspace/server.py >>"$log" 2>&1 &

until curl --fail --silent --show-error http://127.0.0.1:8080/healthz >/dev/null; do
  [ "$(date +%s)" -lt "$deadline" ] || { tail -n 50 "$log" >&2; exit 1; }
  sleep 1
done
echo PARAKEET_SERVER_READY >>"$log"
exec python3 -I /workspace/vast-pyworker/worker.py
