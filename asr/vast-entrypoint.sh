#!/bin/sh
set -eu

log=/workspace/parakeet-server.log
startup_timeout=${STARTUP_TIMEOUT_SECONDS:-300}; startup_timeout=${startup_timeout#\"}; startup_timeout=${startup_timeout%\"}
deadline=$(( $(date +%s) + startup_timeout ))
report_addr="${REPORT_ADDR:-https://run.vast.ai}"

report_error_and_exit() {
  ERROR_MESSAGE=$1 REPORT_ADDR=$report_addr python3 -c '
import json, os
from urllib.request import Request, urlopen
try:
    request = Request(os.environ["REPORT_ADDR"].rstrip("/") + "/worker_status/", data=json.dumps({"id": int(os.environ.get("CONTAINER_ID", "0")), "mtoken": os.environ.get("MASTER_TOKEN", ""), "version": os.environ.get("PYWORKER_VERSION", "0"), "error_msg": os.environ["ERROR_MESSAGE"], "url": os.environ.get("URL", "")}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    urlopen(request, timeout=10).read()
except Exception:
    pass
'
  exit 1
}

[ -n "${CONTAINER_ID:-}" ] || report_error_and_exit "CONTAINER_ID must be set"
if [ "${USE_SSL:-true}" = true ]; then
  openssl req -newkey rsa:2048 -subj '/C=US/ST=CA/CN=pyworker.vast.ai/' -nodes -sha256 -keyout /etc/instance.key -out /etc/instance.csr || report_error_and_exit "failed to create certificate request"
  python3 -c '
import sys
from urllib.request import Request, urlopen
request = Request("https://console.vast.ai/api/v0/sign_cert/?instance_id=" + sys.argv[1], data=sys.stdin.buffer.read(), headers={"Content-Type": "application/octet-stream"}, method="POST")
sys.stdout.buffer.write(urlopen(request, timeout=30).read())
' "$CONTAINER_ID" < /etc/instance.csr > /etc/instance.crt || report_error_and_exit "failed to sign certificate"
fi

export REPORT_ADDR="$report_addr" WORKER_PORT USE_SSL UNSECURED
: > "$log"
( python3 /workspace/server.py 2>&1 | tee -a "$log" >&2 ) &

while :; do
  health=$(python3 -c '
import json
from urllib.error import HTTPError
from urllib.request import urlopen
try:
    response = urlopen("http://127.0.0.1:8080/healthz", timeout=5)
except HTTPError as error:
    response = error
try:
    print(json.load(response).get("status", ""))
except Exception:
    pass
' 2>/dev/null || true)
  [ "$health" != failed ] || { tail -n 50 "$log" >&2; exit 1; }
  [ "$health" = ready ] && break
  [ "$(date +%s)" -lt "$deadline" ] || { tail -n 50 "$log" >&2; exit 1; }
  sleep 1
done
echo PARAKEET_SERVER_READY >>"$log"
exec python3 -I /workspace/vast-pyworker/worker.py
