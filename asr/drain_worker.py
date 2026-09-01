#!/usr/bin/env python3
"""One-lane, credential-minimal durable ASR drain client for the v3 image."""
import base64
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from vast_adapter import LANE, MODEL_ID, MODEL_REVISION, REQUEST_VERSION
except ModuleNotFoundError:
    from asr.vast_adapter import LANE, MODEL_ID, MODEL_REVISION, REQUEST_VERSION

ADMISSION_CAP = max(1, int(os.environ.get("ASR_DRAIN_ADMISSION_CAP", "2")))

class DrainError(RuntimeError): pass
class ContractBatchError(DrainError): pass

def request(base, token, path, payload):
    req = urllib.request.Request(base.rstrip("/")+path, data=json.dumps(payload, separators=(",", ":")).encode(), method="POST", headers={"Authorization":"Bearer "+token,"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        if error.code == 204: return 204, {}
        raise DrainError("control plane HTTP %d" % error.code) from error

def candidate_payload(claim):
    # The CPU producer must persist exact retained duration; GPU never guesses.
    duration = claim.get("safe_parameters", {}).get("audio_duration_seconds")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        raise DrainError("claimed post-CPU audio duration is required")
    with urllib.request.urlopen(claim["audio_url"], timeout=300) as source:
        audio = source.read(512 * 1024 * 1024 + 1)
    if len(audio) > 512 * 1024 * 1024: raise DrainError("audio exceeds bounded request limit")
    return {"request_version":REQUEST_VERSION,"lane":LANE,"model_id":MODEL_ID,"model_revision":MODEL_REVISION,"audio_filename":"claimed.wav","audio_duration_seconds":duration,"audio_base64":base64.b64encode(audio).decode()}

def candidates(claims):
    req = urllib.request.Request("http://127.0.0.1:8080/transcribe-batch", data=json.dumps({"requests":[candidate_payload(claim) for claim in claims]}, separators=(",", ":")).encode(), method="POST", headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as response: return json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code == 400: raise ContractBatchError("GPU rejected batch contract") from error
        raise

def complete(config, lease, result, job_schema_version):
    body = json.dumps(result, separators=(",", ":")).encode(); digest = hashlib.sha256(body).hexdigest()
    _, intent = request(config["base"], config["token"], "/internal/call-pipeline/asr/publish-intent", {"lease":lease,"schema_version":result["schema_version"],"artifact_kind":"asr_candidate","media_type":"application/vnd.adaudit.asr-candidate-v1+json","sha256":digest,"size_bytes":len(body)})
    put = urllib.request.Request(intent["url"], data=body, method="PUT", headers=intent.get("required_headers", {}))
    with urllib.request.urlopen(put, timeout=300) as response: version = response.headers.get("x-amz-version-id") or response.headers.get("X-Bz-File-Id")
    if not version: raise DrainError("candidate upload omitted immutable version")
    request(config["base"], config["token"], "/internal/call-pipeline/asr/complete", {"lease":lease,"schema_version":job_schema_version,"attempt":lease["Attempt"],"result":{"asr_candidate_artifact":{"object_key":intent["object_key"],"version_id":version,"upload_generation":intent["upload_generation"],"media_type":"application/vnd.adaudit.asr-candidate-v1+json","sha256":digest,"size_bytes":len(body)}},"idempotency_key":digest})

def run_once(config):
    # Claim at most one lease at a time in the sole inference lane. The fixed
    # admission cap is intentionally visible for future batch-server support;
    # no unbounded local/provider queue is ever constructed.
    status, claim = request(config["base"], config["token"], "/internal/call-pipeline/asr/claim", {"worker_id":config["worker_id"]})
    if status == 204: return False
    lease = claim["lease"]
    try:
        _, beat = request(config["base"], config["token"], "/internal/call-pipeline/asr/heartbeat", {"lease":lease}); lease = beat["lease"]
        complete(config, lease, candidates([claim])[0], claim.get("job_schema_version", lease.get("SchemaVersion", "asr-v1")))
        return True
    except Exception as error:
        request(config["base"], config["token"], "/internal/call-pipeline/asr/fail", {"lease":lease,"retryable":True,"failure":{"code":"v3_drain_error","detail":str(error)[:256]}})
        return True

def run_window(config, cap=ADMISSION_CAP):
    """Claim one bounded window and complete each lease independently."""
    claimed = []
    for _ in range(max(1, cap)):
        status, claim = request(config["base"], config["token"], "/internal/call-pipeline/asr/claim", {"worker_id":config["worker_id"]})
        if status == 204: break
        _, beat = request(config["base"], config["token"], "/internal/call-pipeline/asr/heartbeat", {"lease":claim["lease"]})
        claim["lease"] = beat["lease"]; claimed.append(claim)
    if not claimed: return False
    def process(items):
        try:
            results = candidates(items)
            for claim, result in zip(items, results):
                try: complete(config, claim["lease"], result, claim.get("job_schema_version", claim["lease"].get("SchemaVersion", "asr-v1")))
                except Exception as error: request(config["base"], config["token"], "/internal/call-pipeline/asr/fail", {"lease":claim["lease"],"retryable":True,"failure":{"code":"v3_drain_error","detail":str(error)[:256]}})
        except Exception as error:
            if ("out of memory" in str(error).lower() or isinstance(error, ContractBatchError)) and len(items) > 1:
                midpoint = max(1, len(items)//2); process(items[:midpoint]); process(items[midpoint:]); return
            for claim in items:
                request(config["base"], config["token"], "/internal/call-pipeline/asr/fail", {"lease":claim["lease"],"retryable":not isinstance(error, ContractBatchError),"failure":{"code":"v3_contract_error" if isinstance(error, ContractBatchError) else "v3_drain_error","detail":str(error)[:256]}})
    process(claimed); return True

def main():
    base, token, worker_id = os.environ.get("CALL_PIPELINE_WORKER_API_URL", ""), os.environ.get("ASR_WORKER_SESSION_TOKEN", ""), os.environ.get("ASR_WORKER_SESSION_WORKER_ID", "")
    if not all((base, token, worker_id)): raise DrainError("control-plane URL, scoped token, and worker ID are required")
    server = subprocess.Popen(["python3", "/workspace/server.py"])
    stopping = [False]; signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    try:
        deadline = time.monotonic() + 300
        while True:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8080/healthz", timeout=5): break
            except Exception:
                if time.monotonic() >= deadline: raise DrainError("model readiness deadline exceeded")
                time.sleep(1)
        while not stopping[0]:
            if not run_window({"base":base,"token":token,"worker_id":worker_id}):
                request(base, token, "/internal/call-pipeline/asr/drained", {})
                time.sleep(2)
    finally: server.terminate(); server.wait(timeout=30)

if __name__ == "__main__": main()
