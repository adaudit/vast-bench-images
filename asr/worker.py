#!/usr/bin/env python3
"""Credential-minimal ASR queue drainer for the mounted RunPod volume."""
import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
import re
from pathlib import Path

MEDIA_TYPE = "application/vnd.adaudit.asr-candidate-v1+json"
FALLBACKS = {
    "parakeet_dual": "whisper_turbo_fallback",
    "parakeet_pyannote_mono": "whisper_turbo_fallback",
    "whisper_turbo_fallback": "whisper_v3_salvage",
}
_MODELS = {}
_MODEL_LOCK = threading.Lock()
_PARAKEET_POOLS = {}
_PARAKEET_POOL_LOCK = threading.Lock()
PARAKEET_INSTANCES_ENV = "PARAKEET_INSTANCES"
DEFAULT_PARAKEET_INSTANCES = 2


class WorkerError(RuntimeError):
    pass


def parakeet_instance_count():
    """Return the configured number of independent Parakeet inference lanes."""
    try:
        return max(1, int(os.environ.get(PARAKEET_INSTANCES_ENV, DEFAULT_PARAKEET_INSTANCES)))
    except ValueError:
        return DEFAULT_PARAKEET_INSTANCES


class ParakeetInstance:
    """One thread-unsafe model and the CUDA stream reserved for its work."""

    def __init__(self, model, stream):
        self.model = model
        self.stream = stream
        self.lock = threading.Lock()


class ParakeetPool:
    """Blocking checkout pool for independent Parakeet inference lanes."""

    def __init__(self, models, instances, loader):
        self.instances = [loader(models) for _ in range(instances)]
        self._available = queue.Queue(maxsize=instances)
        for instance in self.instances:
            self._available.put(instance)

    def checkout(self):
        return self._available.get()

    def checkin(self, instance):
        self._available.put(instance)


def validate_baked_models(models, manifest):
    """Fail closed when a volume-free image is missing a baked model tree."""
    missing = [lane for lane in manifest.get("models", {}) if not (models / lane).is_dir()]
    if missing:
        raise WorkerError("baked image model cache missing: " + ", ".join(sorted(missing)))


def post(base_url, token, path, payload):
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        if error.code == 204:
            return 204, {}
        raise WorkerError(f"{path}: HTTP {error.code}") from error


def fetch(url):
    with urllib.request.urlopen(url, timeout=300) as response:
        suffix = Path(urllib.parse.urlparse(url).path).suffix or ".audio"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(response.read())
            return Path(handle.name)


def normalize_audio(audio):
    """Convert arbitrary claimed audio to the mono 16 kHz WAV ASR contract."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        normalized = Path(handle.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(normalized)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return normalized
    except (OSError, subprocess.CalledProcessError) as error:
        normalized.unlink(missing_ok=True)
        raise WorkerError("ffmpeg could not normalize audio to mono 16 kHz WAV") from error


def candidate_from_whisper(audio, model_dir, lane):
    from faster_whisper import WhisperModel
    model = _cached_model("whisper:" + lane, lambda: WhisperModel(str(model_dir), device="cuda", compute_type="int8_float16"))
    segments, info = model.transcribe(str(audio), beam_size=5, vad_filter=True)
    result = [{"start_seconds": segment.start, "end_seconds": segment.end, "speaker": "S0", "text": segment.text.strip(), "confidence": float(getattr(segment, "avg_logprob", 0.0))} for segment in segments]
    confidence = min(1.0, max(0.0, (sum(s["confidence"] for s in result) / len(result) + 1) if result else 0.0))
    return {"lane": lane, "confidence": confidence, "segments": result, "language": getattr(info, "language", "")}


def _cached_model(key, factory):
    with _MODEL_LOCK:
        if key not in _MODELS:
            _MODELS[key] = factory()
        return _MODELS[key]


def _load_parakeet_instance(models):
    import torch
    from nemo.collections.asr.models import ASRModel
    model = ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2", cache_dir=str(models / "parakeet")
    )
    return ParakeetInstance(model, torch.cuda.Stream())


def parakeet_pool(models, instances=None):
    """Load and cache the configured set of independent Parakeet lanes."""
    instances = parakeet_instance_count() if instances is None else instances
    key = (str(Path(models).resolve()), instances)
    with _PARAKEET_POOL_LOCK:
        if key not in _PARAKEET_POOLS:
            _PARAKEET_POOLS[key] = ParakeetPool(models, instances, _load_parakeet_instance)
        return _PARAKEET_POOLS[key]


def warm_parakeet_pool(models):
    """Initialize all Parakeet lanes before accepting work."""
    return parakeet_pool(models)


def candidate_from_parakeet(audio, models, lane, batch_size):
    import torch
    pool = parakeet_pool(models)
    instance = pool.checkout()
    try:
        with instance.lock:
            with torch.cuda.stream(instance.stream):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hypotheses = instance.model.transcribe([str(audio)], batch_size=batch_size, timestamps=True)
    finally:
        # A lane cannot be safely reused until all work submitted to its stream is done.
        try:
            instance.stream.synchronize()
        finally:
            pool.checkin(instance)
    hypothesis = hypotheses[0]
    text = hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis)
    return {"lane": lane, "confidence": 0.8, "segments": [
        {"start_seconds": 0.0, "end_seconds": 0.0, "speaker": "S0", "text": text.strip(), "confidence": 0.8}
    ], "language": ""}


def diarize(candidate, audio, models):
    from pyannote.audio import Pipeline
    pipeline = _cached_model("pyannote", lambda: Pipeline.from_pretrained(str(models / "pyannote")))
    turns = list(pipeline(str(audio)).itertracks(yield_label=True))
    labels = {label: index for index, label in enumerate(sorted({item[2] for item in turns}))}
    for segment in candidate["segments"]:
        midpoint = (segment["start_seconds"] + segment["end_seconds"]) / 2
        label = next((label for turn, _, label in turns if turn.start <= midpoint <= turn.end), "SPEAKER_00")
        segment["speaker"] = "S" + str(labels.get(label, 0))
    return candidate


def transcribe(audio, lane, models, blocked, batch_size=8):
    if lane in blocked:
        raise WorkerError(f"lane {lane} blocked: HF_TOKEN plus pyannote license acceptance is required")
    if lane == "whisper_turbo_fallback":
        return candidate_from_whisper(audio, models / "whisper_turbo", lane)
    if lane == "whisper_v3_salvage":
        return candidate_from_whisper(audio, models / "whisper_v3", lane)
    candidate = candidate_from_parakeet(audio, models, lane, batch_size)
    return candidate if lane == "parakeet_dual" else diarize(candidate, audio, models)


def is_oom(error):
    return "out of memory" in str(error).lower() or "cuda oom" in str(error).lower()


def transcribe_with_fallback(audio, lane, models, blocked, params=None, transcriber=transcribe):
    """Run the frozen lane ladder, reducing only OOM retries in the same lane."""
    batch_size = max(1, int((params or {}).get("micro_batch", 8)))
    attempted = []
    while lane:
        attempted.append(lane)
        try:
            while True:
                try:
                    candidate = transcriber(audio, lane, models, blocked, batch_size)
                    candidate["lane"] = lane
                    labels = {}
                    for segment in candidate.get("segments", []):
                        label = str(segment.get("speaker", ""))
                        if not re.fullmatch(r"S\d+", label):
                            labels.setdefault(label, len(labels))
                            segment["speaker"] = "S" + str(labels[label])
                    return candidate, lane, attempted
                except Exception as error:
                    if is_oom(error) and batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        continue
                    raise
        except Exception:
            lane = FALLBACKS.get(lane)
    raise WorkerError("all ASR lanes failed")


def publish(base_url, token, lease, schema_version, candidate):
    body = json.dumps({k: candidate[k] for k in ("lane", "confidence", "segments")}, separators=(",", ":")).encode()
    digest = hashlib.sha256(body).hexdigest()
    _, intent = post(base_url, token, "/internal/call-pipeline/asr/publish-intent", {
        "lease": lease, "schema_version": schema_version, "artifact_kind": "asr_candidate",
        "media_type": MEDIA_TYPE, "sha256": digest, "size_bytes": len(body),
    })
    request = urllib.request.Request(intent["url"], data=body, method="PUT", headers=intent.get("required_headers", {}))
    with urllib.request.urlopen(request, timeout=300) as response:
        version = response.headers.get("x-amz-version-id") or response.headers.get("X-Bz-File-Id")
    if not version:
        raise WorkerError("candidate upload did not return immutable version")
    post(base_url, token, "/internal/call-pipeline/asr/complete", {
        "lease": lease, "schema_version": schema_version, "attempt": lease["Attempt"],
        "result": {"asr_candidate_artifact": {"object_key": intent["object_key"], "version_id": version, "upload_generation": intent["upload_generation"], "media_type": MEDIA_TYPE, "sha256": digest, "size_bytes": len(body)}},
        "idempotency_key": digest,
    })


def run_once(config):
    status, claim = post(config["base_url"], config["token"], "/internal/call-pipeline/asr/claim", {"business_id": config["business_id"], "worker_id": config["worker_id"]})
    if status == 204:
        return False
    lease = claim["lease"]
    _, heartbeat = post(config["base_url"], config["token"], "/internal/call-pipeline/asr/heartbeat", {"lease": lease})
    lease = heartbeat["lease"]
    audio = fetch(claim["audio_url"])
    normalized_audio = None
    try:
        normalized_audio = normalize_audio(audio)
        lane = claim["asr_lane"]
        while lane:
            try:
                candidate = transcribe(normalized_audio, lane, config["models"], config["blocked_lanes"])
                publish(config["base_url"], config["token"], lease, config["schema_version"], candidate)
                return True
            except WorkerError:
                lane = FALLBACKS.get(lane)
        raise WorkerError("all ASR lanes failed")
    except Exception as error:
        post(config["base_url"], config["token"], "/internal/call-pipeline/asr/fail", {"lease": lease, "retryable": True, "failure": {"code": "asr_worker_error", "detail": str(error)[:512]}})
        raise
    finally:
        if normalized_audio:
            normalized_audio.unlink(missing_ok=True)
        audio.unlink(missing_ok=True)


def config_from_env():
    root = Path(os.environ.get("WORKSPACE_ROOT", "/workspace"))
    manifest = json.loads((root / "model-pins.json").read_text())
    required = {"base_url": os.environ.get("CALL_PIPELINE_WORKER_API_URL", "").rstrip("/"), "token": os.environ.get("ASR_WORKER_SESSION_TOKEN", ""), "worker_id": os.environ.get("ASR_WORKER_SESSION_WORKER_ID", "")}
    try:
        required["business_id"] = int(os.environ["CALL_PIPELINE_WORKER_BUSINESS_ID"])
    except (KeyError, ValueError):
        required["business_id"] = 0
    if not all(required.values()):
        raise WorkerError("CALL_PIPELINE_WORKER_API_URL, ASR_WORKER_SESSION_TOKEN, ASR_WORKER_SESSION_WORKER_ID, and CALL_PIPELINE_WORKER_BUSINESS_ID are required")
    models = root / "models"
    validate_baked_models(models, manifest)
    required.update(models=models, blocked_lanes=set(manifest.get("blocked_lanes", [])), schema_version=os.environ.get("ASR_SCHEMA_VERSION", "asr-candidate-v1"))
    return required


def main():
    config = config_from_env()
    # PARAKEET_INSTANCES (default 2) controls independently loaded GPU lanes.
    warm_parakeet_pool(config["models"])
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    while not stopping[0]:
        try:
            if not run_once(config):
                time.sleep(2)
        except Exception as error:
            print(error, file=sys.stderr)
            time.sleep(2)


if __name__ == "__main__":
    main()
