"""RunPod Serverless entrypoint; it only receives signed URLs and job metadata."""
import asyncio
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import worker


DEFAULT_CONCURRENCY = 3
_V3_LOCK = threading.Lock()


def _input(job):
    value = job.get("input") if isinstance(job, dict) else None
    required = ("audio_url", "control_plane_base_url", "worker_session_token", "transcript_id", "lease_id", "lease", "lane", "params")
    non_empty = ("audio_url", "control_plane_base_url", "worker_session_token", "transcript_id", "lease_id", "lease", "lane")
    if not isinstance(value, dict) or any(key not in value for key in required) or any(not value.get(key) for key in non_empty) or not isinstance(value["params"], dict):
        raise worker.WorkerError("job input requires the Go RunPod serverless submit contract")
    return value


def concurrency_modifier(_):
    """Keep enough worker threads available to occupy every Parakeet instance."""
    minimum = worker.parakeet_instance_count()
    try:
        return max(minimum, int(os.environ.get("RUNPOD_HANDLER_CONCURRENCY", DEFAULT_CONCURRENCY)))
    except ValueError:
        return max(minimum, DEFAULT_CONCURRENCY)


def _transcriber_with_v3_guard(transcriber):
    def guarded(audio, lane, models, blocked, batch_size):
        if lane == "whisper_v3_salvage":
            with _V3_LOCK:
                return transcriber(audio, lane, models, blocked, batch_size)
        return transcriber(audio, lane, models, blocked, batch_size)
    return guarded


def _handle(job, *, fetcher=worker.fetch, publisher=worker.publish, transcriber=worker.transcribe):
    """Fetch signed audio and finish it through the fenced worker-session lifecycle."""
    started = time.monotonic()
    data = _input(job)
    root = Path(os.environ.get("WORKSPACE_ROOT", "/workspace"))
    manifest = json.loads((root / "model-pins.json").read_text())
    audio = fetcher(data["audio_url"])
    normalized_audio = None
    try:
        normalized_audio = worker.normalize_audio(audio)
        candidate, lane_used, attempted = worker.transcribe_with_fallback(
            normalized_audio, data["lane"], root / "models", set(manifest.get("blocked_lanes", [])),
            data["params"], _transcriber_with_v3_guard(transcriber),
        )
        body = json.dumps({key: candidate[key] for key in ("lane", "confidence", "segments")}, separators=(",", ":")).encode()
        digest, size = hashlib.sha256(body).hexdigest(), len(body)
        publisher(
            data["control_plane_base_url"], data["worker_session_token"], data["lease"],
            data["params"].get("schema_version", "asr-v1"), candidate,
        )
        return {"status": "succeeded", "lane_used": lane_used, "metrics": {
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "attempted_lanes": attempted,
            "candidate_sha256": digest,
            "candidate_size_bytes": size,
        }}
    finally:
        if normalized_audio:
            Path(normalized_audio).unlink(missing_ok=True)
        Path(audio).unlink(missing_ok=True)


async def handler(job, *, fetcher=worker.fetch, publisher=worker.publish, transcriber=worker.transcribe):
    """Run blocking GPU and network work concurrently in bounded worker threads."""
    return await asyncio.to_thread(_handle, job, fetcher=fetcher, publisher=publisher, transcriber=transcriber)


def main():
    import runpod
    root = Path(os.environ.get("WORKSPACE_ROOT", "/workspace"))
    manifest = json.loads((root / "model-pins.json").read_text())
    worker.validate_baked_models(root / "models", manifest)
    worker.warm_parakeet_pool(root / "models")
    runpod.serverless.start({"handler": handler, "concurrency_modifier": concurrency_modifier})


if __name__ == "__main__":
    main()
