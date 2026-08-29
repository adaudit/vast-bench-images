"""Vast PyWorker proxy for the baked Parakeet HTTP server."""
import base64
import hashlib
import os
from pathlib import Path

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

READY_MARKER = "PARAKEET_SERVER_READY"
BENCHMARK_ARTIFACT = Path("/workspace/fixtures/2086-149220-0033.wav")
BENCHMARK_SHA256 = "5fceacff0315d49cb59fcc505bcecf1ed5f2f35c2897b1e65a59f30e5d922150"
BENCHMARK_BYTES = 237964
BENCHMARK_DURATION_SECONDS = 7.435


def benchmark_payload():
    """Use the approved NVIDIA/LibriSpeech sample, never generated silence."""
    artifact = Path(os.environ.get("VAST_BENCHMARK_ARTIFACT", BENCHMARK_ARTIFACT))
    try:
        audio = artifact.read_bytes()
    except OSError as error:
        raise RuntimeError("approved benchmark artifact is unavailable") from error
    if len(audio) != BENCHMARK_BYTES or hashlib.sha256(audio).hexdigest() != BENCHMARK_SHA256 or audio[:4] != b"RIFF":
        raise RuntimeError("benchmark artifact is not the approved canonical WAV")
    return {"requests": [{
        "request_version": "parakeet-v3-offline-request-v1",
        "lane": "parakeet_v3",
        "model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "model_revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        "audio_filename": artifact.name,
        "audio_duration_seconds": BENCHMARK_DURATION_SECONDS,
        "audio_base64": base64.b64encode(audio).decode(),
    }]}


def workload(payload):
    return float(len(payload["requests"]))

Worker(
    WorkerConfig(
        model_server_url="http://127.0.0.1",
        model_server_port=8080,
        model_log_file="/workspace/parakeet-server.log",
        model_healthcheck_url="/healthz",
        handlers=[HandlerConfig(
            route="/transcribe-batch",
            allow_parallel_requests=False,
            max_queue_time=60.0,
            benchmark_config=BenchmarkConfig(generator=benchmark_payload, runs=1, do_warmup=False),
            workload_calculator=workload,
        )],
        log_action_config=LogActionConfig(on_load=[READY_MARKER]),
    )
).run()
