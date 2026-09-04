"""Vast PyWorker proxy for the baked Sortformer HTTP server."""
import base64
import hashlib
import json
import os
from pathlib import Path

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

READY_MARKER = "SORTFORMER_SERVER_READY"
BENCHMARK_ARTIFACT = Path("/workspace/fixtures/2086-149220-0033.wav")
BENCHMARK_SHA256 = "5fceacff0315d49cb59fcc505bcecf1ed5f2f35c2897b1e65a59f30e5d922150"
BENCHMARK_BYTES = 237964
MODEL_REVISION = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d"



def benchmark_payload():
    """Use the approved LibriSpeech WAV, never generated silence."""
    artifact = Path(os.environ.get("VAST_BENCHMARK_ARTIFACT", BENCHMARK_ARTIFACT))
    try:
        audio = artifact.read_bytes()
    except OSError as error:
        raise RuntimeError("approved benchmark artifact is unavailable") from error
    if len(audio) != BENCHMARK_BYTES or hashlib.sha256(audio).hexdigest() != BENCHMARK_SHA256 or audio[:4] != b"RIFF":
        raise RuntimeError("benchmark artifact is not the approved canonical WAV")
    return {"audio_base64": base64.b64encode(audio).decode()}


def workload(_payload):
    return 1.0


def validate_response(payload):
    if not isinstance(payload, dict) or payload.get("engine") != "sortformer" or payload.get("model_revision") != MODEL_REVISION:
        raise ValueError("Sortformer response is malformed")
    turns = payload.get("speaker_turns")
    metrics = payload.get("metrics")
    if not isinstance(turns, list) or not isinstance(metrics, dict):
        raise ValueError("Sortformer response is malformed")
    if any(set(turn) != {"start_s", "end_s", "speaker_idx"} or turn["end_s"] <= turn["start_s"] for turn in turns):
        raise ValueError("Sortformer response has invalid turns")
    if set(metrics) != {"elapsed_ms", "audio_seconds", "num_speakers", "peak_vram_mb"}:
        raise ValueError("Sortformer response has invalid metrics")


async def validated_response(_client_request, model_response):
    """Fail the benchmark if a successful response violates /diarize."""
    from aiohttp import web
    body = await model_response.read()
    if model_response.status == 200:
        validate_response(json.loads(body))
    return web.Response(
        body=body,
        status=model_response.status,
        content_type=model_response.content_type,
        headers={name: value for name, value in model_response.headers.items() if name.lower() != "content-type"},
    )


Worker(
    WorkerConfig(
        model_server_url="http://127.0.0.1",
        model_server_port=8080,
        model_log_file="/workspace/diar-server.log",
        model_healthcheck_url="/healthz",
        handlers=[HandlerConfig(
            route="/diarize",
            allow_parallel_requests=False,
            max_queue_time=60.0,
            benchmark_config=BenchmarkConfig(generator=benchmark_payload, runs=1, do_warmup=False),
            response_generator=validated_response,
            workload_calculator=workload,
        )],
        log_action_config=LogActionConfig(on_load=[READY_MARKER]),
    )
).run()
