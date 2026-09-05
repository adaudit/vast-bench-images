"""Vast PyWorker proxy for the baked Parakeet HTTP server."""
import base64
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

from production_vast_batch import validate_batch

LOGGER = logging.getLogger("parakeet.pyworker")

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


async def validated_response(client_request, model_response):
    """Fail the benchmark when its successful response is not a complete v3 batch."""
    from aiohttp import web

    method = getattr(client_request, "method", "POST")
    route = getattr(client_request, "path", "/transcribe-batch")
    status = getattr(model_response, "status", None)
    body = b""
    try:
        body = await model_response.read()
        if status == 200:
            validate_batch(json.loads(body), (await client_request.json())["requests"])
    except Exception:
        LOGGER.exception(
            "pyworker upstream failure method=%s route=%s status=%s body=%s",
            method, route, status, body[:500].decode("utf-8", "replace"),
        )
        raise
    if status != 200:
        LOGGER.error(
            "pyworker benchmark/proxy outcome method=%s route=%s status=%s body=%s",
            method, route, status, body[:500].decode("utf-8", "replace"),
        )
    else:
        LOGGER.info("pyworker benchmark/proxy outcome method=%s route=%s status=200", method, route)
    return web.Response(
        body=body,
        status=status,
        content_type=model_response.content_type,
        headers={name: value for name, value in model_response.headers.items() if name.lower() != "content-type"},
    )

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
            response_generator=validated_response,
            workload_calculator=workload,
        )],
        log_action_config=LogActionConfig(on_load=[READY_MARKER]),
    )
).run()
