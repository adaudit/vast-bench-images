"""Vast PyWorker proxy for the baked Parakeet HTTP server."""
import base64
import asyncio
import hashlib
import json
import os
import sys
import logging

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

def disable_sdk_sessions(backend_class):
    """The application never uses sessions; disable every SDK session entry."""
    async def rejected_session(self, request):
        from aiohttp import web
        return web.Response(status=404)
    for name in ("session_create_handler", "session_end_handler", "session_get_handler", "session_health_handler"):
        setattr(backend_class, name, rejected_session)

try:
    from vastai.serverless.server.lib import backend, metrics
    disable_sdk_sessions(backend.Backend)
    # SDK debug records include payloads/signatures; never emit them.
    for module in (backend, metrics):
        for value in vars(module).values():
            if isinstance(value, logging.Logger):
                value.disabled = True
except ImportError:
    if getattr(sys.modules.get("vastai"), "__file__", None):
        raise RuntimeError("SDK session hardening unavailable") from None
    # Dependency-free contract tests inject only the public SDK surface.

from production_vast_batch import validate_batch

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
    return float(len(payload["items"] if "items" in payload else payload["requests"]))


async def validated_response(client_request, model_response):
    """Fail the benchmark when its successful response is not a complete v3 batch."""
    from aiohttp import web

    if model_response.content_type == "application/x-ndjson":
        # Forward bounded chunks, not readline(): a long candidate may exceed
        # aiohttp's 64 KiB line limit. Go validates identity and candidate evidence
        # incrementally before each durable per-item completion.
        response = web.StreamResponse(status=model_response.status, headers={"Content-Type": "application/x-ndjson"})
        closed = False
        def close_upstream():
            nonlocal closed
            if not closed:
                closed = True
                close = getattr(model_response, "close", None)
                if callable(close):
                    close()
        owner = asyncio.current_task()
        async def observe_disconnect():
            while True:
                if hasattr(client_request, "transport"):
                    transport = client_request.transport
                    if transport is None or transport.is_closing():
                        close_upstream()
                        owner.cancel()
                        return
                await asyncio.sleep(.025)
        observer = asyncio.create_task(observe_disconnect())
        try:
            await response.prepare(client_request)
            async for chunk in model_response.content.iter_chunked(64 * 1024):
                if hasattr(client_request, "transport"):
                    transport = client_request.transport
                    if transport is None or transport.is_closing():
                        raise ConnectionResetError("client disconnected")
                await response.write(chunk)
            await response.write_eof()
            return response
        finally:
            # On write failure or asyncio cancellation, close the upstream socket
            # so the model server observes disconnect and cancels queued work.
            close_upstream()
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
    body = await model_response.read()
    if model_response.status == 200:
        incoming = await client_request.json()
        validate_batch(json.loads(body), incoming.get("payload", incoming)["requests"])
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
        log_action_config=LogActionConfig(on_load=[READY_MARKER], on_error=["ASR_STARTUP_FAILED"]),
    )
).run()
