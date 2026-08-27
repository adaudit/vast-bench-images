"""Vast PyWorker proxy for the baked Parakeet HTTP server."""
import base64
from io import BytesIO
import wave

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

READY_MARKER = "PARAKEET_SERVER_READY"


def benchmark_payload():
    """One-second generated silence is a rights-cleared, valid WAV benchmark."""
    wav = BytesIO()
    with wave.open(wav, "wb") as output:
        output.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
        output.writeframes(b"\0\0" * 16000)
    return {"requests": [{
        "request_version": "parakeet-v3-offline-request-v1",
        "lane": "parakeet_v3",
        "model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "model_revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        "audio_filename": "benchmark.wav",
        "audio_duration_seconds": 1.0,
        "audio_base64": base64.b64encode(wav.getvalue()).decode(),
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
