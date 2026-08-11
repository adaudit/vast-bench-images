"""Vast serverless PyWorker for the ASR model server.

Deployed as the pyworker entrypoint (start_server.sh runs $SERVER_DIR/worker.py).
It is a proxy: it does NOT run the model, it forwards /transcribe to our
FastAPI model server on 127.0.0.1:MODEL_SERVER_PORT, detects readiness from the
model log, and reports per-request load so Vast's autoscaler can scale.
"""
import os

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", "18100"))
MODEL_LOG_FILE = os.environ.get("MODEL_LOG", "/workspace/model-server.log")

# Must match server.MODEL_SERVER_LOADED_MARKER exactly.
MODEL_LOADED_MARKER = "MODEL_SERVER_LOADED: ready for traffic"

# A short, publicly reachable clip Vast replays to benchmark throughput on boot.
# Overridable so the endpoint can be benchmarked against a representative file.
BENCHMARK_AUDIO_URL = os.environ.get(
    "BENCHMARK_AUDIO_URL",
    "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav",
)

worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url="/ping",
    handlers=[
        HandlerConfig(
            route="/transcribe",
            # Test A: FIFO one-at-a-time keeps the first correctness run clean;
            # 3-lane parallelism is a follow-up tuning step.
            allow_parallel_requests=False,
            max_queue_time=60.0,
            benchmark_config=BenchmarkConfig(
                dataset=[{"audio_url": BENCHMARK_AUDIO_URL, "lane": "parakeet_dual", "params": {}}],
                runs=1,
            ),
            # Non-LLM constant cost per request (perf units).
            workload_calculator=lambda _: 1000.0,
        )
    ],
    log_action_config=LogActionConfig(
        on_load=[MODEL_LOADED_MARKER],
        # Only OUR explicit error markers — NOT a generic "Traceback", which NeMo
        # prints benignly during model restore and would false-positive-kill the
        # worker mid-load (before the ~112s warm even completes).
        on_error=["LANE_FAILED", "TRANSCRIBE_FAILED"],
    ),
)

Worker(worker_config).run()
