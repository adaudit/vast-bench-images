"""Vast PyWorker proxy for the baked Parakeet HTTP server."""
from vastai import HandlerConfig, LogActionConfig, Worker, WorkerConfig

READY_MARKER = "PARAKEET_SERVER_READY"

Worker(
    WorkerConfig(
        model_server_url="http://127.0.0.1",
        model_server_port=8080,
        model_log_file="/workspace/parakeet-server.log",
        model_healthcheck_url="/healthz",
        handlers=[HandlerConfig(route="/transcribe-batch", allow_parallel_requests=False, max_queue_time=60.0)],
        log_action_config=LogActionConfig(on_load=[READY_MARKER]),
    )
).run()
