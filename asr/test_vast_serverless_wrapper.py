import base64
import os
import runpy
import subprocess
import sys
import unittest
import wave
from io import BytesIO
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "asr" / "vast-pyworker.py"


class VastServerlessWrapperTest(unittest.TestCase):
    def _load_worker(self, run_name):
        fake = ModuleType("vastai")
        fake.BenchmarkConfig = lambda **kwargs: type("Benchmark", (), {"kwargs": kwargs})()
        fake.HandlerConfig = lambda **kwargs: type("Handler", (), {"kwargs": kwargs})()
        fake.WorkerConfig = lambda **kwargs: type("WorkerConfig", (), {"kwargs": kwargs})()
        fake.LogActionConfig = lambda **kwargs: type("LogAction", (), {"kwargs": kwargs})()
        fake.Worker = lambda _: type("Worker", (), {"run": lambda self: None})()
        previous = sys.modules.get("vastai")
        sys.modules["vastai"] = fake
        try:
            return runpy.run_path(str(WORKER), run_name=run_name)
        finally:
            if previous is None: sys.modules.pop("vastai", None)
            else: sys.modules["vastai"] = previous

    def test_worker_construction_has_one_serial_benchmark(self):
        captured = {}

        class BenchmarkConfig:
            def __init__(self, **kwargs): self.kwargs = kwargs

        class HandlerConfig:
            def __init__(self, **kwargs):
                if "benchmark_config" not in kwargs:
                    raise TypeError("benchmark_config is required")
                captured["handler"] = self
                self.kwargs = kwargs

        class WorkerConfig:
            def __init__(self, **kwargs): self.kwargs = kwargs

        class LogActionConfig:
            def __init__(self, **kwargs): self.kwargs = kwargs

        class Worker:
            def __init__(self, config): captured["config"] = config
            def run(self): captured["ran"] = True

        fake = ModuleType("vastai")
        fake.BenchmarkConfig, fake.HandlerConfig = BenchmarkConfig, HandlerConfig
        fake.Worker, fake.WorkerConfig, fake.LogActionConfig = Worker, WorkerConfig, LogActionConfig
        previous = sys.modules.get("vastai")
        sys.modules["vastai"] = fake
        try:
            runpy.run_path(str(WORKER), run_name="__main__")
        finally:
            if previous is None: sys.modules.pop("vastai", None)
            else: sys.modules["vastai"] = previous

        handler = captured["handler"].kwargs
        benchmark = handler["benchmark_config"].kwargs
        self.assertTrue(captured["ran"])
        self.assertFalse(handler["allow_parallel_requests"])
        self.assertEqual(benchmark["runs"], 1)
        self.assertFalse(benchmark["do_warmup"])
        payload = benchmark["generator"]()
        request = payload["requests"][0]
        self.assertEqual(set(request), {"request_version", "lane", "model_id", "model_revision", "audio_filename", "audio_duration_seconds", "audio_base64"})
        self.assertTrue(base64.b64decode(request["audio_base64"]).startswith(b"RIFF"))
        self.assertEqual(handler["workload_calculator"](payload), 1.0)

    def test_wrapper_bootstrap_contract(self):
        dockerfile = (ROOT / "asr" / "vast-serverless.Dockerfile").read_text()
        entrypoint = (ROOT / "asr" / "vast-entrypoint.sh").read_text()
        self.assertIn("USER root", dockerfile)
        self.assertIn("python3 -m pip check", dockerfile)
        self.assertLess(dockerfile.index("RUN chmod"), dockerfile.index("ENTRYPOINT"))
        self.assertNotIn("USER 65532", dockerfile)
        self.assertIn("openssl req", entrypoint)
        self.assertIn("instance.crt", entrypoint)
        self.assertIn("worker_status", entrypoint)
        self.assertIn("urllib.request", entrypoint)
        self.assertNotIn("curl --fail", entrypoint)
        self.assertIn("WORKER_PORT", entrypoint)

    def test_startup_timeout_accepts_a_quoted_number_in_sh(self):
        entrypoint = (ROOT / "asr" / "vast-entrypoint.sh").read_text()
        prefix = "\n".join(entrypoint.splitlines()[:6])
        result = subprocess.run(
            ["/bin/sh", "-c", prefix + "\nprintf '%s\\n' \"$deadline\""],
            env={**os.environ, "STARTUP_TIMEOUT_SECONDS": '"300"'},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip().isdigit())

    def test_minimal_benchmark_wav_is_valid(self):
        namespace = self._load_worker("benchmark_fixture")
        payload = namespace["benchmark_payload"]()
        audio = base64.b64decode(payload["requests"][0]["audio_base64"])
        with wave.open(BytesIO(audio)) as source:
            self.assertEqual((source.getframerate(), source.getnchannels(), source.getsampwidth()), (16000, 1, 2))


if __name__ == "__main__":
    unittest.main()
