import base64
import asyncio
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from urllib.error import HTTPError
from pathlib import Path
from types import ModuleType

from asr import server


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "asr" / "vast-pyworker.py"


class VastServerlessWrapperTest(unittest.TestCase):
    @staticmethod
    def _http(request):
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            body = json.loads(error.read()); error.close()
            return error.code, body

    def _benchmark_env(self):
        directory = tempfile.TemporaryDirectory()
        artifact = ROOT / "asr" / "fixtures" / "audio" / "2086-149220-0033.wav"
        return directory, {"VAST_BENCHMARK_ARTIFACT": str(artifact)}

    def _validated_response(self, generator, payload, body):
        class Response:
            def __init__(self, **kwargs): self.kwargs = kwargs
        class Request:
            async def json(self): return payload
        class ModelResponse:
            status, content_type, headers = 200, "application/json", {"Content-Type": "application/json"}
            async def read(self): return body
        fake = ModuleType("aiohttp"); fake.web = type("Web", (), {"Response": Response})
        previous = sys.modules.get("aiohttp"); sys.modules["aiohttp"] = fake
        try:
            return asyncio.run(generator(Request(), ModelResponse()))
        finally:
            if previous is None: sys.modules.pop("aiohttp", None)
            else: sys.modules["aiohttp"] = previous

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
        previous, env = sys.modules.get("vastai"), os.environ.copy()
        sys.modules["vastai"] = fake
        directory, additions = self._benchmark_env(); os.environ.update(additions)
        try:
            runpy.run_path(str(WORKER), run_name="__main__")
            payload = captured["handler"].kwargs["benchmark_config"].kwargs["generator"]()
        finally:
            directory.cleanup(); os.environ.clear(); os.environ.update(env)
            if previous is None: sys.modules.pop("vastai", None)
            else: sys.modules["vastai"] = previous

        handler = captured["handler"].kwargs
        benchmark = handler["benchmark_config"].kwargs
        self.assertTrue(captured["ran"])
        self.assertFalse(handler["allow_parallel_requests"])
        self.assertEqual(benchmark["runs"], 1)
        self.assertFalse(benchmark["do_warmup"])
        self.assertTrue(callable(handler["response_generator"]))
        self.assertEqual(handler["response_generator"].__name__, "validated_response")
        request = payload["requests"][0]
        self.assertEqual(set(request), {"request_version", "lane", "model_id", "model_revision", "audio_filename", "audio_duration_seconds", "audio_base64"})
        self.assertTrue(base64.b64decode(request["audio_base64"]).startswith(b"RIFF"))
        self.assertEqual(handler["workload_calculator"](payload), 1.0)

    def test_isolated_worker_imports_copied_validator_and_validates_responses(self):
        copied = ("production_vast_batch.py", "drain_worker.py", "vast_adapter.py", "offline_entrypoint.py", "vast_failure_guard.py")
        probe = '''
import asyncio, json, runpy, sys, types
captured = {}
class HandlerConfig:
    def __init__(self, **kwargs): captured["handler"] = kwargs
class Worker:
    def __init__(self, config): pass
    def run(self): pass
fake = types.ModuleType("vastai")
fake.BenchmarkConfig = lambda **kwargs: kwargs
fake.HandlerConfig = HandlerConfig
fake.LogActionConfig = lambda **kwargs: kwargs
fake.WorkerConfig = lambda **kwargs: kwargs
fake.Worker = Worker
sys.modules["vastai"] = fake
runpy.run_path(sys.argv[1], run_name="__main__")
request = {"audio_duration_seconds": 7.435}
complete = [{"schema_version": "asr-candidate-v3", "disposition": "speech", "lane": "parakeet_v3", "model_id": "nvidia/parakeet-tdt-0.6b-v3", "model_revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b", "audio_duration_seconds": 7.435, "segments": [{"start_seconds": 0, "end_seconds": 1, "text": "fixture", "confidence": .9}], "selected_segment_indexes": [], "calibration": {"corpus_sha256": "0" * 64, "metric": "segment_brier_score", "threshold": .7, "decision_rule": "calibrated_confidence < threshold", "segment_evidence": [{"segment_index": 0, "raw_confidence": .9, "calibrated_confidence": .8, "timestamp_start_seconds": 0, "timestamp_end_seconds": 1}]}}]
class Request:
    async def json(self): return {"requests": [request]}
class ModelResponse:
    def __init__(self, body, status): self.body, self.status, self.content_type, self.headers = body, status, "application/json", {"Content-Type": "application/json"}
    async def read(self): return self.body
fake_aiohttp = types.ModuleType("aiohttp")
fake_aiohttp.web = types.SimpleNamespace(Response=lambda **kwargs: kwargs)
sys.modules["aiohttp"] = fake_aiohttp
async def response(body, status): return await captured["handler"]["response_generator"](Request(), ModelResponse(body, status))
accepted = asyncio.run(response(json.dumps(complete).encode(), 200))
try: asyncio.run(response(b"[]", 200))
except ValueError: malformed = True
else: malformed = False
non_200 = asyncio.run(response(b"upstream failure", 502))
print(json.dumps({"handler": "response_generator" in captured["handler"], "accepted": accepted["status"] == 200, "malformed": malformed, "non_200": non_200["status"] == 502 and non_200["body"] == b"upstream failure"}))
'''
        with tempfile.TemporaryDirectory() as directory:
            worker_directory = Path(directory, "vast-pyworker"); worker_directory.mkdir()
            shutil.copy(WORKER, worker_directory / "worker.py")
            for name in copied:
                shutil.copy(ROOT / "asr" / name, worker_directory / name)
            result = subprocess.run([sys.executable, "-I", "-c", probe, str(worker_directory / "worker.py")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"handler": True, "accepted": True, "malformed": True, "non_200": True})

    def test_wrapper_bootstrap_contract(self):
        dockerfile = (ROOT / "asr" / "vast-serverless.Dockerfile").read_text()
        entrypoint = (ROOT / "asr" / "vast-entrypoint.sh").read_text()
        self.assertIn("USER root", dockerfile)
        self.assertIn("python3 -m pip check", dockerfile)
        self.assertIn('python3 -I -c "import vastai"', dockerfile)
        self.assertLess(dockerfile.index("RUN chmod"), dockerfile.index("ENTRYPOINT"))
        self.assertNotIn("USER 65532", dockerfile)
        self.assertIn("openssl req", entrypoint)
        self.assertIn("instance.crt", entrypoint)
        self.assertIn("worker_status", entrypoint)
        self.assertIn("urllib.request", entrypoint)
        self.assertNotIn("curl --fail", entrypoint)
        self.assertIn("WORKER_PORT", entrypoint)
        self.assertIn("ENV PARAKEET_INSTANCES=3", dockerfile)
        self.assertIn("COPY asr/server.py asr/parakeet_pool.py /workspace/", dockerfile)

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

    def test_entrypoint_exits_immediately_for_terminal_failed_health(self):
        entrypoint = (ROOT / "asr" / "vast-entrypoint.sh").read_text()
        loop = entrypoint[entrypoint.index("while :; do"):entrypoint.index("done", entrypoint.index("while :; do")) + 4]
        with tempfile.TemporaryDirectory() as directory:
            fake_python = Path(directory, "python3")
            fake_python.write_text("#!/bin/sh\nprintf '%s\\n' failed\n")
            fake_python.chmod(0o755)
            result = subprocess.run(["/bin/sh", "-c", loop], env={**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"]}, capture_output=True, text=True, timeout=2)
        self.assertNotEqual(result.returncode, 0)

    def test_benchmark_requires_the_approved_wav(self):
        namespace = self._load_worker("benchmark_fixture")
        with tempfile.NamedTemporaryFile() as artifact:
            artifact.write(b"RIFFnot-approved"); artifact.flush()
            env = os.environ.copy(); os.environ["VAST_BENCHMARK_ARTIFACT"] = artifact.name
            try:
                with self.assertRaises(RuntimeError): namespace["benchmark_payload"]()
            finally:
                os.environ.clear(); os.environ.update(env)

    def test_benchmark_hook_rejects_incomplete_http_success(self):
        directory, additions = self._benchmark_env(); env = os.environ.copy(); os.environ.update(additions)
        try:
            namespace = self._load_worker("benchmark_response_validation")
            payload = namespace["benchmark_payload"]()
        finally:
            directory.cleanup(); os.environ.clear(); os.environ.update(env)
        runtime = server.Runtime(model_verifier=lambda _: None, model_loader=lambda: type("Model", (), {"transcribe": lambda *_args, **_kwargs: []})(), batch_transcriber=lambda _: [[{"start_seconds": 0, "end_seconds": 1, "text": "fixture", "confidence": .9}]])
        runtime.initialize_once()
        body = runtime.transcribe_batch(payload["requests"])
        response = self._validated_response(namespace["validated_response"], payload, json.dumps(body).encode())
        self.assertEqual(json.loads(response.kwargs["body"]), body)
        with self.assertRaises(ValueError):
            self._validated_response(namespace["validated_response"], payload, b'[]')

    def test_approved_benchmark_reaches_public_http_contract(self):
        directory, additions = self._benchmark_env(); env = os.environ.copy(); os.environ.update(additions)
        try:
            namespace = self._load_worker("approved_benchmark")
            payload = namespace["benchmark_payload"]()
        finally:
            directory.cleanup(); os.environ.clear(); os.environ.update(env)
        runtime = server.Runtime(model_verifier=lambda _: None, model_loader=lambda: type("Model", (), {"transcribe": lambda *_args, **_kwargs: []})(), batch_transcriber=lambda _: [[{"start_seconds": 0, "end_seconds": 1, "text": "fixture", "confidence": .9}]])
        http = server.make_server(("127.0.0.1", 0), runtime)
        thread = threading.Thread(target=http.serve_forever); thread.start()
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{http.server_port}/transcribe-batch", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request) as response:
                body = json.loads(response.read())
        finally:
            http.shutdown(); thread.join(); http.server_close()
        self.assertEqual(body[0]["schema_version"], "asr-candidate-v3")

    def test_http_health_and_post_are_observation_only_across_startup(self):
        entered, release, loads = threading.Event(), threading.Event(), []
        def loader():
            loads.append(1); entered.set(); release.wait(2)
            return type("Model", (), {"transcribe": lambda *_args, **_kwargs: []})()
        runtime = server.Runtime(model_verifier=lambda _: None, model_loader=loader)
        http = server.make_server(("127.0.0.1", 0), runtime)
        thread = threading.Thread(target=http.serve_forever); thread.start()
        try:
            self.assertTrue(entered.wait(1))
            health = [None] * 4
            probes = [threading.Thread(target=lambda index=index: health.__setitem__(index, self._http(f"http://127.0.0.1:{http.server_port}/healthz"))) for index in range(4)]
            [probe.start() for probe in probes]; [probe.join() for probe in probes]
            self.assertEqual(health, [(503, {"status": "loading"})] * 4)
            self.assertEqual(self._http(urllib.request.Request(f"http://127.0.0.1:{http.server_port}/transcribe-batch", data=b'{"requests":[]}', headers={"Content-Type": "application/json"}, method="POST")), (503, {"error": "not ready"}))
            self.assertEqual(len(loads), 1)
            release.set()
            while runtime.health()[0] != 200:
                pass
            self.assertEqual(self._http(f"http://127.0.0.1:{http.server_port}/healthz"), (200, {"status": "ready"}))
            self.assertEqual(len(loads), 3)
        finally:
            release.set(); http.shutdown(); thread.join(); http.server_close()

    def test_http_terminal_failure_is_stable_and_sanitized(self):
        loads = []
        runtime = server.Runtime(model_verifier=lambda _: None, model_loader=lambda: loads.append(1) or (_ for _ in ()).throw(RuntimeError("/tmp/private-model")))
        http = server.make_server(("127.0.0.1", 0), runtime)
        thread = threading.Thread(target=http.serve_forever); thread.start()
        try:
            while runtime.health()[0] != 503 or runtime.health()[1]["status"] != "failed":
                pass
            expected = (503, {"status": "failed", "cause": "initialization failed"})
            self.assertEqual(self._http(f"http://127.0.0.1:{http.server_port}/healthz"), expected)
            self.assertEqual(self._http(f"http://127.0.0.1:{http.server_port}/healthz"), expected)
            self.assertEqual(len(loads), 1)
        finally:
            http.shutdown(); thread.join(); http.server_close()


if __name__ == "__main__":
    unittest.main()
