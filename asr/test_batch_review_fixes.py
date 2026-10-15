import hashlib
import io
import queue
import tempfile
import threading
import unittest
import wave
import asyncio
import json
import runpy
import shutil
import subprocess
import sys
import time
import contextlib
from unittest.mock import patch
from types import ModuleType, SimpleNamespace
from pathlib import Path

from asr import server, offline_entrypoint as pin
from asr.batch_runtime import batch_events
from asr.startup import Stages


class BatchReviewFixesTest(unittest.TestCase):
    def test_extracted_loader_rebuilds_failed_lane_without_archive_or_cuda(self):
        models, connectors, restores, resets = [], [], [], []
        class Connector:
            def __init__(self):
                connectors.append(self)
        def restore(path, *, save_restore_connector, map_location):
            self.assertTrue(Path(path).is_dir())
            self.assertEqual(save_restore_connector.model_extracted_dir, path)
            self.assertEqual(map_location.type, "cpu")
            self.assertFalse(archive.exists())
            restores.append(path)
            model = SimpleNamespace(cfg=SimpleNamespace(decoding=SimpleNamespace()), transcribe=lambda *_args, **_kwargs: [])
            def configure(*_args, **_kwargs):
                if len(models) == 1:
                    raise RuntimeError("synthetic failed decoding setup")
            model.change_decoding_strategy = configure
            models.append(model)
            return model
        modules = {}
        for name in ("nemo", "nemo.collections", "nemo.collections.asr", "nemo.collections.asr.models", "nemo.core", "nemo.core.connectors", "nemo.core.connectors.save_restore_connector", "omegaconf", "torch"):
            modules[name] = ModuleType(name)
        modules["nemo.collections.asr.models"].ASRModel = SimpleNamespace(restore_from=restore, _set_model_restore_state=lambda **kwargs: resets.append(kwargs))
        modules["nemo.core.connectors.save_restore_connector"].SaveRestoreConnector = Connector
        modules["omegaconf"].open_dict = lambda _: contextlib.nullcontext()
        modules["torch"].device = lambda device: SimpleNamespace(type=device)
        modules["torch"].cuda = SimpleNamespace(is_available=lambda: False, Stream=lambda: self.fail("CPU loading touched CUDA"))
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "absent.nemo"
            extracted = archive.parent / "parakeet-v3"
            extracted.mkdir()
            (extracted / "model_config.yaml").write_text("config")
            (extracted / "model_weights.ckpt").write_bytes(b"weights")
            manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in extracted.iterdir()}
            (extracted / "sha256-manifest.json").write_text(json.dumps(manifest))
            with patch.dict(sys.modules, modules), patch.object(server, "MODEL_PATH", archive), patch.object(server, "verify_model", side_effect=AssertionError("runtime touched archive")), patch.object(server.Runtime, "_check_dependencies"):
                runtime = server.Runtime(stages=Stages(sleep=lambda _: None, log=lambda *_, **__: None), benchmark=lambda _: None)
                runtime.initialize_once()
            self.assertTrue(runtime.ready)
            self.assertEqual(len(models), 4)
            self.assertEqual(len(connectors), 4)
            self.assertEqual(len({id(connector) for connector in connectors}), 4)
            self.assertEqual(restores, [str(extracted)] * 4)
            self.assertEqual(resets, [{"is_being_restored": False}])
            self.assertEqual(runtime.pool.instances, models[1:])
            self.assertNotIn(id(models[0]), runtime._streams)

    def test_stage_failure_logs_only_class_and_fixed_reason(self):
        output = io.StringIO()
        stages = Stages(attempts=1, log=lambda line, **_: print(line, file=output))
        def fail():
            raise FileNotFoundError("/secret/model.nemo https://private.invalid/?token=hidden")
        with self.assertRaisesRegex(RuntimeError, "startup stage exhausted"):
            stages.run("lane_1", fail)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("error=FileNotFoundError reason=required_file_missing", lines[1])
        self.assertNotIn("secret", output.getvalue())
        self.assertNotIn("https", output.getvalue())
        self.assertNotIn("hidden", output.getvalue())

    def test_cpu_benchmark_shim_preserves_torch_cuda_types_and_availability(self):
        workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/publish-asr-v3.yml").read_text()
        shim = workflow.split('cat > "$RUNNER_TEMP/cpu-benchmark-shim.py" <<\'SHIM\'\n', 1)[1].split("          SHIM", 1)[0]
        import textwrap
        torch = ModuleType("torch")
        stream_type = type("Stream", (), {})
        available = lambda: False
        torch.cuda = SimpleNamespace(is_available=available, Stream=stream_type, stream=contextlib.nullcontext)
        checks = []
        class Runtime:
            def _check_dependencies(self, *, require_cuda=True):
                checks.append(require_cuda)
        fake_server = ModuleType("server")
        fake_server.Runtime = Runtime
        fake_server.make_server = lambda: SimpleNamespace(serve_forever=lambda: Runtime()._check_dependencies())
        with patch.dict(sys.modules, {"torch": torch, "server": fake_server}), patch("runpy.run_path", side_effect=lambda *_args, **_kwargs: Runtime()._check_dependencies()), contextlib.redirect_stdout(io.StringIO()):
            exec(textwrap.dedent(shim), {})
        self.assertIs(torch.cuda.Stream, stream_type)
        self.assertIs(torch.cuda.is_available, available)
        self.assertEqual(checks, [False])

    def test_cpu_benchmark_bypasses_only_hardware_requirement(self):
        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(is_available=lambda: False)
        runtime = server.Runtime()
        with patch.dict(sys.modules, {"torch": torch}), patch("importlib.metadata.version", return_value="2.4.1"), patch.object(server.logging, "disable"):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                runtime._check_dependencies()
            runtime._check_dependencies(require_cuda=False)
        with patch.dict(sys.modules, {"torch": torch}), patch("importlib.metadata.version", return_value="unexpected"), patch.object(server.logging, "disable"):
            with self.assertRaisesRegex(RuntimeError, "unexpected NeMo version"):
                runtime._check_dependencies(require_cuda=False)

    def test_extract_model_skip_flag_skips_only_nemo_connector_check(self):
        with patch.object(sys, "path", [str(Path(__file__).parent), *sys.path]):
            import extract_model as extractor
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "models" / "model.nemo"
            model.parent.mkdir()
            model.write_bytes(b"pinned model")
            target = model.parent / "parakeet-v3"
            (target).mkdir()
            (target / "model_config.yaml").write_text("config")
            (target / "model_weights.ckpt").write_bytes(b"weights")
            with patch.object(extractor, "MODEL_PATH", model), patch.object(extractor, "verify_model") as verify, patch.object(extractor.tarfile, "open") as archive_open, patch.dict("os.environ", {"ASR_VERIFY_NEMO_CONNECTOR": "0"}):
                archive = archive_open.return_value.__enter__.return_value
                archive.getmembers.return_value = []
                extractor.extract()
            verify.assert_called_once_with(model)
            manifest = json.loads((target / "sha256-manifest.json").read_text())
            self.assertIn("model_config.yaml", manifest)
            self.assertIn("model_weights.ckpt", manifest)

    def test_benchmarks_each_specific_lane_and_retries_same_lane(self):
        models = []
        calls = []
        def loader():
            model = type("Model", (), {"transcribe": lambda *_args, **_kwargs: []})()
            models.append(model)
            return model
        def benchmark(model):
            lane = models.index(model) + 1
            calls.append(lane)
            if lane == 1 and calls.count(1) == 1:
                raise ValueError("synthetic retry")
        runtime = server.Runtime(model_verifier=lambda _: None, model_loader=loader, benchmark=benchmark, stages=Stages(sleep=lambda _: None, log=lambda *_, **__: None))
        runtime.initialize_once()
        self.assertTrue(runtime.ready)
        self.assertEqual(calls, [1, 1, 2, 3])

    def test_broken_third_lane_never_writes_ready_marker(self):
        models = []
        def loader():
            model = type("Model", (), {"transcribe": lambda *_args, **_kwargs: []})()
            models.append(model)
            return model
        def benchmark(model):
            if models.index(model) == 2:
                raise ValueError("synthetic lane failure")
        with tempfile.TemporaryDirectory() as directory:
            runtime = server.Runtime(model_verifier=lambda _: None, model_loader=loader, benchmark=benchmark, stages=Stages(directory, sleep=lambda _: None, log=lambda *_, **__: None))
            runtime.initialize_once()
            self.assertFalse(runtime.ready)
            self.assertFalse((Path(directory) / "ready.done").exists())
            self.assertFalse((Path(directory) / "benchmark_3.done").exists())

    def test_completed_download_is_inferred_before_stalled_download_ends(self):
        output = io.BytesIO()
        with wave.open(output, "wb") as audio:
            audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000); audio.writeframes(bytes(32000))
        body = output.getvalue()
        receipt = {"sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body), "media_type": "audio/wav", "canonical_sample_rate": 16000, "canonical_channel_count": 1, "canonical_frame_count": 16000}
        params = {"request_version": pin.REQUEST_VERSION, "model_id": pin.MODEL_ID, "model_revision": pin.MODEL_REVISION, "audio_duration_seconds": 1, "audio_receipt": receipt}
        payload = {"batch_id": "synthetic", "items": [{"id": c * 64, "audio_url": c, "lane": "parakeet_v3", "params": params} for c in "ab"]}
        release = threading.Event()
        events = queue.Queue()
        class Runtime:
            def transcribe(self, request): return pin.build_candidate(1, [])
        def fetch(url, path):
            if url == "b": release.wait(3)
            path.write_bytes(body)
        def run():
            for event in batch_events(Runtime(), payload, fetcher=fetch, keepalive_seconds=.02):
                if event["type"] == "item": events.put(event)
        thread = threading.Thread(target=run)
        thread.start()
        try:
            first = events.get(timeout=1)
            self.assertEqual(first["id"], "a" * 64)
            self.assertFalse(release.is_set())
        finally:
            release.set(); thread.join(4)
        self.assertFalse(thread.is_alive())

    def test_sdk_session_routes_are_disabled_before_callbacks_can_run(self):
        from asr.test_vast_serverless_wrapper import VastServerlessWrapperTest
        namespace = VastServerlessWrapperTest()._load_worker("synthetic_session_hardening")
        class Backend:
            called = False
            async def session_create_handler(self, request):
                self.called = True
                raise AssertionError("unauthenticated callback accepted")
        namespace["disable_sdk_sessions"](Backend)
        fake = ModuleType("aiohttp")
        fake.web = SimpleNamespace(Response=lambda **kwargs: SimpleNamespace(**kwargs))
        with patch.dict(sys.modules, {"aiohttp": fake}):
            backend = Backend()
            for name in ("session_create_handler", "session_end_handler", "session_get_handler", "session_health_handler"):
                response = asyncio.run(getattr(backend, name)(None))
                self.assertEqual(response.status, 404)
            self.assertFalse(backend.called)

    def test_non_batch_image_packages_shared_pool_and_imports_in_isolation(self):
        root = Path(__file__).resolve().parents[1]
        self.assertIn("asr/worker.py asr/rp_handler.py asr/parakeet_pool.py", (root / "asr.Dockerfile").read_text())
        with tempfile.TemporaryDirectory() as directory:
            for name in ("worker.py", "parakeet_pool.py", "rp_handler.py"):
                shutil.copy(root / "asr" / name, directory)
            probe = "import sys; sys.path.insert(0, sys.argv[1]); import worker; assert worker.ParakeetPool"
            result = subprocess.run([sys.executable, "-I", "-c", probe, directory], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_audio_download_has_wall_clock_deadline_even_when_peer_trickles(self):
        from asr import batch_runtime
        closed = threading.Event()
        class Socket:
            def shutdown(self, *_): closed.set()
        class Response:
            status = 200
            def read1(self, _):
                time.sleep(.01)
                if closed.is_set(): raise OSError("synthetic deadline")
                return b"x"
        class Connection:
            def __init__(self, *_args, **_kwargs): self.sock = Socket()
            def request(self, *_args, **_kwargs): pass
            def getresponse(self): return Response()
            def close(self): closed.set()
        with tempfile.TemporaryDirectory() as directory, patch.object(batch_runtime.http.client, "HTTPSConnection", Connection):
            started = time.monotonic()
            with self.assertRaises((TimeoutError, OSError)):
                batch_runtime.fetch_audio("https://synthetic.example/audio", Path(directory) / "audio", deadline_seconds=.05)
            self.assertLess(time.monotonic() - started, .5)

    def test_jit_refresh_occurs_when_queued_download_starts(self):
        output = io.BytesIO()
        with wave.open(output, "wb") as audio:
            audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000); audio.writeframes(bytes(32000))
        body = output.getvalue()
        params = {"request_version": pin.REQUEST_VERSION, "model_id": pin.MODEL_ID, "model_revision": pin.MODEL_REVISION, "audio_duration_seconds": 1, "audio_receipt": {"sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body), "media_type": "audio/wav", "canonical_sample_rate": 16000, "canonical_channel_count": 1, "canonical_frame_count": 16000}}
        first_started, release = threading.Event(), threading.Event()
        refreshed = []
        events = queue.Queue()
        items = [{"id": format(i, "064x"), "audio_url": "expired", "audio_refresh": {"endpoint": "synthetic", "token": str(i)}, "lane": "parakeet_v3", "params": params} for i in range(9)]
        def refresh(capability):
            refreshed.append(capability["token"])
            return "fresh-" + capability["token"]
        def fetch(url, path):
            self.assertTrue(url.startswith("fresh-"))
            first_started.set(); release.wait(3); path.write_bytes(body)
        class Runtime:
            def transcribe(self, request): return pin.build_candidate(1, [])
        def run():
            events.put(list(batch_events(Runtime(), {"batch_id": "synthetic", "items": items}, fetcher=fetch, refresher=refresh, keepalive_seconds=.01)))
        thread = threading.Thread(target=run); thread.start()
        try:
            self.assertTrue(first_started.wait(1))
            self.assertNotIn("8", refreshed)
        finally:
            release.set(); thread.join(4)
        self.assertFalse(thread.is_alive())
        self.assertEqual(set(refreshed), {str(i) for i in range(9)})
        self.assertEqual(sum(item["type"] == "item" for item in events.get(timeout=1)), 9)

    def test_schema_and_live_candidate_limit_match_backend_50000(self):
        root = Path(__file__).resolve().parent
        for version in (2, 3):
            schema = json.loads((root / "schemas" / f"asr-candidate-v{version}.schema.json").read_text())
            self.assertEqual(schema["properties"]["segments"]["maxItems"], 50000)
        self.assertEqual(pin.MAX_CANDIDATE_SEGMENTS, 50000)
        segment = {"start_seconds": 0, "end_seconds": .1, "text": "synthetic", "confidence": .9}
        self.assertEqual(len(pin.build_candidate(1, [segment.copy() for _ in range(5001)])["segments"]), 5001)
        with self.assertRaises(pin.ContractError):
            pin.build_candidate(1, [segment] * 50001)


    def test_sdk_startup_disables_session_callbacks_before_worker_runs(self):
        called = []
        captured = {}
        class Backend:
            async def session_create_handler(self, request):
                called.append("untrusted callback")
                raise AssertionError("session callback accepted")
        class Worker:
            def __init__(self, config): pass
            def run(self):
                response = asyncio.run(Backend().session_create_handler(None))
                captured["status"] = response.status
        sdk = ModuleType("vastai"); sdk.__path__ = []
        sdk.Worker = Worker
        for name in ("BenchmarkConfig", "HandlerConfig", "LogActionConfig", "WorkerConfig"):
            setattr(sdk, name, lambda **kwargs: kwargs)
        serverless = ModuleType("vastai.serverless"); serverless.__path__ = []
        package = ModuleType("vastai.serverless.server"); package.__path__ = []
        library = ModuleType("vastai.serverless.server.lib")
        library.backend = SimpleNamespace(Backend=Backend)
        library.metrics = SimpleNamespace()
        aiohttp = ModuleType("aiohttp")
        aiohttp.web = SimpleNamespace(Response=lambda **kwargs: SimpleNamespace(**kwargs))
        with patch.dict(sys.modules, {"vastai": sdk, "vastai.serverless": serverless, "vastai.serverless.server": package, "vastai.serverless.server.lib": library, "aiohttp": aiohttp}):
            runpy.run_path(str(Path(__file__).with_name("vast-pyworker.py")), run_name="__main__")
        self.assertEqual(captured["status"], 404)
        self.assertEqual(called, [])


if __name__ == "__main__": unittest.main()
