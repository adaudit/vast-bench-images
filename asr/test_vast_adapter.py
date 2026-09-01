import base64
import importlib.util
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vast_adapter", ROOT / "asr" / "vast_adapter.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
SERVER_SPEC = importlib.util.spec_from_file_location("asr_server", ROOT / "asr" / "server.py")


class VastAdapterTest(unittest.TestCase):
    @staticmethod
    def _model():
        return type("Model", (), {"transcribe": lambda *_args, **_kwargs: []})()

    def test_initialize_once_restores_three_lanes_and_health_never_retries(self):
        server = importlib.util.module_from_spec(SERVER_SPEC); SERVER_SPEC.loader.exec_module(server)
        entered, release = threading.Event(), threading.Event()
        loads = []

        def load():
            loads.append(1); entered.set(); release.wait(2); return self._model()

        runtime = server.Runtime(model_verifier=lambda _: None, model_loader=load)
        starter = threading.Thread(target=runtime.initialize_once); starter.start()
        self.assertTrue(entered.wait(1))
        observed = []
        probes = [threading.Thread(target=lambda: observed.append(runtime.health())) for _ in range(4)]
        [probe.start() for probe in probes]; [probe.join() for probe in probes]
        self.assertEqual(observed, [(503, {"status": "loading"})] * 4)
        with self.assertRaises(server.NotReadyError):
            runtime.transcribe_batch([])
        release.set(); starter.join()
        self.assertEqual((len(loads), runtime.health()), (3, (200, {"status": "ready"})))

    def test_initialize_failure_is_terminal(self):
        server = importlib.util.module_from_spec(SERVER_SPEC); SERVER_SPEC.loader.exec_module(server)
        loads = []
        runtime = server.Runtime(model_verifier=lambda _: None, model_loader=lambda: loads.append(1) or (_ for _ in ()).throw(RuntimeError("boom")))
        runtime.initialize_once()
        self.assertEqual(runtime.health(), (503, {"status": "failed", "cause": "initialization failed"}))
        runtime.initialize_once()
        self.assertEqual(len(loads), 1)

    def test_unusable_or_aliased_lanes_are_terminal_failures(self):
        server = importlib.util.module_from_spec(SERVER_SPEC); SERVER_SPEC.loader.exec_module(server)
        for loader in (lambda: None, lambda: object(), lambda shared=self._model(): shared):
            runtime = server.Runtime(model_verifier=lambda _: None, model_loader=loader)
            runtime.initialize_once()
            self.assertEqual(runtime.health(), (503, {"status": "failed", "cause": "initialization failed"}))

    def test_failure_cause_is_stable_and_sanitized(self):
        server = importlib.util.module_from_spec(SERVER_SPEC); SERVER_SPEC.loader.exec_module(server)
        loads = []
        runtime = server.Runtime(model_verifier=lambda _: None, model_loader=lambda: loads.append(1) or (_ for _ in ()).throw(RuntimeError("/tmp/private-model secret")))
        runtime.initialize_once()
        self.assertEqual(runtime.health(), (503, {"status": "failed", "cause": "initialization failed"}))
        self.assertNotIn("/tmp", str(runtime.health()))
        runtime.initialize_once()
        self.assertEqual(len(loads), 1)

    def test_runtime_readiness_and_transcribe_contract(self):
        server = importlib.util.module_from_spec(SERVER_SPEC)
        SERVER_SPEC.loader.exec_module(server)
        runtime = server.Runtime(
            model_verifier=lambda _: None,
            model_loader=self._model,
            transcriber=lambda request: [{"start_seconds": 0, "end_seconds": request.audio_duration_seconds, "text": "ok", "confidence": .9}],
        )
        runtime.initialize_once()
        payload = {"request_version": adapter.REQUEST_VERSION, "lane": adapter.LANE, "model_id": adapter.MODEL_ID, "model_revision": adapter.MODEL_REVISION, "audio_filename": "sample.wav", "audio_duration_seconds": 1, "audio_base64": base64.b64encode(b"RIFFtest").decode()}
        self.assertEqual(runtime.transcribe(payload)["segments"][0]["text"], "ok")

    def test_batch_loads_once_and_keeps_input_order(self):
        server = importlib.util.module_from_spec(SERVER_SPEC); SERVER_SPEC.loader.exec_module(server)
        loads, calls = [], []
        runtime = server.Runtime(model_verifier=lambda _: None, model_loader=lambda: loads.append(1) or self._model(), batch_transcriber=lambda requests: calls.append(len(requests)) or [[{"start_seconds": 0, "end_seconds": r.audio_duration_seconds, "text": r.audio_filename, "confidence": .9}] for r in requests])
        payload = lambda name: {"request_version": adapter.REQUEST_VERSION, "lane": adapter.LANE, "model_id": adapter.MODEL_ID, "model_revision": adapter.MODEL_REVISION, "audio_filename": name, "audio_duration_seconds": 1, "audio_base64": base64.b64encode(b"RIFFtest").decode()}
        runtime.initialize_once()
        got = runtime.transcribe_batch([payload("a.wav"), payload("b.wav")])
        self.assertEqual((loads, calls, [x["segments"][0]["text"] for x in got]), ([1, 1, 1], [2], ["a.wav", "b.wav"]))
    def test_request_keeps_short_audio_as_one_chunk(self):
        request = adapter.parse_request({
            "request_version": adapter.REQUEST_VERSION,
            "lane": adapter.LANE,
            "model_id": adapter.MODEL_ID,
            "model_revision": adapter.MODEL_REVISION,
            "audio_filename": "sample.wav",
            "audio_duration_seconds": 10,
            "audio_base64": base64.b64encode(b"RIFFtest").decode(),
        })
        self.assertEqual(request.chunks, ((0.0, 10.0),))

    def test_chunks_at_silence_and_restitches_in_original_order(self):
        chunks = adapter.chunk_ranges(120.0, silence_points=(60.0,))
        self.assertEqual(chunks, ((0.0, 61.0), (59.0, 120.0)))
        result = adapter.batch_and_restitch(
            [[{"start_seconds": 0, "end_seconds": 1, "text": "first", "confidence": .9}],
             [{"start_seconds": 1, "end_seconds": 2, "text": "second", "confidence": .8}]],
            chunks,
        )
        self.assertEqual([segment["text"] for segment in result], ["first", "second"])
        self.assertEqual([segment["start_seconds"] for segment in result], [0.0, 60.0])

    def test_restitch_removes_a_segment_owned_by_the_prior_overlap(self):
        result = adapter.batch_and_restitch(
            [[{"start_seconds": 58, "end_seconds": 60, "text": "owned", "confidence": .9}],
             [{"start_seconds": 0, "end_seconds": 1, "text": "owned", "confidence": .9},
              {"start_seconds": 1, "end_seconds": 2, "text": "next", "confidence": .8}]],
            ((0.0, 61.0), (59.0, 120.0)),
        )
        self.assertEqual([segment["text"] for segment in result], ["owned", "next"])

    def test_rejects_foreign_request_identity(self):
        with self.assertRaises(adapter.ContractError):
            adapter.parse_request({"request_version": "wrong"})


if __name__ == "__main__":
    unittest.main()
