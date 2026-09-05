import base64
import io
import importlib.util
import struct
import sys
import threading
import unittest
import wave
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
    @staticmethod
    def _wav(frame_data, rate=16000):
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            wav.writeframes(frame_data)
        return output.getvalue()


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
        with self.assertLogs("parakeet.server", "ERROR") as logs:
            runtime.initialize_once()
        self.assertEqual(runtime.health(), (503, {"status": "failed", "cause": "initialization failed", "error": "boom"}))
        self.assertIn("RuntimeError: boom", "\n".join(logs.output))
        runtime.initialize_once()
        self.assertEqual(len(loads), 1)

    def test_unusable_or_aliased_lanes_are_terminal_failures(self):
        server = importlib.util.module_from_spec(SERVER_SPEC); SERVER_SPEC.loader.exec_module(server)
        for loader in (lambda: None, lambda: object(), lambda shared=self._model(): shared):
            runtime = server.Runtime(model_verifier=lambda _: None, model_loader=loader)
            runtime.initialize_once()
            self.assertEqual(runtime.health(), (503, {"status": "failed", "cause": "initialization failed", "error": "model lanes are unusable"}))

    def test_failure_error_is_stable_and_exposed(self):
        server = importlib.util.module_from_spec(SERVER_SPEC); SERVER_SPEC.loader.exec_module(server)
        loads = []
        runtime = server.Runtime(model_verifier=lambda _: None, model_loader=lambda: loads.append(1) or (_ for _ in ()).throw(RuntimeError("/tmp/private-model secret")))
        runtime.initialize_once()
        self.assertEqual(runtime.health(), (503, {"status": "failed", "cause": "initialization failed", "error": "/tmp/private-model secret"}))
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
        self.assertEqual(chunks, ((0.0, 61.0), (60.0, 120.0)))
        result = adapter.batch_and_restitch(
            [[{"start_seconds": 0, "end_seconds": 1, "text": "first", "confidence": .9}],
             [{"start_seconds": 1, "end_seconds": 2, "text": "second", "confidence": .8}]],
            chunks,
        )
        self.assertEqual([segment["text"] for segment in result], ["first", "second"])
        self.assertEqual([segment["start_seconds"] for segment in result], [0.0, 61.0])

    def test_restitch_removes_a_segment_owned_by_the_prior_overlap(self):
        result = adapter.batch_and_restitch(
            [[{"start_seconds": 58, "end_seconds": 60, "text": "owned", "confidence": .9}],
             [{"start_seconds": 0, "end_seconds": 1, "text": "owned", "confidence": .9},
              {"start_seconds": 1, "end_seconds": 2, "text": "next", "confidence": .8}]],
            ((0.0, 61.0), (59.0, 120.0)),
        )
        self.assertEqual([segment["text"] for segment in result], ["owned", "next"])

    def test_slice_wav_keeps_requested_frames_and_header(self):
        audio = self._wav(struct.pack("<h", 1000) * (16000 * 5))
        sliced = adapter.slice_wav(audio, 1.25, 3.75)
        self.assertTrue(sliced.startswith(b"RIFF"))
        with wave.open(io.BytesIO(sliced), "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()), (1, 2, 16000, 40000))

    def test_ten_minute_silence_and_tone_request_has_ten_chunks(self):
        frames = b"\0\0" * (16000 * 300) + struct.pack("<h", 1000) * (16000 * 300)
        audio = self._wav(frames)
        request = adapter.parse_request({
            "request_version": adapter.REQUEST_VERSION,
            "lane": adapter.LANE,
            "model_id": adapter.MODEL_ID,
            "model_revision": adapter.MODEL_REVISION,
            "audio_filename": "long.wav",
            "audio_duration_seconds": 600,
            "audio_base64": base64.b64encode(audio).decode(),
        })
        self.assertEqual((len(request.chunks), request.chunks[-1][1]), (10, 600.0))

    def test_transcribe_many_slices_restitches_and_synchronizes_each_sub_batch(self):
        server = importlib.util.module_from_spec(SERVER_SPEC); SERVER_SPEC.loader.exec_module(server)

        class Context:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False

        class Stream:
            def __init__(self):
                self.synchronizations = 0
            def synchronize(self):
                self.synchronizations += 1

        class OutOfMemoryError(RuntimeError):
            pass

        class Cuda:
            @staticmethod
            def is_available():
                return True
            @staticmethod
            def Stream():
                return Stream()
            @staticmethod
            def stream(_):
                return Context()

        Cuda.OutOfMemoryError = OutOfMemoryError
        fake_torch = type("Torch", (), {"cuda": Cuda(), "bfloat16": object(), "autocast": staticmethod(lambda **_: Context())})()
        models = []

        class Model:
            def __init__(self):
                self.batch_sizes = []
                self.chunk_index = 0
                self.fail_once = True
            def transcribe(self, paths, *, batch_size, timestamps):
                self.batch_sizes.append(batch_size)
                if self.fail_once:
                    self.fail_once = False
                    raise OutOfMemoryError()
                hypotheses = []
                for _ in paths:
                    index = self.chunk_index
                    self.chunk_index += 1
                    words = (
                        [{"word": "owned", "start": 59, "end": 61}]
                        if index == 0 else
                        [{"word": "owned", "start": 0, "end": 1}, {"word": "next", "start": 1, "end": 2}]
                        if index == 1 else
                        [{"word": f"chunk-{index}", "start": 0, "end": 1}]
                    )
                    hypotheses.append(type("Hypothesis", (), {"timestamp": {"word": words}, "word_confidence": [.9] * len(words)})())
                return hypotheses

        previous_torch = sys.modules.get("torch")
        previous_batch = getattr(server, "CHUNK_BATCH", None)
        sys.modules["torch"] = fake_torch
        server.torch = fake_torch
        server.CHUNK_BATCH = 3
        try:
            runtime = server.Runtime(model_verifier=lambda _: None, model_loader=lambda: models.append(Model()) or models[-1])
            runtime.initialize_once()
            request = adapter.Request(self._wav(b"\0\0" * (16000 * 182)), "long.wav", 182, adapter.chunk_ranges(182))
            segments = runtime._transcribe_many([request])[0]
        finally:
            if previous_torch is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = previous_torch
            if previous_batch is None:
                delattr(server, "CHUNK_BATCH")
            else:
                server.CHUNK_BATCH = previous_batch

        self.assertEqual(models[0].batch_sizes, [3, 1, 3])
        self.assertEqual(runtime.pool.instances[0].stream.synchronizations, 3)
        self.assertEqual([segment["text"] for segment in segments], ["owned", "next", "chunk-2", "chunk-3"])
        self.assertEqual([segment["start_seconds"] for segment in segments], [59.0, 61.0, 120.0, 180.0])
        self.assertTrue(all(previous["end_seconds"] <= current["end_seconds"] for previous, current in zip(segments, segments[1:])))
    def test_rejects_foreign_request_identity(self):
        with self.assertRaises(adapter.ContractError):
            adapter.parse_request({"request_version": "wrong"})


if __name__ == "__main__":
    unittest.main()
