import base64
import io
import json
import tempfile
import threading
import unittest
import urllib.request
import wave
from pathlib import Path
from urllib.error import HTTPError

from diar import server


class FakeModel:
    def __init__(self, segments):
        self.segments = segments
        self.sortformer_modules = type("Modules", (), {})()

    def eval(self):
        return self

    def diarize(self, **_kwargs):
        return [self.segments]


def wav_bytes(*, channels=1, rate=16000, frames=16000):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * frames * channels)
    return output.getvalue()


class DiarServerTest(unittest.TestCase):
    def runtime(self, segments=("0.0 1.0 speaker_a",)):
        model = FakeModel(segments)
        clock_values = iter((10.0, 10.125))
        def clock():
            try:
                return next(clock_values)
            except StopIteration:
                return 10.125
        runtime = server.Runtime(
            model_verifier=lambda _: None,
            model_loader=lambda: model,
            clock=clock,
            peak_memory=lambda: 321,
            reset_peak=lambda: None,
        )
        runtime.initialize_once()
        return runtime, model

    @staticmethod
    def request(url, data=None, content_type=None):
        request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            body = json.loads(error.read())
            error.close()
            return error.code, body

    def serve(self, runtime):
        http = server.make_server(("127.0.0.1", 0), runtime)
        thread = threading.Thread(target=http.serve_forever)
        thread.start()
        self.addCleanup(lambda: (http.shutdown(), thread.join(), http.server_close()))
        return f"http://127.0.0.1:{http.server_port}"

    def test_turns_use_first_chronological_appearance(self):
        self.assertEqual(
            server.turns(((1, 2, "B"), (0, 1, "A"), (2, 3, "B"), (3, 3, "discard"))),
            [
                {"start_s": 0.0, "end_s": 1.0, "speaker_idx": 0},
                {"start_s": 1.0, "end_s": 2.0, "speaker_idx": 1},
                {"start_s": 2.0, "end_s": 3.0, "speaker_idx": 1},
            ],
        )

    def test_runtime_configures_model_and_reports_metrics(self):
        runtime, model = self.runtime(("1.0 2.0 speaker_b", "0.0 1.0 speaker_a", "2.0 3.0 speaker_b"))
        self.assertTrue(runtime.ready)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output:
            path = Path(output.name)
        try:
            path.write_bytes(wav_bytes())
            response = runtime.diarize(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(response["engine"], "sortformer")
        self.assertEqual(response["model_revision"], server.MODEL_REVISION)
        self.assertEqual(response["speaker_turns"], [
            {"start_s": 0.0, "end_s": 1.0, "speaker_idx": 0},
            {"start_s": 1.0, "end_s": 2.0, "speaker_idx": 1},
            {"start_s": 2.0, "end_s": 3.0, "speaker_idx": 1},
        ])
        self.assertEqual(response["metrics"], {"elapsed_ms": 125, "audio_seconds": 1.0, "num_speakers": 2, "peak_vram_mb": 321})
        self.assertEqual(model.diarize.__name__, "diarize")

    def test_json_and_multipart_requests_reach_diarize(self):
        runtime, _ = self.runtime()
        address = self.serve(runtime)
        audio = wav_bytes()
        json_body = json.dumps({"audio_base64": base64.b64encode(audio).decode()}).encode()
        status, response = self.request(address + "/diarize", json_body, "application/json")
        self.assertEqual(status, 200)
        self.assertEqual(response["metrics"]["audio_seconds"], 1.0)
        boundary = "diar-test-boundary"
        multipart = b"\r\n".join((
            f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="audio"; filename="audio.wav"',
            b"Content-Type: audio/wav",
            b"",
            audio,
            f"--{boundary}--".encode(),
            b"",
        ))
        status, response = self.request(address + "/diarize", multipart, f"multipart/form-data; boundary={boundary}")
        self.assertEqual(status, 200)
        self.assertEqual(response["engine"], "sortformer")

    def test_http_error_paths_and_initialization_failure(self):
        runtime, _ = self.runtime()
        address = self.serve(runtime)
        self.assertEqual(self.request(address + "/healthz"), (200, {"status": "ready"}))
        self.assertEqual(self.request(address + "/diarize", b"{", "application/json"), (400, {"error": "invalid request"}))
        invalid_audio = json.dumps({"audio_base64": base64.b64encode(wav_bytes(channels=2)).decode()}).encode()
        self.assertEqual(self.request(address + "/diarize", invalid_audio, "application/json"), (400, {"error": "invalid request"}))
        self.assertEqual(self.request(address + "/diarize", b"{}", "text/plain"), (400, {"error": "invalid request"}))
        failed = server.Runtime(model_verifier=lambda _: None, model_loader=lambda: (_ for _ in ()).throw(RuntimeError("model unavailable")))
        failed.initialize_once()
        self.assertEqual(failed.health(), (503, {"status": "failed", "cause": "initialization failed", "error": "model unavailable"}))
        self.assertFalse(failed.ready)


if __name__ == "__main__":
    unittest.main()
