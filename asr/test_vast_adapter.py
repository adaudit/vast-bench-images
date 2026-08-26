import base64
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vast_adapter", ROOT / "asr" / "vast_adapter.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
SERVER_SPEC = importlib.util.spec_from_file_location("asr_server", ROOT / "asr" / "server.py")


class VastAdapterTest(unittest.TestCase):
    def test_runtime_readiness_and_transcribe_contract(self):
        server = importlib.util.module_from_spec(SERVER_SPEC)
        SERVER_SPEC.loader.exec_module(server)
        runtime = server.Runtime(
            model_verifier=lambda _: None,
            transcriber=lambda request: [{"start_seconds": 0, "end_seconds": request.audio_duration_seconds, "text": "ok", "confidence": .9}],
        )
        runtime.check_ready()
        payload = {"request_version": adapter.REQUEST_VERSION, "lane": adapter.LANE, "model_id": adapter.MODEL_ID, "model_revision": adapter.MODEL_REVISION, "audio_filename": "sample.wav", "audio_duration_seconds": 1, "audio_base64": base64.b64encode(b"RIFFtest").decode()}
        self.assertEqual(runtime.transcribe(payload)["segments"][0]["text"], "ok")
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
