import hashlib
import io
import tempfile
import threading
import unittest
import wave
from pathlib import Path

from asr.batch_runtime import batch_events
from asr.startup import Stages, verify_extracted_model
from asr import offline_entrypoint as pin


class HourlyBatchTest(unittest.TestCase):
    def test_stage_two_failure_retries_only_stage_two(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            stages = Stages(directory, sleep=lambda _: None, log=lambda *_args, **_kwargs: None)
            stages.run("deps", lambda: calls.append("deps"))
            def verify():
                calls.append("model_files")
                if calls.count("model_files") == 1:
                    raise ValueError("synthetic failure")
            stages.run("model_files", verify)
            stages.run("deps", lambda: self.fail("completed stage repeated"))
            stages.run("lane_1", lambda: calls.append("lane_1"))
            self.assertEqual(calls, ["deps", "model_files", "model_files", "lane_1"])
            self.assertEqual({p.name for p in Path(directory).glob("*.done")}, {"deps.done", "model_files.done", "lane_1.done"})

    def test_id_mapping_and_per_item_failure_isolation_with_three_parallel_lanes(self):
        output = io.BytesIO()
        with wave.open(output, "wb") as audio:
            audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000); audio.writeframes(bytes(32000))
        body = output.getvalue()
        receipt = {"sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body), "media_type": "audio/wav", "canonical_sample_rate": 16000, "canonical_channel_count": 1, "canonical_frame_count": 16000}
        params = {"request_version": pin.REQUEST_VERSION, "model_id": pin.MODEL_ID, "model_revision": pin.MODEL_REVISION, "audio_duration_seconds": 1, "audio_receipt": receipt}
        payload = {"batch_id": "synthetic-batch", "items": [{"id": character * 64, "audio_url": "bad" if character == "d" else "synthetic", "lane": "parakeet_v3", "params": params} for character in "abcd"]}
        barrier = threading.Barrier(3)
        class Runtime:
            def transcribe(self, request):
                barrier.wait(timeout=2)
                return pin.build_candidate(1, [])
        def fetch(url, path):
            if url == "bad": raise ValueError("synthetic failure")
            path.write_bytes(body)
        events = list(batch_events(Runtime(), payload, fetcher=fetch, keepalive_seconds=.1))
        self.assertEqual(events[0]["type"], "accepted")
        self.assertEqual(events[-1]["type"], "end")
        results = {item["id"]: item for item in events if item["type"] == "item"}
        self.assertEqual(set(results), {c * 64 for c in "abcd"})
        self.assertEqual(results["d" * 64]["status"], "failed")
        self.assertTrue(all(results[c * 64]["status"] == "succeeded" for c in "abc"))

    def test_duplicate_ids_rejected_before_acceptance(self):
        payload = {"batch_id": "synthetic", "items": [{"id": "a" * 64}, {"id": "a" * 64}]}
        with self.assertRaises(ValueError): next(batch_events(None, payload))


if __name__ == "__main__":
    unittest.main()
