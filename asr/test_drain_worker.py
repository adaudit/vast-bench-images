import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("drain_worker", ROOT / "asr" / "drain_worker.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class DrainWorkerTest(unittest.TestCase):
    def test_single_claim_is_bounded_and_finishes_independently(self):
        lease = {"Attempt": 1}
        calls = []
        def control(_, __, path, payload):
            calls.append((path, payload))
            if path.endswith("/claim"): return 200, {"lease": lease, "audio_url": "https://audio", "safe_parameters": {"audio_duration_seconds": 1}}
            if path.endswith("/heartbeat"): return 200, {"lease": lease}
            if path.endswith("/publish-intent"): return 200, {"url": "https://put", "object_key":"key", "upload_generation":1}
            return 204, {}
        class Put:
            headers = {"x-amz-version-id":"version"}
            def __enter__(self): return self
            def __exit__(self, *_): pass
        with patch.object(worker, "request", side_effect=control), patch.object(worker, "candidates", return_value=[{"schema_version":"asr-candidate-v2","segments":[]}]), patch.object(worker.urllib.request, "urlopen", return_value=Put()):
            self.assertTrue(worker.run_once({"base":"https://control","token":"grant","worker_id":"one"}))
        self.assertEqual([path for path, _ in calls], ["/internal/call-pipeline/asr/claim", "/internal/call-pipeline/asr/heartbeat", "/internal/call-pipeline/asr/publish-intent", "/internal/call-pipeline/asr/complete"])
        self.assertEqual(calls[0][1], {"worker_id":"one"})

    def test_window_halves_oom_and_does_not_block_siblings(self):
        leases = [{"Attempt": 1, "id": n} for n in range(3)]
        calls, batches = [], []
        def control(_, __, path, payload):
            calls.append((path, payload))
            if path.endswith("/claim"):
                return (200, {"lease": leases.pop(0), "audio_url": "https://audio", "safe_parameters": {"audio_duration_seconds": 1}}) if leases else (204, {})
            if path.endswith("/heartbeat"): return 200, {"lease": payload["lease"]}
            if path.endswith("/publish-intent"): return 200, {"url": "https://put", "object_key":"key", "upload_generation":1}
            return 204, {}
        class Put:
            headers = {"x-amz-version-id":"version"}
            def __enter__(self): return self
            def __exit__(self, *_): pass
        def batch(items):
            batches.append(len(items))
            if len(items) > 1: raise RuntimeError("CUDA out of memory")
            return [{"schema_version":"asr-candidate-v2","segments":[],"job": item["lease"]["id"]} for item in items]
        with patch.object(worker, "request", side_effect=control), patch.object(worker, "candidates", side_effect=batch), patch.object(worker.urllib.request, "urlopen", return_value=Put()):
            self.assertTrue(worker.run_window({"base":"https://control","token":"grant","worker_id":"one"}, cap=3))
        self.assertEqual(batches, [3, 1, 2, 1, 1])
        self.assertEqual(sum(1 for path, _ in calls if path.endswith("/complete")), 3)


if __name__ == "__main__": unittest.main()
