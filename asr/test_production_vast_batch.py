import base64
import asyncio
import copy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("production_vast_batch", ROOT / "asr" / "production_vast_batch.py")
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class ProductionVastBatchTest(unittest.TestCase):
    def source(self, directory, name, body, duration):
        path = Path(directory, name)
        path.write_bytes(body)
        return bridge.Source(path, duration, hashlib.sha256(body).hexdigest())

    def speech_candidate(self):
        return {"schema_version": "asr-candidate-v3", "disposition": "speech", "lane": bridge.drain_worker.LANE, "model_id": bridge.drain_worker.MODEL_ID, "model_revision": bridge.drain_worker.MODEL_REVISION, "audio_duration_seconds": 1.0, "segments": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "ok", "confidence": .9}], "selected_segment_indexes": [], "calibration": {"corpus_sha256": "a" * 64, "metric": "segment_brier_score", "threshold": .7, "decision_rule": "calibrated_confidence < threshold", "segment_evidence": [{"segment_index": 0, "raw_confidence": .9, "calibrated_confidence": .9, "timestamp_start_seconds": 0.0, "timestamp_end_seconds": 1.0}]}}

    def test_payload_is_exact_existing_serialization_in_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = (
                self.source(directory, "first.wav", b"RIFFfirst", 1.0),
                self.source(directory, "second.wav", b"RIFFsecond", 2.0),
                self.source(directory, "third.wav", b"RIFFthird", 3.0),
                self.source(directory, "fourth.wav", b"RIFFfourth", 4.0),
            )
            got = bridge.payload(sources)
            expected = {"requests": [bridge.drain_worker.candidate_payload(bridge.verified_claim(source)) for source in sources]}
            self.assertEqual(got, expected)
            self.assertEqual(len(got["requests"]), 4)
            self.assertEqual([base64.b64decode(item["audio_base64"]) for item in got["requests"]], [source.path.read_bytes() for source in sources])
            expected_keys = {"request_version", "lane", "model_id", "model_revision", "audio_filename", "audio_duration_seconds", "audio_base64"}
            self.assertEqual([set(item) for item in got["requests"]], [expected_keys] * 4)

    def test_rejects_hash_mismatch_before_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory, "source.wav", b"RIFFsource", 1.0)
            source.path.write_bytes(b"RIFFchanged")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                bridge.payload((source,))

    def test_bridge_has_no_raw_source_canonicalization_path(self):
        self.assertFalse(hasattr(bridge, "canonicalize"))
        self.assertNotIn("ffmpeg", (ROOT / "asr" / "production_vast_batch.py").read_text())

    def test_rejects_non_wav_before_external_request(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory, "source.m4a", b"ftypM4A ", 1.0)
            with self.assertRaisesRegex(ValueError, "not canonical WAV"):
                bridge.payload((source,))

    def test_validated_batch_requires_every_ordered_candidate(self):
        request = {"audio_duration_seconds": 1.0}
        candidate = self.speech_candidate()
        self.assertEqual(bridge.validate_batch([candidate], [request]), [candidate])
        for bad in ([], {}, [dict(candidate, segments=[dict(candidate["segments"][0], text=" \t")])], [dict(candidate, calibration={"segment_evidence": candidate["calibration"]["segment_evidence"]})]):
            with self.assertRaises(ValueError):
                bridge.validate_batch(bad, [request])

    def test_validated_batch_rejects_non_contract_calibration_and_shapes(self):
        request = {"audio_duration_seconds": 1.0}
        variants = []
        for field, value in (("corpus_sha256", "not-a-digest"), ("metric", "wer"), ("threshold", .2), ("decision_rule", "below_threshold")):
            candidate = copy.deepcopy(self.speech_candidate()); candidate["calibration"][field] = value; variants.append(candidate)
        candidate = copy.deepcopy(self.speech_candidate()); candidate["calibration"]["segment_evidence"][0]["calibrated_confidence"] = True; variants.append(candidate)
        candidate = copy.deepcopy(self.speech_candidate()); candidate["unexpected"] = True; variants.append(candidate)
        candidate = copy.deepcopy(self.speech_candidate()); candidate["segments"][0]["unexpected"] = True; variants.append(candidate)
        candidate = copy.deepcopy(self.speech_candidate()); candidate["calibration"]["segment_evidence"][0]["unexpected"] = True; variants.append(candidate)
        candidate = copy.deepcopy(self.speech_candidate()); candidate["calibration"]["segment_evidence"][0]["segment_index"] = False; variants.append(candidate)
        for candidate in variants:
            with self.assertRaises(ValueError):
                bridge.validate_batch([candidate], [request])

    def test_load_sources_requires_distinct_recording_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory, "source.wav", b"RIFFsource", 1.0)
            manifest = Path(directory, "manifest.json")
            manifest.write_text(__import__("json").dumps({"sources": [{"path": str(source.path), "duration_seconds": source.duration_seconds, "sha256": source.sha256}] * 4}))
            with self.assertRaisesRegex(ValueError, "distinct"):
                bridge.load_sources(manifest)

    def test_load_sources_rejects_same_digest_at_distinct_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = tuple(self.source(directory, f"source-{index}.wav", b"RIFFsame" if index < 2 else b"RIFF" + bytes([index]), 1.0 + index) for index in range(4))
            manifest = Path(directory, "manifest.json")
            manifest.write_text(__import__("json").dumps({"sources": [{"path": str(source.path), "duration_seconds": source.duration_seconds, "sha256": source.sha256} for source in sources]}))
            with self.assertRaisesRegex(ValueError, "distinct"):
                bridge.load_sources(manifest)

    def test_submit_resolves_exact_target_before_request(self):
        events = []
        class Control:
            def resolve_endpoint_target(self, endpoint):
                events.append("target")
                return bridge.failure_guard.Target(endpoint, 9, (10,))
        class Endpoint:
            def request(self, path, body, **kwargs):
                events.append("request")
                return [{"schema_version": "asr-candidate-v3", "disposition": "no_speech", "lane": bridge.drain_worker.LANE, "model_id": bridge.drain_worker.MODEL_ID, "model_revision": bridge.drain_worker.MODEL_REVISION, "audio_duration_seconds": request["audio_duration_seconds"], "segments": [], "selected_segment_indexes": [], "calibration": {"corpus_sha256": "a" * 64, "metric": "segment_brier_score", "threshold": .7, "decision_rule": "calibrated_confidence < threshold", "segment_evidence": []}} for request in body["requests"]]
        class Factory:
            async def get_endpoint(self, *, name): return Endpoint()
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory, "source.wav", b"RIFFsource", 1.0)
            __import__("asyncio").run(bridge.submit_once((source,), Control(), Path(directory, "guard.json"), Factory(), "attempt"))
        self.assertEqual(events, ["target", "request"])

    def test_submit_rejects_invalid_target_before_endpoint_request(self):
        events = []
        class Control:
            def resolve_endpoint_target(self, endpoint):
                events.append("target")
                return bridge.failure_guard.Target(endpoint, 0, (10,))
        class Factory:
            async def get_endpoint(self, *, name):
                events.append("endpoint")
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory, "source.wav", b"RIFFsource", 1.0)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                __import__("asyncio").run(bridge.submit_once((source,), Control(), Path(directory, "guard.json"), Factory(), "attempt"))
        self.assertEqual(events, ["target"])

    def test_main_makes_one_guarded_fake_submission(self):
        class Control:
            def resolve_endpoint_target(self, endpoint):
                return bridge.failure_guard.Target(endpoint, 9, (10,))
        class Endpoint:
            def __init__(self): self.calls = 0
            def request(self, path, body, **kwargs):
                self.calls += 1
                return [{"schema_version": "asr-candidate-v3", "disposition": "no_speech", "lane": bridge.drain_worker.LANE, "model_id": bridge.drain_worker.MODEL_ID, "model_revision": bridge.drain_worker.MODEL_REVISION, "audio_duration_seconds": request["audio_duration_seconds"], "segments": [], "selected_segment_indexes": [], "calibration": {"corpus_sha256": "a" * 64, "metric": "segment_brier_score", "threshold": .7, "decision_rule": "calibrated_confidence < threshold", "segment_evidence": []}} for request in body["requests"]]
        class Factory:
            def __init__(self, endpoint): self.endpoint = endpoint
            async def get_endpoint(self, *, name):
                self.name = name
                return self.endpoint
        with tempfile.TemporaryDirectory() as directory:
            sources = tuple(self.source(directory, f"source-{index}.wav", b"RIFFsource" + bytes([index]), 1.0 + index) for index in range(4))
            manifest = Path(directory, "manifest.json")
            manifest.write_text(__import__("json").dumps({"sources": [{"path": str(source.path), "duration_seconds": source.duration_seconds, "sha256": source.sha256} for source in sources]}))
            endpoint = Endpoint()
            bridge.main(["--manifest", str(manifest), "--state-path", str(Path(directory, "guard.json"))], control_client=Control(), endpoint_factory=Factory(endpoint))
            self.assertEqual(endpoint.calls, 1)


if __name__ == "__main__":
    unittest.main()
