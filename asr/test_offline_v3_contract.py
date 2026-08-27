import hashlib
import importlib.util
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "asr" / "offline_entrypoint.py"
spec = importlib.util.spec_from_file_location("offline_entrypoint", ENTRYPOINT)
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)


class OfflineV3ContractTest(unittest.TestCase):
    def test_image_includes_only_the_v3_adapter(self):
        dockerfile = (ROOT / "asr-v3.Dockerfile").read_text()
        self.assertIn("COPY asr/vast_adapter.py /workspace/vast_adapter.py", dockerfile)
        self.assertIn("COPY asr/server.py /workspace/server.py", dockerfile)
        self.assertIn("pip install --no-deps --no-index --find-links /bootstrap --require-hashes", dockerfile)
        self.assertIn("pip install --no-deps --only-binary=:all: --require-hashes", dockerfile)
        self.assertIn("COPY asr/drain_worker.py /workspace/drain_worker.py", dockerfile)
        self.assertIn("ENTRYPOINT [\"python3\", \"/workspace/drain_worker.py\"]", dockerfile)
        self.assertIn("ADD --checksum=sha256:" + offline.MODEL_SHA256, dockerfile)
        self.assertNotIn("COPY vendor/models", dockerfile)
        for wheel in (
            "antlr4_python3_runtime-4.9.3-py3-none-any.whl",
            "docopt-0.6.2-py2.py3-none-any.whl",
            "texterrors-0.4.4-cp311-cp311-linux_x86_64.whl",
            "wget-3.2-py3-none-any.whl",
        ):
            self.assertIn("COPY vendor/wheelhouses/parakeet-v3/" + wheel + " /bootstrap/", dockerfile)
        bootstrap = (ROOT / "locks" / "parakeet-v3.bootstrap.requirements.txt").read_text().splitlines()
        self.assertEqual(bootstrap, [line for line in (ROOT / "locks" / "parakeet-v3.requirements.txt").read_text().splitlines() if line.split("==", 1)[0] in {"antlr4-python3-runtime", "docopt", "texterrors", "wget"}])
        self.assertNotIn("rp_handler.py", dockerfile)
        self.assertNotIn("qwen", dockerfile.lower())

    def test_schema_mirror_and_calibration_digest(self):
        self.assertEqual((ROOT / "asr" / "schemas" / "asr-candidate-v2.schema.json").read_bytes(), (ROOT.parent / "backend" / "internal" / "service" / "schemas" / "asr-candidate-v2.schema.json").read_bytes())
        fixture = ROOT / "asr" / "fixtures" / "parakeet-v3-calibration.jsonl"
        expected = (ROOT / "asr" / "fixtures" / "parakeet-v3-calibration.sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), expected)
        row = json.loads(fixture.read_text())
        brier = sum((item["calibrated_confidence"] - item["correct"]) ** 2 for item in row["segments"]) / len(row["segments"])
        self.assertLessEqual(brier, .20)
        self.assertEqual(row["threshold"], .70)
        self.assertTrue(any(item["calibrated_confidence"] >= .70 for item in row["segments"]))
        self.assertTrue(any(item["calibrated_confidence"] < .70 for item in row["segments"]))

    def test_candidate_has_aligned_selector_evidence(self):
        candidate = offline.build_candidate(2, [
            {"start_seconds": 0, "end_seconds": 1, "text": "accepted", "confidence": .9},
            {"start_seconds": 1, "end_seconds": 2, "text": "selected", "confidence": .4},
        ])
        self.assertEqual(candidate["selected_segment_indexes"], [1])
        self.assertEqual(candidate["calibration"]["segment_evidence"][1]["timestamp_end_seconds"], 2)

    def test_extracts_real_nemo_word_timestamps_and_parallel_confidence(self):
        result = SimpleNamespace(
            timestamp={"word": [{"word": "accepted", "start": 0.0, "end": 1.0}, {"word": "selected", "start": 1.0, "end": 2.0}]},
            word_confidence=[.9, .4],
        )
        self.assertEqual(offline.extract_aligned_words(result), [
            {"start_seconds": 0.0, "end_seconds": 1.0, "text": "accepted", "confidence": .9},
            {"start_seconds": 1.0, "end_seconds": 2.0, "text": "selected", "confidence": .4},
        ])
        with self.assertRaises(offline.ContractError):
            offline.extract_aligned_words(SimpleNamespace(timestamp=result.timestamp, word_confidence=[.9]))

    def test_request_rejects_url_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "request.json"
            path.write_text(json.dumps({"request_version": offline.REQUEST_VERSION, "lane": offline.LANE, "model_id": offline.MODEL_ID, "model_revision": offline.MODEL_REVISION, "audio_path": "https://example.test/audio.wav", "audio_duration_seconds": 1}))
            with self.assertRaises(offline.ContractError):
                offline.read_request(path)

    def test_static_runtime_boundaries(self):
        source = ENTRYPOINT.read_text()
        image = (ROOT / "asr-v3.Dockerfile").read_text()
        self.assertIn('MODEL_PATH = Path("/workspace/models/parakeet-tdt-0.6b-v3.nemo")', source)
        self.assertNotIn('add_argument("--model-path"', source)
        for control in ("INPUT_ROOT", "OUTPUT_ROOT", "MAX_AUDIO_BYTES", "AUDIO_MAGIC", "is_symlink", "relative_to"):
            self.assertIn(control, source)
        self.assertIn("USER 65532:65532", image)
        self.assertIn("mkdir /workspace/input /workspace/output", image)
        runtime = " ".join((ROOT / "README.md").read_text().split())
        for control in ("read-only root filesystem", "drop every Linux capability", "no-new-privileges"):
            self.assertIn(control, runtime)

    def test_static_image_and_workflow_exclusions(self):
        source = (ROOT / "asr-v3.Dockerfile").read_text().lower() + ENTRYPOINT.read_text().lower()
        for excluded in ("runpod", "whisper", "qwen", "pyannote"):
            self.assertNotIn(excluded, source)
        workflow = (ROOT / ".github" / "workflows" / "publish-asr-v3.yml").read_text()
        self.assertIn("candidate-${{ github.sha }}", workflow)
        self.assertIn('docker pull "$IMAGE@$digest"', workflow)
        self.assertIn('docker save "$IMAGE@$digest" -o image.tar', workflow)
        self.assertIn("scan image --archive /workspace/image.tar", workflow)
        self.assertNotIn("/usr/bin/docker", workflow)
        build_step = workflow.split("name: Build one immutable ASR candidate", 1)[1]
        self.assertIn("set -euo pipefail", build_step)
        self.assertNotIn("continue-on-error", build_step)
        self.assertLess(build_step.index('docker pull "$IMAGE@$digest"'), build_step.index('docker save "$IMAGE@$digest"'))
        self.assertNotIn("vast-bench-diar", workflow)
        manifest = json.loads((ROOT / "locks" / "parakeet-v3.artifacts.json").read_text())
        self.assertFalse(manifest["resolution_required"])
        self.assertEqual(manifest["models"][0]["sha256"], offline.MODEL_SHA256)
        self.assertEqual(manifest["models"][0]["revision"], offline.MODEL_REVISION)
        self.assertIn(manifest["models"][0]["revision"], manifest["models"][0]["source"])


if __name__ == "__main__":
    unittest.main()
