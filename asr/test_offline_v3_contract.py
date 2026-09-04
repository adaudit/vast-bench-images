import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
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
    def test_generated_ghcr_guard_behavior_matrix(self):
        workflow = (ROOT / ".github" / "workflows" / "publish-asr-v3.yml").read_text()
        guard = re.search(r"cat > \"\$guard\" <<'EOF'\n(.*?)\n          EOF", workflow, re.DOTALL).group(1)
        sentinel = "token-leak-sentinel-9F3k"
        unknown = json.dumps({"errors": [{"code": "MANIFEST_UNKNOWN", "message": "not found"}]})
        cases = (
            ("token", json.dumps({"token": sentinel}), 200, unknown, 404, True),
            ("access_token", json.dumps({"access_token": sentinel}), 200, unknown, 404, True),
            ("equal_dual_token", json.dumps({"token": sentinel, "access_token": sentinel}), 200, unknown, 404, True),
            ("zero_token", json.dumps({"token": "0"}), 200, unknown, 404, True),
            ("missing_token", "{}", 200, unknown, 404, False),
            ("empty_token", json.dumps({"token": ""}), 200, unknown, 404, False),
            ("whitespace_token", json.dumps({"token": "bad token"}), 200, unknown, 404, False),
            ("control_token", json.dumps({"token": "bad\\nvalue"}), 200, unknown, 404, False),
            ("array_token", json.dumps({"token": [sentinel]}), 200, unknown, 404, False),
            ("multidocument_token", json.dumps({"token": sentinel}) + "\n{}", 200, unknown, 404, False),
            ("conflicting_dual_token", json.dumps({"token": sentinel, "access_token": "other"}), 200, unknown, 404, False),
            ("whitespace_access_token", json.dumps({"access_token": "bad token"}), 200, unknown, 404, False),
            ("token_http_401", json.dumps({"token": sentinel}), 401, unknown, 404, False),
            ("token_redirect", json.dumps({"token": sentinel}), 302, unknown, 404, False),
            ("token_transport", json.dumps({"token": sentinel}), 200, unknown, 404, False, 7),
            ("token_http_500", json.dumps({"token": sentinel}), 500, unknown, 404, False),
            ("all_unknown_single", json.dumps({"token": sentinel}), 200, unknown, 404, True),
            ("all_unknown_multiple", json.dumps({"token": sentinel}), 200, json.dumps({"errors": [{"code": "MANIFEST_UNKNOWN", "message": "first"}, {"code": "MANIFEST_UNKNOWN", "message": "second"}]}), 404, True),
            ("all_unknown_extra_fields", json.dumps({"token": sentinel}), 200, json.dumps({"errors": [{"code": "MANIFEST_UNKNOWN", "message": "not found", "detail": "ignored"}]}), 404, True),
            ("occupied_200_json", json.dumps({"token": sentinel}), 200, "{}", 200, False),
            ("occupied_200_malformed", json.dumps({"token": sentinel}), 200, "not json", 200, False),
            ("manifest_http_401", json.dumps({"token": sentinel}), 200, "{}", 401, False),
            ("manifest_http_429", json.dumps({"token": sentinel}), 200, "{}", 429, False),
            ("manifest_http_500", json.dumps({"token": sentinel}), 200, "{}", 500, False),
            ("manifest_redirect", json.dumps({"token": sentinel}), 200, "{}", 302, False),
            ("mixed_manifest_errors", json.dumps({"token": sentinel}), 200, json.dumps({"errors": [{"code": "MANIFEST_UNKNOWN", "message": "not found"}, {"code": "UNAUTHORIZED", "message": "no"}]}), 404, False),
            ("empty_manifest_errors", json.dumps({"token": sentinel}), 200, json.dumps({"errors": []}), 404, False),
            ("manifest_errors_missing_code", json.dumps({"token": sentinel}), 200, json.dumps({"errors": [{"message": "not found"}]}), 404, False),
            ("manifest_errors_nonobject", json.dumps({"token": sentinel}), 200, json.dumps({"errors": ["MANIFEST_UNKNOWN"]}), 404, False),
            ("manifest_errors_missing", json.dumps({"token": sentinel}), 200, "{}", 404, False),
            ("manifest_errors_string", json.dumps({"token": sentinel}), 200, json.dumps({"errors": "MANIFEST_UNKNOWN"}), 404, False),
            ("manifest_malformed_json", json.dumps({"token": sentinel}), 200, "not json", 404, False),
            ("manifest_transport", json.dumps({"token": sentinel}), 200, unknown, 404, False, 0, 7),
            ("unknown_blank_message", json.dumps({"token": sentinel}), 200, json.dumps({"errors": [{"code": "MANIFEST_UNKNOWN", "message": ""}]}), 404, False),
            ("unknown_numeric_message", json.dumps({"token": sentinel}), 200, json.dumps({"errors": [{"code": "MANIFEST_UNKNOWN", "message": 7}]}), 404, False),
            ("unknown_with_other_document", json.dumps({"token": sentinel}), 200, unknown + "\n{}", 404, False),
        )
        self.assertEqual(len(cases), 36)
        fake_curl = '''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
is_token = "https://ghcr.io/token" in args
prefix = "TOKEN" if is_token else "MANIFEST"
output = args[args.index("--output") + 1]
open(output, "w").write(os.environ[prefix + "_BODY"])
print(os.environ[prefix + "_STATUS"], end="")
sys.exit(int(os.environ.get(prefix + "_EXIT", "0")))
'''
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            guard_path = temporary_path / "guard"
            curl_path = temporary_path / "curl"
            guard_path.write_text(guard)
            curl_path.write_text(fake_curl)
            guard_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            curl_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            for case in cases:
                name, token_body, token_status, manifest_body, manifest_status, expected, *exits = case
                token_exit, manifest_exit = (exits + [0, 0])[:2]
                environment = os.environ | {
                    "PATH": str(temporary_path) + os.pathsep + os.environ["PATH"],
                    "GITHUB_ACTOR": "actor",
                    "GITHUB_TOKEN": sentinel,
                    "IMAGE": "ghcr.io/owner/image",
                    "TAG": "candidate",
                    "TOKEN_BODY": token_body,
                    "TOKEN_STATUS": str(token_status),
                    "TOKEN_EXIT": str(token_exit),
                    "MANIFEST_BODY": manifest_body,
                    "MANIFEST_STATUS": str(manifest_status),
                    "MANIFEST_EXIT": str(manifest_exit),
                }
                result = subprocess.run(["bash", str(guard_path)], env=environment, text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode == 0, expected, name + ": " + result.stderr)
                self.assertNotIn(sentinel, result.stdout + result.stderr, name)

    def test_image_includes_only_the_v3_adapter(self):
        dockerfile = (ROOT / "asr-v3.Dockerfile").read_text()
        self.assertIn("COPY asr/vast_adapter.py /workspace/vast_adapter.py", dockerfile)
        self.assertIn("COPY asr/server.py /workspace/server.py", dockerfile)
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
        self.assertEqual((ROOT / "asr" / "schemas" / "asr-candidate-v3.schema.json").read_bytes(), (ROOT.parent / "backend" / "internal" / "service" / "schemas" / "asr-candidate-v3.schema.json").read_bytes())
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

    def test_candidate_rejects_blank_speech_text(self):
        for text in ("", " \t\n"):
            with self.assertRaises(offline.ContractError):
                offline.build_candidate(1, [{"start_seconds": 0, "end_seconds": 1, "text": text, "confidence": .9}])

    def test_candidate_rejects_more_than_one_decoder_frame_past_audio_endpoint(self):
        with self.assertRaises(offline.ContractError):
            offline.build_candidate(7.435, [{"start_seconds": 7.36, "end_seconds": 7.516, "text": "fixture", "confidence": .9}])

    def test_candidate_rejects_nonnumeric_final_end_with_contract_error(self):
        with self.assertRaises(offline.ContractError):
            offline.build_candidate(7.435, [{"start_seconds": 7.36, "end_seconds": "7.44", "text": "fixture", "confidence": .9}])

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
        staging_tag = "staging-parakeet-${{ github.sha }}-${{ github.run_id }}-${{ github.run_attempt }}"
        final_tag = "parakeet-${{ github.sha }}"
        self.assertIn(staging_tag, workflow)
        self.assertIn(final_tag, workflow)
        buildx_setup = "docker/setup-buildx-action@e468171a9de216ec08956ac3ada2f0791b6bd435"
        self.assertIn(buildx_setup, workflow)
        self.assertIn("driver: docker-container", workflow)
        self.assertEqual(re.findall(r"^\s+version: (.+)$", workflow, re.MULTILINE), ["v0.36.1"])
        self.assertIn('docker pull "$IMAGE@$digest"', workflow)
        self.assertIn('docker save "$IMAGE@$digest" -o image.tar', workflow)
        self.assertIn("scan image --archive /workspace/image.tar", workflow)
        self.assertIn('docker run --rm -v "$PWD:/workspace:ro"', workflow)
        self.assertNotIn("/usr/bin/docker", workflow)
        source_step = workflow.split("name: Scan exact source", 1)[1].split("name: Prepare fail-closed GHCR tag guard", 1)[0]
        self.assertIn("scan --recursive .", source_step)
        self.assertNotIn("osv_status", source_step)
        self.assertNotIn("continuing by release policy", source_step)
        build_step = workflow.split("name: Build one immutable serverless wrapper", 1)[1].split("name: Attest final OCI index", 1)[0]
        self.assertLess(workflow.index(buildx_setup), workflow.index("name: Build one immutable serverless wrapper"))
        self.assertIn("set -euo pipefail", build_step)
        self.assertNotIn("continue-on-error", build_step)
        self.assertNotIn("osv_status", build_step)
        self.assertNotIn("continuing by release policy", build_step)
        self.assertIn('STAGING_TAG: ' + staging_tag, build_step)
        self.assertIn('-t "$IMAGE:$STAGING_TAG"', build_step)
        self.assertNotIn('-t "$IMAGE:$TAG"', build_step)
        self.assertNotIn('$TAG', build_step)
        self.assertLess(build_step.index('docker pull "$IMAGE@$digest"'), build_step.index('docker save "$IMAGE@$digest"'))
        self.assertLess(build_step.index('docker save "$IMAGE@$digest" -o image.tar'), build_step.index('scan image --archive /workspace/image.tar'))
        self.assertLess(build_step.index('scan image --archive /workspace/image.tar'), build_step.index('docker buildx imagetools inspect "$IMAGE@$digest"'))
        attest_step = workflow.split("name: Attest final OCI index", 1)[1].split("name: Promote", 1)[0]
        self.assertIn("subject-digest: ${{ steps.build.outputs.digest }}", attest_step)
        promote_step = workflow.split("name: Promote", 1)[1]
        expected_promotion = 'docker buildx imagetools create --prefer-index=false --tag "$IMAGE:$TAG" "$IMAGE@$DIGEST"'
        dry_run = 'promotion_manifest="$(docker buildx imagetools create --dry-run --prefer-index=false "$IMAGE@$DIGEST")"'
        planned_digest = 'planned_digest="sha256:$(printf \'%s\' "$promotion_manifest" | sha256sum | awk \'{print $1}\')"'
        recovery_inspect = 'resolved="$(docker buildx imagetools inspect "$IMAGE:$TAG" --format \'{{json .Manifest}}\' 2>/dev/null | jq -er \'.digest\')" || exit 1'
        self.assertIn(dry_run, promote_step)
        self.assertIn(planned_digest, promote_step)
        self.assertIn('test "$planned_digest" = "$DIGEST"', promote_step)
        self.assertIn(expected_promotion, promote_step)
        self.assertNotRegex(promote_step, r"imagetools create[^\n]*(?:--annotation|--filter|--platform)")
        self.assertLess(promote_step.index(dry_run), promote_step.index('bash "$RUNNER_TEMP/refuse-occupied-ghcr-tag"'))
        self.assertLess(promote_step.index('test "$planned_digest" = "$DIGEST"'), promote_step.index('bash "$RUNNER_TEMP/refuse-occupied-ghcr-tag"'))
        self.assertRegex(promote_step, re.escape(expected_promotion) + r"; then\n\s*exit 0\n\s*fi")
        self.assertIn(recovery_inspect, promote_step)
        self.assertIn('test "$resolved" = "$DIGEST"', promote_step)
        self.assertNotIn("{{.Digest}}", promote_step)
        self.assertEqual(promote_step.count("imagetools inspect"), 1)
        self.assertLess(promote_step.index("fi"), promote_step.index(recovery_inspect))
        self.assertLess(workflow.index("name: Attest final OCI index"), workflow.index("name: Promote"))
        self.assertEqual(workflow.rstrip().splitlines()[-1], '          test "$resolved" = "$DIGEST"')
        self.assertNotIn("vast-bench-diar", workflow)
        manifest = json.loads((ROOT / "locks" / "parakeet-v3.artifacts.json").read_text())
        self.assertFalse(manifest["resolution_required"])
        self.assertEqual(manifest["models"][0]["sha256"], offline.MODEL_SHA256)
        self.assertEqual(manifest["models"][0]["revision"], offline.MODEL_REVISION)
        self.assertIn(manifest["models"][0]["revision"], manifest["models"][0]["source"])


if __name__ == "__main__":
    unittest.main()
