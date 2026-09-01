#!/usr/bin/env python3
"""One-shot bridge from the four pinned Zoom sources to the v3 batch schema."""
import asyncio
import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import drain_worker
import vast_failure_guard as failure_guard

ENDPOINT_ID = 35304
MANIFEST = Path(__file__).with_name("approved-gate-b-benchmark.json")
CALIBRATION_METRIC = "segment_brier_score"
CALIBRATION_THRESHOLD = .7
CALIBRATION_RULE = "calibrated_confidence < threshold"


@dataclass(frozen=True)
class Source:
    path: Path
    duration_seconds: float
    sha256: str


def verify_source(source):
    digest = hashlib.sha256()
    with source.path.open("rb") as audio:
        for block in iter(lambda: audio.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != source.sha256:
        raise ValueError("source hash mismatch: " + source.path.name)
    return source


def verified_claim(source):
    verify_source(source)
    # candidate_payload is the production serializer for canonical WAV claims.
    if source.path.read_bytes()[:4] != b"RIFF":
        raise ValueError("source is not canonical WAV: " + source.path.name)
    return {"audio_url": source.path.as_uri(), "safe_parameters": {"audio_duration_seconds": source.duration_seconds}}


def payload(sources):
    return {"requests": [drain_worker.candidate_payload(verified_claim(source)) for source in sources]}


def load_sources(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    rows = manifest.get("sources") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("exactly four approved post-Gate-B sources are required")
    sources = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "duration_seconds", "sha256"}:
            raise ValueError("invalid approved source manifest")
        path, duration, digest = Path(row["path"]), row["duration_seconds"], row["sha256"]
        if not path.is_absolute() or not isinstance(duration, (int, float)) or isinstance(duration, bool) or not 0 < duration <= 86400 or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid approved source manifest")
        sources.append(Source(path, float(duration), digest))
    if len({source.path.resolve(strict=False) for source in sources}) != len(sources) or len({source.sha256 for source in sources}) != len(sources):
        raise ValueError("approved sources must have distinct recording identities")
    return tuple(sources)


def validate_batch(result, requests):
    if not requests or not isinstance(result, list) or len(result) != len(requests):
        raise ValueError("batch response is not a complete ordered list")
    for candidate, request in zip(result, requests):
        if not isinstance(candidate, dict) or set(candidate) != {"schema_version", "disposition", "lane", "model_id", "model_revision", "audio_duration_seconds", "segments", "selected_segment_indexes", "calibration"} or candidate.get("schema_version") != "asr-candidate-v3" or candidate.get("lane") != drain_worker.LANE or candidate.get("model_id") != drain_worker.MODEL_ID or candidate.get("model_revision") != drain_worker.MODEL_REVISION or not isinstance(candidate.get("audio_duration_seconds"), (int, float)) or isinstance(candidate["audio_duration_seconds"], bool) or not math.isfinite(candidate["audio_duration_seconds"]) or not 0 < candidate["audio_duration_seconds"] <= 86400 or candidate["audio_duration_seconds"] != request["audio_duration_seconds"]:
            raise ValueError("batch response identity/order mismatch")
        disposition, segments, selected = candidate.get("disposition"), candidate.get("segments"), candidate.get("selected_segment_indexes")
        calibration = candidate.get("calibration")
        evidence = calibration.get("segment_evidence") if isinstance(calibration, dict) else None
        if disposition not in ("speech", "no_speech") or not isinstance(segments, list) or not isinstance(selected, list) or not isinstance(evidence, list) or set(calibration) != {"corpus_sha256", "metric", "threshold", "decision_rule", "segment_evidence"} or not isinstance(calibration.get("corpus_sha256"), str) or len(calibration["corpus_sha256"]) != 64 or any(digit not in "0123456789abcdef" for digit in calibration["corpus_sha256"]) or calibration.get("metric") != CALIBRATION_METRIC or calibration.get("threshold") != CALIBRATION_THRESHOLD or isinstance(calibration.get("threshold"), bool) or calibration.get("decision_rule") != CALIBRATION_RULE or any(not isinstance(index, int) or isinstance(index, bool) for index in selected) or len(set(selected)) != len(selected):
            raise ValueError("batch response evidence is malformed")
        if disposition == "no_speech":
            if segments or selected or evidence:
                raise ValueError("no-speech response contains evidence")
            continue
        if not segments or len(evidence) != len(segments):
            raise ValueError("speech response lacks aligned evidence")
        wanted, previous = [], (-1.0, -1.0)
        for index, (segment, item) in enumerate(zip(segments, evidence)):
            if not isinstance(segment, dict) or set(segment) != {"start_seconds", "end_seconds", "text", "confidence"} or not isinstance(item, dict) or set(item) != {"segment_index", "raw_confidence", "calibrated_confidence", "timestamp_start_seconds", "timestamp_end_seconds"}:
                raise ValueError("batch segment is malformed")
            start, end, text, confidence = segment.get("start_seconds"), segment.get("end_seconds"), segment.get("text"), segment.get("confidence")
            if not isinstance(text, str) or not text.strip() or not isinstance(item["segment_index"], int) or isinstance(item["segment_index"], bool) or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in (start, end, confidence, item["raw_confidence"], item["calibrated_confidence"], item["timestamp_start_seconds"], item["timestamp_end_seconds"])) or not 0 <= start < end <= candidate["audio_duration_seconds"] or not 0 <= confidence <= 1 or start < previous[0] or end < previous[1] or (item["segment_index"], item["raw_confidence"], item["timestamp_start_seconds"], item["timestamp_end_seconds"]) != (index, confidence, start, end) or not 0 <= item["calibrated_confidence"] <= 1:
                raise ValueError("batch segment/evidence mismatch")
            previous = start, end
            if item["calibrated_confidence"] < calibration["threshold"]:
                wanted.append(index)
        if selected != wanted:
            raise ValueError("batch selected indexes mismatch")
    return result


async def submit_once(sources, control_client=None, state_path=None, endpoint_factory=None, attempt_id=None):
    if control_client is None or state_path is None:
        raise ValueError("control client and durable state path are required")
    requests = payload(sources)["requests"]
    guard = failure_guard.VastFailureGuard(control_client, state_path=state_path)
    if guard.suspended:
        raise RuntimeError("endpoint 35304 is suspended; explicit recreation required")
    attempt_id = attempt_id or uuid.uuid4().hex
    target = control_client.resolve_endpoint_target(ENDPOINT_ID)
    if not isinstance(target, failure_guard.Target) or target.endpoint_id != ENDPOINT_ID or target.workergroup_id <= 0 or not target.instance_ids or len(set(target.instance_ids)) != len(target.instance_ids):
        raise ValueError("ambiguous Vast target")
    try:
        if endpoint_factory is None:
            from vastai import Serverless
            endpoint_factory = Serverless(api_key=os.environ.get("VAST_API_KEY"))
        endpoint = await endpoint_factory.get_endpoint(name=str(ENDPOINT_ID))
        result = endpoint.request("/transcribe-batch", {"requests": requests}, cost=len(requests), timeout=1800)
        if hasattr(result, "__await__"):
            result = await result
        validate_batch(result, requests)
    except Exception:
        guard.failure(attempt_id, target)
        raise
    guard.success(True)
    return result


def main(argv=None, *, control_client=None, endpoint_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--state-path", type=Path, required=True)
    args = parser.parse_args(argv)
    if control_client is None:
        raise SystemExit("an installed exact-target Vast control adapter is required; do not submit")
    if not args.state_path.parent.is_dir() or not os.access(args.state_path.parent, os.W_OK):
        raise SystemExit("a writable durable state path is required")
    try:
        return asyncio.run(submit_once(load_sources(args.manifest), control_client, args.state_path, endpoint_factory))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
