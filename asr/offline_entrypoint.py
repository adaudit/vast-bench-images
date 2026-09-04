#!/usr/bin/env python3
"""Offline-only Parakeet-v3 candidate producer."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

SCHEMA_VERSION = "asr-candidate-v3"
REQUEST_VERSION = "parakeet-v3-offline-request-v1"
LANE = "parakeet_v3"
MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
MODEL_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
MODEL_SHA256 = "3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d"
MODEL_SIZE_BYTES = 2509332480
CALIBRATION_SHA256 = "08575f17a02a229d805003df4cd7f518d4134371d6ac4528ebfb56fa75b16af4"
THRESHOLD = 0.70
DECODER_FRAME_SECONDS = 0.08
MODEL_PATH = Path("/workspace/models/parakeet-tdt-0.6b-v3.nemo")
INPUT_ROOT = Path("/workspace/input")
OUTPUT_ROOT = Path("/workspace/output")
MAX_AUDIO_BYTES = 512 * 1024 * 1024
AUDIO_MAGIC = {
    ".wav": (b"RIFF",),
    ".flac": (b"fLaC",),
    ".mp3": (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"),
    ".ogg": (b"OggS",),
}


class ContractError(ValueError):
    pass


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _contained_regular_file(path, root, *, limit=None, magic=None):
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ContractError("path must be an existing non-symlink regular file")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        raise ContractError("path must remain inside its dedicated root") from None
    if resolved != path or any(parent.is_symlink() for parent in path.parents if parent != root.parent):
        raise ContractError("symlink paths are forbidden")
    if limit is not None and not 0 < resolved.stat().st_size <= limit:
        raise ContractError("file size is outside the permitted limit")
    if magic is not None:
        with resolved.open("rb") as audio:
            if audio.read(4) not in magic:
                raise ContractError("audio type does not match its file signature")
    return resolved


def read_request(path):
    request_path = _contained_regular_file(Path(path), INPUT_ROOT, limit=64 * 1024)
    request = json.loads(request_path.read_text())
    required = {"request_version", "lane", "model_id", "model_revision", "audio_path", "audio_duration_seconds"}
    if not isinstance(request, dict) or set(request) != required:
        raise ContractError("request must contain only the offline candidate fields")
    if (request["request_version"], request["lane"], request["model_id"], request["model_revision"]) != (REQUEST_VERSION, LANE, MODEL_ID, MODEL_REVISION):
        raise ContractError("unexpected offline candidate identity")
    audio_path = Path(request["audio_path"])
    if "://" in request["audio_path"] or audio_path.suffix.lower() not in AUDIO_MAGIC:
        raise ContractError("audio_path must be a permitted local audio file")
    audio_path = _contained_regular_file(audio_path, INPUT_ROOT, limit=MAX_AUDIO_BYTES, magic=AUDIO_MAGIC[audio_path.suffix.lower()])
    duration = request["audio_duration_seconds"]
    if not _finite(duration) or duration <= 0 or duration > 86400:
        raise ContractError("audio_duration_seconds must be finite and bounded")
    return audio_path, float(duration)


def verify_model(path):
    if path != MODEL_PATH or path.is_symlink() or not path.is_file() or path.stat().st_size != MODEL_SIZE_BYTES:
        raise ContractError("exact baked model artifact is required")
    digest = hashlib.sha256()
    with path.open("rb") as model:
        for block in iter(lambda: model.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != MODEL_SHA256:
        raise ContractError("baked model checksum mismatch")


def extract_aligned_words(result):
    timestamp = getattr(result, "timestamp", None)
    words = timestamp.get("word") if isinstance(timestamp, dict) else None
    confidences = getattr(result, "word_confidence", None)
    if not isinstance(words, list) or not isinstance(confidences, list) or len(confidences) != len(words):
        raise ContractError("model produced no aligned word evidence")
    if not words:
        return []
    segments = []
    for word, confidence in zip(words, confidences):
        if not isinstance(word, dict):
            raise ContractError("model produced invalid aligned word evidence")
        text, start, end = word.get("word"), word.get("start"), word.get("end")
        if not isinstance(text, str) or not text.strip() or not all(_finite(value) for value in (start, end, confidence)) or start < 0 or end <= start or not 0 <= confidence <= 1:
            raise ContractError("model produced invalid aligned word evidence")
        segments.append({"start_seconds": start, "end_seconds": end, "text": text.strip(), "confidence": confidence})
    return segments


def decode_with_nemo(model_path, audio_path):
    from nemo.collections.asr.models import ASRModel
    from omegaconf import open_dict

    model = ASRModel.restore_from(str(model_path))
    with open_dict(model.cfg.decoding):
        model.cfg.decoding.compute_timestamps = True
        model.cfg.decoding.preserve_alignments = True
        model.cfg.decoding.confidence_cfg = {"preserve_word_confidence": True}
    model.change_decoding_strategy(model.cfg.decoding, verbose=False)
    return extract_aligned_words(model.transcribe([str(audio_path)], timestamps=True)[0])


def build_candidate(duration, segments):
    # A decoder may legitimately find no words in a tiny retained VAD fragment.
    # Missing or malformed evidence remains a ContractError above.
    if segments == []:
        return {"schema_version": SCHEMA_VERSION, "disposition": "no_speech", "lane": LANE, "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "audio_duration_seconds": duration, "segments": [], "selected_segment_indexes": [], "calibration": {"corpus_sha256": CALIBRATION_SHA256, "metric": "segment_brier_score", "threshold": THRESHOLD, "decision_rule": "calibrated_confidence < threshold", "segment_evidence": []}}
    evidence, selected = [], []
    previous_start = previous_end = -1.0
    for index, segment in enumerate(segments):
        start, end, confidence = segment["start_seconds"], segment["end_seconds"], segment["confidence"]
        if index == len(segments) - 1 and _finite(end) and duration < end <= duration + DECODER_FRAME_SECONDS:
            end = segment["end_seconds"] = duration
        if not isinstance(segment.get("text"), str) or not segment["text"].strip() or not all(_finite(value) for value in (start, end, confidence)) or start < 0 or end <= start or end > duration or not 0 <= confidence <= 1 or start < previous_start or end < previous_end:
            raise ContractError("unaligned segment evidence")
        previous_start, previous_end = start, end
        evidence.append({"segment_index": index, "raw_confidence": confidence, "calibrated_confidence": confidence, "timestamp_start_seconds": start, "timestamp_end_seconds": end})
        if confidence < THRESHOLD:
            selected.append(index)
    return {"schema_version": SCHEMA_VERSION, "disposition": "speech", "lane": LANE, "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "audio_duration_seconds": duration, "segments": segments, "selected_segment_indexes": selected, "calibration": {"corpus_sha256": CALIBRATION_SHA256, "metric": "segment_brier_score", "threshold": THRESHOLD, "decision_rule": "calibrated_confidence < threshold", "segment_evidence": evidence}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ContractError("offline environment variables must be set")
    audio_path, duration = read_request(args.input_json)
    output_path = Path(args.output_json)
    try:
        output_path.resolve(strict=False).relative_to(OUTPUT_ROOT)
    except ValueError:
        raise ContractError("output_json must remain inside the dedicated output root") from None
    verify_model(MODEL_PATH)
    output_path.write_text(json.dumps(build_candidate(duration, decode_with_nemo(MODEL_PATH, audio_path)), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
