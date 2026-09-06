#!/usr/bin/env python3
"""Offline-only Parakeet-v3 candidate producer."""
import argparse
import functools
import hashlib
import json
import logging
import math
import os
from pathlib import Path

LOGGER = logging.getLogger("parakeet.offline")

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


def _number(value):
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        try:
            value = value.item()
        except (AttributeError, TypeError, ValueError, OverflowError, RuntimeError):
            return None
        if isinstance(value, bool):
            return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _finite(value):
    return _number(value) is not None


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


def _aligned_word_error(index, word, start, end, confidence, check):
    word_repr = repr(word)
    if len(word_repr) > 200:
        word_repr = word_repr[:197] + "..."
    return ContractError(
        f"model produced invalid aligned word evidence at index {index}: {check}; "
        f"word={word_repr}; types(start={type(start).__name__}, "
        f"end={type(end).__name__}, confidence={type(confidence).__name__})"
    )


def _token_records(timestamp, confidences):
    chars = timestamp.get("char") if isinstance(timestamp, dict) else None
    if not isinstance(chars, list) or not isinstance(confidences, (list, tuple)):
        return []
    records = []
    token_index = 0
    for char in chars:
        if not isinstance(char, dict):
            continue
        values = char.get("char")
        if not isinstance(values, (list, tuple)):
            values = [values]
        start, end = _number(char.get("start_offset")), _number(char.get("end_offset"))
        for _ in values:
            confidence = _number(confidences[token_index]) if token_index < len(confidences) else None
            token_index += 1
            if start is not None and end is not None and confidence is not None and 0 <= confidence <= 1:
                records.append((start, end, confidence))
    return records


def _word_confidence(word, records):
    start, end = _number(word.get("start_offset")), _number(word.get("end_offset"))
    if start is None or end is None or end < start:
        return 0.0
    confidence = None
    for char_start, char_end, token_confidence in records:
        if start <= char_start and char_end <= end:
            confidence = token_confidence if confidence is None else min(confidence, token_confidence)
    return confidence if confidence is not None else 0.0


def extract_aligned_words(result):
    timestamp = getattr(result, "timestamp", None)
    words = timestamp.get("word") if isinstance(timestamp, dict) else None
    confidences = getattr(result, "token_confidence", None)
    if not isinstance(words, list):
        raise ContractError(
            "model produced no aligned word evidence: "
            f"words={type(words).__name__}(len={len(words) if isinstance(words, list) else 'n/a'}), "
            f"confidences={type(confidences).__name__}(len={len(confidences) if isinstance(confidences, (list, tuple)) else 'n/a'})"
        )
    if not words:
        return []
    records = _token_records(timestamp, confidences)
    segments = []
    for index, word in enumerate(words):
        confidence = _word_confidence(word, records) if isinstance(word, dict) else 0.0
        if not isinstance(word, dict):
            raise _aligned_word_error(index, word, None, None, confidence, "word must be a dict")
        text, start, end = word.get("word"), word.get("start"), word.get("end")
        start_number, end_number, confidence_number = (_number(value) for value in (start, end, confidence))
        if not isinstance(text, str) or not text.strip():
            raise _aligned_word_error(index, word, start, end, confidence, "text must be a non-empty string")
        if start_number is None:
            raise _aligned_word_error(index, word, start, end, confidence, "start must be a finite numeric scalar")
        if end_number is None:
            raise _aligned_word_error(index, word, start, end, confidence, "end must be a finite numeric scalar")
        if confidence_number is None:
            raise _aligned_word_error(index, word, start, end, confidence, "confidence must be a finite numeric scalar")
        if start_number < 0:
            raise _aligned_word_error(index, word, start, end, confidence, "start must be non-negative")
        if end_number < start_number:
            raise _aligned_word_error(index, word, start, end, confidence, "end must not precede start")
        if not 0 <= confidence_number <= 1:
            raise _aligned_word_error(index, word, start, end, confidence, "confidence must be within [0, 1]")
        if end_number == start_number:
            end_number += DECODER_FRAME_SECONDS
        segments.append({"start_seconds": start_number, "end_seconds": end_number, "text": text.strip(), "confidence": confidence_number})
    return segments


_GUARD_ATTRIBUTE = "_leading_punctuation_guard"


def _droppable_leading_entries(char_offsets, supported_punctuation):
    """Count leading offset entries whose decoded tokens are all whitespace or supported punctuation.

    ``char_offsets`` and ``encoded_char_offsets`` are index-aligned: NeMo deepcopies the
    one from the other (rnnt_decoding.py:927) after building both from the same token
    zip (rnnt_decoding.py:1037-1044), so a single count trims both lists.
    """
    dropped = 0
    for entry in char_offsets:
        tokens = entry.get("char") if isinstance(entry, dict) else None
        if not isinstance(tokens, (list, tuple)):
            break
        if not all(
            token.strip() == "" or (supported_punctuation and token.strip() in supported_punctuation)
            for token in tokens
        ):
            break
        dropped += 1
    return dropped


def guard_leading_punctuation(decoding):
    """Wrap ``decoding.get_words_offsets`` against the NeMo 2.4.1 leading-punctuation bug.

    ``RNNTBPEDecoding.get_words_offsets`` reads ``word_offsets[-1]`` before any word
    exists when a chunk's first decoded token is supported punctuation
    (rnnt_decoding.py:1966-1971), turning real 60 s chunks into IndexErrors. The wrapper
    retries once without the leading whitespace/punctuation entries; offsets are
    absolute decoder timesteps, so the trimmed retry keeps correct timings. It must be
    applied after ``change_decoding_strategy`` because that call rebuilds
    ``model.decoding``.
    """
    original = decoding.get_words_offsets
    if getattr(original, _GUARD_ATTRIBUTE, False):
        return

    @functools.wraps(original)
    def guarded(char_offsets, encoded_char_offsets, word_delimiter_char=" ", supported_punctuation=None):
        try:
            return original(
                char_offsets=char_offsets,
                encoded_char_offsets=encoded_char_offsets,
                word_delimiter_char=word_delimiter_char,
                supported_punctuation=supported_punctuation,
            )
        except IndexError:
            dropped = _droppable_leading_entries(char_offsets, supported_punctuation)
            if not dropped:
                raise
            LOGGER.warning("get_words_offsets: dropping %d leading punctuation-only offset entries before retry", dropped)
            trimmed_char_offsets = char_offsets[dropped:]
            trimmed_encoded_char_offsets = encoded_char_offsets[dropped:]
            if not trimmed_char_offsets or not trimmed_encoded_char_offsets:
                return []
            return original(
                char_offsets=trimmed_char_offsets,
                encoded_char_offsets=trimmed_encoded_char_offsets,
                word_delimiter_char=word_delimiter_char,
                supported_punctuation=supported_punctuation,
            )

    setattr(guarded, _GUARD_ATTRIBUTE, True)
    decoding.get_words_offsets = guarded


def decode_with_nemo(model_path, audio_path):
    from nemo.collections.asr.models import ASRModel
    from omegaconf import open_dict

    model = ASRModel.restore_from(str(model_path))
    with open_dict(model.cfg.decoding):
        model.cfg.decoding.compute_timestamps = True
        model.cfg.decoding.preserve_alignments = True
        model.cfg.decoding.confidence_cfg = {"preserve_token_confidence": True, "preserve_word_confidence": False}
        if "greedy" not in model.cfg.decoding:
            model.cfg.decoding.greedy = {}
        # NeMo's CUDA-graph TDT decoder captures a stream per lane and crashes when lanes decode concurrently.
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(model.cfg.decoding, verbose=False)
    guard_leading_punctuation(model.decoding)
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
