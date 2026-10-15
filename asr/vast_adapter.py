"""Small, dependency-free adapter between Vast PyWorker and the v3 ASR core."""
import base64
from dataclasses import dataclass

try:
    from offline_entrypoint import LANE, MODEL_ID, MODEL_REVISION, REQUEST_VERSION
except ModuleNotFoundError:  # local test invocation keeps modules under asr/
    from asr.offline_entrypoint import LANE, MODEL_ID, MODEL_REVISION, REQUEST_VERSION

CHUNK_SECONDS = 60.0
OVERLAP_SECONDS = 1.0
MAX_AUDIO_BYTES = 512 * 1024 * 1024


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class Request:
    audio: bytes
    audio_filename: str
    audio_duration_seconds: float
    chunks: tuple[tuple[float, float], ...]


def chunk_ranges(duration, *, silence_points=()):
    if duration <= CHUNK_SECONDS:
        return ((0.0, duration),)
    chunks, start = [], 0.0
    while start < duration:
        if duration - start <= CHUNK_SECONDS + OVERLAP_SECONDS:
            chunks.append((start, duration))
            break
        target = min(start + CHUNK_SECONDS, duration)
        if target == duration:
            chunks.append((start, duration))
            break
        nearby = [point for point in silence_points if start < point < duration and abs(point - target) <= CHUNK_SECONDS / 2]
        boundary = min(nearby, key=lambda point: abs(point - target)) if nearby else target
        chunks.append((start, min(duration, boundary + OVERLAP_SECONDS)))
        start = max(start + 0.001, boundary - OVERLAP_SECONDS)
    return tuple(chunks)


def parse_request(payload):
    required = {"request_version", "lane", "model_id", "model_revision", "audio_filename", "audio_duration_seconds", "audio_base64"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ContractError("request must contain only the v3 adapter fields")
    if (payload["request_version"], payload["lane"], payload["model_id"], payload["model_revision"]) != (REQUEST_VERSION, LANE, MODEL_ID, MODEL_REVISION):
        raise ContractError("unexpected v3 request identity")
    filename = payload["audio_filename"]
    if not isinstance(filename, str) or filename.rsplit("/", 1)[-1] != filename or not filename.endswith(".wav"):
        raise ContractError("audio_filename must be a plain WAV filename")
    duration = payload["audio_duration_seconds"]
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not 0 < duration <= 86400:
        raise ContractError("audio duration must be finite and bounded")
    try:
        audio = base64.b64decode(payload["audio_base64"], validate=True)
    except (TypeError, ValueError) as error:
        raise ContractError("audio_base64 must be valid base64") from error
    if not 0 < len(audio) <= MAX_AUDIO_BYTES or not audio.startswith(b"RIFF"):
        raise ContractError("audio must be a bounded WAV payload")
    return Request(audio, filename, float(duration), chunk_ranges(float(duration)))


def batch_and_restitch(batch_segments, chunks):
    """Restore absolute times while retaining deterministic call/chunk order."""
    if len(batch_segments) != len(chunks):
        raise ContractError("batch result count must equal chunk count")
    result, covered_end = [], -1.0
    for segments, (offset, _) in zip(batch_segments, chunks):
        for segment in segments:
            restored = {**segment, "start_seconds": segment["start_seconds"] + offset, "end_seconds": segment["end_seconds"] + offset}
            if restored["end_seconds"] <= covered_end:
                continue
            result.append(restored)
            covered_end = restored["end_seconds"]
    return result
