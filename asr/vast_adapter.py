"""Small, dependency-free adapter between Vast PyWorker and the v3 ASR core."""
import base64
import io
import math
import wave
from dataclasses import dataclass

try:
    from offline_entrypoint import LANE, MODEL_ID, MODEL_REVISION, REQUEST_VERSION
except ModuleNotFoundError:  # local test invocation keeps modules under asr/
    from asr.offline_entrypoint import LANE, MODEL_ID, MODEL_REVISION, REQUEST_VERSION

CHUNK_SECONDS = 300.0
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
    """Fallback planner mirroring the backend: latest silence in [start+L/2, start+L],
    else hard cut at start+L, with a 1 s overlap. Boundaries are chosen on the
    integer 16 kHz frame grid and the endpoint alone gets a nextafter nudge so
    float rounding can never push a piece past the strict cap."""
    rate = 16000
    chunk_frames, overlap_frames = round(CHUNK_SECONDS * rate), round(OVERLAP_SECONDS * rate)
    silence_frames = [round(point * rate) for point in silence_points]
    total, start_frame, chunks = round(duration * rate), 0, []
    if total <= chunk_frames + overlap_frames:
        return ((0.0, duration),)
    while start_frame < total:
        start = start_frame / rate
        if total - start_frame <= chunk_frames + overlap_frames:
            chunks.append((start, duration))
            break
        target = start_frame + chunk_frames
        window_start = start_frame + chunk_frames // 2
        nearby = [frame for frame in silence_frames if window_start <= frame <= target]
        boundary = max(nearby) if nearby else target
        end = min(duration, (boundary + overlap_frames) / rate)
        while end - start > CHUNK_SECONDS + OVERLAP_SECONDS:
            end = math.nextafter(end, start)
        chunks.append((start, end))
        start_frame = boundary
    return tuple(chunks)


def validate_chunk_plan(plan, duration):
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 86400:
        raise ContractError("invalid chunk duration")
    if not isinstance(plan, (list, tuple)) or not 0 < len(plan) <= 50000:
        raise ContractError("invalid chunk plan")
    result = []
    previous_start, previous_end = -1.0, 0.0
    for index, interval in enumerate(plan):
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ContractError("invalid chunk interval")
        start, end = interval
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (start, end)):
            raise ContractError("invalid chunk interval")
        if (start < 0 or not start < end or end > duration + 1e-6 or end - start > CHUNK_SECONDS + OVERLAP_SECONDS
                or start <= previous_start or end <= previous_end or start > previous_end + 1e-6
                or previous_end - start > OVERLAP_SECONDS + 1e-6
                or round(end * 16000) <= round(start * 16000)
                or round(end * 16000) - round(start * 16000) > (CHUNK_SECONDS + OVERLAP_SECONDS) * 16000):
            raise ContractError("invalid chunk coverage")
        if index == 0 and start != 0:
            raise ContractError("invalid chunk coverage")
        result.append((float(start), float(end)))
        previous_start, previous_end = start, end
    if abs(result[-1][1] - duration) > 1e-6:
        raise ContractError("incomplete chunk coverage")
    result[-1] = (result[-1][0], float(duration))
    return tuple(result)


def slice_wav(audio_bytes, start, end):
    """Port of the proven frame-aligned 16 kHz mono s16le CPU slicer."""
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in (start, end)) or not 0 <= start < end:
        raise ContractError("WAV slice bounds are invalid")
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ContractError("audio must be 16 kHz mono s16le WAV")
            first, last = round(start * 16000), round(end * 16000)
            if not 0 <= first < last <= source.getnframes():
                raise ContractError("WAV slice exceeds canonical audio")
            source.setpos(first)
            frames = source.readframes(last - first)
            params = source.getparams()
            if len(frames) != (last - first) * 2:
                raise ContractError("incomplete WAV slice")
    except (wave.Error, EOFError):
        raise ContractError("audio must be a valid WAV payload") from None
    output = io.BytesIO()
    with wave.open(output, "wb") as destination:
        destination.setparams(params)
        destination.writeframes(frames)
    return output.getvalue()


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
    """Port of the proven overlap-owner dedupe with monotonic absolute times."""
    if len(batch_segments) != len(chunks):
        raise ContractError("batch result count must equal chunk count")
    result, covered_end, previous_start = [], -1.0, -1.0
    for segments, (offset, _) in zip(batch_segments, chunks, strict=True):
        ownership_cutoff = covered_end
        for segment in segments:
            start, end = segment["start_seconds"] + offset, segment["end_seconds"] + offset
            if end <= ownership_cutoff or (start + end) / 2 <= ownership_cutoff:
                continue
            start = max(start, ownership_cutoff, previous_start)
            if end <= start:
                continue
            result.append({**segment, "start_seconds": start, "end_seconds": end})
            covered_end, previous_start = end, start
    return result
