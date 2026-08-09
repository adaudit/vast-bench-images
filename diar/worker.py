"""GPU diarization engines shared by the standalone RunPod handler."""
from __future__ import annotations

import os
import threading
import wave
from pathlib import Path

SORTFORMER_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
_MODELS: dict[str, object] = {}
_MODEL_LOCK = threading.Lock()


class WorkerError(RuntimeError):
    pass


def validate_audio(path: Path) -> None:
    """Reject anything outside the endpoint's canonical 16 kHz mono WAV contract."""
    try:
        with wave.open(str(path)) as audio:
            if audio.getnframes() == 0:
                raise WorkerError("audio WAV is empty")
            if audio.getframerate() != 16000 or audio.getnchannels() != 1:
                raise WorkerError("audio must be a 16 kHz mono WAV")
    except WorkerError:
        raise
    except (wave.Error, EOFError, OSError) as error:
        raise WorkerError("audio must be a readable WAV") from error


def audio_seconds(path: Path) -> float:
    validate_audio(path)
    with wave.open(str(path)) as audio:
        return round(audio.getnframes() / audio.getframerate(), 3)


def _cache_dir() -> Path:
    path = Path(os.environ.get("HF_HOME", "/workspace/models"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model(engine: str):
    with _MODEL_LOCK:
        if engine not in _MODELS:
            token = os.environ.get("HF_TOKEN") or None
            if engine == "sortformer":
                from nemo.collections.asr.models import SortformerEncLabelModel
                model = SortformerEncLabelModel.from_pretrained(SORTFORMER_MODEL)
                model.eval()
                # NVIDIA's v2.1 streaming configuration from the model card.
                model.sortformer_modules.chunk_len = 340
                model.sortformer_modules.chunk_right_context = 40
                model.sortformer_modules.fifo_len = 40
                model.sortformer_modules.spkcache_update_period = 300
            elif engine == "pyannote":
                import torch
                from pyannote.audio import Pipeline
                torch.__version__ = torch.__version__.split("a", 1)[0].split("+", 1)[0]
                model = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=token, cache_dir=str(_cache_dir()))
                if model is None:
                    raise WorkerError("could not load pyannote; set HF_TOKEN and accept the model license")
            else:
                raise WorkerError(f"unsupported engine: {engine}")
            _MODELS[engine] = model
        return _MODELS[engine]


def _turns(raw_turns) -> list[dict]:
    labels: dict[str, int] = {}
    turns = []
    for start, end, label in raw_turns:
        start, end = round(float(start), 3), round(float(end), 3)
        if end <= start:
            continue
        speaker_idx = labels.setdefault(str(label), len(labels))
        turns.append({"start_s": start, "end_s": end, "speaker_idx": speaker_idx})
    return sorted(turns, key=lambda turn: (turn["start_s"], turn["end_s"], turn["speaker_idx"]))


def _sortformer_turns(audio: Path) -> list[dict]:
    model = _model("sortformer")
    segments = model.diarize(audio=[str(audio)], batch_size=1)[0]
    parsed = []
    for segment in segments:
        fields = str(segment).split()
        if len(fields) < 3:
            continue
        try:
            parsed.append((fields[0], fields[1], fields[-1]))
        except (TypeError, ValueError):
            continue
    return _turns(parsed)


def _pyannote_turns(audio: Path) -> list[dict]:
    diarization = _model("pyannote")(str(audio))
    return _turns((segment.start, segment.end, label) for segment, _, label in diarization.itertracks(yield_label=True))


def diarize(audio: Path, engine: str, params: dict | None = None) -> list[dict]:
    del params  # Reserved for per-engine tuning without expanding the public contract yet.
    validate_audio(audio)
    if engine == "sortformer":
        return _sortformer_turns(audio)
    if engine == "pyannote":
        return _pyannote_turns(audio)
    raise WorkerError("engine must be 'sortformer' or 'pyannote'")
