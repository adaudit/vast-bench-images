"""Public RunPod Serverless contract for the standalone diarization endpoint."""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import worker

ENGINES = {"sortformer", "pyannote"}


def _input(job: dict) -> dict:
    value = job.get("input") if isinstance(job, dict) else None
    if not isinstance(value, dict) or not isinstance(value.get("audio_url"), str) or not value["audio_url"]:
        raise worker.WorkerError("job input requires a non-empty audio_url")
    if value.get("engine") not in ENGINES:
        raise worker.WorkerError("engine must be 'sortformer' or 'pyannote'")
    if "params" in value and not isinstance(value["params"], dict):
        raise worker.WorkerError("params must be an object")
    return value


def fetch(url: str) -> Path:
    with urllib.request.urlopen(url, timeout=300) as response:
        suffix = Path(urllib.parse.urlparse(url).path).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(response.read())
            return Path(handle.name)


def _peak_vram_mb() -> int | None:
    try:
        import torch
        return round(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else None
    except Exception:
        return None


def _reset_peak_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _handle(job: dict, *, fetcher=fetch, diarizer=worker.diarize) -> dict:
    data = _input(job)
    audio = fetcher(data["audio_url"])
    try:
        audio_seconds = worker.audio_seconds(audio)
        _reset_peak_vram()
        started = time.monotonic()  # Fetch time deliberately excluded from the benchmark metric.
        turns = diarizer(audio, data["engine"], data.get("params"))
        elapsed_ms = round((time.monotonic() - started) * 1000)
        return {
            "engine": data["engine"],
            "speaker_turns": turns,
            "metrics": {"elapsed_ms": elapsed_ms, "audio_seconds": audio_seconds, "num_speakers": len({turn["speaker_idx"] for turn in turns}), "peak_vram_mb": _peak_vram_mb()},
        }
    finally:
        audio.unlink(missing_ok=True)


async def handler(job: dict, *, fetcher=fetch, diarizer=worker.diarize) -> dict:
    return await asyncio.to_thread(_handle, job, fetcher=fetcher, diarizer=diarizer)


def concurrency_modifier(current_concurrency):
    try:
        return max(1, int(os.environ.get("DIAR_CONCURRENCY", "8")))
    except (TypeError, ValueError):
        return 8


def main() -> None:
    import runpod
    runpod.serverless.start({"handler": handler, "concurrency_modifier": concurrency_modifier})


if __name__ == "__main__":
    main()
