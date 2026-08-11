"""Test A model server for Vast serverless.

A thin HTTP server the Vast PyWorker proxies to. It reuses the exact ASR core
from the RunPod worker (fetch -> normalize -> transcribe_with_fallback) but,
unlike rp_handler._handle, it returns the transcript in the HTTP response
instead of publishing it to the Go control plane. That makes Test A entirely
self-contained: no control-plane dependency, so it isolates whether the Vast
serverless harness + our fixed Parakeet worker actually run and return results.

Readiness contract: after the Parakeet pool is warm this prints the exact line
MODEL_SERVER_LOADED_MARKER, which the PyWorker watches for in the log file to
tell Vast's autoscaler the worker is ready to receive traffic.
"""
import json
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import worker

MODEL_SERVER_LOADED_MARKER = "MODEL_SERVER_LOADED: ready for traffic"

ROOT = Path(os.environ.get("WORKSPACE_ROOT", "/workspace"))
MODELS = ROOT / "models"

app = FastAPI()
_state = {"ready": False}


class TranscribeRequest(BaseModel):
    audio_url: str
    lane: str = "parakeet_dual"
    params: dict = {}


@app.on_event("startup")
def _startup():
    manifest = json.loads((ROOT / "model-pins.json").read_text())
    worker.validate_baked_models(MODELS, manifest)
    worker.warm_parakeet_pool(MODELS)
    _state["blocked"] = set(manifest.get("blocked_lanes", []))
    _state["ready"] = True
    # This exact line is the PyWorker readiness signal (see pyworker.py on_load).
    print(MODEL_SERVER_LOADED_MARKER, flush=True)


@app.get("/ping")
def ping():
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    """Fetch signed audio, transcribe, and return the transcript directly."""
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="model not loaded")
    started = time.monotonic()
    audio = worker.fetch(req.audio_url)
    normalized = None
    try:
        normalized = worker.normalize_audio(audio)
        candidate, lane_used, attempted = worker.transcribe_with_fallback(
            normalized, req.lane, MODELS, _state["blocked"], req.params,
        )
        return {
            "status": "succeeded",
            "lane_used": lane_used,
            "attempted_lanes": attempted,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "lane": candidate.get("lane"),
            "confidence": candidate.get("confidence"),
            "segments": candidate.get("segments", []),
        }
    except Exception as error:  # surface the failure to the caller for Test A
        print(f"TRANSCRIBE_FAILED: {error}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        if normalized:
            Path(normalized).unlink(missing_ok=True)
        Path(audio).unlink(missing_ok=True)
