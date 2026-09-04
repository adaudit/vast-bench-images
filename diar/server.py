#!/usr/bin/env python3
"""Offline Sortformer HTTP boundary for Vast Serverless."""
import base64
import binascii
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import wave
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
MODEL_REVISION = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d"
MODEL_PATH = Path("/workspace/models/diar_streaming_sortformer_4spk-v2.1.nemo")
MODEL_SIZE_BYTES = 471367680
MODEL_SHA256 = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"
MAX_BODY_BYTES = 700 * 1024 * 1024
LOGGER = logging.getLogger("sortformer.server")


class ContractError(ValueError):
    pass


class NotReadyError(ContractError):
    pass


def verify_model(path=MODEL_PATH):
    path = Path(path)
    if path.stat().st_size != MODEL_SIZE_BYTES:
        raise ValueError("baked model size does not match")
    digest = hashlib.sha256()
    with path.open("rb") as model:
        for chunk in iter(lambda: model.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        raise ValueError("baked model digest does not match")


def audio_seconds(path):
    try:
        with wave.open(str(path)) as audio:
            if audio.getframerate() != 16000 or audio.getnchannels() != 1:
                raise ContractError("audio must be 16 kHz mono WAV")
            frames = audio.getnframes()
    except wave.Error as error:
        raise ContractError("audio must be a readable WAV") from error
    if frames <= 0:
        raise ContractError("audio must not be empty")
    return round(frames / 16000, 3)


def turns(raw_turns):
    parsed = []
    for start, end, label in raw_turns:
        start, end = round(float(start), 3), round(float(end), 3)
        if end > start:
            parsed.append((start, end, str(label)))
    parsed.sort(key=lambda turn: (turn[0], turn[1], turn[2]))
    labels = {}
    return [
        {"start_s": start, "end_s": end, "speaker_idx": labels.setdefault(label, len(labels))}
        for start, end, label in parsed
    ]


def parse_segments(segments):
    raw_turns = []
    for segment in segments:
        fields = str(segment).split()
        if len(fields) < 3:
            continue
        try:
            raw_turns.append((fields[0], fields[1], fields[-1]))
        except (TypeError, ValueError):
            continue
    return turns(raw_turns)


def peak_vram_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 / 1024)
    except Exception:
        pass
    return None


def reset_peak_vram():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


class Runtime:
    def __init__(self, *, model_verifier=verify_model, model_loader=None, clock=time.monotonic, peak_memory=peak_vram_mb, reset_peak=reset_peak_vram):
        if os.environ.get("DIAR_CONCURRENCY", "1") != "1":
            raise ValueError("DIAR_CONCURRENCY must be exactly 1")
        self.model_verifier = model_verifier
        self.model_loader = model_loader or self._load_model
        self.clock = clock
        self.peak_memory = peak_memory
        self.reset_peak = reset_peak
        self.model = None
        self.state = "not_started"
        self.error = None
        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def ready(self):
        return self.state == "ready"

    def initialize_once(self):
        with self._state_lock:
            if self.state != "not_started":
                return
            self.state = "loading"
        try:
            self.model_verifier(MODEL_PATH)
            model = self.model_loader()
            if model is None or not callable(getattr(model, "diarize", None)):
                raise ValueError("Sortformer model is unusable")
        except Exception as error:
            LOGGER.exception("sortformer initialization failed")
            with self._state_lock:
                self.error = str(error)
                self.state = "failed"
            return
        with self._state_lock:
            self.model = model
            self.state = "ready"

    def start_initialization(self):
        threading.Thread(target=self.initialize_once, daemon=True).start()

    def health(self):
        if self.state == "ready":
            return 200, {"status": "ready"}
        if self.state == "failed":
            return 503, {"status": "failed", "cause": "initialization failed", "error": self.error}
        return 503, {"status": "loading" if self.state == "loading" else "not started"}

    def _load_model(self):
        from nemo.collections.asr.models import SortformerEncLabelModel
        model = SortformerEncLabelModel.restore_from(str(MODEL_PATH))
        model.eval()
        model.sortformer_modules.chunk_len = 340
        model.sortformer_modules.chunk_right_context = 40
        model.sortformer_modules.fifo_len = 40
        model.sortformer_modules.spkcache_update_period = 300
        return model

    def diarize(self, path):
        if not self.ready:
            raise NotReadyError("runtime is not ready")
        duration = audio_seconds(path)
        with self._inference_lock:
            self.reset_peak()
            started = self.clock()
            segments = self.model.diarize(audio=[str(path)], batch_size=1)[0]
            elapsed_ms = round((self.clock() - started) * 1000)
            memory = self.peak_memory()
        speaker_turns = parse_segments(segments)
        return {
            "engine": "sortformer",
            "model_revision": MODEL_REVISION,
            "speaker_turns": speaker_turns,
            "metrics": {
                "elapsed_ms": elapsed_ms,
                "audio_seconds": duration,
                "num_speakers": len({turn["speaker_idx"] for turn in speaker_turns}),
                "peak_vram_mb": memory,
            },
        }


def _json_audio(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("audio_base64"), str):
        raise ContractError("JSON request requires audio_base64")
    try:
        return base64.b64decode(payload["audio_base64"], validate=True)
    except (binascii.Error, ValueError) as error:
        raise ContractError("audio_base64 is invalid") from error


def _multipart_audio(content_type, body):
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("ascii") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    for part in message.iter_parts():
        if part.get_content_disposition() == "form-data" and part.get_param("name", header="content-disposition") == "audio":
            audio = part.get_payload(decode=True)
            if audio:
                return audio
    raise ContractError("multipart request requires audio")


def make_server(address=("0.0.0.0", 8080), runtime=None):
    runtime = runtime or Runtime()
    runtime.start_initialization()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/healthz":
                self._send(404, {"error": "not found"})
                return
            status, payload = runtime.health()
            self._send(status, payload)

        def do_POST(self):
            if self.path != "/diarize":
                self._send(404, {"error": "not found"})
                return
            try:
                if not runtime.ready:
                    raise NotReadyError("runtime is not ready")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ContractError("request body is outside the permitted limit")
                body = self.rfile.read(length)
                content_type = self.headers.get("Content-Type", "")
                if content_type.startswith("application/json"):
                    audio = _json_audio(json.loads(body))
                elif content_type.startswith("multipart/form-data"):
                    audio = _multipart_audio(content_type, body)
                else:
                    raise ContractError("unsupported content type")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output:
                    output.write(audio)
                    path = Path(output.name)
                try:
                    self._send(200, runtime.diarize(path))
                finally:
                    path.unlink(missing_ok=True)
            except NotReadyError:
                self._send(503, {"error": "not ready"})
            except (ContractError, json.JSONDecodeError, UnicodeEncodeError, ValueError):
                self._send(400, {"error": "invalid request"})
            except Exception:
                LOGGER.exception("sortformer request failed")
                self._send(500, {"error": "internal error"})

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(address, Handler)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    make_server().serve_forever()
