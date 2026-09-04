#!/usr/bin/env python3
"""Minimal HTTP boundary for the immutable Parakeet-v3 candidate."""
import json
import logging
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from offline_entrypoint import MODEL_PATH, build_candidate, extract_aligned_words, verify_model
    from vast_adapter import ContractError, parse_request
    from parakeet_pool import ParakeetPool
except ModuleNotFoundError:
    from asr.offline_entrypoint import MODEL_PATH, build_candidate, extract_aligned_words, verify_model
    from asr.vast_adapter import ContractError, parse_request
    from asr.parakeet_pool import ParakeetPool


LOGGER = logging.getLogger("parakeet.server")


class NotReadyError(ContractError):
    pass


class Runtime:
    def __init__(self, *, model_verifier=verify_model, model_loader=None, transcriber=None, batch_transcriber=None, instance_count=3):
        if instance_count != 3 or os.environ.get("PARAKEET_INSTANCES", "3") != "3":
            raise ValueError("PARAKEET_INSTANCES must be exactly 3")
        self.model_verifier = model_verifier
        self.model_loader = model_loader or self._load_model
        self.model = None
        self.pool = None
        self.transcriber = transcriber
        self.batch_transcriber = batch_transcriber
        self.instance_count = instance_count
        self.state = "not_started"
        self._state_lock = threading.Lock()

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
            pool = ParakeetPool(self.instance_count, self.model_loader)
            if len(pool.instances) != self.instance_count or len({id(model) for model in pool.instances}) != self.instance_count or any(model is None or not callable(getattr(model, "transcribe", None)) for model in pool.instances):
                raise ValueError("model lanes are unusable")
        except Exception:
            with self._state_lock:
                self.state = "failed"
            return
        with self._state_lock:
            self.pool = pool
            self.model = pool.instances[0]
            self.state = "ready"

    def start_initialization(self):
        threading.Thread(target=self.initialize_once, daemon=True).start()

    def health(self):
        if self.state == "ready":
            return 200, {"status": "ready"}
        if self.state == "failed":
            return 503, {"status": "failed", "cause": "initialization failed"}
        return 503, {"status": "loading" if self.state == "loading" else "not started"}

    def check_ready(self):
        return self.ready

    def _load_model(self):
        from nemo.collections.asr.models import ASRModel
        from omegaconf import open_dict
        model = ASRModel.restore_from(str(MODEL_PATH))
        with open_dict(model.cfg.decoding):
            model.cfg.decoding.compute_timestamps = True
            model.cfg.decoding.preserve_alignments = True
            model.cfg.decoding.confidence_cfg = {"preserve_word_confidence": True}
        model.change_decoding_strategy(model.cfg.decoding, verbose=False)
        return model

    def _transcribe_many(self, requests):
        files = []
        try:
            for request in requests:
                audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                audio.write(request.audio); audio.close(); files.append(audio.name)
            model = self.pool.checkout()
            try:
                LOGGER.info("parakeet_inference stage=start")
                segments = [extract_aligned_words(result) for result in model.transcribe(files, batch_size=len(files), timestamps=True)]
                LOGGER.info("parakeet_inference stage=complete")
                return segments
            finally:
                self.pool.checkin(model)
        finally:
            for path in files:
                try: __import__("os").unlink(path)
                except FileNotFoundError: pass

    def transcribe(self, payload):
        return self.transcribe_batch([payload])[0]

    def transcribe_batch(self, payloads):
        if not self.ready:
            raise NotReadyError("runtime is not ready")
        requests = [parse_request(payload) for payload in payloads]
        if not 0 < len(requests) <= 32: raise ContractError("batch is outside the permitted limit")
        if self.batch_transcriber: segments = self.batch_transcriber(requests)
        elif self.transcriber: segments = [self.transcriber(request) for request in requests]
        else: segments = self._transcribe_many(requests)
        if len(segments) != len(requests): raise ContractError("batch result count must equal request count")
        return [build_candidate(request.audio_duration_seconds, item) for request, item in zip(requests, segments)]


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
            if self.path not in ("/transcribe", "/transcribe-batch") or self.headers.get("Content-Type") != "application/json":
                self._send(404, {"error": "not found"})
                return
            try:
                if not runtime.ready:
                    raise NotReadyError("runtime is not ready")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 700 * 1024 * 1024:
                    raise ContractError("request body is outside the permitted limit")
                payload = json.loads(self.rfile.read(length))
                self._send(200, runtime.transcribe(payload) if self.path == "/transcribe" else runtime.transcribe_batch(payload["requests"]))
            except NotReadyError:
                LOGGER.info("parakeet_http status=503 category=not_ready")
                self._send(503, {"error": "not ready"})
            except json.JSONDecodeError:
                LOGGER.info("parakeet_http status=400 category=json")
                self._send(400, {"error": "invalid request"})
            except ContractError:
                LOGGER.info("parakeet_http status=400 category=contract")
                self._send(400, {"error": "invalid request"})
            except ValueError as error:
                LOGGER.info("parakeet_http status=400 category=%s", type(error).__name__)
                self._send(400, {"error": "invalid request"})
            except Exception as error:
                LOGGER.info("parakeet_http status=500 category=%s", type(error).__name__)
                self._send(500, {"error": "internal error"})

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(address, Handler)


if __name__ == "__main__":
    make_server().serve_forever()
