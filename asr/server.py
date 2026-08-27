#!/usr/bin/env python3
"""Minimal HTTP boundary for the immutable Parakeet-v3 candidate."""
import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from offline_entrypoint import MODEL_PATH, build_candidate, extract_aligned_words, verify_model
    from vast_adapter import ContractError, parse_request
except ModuleNotFoundError:
    from asr.offline_entrypoint import MODEL_PATH, build_candidate, extract_aligned_words, verify_model
    from asr.vast_adapter import ContractError, parse_request


class Runtime:
    def __init__(self, *, model_verifier=verify_model, model_loader=None, transcriber=None, batch_transcriber=None):
        self.model_verifier = model_verifier
        self.model_loader = model_loader or self._load_model
        self.model = None
        self.transcriber = transcriber
        self.batch_transcriber = batch_transcriber
        self.ready = False

    def check_ready(self):
        if not self.ready:
            self.model_verifier(MODEL_PATH)
            self.model = self.model_loader()
            self.ready = True

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
            return [extract_aligned_words(result) for result in self.model.transcribe(files, batch_size=len(files), timestamps=True)]
        finally:
            for path in files:
                try: __import__("os").unlink(path)
                except FileNotFoundError: pass

    def transcribe(self, payload):
        return self.transcribe_batch([payload])[0]

    def transcribe_batch(self, payloads):
        self.check_ready()
        requests = [parse_request(payload) for payload in payloads]
        if not 0 < len(requests) <= 32: raise ContractError("batch is outside the permitted limit")
        if self.batch_transcriber: segments = self.batch_transcriber(requests)
        elif self.transcriber: segments = [self.transcriber(request) for request in requests]
        else: segments = self._transcribe_many(requests)
        if len(segments) != len(requests): raise ContractError("batch result count must equal request count")
        return [build_candidate(request.audio_duration_seconds, item) for request, item in zip(requests, segments)]


def make_server(address=("0.0.0.0", 8080), runtime=None):
    runtime = runtime or Runtime()

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
            try:
                runtime.check_ready()
            except ContractError:
                self._send(503, {"status": "not ready"})
                return
            self._send(200, {"status": "ready"})

        def do_POST(self):
            if self.path not in ("/transcribe", "/transcribe-batch") or self.headers.get("Content-Type") != "application/json":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 700 * 1024 * 1024:
                    raise ContractError("request body is outside the permitted limit")
                payload = json.loads(self.rfile.read(length))
                self._send(200, runtime.transcribe(payload) if self.path == "/transcribe" else runtime.transcribe_batch(payload["requests"]))
            except (ContractError, ValueError, json.JSONDecodeError):
                self._send(400, {"error": "invalid request"})

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(address, Handler)


if __name__ == "__main__":
    make_server().serve_forever()
