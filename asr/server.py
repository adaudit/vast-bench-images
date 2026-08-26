#!/usr/bin/env python3
"""Minimal HTTP boundary for the immutable Parakeet-v3 candidate."""
import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from offline_entrypoint import MODEL_PATH, build_candidate, decode_with_nemo, verify_model
    from vast_adapter import ContractError, parse_request
except ModuleNotFoundError:
    from asr.offline_entrypoint import MODEL_PATH, build_candidate, decode_with_nemo, verify_model
    from asr.vast_adapter import ContractError, parse_request


class Runtime:
    def __init__(self, *, model_verifier=verify_model, transcriber=None):
        self.model_verifier = model_verifier
        self.transcriber = transcriber or self._transcribe
        self.ready = False

    def check_ready(self):
        if not self.ready:
            self.model_verifier(MODEL_PATH)
            self.ready = True

    def _transcribe(self, request):
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            audio.write(request.audio)
            audio.flush()
            return decode_with_nemo(MODEL_PATH, audio.name)

    def transcribe(self, payload):
        self.check_ready()
        request = parse_request(payload)
        return build_candidate(request.audio_duration_seconds, self.transcriber(request))


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
            if self.path != "/transcribe" or self.headers.get("Content-Type") != "application/json":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 700 * 1024 * 1024:
                    raise ContractError("request body is outside the permitted limit")
                self._send(200, runtime.transcribe(json.loads(self.rfile.read(length))))
            except (ContractError, ValueError, json.JSONDecodeError):
                self._send(400, {"error": "invalid request"})

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(address, Handler)


if __name__ == "__main__":
    make_server().serve_forever()
