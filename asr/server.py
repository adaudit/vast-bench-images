#!/usr/bin/env python3
"""Minimal HTTP boundary for the immutable Parakeet-v3 candidate."""
import json
import logging
import os
import tempfile
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from offline_entrypoint import MODEL_PATH, build_candidate, extract_aligned_words, verify_model
    from vast_adapter import ContractError, batch_and_restitch, parse_request, slice_wav
    from parakeet_pool import ParakeetPool
except ModuleNotFoundError:
    from asr.offline_entrypoint import MODEL_PATH, build_candidate, extract_aligned_words, verify_model
    from asr.vast_adapter import ContractError, batch_and_restitch, parse_request, slice_wav
    from asr.parakeet_pool import ParakeetPool


LOGGER = logging.getLogger("parakeet.server")
CHUNK_BATCH = max(1, int(os.environ.get("PARAKEET_CHUNK_BATCH", "8")))


def _cuda_available():
    return torch is not None and torch.cuda.is_available()



def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


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
        self.error = None
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
            models = [getattr(lane, "model", None) for lane in pool.instances]
            if len(pool.instances) != self.instance_count or len({id(model) for model in models}) != self.instance_count or any(not callable(getattr(model, "transcribe", None)) for model in models):
                raise ValueError("model lanes are unusable")
        except Exception as exc:
            LOGGER.exception("parakeet initialization failed")
            with self._state_lock:
                self.error = str(exc)
                self.state = "failed"
            return
        with self._state_lock:
            self.pool = pool
            self.model = models[0]
            self.state = "ready"
        LOGGER.info(
            "parakeet_runtime device=%s dtype_policy=%s lane_count=%d",
            "cuda" if _cuda_available() else "cpu",
            "bf16-autocast" if _cuda_available() else "fp32-cpu",
            self.instance_count,
        )

    def start_initialization(self):
        threading.Thread(target=self.initialize_once, daemon=True).start()

    def health(self):
        if self.state == "ready":
            return 200, {"status": "ready"}
        if self.state == "failed":
            return 503, {"status": "failed", "cause": "initialization failed", "error": self.error}
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
        files, chunk_request_indexes = [], []
        request_batches = [0] * len(requests)
        started = time.monotonic()
        try:
            for request_index, request in enumerate(requests):
                for start, end in request.chunks:
                    audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    files.append(audio.name)
                    try:
                        audio.write(slice_wav(request.audio, start, end))
                    finally:
                        audio.close()
                    chunk_request_indexes.append(request_index)
            lane = self.pool.checkout()
            try:
                chunk_segments = [[] for _ in requests]
                offset = 0
                while offset < len(files):
                    size, retried = min(CHUNK_BATCH, len(files) - offset), False
                    while True:
                        paths = files[offset:offset + size]
                        for request_index in set(chunk_request_indexes[offset:offset + size]):
                            request_batches[request_index] += 1
                        try:
                            if _cuda_available():
                                with torch.cuda.stream(lane.stream):
                                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                        hypotheses = lane.model.transcribe(paths, batch_size=len(paths), timestamps=True)
                            else:
                                hypotheses = lane.model.transcribe(paths, batch_size=len(paths), timestamps=True)
                            break
                        except Exception as error:
                            if retried or size == 1 or not _cuda_available() or not isinstance(error, torch.cuda.OutOfMemoryError):
                                raise
                            size //= 2
                            retried = True
                        finally:
                            if lane.stream is not None:
                                lane.stream.synchronize()
                    if len(hypotheses) != len(paths):
                        raise ContractError("model result count must equal chunk count")
                    for hypothesis, request_index in zip(hypotheses, chunk_request_indexes[offset:offset + size]):
                        timestamp = getattr(hypothesis, "timestamp", None)
                        words = timestamp.get("word") if isinstance(timestamp, dict) else None
                        LOGGER.info(
                            "parakeet_inference result aligned_words=%d word_confidence_present=%s hypothesis_type=%s",
                            len(words) if isinstance(words, list) else 0,
                            getattr(hypothesis, "word_confidence", None) is not None,
                            type(hypothesis).__name__,
                        )
                        chunk_segments[request_index].append(extract_aligned_words(hypothesis))
                    offset += size
                segments = [batch_and_restitch(chunk_segments[index], request.chunks) for index, request in enumerate(requests)]
                elapsed = time.monotonic() - started
                for request_index, request in enumerate(requests):
                    LOGGER.info(
                        "parakeet_chunked_inference filename=%s chunk_count=%d sub_batches=%d elapsed_seconds=%.3f",
                        request.audio_filename,
                        len(request.chunks),
                        request_batches[request_index],
                        elapsed,
                    )
                return segments
            finally:
                self.pool.checkin(lane)
        finally:
            for path in files:
                try: os.unlink(path)
                except FileNotFoundError: pass

    def transcribe(self, payload):
        return self.transcribe_batch([payload])[0]

    def transcribe_batch(self, payloads):
        if not self.ready:
            raise NotReadyError("runtime is not ready")
        requests = [parse_request(payload) for payload in payloads]
        LOGGER.info(
            "parakeet_batch request_count=%d filenames=%s durations=%s audio_bytes=%s",
            len(requests),
            [request.audio_filename for request in requests],
            [request.audio_duration_seconds for request in requests],
            [len(request.audio) for request in requests],
        )
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
            except json.JSONDecodeError as error:
                LOGGER.warning("parakeet_http status=400 category=contract reason=%s", error)
                self._send(400, {"error": "invalid request"})
            except ContractError as error:
                LOGGER.warning("parakeet_http status=400 category=contract reason=%s", error)
                self._send(400, {"error": "invalid request"})
            except ValueError as error:
                LOGGER.warning("parakeet_http status=400 category=contract reason=%s", error)
                self._send(400, {"error": "invalid request"})
            except Exception as error:
                LOGGER.exception("parakeet_http status=500 category=%s", type(error).__name__)
                self._send(500, {"error": "internal error", "type": type(error).__name__, "message": str(error)})

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(address, Handler)


def main():
    configure_logging()
    make_server().serve_forever()


if __name__ == "__main__":
    main()
