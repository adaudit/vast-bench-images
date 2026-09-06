#!/usr/bin/env python3
"""Minimal HTTP boundary for the immutable Parakeet-v3 candidate."""
import json
import logging
import os
import tempfile
import sys
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from offline_entrypoint import MODEL_PATH, build_candidate, extract_aligned_words, guard_model_decoding, verify_model
    from vast_adapter import ContractError, batch_and_restitch, parse_request, slice_wav
    from parakeet_pool import ParakeetPool
except ModuleNotFoundError:
    from asr.offline_entrypoint import MODEL_PATH, build_candidate, extract_aligned_words, guard_model_decoding, verify_model
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
        self._stats_lock = threading.Lock()
        self._recent = deque(maxlen=256)
        self._checked_out_lanes = 0
        self._max_concurrent_lanes = 0

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

    def stats(self):
        with self._stats_lock:
            return {
                "recent": [dict(entry) for entry in self._recent],
                "max_concurrent_lanes": self._max_concurrent_lanes,
                "lane_count": self.instance_count,
            }

    def _checkout_lane(self):
        lane = self.pool.checkout()
        with self._stats_lock:
            self._checked_out_lanes += 1
            self._max_concurrent_lanes = max(self._max_concurrent_lanes, self._checked_out_lanes)
        return lane

    def _checkin_lane(self, lane):
        with self._stats_lock:
            self._checked_out_lanes -= 1
        self.pool.checkin(lane)

    def _record_timing(self, lane_index, chunk_count, sub_batches, audio_seconds, gpu_seconds, started, finished):
        with self._stats_lock:
            max_concurrent_lanes = self._max_concurrent_lanes
            self._recent.append({
                "lane_index": lane_index,
                "chunk_count": chunk_count,
                "sub_batches": sub_batches,
                "audio_seconds": audio_seconds,
                "gpu_seconds": gpu_seconds,
                "started_monotonic": started,
                "finished_monotonic": finished,
                "max_concurrent_lanes": max_concurrent_lanes,
            })
        LOGGER.info(
            "parakeet_request_timing lane_index=%s chunk_count=%d sub_batches=%d audio_seconds=%.3f gpu_seconds=%.3f started_monotonic=%.6f finished_monotonic=%.6f max_concurrent_lanes=%d",
            lane_index,
            chunk_count,
            sub_batches,
            audio_seconds,
            gpu_seconds,
            started,
            finished,
            max_concurrent_lanes,
        )

    def check_ready(self):
        return self.ready

    def _load_model(self):
        from nemo.collections.asr.models import ASRModel
        from omegaconf import open_dict
        model = ASRModel.restore_from(str(MODEL_PATH))
        with open_dict(model.cfg.decoding):
            model.cfg.decoding.compute_timestamps = True
            model.cfg.decoding.preserve_alignments = True
            model.cfg.decoding.confidence_cfg = {"preserve_token_confidence": True, "preserve_word_confidence": False}
            if "greedy" not in model.cfg.decoding:
                model.cfg.decoding.greedy = {}
            # NeMo's CUDA-graph TDT decoder captures a stream per lane and crashes when lanes decode concurrently.
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(model.cfg.decoding, verbose=False)
        guard_model_decoding(model)
        return model

    def _transcribe_many(self, requests):
        files, chunk_request_indexes = [], []
        started = time.monotonic()
        lane = None
        lane_index = None
        gpu_started = None
        sub_batches = 0
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
            lane = self._checkout_lane()
            lane_index = next(index for index, candidate in enumerate(self.pool.instances) if candidate is lane)
            gpu_started = time.monotonic()
            try:
                chunk_segments = [[] for _ in requests]
                offset = 0
                while offset < len(files):
                    size, retried = min(CHUNK_BATCH, len(files) - offset), False
                    while True:
                        paths = files[offset:offset + size]
                        try:
                            sub_batches += 1
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
                            "parakeet_inference result aligned_words=%d token_confidence_present=%s hypothesis_type=%s",
                            len(words) if isinstance(words, list) else 0,
                            getattr(hypothesis, "token_confidence", None) is not None,
                            type(hypothesis).__name__,
                        )
                        chunk_segments[request_index].append(extract_aligned_words(hypothesis))
                    offset += size
                return [batch_and_restitch(chunk_segments[index], request.chunks) for index, request in enumerate(requests)]
            finally:
                self._checkin_lane(lane)
        finally:
            finished = time.monotonic()
            self._record_timing(
                lane_index,
                len(chunk_request_indexes),
                sub_batches,
                sum(request.audio_duration_seconds for request in requests),
                (finished - gpu_started) if gpu_started is not None else 0.0,
                started,
                finished,
            )
            for path in files:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

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
            if self.path not in ("/transcribe", "/transcribe-batch", "/stats") or self.headers.get("Content-Type") != "application/json":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 700 * 1024 * 1024:
                    raise ContractError("request body is outside the permitted limit")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/stats":
                    if payload != {}:
                        raise ContractError("stats request must be an empty JSON object")
                    self._send(200, runtime.stats())
                    return
                if not runtime.ready:
                    raise NotReadyError("runtime is not ready")
                self._send(200, runtime.transcribe(payload) if self.path == "/transcribe" else runtime.transcribe_batch(payload["requests"]))
            except NotReadyError:
                LOGGER.info("parakeet_http status=503 category=not_ready")
                self._send(503, {"error": "not ready"})
            except json.JSONDecodeError as error:
                LOGGER.warning("parakeet_http status=400 category=contract reason=%s", error)
                self._send(400, {"error": "invalid request", "reason": "invalid JSON"})
            except ContractError as error:
                LOGGER.warning("parakeet_http status=400 category=contract reason=%s", error)
                self._send(400, {"error": "invalid request", "reason": str(error)[:300]})
            except ValueError as error:
                LOGGER.warning("parakeet_http status=400 category=contract reason=%s", error)
                self._send(400, {"error": "invalid request", "reason": str(error)[:300]})
            except Exception as error:
                LOGGER.exception("parakeet_http status=500 category=%s", type(error).__name__)
                reason = " | ".join(
                    f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}"
                    for frame in traceback.extract_tb(error.__traceback__)[-3:]
                )[:600]
                self._send(500, {"error": "internal error", "type": type(error).__name__, "message": str(error), "reason": reason})

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(address, Handler)


def main():
    configure_logging()
    make_server().serve_forever()


if __name__ == "__main__":
    main()
