#!/usr/bin/env python3
"""Minimal HTTP boundary for the immutable Parakeet-v3 candidate."""
import copy
import json
import logging
import os
import tempfile
import threading
import base64
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor

try:
    from offline_entrypoint import MODEL_PATH, build_candidate, extract_aligned_words, verify_model
    from vast_adapter import ContractError, parse_request
    from parakeet_pool import ParakeetPool
    from startup import Stages, verify_extracted_model
    from batch_runtime import batch_events
except ModuleNotFoundError:
    from asr.offline_entrypoint import MODEL_PATH, build_candidate, extract_aligned_words, verify_model
    from asr.vast_adapter import ContractError, parse_request
    from asr.parakeet_pool import ParakeetPool
    from asr.startup import Stages, verify_extracted_model
    from asr.batch_runtime import batch_events


LOGGER = logging.getLogger("parakeet.server")


class NotReadyError(ContractError):
    pass


class Runtime:
    def __init__(self, *, model_verifier=None, model_loader=None, transcriber=None, batch_transcriber=None, instance_count=3, stages=None, benchmark=None):
        if instance_count != 3 or os.environ.get("PARAKEET_INSTANCES", "3") != "3":
            raise ValueError("PARAKEET_INSTANCES must be exactly 3")
        self.model_verifier = model_verifier or (lambda path: verify_extracted_model(path.parent / "parakeet-v3") if (path.parent / "parakeet-v3").is_dir() else verify_model(path))
        self._production = model_loader is None
        self.stages = stages or Stages(os.environ.get("ASR_STARTUP_MARKER_DIR", "/tmp/asr-startup-markers") if self._production else None)
        self.benchmark = benchmark or (self._benchmark if self._production else lambda model: None)
        self._lanes = []
        self._streams = {}
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
            self.stages.run("deps", self._check_dependencies)
            self.stages.run("model_files", lambda: self.model_verifier(MODEL_PATH))
            for index in range(self.instance_count):
                def load():
                    model = self.model_loader()
                    if model is None or not callable(getattr(model, "transcribe", None)) or any(model is old for old in self._lanes):
                        raise ValueError("model lane is unusable")
                    self._lanes.append(model)
                    return model
                self.stages.run(f"lane_{index + 1}", load)
            pool = ParakeetPool(self.instance_count, iter(self._lanes).__next__)
            if len(pool.instances) != self.instance_count or len({id(model) for model in pool.instances}) != self.instance_count or any(model is None or not callable(getattr(model, "transcribe", None)) for model in pool.instances):
                raise ValueError("model lanes are unusable")
            self.pool = pool
            self.model = pool.instances[0]
            with ThreadPoolExecutor(max_workers=self.instance_count) as executor:
                benchmarks = [
                    executor.submit(self.stages.run, f"benchmark_{index + 1}", lambda model=model: self.benchmark(model))
                    for index, model in enumerate(pool.instances)
                ]
                for benchmark in benchmarks:
                    benchmark.result()
            self.stages.run("ready", lambda: None)
        except Exception:
            print("ASR_STARTUP_FAILED", flush=True)
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
        from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            if self._lanes:
                # Copy only the evaluated, unbenchmarked lane: no captured graphs
                # or inference buffers exist yet (NeMo 2.4.1 tdt_label_looping.py:252-257).
                model = copy.deepcopy(self._lanes[0])
            else:
                extracted = MODEL_PATH.parent / "parakeet-v3"
                if extracted.is_dir():
                    connector = SaveRestoreConnector()
                    connector.model_extracted_dir = str(extracted)
                    model = ASRModel.restore_from(str(extracted), save_restore_connector=connector, map_location=device)
                else:
                    # The separate non-serverless v3 image retains its archive layout.
                    model = ASRModel.restore_from(str(MODEL_PATH), map_location=device)
            with open_dict(model.cfg.decoding):
                model.cfg.decoding.compute_timestamps = True
                model.cfg.decoding.preserve_alignments = True
                model.cfg.decoding.confidence_cfg = {"preserve_word_confidence": True}
            # CUDA permits only one capture per process. NeMo also rebuilds the
            # decoder on timestamps=True, so keep graphs off for concurrent lanes.
            with open_dict(model.cfg.decoding.greedy):
                model.cfg.decoding.greedy.use_cuda_graph_decoder = False
            # NeMo 2.4.1 rnnt_bpe_models.py:477-497 rebuilds decoding/WER
            # with this lane's decoder and joint, including fresh graph/confidence state.
            model.change_decoding_strategy(model.cfg.decoding, verbose=False)
            model.eval()
            if device.type == "cuda":
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                self._streams[id(model)] = stream
            return model
        except Exception:
            if not self._lanes:
                # NeMo clears this global flag only after a successful restore.
                ASRModel._set_model_restore_state(is_being_restored=False)
            raise

    def _check_dependencies(self, *, require_cuda=True):
        if self._production:
            logging.disable(logging.CRITICAL)
            import importlib.metadata
            if importlib.metadata.version("nemo_toolkit") != "2.4.1":
                raise RuntimeError("unexpected NeMo version")
            import torch
            if require_cuda and not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")

    def _benchmark(self, model):
        audio = Path("/workspace/fixtures/2086-149220-0033.wav").read_bytes()
        payload = {"request_version": "parakeet-v3-offline-request-v1", "lane": "parakeet_v3", "model_id": "nvidia/parakeet-tdt-0.6b-v3", "model_revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b", "audio_filename": "benchmark.wav", "audio_duration_seconds": 7.435, "audio_base64": base64.b64encode(audio).decode()}
        build_candidate(7.435, self._transcribe_many([parse_request(payload)], model=model)[0])

    def _transcribe_many(self, requests, model=None):
        files = []
        borrowed = model is None
        try:
            for request in requests:
                audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                audio.write(request.audio); audio.close(); files.append(audio.name)
            if borrowed:
                model = self.pool.checkout()
            try:
                LOGGER.info("parakeet_inference stage=start")
                stream = self._streams.get(id(model))
                if stream is None:
                    segments = [extract_aligned_words(result) for result in model.transcribe(files, batch_size=len(files), timestamps=True, num_workers=0)]
                else:
                    import torch
                    with torch.cuda.stream(stream):
                        segments = [extract_aligned_words(result) for result in model.transcribe(files, batch_size=len(files), timestamps=True, num_workers=0)]
                LOGGER.info("parakeet_inference stage=complete")
                return segments
            finally:
                try:
                    stream = self._streams.get(id(model))
                    if stream is not None:
                        stream.synchronize()
                finally:
                    if borrowed:
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
                if self.path == "/transcribe-batch" and "items" in payload:
                    events = batch_events(runtime, payload)
                    # Validate the batch envelope before committing HTTP success.
                    first = next(events)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    try:
                        self.wfile.write(json.dumps(first, separators=(",", ":")).encode() + b"\n")
                        self.wfile.flush()
                        for event in events:
                            self.wfile.write(json.dumps(event, separators=(",", ":")).encode() + b"\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        events.close()
                    return
                self._send(200, runtime.transcribe(payload) if self.path == "/transcribe" else runtime.transcribe_batch(payload["requests"]))
                return
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
