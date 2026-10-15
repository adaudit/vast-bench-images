"""Hourly batch protocol, with no credentials or recording content in logs."""
import base64
import hashlib
import http.client
import ipaddress
import json
import math
import re
import socket
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from urllib.parse import urlsplit

try:
    from offline_entrypoint import MODEL_ID, MODEL_REVISION, REQUEST_VERSION, AlignmentError, ConfidenceMismatchError
except ModuleNotFoundError:
    from asr.offline_entrypoint import MODEL_ID, MODEL_REVISION, REQUEST_VERSION, AlignmentError, ConfidenceMismatchError

MAX_AUDIO_BYTES = 512 * 1024 * 1024
_DNS = ThreadPoolExecutor(max_workers=8, thread_name_prefix="asr-dns")
_FAILURE_LOG_LOCK = threading.Lock()


class _ItemFailure(Exception):
    def __init__(self, stage, reason):
        self.stage, self.reason = stage, reason


def _failure_details(error, stage):
    if isinstance(error, _ItemFailure):
        return error.stage, error.reason
    if isinstance(error, MemoryError) or any(
        cls.__name__ == "OutOfMemoryError" and cls.__module__.split(".")[0] == "torch"
        for cls in type(error).__mro__
    ):
        return stage, "oom"
    if stage == "download":
        if isinstance(error, TimeoutError):
            return stage, "download_deadline"
        return stage, "download_rejected" if isinstance(error, ValueError) else "download_error"
    if stage == "validate":
        return stage, "receipt_mismatch"
    if isinstance(error, ConfidenceMismatchError):
        return "confidence", "confidence_mismatch"
    if isinstance(error, AlignmentError):
        return "alignment", "alignment_invalid"
    return stage, "transcribe_error" if isinstance(error, (RuntimeError, ValueError, OSError)) else "unknown"


def _https_request(url, output, *, body=None, limit=MAX_AUDIO_BYTES, deadline_seconds=180):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment or parsed.port not in (None, 443):
        raise ValueError("invalid audio URL")
    deadline = time.monotonic() + deadline_seconds
    expired = threading.Event()
    sockets = []
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=min(60, deadline_seconds))

    def remaining():
        seconds = deadline - time.monotonic()
        if expired.is_set() or seconds <= 0:
            raise TimeoutError("audio deadline exceeded")
        return seconds

    def abort():
        expired.set()
        for sock in [connection.sock, *sockets]:
            if sock is not None:
                try: sock.shutdown(socket.SHUT_RDWR)
                except OSError: pass
        connection.close()

    def public_connect(address, timeout, source_address=None):
        # Bound DNS as well as connect/read. An OS resolver stall cannot occupy
        # the batch download thread indefinitely; the resolver pool is bounded.
        future = _DNS.submit(socket.getaddrinfo, address[0], address[1], type=socket.SOCK_STREAM)
        try:
            addresses = future.result(timeout=remaining())
        finally:
            future.cancel()
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise ValueError("non-public audio address")
        for family, kind, protocol, _, target in addresses:
            sock = socket.socket(family, kind, protocol)
            try:
                sock.settimeout(min(timeout, remaining()))
                sock.connect(target)
                return sock
            except OSError:
                sock.close()
        raise ValueError("audio connection unavailable")

    connection._create_connection = public_connect
    timer = threading.Timer(deadline_seconds, abort)
    timer.daemon = True
    timer.start()
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request("POST" if body is not None else "GET", parsed.path + ("?" + parsed.query if parsed.query else ""), body=body, headers=headers)
        sockets.append(connection.sock)  # retain TLS socket even after HTTP/1.0 headers
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("audio download rejected")
        size = 0
        with output.open("wb") as destination:
            while True:
                remaining()
                block = response.read1(1024 * 1024)
                if not block: break
                size += len(block)
                if size > limit: raise ValueError("audio too large")
                destination.write(block)
        remaining()
    except Exception:
        if expired.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("audio deadline exceeded") from None
        raise
    finally:
        timer.cancel()
        connection.close()


def fetch_audio(url, output, *, deadline_seconds=180):
    _https_request(url, output, deadline_seconds=deadline_seconds)


def refresh_audio_url(capability):
    # The capability is exact-lease, read-only, and not a worker session. This
    # request occurs when a download slot starts, not while waiting in a queue.
    endpoint = capability["endpoint"]
    parsed = urlsplit(endpoint)
    if parsed.path != "/internal/call-pipeline/asr/batch/audio-url" or parsed.query:
        raise ValueError("invalid refresh endpoint")
    with tempfile.TemporaryDirectory(prefix="asr-audio-ticket-") as directory:
        output = Path(directory) / "response.json"
        _https_request(endpoint, output, body=json.dumps({"token": capability["token"]}).encode(), limit=64 * 1024, deadline_seconds=30)
        result = json.loads(output.read_bytes())
        if not isinstance(result.get("audio_url"), str): raise ValueError("invalid refresh response")
        return result["audio_url"]


def validate_audio(path, params):
    receipt = params["audio_receipt"]
    if path.stat().st_size != receipt["size_bytes"]: raise ValueError("audio receipt size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""): digest.update(block)
    if digest.hexdigest() != receipt["sha256"]: raise ValueError("audio receipt checksum mismatch")
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getframerate() != 16000 or audio.getsampwidth() != 2 or audio.getcomptype() != "NONE" or audio.getnframes() != receipt["canonical_frame_count"] or receipt["canonical_sample_rate"] != 16000 or receipt["canonical_channel_count"] != 1 or receipt["media_type"] != "audio/wav" or abs(audio.getnframes() / 16000 - params["audio_duration_seconds"]) > 1e-6:
            raise ValueError("audio receipt format mismatch")


def batch_events(runtime, payload, *, fetcher=fetch_audio, refresher=refresh_audio_url, keepalive_seconds=10):
    batch_id, items = payload["batch_id"], payload["items"]
    if not isinstance(batch_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", batch_id) or not isinstance(items, list) or not items: raise ValueError("invalid batch")
    ids = [item.get("id") for item in items]
    if any(not isinstance(item_id, str) or not re.fullmatch(r"[a-f0-9]{64}", item_id) for item_id in ids) or len(set(ids)) != len(ids): raise ValueError("invalid batch item identities")

    def event(kind, **fields): return {"type": kind, "batch_id": batch_id, **fields}

    def failure(index, error, stage):
        stage, reason = _failure_details(error, stage)
        with _FAILURE_LOG_LOCK:
            print(f"ITEM_FAILED stage={stage} reason={reason}", flush=True)
        return event("item", id=ids[index], status="failed", stage=stage, reason=reason)

    def fetch(index, directory):
        item, params = items[index], items[index]["params"]
        if item["lane"] != "parakeet_v3" or (params["model_id"], params["model_revision"], params["request_version"]) != (MODEL_ID, MODEL_REVISION, REQUEST_VERSION): raise ValueError("batch model identity mismatch")
        duration = params["audio_duration_seconds"]
        if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not math.isfinite(duration) or not 0 < duration <= 86400: raise ValueError("invalid duration")
        path = directory / (str(index) + ".wav")
        refresh = item.get("audio_refresh")
        url = refresher(refresh) if refresh and refresh.get("token") else item["audio_url"]
        stage = "download"
        try:
            fetcher(url, path)
            stage = "validate"
            validate_audio(path, params)
            return path
        except Exception as error:
            path.unlink(missing_ok=True)
            raise _ItemFailure(*_failure_details(error, stage)) from None

    def infer(index, path):
        try:
            params = items[index]["params"]
            request = {key: params[key] for key in ("request_version", "model_id", "model_revision", "audio_duration_seconds")}
            request.update(lane="parakeet_v3", audio_filename="claimed.wav", audio_base64=base64.b64encode(path.read_bytes()).decode())
            return event("item", id=ids[index], status="succeeded", candidate=runtime.transcribe(request))
        except Exception as error:
            return failure(index, error, "transcribe")
        finally:
            path.unlink(missing_ok=True)

    yield event("accepted")
    with tempfile.TemporaryDirectory(prefix="asr-batch-") as directory:
        with ThreadPoolExecutor(max_workers=8) as downloads, ThreadPoolExecutor(max_workers=3) as lanes:
            pending = {downloads.submit(fetch, index, Path(directory)): ("download", index) for index in range(len(items))}
            while pending:
                done, _ = wait(pending, timeout=keepalive_seconds, return_when=FIRST_COMPLETED)
                if not done: yield event("keepalive")
                for future in done:
                    kind, index = pending.pop(future)
                    if kind == "inference":
                        yield future.result()
                    else:
                        try:
                            path = future.result()
                            pending[lanes.submit(infer, index, path)] = ("inference", index)
                        except Exception as error:
                            yield failure(index, error, "download")
    yield event("end")
