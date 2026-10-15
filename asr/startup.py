"""Ordered in-process startup checkpoints; GPU objects never cross processes."""
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class Stages:
    def __init__(self, marker_dir=None, *, attempts=3, sleep=time.sleep, log=print):
        self.marker_dir = Path(marker_dir) if marker_dir else None
        self.attempts, self.sleep, self.log = attempts, sleep, log
        self.completed = {}
        self._log_lock = threading.Lock()
        if self.marker_dir:
            self.marker_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message, **kwargs):
        # Keep each STAGE line intact while independent benchmark actions overlap.
        with self._log_lock:
            self.log(message, **kwargs)

    def run(self, name, action):
        if name in self.completed:
            return self.completed[name]
        for attempt in range(self.attempts):
            started = time.monotonic()
            stamp = datetime.now(timezone.utc).isoformat()
            self._log(f"{stamp} STAGE {name} start elapsed_ms=0", flush=True)
            try:
                result = action()
            except Exception as exc:
                elapsed = round((time.monotonic() - started) * 1000)
                # Fixed reason codes only: exception text may contain credentials,
                # URLs, or sensitive absolute paths.
                reason = next((reason for cls, reason in (
                    (FileNotFoundError, "required_file_missing"),
                    (ImportError, "dependency_import_failed"),
                    (AssertionError, "invariant_failed"),
                    (ValueError, "invalid_configuration"),
                ) if isinstance(exc, cls)), "operation_failed")
                self._log(f"{datetime.now(timezone.utc).isoformat()} STAGE {name} end elapsed_ms={elapsed} status=retry error={type(exc).__name__} reason={reason}", flush=True)
                if attempt + 1 == self.attempts:
                    raise RuntimeError("startup stage exhausted") from None
                self.sleep(min(2 ** attempt, 8))
                continue
            self.completed[name] = result
            elapsed = round((time.monotonic() - started) * 1000)
            if self.marker_dir:
                marker = self.marker_dir / (name + ".done")
                temporary = marker.with_suffix(".tmp")
                temporary.write_text(json.dumps({"pid": os.getpid(), "elapsed_ms": elapsed}) + "\n")
                temporary.replace(marker)
            self._log(f"{datetime.now(timezone.utc).isoformat()} STAGE {name} end elapsed_ms={elapsed} status=ok", flush=True)
            return result
        raise AssertionError("unreachable")


def verify_extracted_model(root=Path("/workspace/models/parakeet-v3")):
    manifest = json.loads((root / "sha256-manifest.json").read_text())
    if not manifest or not isinstance(manifest, dict):
        raise RuntimeError("model manifest missing")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "sha256-manifest.json"}
    if actual != set(manifest):
        raise RuntimeError("model manifest file set mismatch")
    for relative, expected in manifest.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or root.resolve() not in path.resolve().parents:
            raise RuntimeError("model manifest path invalid")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError("model manifest checksum mismatch")
