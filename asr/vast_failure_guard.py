"""Fail-closed, endpoint-scoped circuit breaker for a Vast activation run.

The client is deliberately narrow so tests can prove target safety without
loading credentials or making a provider call.  A caller must resolve the
workergroup and instance IDs before reporting a failure.
"""
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


ENDPOINT_ID = 35304
THRESHOLD = 3


@dataclass(frozen=True)
class Target:
    endpoint_id: int
    workergroup_id: int
    instance_ids: tuple[int, ...]


class VastFailureGuard:
    def __init__(self, client, threshold=THRESHOLD, state_path=None, polls=20):
        self.client, self.threshold, self.state_path, self.polls = client, threshold, Path(state_path) if state_path else None, polls
        self.failures, self.suspended, self.half_open, self.last_attempt = self._load()

    def _load(self):
        if not self.state_path or not self.state_path.exists(): return 0, False, False, ""
        state = json.loads(self.state_path.read_text())
        if not isinstance(state, dict) or set(state) != {"endpoint_id", "failures", "suspended", "half_open", "last_attempt"} or state.get("endpoint_id") != ENDPOINT_ID or not isinstance(state.get("failures"), int) or isinstance(state["failures"], bool) or state["failures"] < 0 or not isinstance(state.get("suspended"), bool) or not isinstance(state.get("half_open"), bool) or not isinstance(state.get("last_attempt"), str) or state["failures"] and not state["last_attempt"] or state["suspended"] and state["half_open"] or (state["failures"] >= self.threshold) != (state["suspended"] or state["half_open"]):
            raise ValueError("wrong endpoint guard state")
        return state["failures"], state["suspended"], state["half_open"], state["last_attempt"]

    def _save(self):
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", dir=self.state_path.parent, delete=False) as state:
                json.dump({"endpoint_id": ENDPOINT_ID, "failures": self.failures, "suspended": self.suspended, "half_open": self.half_open, "last_attempt": self.last_attempt}, state, sort_keys=True)
                state.flush(); os.fsync(state.fileno())
                name = state.name
            os.replace(name, self.state_path)

    def success(self, validated):
        if validated:
            if self.suspended:
                raise RuntimeError("durable half-open transition required")
            self.failures, self.suspended, self.half_open = 0, False, False
            self._save()

    def begin_half_open(self):
        if not self.suspended:
            raise RuntimeError("endpoint is not suspended")
        self.suspended, self.half_open = False, True
        self._save()

    def failure(self, request_id, target):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("distinct request ID required")
        if request_id == self.last_attempt:
            return {"event": "vast_duplicate_failure", "endpoint_id": ENDPOINT_ID, "failures": self.failures}
        if self.suspended:
            return {"event": "vast_endpoint_suspended", "endpoint_id": ENDPOINT_ID, "failures": self.failures}
        self.failures += 1
        self.last_attempt = request_id
        if self.failures >= self.threshold:
            self.suspended, self.half_open = True, False
        self._save()
        if not isinstance(target, Target) or target.endpoint_id != ENDPOINT_ID or target.workergroup_id <= 0 or not target.instance_ids or len(set(target.instance_ids)) != len(target.instance_ids):
            raise ValueError("ambiguous Vast target")
        evidence = {"event": "vast_startup_or_benchmark_failure", "endpoint_id": ENDPOINT_ID, "failures": self.failures}
        if self.failures < self.threshold:
            return evidence
        self.client.cancel(request_id)
        self.client.delete_workergroup(target.workergroup_id)
        for instance_id in target.instance_ids:
            self.client.destroy_instance(instance_id)
        for _ in range(self.polls):
            if not self.client.workergroup_exists(target.workergroup_id) and not any(self.client.instance_exists(i) for i in target.instance_ids): break
        else: raise RuntimeError("targeted Vast cleanup incomplete")
        return {**evidence, "event": "vast_endpoint_suspended", "workergroup_id": target.workergroup_id, "instance_ids": list(target.instance_ids), "explicit_recreation_required": True}
