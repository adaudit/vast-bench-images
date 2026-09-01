import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("vast_failure_guard", Path(__file__).with_name("vast_failure_guard.py"))
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class FakeClient:
    def __init__(self): self.calls, self.groups, self.instances = [], {9}, {10, 11, 99}
    def cancel(self, request): self.calls.append(("cancel", request))
    def delete_workergroup(self, ident): self.calls.append(("group", ident)); self.groups.discard(ident)
    def destroy_instance(self, ident): self.calls.append(("instance", ident)); self.instances.discard(ident)
    def workergroup_exists(self, ident): return ident in self.groups
    def instance_exists(self, ident): return ident in self.instances


class VastFailureGuardTest(unittest.TestCase):
    def test_three_failures_suspend_only_resolved_resources_and_success_resets(self):
        client, breaker = FakeClient(), guard.VastFailureGuard(FakeClient())
        client = breaker.client
        target = guard.Target(35304, 9, (10, 11))
        self.assertEqual(breaker.failure("request-1", target)["failures"], 1)
        self.assertEqual(breaker.failure("request-2", target)["failures"], 2)
        evidence = breaker.failure("request-3", target)
        self.assertTrue(breaker.suspended)
        self.assertTrue(evidence["explicit_recreation_required"])
        self.assertEqual(client.instances, {99})
        self.assertEqual(client.calls, [("cancel", "request-3"), ("group", 9), ("instance", 10), ("instance", 11)])
        breaker.success(False); self.assertTrue(breaker.suspended)
        with self.assertRaisesRegex(RuntimeError, "half-open"):
            breaker.success(True)
        self.assertTrue(breaker.suspended); self.assertEqual(breaker.failures, 3)

    def test_ambiguous_target_fails_closed(self):
        with self.assertRaises(ValueError):
            guard.VastFailureGuard(FakeClient()).failure("request", guard.Target(35304, 0, (10,)))

    def test_state_survives_process_boundary_and_polls_to_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory, "guard.json")
            client = FakeClient()
            target = guard.Target(35304, 9, (10, 11))
            breaker = guard.VastFailureGuard(client, state_path=state)
            breaker.failure("request-1", target); breaker.failure("request-2", target)
            self.assertEqual(guard.VastFailureGuard(client, state_path=state).failures, 2)
            breaker.failure("request-3", target)
            self.assertTrue(guard.VastFailureGuard(client, state_path=state).suspended)

    def test_duplicate_attempt_is_idempotent(self):
        breaker = guard.VastFailureGuard(FakeClient())
        target = guard.Target(35304, 9, (10, 11))
        breaker.failure("attempt", target)
        self.assertEqual(breaker.failure("attempt", target)["event"], "vast_duplicate_failure")
        self.assertEqual(breaker.failures, 1)

    def test_missing_endpoint_id_in_durable_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory, "guard.json")
            state.write_text('{"failures": 2, "suspended": false}')
            with self.assertRaisesRegex(ValueError, "endpoint"):
                guard.VastFailureGuard(FakeClient(), state_path=state)

    def test_malformed_or_inconsistent_durable_state_fails_closed(self):
        valid = {"endpoint_id": 35304, "failures": 2, "suspended": False, "half_open": False, "last_attempt": "two"}
        malformed = [
            {key: value for key, value in valid.items() if key != "last_attempt"},
            dict(valid, failures=True),
            dict(valid, suspended="false"),
            dict(valid, unexpected=True),
            dict(valid, failures=3),
            dict(valid, failures=3, suspended=True, half_open=True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory, "guard.json")
            for value in malformed:
                state.write_text(__import__("json").dumps(value))
                with self.assertRaises(ValueError):
                    guard.VastFailureGuard(FakeClient(), state_path=state)

    def test_persisted_failure_requires_attributed_last_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory, "guard.json")
            state.write_text('{"endpoint_id":35304,"failures":1,"suspended":false,"half_open":false,"last_attempt":""}')
            with self.assertRaises(ValueError):
                guard.VastFailureGuard(FakeClient(), state_path=state)

    def test_third_failure_is_open_before_ambiguous_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory, "guard.json")
            breaker = guard.VastFailureGuard(FakeClient(), state_path=state)
            target = guard.Target(35304, 9, (10, 11))
            breaker.failure("one", target); breaker.failure("two", target)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                breaker.failure("three", guard.Target(35304, 0, (10,)))
            reloaded = guard.VastFailureGuard(FakeClient(), state_path=state)
            self.assertTrue(reloaded.suspended)
            self.assertEqual(reloaded.failures, 3)

    def test_explicit_half_open_success_closes_and_resets(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory, "guard.json")
            breaker = guard.VastFailureGuard(FakeClient(), state_path=state)
            target = guard.Target(35304, 9, (10, 11))
            for attempt in ("one", "two", "three"):
                breaker.failure(attempt, target)
            breaker.begin_half_open()
            self.assertFalse(breaker.suspended)
            self.assertTrue(breaker.half_open)
            self.assertTrue(guard.VastFailureGuard(FakeClient(), state_path=state).half_open)
            breaker.success(True)
            reloaded = guard.VastFailureGuard(FakeClient(), state_path=state)
            self.assertFalse(reloaded.suspended)
            self.assertFalse(reloaded.half_open)
            self.assertEqual(reloaded.failures, 0)


if __name__ == "__main__": unittest.main()
