"""Process-wide bounded inference admission and batch cancellation controls."""
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

LOG_LOCK = threading.Lock()


def emit_line(line):
    with LOG_LOCK:
        try:
            print(line, flush=True)
        except (OSError, ValueError):
            # A closed diagnostic sink must not leak a model lane or its tensors.
            pass


class WorkCancelled(Exception):
    pass


class BatchControl:
    def __init__(self, deadline_seconds=None, cancel_event=None):
        seconds = float(os.environ.get("ASR_BATCH_DEADLINE_SECONDS", "1800") if deadline_seconds is None else deadline_seconds)
        if not math.isfinite(seconds) or not 0 < seconds <= 86400:
            raise ValueError("invalid batch deadline")
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self._metrics_lock = threading.Lock()
        self._active = self._maximum = self._units = 0
        self._audio = 0.0
        self._closing = self._reported = False
        self._cancel_lock = threading.RLock()
        self._callbacks = set()
        self._observer = None

    @property
    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())

    @property
    def cancelled(self):
        if self.remaining == 0:
            self.cancel_event.set()
        return self.cancel_event.is_set()

    def check(self):
        if self.cancelled:
            raise WorkCancelled("batch stopped")

    def cancel(self):
        self.cancel_event.set()
        with self._cancel_lock:
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
            for callback in callbacks:
                try:
                    callback()
                except Exception:
                    pass

    def on_cancel(self, callback):
        observer = None
        with self._cancel_lock:
            cancelled = self.cancelled
            if not cancelled:
                self._callbacks.add(callback)
                if self._observer is None:
                    def observe():
                        self.cancel_event.wait(self.remaining)
                        self.cancel()
                    observer = self._observer = threading.Thread(target=observe, daemon=True)
        if cancelled:
            callback()
        if observer is not None:
            observer.start()
        def detach():
            with self._cancel_lock:
                self._callbacks.discard(callback)
        return detach

    def unit_started(self):
        with self._metrics_lock:
            self.check()
            if self._closing:
                raise WorkCancelled("batch stopped")
            self._active += 1
            self._units += 1
            self._maximum = max(self._maximum, self._active)

    def unit_finished(self, audio_seconds):
        with self._metrics_lock:
            self._active -= 1
            self._audio += audio_seconds
            self._report_if_idle()

    def finish(self):
        self.cancel()
        with self._metrics_lock:
            self._closing = True
            self._report_if_idle()

    def _report_if_idle(self):
        if self._closing and not self._active and not self._reported:
            self._reported = True
            emit_line(f"BATCH_LANES max_concurrent={self._maximum} units={self._units} "
                      f"audio_s={self._audio:.3f} wall_s={time.monotonic() - self.started:.3f}")


def is_cuda_oom(error):
    return any(cls.__name__ == "OutOfMemoryError" and cls.__module__.split(".")[0] == "torch"
               for cls in type(error).__mro__)


class InferenceAdmission:
    """Three shared units, or one OOM retry after all other inference drains."""
    def __init__(self):
        self._condition = threading.Condition()
        self._active = 0
        self._exclusive = False
        self._waiting = 0

    @contextmanager
    def enter(self, *, exclusive=False, control=None):
        with self._condition:
            if exclusive:
                self._waiting += 1
            try:
                while self._exclusive or (self._active if exclusive else self._waiting or self._active >= 3):
                    if control is not None:
                        control.check()
                    self._condition.wait(.05)
                if control is not None:
                    control.check()
                self._active += 1
                self._exclusive = exclusive
            finally:
                if exclusive:
                    self._waiting -= 1
                    self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                if exclusive:
                    self._exclusive = False
                self._condition.notify_all()


ADMISSION = InferenceAdmission()
CHUNK_WORKERS = ThreadPoolExecutor(max_workers=3, thread_name_prefix="asr-chunk")
