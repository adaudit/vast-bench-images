import contextlib
import hashlib
import io
import sys
import threading
import time
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from asr import batch_runtime as batch, offline_entrypoint as pin, server
from asr.inference_runtime import BatchControl


CUDAOutOfMemory = type('OutOfMemoryError', (RuntimeError,), {'__module__': 'torch'})


def audio(seconds=1, silence=None):
    samples = bytearray(b'\xe8\x03' * round(seconds * 16000))
    if silence:
        start, end = (round(point * 16000) for point in silence)
        samples[start * 2:end * 2] = bytes((end - start) * 2)
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(samples)
    return output.getvalue()


def hypothesis():
    return SimpleNamespace(timestamp={'word': [{'word': 'evidence', 'start': .1, 'end': .2}]}, word_confidence=[.8])


def request(seconds=1):
    return SimpleNamespace(audio=audio(seconds), audio_duration_seconds=seconds)


class OOMRecoveryTest(unittest.TestCase):
    def torch(self):
        module = ModuleType('torch')
        module.cuda = SimpleNamespace(is_available=lambda: True, empty_cache=Mock())
        return module

    def test_cuda_oom_retries_once_exclusively_then_succeeds(self):
        torch = self.torch()
        model = SimpleNamespace(transcribe=Mock(side_effect=[CUDAOutOfMemory('PRIVATE'), [hypothesis()]]))
        runtime = server.Runtime(model_loader=lambda: model)
        with patch.dict(sys.modules, {'torch': torch}):
            result = runtime._transcribe_many([request()], model=model)
        self.assertEqual(model.transcribe.call_count, 2)
        self.assertEqual(result[0][0]['text'], 'evidence')
        self.assertGreaterEqual(torch.cuda.empty_cache.call_count, 1)
        self.assertTrue(all(call.kwargs['verbose'] is False for call in model.transcribe.call_args_list))

    def test_second_oom_is_typed_and_does_not_fabricate_partial_success(self):
        torch = self.torch()
        model = SimpleNamespace(transcribe=Mock(side_effect=CUDAOutOfMemory('PRIVATE')))
        runtime = server.Runtime(model_loader=lambda: model)
        with patch.dict(sys.modules, {'torch': torch}), self.assertRaises(MemoryError):
            runtime._transcribe_many([request()], model=model)
        self.assertEqual(model.transcribe.call_count, 2)
        self.assertGreaterEqual(torch.cuda.empty_cache.call_count, 2)

    def test_call_chunks_are_at_most_301_seconds_and_use_three_independent_lanes(self):
        active = maximum = 0
        lock = threading.Lock()
        barrier = threading.Barrier(3)
        calls = []
        def loader():
            model = SimpleNamespace()
            def transcribe(files, **kwargs):
                nonlocal active, maximum
                with wave.open(files[0], 'rb') as wav:
                    duration = wav.getnframes() / wav.getframerate()
                with lock:
                    calls.append((id(model), duration, kwargs['batch_size'], kwargs['verbose']))
                    active += 1
                    maximum = max(maximum, active)
                try:
                    barrier.wait(timeout=3)
                    return [hypothesis()]
                finally:
                    with lock:
                        active -= 1
            model.transcribe = transcribe
            return model
        runtime = server.Runtime(model_verifier=lambda _: None, model_loader=loader)
        runtime.initialize_once()
        self.assertTrue(runtime.ready)
        control = BatchControl()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = runtime._transcribe_many([request(900)], control=control)[0]
            control.finish()
        lines = output.getvalue().splitlines()
        units = [line for line in lines if line.startswith('UNIT ')]
        self.assertEqual(len(units), 3)
        self.assertEqual({int(line.split()[1].split('=')[1]) for line in units}, {1, 2, 3})
        for line in units:
            self.assertRegex(line, r'^UNIT lane=[1-3] audio_ms=[0-9]+ wall_ms=[0-9]+ start=[0-9]{4}-')
        self.assertRegex(lines[-1], r'^BATCH_LANES max_concurrent=3 units=3 audio_s=902[.]000 wall_s=[0-9]+[.][0-9]{3}$')
        self.assertEqual(maximum, 3)
        self.assertEqual(len({row[0] for row in calls}), 3)
        self.assertEqual(sorted(row[1] for row in calls), [300, 301, 301])
        self.assertTrue(all(row[2:] == (1, False) for row in calls))
        self.assertEqual(len(result), 3)
        self.assertEqual([round(word['start_seconds'], 1) for word in result], [.1, 300.1, 600.1])

    def test_oom_retry_waits_for_other_active_lane_to_finish(self):
        entered, release, finished, first_failed, retry_started = [threading.Event() for _ in range(5)]
        def other(*args, **kwargs):
            entered.set()
            release.wait(3)
            finished.set()
            return [hypothesis()]
        attempts = []
        def failing(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                first_failed.set()
                raise CUDAOutOfMemory('PRIVATE')
            self.assertTrue(finished.is_set())
            retry_started.set()
            return [hypothesis()]
        runtime = server.Runtime(model_loader=lambda: None)
        with patch.dict(sys.modules, {'torch': self.torch()}), ThreadPoolExecutor(2) as workers:
            other_future = workers.submit(runtime._transcribe_many, [request()], model=SimpleNamespace(transcribe=other))
            self.assertTrue(entered.wait(1))
            retry_future = workers.submit(runtime._transcribe_many, [request()], model=SimpleNamespace(transcribe=failing))
            try:
                self.assertTrue(first_failed.wait(1))
                self.assertFalse(retry_started.wait(.05))
            finally:
                release.set()
            other_future.result(timeout=4)
            retry_future.result(timeout=4)
        self.assertTrue(retry_started.is_set())


class BatchCancellationTest(unittest.TestCase):
    def payload(self, count=30):
        body = audio()
        receipt = {'sha256': hashlib.sha256(body).hexdigest(), 'size_bytes': len(body), 'media_type': 'audio/wav',
                   'canonical_sample_rate': 16000, 'canonical_channel_count': 1, 'canonical_frame_count': 16000}
        params = {'request_version': pin.REQUEST_VERSION, 'model_id': pin.MODEL_ID, 'model_revision': pin.MODEL_REVISION,
                  'audio_duration_seconds': 1, 'audio_receipt': receipt}
        return body, {'batch_id': 'synthetic', 'items': [{'id': format(i, '064x'), 'lane': 'parakeet_v3',
                                                       'audio_url': 'synthetic', 'params': params} for i in range(count)]}

    def test_closing_stream_stops_queued_transcriptions(self):
        body, payload = self.payload()
        calls = []
        def transcribe(_):
            calls.append(1)
            time.sleep(.02)
            return pin.build_candidate(1, [])
        events = batch.batch_events(SimpleNamespace(transcribe=transcribe), payload,
                                    fetcher=lambda _, path: path.write_bytes(body))
        self.assertEqual(next(events)['type'], 'accepted')
        while next(events)['type'] != 'item':
            pass
        before = len(calls)
        events.close()
        time.sleep(.06)
        self.assertEqual(len(calls), before)
        self.assertLess(len(calls), len(payload['items']))

    def test_batch_deadline_returns_without_draining_all_inference(self):
        body, payload = self.payload()
        calls = []
        finished = threading.Event()
        def transcribe(_):
            calls.append(1)
            time.sleep(.2)
            finished.set()
            return pin.build_candidate(1, [])
        started = time.monotonic()
        events = list(batch.batch_events(SimpleNamespace(transcribe=transcribe), payload,
                                        fetcher=lambda _, path: path.write_bytes(body), deadline_seconds=.04,
                                        keepalive_seconds=.01))
        self.assertLess(time.monotonic() - started, .15)
        self.assertEqual(events[-1]['type'], 'end')
        self.assertLessEqual(len(calls), 3)
        self.assertTrue(finished.wait(1))


if __name__ == '__main__':
    unittest.main()
