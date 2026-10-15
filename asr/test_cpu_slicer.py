import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import wave
from unittest.mock import patch

from asr import vast_adapter as adapter, offline_entrypoint as pin, batch_runtime as batch
from asr.test_oom_cancellation import audio
from asr import test_oom_cancellation as fixtures
from asr import server


class CpuSlicerTest(unittest.TestCase):
    def test_latest_retained_boundary_before_target_and_exact_overlap(self):
        chunks = adapter.chunk_ranges(100, silence_points=(19, 30, 55, 60, 85))
        self.assertEqual(chunks, ((0, 31), (30, 61), (60, 100)))
        self.assertEqual(adapter.validate_chunk_plan(chunks, 100), chunks)
        self.assertTrue(all(end - start <= 41 for start, end in chunks))
        self.assertEqual(adapter.chunk_ranges(41), ((0, 41),))
        self.assertEqual(adapter.chunk_ranges(100, silence_points=(55,))[0], (0, 41))

    def test_plan_rejects_gaps_extra_overlap_long_chunks_and_noncanonical_bounds(self):
        invalid = [None, [], [[1, 40]], [[0, 39]], [[0, 42]], [[0, 20], [21, 40]],
                   [[0, 20], [18, 40]], [[0, 20], [0, 40]], [[0, 20], [19, 20]],
                   [[0, float('nan')]], [[False, 40]], [[0, 20], [19, 41]]]
        for plan in invalid:
            with self.subTest(plan=plan), self.assertRaises(adapter.ContractError):
                adapter.validate_chunk_plan(plan, 40)

    def test_slices_are_frame_aligned_and_preserve_pcm(self):
        body = audio(3)
        result = adapter.slice_wav(body, .25, 1.25)
        with wave.open(io.BytesIO(body), 'rb') as original, wave.open(io.BytesIO(result), 'rb') as sliced:
            original.setpos(4000)
            self.assertEqual(sliced.getnframes(), 16000)
            self.assertEqual(sliced.readframes(16000), original.readframes(16000))
        with self.assertRaises(adapter.ContractError):
            adapter.slice_wav(body, 2, 4)
        with self.assertRaises(adapter.ContractError):
            adapter.slice_wav(body, 1, 1)

    def test_proven_overlap_ownership_dedupe_and_monotonic_clipping(self):
        first = [{'start_seconds': 38, 'end_seconds': 40.6, 'text': 'owned', 'confidence': .9}]
        second = [{'start_seconds': .1, 'end_seconds': .5, 'text': 'owned', 'confidence': .9},
                  {'start_seconds': .4, 'end_seconds': 1, 'text': 'next', 'confidence': .8}]
        words = adapter.batch_and_restitch([first, second], ((0, 41), (40, 80)))
        self.assertEqual([word['text'] for word in words], ['owned', 'next'])
        self.assertEqual(words[-1]['start_seconds'], 40.6)
        candidate = pin.build_candidate(80, words)
        self.assertEqual(candidate['calibration']['segment_evidence'][-1]['timestamp_end_seconds'], 41)
        with self.assertRaises(adapter.ContractError):
            adapter.batch_and_restitch([first], ((0, 41), (40, 80)))

    def test_same_chunk_numeric_tdt_words_survive_restitch(self):
        _, _, spans = json.loads(Path(__file__).with_name('fixtures').joinpath('tdt-alignment-numeric.json').read_text())[2]
        hypothesis = SimpleNamespace(
            timestamp={'word': [{'word': 'word' + str(index), 'start': start, 'end': end}
                                for index, (start, end, _) in enumerate(spans)]},
            word_confidence=[confidence for _, _, confidence in spans])
        expected = pin.build_candidate(41, pin.extract_aligned_words(hypothesis, 41))['segments']
        outputs = iter([[hypothesis], [SimpleNamespace(timestamp={'word': []}, word_confidence=[])]])
        model = SimpleNamespace(transcribe=lambda *args, **kwargs: next(outputs))
        runtime = server.Runtime(model_loader=lambda: model)
        actual = runtime._transcribe_many([fixtures.request(80)], model=model)[0]
        self.assertEqual(len(expected), 6)
        self.assertEqual(actual, expected)

    def test_batch_forwards_plan_and_rejects_bad_plan_before_download(self):
        body, payload = fixtures.BatchCancellationTest().payload(1)
        payload['items'][0]['params']['chunk_plan'] = [[0, 1]]
        captured = []
        class Runtime:
            def transcribe_with_control(self, request, control, plan):
                captured.append(plan)
                return pin.build_candidate(1, [])
        events = list(batch.batch_events(Runtime(), payload, fetcher=lambda _, path: path.write_bytes(body)))
        self.assertEqual(captured, [[[0, 1]]])
        self.assertEqual(sum(event.get('status') == 'succeeded' for event in events), 1)
        payload['items'][0]['params']['chunk_plan'] = [[0, .5]]
        with patch.object(batch, 'validate_audio') as validate:
            events = list(batch.batch_events(Runtime(), payload, fetcher=lambda *_: self.fail('invalid plan downloaded')))
        self.assertFalse(validate.called)
        self.assertEqual(next(event for event in events if event['type'] == 'item')['reason'], 'download_rejected')


if __name__ == '__main__':
    unittest.main()
