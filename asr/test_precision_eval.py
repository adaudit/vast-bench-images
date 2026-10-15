import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from asr.precision_eval import evaluate, compare_words


def word(text, start, end, confidence=.9):
    return dict(text=text, start_seconds=start, end_seconds=end, confidence=confidence)


class PrecisionEvalTests(unittest.TestCase):
    def test_alignment_counts_errors_and_only_compares_matching_words(self):
        result = compare_words([word('A', 0, 1), word('private', 1, 2), word('call', 2, 3)],
                               [word('a', .02, 1.03, .8), word('other', 1, 2), word('call', 2.08, 3.09)])
        self.assertAlmostEqual(result['wer'], 1 / 3)
        self.assertEqual(result['aligned_word_count'], 2)
        self.assertEqual(result['within_50ms_share'], .5)
        self.assertEqual(result['within_100ms_share'], 1)
        self.assertAlmostEqual(result['mean_timing_drift_ms'], 55)
        self.assertAlmostEqual(result['max_timing_drift_ms'], 90)
        self.assertAlmostEqual(result['mean_confidence_diff'], .05)
        self.assertNotIn('private', json.dumps(result))

    def test_mocked_model_runs_four_serial_variants_with_server_contract_and_no_text(self):
        torch = MagicMock()
        torch.cuda.max_memory_allocated.return_value = 123
        torch.cuda.max_memory_reserved.return_value = 456
        torch.cuda.memory_allocated.return_value = 100
        torch.cuda.is_available.return_value = True
        models = []
        def loader():
            model = MagicMock()
            model.float.return_value = model
            model.transcribe.return_value = [SimpleNamespace(
                timestamp={'word': [{'word': 'SECRET', 'start': 0., 'end': .5}]}, word_confidence=[.9])]
            models.append(model)
            return model
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'case_01.wav'
            with wave.open(str(path), 'wb') as audio:
                audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                audio.writeframes(b'\0\0' * 16000)
            result = evaluate(Path(directory), model_loader=loader, torch_module=torch)
        self.assertEqual(len(models), 4)
        for model in models:
            model.transcribe.assert_called_once_with([str(path)], batch_size=1, timestamps=True, num_workers=0)
        self.assertEqual(torch.autocast.call_count, 2)
        torch.autocast.assert_called_with(device_type='cuda', dtype=torch.bfloat16)
        models[0].to.assert_not_called()
        models[1].to.assert_not_called()
        models[2].to.assert_called_once_with(dtype=torch.bfloat16)
        models[2].preprocessor.float.assert_called_once_with()
        models[3].to.assert_not_called()
        self.assertIs(models[3].cfg.decoding.greedy.use_cuda_graph_decoder, True)
        models[3].change_decoding_strategy.assert_called_once_with(models[3].cfg.decoding, verbose=False)
        for model in models[:3]:
            model.change_decoding_strategy.assert_not_called()
        for variant in ('bf16', 'bf16_weights', 'fp32_graphs'):
            self.assertEqual(result['comparisons'][variant]['aggregate']['wer'], 0)
            self.assertEqual(result['comparisons'][variant]['cases']['case_01']['duration_s'], 1)
            self.assertEqual(result['precisions'][variant]['peak_gpu_allocated_bytes'], 123)
            self.assertGreaterEqual(result['precisions'][variant]['load_s'], 0)
            self.assertGreaterEqual(result['precisions'][variant]['realtime_factor'], 0)
        self.assertEqual(result['precisions']['fp32']['peak_gpu_allocated_bytes'], 123)
        self.assertGreater(result['precisions']['fp32_graphs']['transcribe_speedup_vs_fp32'], 0)
        self.assertNotIn('SECRET', json.dumps(result))
        self.assertNotIn(str(path), json.dumps(result))

    def test_empty_and_insertions(self):
        self.assertIsNone(compare_words([], [])['wer'])
        self.assertEqual(compare_words([word('a', 0, 1)], [])['wer'], 1)
        self.assertEqual(compare_words([word('a', 0, 1)], [word('b', 0, 1), word('a', 1, 2)])['wer'], 1)


if __name__ == '__main__':
    unittest.main()
