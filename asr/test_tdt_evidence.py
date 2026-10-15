import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from asr import offline_entrypoint as pin, server


class TDTEvidenceTest(unittest.TestCase):
    def test_numeric_replay_zero_spans_build_strict_candidates(self):
        fixtures = json.loads((Path(__file__).parent / 'fixtures' / 'tdt-alignment-numeric.json').read_text())
        self.assertEqual(sum(row[1] for row in fixtures), 8)
        for duration, zero_count, spans in fixtures:
            with self.subTest(duration=duration):
                self.assertEqual(sum(start == end for start, end, _ in spans), zero_count)
                hypothesis = SimpleNamespace(timestamp={'word': [
                    {'word': 'evidence', 'start': start, 'end': end} for start, end, _ in spans
                ]}, word_confidence=[confidence for _, _, confidence in spans])
                segments = pin.extract_aligned_words(hypothesis, duration)
                candidate = pin.build_candidate(duration, segments)
                self.assertEqual(candidate['disposition'], 'speech')
                self.assertEqual(len(segments), len(spans))
                for segment, (start, end, confidence) in zip(segments, spans, strict=True):
                    self.assertEqual(segment['start_seconds'], start)
                    self.assertEqual(segment['confidence'], confidence)
                    self.assertEqual(segment['end_seconds'], end if end > start else min(start + .08, duration))

    def test_zero_spans_are_bounded_without_relaxing_invalid_evidence(self):
        def hypothesis(start, end, next_end=None):
            words = [{'word': 'evidence', 'start': start, 'end': end}]
            if next_end is not None:
                words.append({'word': 'next', 'start': start, 'end': next_end})
            return SimpleNamespace(timestamp={'word': words}, word_confidence=[.5] * len(words))
        self.assertEqual(pin.extract_aligned_words(hypothesis(.98, .98), 1)[0]['end_seconds'], 1)
        self.assertEqual(pin.extract_aligned_words(hypothesis(.2, .2, .23), 1)[0]['end_seconds'], .23)
        for start, end in ((1, 1), (2, 2), (.4, .3), (-1, -1), (float('nan'), .2)):
            with self.subTest(start=start, end=end), self.assertRaises(pin.ContractError):
                pin.extract_aligned_words(hypothesis(start, end), 1)
        with self.assertRaises(pin.ContractError):
            pin.extract_aligned_words(hypothesis(.2, .2))
        with self.assertRaises(pin.ContractError):
            pin.build_candidate(1, [{'start_seconds': .2, 'end_seconds': .2, 'text': 'evidence', 'confidence': .5}])

    def test_confidence_grouping_matches_timestamp_punctuation_and_survives_rebuild(self):
        tokens = ['▁evidence', '▁,', '▁word']
        groups = []
        class Decoder:
            supported_punctuation = {','}
            def decode_ids_to_tokens(self, ids):
                return [tokens[index] for index in ids]
            def decode_tokens_to_str(self, ids):
                return ''.join(tokens[index] for index in ids).replace('▁', ' ').strip()
            def _aggregate_confidence(self, values):
                groups.append(list(values))
                return min(values)
            def _aggregate_token_confidence_subwords_sentencepiece(self, *args):
                raise RuntimeError('recognized text must never escape')
        pin.configure_nemo_confidence(SimpleNamespace(decoding=Decoder()))
        for decoder in (Decoder(), Decoder()):
            self.assertEqual(decoder._aggregate_token_confidence_subwords_sentencepiece(
                ['evidence,', 'word'], [.9, .8, .7], [0, 1, 2]), [.8, .7])
        self.assertEqual(groups, [[.9, .8], [.7], [.9, .8], [.7]])
        for words, confidence, ids in ((['private phrase'], [.9, .8], [0]),
                                        (['private phrase'] * 3, [.9, .8, .7], [0, 1, 2])):
            with self.assertRaises(pin.ContractError) as raised:
                Decoder()._aggregate_token_confidence_subwords_sentencepiece(words, confidence, ids)
            self.assertNotIn('private', str(raised.exception))

    def test_runtime_passes_canonical_duration_without_mutating_model_output(self):
        hypothesis = SimpleNamespace(timestamp={'word': [
            {'word': 'evidence', 'start': .98, 'end': .98}
        ]}, word_confidence=[.6])
        model = SimpleNamespace(transcribe=lambda *_, **__: [hypothesis])
        runtime = server.Runtime(model_loader=lambda: model)
        request = SimpleNamespace(audio=b'synthetic', audio_duration_seconds=1)
        result = runtime._transcribe_many([request], model=model)
        self.assertEqual(result[0][0]['end_seconds'], 1)
        self.assertEqual(hypothesis.timestamp['word'][0]['end'], .98)
        model.transcribe = lambda *_, **__: []
        with self.assertRaises(ValueError):
            runtime._transcribe_many([request], model=model)

    def test_empty_and_mismatched_confidence_evidence_remain_strict(self):
        self.assertEqual(pin.extract_aligned_words(SimpleNamespace(timestamp={'word': []}, word_confidence=[]), 1), [])
        for confidences in (None, [], [.3, .4]):
            with self.subTest(confidences=confidences), self.assertRaises(pin.ConfidenceMismatchError):
                pin.extract_aligned_words(SimpleNamespace(timestamp={'word': [
                    {'word': 'evidence', 'start': 0, 'end': .1}
                ]}, word_confidence=confidences), 1)


if __name__ == '__main__':
    unittest.main()
