import contextlib
import hashlib
import io
import json
import unittest
import wave
from types import SimpleNamespace

from asr import batch_runtime as batch, offline_entrypoint as pin


class BatchFailureReasonsTest(unittest.TestCase):
    def audio_payload(self):
        output = io.BytesIO()
        with wave.open(output, 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(bytes(32000))
        body = output.getvalue()
        receipt = {'sha256': hashlib.sha256(body).hexdigest(), 'size_bytes': len(body), 'media_type': 'audio/wav',
                   'canonical_sample_rate': 16000, 'canonical_channel_count': 1, 'canonical_frame_count': 16000}
        params = {'request_version': pin.REQUEST_VERSION, 'model_id': pin.MODEL_ID,
                  'model_revision': pin.MODEL_REVISION, 'audio_duration_seconds': 1, 'audio_receipt': receipt}
        return body, {'batch_id': 'test', 'items': [{'id': 'a' * 64, 'lane': 'parakeet_v3',
                                                   'audio_url': 'https://public.invalid/private-capability', 'params': params}]}

    def test_failure_reason_and_stage_are_content_free_and_logged_once(self):
        oom = type('OutOfMemoryError', (RuntimeError,), {'__module__': 'torch'})
        cases = [
            ('download', ValueError('PRIVATE recognized words'), 'download_rejected', 'download'),
            ('download', TimeoutError('PRIVATE signed URL'), 'download_deadline', 'download'),
            ('download', OSError('PRIVATE signed URL'), 'download_error', 'download'),
            ('validate', None, 'receipt_mismatch', 'validate'),
            ('transcribe', RuntimeError('PRIVATE recognized words'), 'transcribe_error', 'transcribe'),
            ('transcribe', pin.AlignmentError('PRIVATE recognized words'), 'alignment_invalid', 'alignment'),
            ('transcribe', pin.ConfidenceMismatchError('PRIVATE recognized words'), 'confidence_mismatch', 'confidence'),
            ('transcribe', oom('PRIVATE recognized words'), 'oom', 'transcribe'),
            ('transcribe', MemoryError('PRIVATE recognized words'), 'oom', 'transcribe'),
            ('transcribe', Exception('PRIVATE recognized words'), 'unknown', 'transcribe'),
        ]
        for location, error, reason, stage in cases:
            with self.subTest(reason=reason):
                body, payload = self.audio_payload()
                def fetch(url, path):
                    if location == 'download':
                        raise error
                    path.write_bytes(b'corrupt' if location == 'validate' else body)
                def transcribe(request):
                    raise error
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    events = list(batch.batch_events(SimpleNamespace(transcribe=transcribe), payload, fetcher=fetch))
                item = next(event for event in events if event['type'] == 'item')
                self.assertEqual(item['status'], 'failed')
                self.assertEqual((item['stage'], item['reason']), (stage, reason))
                self.assertEqual(output.getvalue().splitlines(), [f'ITEM_FAILED stage={stage} reason={reason}'])
                self.assertNotIn('PRIVATE', json.dumps(events) + output.getvalue())
                self.assertNotIn('private-capability', json.dumps(events) + output.getvalue())
                self.assertNotIn('a' * 64, output.getvalue())

    def test_success_does_not_emit_failure_diagnostics(self):
        body, payload = self.audio_payload()
        runtime = SimpleNamespace(transcribe=lambda _: pin.build_candidate(1, []))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            events = list(batch.batch_events(runtime, payload, fetcher=lambda _, path: path.write_bytes(body)))
        item = next(event for event in events if event['type'] == 'item')
        self.assertEqual(item['status'], 'succeeded')
        self.assertNotIn('reason', item)
        self.assertNotIn('stage', item)
        self.assertEqual(output.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
