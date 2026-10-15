import asyncio
import json
import socket
import struct
import sys
import threading
import time
import unittest
import tempfile
from pathlib import Path
from asr.inference_runtime import BatchControl, WorkCancelled
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from asr import server, batch_runtime as batch, offline_entrypoint as pin
from asr import test_oom_cancellation as fixtures
from asr import test_vast_serverless_wrapper as wrapper_tests


class StreamCancelTest(unittest.TestCase):
    def test_proxy_closes_upstream_on_write_failure_and_async_cancellation(self):
        namespace = wrapper_tests.VastServerlessWrapperTest()._load_worker('disconnect_regression')
        for error in (ConnectionResetError('PRIVATE'), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                async def chunks(_):
                    yield b'{"type":"accepted"}\n'
                upstream = SimpleNamespace(status=200, content_type='application/x-ndjson',
                                           content=SimpleNamespace(iter_chunked=chunks), close=Mock())
                class Response:
                    def __init__(self, **kwargs):
                        pass
                    async def prepare(self, request):
                        pass
                    async def write(self, chunk):
                        raise error
                    async def write_eof(self):
                        pass
                aiohttp = ModuleType('aiohttp')
                aiohttp.web = SimpleNamespace(StreamResponse=Response)
                with patch.dict(sys.modules, {'aiohttp': aiohttp}), self.assertRaises(type(error)):
                    asyncio.run(namespace['validated_response'](SimpleNamespace(), upstream))
                upstream.close.assert_called_once_with()

    def test_http_disconnect_sets_cancel_and_stops_new_items(self):
        body, payload = fixtures.BatchCancellationTest().payload(30)
        calls, controls = [], []
        class Runtime:
            ready = True
            def start_initialization(self):
                pass
            def transcribe(self, request):
                calls.append(1)
                time.sleep(.03)
                return pin.build_candidate(1, [])
        def events(runtime, payload, *, cancel_event):
            controls.append(cancel_event)
            return batch.batch_events(runtime, payload, cancel_event=cancel_event, keepalive_seconds=.01,
                                      fetcher=lambda _, path: path.write_bytes(body))
        with patch.object(server, 'batch_events', events):
            http = server.make_server(('127.0.0.1', 0), Runtime())
            thread = threading.Thread(target=http.serve_forever)
            thread.start()
            client = socket.create_connection(http.server_address, timeout=2)
            stream = client.makefile('rb')
            try:
                encoded = json.dumps(payload).encode()
                client.sendall(b'POST /transcribe-batch HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: '
                               + str(len(encoded)).encode() + b'\r\n\r\n' + encoded)
                self.assertIn(b'200', stream.readline())
                while stream.readline() != b'\r\n':
                    pass
                while json.loads(stream.readline())['type'] != 'item':
                    pass
                client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
                stream.close()
                client.close()
                self.assertTrue(controls[0].wait(2))
                before = len(calls)
                time.sleep(.12)
                self.assertEqual(len(calls), before)
                self.assertLess(len(calls), 30)
            finally:
                stream.close()
                client.close()
                http.shutdown()
                thread.join(2)
                http.server_close()

    def test_proxy_observes_disconnect_while_upstream_is_quiet(self):
        namespace = wrapper_tests.VastServerlessWrapperTest()._load_worker('idle_disconnect_regression')
        async def chunks(_):
            await asyncio.sleep(5)
            yield b'never'
        upstream = SimpleNamespace(status=200, content_type='application/x-ndjson',
                                   content=SimpleNamespace(iter_chunked=chunks), close=Mock())
        class Response:
            def __init__(self, **kwargs):
                pass
            async def prepare(self, request):
                pass
            async def write(self, chunk):
                pass
            async def write_eof(self):
                pass
        aiohttp = ModuleType('aiohttp')
        aiohttp.web = SimpleNamespace(StreamResponse=Response)
        async def exercise():
            request = SimpleNamespace(transport=SimpleNamespace(is_closing=lambda: True))
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(namespace['validated_response'](request, upstream), .2)
        with patch.dict(sys.modules, {'aiohttp': aiohttp}):
            asyncio.run(exercise())
        upstream.close.assert_called_once_with()

    def test_quiet_http_disconnect_cancels_before_download_finishes(self):
        body, payload = fixtures.BatchCancellationTest().payload(1)
        started, release = threading.Event(), threading.Event()
        cancel, calls = [], []
        class Runtime:
            ready = True
            def start_initialization(self):
                pass
            def transcribe(self, request):
                calls.append(1)
                return pin.build_candidate(1, [])
        def fetch(_, path):
            started.set()
            release.wait(2)
            path.write_bytes(body)
        def events(runtime, payload, *, cancel_event):
            cancel.append(cancel_event)
            return batch.batch_events(runtime, payload, cancel_event=cancel_event, fetcher=fetch)
        with patch.object(server, 'batch_events', events):
            http = server.make_server(('127.0.0.1', 0), Runtime())
            thread = threading.Thread(target=http.serve_forever)
            thread.start()
            client = socket.create_connection(http.server_address, timeout=2)
            stream = client.makefile('rb')
            try:
                encoded = json.dumps(payload).encode()
                client.sendall(b'POST /transcribe-batch HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: '
                               + str(len(encoded)).encode() + b'\r\n\r\n' + encoded)
                self.assertIn(b'200', stream.readline())
                while stream.readline() != b'\r\n':
                    pass
                self.assertEqual(json.loads(stream.readline())['type'], 'accepted')
                self.assertTrue(started.wait(1))
                client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
                stream.close()
                client.close()
                self.assertTrue(cancel[0].wait(.2))
                release.set()
                time.sleep(.1)
                self.assertEqual(calls, [])
            finally:
                release.set()
                stream.close()
                client.close()
                http.shutdown()
                thread.join(2)
                http.server_close()

    def test_disconnect_and_deadline_abort_blocked_download_socket(self):
        for reason in ('disconnect', 'deadline'):
            with self.subTest(reason=reason):
                _, payload = fixtures.BatchCancellationTest().payload(1)
                payload['items'][0]['audio_url'] = 'https://public.invalid/audio'
                started, release, closed, done = (threading.Event() for _ in range(4))
                cancel, errors = threading.Event(), []
                class Response:
                    status = 200
                    def getheader(self, name):
                        return None
                    def read1(self, size):
                        started.set()
                        release.wait(2)
                        return b''
                class Connection:
                    def __init__(self, *args, **kwargs):
                        self.sock = self
                    def request(self, *args, **kwargs):
                        pass
                    def getresponse(self):
                        return Response()
                    def shutdown(self, how):
                        closed.set()
                        release.set()
                    def close(self):
                        closed.set()
                        release.set()
                def consume():
                    try:
                        list(batch.batch_events(SimpleNamespace(), payload, cancel_event=cancel,
                                               deadline_seconds=.05 if reason == 'deadline' else 2,
                                               keepalive_seconds=.01))
                    except BaseException as error:
                        errors.append(type(error))
                    finally:
                        done.set()
                with patch.object(batch.http.client, 'HTTPSConnection', Connection):
                    thread = threading.Thread(target=consume)
                    thread.start()
                    try:
                        self.assertTrue(started.wait(1))
                        if reason == 'disconnect':
                            cancel.set()
                        self.assertTrue(closed.wait(.3))
                        self.assertTrue(done.wait(.3))
                        self.assertEqual(errors, [])
                    finally:
                        cancel.set()
                        release.set()
                        thread.join(2)

    def test_cancellation_interrupts_tls_handshake_with_live_descriptor(self):
        real_socket = socket.socket
        listener = real_socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        target = listener.getsockname()
        accepted, release, done, cancel = (threading.Event() for _ in range(4))
        errors = []
        class RedirectSocket(real_socket):
            def connect(self, address):
                return super().connect(target if address == ('8.8.8.8', 443) else address)
        def peer():
            connection, _ = listener.accept()
            accepted.set()
            release.wait(3)
            connection.close()
        control = BatchControl(2, cancel)
        def download(output):
            try:
                batch.fetch_audio('https://public.invalid/audio', output, deadline_seconds=2, control=control)
            except Exception as error:
                errors.append(type(error))
            finally:
                done.set()
        peer_thread = threading.Thread(target=peer)
        peer_thread.start()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(socket, 'socket', RedirectSocket), patch.object(socket, 'getaddrinfo', return_value=[
                    (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('8.8.8.8', 443))]):
                thread = threading.Thread(target=download, args=(Path(directory) / 'audio.wav',))
                thread.start()
                try:
                    self.assertTrue(accepted.wait(1))
                    time.sleep(.05)
                    cancel.set()
                    self.assertTrue(done.wait(.2))
                    self.assertEqual(errors, [WorkCancelled])
                finally:
                    cancel.set()
                    release.set()
                    thread.join(3)
                    peer_thread.join(3)
                    listener.close()


if __name__ == '__main__':
    unittest.main()
