import array
import contextlib
import copy
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from asr import server
from asr.startup import Stages


class Parameter:
    """Native storage stand-in keeps the default suite free of GPU dependencies."""
    def __init__(self, value):
        self.data = array.array("f", [value])

    def data_ptr(self):
        return self.data.buffer_info()[0]


class LaneModel:
    def __init__(self, cfg=None, trainer=None):
        self.cfg = cfg if cfg is not None else SimpleNamespace(decoding=SimpleNamespace(greedy=SimpleNamespace(use_cuda_graph_decoder=True)))
        self.decoder = SimpleNamespace(weight=Parameter(1), buffer=Parameter(2))
        self.joint = SimpleNamespace(weight=Parameter(3), buffer=Parameter(4))
        self.training = True
        self.decoding = None
        self.configure_calls = 0

    def __deepcopy__(self, memo):
        raise NotImplementedError("NeMo-like objects cannot be deep-copied")

    def to(self, device):
        self.device = device
        return self

    def state_dict(self):
        return {f"{module}.{name}": getattr(getattr(self, module), name).data
                for module in ("decoder", "joint") for name in ("weight", "buffer")}

    def load_state_dict(self, state, *, strict):
        for key, value in state.items():
            module, name = key.split(".")
            getattr(getattr(self, module), name).data[:] = value

    def eval(self):
        self.training = False
        return self

    def change_decoding_strategy(self, cfg, *, verbose):
        self.configure_calls += 1
        computer = SimpleNamespace(decoder=self.decoder, joint=self.joint, state=None, full_graph=None,
                                   allow_cuda_graphs=cfg.greedy.use_cuda_graph_decoder)
        self.decoding = SimpleNamespace(decoding=SimpleNamespace(decoding_computer=computer))

    def transcribe(self, files, **kwargs):
        return [None for _ in files]


class LaneCopyTest(unittest.TestCase):
    @contextlib.contextmanager
    def runtime(self, *, cuda=False, benchmark=None, model_factory=LaneModel, torch_module=None):
        modules = {name: ModuleType(name) for name in (
            "nemo", "nemo.collections", "nemo.collections.asr", "nemo.collections.asr.models",
            "nemo.core", "nemo.core.connectors", "nemo.core.connectors.save_restore_connector",
            "omegaconf", "torch",
        )}
        restores, streams = [], []
        current = threading.local()
        loading_stream = object()

        class Stream:
            def __init__(self):
                self.waits = []
                self.synchronizations = 0
                streams.append(self)

            def wait_stream(self, stream):
                self.waits.append(stream)

            def synchronize(self):
                self.synchronizations += 1

        @contextlib.contextmanager
        def stream_context(stream):
            old = getattr(current, "stream", None)
            current.stream = stream
            try:
                yield
            finally:
                current.stream = old

        def restore(path, **kwargs):
            restores.append((path, kwargs))
            return model_factory()

        modules["nemo.collections.asr.models"].ASRModel = SimpleNamespace(
            restore_from=restore, _set_model_restore_state=lambda **_: None)
        modules["nemo.core.connectors.save_restore_connector"].SaveRestoreConnector = type("Connector", (), {})
        modules["omegaconf"].open_dict = lambda _: contextlib.nullcontext()
        modules["torch"].device = lambda name: SimpleNamespace(type=name)
        modules["torch"].cuda = SimpleNamespace(
            is_available=lambda: cuda, Stream=Stream, current_stream=lambda: loading_stream, stream=stream_context)
        if torch_module is not None:
            modules["torch"] = torch_module
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "absent.nemo"
            extracted = archive.parent / "parakeet-v3"
            extracted.mkdir()
            stages = Stages(Path(directory) / "stages", sleep=lambda _: None, log=lambda *_, **__: None)
            with patch.dict(sys.modules, modules), patch.object(server, "MODEL_PATH", archive), patch.object(server.Runtime, "_check_dependencies"):
                runtime = server.Runtime(model_verifier=lambda _: None, stages=stages, benchmark=benchmark or (lambda _: None))
                yield SimpleNamespace(runtime=runtime, restores=restores, streams=streams, current=current,
                                      loading_stream=loading_stream, extracted=extracted, markers=stages.marker_dir)

    def test_one_restore_independent_parameters_buffers_and_decoders(self):
        with self.runtime(cuda=True) as fixture:
            runtime = fixture.runtime
            runtime.initialize_once()
            self.assertTrue(runtime.ready)
            self.assertEqual(len(fixture.restores), 1)
            path, kwargs = fixture.restores[0]
            self.assertEqual(path, str(fixture.extracted))
            self.assertEqual(kwargs["save_restore_connector"].model_extracted_dir, path)
            self.assertEqual(kwargs["map_location"].type, "cuda")
            lanes = runtime.pool.instances
            for name in ("decoder", "joint"):
                self.assertEqual(len({id(getattr(lane, name)) for lane in lanes}), 3)
                for storage in ("weight", "buffer"):
                    params = [getattr(getattr(lane, name), storage) for lane in lanes]
                    self.assertEqual(len({param.data_ptr() for param in params}), 3)
                    before = [param.data[0] for param in params]
                    params[1].data[0] = 99
                    self.assertEqual([params[0].data[0], params[2].data[0]], [before[0], before[2]])
            self.assertEqual(len({id(lane.cfg.decoding) for lane in lanes}), 3)
            self.assertEqual(len({id(lane.decoding) for lane in lanes}), 3)
            computers = [lane.decoding.decoding.decoding_computer for lane in lanes]
            self.assertEqual(len({id(computer) for computer in computers}), 3)
            for lane, computer in zip(lanes, computers):
                self.assertFalse(lane.training)
                self.assertIs(computer.decoder, lane.decoder)
                self.assertIs(computer.joint, lane.joint)
                self.assertTrue(lane.cfg.decoding.compute_timestamps)
                self.assertTrue(lane.cfg.decoding.preserve_alignments)
                self.assertEqual(lane.cfg.decoding.confidence_cfg, {"preserve_word_confidence": True})
                self.assertFalse(lane.cfg.decoding.greedy.use_cuda_graph_decoder)
                self.assertFalse(computer.allow_cuda_graphs)
            self.assertEqual([lane.configure_calls for lane in lanes], [1, 1, 1])
            self.assertEqual(len({id(runtime._streams[id(lane)]) for lane in lanes}), 3)
            self.assertTrue(all(stream.waits == [fixture.loading_stream] for stream in fixture.streams))

    def test_archive_layout_keeps_independent_per_lane_restores(self):
        with self.runtime() as fixture, patch.object(server.copy, "deepcopy", side_effect=AssertionError("archive layout must restore")):
            fixture.extracted.rmdir()
            fixture.runtime.initialize_once()
            self.assertTrue(fixture.runtime.ready)
            self.assertEqual(len(fixture.restores), 3)
            self.assertEqual(len({id(lane) for lane in fixture.runtime.pool.instances}), 3)
            for path, kwargs in fixture.restores:
                self.assertEqual(path, str(fixture.extracted.parent / "absent.nemo"))
                self.assertEqual(set(kwargs), {"map_location"})

    def test_failed_copy_retries_without_disk_restore(self):
        deepcopy = copy.deepcopy
        attempts = []
        def flaky(cfg):
            attempts.append(cfg)
            if len(attempts) == 1:
                raise RuntimeError("synthetic copy failure")
            return deepcopy(cfg)
        with self.runtime() as fixture, patch.object(server.copy, "deepcopy", side_effect=flaky):
            fixture.runtime.initialize_once()
            self.assertTrue(fixture.runtime.ready)
            self.assertEqual(len(fixture.restores), 1)
            self.assertEqual(len(attempts), 3)
            self.assertTrue(all(cfg is fixture.runtime.model.cfg for cfg in attempts))
            self.assertEqual(set(fixture.runtime.stages.completed), {
                "deps", "model_files", "lane_1", "lane_2", "lane_3", "benchmark_1", "benchmark_2", "benchmark_3", "ready"})

    def test_non_deepcopyable_model_constructor_retries_without_restore(self):
        attempts = []
        class NonCopyableModel(LaneModel):
            def __init__(self, cfg=None, trainer=None):
                if cfg is not None:
                    attempts.append(cfg)
                    if len(attempts) == 1:
                        raise NotImplementedError("synthetic constructor failure")
                super().__init__(cfg=cfg, trainer=trainer)
        with self.assertRaises(NotImplementedError):
            copy.deepcopy(NonCopyableModel())
        with self.runtime(model_factory=NonCopyableModel) as fixture:
            fixture.runtime.initialize_once()
            self.assertTrue(fixture.runtime.ready)
            self.assertEqual(len(fixture.restores), 1)
            self.assertEqual(len(attempts), 3)
            self.assertIsNot(attempts[0], attempts[1])
            self.assertIsNot(attempts[1], fixture.runtime.model.cfg)

    def test_failed_copy_configuration_retries_copy_not_restore(self):
        configure = LaneModel.change_decoding_strategy
        calls = []
        def flaky(model, cfg, *, verbose):
            calls.append(model)
            if len(calls) == 2:
                raise RuntimeError("synthetic copied decoder failure")
            configure(model, cfg, verbose=verbose)
        with self.runtime(cuda=True) as fixture, patch.object(LaneModel, "change_decoding_strategy", flaky):
            fixture.runtime.initialize_once()
            self.assertTrue(fixture.runtime.ready)
            self.assertEqual(len(fixture.restores), 1)
            self.assertEqual(len(calls), 4)
            self.assertIsNot(calls[1], calls[2])
            self.assertNotIn(id(calls[1]), fixture.runtime._streams)

    def test_exhausted_copy_never_ready_or_reloads_lane_one(self):
        with self.runtime() as fixture, patch.object(server.copy, "deepcopy", side_effect=RuntimeError("broken copy")) as copying:
            fixture.runtime.initialize_once()
            self.assertFalse(fixture.runtime.ready)
            self.assertEqual(fixture.runtime.state, "failed")
            self.assertEqual(len(fixture.restores), 1)
            self.assertEqual(copying.call_count, 3)
            self.assertTrue((fixture.markers / "lane_1.done").exists())
            self.assertFalse((fixture.markers / "lane_2.done").exists())
            self.assertFalse((fixture.markers / "ready.done").exists())

    def test_benchmarks_overlap_on_specific_lane_threads_and_streams(self):
        barrier = threading.Barrier(3)
        calls = []
        def transcribe(model, files, **kwargs):
            calls.append((model, threading.get_ident(), fixture.current.stream))
            self.assertFalse(fixture.runtime.ready)
            self.assertFalse((fixture.markers / "ready.done").exists())
            barrier.wait(timeout=5)
            return [None for _ in files]
        def benchmark(model):
            fixture.runtime._transcribe_many([SimpleNamespace(audio=b"test audio", audio_duration_seconds=1)], model=model)
        with self.runtime(cuda=True, benchmark=benchmark) as fixture, patch.object(LaneModel, "transcribe", transcribe), patch.object(server, "extract_aligned_words", return_value=[]):
            fixture.runtime.initialize_once()
            self.assertTrue(fixture.runtime.ready)
            self.assertEqual(len(calls), 3)
            self.assertEqual({id(call[0]) for call in calls}, {id(lane) for lane in fixture.runtime.pool.instances})
            self.assertEqual(len({call[1] for call in calls}), 3)
            self.assertNotIn(threading.get_ident(), {call[1] for call in calls})
            for model, _, stream in calls:
                self.assertIs(stream, fixture.runtime._streams[id(model)])
                self.assertEqual(stream.synchronizations, 1)
            self.assertTrue(all((fixture.markers / f"benchmark_{index}.done").exists() for index in (1, 2, 3)))

    def test_concurrent_stage_logs_are_serialized_without_serializing_actions(self):
        active = threading.Lock()
        overlaps, lines = [], []
        barrier = threading.Barrier(3)
        def log(message, **kwargs):
            if not active.acquire(blocking=False):
                overlaps.append(message)
                return
            try:
                time.sleep(.01)
                lines.append(message)
            finally:
                active.release()
        stages = Stages(log=log)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(stages.run, f"benchmark_{index}", lambda: barrier.wait(timeout=5)) for index in (1, 2, 3)]
            for future in futures:
                future.result()
        self.assertEqual(overlaps, [])
        self.assertEqual(len(lines), 6)
        self.assertEqual(set(stages.completed), {"benchmark_1", "benchmark_2", "benchmark_3"})

    def test_real_torch_parameter_and_buffer_storage_is_independent(self):
        try:
            import torch
        except ImportError:
            self.skipTest("real PyTorch check runs in the tensor-enabled test environment")
        class TorchModel(torch.nn.Module):
            def __init__(self, cfg=None, trainer=None):
                super().__init__()
                self.decoder = torch.nn.LSTM(2, 2)
                self.joint = torch.nn.Linear(2, 2)
                self.register_buffer("running", torch.ones(2))
                self.cfg = cfg if cfg is not None else SimpleNamespace(decoding=SimpleNamespace(greedy=SimpleNamespace(use_cuda_graph_decoder=True)))
                self.decoding = None

            change_decoding_strategy = LaneModel.change_decoding_strategy
            configure_calls = 0
            transcribe = LaneModel.transcribe
        with self.runtime(model_factory=TorchModel, torch_module=torch) as fixture:
            fixture.runtime.initialize_once()
            self.assertTrue(fixture.runtime.ready)
            lanes = fixture.runtime.pool.instances
            for values in (list(lane.parameters()) for lane in lanes):
                self.assertTrue(values)
            for tensors in zip(*(list(lane.parameters()) + list(lane.buffers()) for lane in lanes)):
                self.assertEqual(len({tensor.data_ptr() for tensor in tensors}), 3)
                before = [tensor.detach().clone() for tensor in tensors]
                self.assertTrue(torch.equal(before[0], before[1]))
                self.assertTrue(torch.equal(before[0], before[2]))
                with torch.no_grad():
                    tensors[1].add_(10)
                self.assertTrue(torch.equal(tensors[0], before[0]))
                self.assertTrue(torch.equal(tensors[2], before[2]))
                self.assertFalse(torch.equal(tensors[1], before[1]))
            self.assertEqual(len(fixture.restores), 1)


if __name__ == "__main__":
    unittest.main()
