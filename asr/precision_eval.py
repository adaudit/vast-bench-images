#!/usr/bin/env python3
"""Metrics-only Parakeet-v3 fp32/bf16 evaluation; no transcript serialization.

All four passes independently restore the baked extracted model through
server.Runtime's loader (identical timestamp/confidence settings).
bf16 uses CUDA autocast with fp32 weights/frontend. bf16_weights additionally
casts model weights/buffers to bfloat16 after restore, then restores preprocessor
buffers to fp32 for stable audio feature extraction; autocast bridges fp32
features to eligible bf16 encoder/decoder operations. No fp32 weight master copy
is retained. fp32 disables TF32. One file/batch, no warmup; wall time includes
first inference. Load time includes dtype conversion; peak memory includes the
initial fp32 restore, so loaded_gpu_allocated_bytes also reports post-cast memory.
fp32_graphs re-enables greedy CUDA graph decoding on one fp32 model, rebuilding
its decoding strategy. This control runs LAST, single-lane and serially, so
only one model ever captures graphs. Its wall time includes first capture
(no warmup), and speedup_vs_fp32 compares against the graphs-off fp32 pass.
The bf16_word_count field denotes candidate word count for every comparison,
including this fp32 graph control; fp32_word_count always denotes the reference.

Timing/confidence compare only equal-word alignment pairs; coverage is reported.
Timing thresholds require BOTH boundaries within threshold; mean drift averages
absolute start AND end differences. Aggregates are word/boundary weighted.
Empty denominators produce null, never NaN. GPU peaks include model loading and
inference (PyTorch allocated/reserved bytes, not total device usage).
"""
import argparse
import contextlib
import gc
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import unicodedata
import wave

try:
    from offline_entrypoint import MODEL_PATH, extract_aligned_words
except ModuleNotFoundError:
    from asr.offline_entrypoint import MODEL_PATH, extract_aligned_words


def _token(text):
    return ''.join(c for c in text.casefold() if not unicodedata.category(c).startswith('P'))


def _align(reference, hypothesis):
    # Exact edit distance with compact traceback (one byte/cell, linear cost rows).
    n, m = len(reference), len(hypothesis)
    steps = [bytearray(m + 1) for _ in range(n + 1)]
    previous = list(range(m + 1))
    for i, ref in enumerate(reference, 1):
        current = [i] + [0] * m
        steps[i][0] = 1
        for j, hyp in enumerate(hypothesis, 1):
            diagonal = previous[j - 1] + (ref != hyp)
            delete, insert = previous[j] + 1, current[j - 1] + 1
            best = min(diagonal, delete, insert)
            current[j] = best
            steps[i][j] = 0 if best == diagonal else (1 if best == delete else 2)
        previous = current
    edits = previous[m]
    pairs = []
    i, j = n, m
    while i or j:
        direction = steps[i][j] if i and j else (1 if i else 2)
        if direction == 0:
            if reference[i - 1] == hypothesis[j - 1]:
                pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    return edits, pairs


def _divide(numerator, denominator):
    return numerator / denominator if denominator else None


def _summarize(counts):
    return {**counts,
            'wer': _divide(counts['edit_count'], counts['fp32_word_count']),
            'alignment_coverage_share': _divide(counts['aligned_word_count'], counts['fp32_word_count']),
            'within_50ms_share': _divide(counts['within_50ms_count'], counts['aligned_word_count']),
            'within_100ms_share': _divide(counts['within_100ms_count'], counts['aligned_word_count']),
            'mean_timing_drift_ms': _divide(counts['timing_drift_sum_ms'], 2 * counts['aligned_word_count']),
            'mean_confidence_diff': _divide(counts['confidence_diff_sum'], counts['aligned_word_count'])}


def compare_words(reference, hypothesis):
    edits, pairs = _align([_token(w['text']) for w in reference], [_token(w['text']) for w in hypothesis])
    counts = dict(fp32_word_count=len(reference), bf16_word_count=len(hypothesis), edit_count=edits,
                  aligned_word_count=len(pairs), within_50ms_count=0, within_100ms_count=0,
                  timing_drift_sum_ms=0., max_timing_drift_ms=None, confidence_diff_sum=0.)
    for i, j in pairs:
        ref, hyp = reference[i], hypothesis[j]
        start = abs(ref['start_seconds'] - hyp['start_seconds']) * 1000
        end = abs(ref['end_seconds'] - hyp['end_seconds']) * 1000
        counts['within_50ms_count'] += max(start, end) <= 50 + 1e-8
        counts['within_100ms_count'] += max(start, end) <= 100 + 1e-8
        counts['timing_drift_sum_ms'] += start + end
        counts['max_timing_drift_ms'] = max(counts['max_timing_drift_ms'] or 0., start, end)
        counts['confidence_diff_sum'] += abs(ref['confidence'] - hyp['confidence'])
    return _summarize(counts)


def load_model():
    try:
        from server import Runtime
        from startup import verify_extracted_model
    except ModuleNotFoundError:
        from asr.server import Runtime
        from asr.startup import verify_extracted_model
    extracted = MODEL_PATH.parent / 'parakeet-v3'
    if not extracted.is_dir():
        raise RuntimeError('baked extracted model required')
    verify_extracted_model(extracted)
    return Runtime()._load_model().float()

def _peak(frames, begin, end):
    return max((abs(int.from_bytes(frames[i:i + 2], 'little', signed=True)) for i in range(begin, end, 2)), default=0)


def _chunk_case(path, directory):
    """Split into <=40s chunks: latest silence in [start+20,start+40] else hard cut; 1s overlap."""
    with wave.open(str(path), 'rb') as audio:
        rate, channels, width = audio.getframerate(), audio.getnchannels(), audio.getsampwidth()
        frames = audio.readframes(audio.getnframes())
    bytes_per_second = 2 * channels * rate
    duration = len(frames) / bytes_per_second
    boundaries, start = [0.0], 0.0
    while start + 40 < duration:
        window_start = int((start + 20) * bytes_per_second)
        window_end = int((start + 40) * bytes_per_second)
        step = int(0.2 * bytes_per_second)
        cut = start + 40
        for offset in range(window_start, window_end - step, step):
            if _peak(frames, offset, offset + step) <= 250:
                cut = offset / bytes_per_second
        boundaries.append(cut)
        start = max(cut - 1.0, start + 1.0)
    boundaries.append(duration)
    chunks = []
    for index in range(len(boundaries) - 1):
        begin, end = boundaries[index], boundaries[index + 1]
        if end - begin <= 0.2:
            continue
        out = directory / f'{path.stem}_chunk{index:03d}.wav'
        with wave.open(str(out), 'wb') as audio:
            audio.setparams((channels, width, rate, 0, 'NONE', 'not compressed'))
            audio.writeframes(frames[int(begin * bytes_per_second):int(end * bytes_per_second)])
        chunks.append((begin, out))
    return chunks


def _transcribe_case(model, path, mode, scratch):
    if mode != 'chunked':
        return extract_aligned_words(model.transcribe([str(path)], batch_size=1, timestamps=True, num_workers=0)[0])
    words = []
    for offset, chunk in _chunk_case(path, scratch):
        for item in extract_aligned_words(model.transcribe([str(chunk)], batch_size=1, timestamps=True, num_workers=0)[0]):
            words.append({**item, 'start_seconds': item['start_seconds'] + offset, 'end_seconds': item['end_seconds'] + offset})
        chunk.unlink()
    deduplicated = []
    for item in words:
        if any(previous['text'].casefold() == item['text'].casefold() and abs(previous['start_seconds'] - item['start_seconds']) < 1.0 for previous in deduplicated[-10:]):
            continue
        deduplicated.append(item)
    return deduplicated

def evaluate(audio_dir, *, model_loader=load_model, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    paths = sorted(Path(audio_dir).glob('*.wav'))
    if not paths or any(not re.fullmatch(r'case_\d{2}\.wav', p.name) or p.is_symlink() for p in paths):
        raise ValueError('synthetic case WAVs required')
    durations = {}
    for path in paths:
        with wave.open(str(path), 'rb') as audio:
            if audio.getframerate() != 16000 or audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise ValueError('canonical 16kHz mono PCM16 WAV required')
            durations[path.stem] = audio.getnframes() / audio.getframerate()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    long_cases = {case for case, duration in durations.items() if duration >= 900}
    scratch = Path(tempfile.mkdtemp(prefix='bf16eval-chunks-'))
    outputs, precisions = {}, {}
    for precision in ('fp32', 'bf16', 'bf16_weights', 'fp32_graphs', 'fp32_chunked', 'fp32_local_attn'):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        model = model_loader()
        if precision == 'bf16_weights':
            model.to(dtype=torch.bfloat16)
            model.preprocessor.float()
        elif precision == 'fp32_graphs':
            model.cfg.decoding.greedy.use_cuda_graph_decoder = True
            model.change_decoding_strategy(model.cfg.decoding, verbose=False)
        elif precision == 'fp32_local_attn':
            model.change_attention_model('rel_pos_local_attn', [128, 128])
        torch.cuda.synchronize()
        load_s = time.perf_counter() - started
        loaded_memory = torch.cuda.memory_allocated()
        outputs[precision] = {}
        selected = paths if not precision.endswith('chunked') and precision != 'fp32_local_attn' else [p for p in paths if p.stem in long_cases]
        mode = 'chunked' if precision.endswith('chunked') else 'whole'
        whole_started = time.perf_counter()
        with torch.inference_mode():
            for path in selected:
                autocast = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if precision in ('bf16', 'bf16_weights') else contextlib.nullcontext()
                with autocast:
                    outputs[precision][path.stem] = _transcribe_case(model, path, mode, scratch)
            whole_elapsed = time.perf_counter() - whole_started
            chunk_elapsed = None
            if precision == 'fp32_local_attn':
                chunk_started = time.perf_counter()
                outputs['fp32_local_attn_chunked'] = {}
                for path in selected:
                    outputs['fp32_local_attn_chunked'][path.stem] = _transcribe_case(model, path, 'chunked', scratch)
                chunk_elapsed = time.perf_counter() - chunk_started
        torch.cuda.synchronize()
        audio_s = sum(durations[p.stem] for p in selected)
        precisions[precision] = dict(load_s=load_s, transcribe_wall_s=whole_elapsed, mode=mode,
                                    case_count=len(selected), audio_duration_s=audio_s,
                                    realtime_factor=_divide(whole_elapsed, audio_s),
                                    loaded_gpu_allocated_bytes=loaded_memory,
                                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        if chunk_elapsed is not None:
            precisions['fp32_local_attn_chunked'] = {**precisions[precision], 'mode': 'chunked',
                                                    'transcribe_wall_s': chunk_elapsed,
                                                    'realtime_factor': _divide(chunk_elapsed, audio_s)}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    shutil.rmtree(scratch, ignore_errors=True)
    comparisons = {}
    for precision, reference in (('bf16', 'fp32'), ('bf16_weights', 'fp32'), ('fp32_graphs', 'fp32'),
                                 ('fp32_local_attn', 'fp32'), ('fp32_local_attn_chunked', 'fp32_chunked')):
        cases = {case: dict(duration_s=duration, **compare_words(outputs[reference][case], outputs[precision][case]))
                 for case, duration in durations.items() if case in outputs[precision]}
        sums = {key: sum(case[key] for case in cases.values()) for key in (
            'fp32_word_count', 'bf16_word_count', 'edit_count', 'aligned_word_count',
            'within_50ms_count', 'within_100ms_count', 'timing_drift_sum_ms', 'confidence_diff_sum')}
        maxima = [case['max_timing_drift_ms'] for case in cases.values() if case['max_timing_drift_ms'] is not None]
        sums['max_timing_drift_ms'] = max(maxima) if maxima else None
        comparisons[precision] = dict(cases=cases, aggregate=dict(
            case_count=len(cases), duration_s=sum(durations[c] for c in outputs[precision]), **_summarize(sums)))
    precisions['fp32_graphs']['transcribe_speedup_vs_fp32'] = _divide(
        precisions['fp32']['transcribe_wall_s'], precisions['fp32_graphs']['transcribe_wall_s'])
    return dict(comparisons=comparisons, precisions=precisions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    # Redirect OS descriptors too: NeMo/tqdm/native code must not leak evidence.
    logging.disable(logging.CRITICAL)
    with open(os.devnull, 'w') as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            result = evaluate(args.audio_dir)
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as output:
                json.dump(result, output, allow_nan=False, separators=(',', ':'))
                output.write('\n')
        except Exception:
            raise SystemExit(1) from None


if __name__ == '__main__':
    main()
