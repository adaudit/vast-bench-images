#!/usr/bin/env python3
"""Metrics-only Parakeet-v3 fp32/bf16 evaluation; no transcript serialization.

All three passes independently restore the baked extracted model through
server.Runtime's loader (identical timestamp/confidence settings, graphs off).
bf16 uses CUDA autocast with fp32 weights/frontend. bf16_weights additionally
casts model weights/buffers to bfloat16 after restore, then restores preprocessor
buffers to fp32 for stable audio feature extraction; autocast bridges fp32
features to eligible bf16 encoder/decoder operations. No fp32 weight master copy
is retained. fp32 disables TF32. One file/batch, no warmup; wall time includes
first inference. Load time includes dtype conversion; peak memory includes the
initial fp32 restore, so loaded_gpu_allocated_bytes also reports post-cast memory.

WER is exact Levenshtein over casefolded, punctuation-stripped aligned words.
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
    outputs, precisions = {}, {}
    for precision in ('fp32', 'bf16', 'bf16_weights'):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        model = model_loader()
        if precision == 'bf16_weights':
            model.to(dtype=torch.bfloat16)
            model.preprocessor.float()
        torch.cuda.synchronize()
        load_s = time.perf_counter() - started
        loaded_memory = torch.cuda.memory_allocated()
        outputs[precision] = {}
        started = time.perf_counter()
        with torch.inference_mode():
            for path in paths:
                autocast = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if precision != 'fp32' else contextlib.nullcontext()
                with autocast:
                    words = extract_aligned_words(model.transcribe([str(path)], batch_size=1, timestamps=True, num_workers=0)[0])
                outputs[precision][path.stem] = words
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        precisions[precision] = dict(load_s=load_s, transcribe_wall_s=elapsed,
                                    realtime_factor=elapsed / sum(durations.values()),
                                    loaded_gpu_allocated_bytes=loaded_memory,
                                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        del model, words
        gc.collect()
        torch.cuda.empty_cache()
    comparisons = {}
    for precision in ('bf16', 'bf16_weights'):
        cases = {case: dict(duration_s=duration, **compare_words(outputs['fp32'][case], outputs[precision][case]))
                 for case, duration in durations.items()}
        sums = {key: sum(case[key] for case in cases.values()) for key in (
            'fp32_word_count', 'bf16_word_count', 'edit_count', 'aligned_word_count',
            'within_50ms_count', 'within_100ms_count', 'timing_drift_sum_ms', 'confidence_diff_sum')}
        maxima = [case['max_timing_drift_ms'] for case in cases.values() if case['max_timing_drift_ms'] is not None]
        sums['max_timing_drift_ms'] = max(maxima) if maxima else None
        comparisons[precision] = dict(cases=cases, aggregate=dict(
            case_count=len(cases), duration_s=sum(durations.values()), **_summarize(sums)))
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
