#!/usr/bin/env python3
"""Benchmark ASR backends (whisper vs parakeet) on generated line audio.

Same input contract as detect_audio_glitches.py: per-line WAVs
(chapter_N.NNNN.wav) plus the expected text (chapter_N.txt, "Line N:"
format). For each backend it measures:

  * model load time
  * transcription wall time per line, total, and RTF vs audio duration
  * accuracy: matched_ratio vs expected text, preamble/truncation flags,
    uncovered-speech ms (speech whisper/parakeet failed to turn into words)
  * word-timing sanity: non-monotonic word starts (alignment defects)

Usage:
    python3 scripts/benchmark_asr.py --chapter 22
    python3 scripts/benchmark_asr.py --chapter 22 --backends whisper parakeet
    python3 scripts/benchmark_asr.py --output-dir voice_test/bwp_output --cpu
"""
import argparse
import glob
import os
import re
import sys
import time
from typing import List

# Reuse the glitch detector's loaders/scoring so benchmark numbers are
# directly comparable with production validation ratios.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, os.path.join(_SCRIPTS_DIR, ".."))
from detect_audio_glitches import analyze, load_expected_lines  # noqa: E402


def audio_duration_seconds(path: str) -> float:
    import soundfile as sf
    try:
        return float(sf.info(path).duration)
    except Exception:
        return 0.0


def load_backend(backend: str, device: str, cpu: bool, fast: bool,
                 warmup_wav: str = None):
    from audiobook_generator.audiobook_generator import setup_validation_model
    t0 = time.perf_counter()
    vm = setup_validation_model(device, cpu=cpu, fast=fast, backend=backend)
    load_seconds = time.perf_counter() - t0

    # Force any lazy CUDA/cuDNN init before the timed loop.
    if warmup_wav:
        from audiobook_generator.utils import transcribe_audio_with_whisper
        transcribe_audio_with_whisper(vm, warmup_wav)
    return vm, load_seconds


def run_backend(backend: str, items, exp_by_num, device: str, cpu: bool,
                fast: bool, repeat: int, verbose: bool):
    """items: list of (wav_path, line_num)."""
    from audiobook_generator.utils import transcribe_audio_with_whisper
    from audiobook_generator.pipeline import uncover_speech_stats

    vm, load_seconds = load_backend(backend, device, cpu, fast,
                                    warmup_wav=items[0][0])

    rows = []
    total_transcribe = 0.0
    total_audio = 0.0
    n_preamble = n_trunc = n_uncovered = n_nonmono = 0
    ratios = []
    for wav, line_num in items:
        expected = exp_by_num.get(line_num, "")
        dur = audio_duration_seconds(wav)
        total_audio += dur

        detected, starts, ends = "", [], []  # type: (str, List[float], List[float])
        for _ in range(max(1, repeat)):
            t0 = time.perf_counter()
            detected, starts, ends = transcribe_audio_with_whisper(vm, wav)
            total_transcribe += time.perf_counter() - t0

        info = analyze(expected, detected) if expected else {"matched_ratio": None,
                                                             "preamble_words": [],
                                                             "truncated_words": []}
        uncovered_ms, _ = uncover_speech_stats(wav, starts, ends)
        nonmono = sum(1 for a, b in zip(starts, starts[1:]) if b < a - 0.01)

        ratios.append(info["matched_ratio"])
        if info["preamble_words"]:
            n_preamble += 1
        if info["truncated_words"]:
            n_trunc += 1
        if uncovered_ms >= 80:
            n_uncovered += 1
        if nonmono:
            n_nonmono += 1
        rows.append((line_num, dur, info["matched_ratio"], uncovered_ms, nonmono))
        if verbose:
            print(f"  [{backend}] L{line_num} ({os.path.basename(wav)}): "
                  f"ratio={info['matched_ratio']} uncovered={uncovered_ms}ms words={len(starts)}")

    n = len(items)
    matched = [r for r in ratios if r is not None]
    summary = {
        "backend": backend,
        "load_seconds": load_seconds,
        "lines": n,
        "audio_seconds": total_audio,
        "transcribe_seconds": total_transcribe,
        "rtf": (total_transcribe / total_audio) if total_audio else float("inf"),
        "avg_matched_ratio": (sum(matched) / len(matched)) if matched else None,
        "preamble": n_preamble,
        "truncated": n_trunc,
        "uncovered": n_uncovered,
        "nonmonotonic": n_nonmono,
    }
    return summary


def print_summary(s: dict):
    r = s["avg_matched_ratio"]
    ratio_str = f"{r:.3f}" if r is not None else "n/a"
    print(f"\n[{s['backend']}]")
    print(f"  load:            {s['load_seconds']:.1f}s")
    print(f"  lines:           {s['lines']}")
    print(f"  audio:           {s['audio_seconds']:.1f}s")
    print(f"  transcribe:      {s['transcribe_seconds']:.1f}s")
    print(f"  RTF (t/audio):   {s['rtf']:.3f}  ({1.0 / s['rtf']:.1f}x realtime)")
    print(f"  matched_ratio:   {ratio_str} (avg vs expected text)")
    print(f"  preamble lines:  {s['preamble']}")
    print(f"  truncated lines: {s['truncated']}")
    print(f"  uncovered lines: {s['uncovered']}")
    print(f"  nonmono timing:  {s['nonmonotonic']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", type=int, default=None,
                    help="Chapter number (per-line WAV mode: chapter_N.NNNN.wav + chapter_N.txt)")
    ap.add_argument("--output-dir", default="voice_test/bwp_output")
    ap.add_argument("--sample-dir", default=None,
                    help="Voice-sample mode: transcribe every .wav in this dir against "
                         "DEFAULTS['static_voice_text'] (works without per-line WAVs)")
    ap.add_argument("--backends", nargs="+", default=["whisper", "parakeet"],
                    choices=["whisper", "parakeet"])
    ap.add_argument("--device", default=None, help="ASR device (default: cuda, or cpu with --cpu)")
    ap.add_argument("--cpu", action="store_true", help="Run ASR on CPU")
    ap.add_argument("--fast", action="store_true", help="Use fast validation model")
    ap.add_argument("--repeat", type=int, default=1, help="Transcriptions per line (timing median)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu else (args.device or "cuda")

    if args.sample_dir:
        # Generic mode: any WAVs, expected text = the static voice text.
        from audiobook_generator.config import DEFAULTS
        sample_wavs = sorted(
            os.path.join(args.sample_dir, f) for f in os.listdir(args.sample_dir)
            if f.endswith(".wav")
        )
        if not sample_wavs:
            print(f"No .wav files in {args.sample_dir}.")
            return
        static_text = DEFAULTS["static_voice_text"]
        # Sample audio contains the full static text (the throwaway-prefix
        # convention only applies to production validation scoring).
        exp_by_num = {i + 1: static_text for i in range(len(sample_wavs))}
        items = [(p, i + 1) for i, p in enumerate(sample_wavs)]
        print(f"Sample mode: {len(sample_wavs)} wavs from {args.sample_dir}, "
              f"backends={args.backends}")
    else:
        if args.chapter is None:
            ap.error("either --chapter or --sample-dir is required")
        ch = str(args.chapter).zfill(2)
        txt_path = os.path.join(args.output_dir, f"chapter_{ch}.txt")
        if not os.path.exists(txt_path):
            print(f"No expected-text file at {txt_path}. Check --chapter/--output-dir.")
            return
        expected_lines = load_expected_lines(txt_path)
        exp_by_num = {n: t for n, t in expected_lines}

        wav_paths = [
            p for p in glob.glob(os.path.join(args.output_dir, f"chapter_{ch}.*.wav"))
            if re.search(rf"chapter_{ch}\.\d+\.wav$", os.path.basename(p))
        ]
        wav_paths = sorted(wav_paths, key=lambda p: int(os.path.basename(p).split(".")[1]))
        if not wav_paths:
            print(f"No line WAVs found for chapter {ch} in {args.output_dir}.")
            return
        items = [(p, int(os.path.basename(p).split(".")[1])) for p in wav_paths]
        print(f"Chapter {ch}: {len(wav_paths)} line audio files, "
              f"{len(expected_lines)} expected lines, backends={args.backends}")

    summaries = []
    for backend in args.backends:
        print(f"\n=== running backend: {backend} ===")
        try:
            summaries.append(run_backend(backend, items, exp_by_num, device,
                                         args.cpu, args.fast, args.repeat, args.verbose))
        except Exception as e:
            print(f"[{backend}] FAILED: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    for s in summaries:
        print_summary(s)

    if len(summaries) == 2:
        a, b = summaries
        speedup = a["transcribe_seconds"] / b["transcribe_seconds"] if b["transcribe_seconds"] else float("inf")
        print(f"\nComparison: {b['backend']} is {speedup:.2f}x faster than {a['backend']} "
              f"on {b['audio_seconds']:.1f}s of audio")


if __name__ == "__main__":
    main()
