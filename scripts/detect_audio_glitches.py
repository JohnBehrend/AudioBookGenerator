#!/usr/bin/env python3
"""Detect audio glitches in generated audiobook lines.

For a given chapter, reads the per-line WAV files (chapter_N.NNNN.wav) and the
expected text (chapter_N.txt, "Line N:" format), transcribes each line with
Whisper, and flags two common problems:

  * PREAMBLE artifact  - extra speech before the expected text starts
                         (e.g. a "teh"/"the"/"um" glitch cloned into the voice).
  * TRUNCATION         - trailing words of the expected text are missing from
                         the transcription (end of speech cut off).

Usage:
    python3 scripts/detect_audio_glitches.py --chapter 22 [--output-dir ...]
                                              [--cpu] [--verbose]
"""
import argparse
import glob
import os
import re
import sys

from difflib import SequenceMatcher


def distill_string(s: str) -> str:
    """Lowercase and strip punctuation, matching the pipeline's distill_string."""
    return re.sub(r"[^\w\s]", "", s.lower()).replace("_", " ")


def load_expected_lines(chapter_txt: str):
    """Return list of (line_num, text) from a 'Line N: ...' chapter file."""
    lines = []
    with open(chapter_txt, encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            m = re.match(r"^Line\s+(\d+):\s*(.*)$", raw, re.IGNORECASE)
            if m:
                lines.append((int(m.group(1)), m.group(2)))
    return lines


def analyze(expected: str, detected: str):
    """Compare expected vs transcribed text. Returns (preamble, truncation) info."""
    exp = [w for w in distill_string(expected).split() if w]
    det = [w for w in distill_string(detected).split() if w]
    info = {"expected": expected, "detected": detected,
            "preamble_words": [], "truncated_words": [], "matched_ratio": 1.0}

    if not exp:
        return info
    if not det:
        info["matched_ratio"] = 0.0
        info["truncated_words"] = exp
        return info

    sm = SequenceMatcher(None, exp, det, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]

    if not blocks:
        info["matched_ratio"] = 0.0
        info["truncated_words"] = exp
        return info

    # First detected index covered by a match, and last expected index covered.
    first_det_idx = min(b[1] for b in blocks)
    last_exp_idx = max(b[0] + b[2] - 1 for b in blocks)

    # Preamble: detected words before the first matched word.
    info["preamble_words"] = det[:first_det_idx]

    # Truncation: expected words after the last matched expected word.
    info["truncated_words"] = exp[last_exp_idx + 1:]

    total = len(exp)
    matched = sum(b[2] for b in blocks)
    info["matched_ratio"] = matched / total if total else 1.0
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", type=int, required=True, help="Chapter number")
    ap.add_argument("--output-dir", default="voice_test/bwp_output")
    ap.add_argument("--cpu", action="store_true", help="Run Whisper on CPU")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from audiobook_generator.audiobook_generator import setup_validation_model
    from audiobook_generator.utils import transcribe_audio_with_whisper

    vm = setup_validation_model("cpu" if args.cpu else "cuda", cpu=args.cpu, fast=True)

    ch = str(args.chapter).zfill(2)
    txt_path = os.path.join(args.output_dir, f"chapter_{ch}.txt")
    expected_lines = load_expected_lines(txt_path)
    exp_by_num = {n: t for n, t in expected_lines}

    wav_paths = [
        p for p in glob.glob(os.path.join(args.output_dir, f"chapter_{ch}.*.wav"))
        if re.search(rf"chapter_{ch}\.\d+\.wav$", os.path.basename(p))
    ]
    wav_paths = sorted(
        wav_paths,
        key=lambda p: int(os.path.basename(p).split(".")[1]),
    )
    if not wav_paths:
        print(f"No line WAVs found for chapter {ch}. Check output dir.")
        return

    print(f"Chapter {ch}: {len(wav_paths)} line audio files, {len(expected_lines)} expected lines")

    n_preamble = 0
    n_trunc = 0
    n_uncovered = 0
    for wav in wav_paths:
        line_num = int(os.path.basename(wav).split(".")[1])
        expected = exp_by_num.get(line_num, "")
        if not expected:
            if args.verbose:
                print(f"  line {line_num}: no expected text")
            continue
        detected, starts, ends = transcribe_audio_with_whisper(vm, wav)
        info = analyze(expected, detected)
        flags = []
        if info["preamble_words"]:
            n_preamble += 1
            flags.append(f"PREAMBLE({len(info['preamble_words'])}):{' '.join(info['preamble_words'])[:40]}")
        if info["truncated_words"]:
            n_trunc += 1
            flags.append(f"TRUNC(miss{len(info['truncated_words'])}):{info['truncated_words'][-3:]}")
        uncovered_ms = _uncovered_speech_ms(wav, starts, ends)
        if uncovered_ms >= 80:
            n_uncovered += 1
            flags.append(f"UNCOVERED({uncovered_ms}ms)")
        if flags:
            print(f"  L{line_num} [{', '.join(flags)}] ratio={info['matched_ratio']:.2f}")
            if args.verbose:
                print(f"      expected: {expected[:90]}")
                print(f"      detected: {detected[:90]}")

    print(f"\nSummary: {n_preamble} preamble, {n_trunc} truncated, {n_uncovered} uncovered-speech lines "
          f"(of {len(wav_paths)} line files)")


def _uncovered_speech_ms(wav_path: str, starts, ends) -> int:
    """Total ms of non-silent audio with no Whisper word covering it."""
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    if not starts:
        return 0
    audio = AudioSegment.from_wav(wav_path)
    ns = detect_nonsilent(audio, min_silence_len=60, silence_thresh=-32)
    total = 0
    for s, e in ns:
        s_ms, e_ms = s / 1000.0, e / 1000.0
        covered = any(s_ms < end and e_ms > start for start, end in zip(starts, ends))
        if not covered:
            total += int((e - s))
    return total


if __name__ == "__main__":
    main()
