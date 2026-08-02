#!/usr/bin/env python3
"""Benchmark TTS end-clipping using the sample Pride & Prejudice EPUB + omni.

For each sampled line this generates the RAW (unclipped) audio for the full
script (line + postfix), transcribes it once, runs the production clipping path
(_validate_and_clip_audio), then transcribes the final clipped audio and reports
whether the line's ending words survived. This isolates premature end-clipping
from TTS quality issues. The ending check is punctuation-tolerant (Whisper adds
trailing quotes/dashes like 'them"', 'he--').

Note: 'premature' flags here are often Whisper re-read variance (the raw and the
clipped audio are different audio, so the re-read can differ slightly) -- verify
by checking the clip transcription.

Usage:
    python scripts/benchmark_clipping.py [--lines N] [--chapters N] [--out DIR]
        [--voice-ref PATH] [--device DEV] [--whisper-cpu]
"""
import argparse
import gc
import json
import re
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audiobook_generator.audiobook_generator import (
    TTSConfig, _validate_and_clip_audio, setup_validation_model,
)
from audiobook_generator.config import DEFAULTS
from audiobook_generator.parse_chapter import parse_epub_to_chapters
from audiobook_generator.pipeline import (
    collect_transcription_segments, prepare_script_for_tts,
)
from audiobook_generator.utils import distill_string
from tts import get_engine

SAMPLE_EPUB = REPO / "voice_test" / "test_pride_and_prejudice.epub"
VOICE_REF = REPO / "voice_test" / "test_voices" / "omni" / "narrator.wav"
POSTFIX = DEFAULTS["short_text_postfix"]

def wav_duration(path: Path) -> float:
    import pydub
    return pydub.AudioSegment.from_wav(str(path)).duration_seconds


def transcribe_full(whisper, path: Path):
    """Transcribe and return (segments, starts, ends, distilled_text)."""
    result = whisper.transcribe(str(path), beam_size=5, word_timestamps=True)
    if isinstance(result, tuple):
        segs_list = result[0]
    elif isinstance(result, dict):
        segs_list = result.get("segments", [])
    else:
        segs_list = result
    segments, starts, ends = collect_transcription_segments(segs_list)
    det = distill_string(" ".join(segments))
    return segments, starts, ends, det


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=30)
    ap.add_argument("--chapters", type=int, default=2)
    ap.add_argument("--out", default="/tmp/abg_clip_bench")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--voice-ref", default=str(VOICE_REF),
                    help="Path to the voice reference sample used for cloning")
    ap.add_argument("--whisper-cpu", action="store_true",
                    help="Run Whisper on CPU (slow; for debugging CPU/GPU split)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    voice_ref = args.voice_ref

    chapters = parse_epub_to_chapters(str(SAMPLE_EPUB), max_chapters=args.chapters)
    lines = []
    for c, ch in enumerate(chapters):
        for obj in ch:
            if obj.text.strip():
                lines.append((c, obj.line_num, obj.text))
    sample = lines[: args.lines]
    print(f"Parsed {len(chapters)} chapter(s), {len(lines)} lines; sampling {len(sample)}")
    print(f"Voice ref: {voice_ref}")

    device = args.device
    engine = get_engine("omni", device=device)
    # Whisper on the GPU by default (fast, low CPU) matching production
    # (whisper_cpu=False). Use --whisper-cpu to force CPU for debugging.
    whisper = setup_validation_model(device, cpu=args.whisper_cpu, fast=args.whisper_cpu)
    tts_config = TTSConfig(
        device=device,
        tts_engine="omni",
        output_dir=str(out),
        short_text_postfix=POSTFIX,
        validation_model=whisper,
        engine=engine,
        verbose=False,
    )

    results = []
    try:
        for i, (c, ln, text) in enumerate(sample):
            full_script, _ = prepare_script_for_tts(text, POSTFIX)
            raw = out / f"raw_c{c}_l{ln}.wav"
            clip = out / f"clip_c{c}_l{ln}.wav"

            ok = engine.generate_line(
                text=full_script, voice_path=voice_ref,
                output_path=str(raw), verbose=False,
            )
            if not ok or not raw.exists():
                results.append({"chapter": c, "line": ln, "error": "generate_line failed"})
                print(f"[FAIL] c{c} l{ln}: generation failed")
                continue

            exp = distill_string(text)
            exp_words = exp.split()
            rs, rstart, rend, raw_det = transcribe_full(whisper, raw)
            raw_words = raw_det.split()

            shutil.copy(raw, clip)
            ratio, last_valid = _validate_and_clip_audio(full_script, str(clip), tts_config)
            _, _, _, clip_det = transcribe_full(whisper, clip)
            clip_words = clip_det.split()

            def tok(w):
                return re.sub(r"[^a-z0-9]", "", w.lower())

            def ending(words, n):
                if not words or len(exp_words) < n:
                    return None
                norm = [tok(w) for w in words]
                return all(tok(w) in norm for w in exp_words[-n:])

            raw_end_ok = ending(raw_words, 1)
            clip_end_ok = ending(clip_words, 1)
            raw_end2 = ending(raw_words, 2)
            clip_end2 = ending(clip_words, 2)

            raw_dur = wav_duration(raw)
            clip_dur = wav_duration(clip)

            premature = bool(raw_end_ok and not clip_end_ok)
            results.append({
                "chapter": c, "line": ln, "text": text,
                "ratio": round(ratio, 3), "last_valid": last_valid,
                "raw_det": raw_det, "clip_det": clip_det,
                "raw_end_ok": raw_end_ok, "clip_end_ok": clip_end_ok,
                "raw_end2": raw_end2, "clip_end2": clip_end2,
                "raw_dur": round(raw_dur, 3), "clip_dur": round(clip_dur, 3),
                "premature": premature,
            })
            status = "PREMATURE" if premature else ("OK" if clip_end_ok else "MISSING")
            print(
                f"[{status}] c{c} l{ln:3d} ratio={ratio:.2f} "
                f"rawEnd={'Y' if raw_end_ok else 'N'} clipEnd={'Y' if clip_end_ok else 'N'} "
                f"dur={raw_dur:.2f}->{clip_dur:.2f} "
                f"| {text[:45]!r}"
            )
            if i % 5 == 4:
                print(f"  ... {i+1}/{len(sample)} (GPU clear) ", end="", flush=True)
                gc.collect(); torch.cuda.empty_cache()
                print("ok")
    finally:
        engine.shutdown_worker()

    with open(out / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    n = [r for r in results if "error" not in r]
    premature = [r for r in n if r["premature"]]
    missing = [r for r in n if not r["clip_end_ok"]]
    ok_lines = [r for r in n if r["clip_end_ok"]]
    avg_ratio = sum(r["ratio"] for r in n) / len(n) if n else 0.0
    print("\n================ SUMMARY ================")
    print(f"Lines analyzed: {len(n)}  (errors: {len(results) - len(n)})")
    print(f"Avg validation ratio: {avg_ratio:.3f}")
    print(f"Ending preserved after clip: {len(ok_lines)}/{len(n)} ({100*len(ok_lines)/max(1,len(n)):.0f}%)")
    print(f"Last word MISSING after clip: {len(missing)}")
    print(f"PREMATURE clips (raw had ending, clip lost it): {len(premature)}")
    if premature:
        print("\nPrematurely-clipped lines (verify: likely Whisper re-read variance):")
        for r in premature:
            print(f"  c{r['chapter']} l{r['line']}: {r['text'][:60]!r}")
            print(f"      raw : ...{r['raw_det'][-70:]}")
            print(f"      clip: ...{r['clip_det'][-70:]}")
    print(f"\nDetailed JSON -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
