#!/usr/bin/env python3
"""
Benchmark script to profile all combinations of voice and TTS engines.

Uses the exact same interface as the Gradio UI:
- gen_voice_samples() for voice generation (voice_engine only)
- generate_audiobook_from_chapters() for TTS generation (tts_engine)

Tests each combination on the first chapter of Pride & Prejudice and saves
results to a CSV file with accuracy metrics and timing.

Usage:
    .venv/bin/python benchmark_engines.py [--gpus cuda:0] [--concurrency 1]
"""

import os
import re
import sys
import csv
import json
import time
import shutil
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audiobook_generator.config import DEFAULTS, AUDIO_SETTINGS
from audiobook_generator import parse_chapter
from audiobook_generator.llm_label_speakers import label_speakers
from audiobook_generator.llm_describe_character import describe_characters
from audiobook_generator.generate_voice_samples import generate_voice_samples as gen_voice_samples, _validate_with_chunkformer
from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters
from audiobook_generator.utils import parse_map_file
from audiobook_generator.pipeline import MIN_RATIO_THRESHOLD

# Voice engines that can generate voice samples
VOICE_ENGINES = ["omni", "vox", "dramabox"]

# TTS engines that can generate line audio
TTS_ENGINES = ["moss", "omni", "vox", "vibevoice", "dramabox"]

# Test EPUB
TEST_EPUB = Path(__file__).parent / "voice_test" / "test_pride_and_prejudice.epub"


def _analyze_audio_quality(voice_path: str) -> Dict:
    """Analyze audio quality metrics for a voice sample.

    Returns dict with duration, snr, clipping, silence_ratio.
    """
    import numpy as np
    import soundfile as sf

    result = {
        "duration": 0.0,
        "snr_db": 0.0,
        "clipping_pct": 0.0,
        "silence_ratio": 0.0,
        "peak_db": 0.0,
        "rms_db": 0.0,
    }

    try:
        audio, sr = sf.read(voice_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        result["duration"] = len(audio) / sr

        # Peak and RMS levels
        peak = np.max(np.abs(audio))
        result["peak_db"] = -np.inf if peak == 0 else 20 * np.log10(peak)
        rms = np.sqrt(np.mean(audio ** 2))
        result["rms_db"] = -np.inf if rms == 0 else 20 * np.log10(rms)

        # Clipping: percentage of samples within 1% of max value
        threshold = 0.99
        clipped = np.sum((audio > threshold) | (audio < -threshold))
        result["clipping_pct"] = (clipped / len(audio)) * 100 if len(audio) > 0 else 0

        # Silence ratio: frames below -40dB threshold
        frame_size = int(0.025 * sr)  # 25ms frames
        if frame_size > 0 and len(audio) >= frame_size:
            frames = [audio[i:i + frame_size] for i in range(0, len(audio) - frame_size + 1, frame_size // 2)]
            silence_threshold = 10 ** (-40 / 20)
            silent_frames = sum(1 for f in frames if np.sqrt(np.mean(f ** 2)) < silence_threshold)
            result["silence_ratio"] = silent_frames / len(frames) if frames else 0

        # SNR: ratio of signal power (above -40dB) to noise power (below -40dB)
        signal = audio[np.abs(audio) > silence_threshold]
        noise = audio[np.abs(audio) <= silence_threshold]
        if len(signal) > 0 and len(noise) > 0:
            signal_power = np.mean(signal ** 2)
            noise_power = np.mean(noise ** 2)
            if noise_power > 0:
                result["snr_db"] = 10 * np.log10(signal_power / noise_power)
            else:
                result["snr_db"] = float('inf')

    except Exception:
        pass

    return result


def _free_gpu_memory():
    """Free GPU memory."""
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    import gc
    gc.collect()


def run_single_combination(
    voice_engine: str,
    tts_engine: str,
    chapters: List,
    chapter_maps: Dict,
    character_descriptions: Dict,
    output_base_dir: str,
    device: str,
    gpus: Optional[List[str]],
    concurrency: int,
    whisper_cpu: bool,
    voice_only: bool = False,
    verbose: bool = False,
) -> Dict:
    """Run a single voice_engine + tts_engine combination and return metrics.

    Uses the exact same interface as the Gradio UI pipeline:
    1. gen_voice_samples() with voice_engine
    2. generate_audiobook_from_chapters() with tts_engine
    """
    combo_dir = os.path.join(output_base_dir, f"{voice_engine}_v_{tts_engine}_t")
    os.makedirs(combo_dir, exist_ok=True)

    result = {
        "voice_engine": voice_engine,
        "tts_engine": tts_engine,
        "status": "failed",
        "total_lines": 0,
        "successful_lines": 0,
        "failed_lines": 0,
        "avg_ratio": 0.0,
        "min_ratio": 1.0,
        "max_ratio": 0.0,
        "voice_gen_time": 0.0,
        "tts_gen_time": 0.0,
        "total_time": 0.0,
        "errors": [],
    }

    try:
        # Step 1: Generate voice samples using voice_engine (same as UI Stage 4)
        if verbose:
            print(f"\n  [Voice Samples] voice_engine={voice_engine}, device={device}")

        t0 = time.time()
        status_msg, generated_voices = gen_voice_samples(
            descriptions=character_descriptions,
            output_dir=combo_dir,
            device=device,
            voice_engine=voice_engine,
            verbose=verbose,
            use_chunkformer=False,
            seed_characters=None,
        )
        result["voice_gen_time"] = time.time() - t0

        if verbose:
            print(f"  [Voice Samples] {status_msg}")

        if not generated_voices:
            result["errors"].append("No voice samples generated")
            return result

        # Build voices_map from generated files
        voices_map = {}
        for f in Path(combo_dir).glob("*.wav"):
            if not f.name.endswith(".tmp.wav"):
                voices_map[f.stem] = str(f)

        if not voices_map:
            result["errors"].append("No voice samples generated")
            return result

        # Step 2: Generate TTS audio using tts_engine (same as UI Stage 5)
        if voice_only:
            # Validate voices with ChunkFormer and retry until all pass
            max_retries = 5
            cf_model = None
            char_attempts = {}  # char_name -> attempts to pass
            char_details = {}   # char_name -> {gender, age, dialect, passed}

            try:
                from audiobook_generator.utils import get_chunkformer_model
                cf_model = get_chunkformer_model()
            except Exception as e:
                result["errors"].append(f"ChunkFormer failed to load: {e}")
                if verbose:
                    print(f"  [CHUNKFORMER] Failed to load model: {e}")
                return result

            # Track which characters still need validation
            pending = {k for k in generated_voices if k != "narrator"}
            for char_name in pending:
                char_attempts[char_name] = 0

            for attempt in range(1, max_retries + 1):
                if attempt > 1 and pending:
                    if verbose:
                        print(f"\n  [CHUNKFORMER] Retry attempt {attempt}/{max_retries} for {len(pending)} failed voices")
                    # Remove failed voice files so they regenerate
                    for char_name in list(pending):
                        failed_path = os.path.join(combo_dir, f"{char_name}.wav")
                        if os.path.exists(failed_path):
                            os.remove(failed_path)

                    # Regenerate only failed voices
                    failed_descriptions = {k: v for k, v in character_descriptions.items() if k in pending}
                    if failed_descriptions:
                        if verbose:
                            print(f"  [CHUNKFORMER] Regenerating {len(failed_descriptions)} failed voices")
                        status_msg, regenerated = gen_voice_samples(
                            descriptions=failed_descriptions,
                            output_dir=combo_dir,
                            device=device,
                            voice_engine=voice_engine,
                            verbose=verbose,
                            use_chunkformer=False,
                            seed_characters=None,
                            force_regenerate=True,
                        )
                        result["voice_gen_time"] += time.time() - t0
                        t0 = time.time()
                        generated_voices.update(regenerated)

                # Validate pending voices
                still_pending = set()
                if cf_model:
                    for char_name in list(pending):
                        voice_path = generated_voices.get(char_name)
                        if not voice_path or not os.path.exists(voice_path):
                            still_pending.add(char_name)
                            continue
                        char_attempts[char_name] = attempt
                        desc = character_descriptions.get(char_name, "")
                        passed, log_json = _validate_with_chunkformer(
                            voice_path, desc, cf_model, verbose=verbose,
                            check_fields=["gender", "age", "dialect"]
                        )
                        try:
                            log_data = json.loads(log_json)
                            char_details[char_name] = {
                                "gender": log_data["classification"]["gender"],
                                "age": log_data["classification"]["age"],
                                "dialect": log_data["classification"]["dialect"],
                                "passed": passed,
                                "attempts": attempt,
                            }
                        except (json.JSONDecodeError, KeyError):
                            char_details[char_name] = {"passed": passed, "attempts": attempt}

                        if not passed:
                            still_pending.add(char_name)

                pending = still_pending
                if not pending:
                    break

            # Build voices_map from generated files
            voices_map = {}
            for f in Path(combo_dir).glob("*.wav"):
                if not f.name.endswith(".tmp.wav"):
                    voices_map[f.stem] = str(f)

            if not voices_map:
                result["errors"].append("No voice samples generated")
                return result

            # Calculate validation stats
            total_chars = len(char_details)
            passed_chars = sum(1 for d in char_details.values() if d.get("passed"))
            failed_chars = total_chars - passed_chars
            avg_attempts = sum(d.get("attempts", 0) for d in char_details.values()) / total_chars if total_chars > 0 else 0

            # Analyze audio quality for each voice
            durations = []
            for char_name, voice_path in generated_voices.items():
                quality = _analyze_audio_quality(voice_path)
                char_details[char_name]["duration_s"] = quality["duration"]
                char_details[char_name]["snr_db"] = quality["snr_db"]
                char_details[char_name]["clipping_pct"] = quality["clipping_pct"]
                char_details[char_name]["silence_ratio"] = quality["silence_ratio"]
                char_details[char_name]["peak_db"] = quality["peak_db"]
                char_details[char_name]["rms_db"] = quality["rms_db"]
                if char_name != "narrator":
                    durations.append(quality["duration"])

            # Duration consistency: std/mean of durations (lower is better)
            duration_cv = 0.0
            if len(durations) > 1:
                import statistics
                duration_cv = statistics.stdev(durations) / statistics.mean(durations) if statistics.mean(durations) > 0 else 0

            # Retry rate: fraction of characters that needed >1 attempt
            retry_rate = sum(1 for d in char_details.values() if d.get("attempts", 0) > 1) / total_chars if total_chars > 0 else 0

            result["status"] = "voice_only"
            result["total_lines"] = len(generated_voices)
            result["successful_lines"] = passed_chars
            result["failed_lines"] = failed_chars
            result["avg_ratio"] = passed_chars / total_chars if total_chars > 0 else 1.0
            result["min_ratio"] = 0.0 if failed_chars > 0 else 1.0
            result["max_ratio"] = 1.0
            # Store extra metrics in errors field for CSV persistence
            result["errors"] = [json.dumps({
                "avg_attempts": avg_attempts,
                "duration_cv": duration_cv,
                "retry_rate": retry_rate,
            })]

            # Print per-character breakdown
            if verbose and char_details:
                print(f"\n  {'='*70}")
                print(f"  Voice Quality Results ({voice_engine})")
                print(f"  {'='*70}")
                print(f"  {'Character':<18} {'Gender':<8} {'Age':<8} {'Dialect':<10} {'Pass':<5} {'Tries':<6} {'Dur':<6} {'SNR':<7} {'Clip%':<6} {'Sil%':<6}")
                print(f"  {'-'*86}")
                for char_name, details in sorted(char_details.items()):
                    gender = details.get("gender", "?")
                    age = details.get("age", "?")
                    dialect = details.get("dialect", "?")
                    passed = "✓" if details.get("passed") else "✗"
                    tries = details.get("attempts", 0)
                    dur = f"{details.get('duration_s', 0):.1f}s"
                    snr = f"{details.get('snr_db', 0):.0f}dB" if details.get('snr_db', 0) > 0 else "?"
                    clip = f"{details.get('clipping_pct', 0):.1f}%"
                    silence = f"{details.get('silence_ratio', 0)*100:.0f}%"
                    print(f"  {char_name:<18} {gender:<8} {age:<8} {dialect:<10} {passed:<5} {tries:<6} {dur:<6} {snr:<7} {clip:<6} {silence:<6}")
                print(f"  {'-'*86}")
                print(f"  Score: {passed_chars}/{total_chars} passed, avg {avg_attempts:.1f} tries, duration CV={duration_cv:.2f}, retry rate={retry_rate:.0%}")
                print(f"  {'='*70}\n")

            return result

        if verbose:
            print(f"  [TTS] tts_engine={tts_engine}, concurrency={concurrency}")

        t1 = time.time()

        # Capture [LINE_PROGRESS] output to extract per-line ratios
        import io
        import contextlib

        stdout_capture = io.StringIO()
        capture_ctx = contextlib.redirect_stdout(stdout_capture) if not verbose else contextlib.nullcontext()

        with capture_ctx:
            status_msg, chapters_processed = generate_audiobook_from_chapters(
                chapters=chapters,
                chapter_maps=chapter_maps,
                voices_map=voices_map,
                output_dir=combo_dir,
                device=device,
                tts_engine=tts_engine,
                max_chapters=1,
                verbose=True,
                whisper_cpu=whisper_cpu,
                concurrency=concurrency,
                gpus=gpus,
                whisper_concurrency=1,
                whisper_fast=True,
            )

        result["tts_gen_time"] = time.time() - t1

        if verbose:
            print(f"  [TTS] {status_msg}")

        # Parse per-line ratios from [LINE_PROGRESS] output
        captured = stdout_capture.getvalue()
        ratio_pattern = re.compile(r'\[LINE_PROGRESS\].*Ratio:\s*(\d+)')
        ratios = []
        for match in ratio_pattern.finditer(captured):
            ratio_pct = int(match.group(1))
            ratios.append(ratio_pct / 100.0)

        # Count total lines from chapter
        result["total_lines"] = len(chapters[0]) if chapters else 0

        if ratios:
            result["successful_lines"] = sum(1 for r in ratios if r >= MIN_RATIO_THRESHOLD)
            result["failed_lines"] = sum(1 for r in ratios if r < MIN_RATIO_THRESHOLD)
            result["avg_ratio"] = sum(ratios) / len(ratios)
            result["min_ratio"] = min(ratios)
            result["max_ratio"] = max(ratios)
            result["status"] = "completed"
        else:
            result["status"] = "no_lines_processed"
            if verbose:
                print(f"  [TTS] No ratios captured from output")

    except Exception as e:
        result["errors"].append(str(e))
        if verbose:
            import traceback
            traceback.print_exc()

    result["total_time"] = result["voice_gen_time"] + result["tts_gen_time"]
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark TTS engine combinations")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--gpus", nargs="+", default=["cuda:0"], help="GPU devices")
    parser.add_argument("--concurrency", type=int, default=1, help="TTS concurrency")
    parser.add_argument("--whisper-cpu", action="store_true", help="Run Whisper on CPU")
    parser.add_argument("--voice-engines", nargs="+", choices=VOICE_ENGINES, default=None,
                        help="Voice engines to test (default: all)")
    parser.add_argument("--tts-engines", nargs="+", choices=TTS_ENGINES, default=None,
                        help="TTS engines to test (default: all)")
    parser.add_argument("--voice-only", action="store_true", help="Only benchmark voice generation, skip TTS")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results, skipping done combinations")
    args = parser.parse_args()

    voice_engines = args.voice_engines or VOICE_ENGINES
    tts_engines = args.tts_engines or TTS_ENGINES

    if args.voice_only:
        tts_engines = [""]  # No TTS engines needed

    print(f"AudioBook Engine Benchmark")
    print(f"==========================")
    print(f"Test EPUB: {TEST_EPUB}")
    print(f"Voice engines: {voice_engines}")
    print(f"TTS engines: {tts_engines if not args.voice_only else '(voice only)'}")
    print(f"Combinations: {len(voice_engines) * len(tts_engines)}")
    print(f"Output: {args.output_dir}")
    print()

    # Find existing output directory for resume
    existing_output_base = None
    done_combos = set()
    existing_results = []

    if args.resume:
        for entry in sorted(Path(args.output_dir).glob("2026*"), reverse=True):
            if entry.is_dir():
                existing_output_base = str(entry)
                break
        if existing_output_base:
            print(f"[RESUME] Found existing run: {existing_output_base}")

            # Load CSV results if available
            for csv_file in sorted(Path(args.output_dir).glob("benchmark_*.csv"), reverse=True):
                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_results.append(row)
                if existing_results:
                    break

            if existing_results:
                done_combos = {(r["voice_engine"], r["tts_engine"]) for r in existing_results if r.get("status") == "completed"}
                print(f"[RESUME] Skipping {len(done_combos)} completed combinations (from CSV)")
            else:
                # Fallback: use directory contents to determine what's done
                for subdir in Path(existing_output_base).glob("*_v_*_t"):
                    if subdir.is_dir():
                        mp3_count = len(list(subdir.glob("chapter_*.mp3")))
                        name = subdir.name
                        parts = name.replace("_v_", "|").replace("_t", "").split("|")
                        if len(parts) == 2:
                            ve_name, te_name = parts
                            if mp3_count > 0:
                                done_combos.add((ve_name, te_name))
                                existing_results.append({"voice_engine": ve_name, "tts_engine": te_name, "status": "completed"})
                print(f"[RESUME] Skipping {len(done_combos)} completed combinations (from directories)")
        else:
            existing_output_base = os.path.join(args.output_dir, time.strftime("%Y%m%d_%H%M%S"))
    else:
        existing_output_base = os.path.join(args.output_dir, time.strftime("%Y%m%d_%H%M%S"))

    # Parse EPUB and label speakers (once, shared across all combinations)
    print("[1/3] Parsing EPUB...")
    chapters = parse_chapter.parse_epub_to_chapters(str(TEST_EPUB), max_chapters=1)
    if not chapters:
        print("ERROR: Failed to parse EPUB")
        sys.exit(1)

    print(f"  Parsed {len(chapters)} chapter(s), {len(chapters[0])} lines")

    # Cache directory for labeling/descriptions (persists across runs)
    cache_dir = Path(args.output_dir) / ".benchmark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Write chapter files for labeling
    parse_chapter.write_chapters_to_txt(chapters, str(cache_dir))

    # Label speakers only if map files don't exist
    map_files = sorted(cache_dir.glob("chapter_*.map.json"))
    if not map_files:
        print("[2/3] Labeling speakers...")
        for txt_file in sorted(cache_dir.glob("chapter_*.txt")):
            label_speakers(
                str(txt_file),
                api_key="lm-studio",
                port=2136,
                model="coder-model",
                num_attempts=1,
                verbose=args.verbose,
            )
    else:
        print("[2/3] Labeling speakers... SKIPPED (cached)")

    # Load chapter maps
    chapter_maps = {}
    for map_file in sorted(cache_dir.glob("chapter_*.map.json")):
        result = parse_map_file(map_file)
        if result:
            char_map, line_map = result
            chapter_idx = int(map_file.stem.replace("chapter_", "").replace(".map", ""))
            chapter_maps[chapter_idx] = (char_map, line_map)

    # Describe characters only if not cached
    desc_file = cache_dir / "characters_descriptions.json"
    if not desc_file.exists():
        print("[3/3] Describing characters...")
        desc_result = describe_characters(
            output_dir=str(cache_dir),
            chapters_dir=str(cache_dir),
            api_key="lm-studio",
            port=2136,
            model="coder-model",
            voice_engine="omni",
            verbose=args.verbose,
        )
        if isinstance(desc_result, tuple):
            char_descriptions = desc_result[1]
        else:
            char_descriptions = desc_result
    else:
        print("[3/3] Describing characters... SKIPPED (cached)")
        with open(desc_file) as f:
            char_descriptions = json.load(f)

    print(f"\n  Characters: {len(char_descriptions)}")
    print(f"\nStarting benchmark...")
    print("=" * 80)

    # Run combinations
    results = list(existing_results) if args.resume else []
    total_combos = len(voice_engines) * len(tts_engines)
    combo_num = 0

    output_base = existing_output_base
    os.makedirs(output_base, exist_ok=True)

    # CSV path — written incrementally after each combination so we can resume anytime
    csv_path = os.path.join(args.output_dir, f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    fieldnames = [
        "voice_engine", "tts_engine", "status",
        "total_lines", "successful_lines", "failed_lines",
        "avg_ratio", "min_ratio", "max_ratio",
        "voice_gen_time", "tts_gen_time", "total_time",
        "errors",
    ]

    # Write header once
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for ve in voice_engines:
        print(f"\n{'='*60}")
        print(f"Voice engine: {ve}")
        print(f"{'='*60}")

        prev_te = None
        for te in tts_engines:
            combo_num += 1

            # Skip completed combinations on resume
            if args.resume and (ve, te) in done_combos:
                print(f"\n[{combo_num}/{total_combos}] voice={ve}, tts={te} — SKIPPED (already completed)")
                prev_te = te
                continue

            print(f"\n[{combo_num}/{total_combos}] voice={ve}, tts={te}")
            print("-" * 60)

            # Cleanup when TTS engine changes to free GPU memory
            if prev_te is not None:
                print(f"  [MEMORY] Cleaning up between engine switches")
                _free_gpu_memory()

            result = run_single_combination(
                voice_engine=ve,
                tts_engine=te,
                chapters=chapters,
                chapter_maps=chapter_maps,
                character_descriptions=char_descriptions,
                output_base_dir=output_base,
                device=args.gpus[0],
                gpus=args.gpus,
                concurrency=args.concurrency,
                whisper_cpu=args.whisper_cpu,
                voice_only=args.voice_only,
                verbose=args.verbose,
            )
            results.append(result)

            # Append result to CSV immediately so we can resume anytime
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                row = result.copy()
                row["errors"] = "; ".join(row["errors"])
                writer.writerow(row)

            print(f"  Status: {result['status']}")
            print(f"  Lines: {result['successful_lines']}/{result['total_lines']} successful")
            print(f"  Avg ratio: {result['avg_ratio']:.3f}")
            print(f"  Time: voice={result['voice_gen_time']:.0f}s, tts={result['tts_gen_time']:.0f}s")

            prev_te = te

        # Cleanup after all TTS engines for this voice engine
        print(f"\n  [MEMORY] Cleaning up after voice engine: {ve}")
        _free_gpu_memory()

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if args.voice_only:
        print(f"{'Voice':<12} {'Pass':<6} {'Fail':<6} {'Score':<8} {'AvgTries':<9} {'DurCV':<8} {'Time':<10}")
        print("-" * 65)
        sorted_results = sorted(results, key=lambda x: x["avg_ratio"], reverse=True)
        for r in sorted_results:
            score_str = f"{r['avg_ratio']:.0%}"
            time_str = f"{r['total_time']:.0f}s"
            # Extract avg_attempts and duration_cv from errors field (stored as JSON)
            avg_tries = "?"
            dur_cv = "?"
            if r["errors"]:
                try:
                    meta = json.loads(r["errors"])
                    avg_tries = f"{meta.get('avg_attempts', 0):.1f}"
                    dur_cv = f"{meta.get('duration_cv', 0):.2f}"
                except (json.JSONDecodeError, KeyError):
                    pass
            print(f"{r['voice_engine']:<12} {r['successful_lines']:<6} {r['failed_lines']:<6} {score_str:<8} {avg_tries:<9} {dur_cv:<8} {time_str:<10}")
    else:
        print(f"{'Voice':<12} {'TTS':<12} {'Status':<12} {'Lines':<10} {'Avg Ratio':<12} {'Time':<12}")
        print("-" * 80)
        sorted_results = sorted(results, key=lambda x: x["avg_ratio"], reverse=True)
        for r in sorted_results:
            lines_str = f"{r['successful_lines']}/{r['total_lines']}"
            time_str = f"{r['total_time']:.0f}s"
            print(f"{r['voice_engine']:<12} {r['tts_engine']:<12} {r['status']:<12} {lines_str:<10} {r['avg_ratio']:<12.3f} {time_str:<12}")

    print(f"\nResults saved to: {csv_path}")
    print(f"Audio files saved to: {output_base}")


if __name__ == "__main__":
    main()
