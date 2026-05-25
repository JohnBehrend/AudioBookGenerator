#!/usr/bin/env python3
"""
Audiobook Generator - Module to generate audiobook audio from chapters.

This module provides:
- CLI entry point for sequential pipeline execution
- Gradio interface via --gradio flag
- Stage 5: TTS audio generation from chapter maps and voice samples

Usage:
    # Launch Gradio interface
    audiobook-interface --gradio

    # Run full pipeline via CLI
    audiobook-interface <epub_file> [--output-dir] [--verbose]

    # Or as a module
    python -m audiobook_generator --gradio
"""

import os
import sys
import time
import json
import re
import glob
import gc
import shutil
import traceback
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

from .utils import _get_attn_implementation
from .config import DEFAULTS, LLM_SETTINGS, AUDIO_SETTINGS
from .utils import natural_sort_key

TTS_ENGINE = os.environ.get('TTS_ENGINE', AUDIO_SETTINGS["default_tts_engine"])

from difflib import SequenceMatcher

from . import parse_chapter
from .llm_label_speakers import label_speakers
from .llm_describe_character import describe_characters
from .generate_voice_samples import generate_voice_samples as gen_voice_samples

from .utils import (
    get_chapters_dir,
    get_temp_dir,
    cleanup_temp_dir,
    load_temp_dir,
    save_temp_dir,
    get_chapters_dir_from_saved,
    ProgressHandler,
    copy_mp3_files_to_chapters,
    load_json_file as load_json,
    get_character_wav_file,
    load_seed_characters,
    get_chapter_map_files,
    parse_map_file,
    count_lines_per_character,
    transcribe_audio_with_whisper,
    distill_string,
    validate_audio_clean,
    get_validation_client,
)

# Import VoiceMapper for centralized TTS management
from .voice_mapper import VoiceMapper

# Import pipeline pure functions
from .pipeline import (
    normalize_script,
    add_postfix,
    prepare_script_for_tts,
    score_strings_pop,
    calculate_clip_points,
    should_retry,
    generate_output_filename,
    get_temp_filenames,
    is_generation_success,
    MIN_RATIO_THRESHOLD,
    MAX_RETRIES,
)


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class TTSConfig:
    """Configuration for TTS generation."""
    device: str = "cuda"
    tts_engine: str = "vibevoice"
    cfg_scale: float = 1.3
    output_dir: str = ""
    short_text_postfix: Optional[str] = DEFAULTS["short_text_postfix"]
    validation_model: Optional[Any] = None
    validation_client: Optional[Any] = None
    validate_clean: bool = False
    verbose: bool = False
    whisper_lock: Optional[threading.Lock] = None
    whisper_pool: Optional[Any] = None
    engine: Optional[Any] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def setup_validation_model(device: str, cpu: bool = False, fast: bool = False) -> Any:
    """Setup the Whisper validation model for audio transcription.

    Args:
        device: Device to run the model on (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu')
        cpu: If True, use CPU with float32 instead of GPU with float16
        fast: If True, use smaller model for faster transcription

    Returns:
        WhisperModel instance for audio validation
    """
    from faster_whisper import WhisperModel

    model_name = DEFAULTS["validation_model_name_fast"] if fast else DEFAULTS["validation_model_name"]
    if cpu:
        return WhisperModel(model_name, device="cpu", compute_type="float32")
    else:
        whisper_device = "cuda" if device.startswith("cuda") else device
        return WhisperModel(model_name, device=whisper_device, compute_type="float16")


def get_non_silent_audio_from_wavs(wav_filepath_list: List[str], min_silence_len: int = 1250, silence_thresh: int = -60) -> Any:
    """Remove silent audio from list of wave filepaths of wavs together. Return AudioSegment."""
    import pydub

    all_audio_segments = None
    for wav in wav_filepath_list:
        raw_audio_segment = pydub.AudioSegment.from_wav(wav)
        this_audio_segment = pydub.AudioSegment.empty()
        for (start_time, end_time) in pydub.silence.detect_nonsilent(raw_audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh):
            this_audio_segment += raw_audio_segment[start_time:end_time]
        if all_audio_segments is None:
            all_audio_segments = this_audio_segment
        else:
            all_audio_segments = all_audio_segments + this_audio_segment
    return all_audio_segments


def color_word(word: str, score: float) -> str:
    """
    Annotates terminal color strings around word.
    The gradient progresses from red (score=0) to green (score=1).
    """
    reset_code = "\033[0m"
    red = int(255 * (1 - score))
    green = int(255 * score)
    blue = 0

    color_code = f"\033[38;2;{red};{green};{blue}m"
    return f"{color_code}{word}{reset_code}"


def split_text_for_echo_tts(text: str, max_chunk_size: int = 500) -> list:
    """Split text into chunks that fit within Echo TTS token limits.

    Echo TTS has a sequence_length limit of 640 tokens (~30 seconds of audio).
    Splits at sentence boundaries first, then commas, then word boundaries.
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chunk_size:
            chunks.append(remaining)
            break

        # Find the best split point (sentence boundary)
        split_point = max_chunk_size

        # Look for sentence endings first
        for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            idx = remaining[:max_chunk_size].rfind(sep)
            if idx > max_chunk_size // 2:  # Must be past the middle
                split_point = idx + len(sep)
                break

        # If no sentence boundary, look for comma
        if split_point == max_chunk_size:
            for sep in [', ', '; ']:
                idx = remaining[:max_chunk_size].rfind(sep)
                if idx > max_chunk_size // 2:
                    split_point = idx + len(sep)
                    break

        # If still no good split, split at word boundary
        if split_point == max_chunk_size:
            last_space = remaining[:max_chunk_size].rfind(' ')
            if last_space > max_chunk_size // 2:
                split_point = last_space

        chunks.append(remaining[:split_point].strip())
        remaining = remaining[split_point:].strip()

    return chunks


def _get_ref_text_for_voice(voice_path: str, validation_model: Any, voice_name: str, verbose: bool) -> str:
    """Transcribe voice sample to get reference text for cloning, with fallback."""
    try:
        ref_text, _, _ = transcribe_audio_with_whisper(validation_model, voice_path)
        if verbose:
            print(f"    Transcribed ref_text for '{voice_name}': {ref_text[:80]}...")
        return ref_text
    except Exception as e:
        if verbose:
            print(f"    Warning: Failed to transcribe ref_text for '{voice_name}': {e}")
            print(f"    Falling back to static_voice_text (cloning quality may be degraded)")
        return DEFAULTS["static_voice_text"]


# ============================================================================
# STAGE 5: TTS AUDIO GENERATION
# ============================================================================


def _tts_generate_only(
    chapter_idx: int,
    line_idx: int,
    text: str,
    voice_name: str,
    voice_mapper: VoiceMapper,
    tts_config: TTSConfig,
    voice_path: Optional[str] = None,
) -> Optional[str]:
    """Generate TTS audio for a single line (no validation).

    Returns output_path on success, None on failure.
    """
    if not text or not text.strip():
        return None

    full_script, _ = prepare_script_for_tts(text, tts_config.short_text_postfix)

    output_path = generate_output_filename(tts_config.output_dir, chapter_idx, line_idx, is_final=False)

    if voice_path is None:
        voice_path = voice_mapper.get_voice_path(voice_name)
    if voice_path is None:
        raise Exception(f"No voice path found for '{voice_name}'")

    engine = tts_config.engine if tts_config.engine is not None else voice_mapper.get_engine()
    try:
        engine.generate_line(
            text=full_script,
            voice_path=voice_path,
            output_path=output_path,
            device=tts_config.device,
            validation_model=tts_config.validation_model,
            cfg_scale=tts_config.cfg_scale,
            verbose=tts_config.verbose,
        )
    except Exception as e:
        print(f"    Engine generation failed: {e}")
        if tts_config.verbose:
            traceback.print_exc()
        return None

    gc.collect()
    import torch
    torch.cuda.empty_cache()

    from scipy.io import wavfile
    sample_rate, waveform = wavfile.read(output_path)
    wavfile.write(output_path, sample_rate, waveform)

    return output_path


def _whisper_validate_and_clip(
    chapter_idx: int,
    line_idx: int,
    text: str,
    output_path: str,
    tts_config: TTSConfig,
    max_ratio: float,
) -> Tuple[float, float]:
    """Validate audio with Whisper and clip if needed.

    Returns (new_ratio, new_max_ratio). If ratio > max_ratio, renames to final.
    """
    full_script, postfix_detect_token = prepare_script_for_tts(text, tts_config.short_text_postfix)
    input_string = distill_string(full_script)

    detected_string = ""
    segments = []
    start_times = []
    end_times = []
    last_valid_token = None
    ratio = 0.0

    if tts_config.validation_model is not None:
        if tts_config.whisper_pool is not None:
            segments_list, info = tts_config.whisper_pool.transcribe(output_path, beam_size=5, word_timestamps=True)
        elif tts_config.whisper_lock:
            with tts_config.whisper_lock:
                segments_list, info = tts_config.validation_model.transcribe(output_path, beam_size=5, word_timestamps=True)
        else:
            segments_list, info = tts_config.validation_model.transcribe(output_path, beam_size=5, word_timestamps=True)

        segments = []
        start_times = []
        end_times = []
        for segment in segments_list:
            for word in segment.words:
                segments.append(distill_string(word.word.strip()))
                start_times.append(word.start)
                end_times.append(word.end)

        detected_string = distill_string(" ".join(segments))
        if tts_config.verbose:
            print(f"  [STT] Original text: {input_string}")
            print(f"  [STT] Whisper transcribed: {detected_string}")
        postfix_for_score = distill_string(tts_config.short_text_postfix) if tts_config.short_text_postfix else ""
        ratio, last_valid_token = score_strings_pop(distill_string(input_string), detected_string, lookahead=5, postfix=postfix_for_score)

        if tts_config.validate_clean and tts_config.validation_client is not None and ratio >= 0.85:
            is_clean, clean_msg = validate_audio_clean(
                audio_path=output_path,
                client=tts_config.validation_client,
                verbose=tts_config.verbose
            )
            if not is_clean:
                if tts_config.verbose:
                    print(f"  [Clean Check] FAILED: {clean_msg}")
                ratio = 0
            else:
                if tts_config.verbose:
                    print(f"  [Clean Check] PASSED: {clean_msg}")

    # Clipping
    if tts_config.validation_model is not None:
        if tts_config.short_text_postfix and (distill_string(tts_config.short_text_postfix) in detected_string) and (postfix_detect_token in segments):
            if detected_string.startswith(distill_string(tts_config.short_text_postfix)):
                if tts_config.verbose:
                    print("\nERROR: POSTFIX DETECTED BUT ONLY POSTFIX! -> Ratio 0\n")
                ratio = 0
            else:
                postfix_start_index = segments[::-1].index(postfix_detect_token)
                if len(end_times) > postfix_start_index + 1:
                    clip_end2 = end_times[::-1][postfix_start_index + 1]
                else:
                    clip_end2 = end_times[-1]
                clip_end1 = start_times[::-1][postfix_start_index]
                if tts_config.verbose:
                    print(f"\nPOSTFIX DETECTED CLIPPING to {clip_end1} - {clip_end2}\n")
                import pydub
                audio = pydub.AudioSegment.from_wav(output_path)
                trimmed_audio = audio[0:((clip_end1 + clip_end2) * 500)]
                trimmed_audio.export(output_path, format="wav")
        elif tts_config.short_text_postfix:
            if ((last_valid_token is None) or (last_valid_token == "")):
                if tts_config.verbose:
                    print("\nERROR: POSTFIX UN-DETECTED and INVALID VALUES. SKIP.\n")
            elif last_valid_token in segments:
                lastvalid_index = segments[::-1].index(last_valid_token)
                clip_end1 = end_times[::-1][lastvalid_index]
                if tts_config.verbose:
                    print(f"\nERROR: POSTFIX UN-DETECTED LAST VALID CLIPPING TO {last_valid_token} {clip_end1}\n")
                import pydub
                audio = pydub.AudioSegment.from_wav(output_path)
                trimmed_audio = audio[0:(clip_end1 * 1000)]
                trimmed_audio.export(output_path, format="wav")
            else:
                if tts_config.verbose:
                    print(f"\nERROR: POSTFIX UN-DETECTED even though last_valid_token = {last_valid_token} should be in {segments}\n")

    if ratio > max_ratio:
        max_ratio = ratio
        final_path = generate_output_filename(tts_config.output_dir, chapter_idx, line_idx, is_final=True)
        if os.path.exists(final_path):
            os.unlink(final_path)
        os.rename(output_path, final_path)

    return ratio, max_ratio


def generate_tts_for_line(
    chapter_idx: int,
    line_idx: int,
    text: str,
    voice_name: str,
    voice_mapper: VoiceMapper,
    tts_config: TTSConfig,
    voice_path: Optional[str] = None,
) -> Tuple[bool, float]:
    """Generate TTS audio for a single line.

    Args:
        chapter_idx: Chapter index
        line_idx: Line index within chapter
        text: Text to synthesize
        voice_name: Name of the voice/character
        voice_mapper: VoiceMapper instance
        tts_config: TTS configuration (device, engine, cfg_scale, output_dir, etc.)
        voice_path: Optional path to voice sample file. If not provided, uses voice_mapper.

    Returns:
        Tuple of (success: bool, ratio: float)
    """
    if not text or not text.strip():
        if tts_config.verbose:
            print(f"  Skipping line {line_idx} (empty text: '{text}')")
        return (True, 1.0)

    full_script, _ = prepare_script_for_tts(text, tts_config.short_text_postfix)

    ratio = 0.0
    max_ratio = float('-inf')
    retries = 0

    from transformers import set_seed
    set_seed(42)
    while should_retry(ratio, max_ratio, retries, MAX_RETRIES, MIN_RATIO_THRESHOLD):
        set_seed(42 + retries)

        output_path = _tts_generate_only(
            chapter_idx, line_idx, full_script, voice_name,
            voice_mapper, tts_config, voice_path,
        )
        if output_path is None:
            ratio = 0.0
            retries += 1
            continue

        ratio, max_ratio = _whisper_validate_and_clip(
            chapter_idx, line_idx, full_script, output_path,
            tts_config, max_ratio,
        )
        retries += 1

    temp_path = generate_output_filename(tts_config.output_dir, chapter_idx, line_idx, is_final=False)
    if os.path.exists(temp_path):
        os.unlink(temp_path)

    return is_generation_success(max_ratio, MIN_RATIO_THRESHOLD), max_ratio


# ============================================================================
# MAIN GENERATION FUNCTIONS
# ============================================================================


def generate_audiobook_from_chapters(
    chapters: list,
    chapter_maps: Dict[int, Tuple[Dict, Dict]],
    voices_map: Dict[str, str],
    output_dir: str,
    device: str = AUDIO_SETTINGS["default_device"],
    tts_engine: str = AUDIO_SETTINGS["default_tts_engine"],
    cfg_scale: float = DEFAULTS["cfg_scale"],
    max_chapters: Optional[int] = None,
    verbose: bool = False,
    turbo: bool = False,
    progress: Optional[callable] = None,
    duplicate_replacement_map: Optional[Dict[str, str]] = None,
    seed_voice_map: Optional[str] = None,
    whisper_device: Optional[str] = None,
    whisper_alt_gpu: bool = False,
    whisper_cpu: bool = False,
    debug_tts: bool = False,
    validate_clean: bool = False,
    max_retries: Optional[int] = None,
    enable_postfix: bool = True,
    concurrency: int = 1,
    gpus: Optional[List[str]] = None,
    whisper_concurrency: int = 1,
    whisper_fast: bool = False,
) -> Tuple[str, int]:
    """Generate audiobook from parsed chapters.

    This is a simplified interface for calling audiobook generation directly
    from the Gradio UI without subprocess.

    Args:
        chapters: List of chapter objects from parse_chapter.parse_epub_to_chapters
        chapter_maps: Dict mapping chapter_idx -> (character_map, line_map)
        voices_map: Dict mapping character names to voice file paths (wav)
        output_dir: Output directory for audio files
        device: Device to run on
        tts_engine: 'vibevoice', 'moss', 'echo-tts', 'omni', or 'vox'
        cfg_scale: CFG scale value
        max_chapters: Maximum number of chapters to process
        verbose: Print verbose output
        whisper_device: Device for Whisper validation model (defaults to device if None)
        turbo: Use KugelAudio turbo model (kugel-1-turbo)
        progress: Optional progress callback (gr.Progress() for Gradio, None for CLI)
        duplicate_replacement_map: Dict mapping duplicate character names to canonical names
        seed_voice_map: Path to existing voices_map.json (if provided)
        validate_clean: If True, validate audio contains only clean speech (no music/SFX)

    Returns:
        Tuple of (status_message, chapters_processed)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Limit chapters if specified
    chapters_to_process = chapters[:max_chapters] if max_chapters else chapters

    # Create unified progress handler for both Gradio and CLI
    # Using context manager for proper cleanup
    with ProgressHandler(progress=progress, total=len(chapters_to_process), desc="Audiobook Generation") as progress_handler:
        # Setup validation model (Whisper always runs)
        whisper_device = whisper_device if whisper_device is not None else device
        # Apply whisper_alt_gpu to override device to cuda:1 if not explicitly set
        if whisper_alt_gpu and whisper_device == device:
            whisper_device = "cuda:1"
        validation_model = setup_validation_model(whisper_device, cpu=whisper_cpu, fast=whisper_fast)
        # This avoids crashes from faster-whisper dependencies if validation_model is None
        validation_client = get_validation_client() if validate_clean else None
        # Initialize VoiceMapper with output_dir so it looks in the correct location
        voice_mapper = VoiceMapper(output_dir=output_dir, device=device, tts_engine=tts_engine, duplicate_replacement_map=duplicate_replacement_map)

        for voice, path in voices_map.items():
            # Use absolute path directly if file exists, otherwise try output_dir
            if os.path.isabs(path) and os.path.exists(path):
                voice_path = path
            else:
                voice_basename = os.path.basename(path)
                voice_path = os.path.join(output_dir, voice_basename)
            voice_mapper.add_voice_path(voice, voice_path)
        short_text_postfix = DEFAULTS["short_text_postfix"]

        # Resolve max_retries: use provided value, fall back to config
        if max_retries is None:
            max_retries = DEFAULTS.get("max_retries", 1)

        # Create Whisper pool for parallel validation
        whisper_lock = None
        whisper_pool = None
        if whisper_concurrency > 1:
            from .engines.pool import WhisperPool
            # Distribute Whisper models across available devices
            if not whisper_cpu and gpus and len(gpus) > 1:
                whisper_devices = gpus
            else:
                whisper_devices = None
            whisper_pool = WhisperPool(
                lambda dev=whisper_device: setup_validation_model(dev, cpu=whisper_cpu, fast=whisper_fast),
                size=whisper_concurrency,
                devices=whisper_devices,
            )
            if verbose:
                distro = whisper_devices if whisper_devices else [whisper_device] * whisper_concurrency
                print(f"[WHISPER] Created pool with {whisper_concurrency} models on {distro}")
        else:
            whisper_lock = threading.Lock()

        # Create multi-GPU worker pool if multiple GPUs specified
        worker_pool = None
        if gpus and len(gpus) > 1:
            if verbose:
                print(f"[MULTI-GPU] Creating worker pool with {len(gpus)} GPUs: {gpus}")
            engine_cls = voice_mapper.get_engine().__class__.__name__
            from .engines.pool import WorkerPool
            worker_pool = WorkerPool(tts_engine, engine_cls, gpus)
            worker_pool.start()
        elif gpus and len(gpus) == 1:
            if verbose:
                print(f"[MULTI-GPU] Single GPU: {gpus[0]}")
            device = gpus[0]

        processed = 0
        for i, chapter in enumerate(chapters_to_process):
            # Check if already generated (resume mode)
            mp3_path = os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.mp3")
            if os.path.exists(mp3_path):
                if verbose:
                    print(f"Skipping chapter {i} (already exists)")
                processed += 1
                continue

            progress_handler.update((i + 1)/len(chapters_to_process), desc=f"Processing Chapter {i}")
            if verbose:
                print(f"[CHAPTER_START] Chapter {i}/{len(chapters_to_process)}")

            # Get chapter map if available
            chapter_map = chapter_maps.get(i)
            if chapter_map:
                character_map, line_map = chapter_map
                character_map = {int(k): v for k, v in character_map.items()}
                line_map = {int(k): v for k, v in line_map.items()}
                # Use .get() with fallback to "narrator" for invalid character references
                line_to_character_map = {k: character_map.get(v, "narrator") for k, v in line_map.items()}
            else:
                # Fallback: assume narrator for all lines
                line_to_character_map = {}

            # Assign speakers based on line map
            for cobj in chapter:
                if not cobj.has_quotes:
                    # Unquoted lines always use narrator, overriding any LLM mapping
                    cobj.set_speaker("narrator")
                elif cobj.line_num in line_map:
                    # Quoted line is explicitly mapped to a speaker
                    char_name = line_to_character_map.get(cobj.line_num, "narrator")
                    cobj.set_speaker(char_name)
                else:
                    # Quoted line not in map - default to narrator
                    cobj.set_speaker("narrator")

            # Get unique voices used in this chapter
            voices_used = []
            for chapter_obj in chapter:
                speaker = chapter_obj.get_speaker()
                if speaker not in voices_used:
                    voices_used.append(speaker)

            # Pre-compute already generated line indices for this chapter (O(1) lookup)
            already_generated = {
                int(x.split(".")[-2])
                for x in glob.glob(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.*.wav"))
                if not x.endswith(".tmp.wav")
            }

            # Collect all work items for this chapter
            work_items = []
            for voice in voices_used:
                for j, chapter_obj in enumerate(chapter):
                    if voice != chapter_obj.get_speaker():
                        continue
                    line_num = chapter_obj.line_num
                    if line_num in already_generated:
                        if verbose:
                            print(f"Skipping chapter {i}.{line_num} (already generated)")
                        continue

                    # Debug TTS mode: print instead of generate
                    if debug_tts:
                        print(f"Chapter {i}, Line {line_num}, Speaker {voice}")
                        continue

                    # Get the voice path for this character
                    canonical_voice = voice
                    if duplicate_replacement_map and voice in duplicate_replacement_map:
                        canonical_voice = duplicate_replacement_map[voice]
                    voice_path = None
                    if canonical_voice in voices_map:
                        mapped_path = voices_map[canonical_voice]
                        if os.path.isabs(mapped_path):
                            voice_path = mapped_path
                        else:
                            voice_path = os.path.join(output_dir, mapped_path)
                        if not os.path.exists(voice_path):
                            voice_path = voice_mapper.get_voice_path(canonical_voice)
                    else:
                        voice_path = voice_mapper.get_voice_path(canonical_voice)
                    if voice_path is None:
                        print(f"  [WARN] Skipping line for '{voice}' — no voice file available")
                        continue

                    work_items.append({
                        "chapter_idx": i,
                        "line_idx": line_num,
                        "text": chapter_obj.text,
                        "voice_name": voice,
                        "voice_path": voice_path,
                        "enumerate_idx": j,
                    })

            # Process work items with thread pool (concurrency=1 preserves sequential behavior)
            tts_config = TTSConfig(
                device=device,
                tts_engine=tts_engine,
                cfg_scale=cfg_scale,
                output_dir=output_dir,
                short_text_postfix=(short_text_postfix if (validation_model is not None) else None),
                validation_model=validation_model,
                verbose=verbose,
                validation_client=validation_client,
                validate_clean=validate_clean,
                whisper_lock=whisper_lock,
                whisper_pool=whisper_pool,
                engine=worker_pool,
            )

            if concurrency > 1:
                # Streaming pipeline: TTS workers generate, validators validate concurrently
                if verbose:
                    print(f"[STREAM] Processing {len(work_items)} lines with {concurrency} TTS workers, {whisper_concurrency} validators")

                # Per-line state tracking
                line_state: Dict[str, dict] = {}
                for item in work_items:
                    key = f"{item['chapter_idx']}_{item['line_idx']}"
                    line_state[key] = {"retries": 0, "max_ratio": float('-inf')}
                line_state_lock = threading.Lock()
                completed_count = 0
                completed_lock = threading.Lock()
                total_items = len(work_items)
                shutdown_tts = threading.Event()
                shutdown_val = threading.Event()

                # Queues: work_queue -> TTS -> validation_queue -> Validator -> (finalize or retry back to work_queue)
                work_queue: queue.Queue = queue.Queue()
                validation_queue: queue.Queue = queue.Queue(maxsize=concurrency * 2)
                for item in work_items:
                    work_queue.put(item)

                def tts_worker():
                    nonlocal completed_count
                    my_thread_id = threading.current_thread().ident
                    while not shutdown_tts.is_set():
                        try:
                            item = work_queue.get(timeout=0.5)
                        except queue.Empty:
                            continue

                        if item is None:
                            work_queue.task_done()
                            break

                        if not item["text"] or not item["text"].strip():
                            work_queue.task_done()
                            continue

                        key = f"{item['chapter_idx']}_{item['line_idx']}"
                        with line_state_lock:
                            state = line_state[key]
                            if state["retries"] >= MAX_RETRIES:
                                with completed_lock:
                                    completed_count += 1
                                work_queue.task_done()
                                continue
                            retry_num = state["retries"]

                        from transformers import set_seed
                        set_seed(42 + retry_num)

                        full_script, _ = prepare_script_for_tts(item["text"], tts_config.short_text_postfix)
                        output_path = generate_output_filename(tts_config.output_dir, item["chapter_idx"], item["line_idx"], is_final=False, thread_id=my_thread_id)

                        try:
                            engine = tts_config.engine if tts_config.engine is not None else voice_mapper.get_engine()
                            engine.generate_line(
                                text=full_script,
                                voice_path=item["voice_path"],
                                output_path=output_path,
                                device=tts_config.device,
                                validation_model=tts_config.validation_model,
                                cfg_scale=tts_config.cfg_scale,
                                verbose=tts_config.verbose,
                            )
                        except Exception as e:
                            print(f"    Engine generation failed: {e}")
                            if tts_config.verbose:
                                traceback.print_exc()
                            with line_state_lock:
                                state["retries"] += 1
                            work_queue.task_done()
                            continue

                        gc.collect()
                        import torch
                        torch.cuda.empty_cache()

                        validation_queue.put((item, output_path, key))
                        work_queue.task_done()

                def validator_worker():
                    nonlocal completed_count
                    while not shutdown_val.is_set():
                        try:
                            item, output_path, key = validation_queue.get(timeout=0.5)
                        except queue.Empty:
                            continue

                        if item is None:
                            validation_queue.task_done()
                            continue

                        if not os.path.exists(output_path):
                            validation_queue.task_done()
                            continue

                        full_script, postfix_detect_token = prepare_script_for_tts(item["text"], tts_config.short_text_postfix)
                        input_string = distill_string(full_script)

                        detected_string = ""
                        segments = []
                        start_times = []
                        end_times = []
                        last_valid_token = None
                        ratio = 0.0

                        if tts_config.validation_model is not None:
                            try:
                                if tts_config.whisper_pool is not None:
                                    segments_list, info = tts_config.whisper_pool.transcribe(output_path, beam_size=5, word_timestamps=True)
                                elif tts_config.whisper_lock:
                                    with tts_config.whisper_lock:
                                        segments_list, info = tts_config.validation_model.transcribe(output_path, beam_size=5, word_timestamps=True)
                                else:
                                    segments_list, info = tts_config.validation_model.transcribe(output_path, beam_size=5, word_timestamps=True)
                            except Exception as e:
                                print(f"    Whisper validation failed: {e}")
                                if tts_config.verbose:
                                    traceback.print_exc()
                                with line_state_lock:
                                    state = line_state[key]
                                    if should_retry(0, state["max_ratio"], state["retries"], MAX_RETRIES, MIN_RATIO_THRESHOLD):
                                        state["retries"] += 1
                                        work_queue.put(item)
                                    else:
                                        with completed_lock:
                                            completed_count += 1
                                validation_queue.task_done()
                                continue

                            segments = []
                            start_times = []
                            end_times = []
                            for segment in segments_list:
                                for word in segment.words:
                                    segments.append(distill_string(word.word.strip()))
                                    start_times.append(word.start)
                                    end_times.append(word.end)

                            detected_string = distill_string(" ".join(segments))
                            if tts_config.verbose:
                                print(f"  [STT] Original text: {input_string}")
                                print(f"  [STT] Whisper transcribed: {detected_string}")
                            postfix_for_score = distill_string(tts_config.short_text_postfix) if tts_config.short_text_postfix else ""
                            ratio, last_valid_token = score_strings_pop(distill_string(input_string), detected_string, lookahead=5, postfix=postfix_for_score)

                            if tts_config.validate_clean and tts_config.validation_client is not None and ratio >= 0.85:
                                is_clean, clean_msg = validate_audio_clean(
                                    audio_path=output_path,
                                    client=tts_config.validation_client,
                                    verbose=tts_config.verbose
                                )
                                if not is_clean:
                                    if tts_config.verbose:
                                        print(f"  [Clean Check] FAILED: {clean_msg}")
                                    ratio = 0
                                else:
                                    if tts_config.verbose:
                                        print(f"  [Clean Check] PASSED: {clean_msg}")

                        # Clipping
                        if tts_config.validation_model is not None:
                            if tts_config.short_text_postfix and (distill_string(tts_config.short_text_postfix) in detected_string) and (postfix_detect_token in segments):
                                if detected_string.startswith(distill_string(tts_config.short_text_postfix)):
                                    if tts_config.verbose:
                                        print("\nERROR: POSTFIX DETECTED BUT ONLY POSTFIX! -> Ratio 0\n")
                                    ratio = 0
                                else:
                                    postfix_start_index = segments[::-1].index(postfix_detect_token)
                                    if len(end_times) > postfix_start_index + 1:
                                        clip_end2 = end_times[::-1][postfix_start_index + 1]
                                    else:
                                        clip_end2 = end_times[-1]
                                    clip_end1 = start_times[::-1][postfix_start_index]
                                    if tts_config.verbose:
                                        print(f"\nPOSTFIX DETECTED CLIPPING to {clip_end1} - {clip_end2}\n")
                                    import pydub
                                    audio = pydub.AudioSegment.from_wav(output_path)
                                    trimmed_audio = audio[0:((clip_end1 + clip_end2) * 500)]
                                    trimmed_audio.export(output_path, format="wav")
                            elif tts_config.short_text_postfix:
                                if ((last_valid_token is None) or (last_valid_token == "")):
                                    if tts_config.verbose:
                                        print("\nERROR: POSTFIX UN-DETECTED and INVALID VALUES. SKIP.\n")
                                elif last_valid_token in segments:
                                    lastvalid_index = segments[::-1].index(last_valid_token)
                                    clip_end1 = end_times[::-1][lastvalid_index]
                                    if tts_config.verbose:
                                        print(f"\nERROR: POSTFIX UN-DETECTED LAST VALID CLIPPING TO {last_valid_token} {clip_end1}\n")
                                    import pydub
                                    audio = pydub.AudioSegment.from_wav(output_path)
                                    trimmed_audio = audio[0:(clip_end1 * 1000)]
                                    trimmed_audio.export(output_path, format="wav")
                                else:
                                    if tts_config.verbose:
                                        print(f"\nERROR: POSTFIX UN-DETECTED even though last_valid_token = {last_valid_token} should be in {segments}\n")

                        with line_state_lock:
                            state = line_state[key]
                            if ratio > state["max_ratio"]:
                                state["max_ratio"] = ratio
                                final_path = generate_output_filename(tts_config.output_dir, item["chapter_idx"], item["line_idx"], is_final=True)
                                if os.path.exists(final_path):
                                    os.unlink(final_path)
                                os.rename(output_path, final_path)
                            else:
                                if os.path.exists(output_path):
                                    os.unlink(output_path)

                            if should_retry(ratio, state["max_ratio"], state["retries"], MAX_RETRIES, MIN_RATIO_THRESHOLD):
                                state["retries"] += 1
                                work_queue.put(item)
                            else:
                                for temp_path in get_temp_filenames(tts_config.output_dir, item["chapter_idx"], item["line_idx"]):
                                    if os.path.exists(temp_path):
                                        os.unlink(temp_path)
                                with completed_lock:
                                    completed_count += 1
                                if verbose:
                                    print(f"[LINE_PROGRESS] Chapter {item['chapter_idx']}, Line {item['line_idx']}, Voice: {item['voice_name']}, Ratio: {int(state['max_ratio'] * 100)}")
                                progress_handler.update(
                                    completed_count / total_items,
                                    desc=f"Processing Chapter {item['chapter_idx']} Line {item['line_idx']} Ratio {int(state['max_ratio'] * 100)}"
                                )

                        validation_queue.task_done()

                # Start TTS workers
                tts_threads = []
                for _ in range(concurrency):
                    t = threading.Thread(target=tts_worker)
                    t.start()
                    tts_threads.append(t)

                # Start validator workers
                val_threads = []
                for _ in range(whisper_concurrency):
                    t = threading.Thread(target=validator_worker)
                    t.start()
                    val_threads.append(t)

                # Wait for all items to complete, then signal shutdown
                while True:
                    with completed_lock:
                        if completed_count >= total_items:
                            break
                    time.sleep(0.5)

                # Send sentinels and wait for clean shutdown
                shutdown_tts.set()
                shutdown_val.set()
                for _ in range(concurrency):
                    work_queue.put(None)
                for _ in range(whisper_concurrency):
                    validation_queue.put((None, None, None))

                for t in tts_threads + val_threads:
                    t.join(timeout=10)

                # Final cleanup of any remaining temp files
                for item in work_items:
                    for temp_path in get_temp_filenames(tts_config.output_dir, item["chapter_idx"], item["line_idx"]):
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
            else:
                # Sequential processing (original behavior)
                for item in work_items:
                    success, ratio = generate_tts_for_line(
                        chapter_idx=item["chapter_idx"],
                        line_idx=item["line_idx"],
                        text=item["text"],
                        voice_name=item["voice_name"],
                        voice_mapper=voice_mapper,
                        tts_config=tts_config,
                        voice_path=item["voice_path"],
                    )
                    progress_handler.update(
                        (item["enumerate_idx"] + 1) / len(chapter),
                        desc=f"Processing Chapter {i} Line {item['line_idx']} Ratio {int(ratio * 100)}"
                    )
                    if verbose:
                        print(f"[LINE_PROGRESS] Chapter {i}, Line {item['line_idx']}, Voice: {item['voice_name']}, Ratio: {int(ratio * 100)}")

            # Assemble chapter MP3 from WAV files
            progress_handler.update(1, desc=f"Assembling Chapters")
            wav_files = sorted(glob.glob(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.*.wav")), key=natural_sort_key)
            if wav_files:
                audio = get_non_silent_audio_from_wavs(wav_files)
                mp3_path = os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.mp3")
                audio.export(str(mp3_path), format="mp3")

                # Clean up individual WAV files
                for wav in wav_files:
                    os.unlink(wav)

                if verbose:
                    print(f"Chapter {i}: Created {os.path.basename(mp3_path)} from {len(wav_files)} audio segments.")
            else:
                if verbose:
                    print(f"Chapter {i}: No WAV files generated.")

            if verbose:
                print(f"[CHAPTER_COMPLETE] Chapter {i}/{len(chapters_to_process)}")

            # Clear cache after each chapter to free VRAM
            import torch
            torch.cuda.empty_cache()
            gc.collect()

            processed += 1

        # Shutdown multi-GPU worker pool if used
        if worker_pool is not None:
            worker_pool.shutdown()
            if verbose:
                print("[MULTI-GPU] Worker pool shutdown")

        return f"Generated {processed} chapters successfully.", processed


def _get_mp3_duration(mp3_path: str) -> float:
    """Get duration of an MP3 file in seconds using ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", mp3_path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def assemble_audiobook_m4b(output_dir: str, verbose: bool = False,
                           max_chapters_per_part: int = 255,
                           max_hours_per_part: float = 12.0,
                           book_name: str = "audiobook") -> str:
    """Assemble chapter MP3 files into .m4b audiobook(s), splitting at boundaries.

    Splits into multiple parts when exceeding max_chapters_per_part (MP4 chapter
    marker limit) or max_hours_per_part (hardware compatibility).

    Args:
        output_dir: Directory containing chapter_XX.mp3 files
        verbose: Print verbose output
        max_chapters_per_part: Max chapters per m4b (MP4 limit is 255)
        max_hours_per_part: Max hours per m4b for hardware compatibility

    Returns:
        Path to the assembled .m4b file(s), or empty string if no chapters found
    """
    import subprocess

    mp3_files = sorted(glob.glob(os.path.join(output_dir, "chapter_*.mp3")), key=natural_sort_key)
    if not mp3_files:
        if verbose:
            print("[M4B] No chapter MP3 files found to assemble.")
        return ""

    # Get durations for all chapters
    if verbose:
        print(f"[M4B] Calculating durations for {len(mp3_files)} chapters...")
    durations = []
    for mp3 in mp3_files:
        dur = _get_mp3_duration(mp3)
        durations.append(dur)

    total_hours = sum(durations) / 3600
    if verbose:
        print(f"[M4B] Total audiobook duration: {total_hours:.1f} hours")

    # Split into parts based on chapter count and duration limits
    parts = []
    current_part = []
    current_duration = 0.0

    for i, (mp3, dur) in enumerate(zip(mp3_files, durations)):
        current_part.append((mp3, dur))
        current_duration += dur

        hours_in_part = current_duration / 3600

        # Check if we need to split (not on last chapter)
        if i < len(mp3_files) - 1 and (
            len(current_part) >= max_chapters_per_part or
            hours_in_part >= max_hours_per_part
        ):
            parts.append(list(current_part))
            current_part = []
            current_duration = 0.0

    if current_part:
        parts.append(current_part)

    if len(parts) > 1 and verbose:
        print(f"[M4B] Splitting into {len(parts)} parts:")
        for idx, part in enumerate(parts):
            part_hours = sum(d for _, d in part) / 3600
            print(f"  Part {idx + 1}: {len(part)} chapters, {part_hours:.1f} hours")

    # Assemble each part
    m4b_paths = []
    for part_idx, part in enumerate(parts):
        if len(parts) == 1:
            m4b_path = os.path.join(output_dir, f"{book_name}.m4b")
        else:
            m4b_path = os.path.join(output_dir, f"{book_name}_part{part_idx + 1}.m4b")

        # Build concat input list with chapter markers
        concat_lines = []
        cumulative_time = 0.0
        for mp3, dur in part:
            concat_lines.append(f"file '{os.path.abspath(mp3)}'")
            cumulative_time += dur

        concat_tmp = os.path.join(output_dir, f"_ffmpeg_concat_{part_idx}.txt")
        with open(concat_tmp, "w") as f:
            f.write("\n".join(concat_lines) + "\n")

        # First pass: concatenate
        temp_m4b = os.path.join(output_dir, f"_temp_{part_idx}.m4b")
        cmd = [
            "ffmpeg", "-y",
            "-threads", "0",
            "-f", "concat", "-safe", "0", "-i", concat_tmp,
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-metadata", "title=Audiobook",
            temp_m4b
        ]

        if verbose:
            part_hours = sum(d for _, d in part) / 3600
            print(f"[M4B] Assembling {len(part)} chapters ({part_hours:.1f}h) into {m4b_path}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(concat_tmp)

        if result.returncode != 0:
            if verbose:
                print(f"[M4B] ffmpeg failed: {result.stderr[:500]}")
            continue

        # Second pass: add chapter markers
        chapter_meta = os.path.join(output_dir, f"_chapters_{part_idx}.txt")
        cumulative_time = 0.0
        with open(chapter_meta, "w") as f:
            f.write(";FFMETADATA1\n")
            for idx, (mp3, dur) in enumerate(part):
                chapter_name = os.path.splitext(os.path.basename(mp3))[0]
                start_ms = int(cumulative_time * 1000)
                cumulative_time += dur
                end_ms = int(cumulative_time * 1000)
                f.write(f"[CHAPTER]\n")
                f.write(f"TIMEBASE=1/1000\n")
                f.write(f"START={start_ms}\n")
                f.write(f"END={end_ms}\n")
                f.write(f"title={chapter_name}\n")
                if idx < len(part) - 1:
                    f.write("\n")

        cmd = [
            "ffmpeg", "-y",
            "-i", temp_m4b,
            "-i", chapter_meta,
            "-map_metadata", "1",
            "-c", "copy",
            "-movflags", "+faststart",
            m4b_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(chapter_meta)
        os.unlink(temp_m4b)

        if result.returncode != 0:
            if verbose:
                print(f"[M4B] ffmpeg failed: {result.stderr[:500]}")
            continue

        if verbose:
            size_mb = os.path.getsize(m4b_path) / (1024 * 1024)
            print(f"[M4B] Created {m4b_path} ({size_mb:.1f} MB)")

        m4b_paths.append(m4b_path)

    return m4b_paths[0] if m4b_paths else ""


# ============================================================================
# STATE MANAGEMENT (Internal to this module)
# ============================================================================


class PipelineState:
    """Manages state for the audiobook pipeline."""

    def __init__(self, output_dir: str, voice_engine: str = None):
        self.output_dir = Path(output_dir)
        # Use output_dir directly as chapters_dir (not nested)
        # This ensures consistent behavior between CLI and Gradio
        self.chapters_dir = self.output_dir
        self.chapters_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_state = None
        self.chapters = None
        self.chapter_maps = {}
        self.characters = []
        self.character_descriptions = {}
        self.voice_map = {}
        self.selected_character = None
        self.voice_engine = voice_engine or "omni"  # Default to omni for backwards compatibility

    def load_chapter_maps(self):
        """Load all chapter map files from the chapters directory."""
        self.chapter_maps = {}
        # Only match map files with simple numeric names (chapter_X.map.json, not chapter_X.result.N.map.json)
        map_files = sorted([f for f in self.chapters_dir.glob("*.map.json")
                           if re.match(r"^chapter_\d+\.map\.json$", f.name)],
                          key=natural_sort_key)

        for map_file in map_files:
            try:
                with open(map_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) >= 2:
                    character_map = data[0]
                    line_map = data[1]
                elif isinstance(data, dict):
                    character_map = data.get("character_map", {})
                    line_map = data.get("line_map", {})
                else:
                    continue

                character_map = {int(k): v for k, v in character_map.items()}
                line_map = {int(k): v for k, v in line_map.items()}

                map_filename = map_file.name.replace(".map.json", "")
                chapter_idx = int(map_filename.replace("chapter_", ""))
                self.chapter_maps[chapter_idx] = (character_map, line_map)
            except Exception as e:
                print(f"Error loading map file {map_file}: {e}")

        return self.chapter_maps

    def get_characters(self) -> List[str]:
        """Extract unique character names from map files."""
        from .utils import get_characters_from_map_files
        self.characters = get_characters_from_map_files(self.chapters_dir)
        return self.characters

    def load_character_descriptions(self):
        """Load character descriptions from JSON file."""
        descriptions_file = self.output_dir / "characters_descriptions.json"
        if descriptions_file.exists():
            self.character_descriptions = load_json(str(descriptions_file))
        return self.character_descriptions

    def load_voice_map(self):
        """Load voice map from JSON file."""
        voices_map_file = self.chapters_dir / "voices_map.json"
        if voices_map_file.exists():
            self.voice_map = load_json(str(voices_map_file))
        else:
            # Create voice map from character descriptions
            self.voice_map = {"narrator": "narrator.wav"}
            for char_name in self.character_descriptions.keys():
                voice_file = f"{char_name}.wav"
                self.voice_map[char_name] = voice_file
        return self.voice_map

    def get_pipeline_state(self) -> str:
        """Determine current pipeline state based on existing files."""
        # Check for Stage 1 completion (chapter text files)
        chapter_files = sorted(self.chapters_dir.glob("chapter_*.txt"), key=natural_sort_key)
        if not chapter_files:
            return "initial"

        # Check for Stage 2 completion (map files)
        map_files = sorted(self.chapters_dir.glob("*.map.json"), key=natural_sort_key)
        if not map_files:
            return "epub_parsed"

        # Check for Stage 3 completion (characters descriptions)
        descriptions_file = self.output_dir / "characters_descriptions.json"
        if not descriptions_file.exists():
            return "labels_complete"

        # Check for Stage 4 completion (voice samples)
        wav_files = list(self.chapters_dir.glob("*.wav"))
        if not wav_files:
            return "characters_described"

        # Check for Stage 5 completion (final audiobook)
        mp3_files = sorted(self.chapters_dir.glob("chapter_*.mp3"), key=natural_sort_key)
        if not mp3_files:
            return "voice_samples_complete"

        return "audiobook_complete"

    def write_chapter_text_files(self, chapters):
        """Write chapter objects to text files."""
        from .parse_chapter import write_chapters_to_txt
        return write_chapters_to_txt(chapters, str(self.chapters_dir))


# ============================================================================
# CLI ORCHESTRATION
# ============================================================================


def run_full_pipeline(epub_path: str, output_dir: str, max_chapters: int = None,
                      verbose: bool = False, api_key: str = None, llm_port: str = None,
                      voice_engine: str = "moss", tts_engine: str = "omni", turbo: bool = False,
                      device: str = AUDIO_SETTINGS["default_device"], seed_voice_map: str = None,
                      num_llm_attempts: int = DEFAULTS["num_llm_attempts"],
                      resume: bool = False, whisper_device: str = None, whisper_alt_gpu: bool = False,
                      whisper_cpu: bool = False, debug_tts: bool = False, validate: bool = False,
                      validate_clean: bool = False, max_retries: int = None,
                       enable_postfix: bool = True, concurrency: int = 1,
                       gpus: Optional[List[str]] = None, whisper_concurrency: int = 1,
                       whisper_fast: bool = False,
                        llm_model: str = None,
                       use_chunkformer: bool = False) -> str:
    """Run the full audiobook pipeline from EPUB to MP3.

    Args:
        epub_path: Path to the EPUB file
        output_dir: Output directory for all generated files
        max_chapters: Maximum number of chapters to process
        verbose: Print verbose output
        api_key: LLM API key for speaker labeling and character descriptions
        llm_port: LLM endpoint port (e.g., LM Studio)
        voice_engine: TTS engine for voice sample generation ('moss', 'omni', 'vox')
        tts_engine: TTS engine for audiobook generation ('vibevoice', 'moss', 'echo-tts', 'omni', 'vox')
        turbo: Reserved for future turbo models
        device: CUDA device (e.g., 'cuda', 'cuda:1')
        seed_voice_map: Path to existing voices_map.json to seed voices
        num_llm_attempts: Number of LLM attempts for speaker labeling
        resume: If True, detect existing state and resume from where it left off
        whisper_device: Device for Whisper validation model (defaults to device if None)
        validate: If True, enable LLM validation for voice sample generation
        validate_clean: If True, validate audio contains only clean speech (no music/SFX)

    Returns:
        Status message
    """
    # Apply CLI overrides to centralized config so all downstream functions pick them up
    if api_key:
        LLM_SETTINGS["api_key"] = api_key
    if llm_port:
        LLM_SETTINGS["port"] = int(llm_port)
    if llm_model:
        LLM_SETTINGS["default_model"] = llm_model

    # Initialize state
    state = PipelineState(output_dir)

    if resume:
        # Detect current state and load existing data
        state.pipeline_state = state.get_pipeline_state()
        state.load_chapter_maps()
        state.get_characters()
        state.load_character_descriptions()
        state.load_voice_map()

        if verbose:
            print(f"[RESUME] Detected state: {state.pipeline_state}")
            print(f"[RESUME] Loaded {len(state.chapter_maps)} chapter maps")
            print(f"[RESUME] Found {len(state.characters)} characters")
            print(f"[RESUME] Loaded {len(state.character_descriptions)} character descriptions")
            print(f"[RESUME] Loaded {len(state.voice_map)} voice mappings")

    # Load duplicate replacement map if available (from Stage 3)
    duplicate_replacement_map = {}
    replacement_map_file = os.path.join(output_dir, "duplicate_replacement_map.json")
    if os.path.exists(replacement_map_file):
        duplicate_replacement_map = load_json(replacement_map_file)
        if verbose and duplicate_replacement_map:
            print(f"[DUPLICATE MAP] Loaded {len(duplicate_replacement_map)} replacements from duplicate_replacement_map.json")

    # Check for missing voice file mappings and add fallbacks
    # This handles cases where LLM labeled characters differently than the voice file names
    # e.g., "elan morin tedronai" should map to "baalzamon.wav"
    if resume:
        # Get list of available voice files in the output directory
        available_voices = set()
        for f in Path(output_dir).glob("*.wav"):
            voice_name = f.stem  # filename without extension
            available_voices.add(voice_name)
        if verbose and available_voices:
            print(f"[VOICE CHECK] Found {len(available_voices)} voice files: {sorted(available_voices)}")

        # Check for characters in voice_map that don't have corresponding voice files
        missing_mappings = {}
        for char_name, voice_path_rel in state.voice_map.items():
            voice_path_full = os.path.join(output_dir, voice_path_rel)
            if not os.path.exists(voice_path_full):
                if verbose:
                    print(f"[VOICE CHECK] Missing voice file for '{char_name}': {voice_path_full}")
                # Try to find a matching voice file by looking for similar names
                # e.g., "elan morin tedronai" -> "baalzamon"
                for available in available_voices:
                    if available in char_name or char_name in available:
                        missing_mappings[char_name] = available
                        if verbose:
                            print(f"[VOICE CHECK]   -> Mapped to available voice: '{available}'")
                        break

    # Load seed voices if provided
    seed_characters = None
    if seed_voice_map:
        seed_characters = load_json(seed_voice_map)
        if seed_characters:
            # Resolve relative paths to absolute paths based on seed_voice_map location
            seed_voice_map_dir = os.path.dirname(os.path.abspath(seed_voice_map))
            resolved_seed_characters = {}
            for char_name, voice_path in seed_characters.items():
                if os.path.isabs(voice_path):
                    resolved_seed_characters[char_name] = voice_path
                else:
                    resolved_seed_characters[char_name] = os.path.join(seed_voice_map_dir, voice_path)
            seed_characters = resolved_seed_characters
            # Merge seed voices into state.voice_map
            for char_name, voice_path in seed_characters.items():
                state.voice_map[char_name] = voice_path
            if verbose:
                print(f"[SEED] Loaded {len(seed_characters)} seeded characters from {seed_voice_map}")
        else:
            if verbose:
                print(f"[SEED] No characters found in seed file: {seed_voice_map}")

    # Check if EPUB is already parsed (for resume mode)
    chapter_files = sorted(state.chapters_dir.glob("chapter_*.txt"), key=natural_sort_key)
    epub_parsed = len(chapter_files) > 0

    # Store chapters in state for reuse (avoid re-parsing)
    # Stage 1: Parse EPUB with progress (skip if already done in resume mode)
    if resume and epub_parsed:
        if verbose:
            print(f"[STAGE 1] Skipping - EPUB already parsed ({len(chapter_files)} chapters found)")
        # Load existing chapters from text files (don't re-parse EPUB)
        if not state.chapters:
            from .parse_chapter import load_chapters_from_txt
            state.chapters = load_chapters_from_txt(
                str(state.output_dir),
                max_chapters=max_chapters
            )
            if verbose:
                print(f"[STAGE 1] Loaded {len(state.chapters)} chapters from existing files")
    else:
        if verbose:
            print(f"[STAGE 1] Parsing EPUB: {epub_path}")

        # Use tqdm for progress tracking (falls back to verbose print if tqdm not available)
        total_lines_estimate = None  # We'll estimate from the EPUB parsing
        with ProgressHandler(progress=None, use_tqdm=True, total=0, desc="Parsing EPUB") as handler:
            chapters = parse_chapter.parse_epub_to_chapters(epub_path, max_chapters=max_chapters)
            if not chapters:
                return "Error: No chapters found in EPUB file."

            # Store chapters in state for Stage 5 reuse
            state.chapters = chapters
            state.write_chapter_text_files(chapters)
            if verbose:
                print(f"[STAGE 1] Parsed {len(chapters)} chapters")

    # Stage 2: Label Speakers with progress
    if verbose:
        print(f"[STAGE 2] Labeling speakers...")
    chapter_files = sorted([f for f in state.chapters_dir.glob("chapter_*.txt")
                           if re.match(r"^chapter_\d+\.txt$", f.name)],
                          key=natural_sort_key)
    num_chapters = len(chapter_files)

    # Check if speakers are already labeled (for resume mode)
    map_files = sorted(state.chapters_dir.glob("*.map.json"), key=natural_sort_key)
    speakers_labeled = len(map_files) == num_chapters

    if resume and speakers_labeled:
        if verbose:
            print(f"[STAGE 2] Skipping - Speakers already labeled ({len(map_files)} map files found)")
        state.load_chapter_maps()
        # Note: characters already loaded in resume block at start of function
    else:
        with ProgressHandler(progress=None, use_tqdm=True, total=num_chapters, desc="Labeling speakers") as handler:
            for i, chapter_file in enumerate(chapter_files):
                # Skip if already labeled (for resume after partial completion)
                map_file = state.chapters_dir / f"chapter_{i:02d}.map.json"
                if resume and map_file.exists():
                    if verbose:
                        print(f"[STAGE 2] Skipping chapter {i} - already labeled")
                    handler.update((i + 1) / num_chapters, desc=f"Labeling chapter {i + 1}/{num_chapters}")
                    continue

                handler.update((i + 1) / num_chapters, desc=f"Labeling chapter {i + 1}/{num_chapters}")

                result_msg, char_map, line_map = label_speakers(
                    txt_file=str(chapter_file),
                    num_attempts=num_llm_attempts,
                    verbose=verbose,
                    seed_characters=seed_characters
                )

                if verbose:
                    print(f"  {result_msg}")

        state.load_chapter_maps()
        state.get_characters()

    # Stage 3: Describe Characters with progress
    descriptions_file = state.output_dir / "characters_descriptions.json"
    descriptions_exist = descriptions_file.exists()

    # Check if voice engine changed (for resume mode)
    force_regenerate_descriptions = False
    if resume and descriptions_exist:
        metadata_file = state.output_dir / "description_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as mf:
                metadata = json.load(mf)
                old_voice_engine = metadata.get("voice_engine", "omni")
                if old_voice_engine != voice_engine:
                    force_regenerate_descriptions = True
                    if verbose:
                        print(f"[STAGE 3] Voice engine changed from '{old_voice_engine}' to '{voice_engine}' - regenerating descriptions")
        else:
            # No metadata file - assume omni for backwards compatibility
            if voice_engine != "omni":
                force_regenerate_descriptions = True
                if verbose:
                    print(f"[STAGE 3] No metadata file found, voice engine is '{voice_engine}' - regenerating descriptions")

    if resume and descriptions_exist and not force_regenerate_descriptions:
        if verbose:
            print(f"[STAGE 3] Skipping - character descriptions already exist")
        state.load_character_descriptions()
        if verbose:
            print(f"[STAGE 3] Loaded {len(state.character_descriptions)} character descriptions")
    else:
        if verbose:
            print(f"[STAGE 3] Describing {len(state.characters)} characters...")

        num_chars = len(state.characters)
        with ProgressHandler(progress=None, use_tqdm=True, total=num_chars, desc="Describing characters") as handler:
            result_msg, character_descriptions = describe_characters(
                output_dir=str(state.output_dir),
                chapters_dir=str(state.output_dir),
                verbose=verbose,
                seed_characters=seed_characters,
                progress_callback=handler.update,
                voice_engine=voice_engine
            )

            if verbose:
                print(f"  {result_msg}")

            state.load_character_descriptions()

            # Save metadata to track which voice engine was used
            metadata_file = state.output_dir / "description_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as mf:
                json.dump({"voice_engine": voice_engine}, mf)

    # Stage 4: Generate Voice Samples with progress
    # Check for existing voice samples (resume mode)
    wav_files = list(state.chapters_dir.glob("*.wav"))
    wav_stems = {f.stem.lower().replace(" ", "") for f in wav_files}
    # All described characters must have voice files to skip Stage 4
    char_stems = {c.lower().replace(" ", "") for c in state.character_descriptions}
    voice_samples_exist = len(state.character_descriptions) > 0 and char_stems.issubset(wav_stems)

    if resume and voice_samples_exist:
        if verbose:
            print(f"[STAGE 4] Skipping - voice samples already exist ({len(wav_files)} files found)")
        state.load_voice_map()
        if verbose:
            print(f"[STAGE 4] Loaded {len(state.voice_map)} voice mappings")
    else:
        if verbose:
            print(f"[STAGE 4] Generating voice samples...")

        num_characters = len(state.character_descriptions)
        with ProgressHandler(progress=None, use_tqdm=True, total=num_characters, desc="Generating voice samples") as handler:
            # Build fallback chain: try all cloning-capable engines if primary fails
            _all_engines = ["omni", "vibevoice", "vox", "moss", "echo-tts"]
            _fallback_engines = [e for e in _all_engines if e != tts_engine]

            result_msg, generated_voices = gen_voice_samples(
                descriptions=state.character_descriptions,
                output_dir=str(state.output_dir),
                device=device,
                verbose=verbose,
                progress=None,  # CLI mode, no gr.Progress
                seed_characters=seed_characters,
                voice_engine=voice_engine,
                validate=validate,
                tts_engine=tts_engine,
                use_chunkformer=use_chunkformer,
                seed_clone_fallback_engines=_fallback_engines,
            )

            if verbose:
                print(f"  {result_msg}")

            handler.update(num_characters, desc="Voice samples complete")
            state.load_voice_map()

    # Stage 5: Generate Audiobook with progress
    if verbose:
        print(f"[STAGE 5] Generating audiobook...")

    if verbose:
        print(f"  TTS engine: {tts_engine}")
        if turbo:
            print(f"  Warning: --turbo flag is reserved for future use")
        print(f"  Device: {device}")

    # Generate TTS for all chapters
    try:
        # Count chapters for progress
        num_chapters_to_process = len(state.chapters) if state.chapters else 0

        # merged_voices_map already includes seed voices (merged at load_voice_map time)
        merged_voices_map = dict(state.voice_map)

        status, processed = generate_audiobook_from_chapters(
            chapters=state.chapters,
            chapter_maps=state.chapter_maps,
            voices_map=merged_voices_map,
            output_dir=str(state.chapters_dir),
            device=device,
            tts_engine=tts_engine,
            turbo=turbo,
            verbose=verbose,
            debug_tts=debug_tts,
            progress=None,
            duplicate_replacement_map=duplicate_replacement_map,
            seed_voice_map=seed_voice_map,
            whisper_alt_gpu=whisper_alt_gpu,
            validate_clean=validate_clean,
            max_retries=max_retries,
            enable_postfix=enable_postfix,
            concurrency=concurrency,
            gpus=gpus,
            whisper_concurrency=whisper_concurrency,
            whisper_fast=whisper_fast)

        if verbose:
            print(f"  {status}")

        # MP3 files are created during generate_audiobook_from_chapters
        mp3_files = sorted(glob.glob(str(state.chapters_dir / "chapter_*.mp3")), key=natural_sort_key)

        if verbose:
            print(f"[STAGE 5] Generated {len(mp3_files)} chapter MP3 files")

        # Assemble into single .m4b
        if mp3_files:
            book_name = os.path.splitext(os.path.basename(epub_path))[0] if epub_path else os.path.basename(state.chapters_dir)
            m4b_path = assemble_audiobook_m4b(str(state.chapters_dir), verbose=verbose, book_name=book_name)
            if m4b_path:
                return f"Audiobook generation complete! Generated {len(mp3_files)} chapter MP3 files and {m4b_path}."
            return f"Audiobook generation complete! Generated {len(mp3_files)} chapter MP3 files (m4b assembly failed)."
        return f"Audiobook generation complete! Generated {len(mp3_files)} chapter MP3 files."

    except Exception as e:
        error_msg = f"Error in Stage 5: {str(e)}\n{traceback.format_exc()}"
        if verbose:
            print(error_msg)
        return error_msg


# ============================================================================
# GRADIO INTERFACE
# ============================================================================


def create_gradio_interface(output_dir: str = "chapters", api_key: str = None,
                            llm_port: str = None, gradio_port: int = None,
                            num_attempts: int = 2, max_chapters: int = 10,
                            seed_voice_map: str = None, epub_file: str = None,
                            saved_temp_dir: str = None, tts_engine: str = None,
                            voice_engine: str = None, verbose: bool = False) -> None:
    """Create and launch the Gradio interface for the audiobook pipeline.

    This function launches the Gradio interface imported from the package's
    gradio_ui module. The actual UI definition is in gradio_ui.py.

    Args:
        output_dir: Output directory for generated files
        api_key: LLM API key
        llm_port: Port for LLM endpoint (e.g., LM Studio)
        gradio_port: Port for Gradio web interface
        num_attempts: Number of LLM attempts
        max_chapters: Max chapters to process
        seed_voice_map: Path to existing voices_map.json to seed voices
        epub_file: Path to EPUB file to pre-load in the interface
        saved_temp_dir: Optional path to a saved temp directory to restore from
        tts_engine: TTS engine to use ('vibevoice', 'moss', 'echo-tts', 'omni', 'vox')
    """
    # Set TTS_ENGINE environment variable for Gradio UI
    if tts_engine:
        os.environ['TTS_ENGINE'] = tts_engine
    try:
        from .gradio_ui import create_interface, cleanup_temp_dir
        import gradio as gr
        import shutil

        # Use provided LLM port or default
        effective_llm_port = llm_port or str(LLM_SETTINGS["port"])

        # Use provided gradio port or default from config
        effective_gradio_port = gradio_port if gradio_port is not None else AUDIO_SETTINGS.get("gradio_port", 7860)

        # Resolve seed_voice_map to absolute path if provided
        seed_voice_map_path = None
        if seed_voice_map:
            seed_voice_map_path = os.path.abspath(seed_voice_map)
            if not os.path.exists(seed_voice_map_path):
                print(f"Warning: Seed voice map file not found: {seed_voice_map_path}")
            else:
                # Set environment variable for MCP tools to access
                os.environ['SEED_VOICE_MAP'] = seed_voice_map_path

        # Handle EPUB file - copy to output_dir if provided for Gradio to access
        epub_path_default = None
        if epub_file:
            epub_path_default = os.path.abspath(epub_file)
            if not os.path.exists(epub_path_default):
                print(f"Error: EPUB file not found: {epub_path_default}")
                sys.exit(1)
            print(f"Pre-loading EPUB: {epub_path_default}")

        demo = create_interface(
            api_key_default=api_key or DEFAULTS.get("api_key", "lm-studio"),
            port_default=effective_llm_port,
            num_attempts_default=num_attempts,
            max_chapters_default=max_chapters,
            seed_voice_map_default=seed_voice_map_path,
            epub_path_default=epub_path_default,
            saved_temp_dir=saved_temp_dir,
            tts_engine_default=tts_engine,
            voice_engine_default=voice_engine,
            verbose=verbose
        )

        try:
            demo.launch(share=False, theme=gr.themes.Soft(), server_port=effective_gradio_port, server_name="0.0.0.0", mcp_server=True)
        except KeyboardInterrupt:
            print("\n\nGradio server stopped by user (Ctrl-C)")
            # Offer to save temp directory before cleanup
            if saved_temp_dir:
                print(f"\nTemp directory: {saved_temp_dir}")
                print("\nOptions before cleanup:")
                print("  's' - Save temp directory as zip archive for later resume")
                print("  'd' - Discard and clean up temp directory")

                try:
                    choice = input("\nPress 's' to save or 'd' to discard: ").strip().lower()
                    if choice == 's':
                        saved_path = save_temp_dir(saved_temp_dir)
                        print(f"\nSaved to: {saved_path}")
                        print("To resume later: --resume_from " + saved_path)
                    elif choice == 'd':
                        print("\nDiscarding temp directory...")
                    else:
                        print(f"\nUnknown choice '{choice}'. Defaulting to discard.")
                except (EOFError, KeyboardInterrupt):
                    print("\nNo input received. Discarding...")

                # Clean up after offering save option
                cleanup_temp_dir()
            else:
                print("No temp directory to clean up.")

    except ImportError as e:
        print(f"Error: Could not import gradio_ui module")
        print(f"Make sure the module is in place: {e}")


def main():
    """CLI entry point for the audiobook generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Audiobook Generator - Convert EPUB to Audiobook")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio interface")
    parser.add_argument("--output-dir", default="chapters", help="Output directory for generated files")
    parser.add_argument("--api-key", default=None, help="LLM API key")
    parser.add_argument("--port", dest="llm_port", default=None, help="Port for LLM endpoint (e.g., LM Studio)")
    parser.add_argument("--gradio-port", type=int, default=None, help="Port for Gradio web interface")
    parser.add_argument("--num-attempts", type=int, default=2, help="Number of LLM attempts")
    parser.add_argument("--max-chapters", type=int, default=None, help="Maximum chapters to process (default: all)")
    parser.add_argument("--seed-voice-map", help="Path to existing voices_map.json to seed voices")
    parser.add_argument("epub_file", nargs="?", help="Path to EPUB file to process")
    parser.add_argument("--saved-temp-dir", help="Path to saved temp directory to restore from")
    parser.add_argument("--tts-engine", choices=["vibevoice", "moss", "echo-tts", "omni", "vox", "dramabox"], help="TTS engine to use")
    parser.add_argument("--model", default=None, help="LLM model name (e.g., coder-model)")
    parser.add_argument("--voice-engine", choices=["omni", "vox", "dramabox"], default="omni", help="Voice engine for character descriptions")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--resume", nargs="?", const=True, default=None, metavar="DIR",
                        help="Resume from existing output directory (use --output-dir or specify DIR)")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent lines to process (default: 1)")
    parser.add_argument("--whisper-cpu", action="store_true", help="Run Whisper validation on CPU (frees GPU for TTS)")
    parser.add_argument("--whisper-concurrency", type=int, default=1, help="Number of concurrent Whisper models for validation (default: 1)")
    parser.add_argument("--whisper-fast", action="store_true", help="Use faster Whisper settings (medium model, beam_size=3)")
    parser.add_argument("--gpus", nargs="+", default=None, help="GPU devices to use (e.g., --gpus cuda:0 cuda:1)")
    parser.add_argument("--use-chunkformer", action="store_true", help="Enable ChunkFormer voice validation (gender/emotion/dialect/age classification)")

    args = parser.parse_args()

    if args.epub_file is None and not args.gradio and args.saved_temp_dir is None and not args.resume:
        parser.print_help()
        print("\nError: Missing required argument. Provide an EPUB file, use --gradio, --resume, or --saved-temp-dir.")
        print("\nExamples:")
        print("  audiobook-interface book.epub --tts-engine omni")
        print("  audiobook-interface --gradio --tts-engine omni")
        print("  audiobook-interface --resume --output-dir voice_test/warbreaker --tts-engine omni")
        print()
        print("Performance options:")
        print("  # 1 worker, sequential (default)")
        print("  audiobook-interface book.epub")
        print()
        print("  # 4 concurrent lines, 1 GPU (thread pool only)")
        print("  audiobook-interface book.epub --concurrency 4 --whisper-cpu")
        print()
        print("  # 2 workers on 1 GPU (not supported — each worker needs its own GPU)")
        print("  # Use --concurrency instead to overlap TTS+validation on 1 GPU")
        print()
        print("  # 2 workers on 2 GPUs, 2 concurrent lines per GPU")
        print("  audiobook-interface book.epub --gpus cuda:0 cuda:1 --concurrency 2 --whisper-cpu")
        print()
        print("  # 4 workers on 4 GPUs, 2 concurrent lines per GPU")
        print("  audiobook-interface book.epub --gpus cuda:0 cuda:1 cuda:2 cuda:3 --concurrency 2 --whisper-cpu")
        sys.exit(1)

    # Resolve resume directory: --resume DIR, --resume --output-dir DIR, or error
    if args.resume is not None:
        if args.resume is True:
            # --resume flag only, use --output-dir
            if args.output_dir == "chapters":
                parser.print_help()
                print("\nError: --resume requires a directory. Use --resume DIR or --resume --output-dir DIR.")
                print("\nExample:")
                print("  audiobook-interface --resume voice_test/warbreaker --tts-engine omni")
                sys.exit(1)
        else:
            # --resume DIR, override output_dir
            if args.output_dir != "chapters":
                print(f"\nWarning: --resume DIR overrides --output-dir. Using {args.resume} instead of {args.output_dir}.")
            args.output_dir = args.resume

    # Validate --resume + --gradio conflict
    if args.resume is not None and args.gradio:
        print("\nError: --resume cannot be used with --gradio.")
        print("  Use the Gradio UI to resume, or run --resume without --gradio.")
        sys.exit(1)

    # Validate --resume + --saved-temp-dir conflict
    if args.resume is not None and args.saved_temp_dir is not None:
        print("\nError: --resume cannot be used with --saved-temp-dir.")
        print("  Use one or the other to specify a directory to resume from.")
        sys.exit(1)

    # Validate resume directory exists
    if args.resume is not None:
        resume_dir = Path(args.output_dir)
        if not resume_dir.exists():
            print(f"\nError: Resume directory does not exist: {resume_dir}")
            sys.exit(1)
        if not resume_dir.is_dir():
            print(f"\nError: Resume path is not a directory: {resume_dir}")
            sys.exit(1)
        # Check for expected files
        chapter_files = list(resume_dir.glob("chapter_*.txt"))
        map_files = list(resume_dir.glob("*.map.json"))
        voice_map = resume_dir / "voices_map.json"
        if not chapter_files:
            print(f"\nError: No chapter files found in {resume_dir}. Cannot resume.")
            print("  A valid resume directory must contain chapter_*.txt files.")
            sys.exit(1)
        if not map_files:
            print(f"\nWarning: No speaker map files found in {resume_dir}. Will regenerate from scratch.")
        if not voice_map.exists():
            print(f"\nWarning: voices_map.json not found in {resume_dir}. Will regenerate voices.")

        # Warn if EPUB is also provided (it will be ignored)
        if args.epub_file is not None:
            print(f"\nWarning: --resume ignores EPUB file. Chapters will be loaded from {resume_dir}.")

    # Validate --output-dir without --resume (warn if existing chapter files found)
    if args.resume is None and args.epub_file is not None:
        output_path = Path(args.output_dir)
        if output_path.exists() and list(output_path.glob("chapter_*.txt")):
            print(f"\nWarning: Output directory already contains chapter files: {output_path}")
            print("  Existing files will be skipped. Use --resume to explicitly continue, or specify a different --output-dir.")

    # Validate --gpus
    if args.gpus:
        import torch
        if not torch.cuda.is_available():
            print(f"\nError: CUDA not available. Cannot use --gpus.")
            sys.exit(1)
        for gpu in args.gpus:
            if not gpu.startswith("cuda:"):
                print(f"\nError: Invalid GPU device '{gpu}'. Must be 'cuda:N' (e.g., cuda:0, cuda:1).")
                sys.exit(1)
            gpu_idx = int(gpu.split(":")[1])
            if gpu_idx >= torch.cuda.device_count():
                print(f"\nError: GPU {gpu} not found. This system has {torch.cuda.device_count()} GPU(s) (cuda:0 to cuda:{torch.cuda.device_count()-1}).")
                sys.exit(1)
        # Check for duplicates
        if len(args.gpus) != len(set(args.gpus)):
            print(f"\nError: Duplicate GPU devices specified: {args.gpus}")
            sys.exit(1)

    # Validate --concurrency
    if args.concurrency < 1:
        print(f"\nError: --concurrency must be >= 1, got {args.concurrency}")
        sys.exit(1)

    # Validate --tts-engine with --gpus
    if args.gpus and len(args.gpus) > 1 and not args.tts_engine:
        print(f"\nWarning: --gpus specified without --tts-engine. Using default: {AUDIO_SETTINGS['default_tts_engine']}")

    if args.gradio:
        create_gradio_interface(
            output_dir=args.output_dir,
            api_key=args.api_key,
            llm_port=args.llm_port,
            gradio_port=args.gradio_port,
            num_attempts=args.num_attempts,
            max_chapters=args.max_chapters,
            seed_voice_map=args.seed_voice_map,
            epub_file=args.epub_file,
            saved_temp_dir=args.saved_temp_dir,
            tts_engine=args.tts_engine,
            voice_engine=args.voice_engine,
            verbose=args.verbose
        )
    else:
        # Non-interactive pipeline run
        import json
        import glob
        import shutil
        import torch

        from .parse_chapter import parse_epub_to_chapters, load_chapters_from_txt
        from .llm_label_speakers import label_speakers
        from .llm_describe_character import describe_characters as describe_chars
        from .generate_voice_samples import generate_voice_samples as gen_voice_samples
        from .config import DEFAULTS, LLM_SETTINGS
        from .utils import (
            get_chapters_dir, get_temp_dir, cleanup_temp_dir,
            natural_sort_key, get_character_wav_file, load_seed_characters,
            count_lines_per_character, ProgressHandler,
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.model:
            LLM_SETTINGS["default_model"] = args.model
        if args.api_key:
            LLM_SETTINGS["api_key"] = args.api_key
        if args.llm_port:
            LLM_SETTINGS["port"] = int(args.llm_port)

        device = AUDIO_SETTINGS["default_device"] if torch.cuda.is_available() else "cpu"

        if args.resume:
            # Use run_full_pipeline for resume (has full stage-skipping logic)
            result = run_full_pipeline(
                epub_path=args.epub_file,
                output_dir=str(output_dir),
                max_chapters=args.max_chapters,
                verbose=args.verbose,
                api_key=args.api_key,
                llm_port=args.llm_port,
                voice_engine=args.voice_engine,
                tts_engine=args.tts_engine,
                device=device,
                seed_voice_map=args.seed_voice_map,
                num_llm_attempts=args.num_attempts,
                resume=True,
                whisper_cpu=args.whisper_cpu,
                concurrency=args.concurrency,
                gpus=args.gpus,
                whisper_concurrency=args.whisper_concurrency,
                whisper_fast=args.whisper_fast,
                use_chunkformer=args.use_chunkformer,
            )
            print(result)
        else:
            # Stage 1: Parse EPUB
            print(f"=== Stage 1: Parsing EPUB {args.epub_file} ===")
            chapters = parse_epub_to_chapters(args.epub_file, max_chapters=args.max_chapters)
            if not chapters:
                print("Error: No chapters found in EPUB file.")
                sys.exit(1)

            for i, chapter in enumerate(chapters):
                output_file = output_dir / f"chapter_{i}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    for cobj in chapter:
                        f.write(f"Line {cobj.line_num}: ")
                        if cobj.has_quotes:
                            f.write('"')
                        f.write(cobj.text)
                        if cobj.has_quotes:
                            f.write('"')
                        f.write("\n")
            print(f"Parsed {len(chapters)} chapters")

            # Stage 2: Label speakers
            print("=== Stage 2: Labeling speakers ===")
            all_characters = set()
            for i in range(len(chapters)):
                chapter_file = output_dir / f"chapter_{i}.txt"
                result_msg, char_map, line_map = label_speakers(
                    txt_file=chapter_file,
                    num_attempts=args.num_attempts,
                    verbose=args.verbose,
                    seed_characters=load_seed_characters(args.seed_voice_map),
                )
                if char_map:
                    all_characters.update(char_map.values())

            # Stage 3: Describe characters
            print("=== Stage 3: Describing characters ===")
            result_msg, descriptions = describe_chars(
                output_dir=str(output_dir),
                chapters_dir=str(output_dir),
                verbose=args.verbose,
                seed_characters=load_seed_characters(args.seed_voice_map),
                progress_callback=None,
                voice_engine=args.voice_engine,
            )
            print(result_msg)

            # Stage 4: Generate voice samples
            print("=== Stage 4: Generating voice samples ===")
            _all_engines = ["omni", "vibevoice", "vox", "moss", "echo-tts"]
            _fallback_engines = [e for e in _all_engines if e != args.tts_engine]

            result_msg, generated = gen_voice_samples(
                descriptions=descriptions,
                output_dir=str(output_dir),
                verbose=args.verbose,
                progress=None,
                seed_characters=load_seed_characters(args.seed_voice_map),
                voice_engine=args.tts_engine,
                validate=False,
                use_chunkformer=args.use_chunkformer,
                seed_clone_fallback_engines=_fallback_engines,
            )
            print(result_msg)

            # Stage 5: Generate audiobook
            print("=== Stage 5: Generating audiobook ===")
            chapter_maps = {}
            for i in range(len(chapters)):
                map_file = output_dir / f"chapter_{i}.map.json"
                if map_file.exists():
                    with open(map_file) as f:
                        chapter_maps[i] = json.load(f)

            voices_map = {}
            for char in descriptions:
                wav_path = get_character_wav_file(char, output_dir)
                if wav_path and Path(wav_path).exists():
                    voices_map[char] = Path(wav_path).name

            status, processed = generate_audiobook_from_chapters(
                chapters=chapters,
                chapter_maps=chapter_maps,
                voices_map=voices_map,
                output_dir=str(output_dir),
                device=device,
                tts_engine=args.tts_engine,
                cfg_scale=DEFAULTS["cfg_scale"],
                max_chapters=args.max_chapters,
                verbose=args.verbose,
                concurrency=args.concurrency,
                whisper_cpu=args.whisper_cpu,
                whisper_concurrency=args.whisper_concurrency,
                whisper_fast=args.whisper_fast,
                gpus=args.gpus,
            )
            print(status)
            print(f"Done! Generated {processed} chapters.")


if __name__ == "__main__":
    main()



