#!/usr/bin/env python3
"""
Audiobook Generator - Module to generate audiobook audio from chapters.

This module provides:
- CLI entry point for sequential pipeline execution
- Gradio interface via --gradio flag
- Stage 5: TTS audio generation from chapter maps and voice samples

Usage:
    # Run full pipeline via CLI
    python audiobook_generator.py <epub_file> [--output-dir] [--verbose]

    # Launch Gradio interface
    python audiobook_generator.py --gradio [--api-key KEY] [--llm-port PORT] [--gradio-port PORT]
"""

import argparse
import sys
import os
import time
import json
import re
import glob
import gc
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import torch
from scipy.io import wavfile
import numpy as np

# Helper to check if flash-attn is available
def _get_attn_implementation() -> Optional[str]:
    """Return flash_attention_2 if available, otherwise None."""
    try:
        import flash_attn
        return "flash_attention_2"
    except ImportError:
        return None

# Import config for default values
from config import DEFAULTS, LLM_SETTINGS, AUDIO_SETTINGS

# Text to speech generation - imports moved to setup_tts_engine() for lazy loading
TTS_ENGINE = os.environ.get('TTS_ENGINE', AUDIO_SETTINGS["default_tts_engine"])

# Voice to Text for validation
from difflib import SequenceMatcher
from faster_whisper import WhisperModel

# combine audio files
import pydub
from pydub import AudioSegment

import pandas as pd

# consistent seeding
from transformers import set_seed

# Import modular stage functions - using clean public interfaces
import parse_chapter
from llm_label_speakers import label_speakers  # Clean public function
from llm_describe_character import describe_characters  # Clean public function
from generate_voice_samples import generate_voice_samples as gen_voice_samples

# Import shared utilities for consistent temp directory handling
from utils import (
    get_chapters_dir,
    get_temp_dir,
    cleanup_temp_dir,
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
)

# Import VoiceMapper for centralized TTS management
from voice_mapper import VoiceMapper


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def setup_validation_model(device: str) -> WhisperModel:
    """Setup the Whisper validation model for audio transcription.

    Args:
        device: Device to run the model on (e.g., 'cuda', 'cuda:0', 'cpu')

    Returns:
        WhisperModel instance for audio validation
    """
    model_name = DEFAULTS["validation_model_name"]
    return WhisperModel(model_name, device=device, compute_type="float16")


def get_non_silent_audio_from_wavs(wav_filepath_list, min_silence_len=1250, silence_thresh=-60):
    """Remove silent audio from list of wave filepaths of wavs together. Return AudioSegment."""
    all_audio_segments = None
    for wav in wav_filepath_list:
        raw_audio_segment = pydub.AudioSegment.from_wav(wav)
        # remove silence
        this_audio_segment = pydub.AudioSegment.empty()
        for (start_time, end_time) in pydub.silence.detect_nonsilent(raw_audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh):
            this_audio_segment += raw_audio_segment[start_time:end_time]
        if all_audio_segments is None:
            all_audio_segments = this_audio_segment
        else:
            all_audio_segments = all_audio_segments + this_audio_segment
    return all_audio_segments


def color_word(word, score):
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


def score_strings_pop(i_str, d_str, lookahead=5, postfix="and also with you"):
    # Ensure lookahead is non-negative
    lookahead = max(0, lookahead)
    prev_undetected = False
    results = []
    input_tokens = i_str.split(" ")
    detected_tokens = d_str.split(" ")
    diff_list = []
    for i, i_tok in enumerate(input_tokens):
        if i_tok in diff_list:
            detected = True
            this_idx = diff_list.index(i_tok)
            detected_tokens = diff_list[this_idx+1:] + detected_tokens
            diff_list = diff_list[:this_idx]
        else:
            detected = False
            if prev_undetected and len(diff_list) > 0:
                diff_list.pop(0)
            else:
                diff_list = []

            if detected_tokens:
                n = max(min(lookahead, len(detected_tokens) - len(diff_list)), 0)
                for j in range(n):
                    d_tok = detected_tokens.pop(0)
                    diff_list.append(d_tok)
                    if i_tok in diff_list:
                        detected = True
                        break
                if not detected:
                    prev_undetected = True

        diff_str = " ".join(diff_list)
        results.append((i, i_tok, diff_str, detected, " ".join(detected_tokens[:lookahead])))
    df_temp = pd.DataFrame(results, columns=["i", "i_tok", "diff", "found", "next_tokens"])
    last_valid_token_index = df_temp[df_temp["found"] == True]["i"].max()
    last_valid_token = df_temp[df_temp["i"] == last_valid_token_index]["i_tok"]
    if len(last_valid_token.values) == 0:
        return 0, None
    else:
        return float(df_temp["found"].mean()) - 0.5 * (postfix not in d_str[-len(postfix):]), last_valid_token.values[0]


# ============================================================================
# STAGE 5: TTS AUDIO GENERATION
# ============================================================================


def generate_tts_for_line(
    chapter_idx: int,
    line_idx: int,
    text: str,
    voice_name: str,
    tts_model,
    processor,
    voice_mapper: VoiceMapper,
    device: str,
    tts_engine: str,
    cfg_scale: float,
    output_dir: str,
    short_text_postfix: str = "and also with you?",
    validation_model = None,
    verbose: bool = False,
    voice_path: str = None,
):
    """Generate TTS audio for a single line.

    Args:
        chapter_idx: Chapter index
        line_idx: Line index within chapter
        text: Text to synthesize
        voice_name: Name of the voice/character
        tts_model: Initialized TTS model
        processor: TTS processor
        voice_mapper: VoiceMapper instance
        device: Device to run on
        tts_engine: 'kugelaudio', 'vibevoice', or 'moss'
        voice_path: Optional path to voice sample file. If not provided, uses voice_mapper.
        cfg_scale: CFG scale value
        output_dir: Output directory for audio files
        short_text_postfix: Postfix for validation
        validation_model: Faster-whisper model for validation
        verbose: Print verbose output
        Returns:
        Tuple of (success: bool, ratio: float)
    """
    end_characters = ["?", ".", "-", ";", ",", "!"]

    full_script = str(text[0].upper() + text[1:])
    full_script = re.sub(r"(\s\.)+", r".", full_script)

    # Only add short_text_postfix when validation_model is provided
    # The validation loop is required to detect and dynamically remove the postfix
    postfix_detect_token = None
    if short_text_postfix:
        full_script = full_script + (" " if full_script[-1] in end_characters else ". ") + short_text_postfix
        postfix_detect_token = distill_string(short_text_postfix.strip().split(" ")[0]) # first word, ideal separator

    ratio = 0.0
    max_ratio = float('-inf')  # Initialize to lowest possible value to ensure first attempt is always saved
    retries = 0
    input_string = distill_string(full_script)
    
    set_seed(42)
    while ratio < 0.85 and retries < 5:
        set_seed(42 + retries)
        inputs = None
        outputs = None
        if tts_engine == 'kugelaudio':
            # Use KugelAudio processor with voice prompt
            # Use provided voice_path if available, otherwise look up via voice_mapper
            if voice_path is None:
                voice_path = voice_mapper.get_voice_path(voice_name)
            inputs = processor(
                text=full_script,
                voice_prompt=voice_path,
                padding=True,
                return_tensors="pt",
            )
        elif tts_engine == 'vibevoice':
            # Use VibeVoice processor with voice_samples
            if voice_path is None:
                voice_path = voice_mapper.get_voice_path(voice_name)
            inputs = processor(
                text=["Speaker 1: ? " + full_script],
                voice_samples=[voice_path],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
        elif tts_engine == 'moss':
            # MOSS-TTS uses reference audio for zero-shot cloning
            if voice_path is None:
                voice_path = voice_mapper.get_voice_path(voice_name)
            if voice_path is None:
                raise Exception(f"No voice path found for '{voice_name}'")
            # MOSS-TTS will be handled in the generation block below
            pass

        output_path = os.path.join(output_dir, f"chapter_{str(chapter_idx).zfill(2)}.{str(line_idx).zfill(4)}.tmp.wav")

        if tts_engine == 'moss':
            # MOSS-TTS zero-shot voice cloning
            import torchaudio

            if voice_path is None:
                voice_path = voice_mapper.get_voice_path(voice_name)
            if voice_path is None:
                raise Exception(f"No voice path found for '{voice_name}'")

            # Build conversation with reference audio
            conversations = [
                processor.build_user_message(text=full_script, reference=[voice_path])
            ]

            # Prepare batch for generation
            with torch.no_grad():
                batch = processor(conversations, mode="generation")
                outputs = tts_model.generate(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    max_new_tokens=DEFAULTS["max_new_tokens"],
                    # MOSS-TTS hyperparameters
                    audio_temperature=DEFAULTS["moss_audio_temperature"],
                    audio_top_p=DEFAULTS["moss_audio_top_p"],
                    audio_top_k=DEFAULTS["moss_audio_top_k"],
                    audio_repetition_penalty=DEFAULTS["moss_audio_repetition_penalty"],
                )

                # Decode output
                message = processor.decode(outputs)[0]
                audio = message.audio_codes_list[0]
                sr = processor.model_config.sampling_rate

            # Save audio
            if audio is not None and audio.numel() > 0:
                torchaudio.save(output_path, audio.unsqueeze(0), sr)
            else:
                raise Exception("Empty MOSS-TTS generation for chapter text")

        else:
            # KugelAudio and VibeVoice use processor inputs
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)

            outputs = tts_model.generate(
                **inputs,
                max_new_tokens=DEFAULTS["max_new_tokens"],
                cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer,
                do_sample=False,
                verbose=False
            )

            if tts_engine == 'kugelaudio':
                # KugelAudio returns audio directly in speech_outputs
                processor.save_audio(
                    outputs.speech_outputs[0],
                    output_path=output_path,
                )
            else:
                # VibeVoice also returns audio in speech_outputs
                processor.save_audio(
                    outputs.speech_outputs[0],
                    output_path=output_path,
                )

        del inputs
        del outputs

        gc.collect()
        torch.cuda.empty_cache()

        # send through a cleaning ML algo
        sample_rate, waveform = wavfile.read(output_path)
        wavfile.write(output_path, sample_rate, waveform)

        # Initialize variables for clipping/validation logic
        detected_string = ""
        segments = []
        start_times = []
        end_times = []
        last_valid_token = None

        if short_text_postfix and (validation_model is not None):
            # Use faster-whisper for validation with word timestamps for token-level matching
            segments_list, info = validation_model.transcribe(output_path, beam_size=5, word_timestamps=True)

            # Collect segments (words) and timestamps
            # Whisper word_timestamps=True should give individual words, but we need to split on spaces just in case
            segments = []
            start_times = []
            end_times = []
            for segment in segments_list:
                for word in segment.words:
                    segments.append(distill_string(word.word.strip()))
                    start_times.append(word.start)
                    end_times.append(word.end)

            detected_string = distill_string(" ".join(segments))
            ratio, last_valid_token = score_strings_pop(distill_string(input_string), detected_string, lookahead=5, postfix=distill_string(short_text_postfix))

        # Clipping based on postfix detection (only when validation is available)
        if short_text_postfix and (validation_model is not None):
            if (distill_string(short_text_postfix) in detected_string) and (postfix_detect_token in segments):
                if detected_string.startswith(distill_string(short_text_postfix)):
                    if verbose:
                        print("\nERROR: POSTFIX DETECTED BUT ONLY POSTFIX! -> Ratio 0\n")
                    ratio = 0
                else:
                    postfix_start_index = segments[::-1].index(postfix_detect_token)
                    if len(end_times) > postfix_start_index + 1:
                        clip_end2 = end_times[::-1][postfix_start_index + 1]
                    else:
                        clip_end2 = end_times[-1]
                    clip_end1 = start_times[::-1][postfix_start_index]
                    if verbose:
                        print(f"\nPOSTFIX DETECTED CLIPPING to {clip_end1} - {clip_end2}\n")
                    audio = pydub.AudioSegment.from_wav(output_path)
                    trimmed_audio = audio[0:((clip_end1 + clip_end2) * 500)]
                    trimmed_audio.export(output_path, format="wav")
            else:
                if ((last_valid_token is None) or (last_valid_token == "")):
                    if verbose:
                        print("\nERROR: POSTFIX UN-DETECTED and INVALID VALUES. SKIP.\n")
                elif last_valid_token in segments:
                    lastvalid_index = segments[::-1].index(last_valid_token)
                    clip_end1 = end_times[::-1][lastvalid_index]
                    if verbose:
                        print(f"\nERROR: POSTFIX UN-DETECTED LAST VALID CLIPPING TO {last_valid_token} {clip_end1}\n")
                    audio = pydub.AudioSegment.from_wav(output_path)
                    trimmed_audio = audio[0:(clip_end1 * 1000)]
                    trimmed_audio.export(output_path, format="wav")
                else:
                    if verbose:
                        print(f"\nERROR: POSTFIX UN-DETECTED even though last_valid_token = {last_valid_token} should be in {segments}\n")

        if ratio > max_ratio:
            max_ratio = ratio
            final_path = os.path.join(output_dir, f"chapter_{str(chapter_idx).zfill(2)}.{str(line_idx).zfill(4)}.wav")
            if os.path.exists(final_path):
                os.unlink(final_path)
            time.sleep(2)
            os.rename(output_path, final_path)

        retries += 1
    # Clean up temp file if it exists
    temp_path = os.path.join(output_dir, f"chapter_{str(chapter_idx).zfill(2)}.{str(line_idx).zfill(4)}.tmp.wav")
    if os.path.exists(temp_path):
        os.unlink(temp_path)

    return max_ratio >= 0.85, max_ratio


# ============================================================================
# MAIN GENERATION FUNCTIONS
# ============================================================================


def generate_audiobook_from_chapters(
    chapters: list,
    chapter_maps: Dict[int, Tuple[Dict, Dict]],
    voices_map: Dict[str, str],
    output_dir: str,
    device: str = "cuda",
    tts_engine: str = "kugelaudio",
    cfg_scale: float = 1.30,
    max_chapters: Optional[int] = None,
    verbose: bool = False,
    turbo: bool = False,
    progress: Optional[callable] = None,
    duplicate_replacement_map: Dict[str, str] = None,
    seed_voice_map: str = None
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
        tts_engine: 'kugelaudio', 'vibevoice', or 'moss'
        cfg_scale: CFG scale value
        max_chapters: Maximum number of chapters to process
        verbose: Print verbose output
        turbo: Use KugelAudio turbo model (kugel-1-turbo)
        progress: Optional progress callback (gr.Progress() for Gradio, None for CLI)
        duplicate_replacement_map: Dict mapping duplicate character names to canonical names
        seed_voice_map: Path to existing voices_map.json (if provided)

    Returns:
        Tuple of (status_message, chapters_processed)
    """
    if True:
        os.makedirs(output_dir, exist_ok=True)

        # Limit chapters if specified
        chapters_to_process = chapters[:max_chapters] if max_chapters else chapters

        # Create unified progress handler for both Gradio and CLI
        # Using context manager for proper cleanup
        with ProgressHandler(progress=progress, total=len(chapters_to_process), desc="Audiobook Generation") as progress_handler:
            # Setup validation model
            validation_model = setup_validation_model(device)
            # This avoids crashes from faster-whisper dependencies if validation_model is None
            # Initialize VoiceMapper with output_dir so it looks in the correct location
            voice_mapper = VoiceMapper(output_dir=output_dir, device=device, tts_engine=tts_engine)
            tts_model_read_chapters, processor, _, _ = voice_mapper.setup_tts_engine(turbo=turbo)
            for voice, relative_path in voices_map.items():
                voice_path = os.path.join(output_dir, relative_path)
                # Cache voice path in VoiceMapper
                voice_mapper.add_voice_path(voice, voice_path)
            short_text_postfix = DEFAULTS["short_text_postfix"]

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
                    line_to_character_map = {k: character_map[v] for k, v in line_map.items()}
                else:
                    # Fallback: assume narrator for all lines
                    line_to_character_map = {}

                # Assign speakers based on line map
                for cobj in chapter:
                    if cobj.has_quotes:
                        if cobj.line_num in line_map:
                            char_name = line_to_character_map.get(cobj.line_num, "narrator")
                            cobj.set_speaker(char_name)
                        else:
                            cobj.set_speaker("narrator")
                    else:
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

                # Generate TTS for each voice in this chapter
                for voice in voices_used:
                    progress_handler.update(0, desc=f"Processing Chapter {i} Voice {voice}")

                    for j, chapter_obj in enumerate(chapter):
                        progress_handler.update((j + 1)/ len(chapter), desc=f"Processing Chapter {i} Voice {voice} Line {j}")
                        if voice != chapter_obj.get_speaker():
                            continue
                        if j in already_generated:
                            if verbose:
                                print(f"Skipping chapter {i}.{j} (already generated)")
                            continue

                        # Get the voice path for this character
                        # Apply duplicate replacement map if available
                        canonical_voice = voice
                        if duplicate_replacement_map and voice in duplicate_replacement_map:
                            canonical_voice = duplicate_replacement_map[voice]
                            if verbose:
                                print(f"  Remapping duplicate voice '{voice}' -> '{canonical_voice}'")
                        if canonical_voice in voices_map:
                            # voices_map contains relative paths, prepend output_dir
                            voice_path = os.path.join(output_dir, voices_map[canonical_voice])
                        else:
                            voice_path = voice_mapper.get_voice_path(canonical_voice)

                        # Generate TTS for this line
                        success, ratio = generate_tts_for_line(
                            chapter_idx=i,
                            line_idx=j,
                            text=chapter_obj.text,
                            voice_name=voice,
                            tts_model=tts_model_read_chapters,
                            processor=processor,
                            voice_mapper=voice_mapper,
                            device=device,
                            tts_engine=tts_engine,
                            cfg_scale=cfg_scale,
                            output_dir=output_dir,
                            short_text_postfix=(short_text_postfix if (validation_model is not None) else None),
                            validation_model=validation_model,
                            verbose=verbose,
                            voice_path=voice_path
                        )
                        progress_handler.update((j + 1)/len(chapter), desc=f"Processing Chapter {i} Voice {voice} Line {j} Ratio {int(ratio * 100)}")
                        if verbose:
                            print(f"[LINE_PROGRESS] Chapter {i}, Line {j+1}/{len(chapter)}, Voice: {voice}, Ratio: {int(ratio * 100)}")

                # Assemble chapter MP3 from WAV files
                progress_handler.update(1, desc=f"Assembling Chapters")
                wav_files = sorted(glob.glob(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.*.wav")))
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

                # Clear cache periodically
                torch.cuda.empty_cache()

                processed += 1

            return f"Generated {processed} chapters successfully.", processed

    # except Exception as e:
    #     import traceback
    #     error_msg = f"Error generating audiobook: {str(e)}\n{traceback.format_exc()}"
    #     if verbose:
    #         print(error_msg)
    #     return error_msg, 0


# ============================================================================
# STATE MANAGEMENT (Internal to this module)
# ============================================================================


class PipelineState:
    """Manages state for the audiobook pipeline."""

    def __init__(self, output_dir: str):
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

    def load_chapter_maps(self):
        """Load all chapter map files from the chapters directory."""
        self.chapter_maps = {}
        # Only match map files with simple numeric names (chapter_X.map.json, not chapter_X.result.N.map.json)
        map_files = sorted([f for f in self.chapters_dir.glob("*.map.json")
                           if re.match(r"^chapter_\d+\.map\.json$", f.name)])

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
        from utils import get_characters_from_map_files
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
        chapter_files = sorted(self.chapters_dir.glob("chapter_*.txt"))
        if not chapter_files:
            return "initial"

        # Check for Stage 2 completion (map files)
        map_files = sorted(self.chapters_dir.glob("*.map.json"))
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
        mp3_files = sorted(self.chapters_dir.glob("chapter_*.mp3"))
        if not mp3_files:
            return "voice_samples_complete"

        return "audiobook_complete"

    def write_chapter_text_files(self, chapters):
        """Write chapter objects to text files."""
        from parse_chapter import write_chapters_to_txt
        return write_chapters_to_txt(chapters, str(self.chapters_dir))


# ============================================================================
# CLI ORCHESTRATION
# ============================================================================


def run_full_pipeline(epub_path: str, output_dir: str, max_chapters: int = None,
                      verbose: bool = False, api_key: str = None, llm_port: str = None,
                      tts_engine: str = "kugelaudio", turbo: bool = False,
                      device: str = "cuda", seed_voice_map: str = None,
                      num_llm_attempts: int = 2) -> str:
    """Run the full audiobook pipeline from EPUB to MP3.

    Args:
        epub_path: Path to the EPUB file
        output_dir: Output directory for all generated files
        max_chapters: Maximum number of chapters to process
        verbose: Print verbose output
        api_key: LLM API key for speaker labeling and character descriptions
        llm_port: LLM endpoint port (e.g., LM Studio)
        tts_engine: 'kugelaudio' or 'vibevoice'
        turbo: Use KugelAudio turbo model (kugel-1-turbo)
        device: CUDA device (e.g., 'cuda', 'cuda:1')
        seed_voice_map: Path to existing voices_map.json to seed voices
        num_llm_attempts: Number of LLM attempts for speaker labeling

    Returns:
        Status message
    """
    # Initialize state
    state = PipelineState(output_dir)

    # Load duplicate replacement map if available (from Stage 3)
    duplicate_replacement_map = {}
    replacement_map_file = os.path.join(output_dir, "chapters", "duplicate_replacement_map.json")
    if os.path.exists(replacement_map_file):
        duplicate_replacement_map = load_json(replacement_map_file)
        if verbose and duplicate_replacement_map:
            print(f"[DUPLICATE MAP] Loaded {len(duplicate_replacement_map)} replacements from duplicate_replacement_map.json")

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
            if verbose:
                print(f"[SEED] Loaded {len(seed_characters)} seeded characters from {seed_voice_map}")
        else:
            if verbose:
                print(f"[SEED] No characters found in seed file: {seed_voice_map}")

    # Store chapters in state for reuse (avoid re-parsing)
    # Stage 1: Parse EPUB with progress
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
                           if re.match(r"^chapter_\d+\.txt$", f.name)])
    num_chapters = len(chapter_files)

    with ProgressHandler(progress=None, use_tqdm=True, total=num_chapters, desc="Labeling speakers") as handler:
        for i, chapter_file in enumerate(chapter_files):
            handler.update((i + 1) / num_chapters, desc=f"Labeling chapter {i + 1}/{num_chapters}")

            result_msg, char_map, line_map = label_speakers(
                txt_file=str(chapter_file),
                api_key=api_key or DEFAULTS.get("api_key", "lm-studio"),
                port=llm_port or LLM_SETTINGS.get("port", "1234"),
                num_attempts=num_llm_attempts,
                verbose=verbose,
                seed_characters=seed_characters
            )

            if verbose:
                print(f"  {result_msg}")

    state.load_chapter_maps()
    state.get_characters()

    # Stage 3: Describe Characters with progress
    if verbose:
        print(f"[STAGE 3] Describing {len(state.characters)} characters...")

    with ProgressHandler(progress=None, use_tqdm=True, total=1, desc="Describing characters") as handler:
        result_msg, character_descriptions = describe_characters(
            output_dir=str(state.output_dir),
            chapters_dir=str(state.output_dir),
            api_key=api_key or DEFAULTS.get("api_key", "lm-studio"),
            port=llm_port or LLM_SETTINGS.get("port", "1234"),
            verbose=verbose,
            seed_characters=seed_characters,
            progress_callback=handler.update
        )

        if verbose:
            print(f"  {result_msg}")

        handler.update(1, desc="Character descriptions complete")
        state.load_character_descriptions()

    # Stage 4: Generate Voice Samples with progress
    if verbose:
        print(f"[STAGE 4] Generating voice samples...")

    num_characters = len(state.character_descriptions)
    with ProgressHandler(progress=None, use_tqdm=True, total=num_characters, desc="Generating voice samples") as handler:
        result_msg, generated_voices = gen_voice_samples(
            descriptions=state.character_descriptions,
            output_dir=str(state.output_dir),
            device=device,
            verbose=verbose,
            progress=None,  # CLI mode, no gr.Progress
            seed_characters=seed_characters,
            tts_engine=tts_engine
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
        if turbo and tts_engine == "kugelaudio":
            print(f"  Using turbo model (kugel-1-turbo)")
        elif turbo and tts_engine != "kugelaudio":
            print(f"  Warning: --turbo flag is only valid for kugelaudio engine, ignoring for {tts_engine}")
        print(f"  Device: {device}")

    # Generate TTS for all chapters
    try:
        # Count chapters for progress
        num_chapters_to_process = len(chapters) if chapters else 0

        status, processed = generate_audiobook_from_chapters(
            chapters=chapters,
            chapter_maps=state.chapter_maps,
            voices_map=state.voice_map,
            output_dir=str(state.chapters_dir),
            device=device,
            tts_engine=tts_engine,
            turbo=turbo,
            verbose=verbose,
            progress=None,
            duplicate_replacement_map=duplicate_replacement_map,
            seed_voice_map=seed_voice_map)

        if verbose:
            print(f"  {status}")

        # MP3 files are created during generate_audiobook_from_chapters
        mp3_files = sorted(glob.glob(str(state.chapters_dir / "chapter_*.mp3")))

        if verbose:
            print(f"[STAGE 5] Generated {len(mp3_files)} chapter MP3 files")

        return f"Audiobook generation complete! Generated {len(mp3_files)} chapter MP3 files."

    except Exception as e:
        import traceback
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
                            seed_voice_map: str = None) -> None:
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
    """
    try:
        from gradio_ui import create_interface, cleanup_temp_dir
        import gradio as gr

        # Use provided LLM port or default
        effective_llm_port = llm_port or LLM_SETTINGS.get("port", "1234")

        # Use provided gradio port or default from config
        effective_gradio_port = gradio_port if gradio_port is not None else AUDIO_SETTINGS.get("gradio_port", 7860)

        # Resolve seed_voice_map to absolute path if provided
        seed_voice_map_path = None
        if seed_voice_map:
            seed_voice_map_path = os.path.abspath(seed_voice_map)
            if not os.path.exists(seed_voice_map_path):
                print(f"Warning: Seed voice map file not found: {seed_voice_map_path}")

        demo = create_interface(
            api_key_default=api_key or DEFAULTS.get("api_key", "lm-studio"),
            port_default=effective_llm_port,
            num_attempts_default=num_attempts,
            max_chapters_default=max_chapters,
            seed_voice_map_default=seed_voice_map_path
        )

        demo.launch(share=False, theme=gr.themes.Soft(), server_port=effective_gradio_port, server_name="0.0.0.0")

    except ImportError as e:
        print(f"Error: Could not import gradio_ui module")
        print(f"Make sure the module is in place: {e}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Audiobook Generator - Parse EPUB and generate audiobook audio"
    )
    parser.add_argument("epub_file", nargs="?", help="Path to the EPUB file")
    parser.add_argument("-output_dir", default=None, help="Output directory for generated files (default: temp directory)")
    parser.add_argument("-max_chapters", type=int, help="Maximum number of chapters to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")
    parser.add_argument("-api_key", help="LLM API key for speaker labeling")
    parser.add_argument("-llm_port", default="1234", help="LLM endpoint port (for LM Studio)")
    parser.add_argument("-tts_engine", default="kugelaudio", choices=["kugelaudio", "vibevoice", "moss"],
                        help="TTS engine to use")
    parser.add_argument("--turbo", action="store_true", help="Use KugelAudio turbo model (kugel-1-turbo)")
    parser.add_argument("-device", default="cuda", help="CUDA device to use")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio interface instead of CLI")
    parser.add_argument("-num_llm_attempts", type=int, default=DEFAULTS["num_llm_attempts"], help="Number of LLM attempts for speaker labeling")
    parser.add_argument("-gradio_port", type=int, default=None, help="Port for Gradio web interface")
    parser.add_argument("-seed_voice_map", help="Path to existing voices_map.json to seed voices")

    args = parser.parse_args()
    print(args)
    if args.gradio:
        # Launch Gradio interface
        create_gradio_interface(
            output_dir=args.output_dir,
            api_key=args.api_key,
            llm_port=args.llm_port,
            gradio_port=args.gradio_port,
            num_attempts=args.num_llm_attempts,
            max_chapters=args.max_chapters or DEFAULTS.get("max_chapters", 10),
            seed_voice_map=args.seed_voice_map
        )
    else:
        # Run CLI pipeline
        if not args.epub_file:
            parser.print_help()
            print("\nError: EPUB file is required for CLI mode.")
            sys.exit(1)

        if not os.path.exists(args.epub_file):
            print(f"Error: EPUB file not found: {args.epub_file}")
            sys.exit(1)

        # Use temp directory by default, or user-specified output_dir if provided
        used_temp_dir = False
        if args.output_dir is None:
            output_dir = str(get_chapters_dir())
            used_temp_dir = True
            print(f"Using temporary directory: {get_temp_dir()}")
        else:
            output_dir = args.output_dir

        status = run_full_pipeline(
            epub_path=args.epub_file,
            output_dir=output_dir,
            max_chapters=args.max_chapters,
            verbose=args.verbose,
            api_key=args.api_key,
            llm_port=args.llm_port,
            tts_engine=args.tts_engine,
            turbo=args.turbo,
            device=args.device,
            seed_voice_map=args.seed_voice_map,
            num_llm_attempts=args.num_llm_attempts
        )

        print(status)

        # Copy MP3 files to ./chapters/ if we used a temporary directory
        if used_temp_dir:
            copy_mp3_files_to_chapters(output_dir)
            print(f"\nMP3 files have been copied to ./chapters/ directory.")
            print(f"Temporary directory will be cleaned up on exit.")


if __name__ == "__main__":
    main()
    input("Press any key to cleanup directory and close.")
