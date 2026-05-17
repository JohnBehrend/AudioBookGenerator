#!/usr/bin/env python3
"""
Gradio Interface for Audiobook Pipeline

A unified web interface for the 5-stage audiobook creation pipeline:
1. EPUB Parsing
2. LLM Speaker Labeling
3. Character Descriptions
4. Voice Sample Generation
5. Full Audiobook Generation

State machine pattern ensures each stage only runs when dependencies are met.
"""

import sys
import io
import re

# ============================================================================
# Suppress sox binary warning (sox Python package requires sox CLI binary)
# ============================================================================
# The sox package prints a warning to stderr when the sox binary is not found.
# We suppress this by temporarily redirecting stderr during the sox import.
# This must run BEFORE any imports that might trigger sox-related dependencies.

original_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    # Import sox here - any warning goes to our buffer
    import sox
except ImportError:
    pass
finally:
    sys.stderr = original_stderr

import gradio as gr
import os
import json
import glob
import shutil
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

# Import from package modules - using clean public interfaces
from . import parse_chapter
from .llm_label_speakers import label_speakers  # Clean public function
from .llm_describe_character import describe_characters as describe_chars  # Clean public function
from .generate_voice_samples import generate_voice_samples as gen_voice_samples
from .utils import (
    natural_sort_key,
    get_chapters_dir,
    get_temp_dir,
    cleanup_temp_dir,
    save_temp_dir,
    load_temp_dir,
    get_available_saved_audiobooks,
    ProgressHandler,
    copy_mp3_files_to_chapters,
    load_json_file,
    get_character_wav_file,
    load_seed_characters,
    get_chapter_map_files,
    parse_map_file,
    count_lines_per_character,
)
from .audiobook_generator import (
    PipelineState,
    generate_audiobook_from_chapters,
    get_non_silent_audio_from_wavs,
    VoiceMapper,
)
from .config import DEFAULTS, LLM_SETTINGS, AUDIO_SETTINGS, DEFAULT_EPUB_FILE


# ============================================================================
# CONSTANTS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Default UI values (imported from config.py for single source of truth)
DEFAULT_API_KEY = LLM_SETTINGS["api_key"]
DEFAULT_PORT = LLM_SETTINGS["port"]
DEFAULT_NUM_ATTEMPTS = DEFAULTS["num_llm_attempts"]
DEFAULT_MAX_CHAPTERS = DEFAULTS["max_chapters"]


# ============================================================================
# HELPER FUNCTIONS
def get_characters_descriptions_file() -> Optional[Path]:
    """Get the path to characters_descriptions.json in the temp directory."""
    chapters_dir = get_chapters_dir()
    if not chapters_dir:
        return None
    return chapters_dir / "characters_descriptions.json"


def get_duplicate_replacement_map_file() -> Optional[Path]:
    """Get the path to duplicate_replacement_map.json in the temp directory."""
    chapters_dir = get_chapters_dir()
    if not chapters_dir:
        return None
    return chapters_dir / "duplicate_replacement_map.json"


def get_description_metadata_file() -> Optional[Path]:
    """Get the path to description_metadata.json in the temp directory."""
    chapters_dir = get_chapters_dir()
    if not chapters_dir:
        return None
    return chapters_dir / "description_metadata.json"


def get_all_character_wav_files(chapters_dir: Path) -> Dict[str, str]:
    """Get all generated character WAV files."""
    wav_files = {}
    for base_dir in [chapters_dir, SCRIPT_DIR]:
        if base_dir and base_dir.exists():
            for wav_path in base_dir.glob("*.wav"):
                wav_name = wav_path.name
                if not wav_name.startswith("chapter_") and wav_name not in ("narrator.wav", "uploaded.epub"):
                    wav_files[wav_path.stem] = str(wav_path)
    return wav_files


# ============================================================================
# STAGE 1: EPUB PARSING
# ============================================================================


def parse_epub_to_file(
    epub_file, max_chapters: Optional[int], progress=gr.Progress()
) -> Tuple[str, PipelineState]:
    """Stage 1: Parse EPUB file into chapter text files.

    Returns:
        Tuple of (log_output, PipelineState instance)
    """
    if epub_file is None:
        return "Error: No EPUB file uploaded.", None

    chapters_dir = get_chapters_dir()
    if not chapters_dir:
        return "Error: Failed to create temporary directory.", None

    try:
        # Create PipelineState
        state = PipelineState(str(chapters_dir))

        # Check if EPUB is already parsed (resume from existing state)
        existing_chapter_files = sorted(chapters_dir.glob("chapter_*.txt"), key=natural_sort_key)
        if existing_chapter_files:
            # EPUB already parsed - preserve existing progress
            log_msg = f"=== Stage 1: EPUB already parsed ({len(existing_chapter_files)} chapters found) ===\n"
            log_msg += "Preserving existing state. To start fresh, use 'Load Temp' with a different directory.\n"
            state.chapters = parse_chapter.load_chapters_from_txt(str(chapters_dir), max_chapters=int(max_chapters) if max_chapters else None)
            state.pipeline_state = "epub_parsed"
            return log_msg, state

        # Clean up existing files in the chapters directory before starting fresh
        for item in chapters_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

        # Copy the EPUB file to temp directory for Stage 6
        epub_dest = chapters_dir / "uploaded.epub"
        shutil.copy2(epub_file.name, str(epub_dest))

        # Parse the EPUB with progress tracking
        progress(0, desc=f"Starting EPUB parsing... (temp: {chapters_dir.parent})")
        chapters = parse_chapter.parse_epub_to_chapters(
            epub_file.name,
            max_chapters=int(max_chapters) if max_chapters else None
        )

        if not chapters:
            return "Error: No chapters found in EPUB file.", None

        # Store parsed chapters in state for reuse (avoid re-parsing)
        state.chapters = chapters

        # Count total lines across all chapters for progress tracking
        total_lines = sum(len(chapter) for chapter in chapters)
        total_chapters = len(chapters)
        progress(0, desc=f"Parsing {total_chapters} chapters with {total_lines} lines... (temp: {chapters_dir.parent})")

        # Save each chapter as a text file with progress updates
        lines_processed = 0
        for i, chapter in enumerate(chapters):
            chapter_line_count = len(chapter)
            for j, cobj in enumerate(chapter):
                lines_processed += 1
                progress(
                    lines_processed / total_lines,
                    desc=f"Parsing chapter {i + 1}/{total_chapters}: line {lines_processed}/{total_lines}... (temp: {chapters_dir.parent})"
                )

            output_file = chapters_dir / f"chapter_{i}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                for cobj in chapter:
                    f.write(f"Line {cobj.line_num}: ")
                    if cobj.has_quotes:
                        f.write('"')
                    f.write(cobj.text)
                    if cobj.has_quotes:
                        f.write('"')
                    f.write("\n")

        progress(1.0, desc=f"Successfully parsed {total_chapters} chapters with {total_lines} lines. (temp: {chapters_dir.parent})")
        state.pipeline_state = "epub_parsed"
        return "=== Stage 1: EPUB Parsing Complete ===\n", state
    except Exception as e:
        log_output = f"Error parsing EPUB: {str(e)}"
        log_output += f"\n{traceback.format_exc()}"
        return log_output, None


# ============================================================================
# STAGE 2: LLM SPEAKER LABELING
# ============================================================================


def process_chapters_for_labels(
    api_key: str,
    port: str,
    num_attempts: int,
    pipeline_state: PipelineState,
    log_output: str,
    seed_voice_map: str = None,
    progress=gr.Progress()
) -> Tuple[str, PipelineState]:
    """Stage 2: Run LLM to label speakers in all chapters."""
    LLM_SETTINGS["api_key"] = api_key
    LLM_SETTINGS["port"] = int(port)

    if pipeline_state is None:
        progress(1.0, desc="Pipeline state not initialized. Please run Stage 1 (Parse EPUB) first.")
        log_output += "\nPipeline state not initialized. Please run Stage 1 (Parse EPUB) first."
        return log_output, None

    chapters_dir = get_chapters_dir()
    if not chapters_dir:
        progress(1.0, desc=f"Error: Chapters directory not initialized. (temp: {chapters_dir.parent})")
        log_output += "\nError: Chapters directory not initialized."
        return log_output, pipeline_state

    chapter_files = sorted([f for f in chapters_dir.glob("chapter_*.txt")
                           if re.match(r"^chapter_\d+\.txt$", f.name)],
                          key=natural_sort_key)

    if not chapter_files:
        progress(1.0, desc="No chapter files found. Please run Stage 1 (Parse EPUB) first.")
        log_output += "\nNo chapter files found. Please run Stage 1 (Parse EPUB) first."
        return log_output, pipeline_state

    # Check for existing map files (resume from partial completion)
    existing_map_files = get_chapter_map_files(chapters_dir)
    labeled_chapters = {int(f.name.replace("chapter_", "").replace(".map.json", ""))
                       for f in existing_map_files}

    num_chapters = len(chapter_files)
    num_labeled = len(labeled_chapters)

    if num_labeled > 0:
        log_output += f"\nFound {num_labeled} already labeled chapters (resume mode). Processing remaining..."
        log_output += f" (temp: {chapters_dir.parent})"

    log_output += f"\nProcessing {num_chapters} chapters with LLM... (temp: {chapters_dir.parent})"

    all_character_names = set()

    for i, chapter_file in enumerate(chapter_files):
        # Skip if already labeled
        if i in labeled_chapters:
            progress((i + 1) / num_chapters, desc=f"Skipping chapter {i + 1}/{num_chapters} (already labeled)... (temp: {chapters_dir.parent})")
            log_output += f"\nSkipping chapter {i + 1} (already labeled)"
            continue

        progress(i / num_chapters, desc=f"Labeling speakers in chapter {i + 1}/{num_chapters}... (temp: {chapters_dir.parent})")
        log_output += f"\nProcessing: {chapter_file}"

        try:
            # Import and call directly instead of subprocess
            result_msg, char_map, line_map = label_speakers(
                txt_file=chapter_file,
                num_attempts=num_attempts,
                verbose=False,
                seed_characters=load_seed_characters(seed_voice_map)
            )

            log_output += f"\n{result_msg}"

            # Extract character names from char_map (char_map is {key: "char_name"})
            if isinstance(char_map, dict):
                for char_name in char_map.values():
                    if isinstance(char_name, str):
                        all_character_names.add(char_name)

        except Exception as e:
            log_output += f"\nError processing {chapter_file}: {str(e)}"
            log_output += f"\n{traceback.format_exc()}"

    # Load chapter maps and characters into state
    pipeline_state.load_chapter_maps()
    pipeline_state.get_characters()
    pipeline_state.pipeline_state = "labels_complete"

    log_output += f"\n\nStage 2 complete! State: {pipeline_state.pipeline_state}"
    log_output += f"\nFound {len(pipeline_state.characters)} characters: {', '.join(pipeline_state.characters)}"

    progress(1.0, desc=f"LLM speaker labeling complete. (temp: {chapters_dir.parent})")
    return log_output, pipeline_state


# ============================================================================
# STAGE 3: CHARACTER DESCRIPTIONS
# ============================================================================


def describe_characters_ui(
    api_key: str,
    port: str,
    pipeline_state: PipelineState,
    log_output: str,
    seed_voice_map: str = None,
    voice_engine: str = "omni",
    progress=gr.Progress()
) -> Tuple[str, PipelineState]:
    """Stage 3: Use LLM to describe characters."""
    LLM_SETTINGS["api_key"] = api_key
    LLM_SETTINGS["port"] = int(port)

    if pipeline_state is None:
        progress(1.0, desc="Pipeline state not initialized. Please run Stage 2 (Label Speakers) first.")
        log_output += "\nPipeline state not initialized. Please run Stage 2 (Label Speakers) first."
        return log_output, None

    # Update pipeline state's voice_engine setting
    pipeline_state.voice_engine = voice_engine

    progress(0, desc="Starting character description generation...")
    log_output += f"\n\nGenerating character descriptions with voice engine '{voice_engine}'..."

    try:
        chapters_dir = get_chapters_dir()
        if not chapters_dir:
            log_output += "\nError: Chapters directory not initialized."
            return log_output, pipeline_state

        # Check if map files exist
        map_files = glob.glob(str(chapters_dir / "*.map.json"))
        if not map_files:
            log_output += "\nNo .map.json files found. Please run Stage 2 (Label Speakers) first."
            return log_output, pipeline_state

        # Check if descriptions already exist (resume mode)
        descriptions_file = get_characters_descriptions_file()
        if descriptions_file and descriptions_file.exists():
            with open(descriptions_file, "r", encoding="utf-8") as f:
                existing_descriptions = json.load(f)
            if existing_descriptions:
                # Check if voice engine changed - if so, force regeneration
                metadata_file = get_description_metadata_file()
                old_voice_engine = "omni"  # Default for backwards compatibility
                if metadata_file and metadata_file.exists():
                    with open(metadata_file, "r", encoding="utf-8") as mf:
                        metadata = json.load(mf)
                        old_voice_engine = metadata.get("voice_engine", "omni")

                if old_voice_engine != voice_engine:
                    log_output += f"\nVoice engine changed from '{old_voice_engine}' to '{voice_engine}' - regenerating descriptions with new prompt format..."
                else:
                    log_output += f"\nFound existing character descriptions ({len(existing_descriptions)} characters) - preserving (resume mode)"
                    pipeline_state.character_descriptions = existing_descriptions
                    pipeline_state.pipeline_state = "characters_described"
                    log_output += f" State: {pipeline_state.pipeline_state}"
                    return log_output, pipeline_state

        # Get characters from existing state
        characters = pipeline_state.get_characters()
        num_characters = len(characters)

        if num_characters == 0:
            log_output += "\nNo characters found in map files. Please run Stage 2 first."
            return log_output, pipeline_state

        log_output += f"\nFound {num_characters} characters from map files. (temp: {chapters_dir.parent})"

        # Call describe_chars from llm_describe_character module with progress
        result_msg, character_descriptions = describe_chars(
            output_dir=str(chapters_dir),
            chapters_dir=str(chapters_dir),
            verbose=False,
            seed_characters=load_seed_characters(seed_voice_map),
            progress_callback=progress,
            voice_engine=voice_engine
        )

        log_output += f"\n{result_msg}"
        progress(1.0, desc="Character description generation complete.")

        # Save metadata to track which voice engine was used
        metadata_file = get_description_metadata_file()
        if metadata_file:
            with open(metadata_file, "w", encoding="utf-8") as mf:
                json.dump({"voice_engine": voice_engine}, mf)

        # Load and store character descriptions in state
        pipeline_state.load_character_descriptions()
        pipeline_state.pipeline_state = "characters_described"

        log_output += f" State: {pipeline_state.pipeline_state}"
        return log_output, pipeline_state

    except Exception as e:
        log_output += f"\nError describing characters: {str(e)}"
        log_output += f"\n{traceback.format_exc()}"
        return log_output, pipeline_state


# ============================================================================
# STAGE 4: VOICE SAMPLE GENERATION
# ============================================================================


def generate_voice_samples(
    pipeline_state: PipelineState,
    log_output: str,
    seed_voice_map: str = None,
    voice_engine: str = None,
    progress=gr.Progress()
) -> Tuple[str, PipelineState]:
    """Stage 4: Generate voice samples for each character."""
    if pipeline_state is None:
        progress(1.0, desc="Pipeline state not initialized. Please run Stage 3 (Describe Characters) first.")
        log_output += "\nPipeline state not initialized. Please run Stage 3 (Describe Characters) first."
        return log_output, None

    # Fall back to default seed_voice_map if not provided
    if not seed_voice_map:
        default_seed_path = os.environ.get('SEED_VOICE_MAP')
        if default_seed_path and os.path.exists(default_seed_path):
            seed_voice_map = default_seed_path

    log_output += "\n\nGenerating voice samples..."

    try:
        progress(0, desc="Starting voice sample generation...")
        chapters_dir = get_chapters_dir()
        if not chapters_dir:
            log_output += "\nError: Chapters directory not initialized."
            return log_output, pipeline_state

        # Copy seed voice files to chapters directory if seed_voice_map is provided
        if seed_voice_map:
            seed_chars = load_seed_characters(seed_voice_map)
            if seed_chars:
                # Resolve voices_archive path relative to seed_voice_map location
                seed_map_path = Path(seed_voice_map) if isinstance(seed_voice_map, str) else Path(seed_voice_map.get("name", ""))
                voices_archive = seed_map_path.parent if seed_map_path.exists() else None
                if voices_archive:
                    for char_name, voice_file in seed_chars.items():
                        src = voices_archive / voice_file
                        dest = chapters_dir / voice_file
                        if src.exists() and not dest.exists():
                            shutil.copy2(src, dest)
                            log_output += f"\nCopied seed voice: {voice_file}"

        descriptions_file = get_characters_descriptions_file()
        if not descriptions_file or not descriptions_file.exists():
            log_output += "\ncharacters_descriptions.json not found. Please run Stage 3 (Describe Characters) first."
            return log_output, pipeline_state

        # Load descriptions to get count for progress
        with open(descriptions_file, "r", encoding="utf-8") as f:
            descriptions = json.load(f)

        num_characters = len(descriptions)
        if num_characters == 0:
            log_output += "\nNo characters found in descriptions. Please run Stage 3 (Describe Characters) first."
            return log_output, pipeline_state
        log_output += f"\nFound {num_characters} characters to process. (temp: {chapters_dir.parent})"

        # Use provided voice_engine or default to moss
        if voice_engine is None:
            voice_engine = "moss"

        # Call generate_voice_samples from package
        progress(0, desc=f"Generating voice samples for {num_characters} characters with TTS engine '{voice_engine}'... (temp: {chapters_dir.parent})")
        result_msg, generated_voices = gen_voice_samples(
            descriptions=descriptions,
            output_dir=str(chapters_dir),
            verbose=False,
            progress=progress,
            seed_characters=load_seed_characters(seed_voice_map),
            voice_engine=voice_engine,
            validate=False,
            nemotron_endpoint=None,
        )

        log_output += f"\n{result_msg}"
        progress(1.0, desc=f"Voice sample generation complete. (temp: {chapters_dir.parent})")

        # Load voice map into state
        pipeline_state.load_voice_map()
        pipeline_state.pipeline_state = "voice_samples_complete"
        log_output += f" State: {pipeline_state.pipeline_state}"
        return log_output, pipeline_state

    except Exception as e:
        log_output += f"\nError generating voice samples: {str(e)}"
        log_output += f"\n{traceback.format_exc()}"
        return log_output, pipeline_state


def regenerate_voice_sample(
    character_name: str,
    pipeline_state: PipelineState,
    log_output: str,
    progress=gr.Progress()
) -> Tuple[str, PipelineState, Optional[str]]:
    """Regenerate a single voice sample for a character."""
    log_output += f"\n\nRegenerating voice sample for: {character_name}"

    try:
        chapters_dir = get_chapters_dir()
        if not chapters_dir:
            log_output += "\nError: Chapters directory not initialized."
            return log_output, pipeline_state, None

        descriptions_file = get_characters_descriptions_file()
        if not descriptions_file or not descriptions_file.exists():
            log_output += "\ncharacters_descriptions.json not found. Please run Stage 3 (Describe Characters) first."
            return log_output, pipeline_state, None

        # Load the specific character's description
        with open(descriptions_file, "r", encoding="utf-8") as f:
            descriptions = json.load(f)

        if character_name not in descriptions:
            log_output += f"\nCharacter '{character_name}' not found in descriptions."
            return log_output, pipeline_state, None

        char_description = descriptions[character_name]

        # Voice generation uses moss by default
        voice_engine = "moss"

        # Call generate_voice_samples from package
        progress(0, desc=f"Regenerating voice sample for {character_name} with TTS engine '{voice_engine}'...")
        result_msg, generated_voices = gen_voice_samples(
            descriptions={character_name: char_description},
            output_dir=str(chapters_dir),
            single_character=character_name,
            verbose=False,
            progress=progress,
            voice_engine=voice_engine,
            force_regenerate=True,
            nemotron_endpoint=None,
        )

        log_output += f"\n{result_msg}"

        # Return the path to the regenerated file
        wav_path = get_character_wav_file(character_name, chapters_dir)
        log_output += f"\n\nVoice sample regenerated for: {character_name} (temp: {chapters_dir.parent})"

        # Reload voice map
        pipeline_state.load_voice_map()
        return log_output, pipeline_state, wav_path

    except Exception as e:
        log_output += f"\nError regenerating voice sample: {str(e)}"
        log_output += f"\n{traceback.format_exc()}"
        return log_output, pipeline_state, None


# ============================================================================
# STAGE 5: FULL AUDIOBOOK GENERATION
# ============================================================================


def generate_tts_audio(
    pipeline_state: PipelineState,
    log_output: str,
    max_chapters: Optional[int],
    turbo: bool = False,
    seed_voice_map: str = None,
    whisper_cpu: bool = False,
    validate_clean: bool = False,
    concurrency: int = 1,
    progress=gr.Progress()
) -> Tuple[str, PipelineState]:
    """Stage 5.1: Generate TTS audio for each line/voice.

    Args:
        pipeline_state: Current pipeline state (contains chapters from Stage 1)
        log_output: Current log output string
        max_chapters: Maximum number of chapters to process
        turbo: Use KugelAudio turbo model (kugel-1-turbo)
        seed_voice_map: Path to seed voices_map.json for reusing existing voices
        whisper_cpu: Use CPU for Whisper
        validate_clean: If True, validate audio contains only clean speech (no music/SFX)
        progress: Gradio progress callback

    Returns:
        Tuple of (log_output, pipeline_state)
    """
    if pipeline_state is None:
        progress(1.0, desc="Pipeline state not initialized. Please run Stage 4 (Generate Voices) first.")
        log_output += "\nPipeline state not initialized. Please run Stage 4 (Generate Voices) first."
        return log_output, None

    log_output += "\n\n=== Stage 5.1: Generating TTS Audio ==="
    progress(0, desc=f"Starting TTS audio generation... (temp: {get_chapters_dir().parent})")
    try:
        chapters_dir = get_chapters_dir()
        if not chapters_dir:
            log_output += "\nError: Chapters directory not initialized."
            return log_output, pipeline_state

        # Check if chapter map files exist
        map_files = glob.glob(str(chapters_dir / "*.map.json"))
        if not map_files:
            log_output += "\nNo .map.json files found. Please run Stage 2 (Label Speakers) first."
            return log_output, pipeline_state

        # Check if characters_descriptions.json exists
        descriptions_file = get_characters_descriptions_file()
        if not descriptions_file or not descriptions_file.exists():
            log_output += "\ncharacters_descriptions.json not found. Please run Stage 3 (Describe Characters) first."
            return log_output, pipeline_state

        # Load character descriptions as voices_map
        with open(descriptions_file, "r", encoding="utf-8") as f:
            descriptions = json.load(f)

        # Load seed voices if provided
        seed_characters = load_seed_characters(seed_voice_map)

        voices_map = {}
        if seed_characters:
            for char_name, voice_path in seed_characters.items():
                voices_map[char_name] = os.path.basename(voice_path)
            progress(0, desc=f"Loaded {len(seed_characters)} seeded voices from seed voice map")

        for char_name in descriptions.keys():
            progress(0, desc=f"Finding voice sample for character: {char_name}... (temp: {chapters_dir.parent})")
            if char_name in voices_map:
                continue
            wav_path = get_character_wav_file(char_name, chapters_dir)
            if wav_path and os.path.exists(wav_path):
                # Store relative path (basename) to avoid issues with temp directory paths
                voices_map[char_name] = os.path.basename(wav_path)
            else:
                # Use a default narrator voice if no sample found
                narrator_path = get_character_wav_file("narrator", chapters_dir)
                if narrator_path and os.path.exists(narrator_path):
                    voices_map[char_name] = os.path.basename(narrator_path)
        progress(0.5, desc=f"Prepared voices for {len(voices_map)} characters. (temp: {chapters_dir.parent})")
        if not voices_map:
            log_output += "\nNo voice samples found. Please run Stage 4 (Generate Voices) first."
            return log_output, pipeline_state

        # Load chapter maps from existing state
        pipeline_state.load_chapter_maps()
        chapter_maps = pipeline_state.chapter_maps

        # Get chapters from state (no re-parsing!)
        chapters = pipeline_state.chapters
        if not chapters:
            log_output += "\nError: Chapters not found in state. Please parse EPUB first."
            return log_output, pipeline_state

        # Count chapters for progress tracking
        num_chapters = len(chapters)
        if max_chapters:
            num_chapters = min(num_chapters, int(max_chapters))
        log_output += f"\nGenerating TTS audio for {num_chapters} chapters... (temp: {chapters_dir.parent})"

        log_output += "\nUsing chapters from Stage 1 (no re-parsing)."

        # Determine device (use CUDA if available, default to cuda:0)
        import torch
        device = AUDIO_SETTINGS["default_device"] if torch.cuda.is_available() else "cpu"
        # Use CPU for Whisper if requested
        if whisper_cpu:
            whisper_device = "cpu"
            log_output += f"\nWhisper will use CPU"
        else:
            whisper_device = device
        log_output += f"\nUsing device: {device}"

        # Use the unified generate_audiobook_from_chapters function from package
        verbose = True

        # Load duplicate replacement map if available (from Stage 3)
        duplicate_replacement_map = {}
        dup_map_file = get_duplicate_replacement_map_file()
        if dup_map_file and dup_map_file.exists():
            with open(dup_map_file, "r", encoding="utf-8") as f:
                duplicate_replacement_map = json.load(f)
            if verbose and duplicate_replacement_map:
                log_output += f"\n[DUPLICATE MAP] Loaded {len(duplicate_replacement_map)} replacements from duplicate_replacement_map.json"

        status, processed = generate_audiobook_from_chapters(
            chapters=chapters,
            chapter_maps=chapter_maps,
            voices_map=voices_map,
            output_dir=str(chapters_dir),
            device=device,
            max_chapters=max_chapters,
            turbo=turbo,
            verbose=verbose,
            progress=progress,
            duplicate_replacement_map=duplicate_replacement_map,
            seed_voice_map=seed_voice_map,
            whisper_device=whisper_device,
            validate_clean=validate_clean,
            concurrency=concurrency,
        )

        log_output += f"\n{status}"
        log_output += "\n\nStage 5.1 (TTS Audio) complete!"

        # Update state to audiobook complete
        pipeline_state.pipeline_state = "audiobook_complete"
        return log_output, pipeline_state

    except Exception as e:
        log_output += f"\nError generating TTS audio: {str(e)}"
        log_output += f"\n{traceback.format_exc()}"
        return log_output, pipeline_state


def generate_full_audiobook(
    pipeline_state: PipelineState,
    log_output: str,
    max_chapters: Optional[int],
    turbo: bool = False,
    seed_voice_map: str = None,
    whisper_cpu: bool = False,
    validate_clean: bool = False,
    concurrency: int = 1,
) -> Tuple[str, PipelineState]:
    """Stage 5: Generate full audiobook using generate_audiobook_from_chapters().

    Args:
        pipeline_state: Current pipeline state (contains chapters from Stage 1)
        log_output: Current log output string
        max_chapters: Maximum number of chapters to process
        turbo: Use KugelAudio turbo model (kugel-1-turbo)
        seed_voice_map: Path to seed voices_map.json for reusing existing voices
        whisper_cpu: Use CPU for Whisper
        validate_clean: If True, validate audio contains only clean speech (no music/SFX)
    """
    log_output += "\n\n=== Stage 5: Full Audiobook Generation ==="

    try:
        # Use the unified generate_audiobook_from_chapters function
        log_output, pipeline_state = generate_tts_audio(pipeline_state, log_output, max_chapters, turbo, seed_voice_map, whisper_cpu, validate_clean, concurrency)

        # Update state to audiobook complete (MP3s are created during generate_audiobook_from_chapters)
        log_output += f"\n\n=== Stage 5 Complete: Full Audiobook Generation === State: {pipeline_state.pipeline_state}"
        return log_output, pipeline_state
    except Exception as e:
        log_output += f"\nError generating audiobook: {str(e)}"
        log_output += f"\n{traceback.format_exc()}"
        return log_output, pipeline_state


# ============================================================================
# UI HELPERS
# ============================================================================


def update_character_table(
    pipeline_state: Optional[str],
    characters_state: Optional[Any]
) -> gr.Dataframe:
    """
    Update the character table based on stored character state.

    Args:
        pipeline_state: Current pipeline state to determine what data is available
        characters_state: gr.State containing either:
            - dict of character_name -> description (after Stage 3)
            - list of character names (after Stage 2)
            - None (before Stage 2)

    Returns:
        Updated Dataframe component with character data (3 columns: name, description, lines_spoken)
        sorted by lines spoken (descending)
    """
    chapters_dir = get_chapters_dir()

    # Check if we have descriptions (Stage 3+)
    descriptions_file = get_characters_descriptions_file()
    if descriptions_file and descriptions_file.exists() and characters_state is not None:
        try:
            with open(descriptions_file, "r", encoding="utf-8") as f:
                descriptions = json.load(f)

            # Get character lines from map files
            character_lines = count_lines_per_character(chapters_dir) if chapters_dir else {}

            # Build table data with character name, truncated description, and lines spoken
            table_data = []
            for char_name, char_desc in descriptions.items():
                truncated_desc = char_desc[:100] + ("..." if len(char_desc) > 100 else "")
                lines_spoken = character_lines.get(char_name, 0)
                table_data.append([char_name, truncated_desc, lines_spoken])

            # Sort by lines spoken (descending)
            table_data.sort(key=lambda x: x[2], reverse=True)

            return gr.Dataframe(
                headers=["Character", "Description", "Lines Spoken"],
                datatype=["str", "str", "int"],
                value=table_data,
                wrap=True,
            )
        except Exception:
            return gr.Dataframe(value=[])

    # Check if we have character list (Stage 2)
    if characters_state is not None:
        if isinstance(characters_state, list):
            # Build table data with just character names (no descriptions or line counts yet)
            table_data = [[char_name, "", 0] for char_name in sorted(characters_state)]
            return gr.Dataframe(
                headers=["Character", "Description", "Lines Spoken"],
                datatype=["str", "str", "int"],
                value=table_data,
                wrap=True,
            )
        # Check if we have character descriptions as a dict (Stage 3+)
        elif isinstance(characters_state, dict):
            # Build table from characters_state dict (character_name -> description)
            character_lines = count_lines_per_character(chapters_dir) if chapters_dir else {}
            table_data = []
            for char_name, char_desc in characters_state.items():
                truncated_desc = char_desc[:100] + ("..." if len(char_desc) > 100 else "")
                lines_spoken = character_lines.get(char_name, 0)
                table_data.append([char_name, truncated_desc, lines_spoken])
            table_data.sort(key=lambda x: x[2], reverse=True)
            return gr.Dataframe(
                headers=["Character", "Description", "Lines Spoken"],
                datatype=["str", "str", "int"],
                value=table_data,
                wrap=True,
            )

    return gr.Dataframe(value=[])


def create_or_get_pipeline_state(output_dir: str = None, voice_engine: str = None) -> PipelineState:
    """Create a new PipelineState or get the existing one from gr.State().

    Args:
        output_dir: Optional output directory path. If None, uses get_chapters_dir().
        voice_engine: Voice engine for character descriptions ('omni' or 'vox')

    Returns:
        PipelineState instance
    """
    if output_dir is None:
        chapters_dir = get_chapters_dir()
        output_dir = str(chapters_dir)
    return PipelineState(output_dir, voice_engine=voice_engine)


STATE_LABELS = {
    None: "Ready",
    "epub_parsed": "EPUB Parsed",
    "labels_complete": "Speakers Labeled",
    "characters_described": "Characters Described",
    "voice_samples_complete": "Voice Samples Ready",
    "audiobook_complete": "Audiobook Complete",
}

BUTTON_STATES = {
    None: [True, False, False, False, False, False],
    "epub_parsed": [True, True, False, False, False, False],
    "labels_complete": [True, True, True, False, False, False],
    "characters_described": [True, True, True, True, True, False],
    "voice_samples_complete": [True, True, True, True, True, True],
    "audiobook_complete": [True, True, True, True, True, True],
}

RECOMMENDATIONS = {
    None: "Upload an EPUB file and click '1. Parse' to begin.",
    "epub_parsed": "Click '2. Label' to use LLM for speaker labeling.",
    "labels_complete": "Click '3. Describe' to generate character descriptions.",
    "characters_described": "Click '4. Voices' to generate voice samples for each character.",
    "voice_samples_complete": "Click 'Read Chapters' to generate the full audiobook.",
    "audiobook_complete": "Audiobook generation complete! You can start a new project.",
}


def update_state_display(state: Optional[str]) -> gr.Textbox:
    """Update log label with state based on pipeline state."""
    state_text = STATE_LABELS.get(state, "Unknown")
    next_step = RECOMMENDATIONS.get(state, "Unknown state.")
    return gr.update(label=f"Log (State: {state_text}) - {next_step}")


def update_button_visibility_from_state(pipeline_state: PipelineState):
    """
    Update button enabled state based on pipeline state.
    Returns tuple of updates for all buttons and components.
    """
    state = pipeline_state.pipeline_state if pipeline_state else None
    _states = BUTTON_STATES.get(state, BUTTON_STATES["audiobook_complete"])

    # Define button labels with icons for each state
    # Icons: ⬜ = pending, ✅ = complete, 🔵 = active (current stage)
    # 6 buttons: parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn (Regen), tts_btn
    button_configs = {
        None: {
            "buttons": ["⬜ 1. Parse", "⬜ 2. Label", "⬜ 3. Describe", "⬜ 4. Voices", "⬜ Regen", "⬜ Read Chapters"],
            "classes": ["", "", "", "", "", ""]
        },
        "epub_parsed": {
            "buttons": ["✅ 1. Parse", "🔵 2. Label", "⬜ 3. Describe", "⬜ 4. Voices", "⬜ Regen", "⬜ Read Chapters"],
            "classes": ["stage-complete", "stage-active", "", "", "", ""]
        },
        "labels_complete": {
            "buttons": ["✅ 1. Parse", "✅ 2. Label", "🔵 3. Describe", "⬜ 4. Voices", "⬜ Regen", "⬜ Read Chapters"],
            "classes": ["stage-complete", "stage-complete", "stage-active", "", "", ""]
        },
        "characters_described": {
            "buttons": ["✅ 1. Parse", "✅ 2. Label", "✅ 3. Describe", "🔵 4. Voices", "⬜ Regen", "⬜ Read Chapters"],
            "classes": ["stage-complete", "stage-complete", "stage-complete", "stage-active", "", ""]
        },
        "voice_samples_complete": {
            "buttons": ["✅ 1. Parse", "✅ 2. Label", "✅ 3. Describe", "✅ 4. Voices", "🔵 Regen", "⬜ Read Chapters"],
            "classes": ["stage-complete", "stage-complete", "stage-complete", "stage-complete", "stage-active", ""]
        },
        "audiobook_complete": {
            "buttons": ["✅ 1. Parse", "✅ 2. Label", "✅ 3. Describe", "✅ 4. Voices", "✅ Regen", "✅ Read Chapters"],
            "classes": ["stage-complete", "stage-complete", "stage-complete", "stage-complete", "stage-complete", "stage-complete"]
        },
    }

    config = button_configs.get(state, button_configs[None])
    button_updates = []

    for i, (label, interactive, css_class) in enumerate(zip(config["buttons"], _states, config["classes"])):
        update = gr.update(
            value=label,
            interactive=interactive
        )
        if css_class:
            update["elem_classes"] = css_class
        button_updates.append(update)

    return tuple(button_updates)


def update_state_display_from_state(pipeline_state: PipelineState) -> gr.Textbox:
    """Update log label with state based on pipeline state."""
    state = pipeline_state.pipeline_state if pipeline_state else None
    state_text = STATE_LABELS.get(state, "Unknown")
    next_step = RECOMMENDATIONS.get(state, "Unknown state.")
    return gr.update(label=f"Log (State: {state_text}) - {next_step}")


def update_character_table_from_state(pipeline_state: PipelineState) -> gr.Dataframe:
    """Update character table based on PipelineState (simple list, no audio)."""
    chapters_dir = get_chapters_dir()

    # Check if we have descriptions (Stage 3+)
    descriptions_file = get_characters_descriptions_file()
    if descriptions_file and descriptions_file.exists() and pipeline_state and pipeline_state.character_descriptions:
        try:
            descriptions = pipeline_state.character_descriptions

            # Get character lines from map files
            character_lines = count_lines_per_character(chapters_dir) if chapters_dir else {}

            # Build table data with character name, truncated description, lines spoken
            table_data = []
            for char_name, char_desc in descriptions.items():
                truncated_desc = char_desc[:100] + ("..." if len(char_desc) > 100 else "")
                lines_spoken = character_lines.get(char_name, 0)
                table_data.append([char_name, truncated_desc, lines_spoken])

            # Sort by lines spoken (descending)
            table_data.sort(key=lambda x: x[2], reverse=True)

            return gr.Dataframe(
                headers=["Character", "Description", "Lines Spoken"],
                datatype=["str", "str", "int"],
                value=table_data,
                wrap=True,
            )
        except Exception:
            return gr.Dataframe(value=[])

    # Check if we have character list (Stage 2)
    if pipeline_state and pipeline_state.characters:
        if isinstance(pipeline_state.characters, list):
            # Build table data with just character names
            table_data = [[char_name, "", 0] for char_name in sorted(pipeline_state.characters)]
            return gr.Dataframe(
                headers=["Character", "Description", "Lines Spoken"],
                datatype=["str", "str", "int"],
                value=table_data,
                wrap=True,
            )

    return gr.Dataframe(value=[])


def update_character_gallery_from_state(pipeline_state: PipelineState) -> gr.Radio:
    """Update character gallery with clickable character badges using Radio component."""
    # Check if we have descriptions (Stage 3+)
    descriptions_file = get_characters_descriptions_file()
    if descriptions_file and descriptions_file.exists() and pipeline_state and pipeline_state.character_descriptions:
        try:
            descriptions = pipeline_state.character_descriptions
            # Return list of character names for Radio component
            return gr.Radio(choices=list(descriptions.keys()), label="Select Character")
        except Exception:
            return gr.Radio(choices=[], label="No characters available")

    # Check if we have character list (Stage 2)
    if pipeline_state and pipeline_state.characters:
        if isinstance(pipeline_state.characters, list):
            return gr.Radio(choices=sorted(pipeline_state.characters), label="Select Character")

    return gr.Radio(choices=[], label="Characters will appear here after speaker labeling")


def update_chapter_progress_from_state(pipeline_state: PipelineState) -> gr.HTML:
    """Update chapter progress display."""
    chapters_dir = get_chapters_dir()

    if not chapters_dir or not chapters_dir.exists():
        return gr.HTML(value="<div style='color: #888; padding: 20px; text-align: center;'>Chapters will appear here after EPUB parsing.</div>")

    # Get chapter text files (only chapter_N.txt format, exclude intermediate files)
    import re
    chapter_txt_files = sorted([f for f in chapters_dir.glob("chapter_*.txt") if re.match(r'chapter_\d+\.txt$', f.name)],
                              key=natural_sort_key)
    # Get chapter map files (speaker labeled)
    chapter_map_files = sorted(chapters_dir.glob("chapter_*.map.json"), key=natural_sort_key)
    # Get chapter audio files
    chapter_mp3_files = sorted(chapters_dir.glob("chapter_*.mp3"), key=natural_sort_key)

    if not chapter_txt_files:
        return gr.HTML(value="<div style='color: #888; padding: 20px; text-align: center;'>Chapters will appear here after EPUB parsing.</div>")

    # Create progress bar
    total_chapters = len(chapter_txt_files)
    labeled_chapters = len(chapter_map_files)
    audio_chapters = len(chapter_mp3_files)

    # Progress percentages
    label_progress = (labeled_chapters / total_chapters * 100) if total_chapters > 0 else 0
    audio_progress = (audio_chapters / total_chapters * 100) if total_chapters > 0 else 0

    html = f'''
    <div style="padding: 16px; max-width: 800px; margin: 0 auto;">
        <h3 style="color: #667eea; margin-bottom: 16px;">📚 Chapter Progress</h3>

        <!-- Overall Progress -->
        <div style="margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #b2bec3;">Overall Progress</span>
                <span style="color: #667eea;">{total_chapters} chapters</span>
            </div>
            <div style="height: 8px; background: rgba(102, 126, 234, 0.2); border-radius: 4px; overflow: hidden;">
                <div style="width: 100%; height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);"></div>
            </div>
        </div>

        <!-- Labeling Progress -->
        <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #b2bec3;">🏷️ Speaker Labeling</span>
                <span style="color: #00b894;">{labeled_chapters}/{total_chapters} ({label_progress:.0f}%)</span>
            </div>
            <div style="height: 6px; background: rgba(0, 184, 148, 0.2); border-radius: 3px; overflow: hidden;">
                <div style="width: {label_progress}%; height: 100%; background: linear-gradient(90deg, #00b894 0%, #00cec9 100%); transition: width 0.3s;"></div>
            </div>
        </div>

        <!-- Audio Progress -->
        <div style="margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #b2bec3;">🎵 Audio Generated</span>
                <span style="color: #f39c12;">{audio_chapters}/{total_chapters} ({audio_progress:.0f}%)</span>
            </div>
            <div style="height: 6px; background: rgba(243, 156, 18, 0.2); border-radius: 3px; overflow: hidden;">
                <div style="width: {audio_progress}%; height: 100%; background: linear-gradient(90deg, #f39c12 0%, #f1c40f 100%); transition: width 0.3s;"></div>
            </div>
        </div>

        <!-- Chapter List -->
        <div style="background: rgba(26, 26, 46, 0.6); border-radius: 8px; padding: 12px;">
            <h4 style="color: #667eea; margin-bottom: 12px;">Chapter Files</h4>
    '''

    # Add chapter list
    for i, txt_file in enumerate(chapter_txt_files):
        chapter_num = i + 1
        has_map = any(f.stem == txt_file.stem for f in chapter_map_files)
        has_audio = any(f.stem == txt_file.stem for f in chapter_mp3_files)

        status_icon = "✅🎵" if has_audio else ("✅" if has_map else "⬜")
        audio_link = f'<a href="/file={chapters_dir / f"{txt_file.stem}.mp3"}" style="color: #00b894; text-decoration: none; margin-left: 8px;">🎧 Download</a>' if has_audio else ""

        html += f'''
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; margin-bottom: 4px; background: rgba(102, 126, 234, 0.1); border-radius: 6px;">
                <span style="color: #e0e0e0;">Chapter {chapter_num}</span>
                <span style="color: #888;">{status_icon} {audio_link}</span>
            </div>
        '''

    html += '''
        </div>
    </div>
    '''

    return gr.HTML(value=html)


# ============================================================================
# GRADIO INTERFACE
# ============================================================================


def create_interface(
    api_key_default: str = DEFAULT_API_KEY,
    port_default: str = DEFAULT_PORT,
    num_attempts_default: int = DEFAULT_NUM_ATTEMPTS,
    epub_path_default: Optional[str] = str(DEFAULT_EPUB_FILE),
    max_chapters_default: int = DEFAULT_MAX_CHAPTERS,
    seed_voice_map_default: Optional[str] = None,
    saved_temp_dir: Optional[str] = None,
    tts_engine_default: Optional[str] = None,
    voice_engine_default: Optional[str] = None,
    verbose: bool = False,
):
    """Create the Gradio interface with all stages using a state machine pattern.

    If epub_path_default is provided and points to an existing file, the interface
    will automatically trigger parsing on startup.

    Args:
        saved_temp_dir: Optional path to a saved temp directory to restore from.
                       If provided, restores the pipeline state from this directory.
        voice_engine_default: Default voice engine for character descriptions ('omni' or 'vox')
    """

    with gr.Blocks() as demo:
        # Custom CSS for audio/voice aesthetic
        gr.HTML("""
        <style>
        /* Custom scrollbar with audio-wave aesthetic */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #1a1a2e;
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #764ba2 0%, #f093fb 100%);
        }

        /* Main background with dark audio-theme */
        .gradio-container {
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
            min-height: 100vh;
        }

        /* Header styling */
        h1 {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800;
            letter-spacing: -1px;
        }

        /* Accordion styling */
        .gr-accordion {
            background: rgba(26, 26, 46, 0.8) !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            border-radius: 12px !important;
        }
        .gr-accordion-header {
            background: linear-gradient(90deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%) !important;
            color: #e0e0e0 !important;
        }

        /* Button styling with gradient */
        .gr-button-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        .gr-button-primary:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
        }
        .gr-button-secondary {
            background: linear-gradient(135deg, #2d3436 0%, #636e72 100%) !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            color: #e0e0e0 !important;
            transition: all 0.3s ease !important;
        }
        .gr-button-secondary:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border-color: transparent !important;
        }

        /* Pulse animation for active stage */
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 10px rgba(102, 126, 234, 0.5); }
            50% { box-shadow: 0 0 25px rgba(102, 126, 234, 0.9); }
        }
        .stage-active {
            animation: pulse-glow 2s infinite;
        }

        /* Completed stage styling */
        .stage-complete {
            background: linear-gradient(135deg, #00b894 0%, #00cec9 100%) !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(0, 184, 148, 0.4) !important;
        }

        /* Log display styling */
        .log-display {
            background: #0d0d1a !important;
            border: 1px solid rgba(102, 126, 234, 0.5) !important;
            border-radius: 8px !important;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
            font-size: 13px !important;
            line-height: 1.5 !important;
        }

        /* Tab styling */
        .gr-tab {
            background: rgba(26, 26, 46, 0.5) !important;
            border-color: rgba(102, 126, 234, 0.3) !important;
        }
        .gr-tab-active {
            background: rgba(102, 126, 234, 0.2) !important;
            border-color: #667eea !important;
        }

        /* Dataframe styling */
        .dataframe-container {
            background: rgba(26, 26, 46, 0.6) !important;
            border-radius: 8px !important;
        }
        .dataframe thead {
            background: linear-gradient(90deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%) !important;
            color: #e0e0e0 !important;
        }
        .dataframe tbody tr:hover {
            background: rgba(102, 126, 234, 0.1) !important;
        }

        /* Audio player styling */
        audio {
            filter: drop-shadow(0 2px 8px rgba(102, 126, 234, 0.3));
        }

        /* Progress bar styling */
        .progress-bar {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        }

        /* Card styling for character gallery */
        .character-card {
            background: linear-gradient(135deg, rgba(45, 52, 54, 0.8) 0%, rgba(118, 75, 162, 0.2) 100%);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 12px;
            padding: 16px;
            margin: 8px;
            transition: all 0.3s ease;
        }
        .character-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            border-color: rgba(102, 126, 234, 0.6);
        }
        .character-card-name {
            font-size: 18px;
            font-weight: 600;
            color: #667eea;
            margin-bottom: 8px;
        }
        .character-card-desc {
            font-size: 14px;
            color: #b2bec3;
            margin-bottom: 12px;
        }
        .character-card-lines {
            font-size: 12px;
            color: #636e72;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Waveform animation */
        @keyframes waveform {
            0%, 100% { height: 10px; }
            50% { height: 25px; }
        }
        .waveform-bar {
            display: inline-block;
            width: 4px;
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            border-radius: 2px;
            animation: waveform 0.5s ease-in-out infinite;
            margin: 0 2px;
        }
        </style>
        <script>
        // JavaScript helper for character gallery clicks
        function updateGallerySelection(charName) {
            console.log('updateGallerySelection called with:', charName);
            alert('Clicked: ' + charName);
            // Find the hidden textbox by its elem_id
            const textbox = document.getElementById('char-gallery-select');
            console.log('Found textbox:', textbox);
            if (textbox) {
                textbox.value = charName;
                // Dispatch change event to trigger Gradio handler
                textbox.dispatchEvent(new Event('change', { bubbles: true }));
                console.log('Change event dispatched');
            } else {
                console.log('Textbox not found!');
            }
        }
        </script>
        """)

        gr.Markdown("# Audiobook Voice Generator")

        # Collapsible input area (settings only, buttons stay visible)
        with gr.Accordion(label="Settings", open=False):
            with gr.Row():
                api_key_input = gr.Textbox(label="API", value=api_key_default, scale=2)
                port_input = gr.Textbox(label="Port", value=port_default, scale=1)
                num_attempts_input = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=num_attempts_default,
                    step=1,
                    label="LLM Attempts",
                    scale=2,
                )
                max_chapters_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=max_chapters_default,
                    step=1,
                    label="Max Chapters",
                    scale=2,
                )
            with gr.Row():
                whisper_cpu_checkbox = gr.Checkbox(
                    label="Use CPU for Whisper",
                    value=False,
                    scale=3
                )
                validate_voice_checkbox = gr.Checkbox(
                    label="Validate Voices with LLM",
                    value=False,
                    scale=3,
                    info="Use LLM to verify voices match character descriptions"
                )
            with gr.Row():
                validate_clean_checkbox = gr.Checkbox(
                    label="Validate Clean Audio",
                    value=False,
                    scale=3,
                    info="Check for music/SFX in generated audio (requires Validate Voices)"
                )
                concurrency_slider = gr.Slider(
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=1,
                    label="Concurrency",
                    scale=2,
                    info="Lines processed in parallel (higher = faster, uses more VRAM)"
                )

            # EPUB upload - use the provided path as default if it exists
            # If epub_path_default is a string path, convert it to a tuple for Gradio's File component
            epub_upload_default = None
            if epub_path_default and os.path.exists(epub_path_default):
                # Gradio File expects file path string or None
                epub_upload_default = epub_path_default
            epub_upload = gr.File(label="EPUB", file_types=[".epub"], value=epub_upload_default)

            # Seed voice map
            seed_voice_map_input = gr.File(
                label="Seed Voice Map (optional)",
                file_types=[".json"],
                value=seed_voice_map_default
            )

            # Voice engine selection (hidden, passed through state)
            voice_engine_input = gr.Radio(
                choices=["omni", "vox"],
                label="Voice Engine for Character Descriptions",
                value=voice_engine_default or "omni",
                info="omni=OmniVoice format, vox=VoxCPM format"
            )

            # Save and Load
            with gr.Row():
                save_temp_btn = gr.Button("Save Temp", variant="secondary")
                load_temp_file_input = gr.File(
                    label="Load .zip (select file to restore)",
                    file_types=[".zip"],
                    type="filepath",
                    visible=True
                )

        # All 6 buttons in a single row - only Parse is clickable initially
        with gr.Row():
            parse_btn = gr.Button("1. Parse", variant="primary", scale=1, interactive=True)
            label_btn = gr.Button("2. Label", variant="secondary", scale=1, interactive=False)
            describe_btn = gr.Button("3. Describe", variant="secondary", scale=1, interactive=False)
            voice_samples_btn = gr.Button("4. Voices", variant="secondary", scale=1, interactive=False)
            tts_btn = gr.Button("Read Chapters", variant="primary", scale=1, interactive=False)
        # Initialize log label based on restored state
        if saved_temp_dir:
            restored_state, restore_msg = restore_pipeline_state(saved_temp_dir)
            initial_log_label = f"=== Session Restored ===\n{restore_msg}"
        else:
            initial_log_label = "Log (State: Ready)"
            restored_state = None

        with gr.Row():
            log_output = gr.Textbox(label=initial_log_label, lines=3, max_lines=3, interactive=False, elem_classes=["log-display"])

        with gr.Row():
            with gr.Tab("🎭 Characters"):
                # Character gallery using Radio for native selection (now first)
                character_gallery = gr.Radio(
                    choices=[],
                    label="Select Character",
                    interactive=True
                )

                # Active character card (shown when character selected)
                with gr.Row(visible=False) as active_character_row:
                    with gr.Column(scale=1):
                        active_char_name = gr.Markdown("### Character")
                        active_char_desc = gr.Textbox(
                            label="Description (editable)",
                            lines=4,
                            value="Description"
                        )
                        active_char_lines = gr.Markdown("**Lines spoken:** 0")
                        character_audio = gr.Audio(label="Voice Sample", type="filepath", visible=True)
                        with gr.Row():
                            save_desc_btn = gr.Button("Save Description", variant="secondary", size="sm")
                            generate_char_btn = gr.Button("Regenerate Voice", variant="secondary", size="sm")

                # Hidden component to store selected character from gallery
                selected_char_gallery = gr.Textbox(
                    label="Selected Character",
                    visible=False,
                    interactive=False,
                    elem_id="char-gallery-select"
                )

                # Character info table (hidden - no longer needed)
                character_table = gr.Dataframe(
                    headers=["Character", "Description", "Lines Spoken"],
                    datatype=["str", "str", "int"],
                    wrap=True,
                    max_height=300,
                    visible=False,
                )

            with gr.Tab("📚 Chapters"):
                # Chapter progress display
                chapter_progress = gr.HTML(
                    value="<div style='color: #888; padding: 20px; text-align: center;'>Chapters will appear here after EPUB parsing.</div>"
                )

            # Hidden file input for Load Temp (required for button to work)
            load_temp_path_input = gr.File(
                label="Load .zip",
                file_types=[".zip"],
                visible=False
            )

        # Use PipelineState as the unified state for both CLI and Gradio
        pipeline_state_obj = gr.State(restored_state)

        # ============================================================================
        # EVENT HANDLERS
        # ============================================================================

        # Parse EPUB - Stage 1
        parse_btn.click(
            fn=parse_epub_to_file,
            inputs=[epub_upload, max_chapters_slider],
            outputs=[log_output, pipeline_state_obj],
        ).then(
            fn=update_button_visibility_from_state,
            inputs=pipeline_state_obj,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn],
        ).then(
            fn=update_state_display_from_state,
            inputs=pipeline_state_obj,
            outputs=log_output,
        ).then(
            fn=update_character_gallery_from_state,
            inputs=pipeline_state_obj,
            outputs=character_gallery,
        ).then(
            fn=update_chapter_progress_from_state,
            inputs=pipeline_state_obj,
            outputs=chapter_progress,
        )

        # Label Speakers - Stage 2
        label_btn.click(
            fn=process_chapters_for_labels,
            inputs=[api_key_input, port_input, num_attempts_input, pipeline_state_obj, log_output, seed_voice_map_input],
            outputs=[log_output, pipeline_state_obj],
        ).then(
            fn=update_button_visibility_from_state,
            inputs=pipeline_state_obj,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn],
        ).then(
            fn=update_state_display_from_state,
            inputs=pipeline_state_obj,
            outputs=log_output,
        ).then(
            fn=update_character_table_from_state,
            inputs=pipeline_state_obj,
            outputs=character_table,
        ).then(
            fn=update_character_gallery_from_state,
            inputs=pipeline_state_obj,
            outputs=character_gallery,
        ).then(
            fn=update_chapter_progress_from_state,
            inputs=pipeline_state_obj,
            outputs=chapter_progress,
        )

        # Describe Characters - Stage 3
        describe_btn.click(
            fn=describe_characters_ui,
            inputs=[api_key_input, port_input, pipeline_state_obj, log_output, seed_voice_map_input, voice_engine_input],
            outputs=[log_output, pipeline_state_obj],
        ).then(
            fn=update_button_visibility_from_state,
            inputs=pipeline_state_obj,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn],
        ).then(
            fn=update_state_display_from_state,
            inputs=pipeline_state_obj,
            outputs=log_output,
        ).then(
            fn=update_character_table_from_state,
            inputs=pipeline_state_obj,
            outputs=character_table,
        ).then(
            fn=update_character_gallery_from_state,
            inputs=pipeline_state_obj,
            outputs=character_gallery,
        ).then(
            fn=update_chapter_progress_from_state,
            inputs=pipeline_state_obj,
            outputs=chapter_progress,
        )

        # Generate All Voice Samples - Stage 4
        voice_samples_btn.click(
            fn=generate_voice_samples,
            inputs=[pipeline_state_obj, log_output, seed_voice_map_input, validate_voice_checkbox],
            outputs=[log_output, pipeline_state_obj],
        ).then(
            fn=update_button_visibility_from_state,
            inputs=pipeline_state_obj,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn],
        ).then(
            fn=update_state_display_from_state,
            inputs=pipeline_state_obj,
            outputs=log_output,
        ).then(
            fn=update_character_gallery_from_state,
            inputs=pipeline_state_obj,
            outputs=character_gallery,
        ).then(
            fn=update_chapter_progress_from_state,
            inputs=pipeline_state_obj,
            outputs=chapter_progress,
        )

        # Handle row selection in character table - show active character card
        def on_character_select(evt: gr.SelectData, _pipeline_state_obj):
            """Handle row selection in the character table."""
            if evt is None or evt.index is None:
                return (gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False))

            character_name = evt.row_value[0] if evt.row_value and len(evt.row_value) > 0 else None

            if not character_name:
                return (gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False))

            if not _pipeline_state_obj:
                return (gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False))

            chapters_dir = get_chapters_dir()
            if not chapters_dir:
                return (gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False))

            wav_path = get_character_wav_file(character_name, chapters_dir)

            if wav_path and os.path.exists(wav_path):
                # Update selected character in pipeline state
                _pipeline_state_obj.selected_character = character_name

                # Get character description if available
                char_desc = ""
                if _pipeline_state_obj.character_descriptions:
                    char_desc = _pipeline_state_obj.character_descriptions.get(character_name, "No description available.")
                    if len(char_desc) > 200:
                        char_desc = char_desc[:200] + "..."

                # Get lines spoken
                character_lines = count_lines_per_character(chapters_dir) if chapters_dir else {}
                lines_spoken = character_lines.get(character_name, 0)

                return (
                    gr.update(visible=True),  # active_character_row
                    gr.update(value=f"### {character_name}"),  # active_char_name
                    gr.update(value=char_desc),  # active_char_desc
                    gr.update(value=f"**Lines spoken:** {lines_spoken}"),  # active_char_lines
                    gr.update(visible=True, value=wav_path),  # character_audio
                    gr.update(visible=True)  # generate_char_btn
                )
            else:
                return (gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False))

        character_table.select(
            fn=on_character_select,
            inputs=pipeline_state_obj,
            outputs=[active_character_row, active_char_name, active_char_desc, active_char_lines, character_audio, generate_char_btn],
        )

        # Handle character badge click from gallery
        def on_gallery_select(character_name: str, pipeline_state_obj):
            """Handle character selection from gallery badges."""
            if not character_name or not pipeline_state_obj:
                return (gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False))

            chapters_dir = get_chapters_dir()
            if not chapters_dir:
                return (gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False))

            wav_path = get_character_wav_file(character_name, chapters_dir)
            has_audio = wav_path and os.path.exists(wav_path)

            # Update selected character in pipeline state
            pipeline_state_obj.selected_character = character_name

            # Get character description if available (full description, no truncation)
            char_desc = ""
            if pipeline_state_obj.character_descriptions:
                char_desc = pipeline_state_obj.character_descriptions.get(character_name, "No description available.")

            # Get lines spoken
            character_lines = count_lines_per_character(chapters_dir) if chapters_dir else {}
            lines_spoken = character_lines.get(character_name, 0)

            return (
                gr.update(visible=True),  # active_character_row
                gr.update(value=f"### {character_name}"),  # active_char_name
                gr.update(value=char_desc),  # active_char_desc
                gr.update(value=f"**Lines spoken:** {lines_spoken}"),  # active_char_lines
                gr.update(visible=has_audio, value=wav_path if has_audio else None),  # character_audio
                gr.update(visible=True)  # generate_char_btn
            )

        character_gallery.change(
            fn=on_gallery_select,
            inputs=[character_gallery, pipeline_state_obj],
            outputs=[active_character_row, active_char_name, active_char_desc, active_char_lines, character_audio, generate_char_btn],
        )

        # Save character description
        def on_save_description(pipeline_state_obj, description_text, log_output):
            """Save the edited character description to the descriptions file."""
            if not pipeline_state_obj:
                log_output += "\nError: Pipeline state not initialized."
                return log_output

            character_name = pipeline_state_obj.selected_character
            if not character_name:
                log_output += "\nNo character selected."
                return log_output

            try:
                descriptions_file = get_characters_descriptions_file()
                if not descriptions_file or not descriptions_file.exists():
                    log_output += f"\nError: Descriptions file not found: {descriptions_file}"
                    return log_output

                # Load existing descriptions
                with open(descriptions_file, "r", encoding="utf-8") as f:
                    descriptions = json.load(f)

                # Update the description for the selected character
                descriptions[character_name] = description_text

                # Save back to file
                with open(descriptions_file, "w", encoding="utf-8") as f:
                    json.dump(descriptions, f, indent=2, ensure_ascii=False)

                # Update pipeline state
                pipeline_state_obj.character_descriptions = descriptions
                pipeline_state_obj.selected_character = character_name

                log_output += f"\nSaved description for '{character_name}'."
                return log_output

            except Exception as e:
                log_output += f"\nError saving description: {str(e)}"
                return log_output

        save_desc_btn.click(
            fn=on_save_description,
            inputs=[pipeline_state_obj, active_char_desc, log_output],
            outputs=[log_output],
        )

        # Regenerate voice sample for selected character
        def on_regenerate_click(pipeline_state_obj, log_output):
            """Regenerate voice sample for the selected character."""
            character_name = pipeline_state_obj.selected_character if pipeline_state_obj else None
            if not character_name:
                log_output += "\nNo character selected."
                return log_output, pipeline_state_obj, None

            return regenerate_voice_sample(character_name, pipeline_state_obj, log_output)

        generate_char_btn.click(
            fn=on_regenerate_click,
            inputs=[pipeline_state_obj, log_output],
            outputs=[log_output, pipeline_state_obj, character_audio],
        ).then(
            fn=update_button_visibility_from_state,
            inputs=pipeline_state_obj,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn],
        ).then(
            fn=update_state_display_from_state,
            inputs=pipeline_state_obj,
            outputs=log_output,
        ).then(
            fn=update_character_table_from_state,
            inputs=pipeline_state_obj,
            outputs=character_table,
        ).then(
            fn=update_character_gallery_from_state,
            inputs=pipeline_state_obj,
            outputs=character_gallery,
        )

        # Generate Full Audiobook - Stage 5
        tts_btn.click(
            fn=generate_full_audiobook,
            inputs=[pipeline_state_obj, log_output, max_chapters_slider, seed_voice_map_input, whisper_cpu_checkbox, validate_clean_checkbox, concurrency_slider],
            outputs=[log_output, pipeline_state_obj],
        ).then(
            fn=update_button_visibility_from_state,
            inputs=pipeline_state_obj,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn],
        ).then(
            fn=update_state_display_from_state,
            inputs=pipeline_state_obj,
            outputs=log_output,
        ).then(
            fn=update_character_gallery_from_state,
            inputs=pipeline_state_obj,
            outputs=character_gallery,
        ).then(
            fn=update_chapter_progress_from_state,
            inputs=pipeline_state_obj,
            outputs=chapter_progress,
        )

        # Save Temp button - save the current temp directory as a zip archive
        def save_temp_handler():
            """Save the current temp directory as a zip archive."""
            temp_dir = get_temp_dir()
            if not temp_dir:
                return "No temp directory to save"
            try:
                # Check if there's any work to save
                chapters_dir = Path(temp_dir) / "chapters"
                chapter_files = list(chapters_dir.glob("chapter_*.txt")) if chapters_dir.exists() else []
                if not chapter_files:
                    return "No work to save. Please parse an EPUB file first."
                archive_path = save_temp_dir(temp_dir)
                # Show the chapters directory path for consistency
                from utils import get_chapters_dir
                chapters_dir = get_chapters_dir()
                return f"Saved to: {archive_path}\nChapters: {chapters_dir}"
            except Exception as e:
                return f"Error saving: {str(e)}"

        save_temp_btn.click(
            fn=save_temp_handler,
            inputs=None,
            outputs=[log_output],
        )

        # Load temp directory from path input
        def load_temp_handler(path):
            """Load from a saved temp directory zip archive and restore full state.

            Args:
                path: Path to the saved temp directory zip archive (from File component)

            Returns:
                Tuple of (log_output, pipeline_state, character_table, character_gallery, chapter_progress, button_updates)
            """
            # Handle Gradio File component output (dict with 'name' key or path string)
            if isinstance(path, dict):
                path = path.get('name')

            if not path or not str(path).strip():
                return ("Please select a zip file to load.", None, gr.update(), gr.update(), gr.update(),
                        *[gr.update() for _ in range(6)])

            path = str(path).strip()
            path_obj = Path(path)

            if not path_obj.exists():
                return (f"Error: File not found: {path}", None, gr.update(), gr.update(), gr.update(),
                        *[gr.update() for _ in range(6)])

            if path_obj.suffix != ".zip":
                return ("Error: Please select a .zip file.", None, gr.update(), gr.update(), gr.update(),
                        *[gr.update() for _ in range(6)])

            try:
                temp_dir = load_temp_dir(str(path_obj))

                if temp_dir:
                    # Get the chapters directory for display
                    from utils import get_chapters_dir_from_saved
                    chapters_dir = get_chapters_dir_from_saved(temp_dir)
                    # temp_dir is the extracted archive root, chapters_dir is the "chapters" subdirectory
                    # The actual files are in chapters_dir, so pass that to PipelineState
                    output_dir = str(chapters_dir) if chapters_dir else temp_dir

                    # Detect pipeline state from loaded files
                    state = PipelineState(output_dir) if output_dir else None
                    detected_state = state.get_pipeline_state() if state else "initial"

                    if detected_state != "initial":
                        # Load the actual state
                        state.pipeline_state = detected_state
                        state.load_chapter_maps()
                        state.get_characters()
                        state.load_character_descriptions()
                        state.load_voice_map()

                        # Build restoration message
                        msg = f"=== Session Restored ===\nState: {detected_state}\nChapters: {chapters_dir}\n\nUI updated below."

                        # Update all UI components
                        button_updates = update_button_visibility_from_state(state)
                        char_table = update_character_table_from_state(state)
                        char_gallery = update_character_gallery_from_state(state)
                        chap_progress = update_chapter_progress_from_state(state)

                        return (msg, state, char_table, char_gallery, chap_progress, *button_updates)

                    return ("Loaded files but no pipeline state detected.", None, gr.update(), gr.update(), gr.update(),
                            *[gr.update() for _ in range(6)])

                return ("Failed to load archive.", None, gr.update(), gr.update(), gr.update(),
                        *[gr.update() for _ in range(6)])
            except Exception as e:
                import traceback
                return (f"Error loading: {str(e)}\n{traceback.format_exc()}", None, gr.update(), gr.update(), gr.update(),
                        *[gr.update() for _ in range(6)])

        # Load temp on button click - uses the file input that was just uploaded
        def on_load_temp_click():
            """Triggered when Load .zip button is clicked - reads the file from the hidden File component."""
            # This is a placeholder - the actual file handling happens via the File component
            return "Please select a zip file first, then click the Load .zip button."

        # Connect the File component change event to load the temp directory
        load_temp_file_input.change(
            fn=load_temp_handler,
            inputs=[load_temp_file_input],
            outputs=[log_output, pipeline_state_obj, character_table, character_gallery, chapter_progress,
                     parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn],
        )

        # Initialize UI state on load (shows restored state if available)
        def initialize_ui(pipeline_state, current_log):
            """Initialize UI with restored state."""
            if pipeline_state:
                # Update button visibility based on restored state
                button_updates = update_button_visibility_from_state(pipeline_state)
                # Update character table
                char_table = update_character_table_from_state(pipeline_state)
                # Update character gallery
                char_gallery = update_character_gallery_from_state(pipeline_state)
                # Update chapter progress
                chap_progress = update_chapter_progress_from_state(pipeline_state)
                # Prepend restoration message to existing log
                new_log = f"=== Session Restored ===\n{current_log}"

                # If we have characters, auto-select the first one to show the preview
                if hasattr(char_gallery, 'choices') and char_gallery.choices and len(char_gallery.choices) > 0:
                    first_char = char_gallery.choices[0] if isinstance(char_gallery.choices[0], str) else char_gallery.choices[0][0]
                    # Get the character preview updates
                    preview_updates = on_gallery_select(first_char, pipeline_state)
                    return (new_log, char_table, char_gallery, chap_progress, *button_updates, preview_updates[0], preview_updates[1], preview_updates[2], preview_updates[3], preview_updates[4], preview_updates[5])

                return (new_log, char_table, char_gallery, chap_progress, *button_updates)
            return (current_log, gr.update(), gr.update(), gr.update(),
                    *[gr.update() for _ in range(6)])

        def check_and_restore_state(current_log):
            """Check if there's existing work in the temp directory and restore it."""
            temp_dir = get_temp_dir()
            if not temp_dir:
                return (current_log, gr.State(None), gr.update(), gr.update(), gr.update(),
                        *[gr.update() for _ in range(6)])

            # Check for existing work
            chapters_path = Path(temp_dir)
            if not chapters_path.exists():
                return (current_log, gr.State(None), gr.update(), gr.update(), gr.update(),
                        *[gr.update() for _ in range(6)])

            # Detect pipeline state from existing files
            state = PipelineState(str(chapters_path))
            detected_state = state.get_pipeline_state()

            if detected_state != "initial":
                # Load the actual state
                state.pipeline_state = detected_state
                state.load_chapter_maps()
                state.get_characters()
                state.load_character_descriptions()
                state.load_voice_map()

                # Update UI
                button_updates = update_button_visibility_from_state(state)
                char_table = update_character_table_from_state(state)
                char_gallery = update_character_gallery_from_state(state)
                chap_progress = update_chapter_progress_from_state(state)
                new_log = f"=== Session Restored ===\nState: {detected_state}\n" + current_log
                return (new_log, gr.State(state), char_table, char_gallery, chap_progress, *button_updates)

            return (current_log, gr.State(None), gr.update(), gr.update(), gr.update(),
                    *[gr.update() for _ in range(6)])

        # Always check for restored state on page load
        demo.load(
            fn=check_and_restore_state,
            inputs=[log_output],
            outputs=[log_output, pipeline_state_obj, character_table, character_gallery, chapter_progress,
                     parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn]
        )

        # Also initialize from saved_temp_dir if provided (CLI --resume_from)
        if saved_temp_dir:
            demo.load(
                fn=initialize_ui,
                inputs=[pipeline_state_obj, log_output],
                outputs=[log_output, character_table, character_gallery, chapter_progress,
                         parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn,
                         active_character_row, active_char_name, active_char_desc, active_char_lines, character_audio, generate_char_btn]
            )

        # Load selected audiobook
        def load_selected_audiobook(archive_path):
            """Load a saved audiobook from archive and initialize state."""
            if not archive_path:
                return "No archive selected"
            try:
                # Clear any existing temp context
                cleanup_temp_dir()
                # Load the archive and extract to a new temp dir
                temp_dir = load_temp_dir(archive_path)
                if temp_dir:
                    # Get the chapters directory path for display
                    from utils import get_chapters_dir_from_saved
                    chapters_dir = get_chapters_dir_from_saved(temp_dir)
                    # Note: Pipeline state is not automatically restored here.
                    # User should use --resume_from at startup for full state restoration.
                    return f"Loaded: {chapters_dir}\nNote: Refresh page to see restored state."
                return "Failed to load archive"
            except Exception as e:
                return f"Error loading: {str(e)}"

    return demo


def restore_pipeline_state(saved_temp_dir: str) -> Tuple[PipelineState, str]:
    """Restore PipelineState from a saved temp directory.

    Args:
        saved_temp_dir: Path to the saved temp directory

    Returns:
        Tuple of (PipelineState, log_message)
    """
    from utils import get_chapters_dir_from_saved

    chapters_dir = get_chapters_dir_from_saved(saved_temp_dir)

    # Check for chapter files
    chapter_files = sorted(chapters_dir.glob("chapter_*.txt"), key=natural_sort_key)

    state = PipelineState(str(chapters_dir))

    log_msg = f"Restoring state from: {saved_temp_dir}\n"

    # Detect pipeline state
    state.pipeline_state = state.get_pipeline_state()
    log_msg += f"Detected state: {state.pipeline_state}\n"

    # Load existing data based on state
    # Always load chapters if they exist (applies to all non-initial states)
    chapter_files = sorted(chapters_dir.glob("chapter_*.txt"), key=natural_sort_key)
    if chapter_files and state.pipeline_state != "initial":
        state.chapters = parse_chapter.load_chapters_from_txt(str(chapters_dir))
        log_msg += f"Loaded {len(state.chapters)} chapter(s)\n"

    if state.pipeline_state in ["labels_complete", "characters_described", "voice_samples_complete", "audiobook_complete"]:
        # Load chapter maps
        state.load_chapter_maps()
        log_msg += f"Loaded {len(state.chapter_maps)} chapter map(s)\n"

        # Load characters
        state.get_characters()
        log_msg += f"Found {len(state.characters)} character(s)\n"

    if state.pipeline_state in ["characters_described", "voice_samples_complete", "audiobook_complete"]:
        # Load character descriptions
        state.load_character_descriptions()
        log_msg += f"Loaded {len(state.character_descriptions)} description(s)\n"

    if state.pipeline_state in ["voice_samples_complete", "audiobook_complete"]:
        # Load voice map
        state.load_voice_map()
        log_msg += f"Loaded {len(state.voice_map)} voice mapping(s)\n"

    if state.pipeline_state == "initial":
        log_msg += "No previous work found. Start by parsing an EPUB file.\n"

    return state, log_msg


def refresh_saved_audiobooks() -> gr.update:
    """Helper function to refresh the saved audiobooks dropdown choices.

    Returns:
        gr.update with updated choices list
    """
    return gr.update(choices=[(f"{a['name']} ({a['timestamp']})", a['archive'])
                               for a in get_available_saved_audiobooks()])