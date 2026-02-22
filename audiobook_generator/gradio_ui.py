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

# ============================================================================
# Suppress sox binary warning (sox Python package requires sox CLI binary)
# ============================================================================
# The sox package prints a warning to stderr when the sox binary is not found.
# We suppress this by temporarily redirecting stderr during the sox import.
# This must run BEFORE any imports that might trigger qwen-tts -> sox.

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
import tempfile
import shutil
import traceback
import atexit
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from time import sleep

# Import from package modules - using clean public interfaces
import parse_chapter
from llm_label_speakers import label_speakers  # Clean public function
from llm_describe_character import describe_characters as describe_chars  # Clean public function
from generate_voice_samples import generate_voice_samples as gen_voice_samples
from utils import get_characters_from_map_files  # Use shared version
from audiobook_generator import (
    PipelineState,
    generate_audiobook_from_chapters,
    get_non_silent_audio_from_wavs,
    VoiceMapper,
    setup_tts_engine,
    setup_validation_model,
)
from config import DEFAULTS, LLM_SETTINGS, AUDIO_SETTINGS, DEFAULT_EPUB_FILE


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
# ============================================================================


def get_chapters_dir() -> Optional[Path]:
    """Get or create a temporary chapters directory for this session."""
    if not hasattr(get_chapters_dir, "_temp_dir"):
        # Use TemporaryDirectory which auto-cleans on program exit
        get_chapters_dir._temp_context = tempfile.TemporaryDirectory(prefix="jbab_chapters_")
        get_chapters_dir._temp_dir = get_chapters_dir._temp_context.name
        # Register cleanup on normal exit
        atexit.register(cleanup_temp_dir)
    if not hasattr(get_chapters_dir, "_chapters_dir"):
        get_chapters_dir._chapters_dir = Path(get_chapters_dir._temp_dir) / "chapters"
        get_chapters_dir._chapters_dir.mkdir(parents=True, exist_ok=True)
    return get_chapters_dir._chapters_dir


def get_temp_dir() -> Optional[str]:
    """Get the temporary directory path for display purposes."""
    if hasattr(get_chapters_dir, "_temp_dir") and get_chapters_dir._temp_dir:
        return get_chapters_dir._temp_dir
    return None


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


def cleanup_temp_dir() -> None:
    """Clean up the temporary directory when done."""
    if hasattr(get_chapters_dir, "_temp_dir") and get_chapters_dir._temp_dir:
        temp_dir = Path(get_chapters_dir._temp_dir)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        get_chapters_dir._temp_dir = None
        get_chapters_dir._chapters_dir = None
    # Clean up the TemporaryDirectory context if it exists
    if hasattr(get_chapters_dir, "_temp_context") and get_chapters_dir._temp_context:
        get_chapters_dir._temp_context.cleanup()
        get_chapters_dir._temp_context = None


def get_pipeline_state() -> Optional[str]:
    """Get the current pipeline state based on existing files."""
    chapters_dir = get_chapters_dir()
    if not chapters_dir:
        return None

    # Use PipelineState class for state determination
    state_manager = PipelineState(str(chapters_dir.parent))
    state_manager.chapters_dir = chapters_dir
    return state_manager.get_pipeline_state()


def get_character_wav_file(character_name: str, chapters_dir: Path) -> Optional[str]:
    """Get the path to a character's generated WAV file."""
    for base_dir in [chapters_dir, SCRIPT_DIR]:
        wav_path = base_dir / f"{character_name}.wav"
        if wav_path.exists():
            return str(wav_path)
    return None


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


def progress_iterator(items, progress=None, desc="Processing"):
    """Generator that yields items and updates progress bar if available."""
    if progress is None:
        for item in items:
            yield item
        return

    total = len(items) if hasattr(items, "__len__") else None
    for i, item in enumerate(items):
        if total:
            progress((i + 1) / total, desc=f"{desc} ({i+1}/{total})")
        else:
            progress(i, desc=f"{desc} ({i+1})")
        yield item


# ============================================================================
# STAGE 1: EPUB PARSING
# ============================================================================


def parse_epub_to_file(
    epub_file, max_chapters: Optional[int], progress=gr.Progress()
) -> Tuple[str, int]:
    """Stage 1: Parse EPUB file into chapter text files."""
    if epub_file is None:
        return "Error: No EPUB file uploaded.", 0

    chapters_dir = get_chapters_dir()
    if not chapters_dir:
        return "Error: Failed to create temporary directory.", 0

    try:
        # Clean up existing files in the chapters directory before starting fresh
        if chapters_dir.exists():
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
            return "Error: No chapters found in EPUB file.", 0

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
        return "=== Stage 1: EPUB Parsing Complete ===\n", "epub_parsed"
    except Exception as e:
        log_output = f"Error parsing EPUB: {str(e)}"
        log_output += f"\n{traceback.format_exc()}"
        return log_output, 0


# ============================================================================
# STAGE 2: LLM SPEAKER LABELING
# ============================================================================


def process_chapters_for_labels(
    api_key: str,
    port: str,
    num_attempts: int,
    pipeline_state: Optional[str],
    log_output: str,
    progress=gr.Progress()
) -> Tuple[str, str, List[str]]:
    """Stage 2: Run LLM to label speakers in all chapters."""
    chapters_dir = get_chapters_dir()
    if not chapters_dir:
        progress(1.0, desc=f"Error: Chapters directory not initialized. (temp: {chapters_dir.parent})")
        return log_output + "\nError: Chapters directory not initialized.", pipeline_state, []

    chapter_files = sorted(glob.glob(str(chapters_dir / "chapter_*.txt")))

    if not chapter_files:
        progress(1.0, desc="No chapter files found. Please run Stage 1 (Parse EPUB) first.")
        return log_output + "\nNo chapter files found. Please run Stage 1 (Parse EPUB) first.", pipeline_state, []

    num_chapters = len(chapter_files)
    log_output += f"\nProcessing {num_chapters} chapters with LLM... (temp: {chapters_dir.parent})"

    all_character_names = set()

    for i, chapter_file in enumerate(chapter_files):
        progress(i / num_chapters, desc=f"Labeling speakers in chapter {i + 1}/{num_chapters}... (temp: {chapters_dir.parent})")
        log_output += f"\nProcessing: {chapter_file}"

        try:
            # Import and call directly instead of subprocess
            result_msg, char_map, line_map = label_speakers(
                txt_file=chapter_file,
                api_key=api_key,
                port=port,
                num_attempts=num_attempts,
                verbose=False
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

    new_state = "labels_complete"
    log_output += f"\n\nStage 2 complete! State: {new_state}"

    # Convert to sorted list for consistent ordering
    character_list = sorted(list(all_character_names))
    log_output += f"\nFound {len(character_list)} characters: {', '.join(character_list)}"

    progress(1.0, desc=f"LLM speaker labeling complete. (temp: {chapters_dir.parent})")
    return log_output, new_state, character_list


# ============================================================================
# STAGE 3: CHARACTER DESCRIPTIONS
# ============================================================================


def describe_characters_ui(
    api_key: str,
    port: str,
    pipeline_state: Optional[str],
    log_output: str,
    progress=gr.Progress()
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Stage 3: Use LLM to describe characters."""
    progress(0, desc="Starting character description generation...")
    log_output += "\n\nGenerating character descriptions..."

    try:
        chapters_dir = get_chapters_dir()
        if not chapters_dir:
            return log_output + "\nError: Chapters directory not initialized.", pipeline_state, None

        # Check if map files exist
        map_files = glob.glob(str(chapters_dir / "*.map.json"))
        if not map_files:
            log_output += "\nNo .map.json files found. Please run Stage 2 (Label Speakers) first."
            return log_output, pipeline_state, None

        # Extract characters using PipelineState
        state_manager = PipelineState(str(chapters_dir.parent))
        state_manager.chapters_dir = chapters_dir
        characters = state_manager.get_characters()
        num_characters = len(characters)

        if num_characters == 0:
            log_output += "\nNo characters found in map files. Please run Stage 2 first."
            return log_output, pipeline_state, None

        log_output += f"\nFound {num_characters} characters from map files. (temp: {chapters_dir.parent})"

        # Call describe_chars from llm_describe_character module
        progress(0.5, desc=f"Describing {num_characters} characters with LLM... (temp: {chapters_dir.parent})")
        result_msg, character_descriptions = describe_chars(
            output_dir=str(chapters_dir),
            api_key=api_key,
            port=port,
            verbose=False
        )

        log_output += f"\n{result_msg}"
        progress(1.0, desc=f"Character description generation complete. (temp: {chapters_dir.parent})")
        # Load and return the character descriptions for state tracking
        descriptions_file = get_characters_descriptions_file()
        try:
            with open(descriptions_file, "r", encoding="utf-8") as f:
                characters_state = json.load(f)
        except Exception:
            characters_state = {}

        new_state = "characters_described"
        log_output += f" State: {new_state}"
        return log_output, new_state, characters_state

    except Exception as e:
        log_output += f"\nError describing characters: {str(e)}"
        log_output += f"\n{traceback.format_exc()}"
        return log_output, pipeline_state, None


# ============================================================================
# STAGE 4: VOICE SAMPLE GENERATION
# ============================================================================


def generate_voice_samples(
    pipeline_state: Optional[str],
    log_output: str,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Stage 4: Generate voice samples for each character."""
    log_output += "\n\nGenerating voice samples..."

    try:
        progress(0, desc="Starting voice sample generation...")
        chapters_dir = get_chapters_dir()
        if not chapters_dir:
            return log_output + "\nError: Chapters directory not initialized.", pipeline_state

        descriptions_file = get_characters_descriptions_file()
        if not descriptions_file or not descriptions_file.exists():
            log_output += "\ncharacters_descriptions.json not found. Please run Stage 3 (Describe Characters) first."
            return log_output, pipeline_state

        # Load descriptions to get count for progress
        with open(descriptions_file, "r", encoding="utf-8") as f:
            descriptions = json.load(f)

        num_characters = len(descriptions)
        log_output += f"\nFound {num_characters} characters to process. (temp: {chapters_dir.parent})"

        # Call generate_voice_samples from package
        progress(0, desc=f"Generating voice samples for {num_characters} characters with TTS engine... (temp: {chapters_dir.parent})")
        result_msg, generated_voices = gen_voice_samples(
            descriptions=descriptions,
            output_dir=str(chapters_dir),
            verbose=False,
            progress=progress
        )

        log_output += f"\n{result_msg}"
        progress(1.0, desc=f"Voice sample generation complete. (temp: {chapters_dir.parent})")
        new_state = "voice_samples_complete"
        log_output += f" State: {new_state}"
        return log_output, new_state

    except Exception as e:
        import traceback
        log_output += f"\nError generating voice samples: {str(e)}"
        log_output += f"\n{traceback.format_exc()}"
        return log_output, pipeline_state


def regenerate_voice_sample(
    character_name: str,
    pipeline_state: Optional[str],
    log_output: str,
    progress=gr.Progress()
) -> Tuple[str, Optional[str], Optional[str]]:
    """Regenerate a single voice sample for a character."""
    log_output += f"\n\nRegenerating voice sample for: {character_name}"

    try:
        chapters_dir = get_chapters_dir()
        if not chapters_dir:
            return log_output + "\nError: Chapters directory not initialized.", pipeline_state, None

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

        # Call generate_voice_samples from package
        progress(0, desc=f"Regenerating voice sample for {character_name} with TTS engine...")
        result_msg, generated_voices = gen_voice_samples(
            descriptions={character_name: char_description},
            output_dir=str(chapters_dir),
            single_character=character_name,
            verbose=False,
            progress=progress
        )

        log_output += f"\n{result_msg}"

        # Return the path to the regenerated file
        wav_path = get_character_wav_file(character_name, chapters_dir)
        log_output += f"\n\nVoice sample regenerated for: {character_name} (temp: {chapters_dir.parent})"
        return log_output, pipeline_state, wav_path

    except Exception as e:
        import traceback
        log_output += f"\nError regenerating voice sample: {str(e)}"
        log_output += f"\n{traceback.format_exc()}"
        return log_output, pipeline_state, None


# ============================================================================
# STAGE 5: FULL AUDIOBOOK GENERATION
# ============================================================================


def generate_tts_audio(
    pipeline_state: Optional[str],
    log_output: str,
    max_chapters: Optional[int],
    progress=gr.Progress()
) -> Tuple[str, Optional[str]]:
    """Stage 5.1: Generate TTS audio for each line/voice."""
    log_output += "\n\n=== Stage 5.1: Generating TTS Audio ==="
    progress(0, desc=f"Starting TTS audio generation... (temp: {get_chapters_dir().parent})")
    try:
        chapters_dir = get_chapters_dir()
        if not chapters_dir:
            return log_output + "\nError: Chapters directory not initialized.", pipeline_state

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

        # Create voices_map: character_name -> voice_path (wav file)
        voices_map = {}
        for char_name in descriptions.keys():
            progress(0, desc=f"Finding voice sample for character: {char_name}... (temp: {chapters_dir.parent})")
            wav_path = get_character_wav_file(char_name, chapters_dir)
            if wav_path and os.path.exists(wav_path):
                voices_map[char_name] = wav_path
            else:
                # Use a default narrator voice if no sample found
                narrator_path = get_character_wav_file("narrator", chapters_dir)
                if narrator_path and os.path.exists(narrator_path):
                    voices_map[char_name] = narrator_path
        progress(0.5, desc=f"Prepared voices for {len(voices_map)} characters. (temp: {chapters_dir.parent})")
        if not voices_map:
            log_output += "\nNo voice samples found. Please run Stage 4 (Generate Voices) first."
            return log_output, pipeline_state

        # Load chapter maps for character/line mapping using PipelineState
        temp_output_dir = str(chapters_dir.parent) if chapters_dir else "chapters"
        state = PipelineState(temp_output_dir)
        state.chapters_dir = chapters_dir
        chapter_maps = state.load_chapter_maps()

        # Count chapters for progress tracking
        chapter_files = sorted(glob.glob(str(chapters_dir / "chapter_*.txt")))
        num_chapters = len(chapter_files)
        if max_chapters:
            num_chapters = min(num_chapters, int(max_chapters))
        log_output += f"\nGenerating TTS audio for {num_chapters} chapters... (temp: {chapters_dir.parent})"

        # Parse the EPUB to get chapters (same as Stage 1)
        epub_path = str(chapters_dir / "uploaded.epub")

        chapters = parse_chapter.parse_epub_to_chapters(
            epub_path,
            max_chapters=int(max_chapters) if max_chapters else None
        )

        if not chapters:
            log_output += "\nError: No chapters found in EPUB file."
            return log_output, pipeline_state

        # Determine device (use CUDA if available, default to cuda:0)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_output += f"\nUsing device: {device}"

        # Setup validation model (WhisperX) for audio validation
        # This enables ratio-based validation of generated audio
        try:
            validation_model = setup_validation_model(device)
            log_output += "\nValidation model (WhisperX) loaded successfully."
        except Exception as e:
            log_output += f"\nWarning: Could not load validation model: {e}. Audio validation will be skipped."
            validation_model = None

        # Use the unified generate_audiobook_from_chapters function from package
        verbose = True
        tts_engine = os.environ.get('TTS_ENGINE', AUDIO_SETTINGS["default_tts_engine"])
        cfg_scale = 1.30

        status, processed = generate_audiobook_from_chapters(
            chapters=chapters,
            chapter_maps=chapter_maps,
            voices_map=voices_map,
            output_dir=str(chapters_dir),
            device=device,
            tts_engine=tts_engine,
            cfg_scale=cfg_scale,
            max_chapters=max_chapters,
            verbose=verbose,
            progress=progress,
            validation_model=validation_model
        )

        log_output += f"\n{status}"
        log_output += "\n\nStage 5.1 (TTS Audio) complete!"
        return log_output, pipeline_state

    except Exception as e:
        import traceback
        log_output += f"\nError generating TTS audio: {str(e)}"
        log_output += f"\n{traceback.format_exc()}"
        return log_output, pipeline_state


def generate_full_audiobook(
    pipeline_state: Optional[str],
    log_output: str,
    max_chapters: Optional[int],
) -> Tuple[str, str]:
    """Stage 5: Generate full audiobook using generate_audiobook_from_chapters()."""
    log_output += "\n\n=== Stage 5: Full Audiobook Generation ==="

    try:
        # Use the unified generate_audiobook_from_chapters function
        log_output, pipeline_state = generate_tts_audio(pipeline_state, log_output, max_chapters)

        # Update state to audiobook complete (MP3s are created during generate_audiobook_from_chapters)
        new_state = "audiobook_complete"
        log_output += f"\n\n=== Stage 5 Complete: Full Audiobook Generation === State: {new_state}"
        return log_output, new_state
    except Exception as e:
        log_output += f"\nError generating audiobook: {str(e)}"
        log_output += f"\n{traceback.format_exc()}"
        return log_output, pipeline_state or "error"


# ============================================================================
# UI HELPERS
# ============================================================================


def count_lines_per_character(chapters_dir: Path) -> Dict[str, int]:
    """
    Count lines spoken per character from map.json files.

    Args:
        chapters_dir: Path to the chapters directory

    Returns:
        Dict mapping character name to line count (including narrator for unlabeled lines)
    """
    character_lines = {}

    map_files = sorted(glob.glob(str(chapters_dir / "*.map.json")))
    for map_file in map_files:
        try:
            with open(map_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # data is typically [character_map, line_map]
            if isinstance(data, list) and len(data) >= 2:
                character_map = data[0]
                line_map = data[1]
            elif isinstance(data, dict):
                character_map = data.get("character_map", {})
                line_map = data.get("line_map", {})
            else:
                continue

            # Convert character_map values to get character names by ID
            char_by_id = {int(k): v for k, v in character_map.items()}

            # Count total spoken lines in this chapter from the corresponding txt file
            # Only count lines that actually have content (not just line headers)
            txt_file = map_file.replace(".map.json", ".txt")
            spoken_lines = set()
            if os.path.exists(txt_file):
                with open(txt_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        # Only count lines that start with "Line N:" and have content after
                        if line.startswith("Line ") and ": " in line:
                            # Extract line number and check if there's actual content
                            try:
                                line_num_str = line.split(": ", 1)[0].replace("Line ", "")
                                line_num = int(line_num_str)
                                content = line.split(": ", 1)[1] if ": " in line else ""
                                if content.strip():
                                    spoken_lines.add(line_num)
                            except (ValueError, IndexError):
                                pass

            # Count labeled lines for each character
            labeled_count = 0
            for char_id in line_map.values():
                char_name = char_by_id.get(char_id)
                if char_name:
                    character_lines[char_name] = character_lines.get(char_name, 0) + 1
                    labeled_count += 1

            # Add unlabeled spoken lines to narrator
            narrator_lines = len(spoken_lines) - labeled_count
            if narrator_lines > 0:
                character_lines["narrator"] = character_lines.get("narrator", 0) + narrator_lines

        except Exception:
            pass

    return character_lines


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


def get_next_step_recommendation(state: Optional[str]) -> str:
    """Get a recommendation for the next step based on pipeline state."""
    recommendations = {
        None: "Upload an EPUB file and click '1. Parse' to begin.",
        "epub_parsed": "Click '2. Label' to use LLM for speaker labeling.",
        "labels_complete": "Click '3. Describe' to generate character descriptions.",
        "characters_described": "Click '4. Voices' to generate voice samples for each character.",
        "voice_samples_complete": "Click 'Read Chapters' to generate the full audiobook.",
        "audiobook_complete": "Audiobook generation complete! You can start a new project.",
    }
    return recommendations.get(state, "Unknown state.")


def update_state_display(state: Optional[str]) -> gr.Textbox:
    """Update log label with state based on pipeline state."""
    state_labels = {
        None: "Ready",
        "epub_parsed": "EPUB Parsed",
        "labels_complete": "Speakers Labeled",
        "characters_described": "Characters Described",
        "voice_samples_complete": "Voice Samples Ready",
        "audiobook_complete": "Audiobook Complete",
    }
    state_text = state_labels.get(state, "Unknown")
    next_step = get_next_step_recommendation(state)
    return gr.update(label=f"Log (State: {state_text}) - {next_step}")


def update_button_visibility(state: Optional[str]):
    """
    Update button enabled state based on pipeline state.
    Returns tuple of updates for all buttons and components.
    """
    if state is None:
        # Initial state: only Parse EPUB is enabled
        _states = [True, False, False, False, False, False, False]
    elif state == "epub_parsed":
        _states = [True,  True, False, False, False, False, False]
    elif state == "labels_complete":
        _states = [True,  True,  True, False, False, False, False]
    elif state == "characters_described":
        _states = [True,  True,  True,  True,  True, False, False]
    elif state == "voice_samples_complete":
        _states = [True,  True,  True,  True,  True,  True, True]
    else:  # AUDIOBOOK_COMPLETE or beyond
        _states = [True,  True,  True,  True,  True,  True,  True]
    return tuple(gr.update(interactive=state) for state in _states)


# ============================================================================
# GRADIO INTERFACE
# ============================================================================


def create_interface(
    api_key_default: str = DEFAULT_API_KEY,
    port_default: str = DEFAULT_PORT,
    num_attempts_default: int = DEFAULT_NUM_ATTEMPTS,
    epub_path_default: Optional[str] = str(DEFAULT_EPUB_FILE),
    max_chapters_default: int = DEFAULT_MAX_CHAPTERS,
):
    """Create the Gradio interface with all stages using a state machine pattern."""

    with gr.Blocks() as demo:
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

            # EPUB upload
            epub_upload = gr.File(label="EPUB", file_types=[".epub"], value=epub_path_default)

        # All 6 buttons in a single row - only Parse is clickable initially
        with gr.Row():
            parse_btn = gr.Button("1. Parse", variant="primary", scale=1, interactive=True)
            label_btn = gr.Button("2. Label", variant="secondary", scale=1, interactive=False)
            describe_btn = gr.Button("3. Describe", variant="secondary", scale=1, interactive=False)
            voice_samples_btn = gr.Button("4. Voices", variant="secondary", scale=1, interactive=False)
            tts_btn = gr.Button("Read Chapters", variant="primary", scale=1, interactive=False)
        with gr.Row():
            # Log output with state on same element
            log_output = gr.Textbox(label="Log (State: Ready)", lines=3, max_lines=3, interactive=False)
        with gr.Row():
            stop_btn = gr.Button("Stop", variant="stop", scale=1)
        with gr.Row():
            with gr.Tab("Characters"):
                # Character info
                character_table = gr.Dataframe(
                    headers=["Character", "Description", "Lines Spoken"],
                    datatype=["str", "str", "int"],
                    wrap=True,
                    max_height=300,
                )
                character_audio = gr.Audio(label="", type="filepath", visible=False, scale=1)
                generate_char_btn = gr.Button("Regen", variant="secondary", scale=0, visible=False)
            with gr.Tab("Chapters"):
                chapter_table = gr.Dataframe(
                    headers=["Chapter", "Line", "Character", "Text"],
                    datatype=["str", "str", "str", "str"],
                    wrap=True,
                    max_height=300,
                )
                chapter_audio = gr.Audio(label="", type="filepath", visible=False, scale=1)
                generate_chap_btn = gr.Button("Regen", variant="secondary", scale=0, visible=False)

        # State to track characters
        characters_state = gr.State(None)

        # State to track selected character for Regen button
        selected_character = gr.State(None)

        # State to track pipeline state
        pipeline_state = gr.State(None)

        # ============================================================================
        # EVENT HANDLERS
        # ============================================================================

        # Parse EPUB - Stage 1
        parse_btn.click(
            fn=parse_epub_to_file,
            inputs=[epub_upload, max_chapters_slider],
            outputs=[log_output, pipeline_state],
        ).then(
            fn=lambda: (None, None),  # Clear characters_state and selected_character
            inputs=[],
            outputs=[characters_state, selected_character],
        ).then(
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, generate_chap_btn, tts_btn],
        ).then(
            fn=update_state_display,
            inputs=pipeline_state,
            outputs=log_output,
        )

        # Label Speakers - Stage 2
        label_btn.click(
            fn=process_chapters_for_labels,
            inputs=[api_key_input, port_input, num_attempts_input, pipeline_state, log_output],
            outputs=[log_output, pipeline_state, characters_state],
        ).then(
            fn=lambda log, state, chars: (update_character_table(state, chars), state),
            inputs=[log_output, pipeline_state, characters_state],
            outputs=[character_table, pipeline_state],
        ).then(
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, generate_chap_btn, tts_btn],
        ).then(
            fn=update_state_display,
            inputs=pipeline_state,
            outputs=log_output,
        )

        # Describe Characters - Stage 3
        describe_btn.click(
            fn=describe_characters_ui,
            inputs=[api_key_input, port_input, pipeline_state, log_output],
            outputs=[log_output, pipeline_state, characters_state],
        ).then(
            fn=lambda log, state, chars: (update_character_table(state, chars), state),
            inputs=[log_output, pipeline_state, characters_state],
            outputs=[character_table, pipeline_state],
        ).then(
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, generate_chap_btn, tts_btn],
        ).then(
            fn=update_state_display,
            inputs=pipeline_state,
            outputs=log_output,
        )

        # Generate All Voice Samples - Stage 4
        voice_samples_btn.click(
            fn=generate_voice_samples,
            inputs=[pipeline_state, log_output],
            outputs=[log_output, pipeline_state],
        ).then(
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, generate_chap_btn, tts_btn],
        ).then(
            fn=update_state_display,
            inputs=pipeline_state,
            outputs=log_output,
        )

        # Handle row selection in character table - show audio when character selected
        def on_character_select(evt: gr.SelectData, _characters_state, _selected_character):
            """Handle row selection in the character table."""
            # If clicking away (no row selected), preserve the existing selected character
            if evt is None or evt.index is None:
                return gr.update(visible=False, value=None), gr.update(visible=False), gr.update(value=_selected_character)

            character_name = evt.row_value[0] if evt.row_value and len(evt.row_value) > 0 else None

            if not character_name:
                # If no character name, preserve existing selection
                return gr.update(visible=False, value=None), gr.update(visible=False), gr.update(value=_selected_character)

            chapters_dir = get_chapters_dir()
            if not chapters_dir:
                return gr.update(visible=False, value=None), gr.update(visible=False), gr.update(value=_selected_character)

            wav_path = get_character_wav_file(character_name, chapters_dir)

            if wav_path and os.path.exists(wav_path):
                return gr.update(visible=True, value=wav_path), gr.update(visible=True), gr.update(value=character_name)
            else:
                return gr.update(visible=False, value=None), gr.update(visible=False), gr.update(value=_selected_character)

        character_table.select(
            fn=on_character_select,
            inputs=[characters_state, selected_character],
            outputs=[character_audio, generate_char_btn, selected_character],
        )

        # Regenerate voice sample for selected character
        def on_regenerate_click(pipeline_state, log_output, selected_char):
            """Regenerate voice sample for the selected character."""
            if not selected_char:
                log_output += "\nNo character selected."
                return log_output, pipeline_state, None

            return regenerate_voice_sample(selected_char, pipeline_state, log_output)

        generate_char_btn.click(
            fn=on_regenerate_click,
            inputs=[pipeline_state, log_output, selected_character],
            outputs=[log_output, pipeline_state, character_audio],
        ).then(
            fn=lambda log, state, chars: (update_character_table(state, chars), state),
            inputs=[log_output, pipeline_state, characters_state],
            outputs=[character_table, pipeline_state],
        ).then(
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, generate_chap_btn, tts_btn],
        ).then(
            fn=update_state_display,
            inputs=pipeline_state,
            outputs=log_output,
        )

        # Generate Full Audiobook - Stage 5
        tts_btn.click(
            fn=generate_full_audiobook,
            inputs=[pipeline_state, log_output, max_chapters_slider],
            outputs=[log_output, pipeline_state],
        ).then(
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, generate_chap_btn, tts_btn],
        ).then(
            fn=update_state_display,
            inputs=pipeline_state,
            outputs=log_output,
        )

        # Stop button - clean up and exit
        stop_btn.click(
            fn=cleanup_temp_dir,
            inputs=None,
            outputs=None,
        ).then(
            fn=lambda: sys.exit(0),
            inputs=None,
            outputs=None,
        )

    return demo


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audiobook Pipeline Gradio Interface")
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY, help="LLM API Key to pre-fill")
    parser.add_argument("--port", type=str, default=DEFAULT_PORT, help="LLM Port to pre-fill")
    parser.add_argument("--num_llm_attempts", type=int, default=DEFAULT_NUM_ATTEMPTS, help="Number of LLM attempts to pre-fill")
    parser.add_argument("--epub", type=str, help="EPUB file path to pre-load")
    parser.add_argument("--max_chapters", type=int, default=DEFAULT_MAX_CHAPTERS, help="Max chapters to pre-fill")
    args = parser.parse_args()

    demo = create_interface(
        api_key_default=args.api_key,
        port_default=args.port,
        num_attempts_default=args.num_llm_attempts,
        epub_path_default=args.epub,
        max_chapters_default=args.max_chapters,
    )

    try:
        demo.launch(share=False, theme=gr.themes.Soft())
    finally:
        cleanup_temp_dir()
