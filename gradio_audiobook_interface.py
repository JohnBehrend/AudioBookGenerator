#!/usr/bin/env python3
"""
Gradio Interface for Audiobook Pipeline

A unified web interface for the 6-stage audiobook creation pipeline:
1. EPUB Parsing
2. LLM Speaker Labeling
3. Character Descriptions
4. Voice Sample Generation
5. Full Audiobook Generation

State machine pattern ensures each stage only runs when dependencies are met.
"""

import gradio as gr
import os
import sys
import json
import glob
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List


# ============================================================================
# CONSTANTS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Pipeline States
PIPELINE_STATE_EPUB_PARSED = "epub_parsed"
PIPELINE_STATE_LABELS_COMPLETE = "labels_complete"
PIPELINE_STATE_CHARACTERS_DESCRIBED = "characters_described"
PIPELINE_STATE_VOICE_SAMPLES_COMPLETE = "voice_samples_complete"
PIPELINE_STATE_AUDIOBOOK_COMPLETE = "audiobook_complete"

# File paths - these are now managed dynamically per run

# Default UI values
DEFAULT_API_KEY = "lm-studio"
DEFAULT_PORT = "1234"
DEFAULT_NUM_ATTEMPTS = 10
DEFAULT_MAX_CHAPTERS = 10

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_chapters_dir() -> Optional[Path]:
    """Get or create a temporary chapters directory for this session."""
    if not hasattr(get_chapters_dir, "_temp_dir"):
        get_chapters_dir._temp_dir = tempfile.mkdtemp(prefix="jbab_chapters_")
    if not hasattr(get_chapters_dir, "_chapters_dir"):
        get_chapters_dir._chapters_dir = Path(get_chapters_dir._temp_dir) / "chapters"
        get_chapters_dir._chapters_dir.mkdir(parents=True, exist_ok=True)
    return get_chapters_dir._chapters_dir


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


def get_pipeline_state() -> Optional[str]:
    """Get the current pipeline state based on existing files."""
    chapters_dir = get_chapters_dir()
    if not chapters_dir:
        return None

    # Check for Stage 1 completion
    chapter_files = sorted(glob.glob(str(chapters_dir / "chapter_*.txt")))
    if not chapter_files:
        return None

    # Check for Stage 2 completion (map files)
    map_files = sorted(glob.glob(str(chapters_dir / "*.map.json")))
    if not map_files:
        return PIPELINE_STATE_EPUB_PARSED

    # Check for Stage 3 completion (characters descriptions)
    descriptions_file = get_characters_descriptions_file()
    if not descriptions_file or not descriptions_file.exists():
        return PIPELINE_STATE_LABELS_COMPLETE

    # Check for Stage 4 completion (voice samples)
    wav_files = list(chapters_dir.glob("*.wav"))
    if not wav_files:
        return PIPELINE_STATE_CHARACTERS_DESCRIBED

    # Check for Stage 5 completion (final audiobook)
    mp3_files = sorted(glob.glob(str(chapters_dir / "chapter_*.mp3")))
    if not mp3_files:
        return PIPELINE_STATE_VOICE_SAMPLES_COMPLETE

    return PIPELINE_STATE_AUDIOBOOK_COMPLETE


def get_characters_from_map_files(chapters_dir: Path) -> List[str]:
    """Extract unique character names from map.json files."""
    characters = set()

    map_files = glob.glob(str(chapters_dir / "*.map.json"))
    for map_file in map_files:
        try:
            with open(map_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            character_map = data[0] if isinstance(data, list) and len(data) > 0 else data.get("character_map", {})
            if isinstance(character_map, dict):
                for char_name in character_map.values():
                    if isinstance(char_name, str):
                        characters.add(char_name)
        except Exception:
            pass

    characters.discard("narrator")
    return sorted(list(characters))


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
    epub_file, max_chapters: Optional[int]
) -> Tuple[str, int, List[str]]:
    """Stage 1: Parse EPUB file into chapter text files."""
    if epub_file is None:
        return "Error: No EPUB file uploaded.", 0, []

    try:
        from parse_chapter import parse_epub_to_chapters

        chapters_dir = get_chapters_dir()
        if not chapters_dir:
            return "Error: Failed to create temporary directory.", 0, []

        # Copy the EPUB file to temp directory for Stage 6
        epub_dest = chapters_dir / "uploaded.epub"
        shutil.copy2(epub_file.name, str(epub_dest))

        # Parse the EPUB with max_chapters limit if specified
        chapters = parse_epub_to_chapters(
            epub_file.name,
            max_chapters=int(max_chapters) if max_chapters else None
        )

        if not chapters:
            return "Error: No chapters found in EPUB file.", 0, []

        # Save each chapter as a text file
        chapter_files = []
        total_chapters = len(chapters)
        for i, chapter in enumerate(chapters):
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
            chapter_files.append(str(output_file))

        return f"Successfully parsed {len(chapters)} chapters.", len(chapters), chapter_files

    except Exception as e:
        return f"Error parsing EPUB: {str(e)}", 0, []


# ============================================================================
# STAGE 2: LLM SPEAKER LABELING
# ============================================================================


def process_chapters_for_labels(
    api_key: str,
    port: str,
    num_attempts: int,
    pipeline_state: Optional[str],
    log_output: str,
) -> Tuple[str, str]:
    """Stage 2: Run LLM to label speakers in all chapters."""
    chapters_dir = get_chapters_dir()
    if not chapters_dir:
        return log_output + "\nError: Chapters directory not initialized.", pipeline_state

    chapter_files = sorted(glob.glob(str(chapters_dir / "chapter_*.txt")))

    if not chapter_files:
        return log_output + "\nNo chapter files found. Please run Stage 1 (Parse EPUB) first.", pipeline_state

    num_chapters = len(chapter_files)
    log_output += f"\nProcessing {num_chapters} chapters with LLM..."

    for i, chapter_file in enumerate(chapter_files):
        log_output += f"\nProcessing: {chapter_file}"

        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "llm_label_speakers.py"),
            "-txt_file", chapter_file,
            "-num_llm_attempts", str(num_attempts),
            "-api_key", api_key,
            "-port", port,
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,  # 5 minute timeout per chapter
                cwd=str(SCRIPT_DIR),
            )
            log_output += result.stdout
            if result.stderr:
                log_output += f"\nErrors: {result.stderr}"
        except subprocess.TimeoutExpired:
            log_output += "\nTimeout - chapter processing took too long."
        except Exception as e:
            log_output += f"\nError processing {chapter_file}: {str(e)}"

    new_state = PIPELINE_STATE_LABELS_COMPLETE
    log_output += f"\n\nStage 2 complete! State: {new_state}"
    return log_output, new_state


# ============================================================================
# STAGE 3: CHARACTER DESCRIPTIONS
# ============================================================================


def describe_characters(
    api_key: str,
    port: str,
    pipeline_state: Optional[str],
    log_output: str,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Stage 3: Use LLM to describe characters."""
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

        # Extract characters directly from map files
        characters = get_characters_from_map_files(chapters_dir)
        num_characters = len(characters)

        if num_characters == 0:
            log_output += "\nNo characters found in map files. Please run Stage 2 first."
            return log_output, pipeline_state, None

        log_output += f"\nFound {num_characters} characters from map files."

        # Run llm_describe_character.py
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "llm_describe_character.py"),
            "--api_key", api_key,
            "--port", port,
            "--verbose",
            "--output-dir", str(chapters_dir),
        ]

        # Write characters to a temp file for the script to read
        temp_characters_file = chapters_dir / "temp_characters.json"
        with open(temp_characters_file, "w", encoding="utf-8") as f:
            json.dump({"characters": characters}, f)

        cmd.insert(2, str(temp_characters_file))
        cmd.insert(3, str(chapters_dir))

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(SCRIPT_DIR),
        )

        # Read output line by line to track progress
        for line in iter(process.stdout.readline, ""):
            if line:
                log_output += f"\n{line.strip()}"

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            if process.stderr:
                log_output += f"\nErrors: {process.stderr.read()}"
            log_output += "\nWarning: Process exited with non-zero code."

        log_output += "\n\nStage 3 complete!"

        # Load and return the character descriptions for state tracking
        descriptions_file = get_characters_descriptions_file()
        try:
            with open(descriptions_file, "r", encoding="utf-8") as f:
                characters_state = json.load(f)
        except Exception:
            characters_state = {}

        new_state = PIPELINE_STATE_CHARACTERS_DESCRIBED
        log_output += f" State: {new_state}"
        return log_output, new_state, characters_state

    except Exception as e:
        log_output += f"\nError describing characters: {str(e)}"
        return log_output, pipeline_state, None


# ============================================================================
# STAGE 4: VOICE SAMPLE GENERATION
# ============================================================================


def generate_voice_samples(
    pipeline_state: Optional[str],
    log_output: str,
) -> Tuple[str, str]:
    """Stage 4: Generate voice samples for each character."""
    log_output += "\n\nGenerating voice samples..."

    try:
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
        log_output += f"\nFound {num_characters} characters to process."

        # Run generate_voice_samples.py
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "generate_voice_samples.py"),
            "--descriptions", str(descriptions_file),
            "--output-dir", str(chapters_dir),
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(SCRIPT_DIR),
        )

        # Read output line by line to track progress
        for line in iter(process.stdout.readline, ""):
            if line:
                log_output += f"\n{line.strip()}"

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            if process.stderr:
                log_output += f"\nErrors: {process.stderr.read()}"
            log_output += "\nWarning: Process exited with non-zero code."

        log_output += "\n\nStage 4 complete!"

        new_state = PIPELINE_STATE_VOICE_SAMPLES_COMPLETE
        log_output += f" State: {new_state}"
        return log_output, new_state

    except Exception as e:
        log_output += f"\nError generating voice samples: {str(e)}"
        return log_output, pipeline_state


def regenerate_voice_sample(
    character_name: str,
    pipeline_state: Optional[str],
    log_output: str,
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

        # Build command to regenerate just this character
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "generate_voice_samples.py"),
            "--descriptions", str(descriptions_file),
            "--output-dir", str(chapters_dir),
            "--single-character", character_name,
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(SCRIPT_DIR),
        )

        for line in iter(process.stdout.readline, ""):
            if line:
                log_output += f"\n{line.strip()}"

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            if process.stderr:
                log_output += f"\nErrors: {process.stderr.read()}"
            log_output += "\nWarning: Process exited with non-zero code."
            return log_output, pipeline_state, None

        # Return the path to the regenerated file
        wav_path = get_character_wav_file(character_name, chapters_dir)
        log_output += f"\n\nVoice sample regenerated for: {character_name}"
        return log_output, pipeline_state, wav_path

    except Exception as e:
        log_output += f"\nError regenerating voice sample: {str(e)}"
        return log_output, pipeline_state, None


# ============================================================================
# STAGE 5: FULL AUDIOBOOK GENERATION
# ============================================================================


def generate_tts_audio(
    pipeline_state: Optional[str],
    log_output: str,
    max_chapters: Optional[int],
) -> Tuple[str, Optional[str]]:
    """Stage 5.1: Generate TTS audio for each line/voice."""
    log_output += "\n\n=== Stage 5.1: Generating TTS Audio ==="

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

        # Check if uploaded EPUB exists in temp directory
        epub_path = str(chapters_dir / "uploaded.epub")
        if not os.path.exists(epub_path):
            log_output += "\nUploaded EPUB file not found. Please run Stage 1 (Parse EPUB) first."
            return log_output, pipeline_state

        # Count chapters for progress tracking
        chapter_files = sorted(glob.glob(str(chapters_dir / "chapter_*.txt")))
        num_chapters = len(chapter_files)
        if max_chapters:
            num_chapters = min(num_chapters, int(max_chapters))
        log_output += f"\nGenerating TTS audio for {num_chapters} chapters..."

        # Run parse_epub.py with --resume to generate TTS audio only
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "parse_epub.py"),
            epub_path,
            "--resume",
            "--output-dir",
            str(chapters_dir),
        ]
        if max_chapters:
            cmd.extend(["--max-chapters", str(max_chapters)])

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(SCRIPT_DIR),
        )

        # Track TTS generation progress
        for line in iter(process.stdout.readline, ""):
            if line:
                log_output += f"\n{line.strip()}"

        process.stdout.close()
        process.wait()

        log_output += "\n\nStage 5.1 (TTS Audio) complete!"
        return log_output, pipeline_state

    except Exception as e:
        log_output += f"\nError generating TTS audio: {str(e)}"
        return log_output, pipeline_state


def assemble_chapter_audiobooks(
    pipeline_state: Optional[str],
    log_output: str,
) -> Tuple[str, Optional[str]]:
    """Stage 5.3: Assemble final chapter MP3 files from WAV files."""
    log_output += "\n\n=== Stage 5.3: Assembling Chapter MP3 Files ==="

    try:
        chapters_dir = get_chapters_dir()
        if not chapters_dir:
            return log_output + "\nError: Chapters directory not initialized.", pipeline_state

        chapter_files = sorted(glob.glob(str(chapters_dir / "chapter_*.txt")))
        if not chapter_files:
            log_output += "\nNo chapter text files found. Please run Stage 5.1 first."
            return log_output, pipeline_state

        num_chapters = len(chapter_files)
        log_output += f"\nFound {num_chapters} chapters to assemble."

        def get_non_silent_audio_from_wavs(wav_filepath_list, min_silence_len=1250, silence_thresh=-60):
            """Remove silent audio from list of wave filepaths."""
            all_audio_segments = None
            for wav in wav_filepath_list:
                raw_audio_segment = gr.Audio.from_wav(wav)
                this_audio_segment = gr.Audio.empty()
                for (start_time, end_time) in gr.Audio.silence.detect_nonsilent(
                    raw_audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh
                ):
                    this_audio_segment += raw_audio_segment[start_time:end_time]
                if all_audio_segments is None:
                    all_audio_segments = this_audio_segment
                else:
                    all_audio_segments = all_audio_segments + this_audio_segment
            return all_audio_segments

        # Process each chapter
        for i in range(len(chapter_files)):
            # Find all WAV files for this chapter
            wav_files = sorted(glob.glob(str(chapters_dir / f"chapter_{str(i).zfill(2)}.*.wav")))

            if not wav_files:
                log_output += f"\nNo WAV files found for chapter {i}, skipping."
                continue

            # Combine audio files
            audio = get_non_silent_audio_from_wavs(wav_files)
            output_mp3 = chapters_dir / f"chapter_{str(i).zfill(2)}.mp3"
            audio.export(str(output_mp3), format="mp3")

            # Clean up individual WAV files
            for wav in wav_files:
                os.unlink(wav)

            log_output += f"\nChapter {i}: Created {output_mp3.name} from {len(wav_files)} audio segments."

        log_output += "\n\nStage 5.3 (Assembly) complete!"
        return log_output, pipeline_state

    except Exception as e:
        log_output += f"\nError assembling audiobooks: {str(e)}"
        return log_output, pipeline_state


def generate_full_audiobook(
    pipeline_state: Optional[str],
    log_output: str,
    max_chapters: Optional[int],
) -> Tuple[str, str]:
    """Stage 5: Generate full audiobook - runs TTS generation and assembly."""
    log_output += "\n\n=== Stage 5: Full Audiobook Generation ==="

    # Step 5.1: Generate TTS audio
    log_output, pipeline_state = generate_tts_audio(pipeline_state, log_output, max_chapters)

    # Step 5.2: Assemble chapter MP3s
    log_output, pipeline_state = assemble_chapter_audiobooks(pipeline_state, log_output)

    # Update state to audiobook complete
    new_state = PIPELINE_STATE_AUDIOBOOK_COMPLETE
    log_output += f"\n\n=== Stage 5 Complete: Full Audiobook Generation === State: {new_state}"
    return log_output, new_state


# ============================================================================
# UI HELPERS
# ============================================================================


def update_character_table(characters_state: Optional[Dict[str, Any]]) -> gr.Dataframe:
    """
    Update the character table based on stored character state.

    Args:
        characters_state: gr.State containing dict of character_name -> description

    Returns:
        Updated Dataframe component with character data (2 columns: name, truncated_description)
    """
    descriptions_file = get_characters_descriptions_file()
    if not descriptions_file or not descriptions_file.exists() or characters_state is None:
        return gr.Dataframe(value=[])

    try:
        with open(descriptions_file, "r", encoding="utf-8") as f:
            descriptions = json.load(f)

        # Build table data with only character name and truncated description
        table_data = []
        for char_name, char_desc in descriptions.items():
            truncated_desc = char_desc[:100] + ("..." if len(char_desc) > 100 else "")
            table_data.append([char_name, truncated_desc])

        return gr.Dataframe(
            headers=["Character", "Description"],
            datatype=["str", "str"],
            value=table_data,
            wrap=True,
        )

    except Exception:
        return gr.Dataframe(value=[])


def get_next_step_recommendation(state: Optional[str]) -> str:
    """Get a recommendation for the next step based on pipeline state."""
    recommendations = {
        None: "Upload an EPUB file and click '1. Parse' to begin.",
        PIPELINE_STATE_EPUB_PARSED: "Click '2. Label' to use LLM for speaker labeling.",
        PIPELINE_STATE_LABELS_COMPLETE: "Click '3. Describe' to generate character descriptions.",
        PIPELINE_STATE_CHARACTERS_DESCRIBED: "Click '4. Voices' to generate voice samples for each character.",
        PIPELINE_STATE_VOICE_SAMPLES_COMPLETE: "Click '6. Audiobook' to generate the full audiobook.",
        PIPELINE_STATE_AUDIOBOOK_COMPLETE: "Audiobook generation complete! You can start a new project.",
    }
    return recommendations.get(state, "Unknown state.")


def update_state_display(state: Optional[str]) -> gr.Textbox:
    """Update log label with state based on pipeline state."""
    state_labels = {
        None: "Ready",
        PIPELINE_STATE_EPUB_PARSED: "EPUB Parsed",
        PIPELINE_STATE_LABELS_COMPLETE: "Speakers Labeled",
        PIPELINE_STATE_CHARACTERS_DESCRIBED: "Characters Described",
        PIPELINE_STATE_VOICE_SAMPLES_COMPLETE: "Voice Samples Ready",
        PIPELINE_STATE_AUDIOBOOK_COMPLETE: "Audiobook Complete",
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
        return (
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    elif state == PIPELINE_STATE_EPUB_PARSED:
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    elif state == PIPELINE_STATE_LABELS_COMPLETE:
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    elif state == PIPELINE_STATE_CHARACTERS_DESCRIBED:
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
    elif state == PIPELINE_STATE_VOICE_SAMPLES_COMPLETE:
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    else:  # AUDIOBOOK_COMPLETE or beyond
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )


# ============================================================================
# GRADIO INTERFACE
# ============================================================================


def create_interface(
    api_key_default: str = DEFAULT_API_KEY,
    port_default: str = DEFAULT_PORT,
    num_attempts_default: int = DEFAULT_NUM_ATTEMPTS,
    epub_path_default: Optional[str] = None,
    max_chapters_default: int = DEFAULT_MAX_CHAPTERS,
):
    """Create the Gradio interface with all stages using a state machine pattern."""

    with gr.Blocks() as demo:
        gr.Markdown("# Audiobook Pipeline")

        # Progress bar at top
        progress_bar = gr.Progress()

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

        # All 6 buttons in a single row
        with gr.Row():
            parse_btn = gr.Button("1. Parse", variant="primary", scale=1)
            label_btn = gr.Button("2. Label", variant="secondary", scale=1)
            describe_btn = gr.Button("3. Describe", variant="secondary", scale=1)
            voice_samples_btn = gr.Button("4. Voices", variant="secondary", scale=1)
            tts_btn = gr.Button("6. Audiobook", variant="primary", scale=1)

        # Log output with state on same element
        log_output = gr.Textbox(label="Log (State: Ready)", lines=4, max_lines=6)

        # Character info
        character_table = gr.Dataframe(
            headers=["Character", "Description"],
            datatype=["str", "str"],
            wrap=True,
            max_height=100,
        )

        # Audio player with Regen button
        with gr.Row():
            character_audio = gr.Audio(label="", type="filepath", visible=False)
            generate_char_btn = gr.Button("Regen", variant="secondary", scale=1, visible=False)

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
            outputs=[log_output, log_output, log_output],  # dummy outputs to satisfy function signature
        ).then(
            # Reset log for new session
            fn=lambda: "=== Stage 1: EPUB Parsing Complete ===\n",
            outputs=log_output,
        ).then(
            fn=lambda: PIPELINE_STATE_EPUB_PARSED,
            outputs=pipeline_state,
        ).then(
            fn=lambda s: list(update_button_visibility(s)) + [update_state_display(s)],
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn, log_output],
        )

        # Label Speakers - Stage 2
        label_btn.click(
            fn=process_chapters_for_labels,
            inputs=[api_key_input, port_input, num_attempts_input, pipeline_state, log_output],
            outputs=[log_output, pipeline_state],
        ).then(
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn],
        ).then(
            fn=update_state_display,
            inputs=pipeline_state,
            outputs=log_output,
        )

        # Describe Characters - Stage 3
        describe_btn.click(
            fn=describe_characters,
            inputs=[api_key_input, port_input, pipeline_state, log_output],
            outputs=[log_output, pipeline_state, characters_state],
        ).then(
            fn=update_character_table,
            inputs=characters_state,
            outputs=character_table,
        ).then(
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn],
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
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn],
        ).then(
            fn=update_state_display,
            inputs=pipeline_state,
            outputs=log_output,
        )

        # Handle row selection in character table - show audio when character selected
        def on_character_select(evt: gr.SelectData, _characters_state, _selected_character):
            """Handle row selection in the character table."""
            if evt is None or evt.index is None:
                return gr.update(visible=False, value=None), gr.update(visible=False), gr.update(value=None)

            character_name = evt.row_value[0] if evt.row_value and len(evt.row_value) > 0 else None

            if not character_name:
                return gr.update(visible=False, value=None), gr.update(visible=False), gr.update(value=None)

            chapters_dir = get_chapters_dir()
            if not chapters_dir:
                return gr.update(visible=False, value=None), gr.update(visible=False), gr.update(value=None)

            wav_path = get_character_wav_file(character_name, chapters_dir)

            if wav_path and os.path.exists(wav_path):
                return gr.update(visible=True, value=wav_path), gr.update(visible=True), gr.update(value=character_name)
            else:
                return gr.update(visible=False, value=None), gr.update(visible=False), gr.update(value=None)

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
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn],
        ).then(
            fn=update_state_display,
            inputs=pipeline_state,
            outputs=log_output,
        ).then(
            fn=update_character_table,
            inputs=characters_state,
            outputs=character_table,
        )

        # Generate Full Audiobook - Stage 5
        tts_btn.click(
            fn=generate_full_audiobook,
            inputs=[pipeline_state, log_output, max_chapters_slider],
            outputs=[log_output, pipeline_state],
        ).then(
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn],
        ).then(
            fn=update_state_display,
            inputs=pipeline_state,
            outputs=log_output,
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