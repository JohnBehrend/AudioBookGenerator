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
import re
from pathlib import Path



def progress_iterator(items, progress: gr.Progress = None, desc="Processing"):
    """Generator that yields items and updates progress bar if available."""
    if progress is None:
        for item in items:
            yield item
        return

    total = len(items) if hasattr(items, '__len__') else None
    for i, item in enumerate(items):
        if total:
            progress((i + 1) / total, desc=f"{desc} ({i+1}/{total})")
        else:
            progress(i, desc=f"{desc} ({i+1})")
        yield item


# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Global temp directory for chapters - created per session
TEMP_DIR = None
CHAPTERS_DIR = None


def get_chapters_dir():
    """Get or create a temporary chapters directory for this session."""
    global TEMP_DIR, CHAPTERS_DIR

    if CHAPTERS_DIR is None:
        # Create a temporary directory with a unique name
        TEMP_DIR = tempfile.mkdtemp(prefix="jbab_chapters_")
        CHAPTERS_DIR = Path(TEMP_DIR) / "chapters"
        CHAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    return CHAPTERS_DIR




# ============================================================================
# STATE MACHINE - Pipeline State Tracking
# ============================================================================

class PipelineState:
    """Tracks the current state of the audiobook pipeline."""
    EPUB_PARSED = "epub_parsed"
    LABELS_COMPLETE = "labels_complete"
    CHARACTERS_DESCRIBED = "characters_described"
    VOICE_SAMPLES_COMPLETE = "voice_samples_complete"
    AUDIOBOOK_COMPLETE = "audiobook_complete"


def get_pipeline_state():
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
        return PipelineState.EPUB_PARSED

    # Check for Stage 3 completion (characters descriptions)
    descriptions_file = get_character_descriptions_file()
    if not os.path.exists(str(descriptions_file)):
        return PipelineState.LABELS_COMPLETE

    # Check for Stage 4 completion (voice samples)
    wav_files = list(chapters_dir.glob("*.wav"))
    if not wav_files:
        return PipelineState.CHARACTERS_DESCRIBED

    # Check for Stage 5 completion (final audiobook)
    mp3_files = sorted(glob.glob(str(chapters_dir / "chapter_*.mp3")))
    if not mp3_files:
        return PipelineState.VOICE_SAMPLES_COMPLETE

    return PipelineState.AUDIOBOOK_COMPLETE


# ============================================================================
# Stage 1: EPUB Parsing
# ============================================================================

def parse_epub_to_file(epub_file, max_chapters=None, progress=gr.Progress()):
    """Stage 1: Parse EPUB file into chapter text files."""
    if epub_file is None:
        return "Error: No EPUB file uploaded.", 0, []

    try:
        # Import parse_chapter module
        from parse_chapter import parse_epub_to_chapters
        import shutil

        # Get the temp chapters directory
        chapters_dir = get_chapters_dir()

        # Copy the EPUB file to temp directory for Stage 6
        epub_dest = chapters_dir / "uploaded.epub"
        shutil.copy2(epub_file.name, str(epub_dest))

        # Parse the EPUB with max_chapters limit if specified
        chapters = parse_epub_to_chapters(epub_file.name, max_chapters=int(max_chapters) if max_chapters else None)

        if not chapters:
            return "Error: No chapters found in EPUB file.", 0, []

        # Gradio 6.x: progress(progress_float, desc="...")
        progress(0, desc="Parsing EPUB...")

        # Save each chapter as a text file
        chapter_files = []
        total_chapters = len(chapters)
        for i, chapter in enumerate(chapters):
            progress((i + 1) / total_chapters, desc=f"Saving chapter {i+1}/{total_chapters}")
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
# Stage 2: LLM Speaker Labeling
# ============================================================================

def process_chapters_for_labels(api_key, port, num_attempts, pipeline_state, log_output, progress=gr.Progress()):
    """Stage 2: Run LLM to label speakers in all chapters."""
    # Check dependency: EPUB must be parsed first
    chapters_dir = get_chapters_dir()
    chapter_files = sorted(glob.glob(str(chapters_dir / "chapter_*.txt")))

    if not chapter_files:
        log_output += "\nNo chapter files found. Please run Stage 1 (Parse EPUB) first."
        return log_output, pipeline_state

    # Process all chapters (chapters were selected in Stage 1 via max_chapters_slider)
    selected_chapters = chapter_files

    num_chapters = len(selected_chapters)
    log_output += f"\nProcessing {num_chapters} chapters with LLM..."

    for i, chapter_file in enumerate(selected_chapters):
        # Update progress before processing each chapter
        progress(i / num_chapters, desc=f"Chapter {i}/{num_chapters}")
        log_output += f"\nProcessing: {chapter_file}"

        # Build command to call llm_label_speakers.py
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "llm_label_speakers.py"),
            "-txt_file", chapter_file,
            "-num_llm_attempts", str(num_attempts),
            "-api_key", api_key,
            "-port", port
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,  # 5 minute timeout per chapter
                cwd=str(SCRIPT_DIR)
            )
            log_output += result.stdout
            if result.stderr:
                log_output += f"\nErrors: {result.stderr}"
        except subprocess.TimeoutExpired:
            log_output += "\nTimeout - chapter processing took too long."
        except Exception as e:
            log_output += f"\nError processing {chapter_file}: {str(e)}"

    # Update state after successful completion
    new_state = PipelineState.LABELS_COMPLETE
    log_output += f"\n\nStage 2 complete! State: {new_state}"
    return log_output, new_state


# ============================================================================
# Stage 3: Character Descriptions
# ============================================================================

def get_characters_from_map_files(chapters_dir):
    """Extract unique character names from map.json files."""
    import glob
    characters = set()

    map_files = glob.glob(str(chapters_dir / "*.map.json"))
    for map_file in map_files:
        try:
            with open(map_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            character_map = data[0] if isinstance(data, list) and len(data) > 0 else data.get("character_map", {})
            # Get character names from character_map values
            if isinstance(character_map, dict):
                for char_name in character_map.values():
                    if isinstance(char_name, str):
                        characters.add(char_name)
        except Exception:
            pass

    # Remove narrator from the list (described separately or not needed)
    characters.discard("narrator")
    return sorted(list(characters))


def describe_characters(api_key, port, pipeline_state, log_output, progress=gr.Progress()):
    """Stage 3: Use LLM to describe characters."""
    # Check dependency: Labels must exist first
    log_output += "\n\nGenerating character descriptions..."

    try:
        chapters_dir = get_chapters_dir()

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

        progress(0, desc=f"Character 0/{num_characters}")

        log_output += f"\nFound {num_characters} characters from map files."

        # Run llm_describe_character.py with progress tracking
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "llm_describe_character.py"),
            "--api_key", api_key,
            "--port", port,
            "--verbose"
        ]

        # Write characters to a temp file for the script to read
        temp_characters_file = chapters_dir / "temp_characters.json"
        with open(temp_characters_file, "w", encoding="utf-8") as f:
            json.dump({"characters": characters}, f)

        cmd.insert(2, str(temp_characters_file))
        cmd.insert(3, str(chapters_dir))

        # Use subprocess with progress tracking
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(SCRIPT_DIR)
        )

        # Read output line by line to track progress
        processed_chars = 0
        for line in iter(process.stdout.readline, ''):
            if line:
                log_output += f"\n{line.strip()}"
                # Update progress based on output
                if "Loading" in line or "Loaded" in line:
                    progress(processed_chars / num_characters, desc="Loading...")
                elif "Describing" in line or "Character" in line:
                    progress(processed_chars / num_characters, desc=f"Character {processed_chars}/{num_characters}")
                    processed_chars += 1

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            if process.stderr:
                log_output += f"\nErrors: {process.stderr.read()}"
            log_output += "\nWarning: Process exited with non-zero code."

        progress(1, desc="Stage 3 Complete")
        log_output += f"\n\nStage 3 complete!"

        # Load and return the character descriptions for state tracking
        try:
            with open(get_character_descriptions_file(), "r", encoding="utf-8") as f:
                characters_state = json.load(f)
        except Exception:
            characters_state = {}

        new_state = PipelineState.CHARACTERS_DESCRIBED
        log_output += f" State: {new_state}"
        return log_output, new_state, characters_state

    except Exception as e:
        log_output += f"\nError describing characters: {str(e)}"
        return log_output, pipeline_state, None


# ============================================================================
# Stage 5: Voice Sample Generation
# ============================================================================

def get_character_descriptions_file():
    """Get the path to characters_descriptions.json."""
    return SCRIPT_DIR / "characters_descriptions.json"


def get_character_wav_file(character_name, chapters_dir):
    """Get the path to a character's generated WAV file."""
    # Check in chapters_dir first, then parent directory
    for base_dir in [chapters_dir, SCRIPT_DIR]:
        wav_path = base_dir / f"{character_name}.wav"
        if os.path.exists(wav_path):
            return str(wav_path)
    return None


def get_all_character_wav_files(chapters_dir):
    """Get all generated character WAV files."""
    wav_files = {}
    # Check in chapters_dir first, then parent directory
    for base_dir in [chapters_dir, SCRIPT_DIR]:
        if base_dir and os.path.exists(base_dir):
            for wav_path in base_dir.glob("*.wav"):
                # Only include character voice samples (not narrator or chapter files)
                wav_name = wav_path.name
                if not wav_name.startswith("chapter_") and wav_name != "narrator.wav" and wav_name != "uploaded.epub":
                    char_name = wav_path.stem
                    wav_files[char_name] = str(wav_path)
    return wav_files


def generate_voice_samples(pipeline_state, log_output, progress=gr.Progress()):
    """Stage 4: Generate voice samples for each character."""
    # Check dependency: Characters must be described first
    log_output += "\n\nGenerating voice samples..."

    try:
        # Check if characters_descriptions.json exists
        chapters_dir = get_chapters_dir()
        descriptions_file = get_character_descriptions_file()
        if not os.path.exists(str(descriptions_file)):
            log_output += "\ncharacters_descriptions.json not found. Please run Stage 3 (Describe Characters) first."
            return log_output, pipeline_state

        # Load descriptions to get count for progress
        with open(descriptions_file, "r", encoding="utf-8") as f:
            descriptions = json.load(f)

        num_characters = len(descriptions)
        log_output += f"\nFound {num_characters} characters to process."

        # Run generate_voice_samples.py with progress tracking
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "generate_voice_samples.py"),
            "--descriptions", str(descriptions_file),
            "--output-dir", str(chapters_dir)
        ]

        # Use subprocess with progress tracking
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(SCRIPT_DIR)
        )

        # Read output line by line to track progress
        processed_chars = 0
        for line in iter(process.stdout.readline, ''):
            if line:
                line_stripped = line.strip()
                log_output += f"\n{line_stripped}"

                # Parse progress from output format: [1/10] character_name
                match = re.search(r'\[(\d+)/(\d+)\]\s+(\w+)', line_stripped)
                if match:
                    processed_chars = int(match.group(1))
                    total_chars = int(match.group(2))
                    char_name = match.group(3)
                    progress(processed_chars / total_chars, desc=f"Creating {char_name}.wav ({processed_chars}/{total_chars})")

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            if process.stderr:
                log_output += f"\nErrors: {process.stderr.read()}"
            log_output += "\nWarning: Process exited with non-zero code."

        progress(1, desc="Stage 4 Complete")
        log_output += f"\n\nStage 4 complete!"

        new_state = PipelineState.VOICE_SAMPLES_COMPLETE
        log_output += f" State: {new_state}"
        return log_output, new_state

    except Exception as e:
        log_output += f"\nError generating voice samples: {str(e)}"
        return log_output, pipeline_state


def regenerate_voice_sample(character_name, api_key, port, pipeline_state, log_output, progress=gr.Progress()):
    """Regenerate a single voice sample for a character."""
    # Check dependency: Characters must be described first
    log_output += f"\n\nRegenerating voice sample for: {character_name}"

    try:
        chapters_dir = get_chapters_dir()
        descriptions_file = get_character_descriptions_file()

        if not os.path.exists(str(descriptions_file)):
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
            "--single-character", character_name
        ]

        progress(0, desc=f"Regenerating {character_name}...")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(SCRIPT_DIR)
        )

        for line in iter(process.stdout.readline, ''):
            if line:
                line_stripped = line.strip()
                log_output += f"\n{line_stripped}"
                if "Generated:" in line_stripped:
                    progress(1, desc="Done!")

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            if process.stderr:
                log_output += f"\nErrors: {process.stderr.read()}"
            log_output += "\nWarning: Process exited with non-zero code."
            return log_output, pipeline_state, None

        # Return the path to the regenerated file
        wav_path = get_character_wav_file(character_name, chapters_dir)
        progress(1, desc="Done!")
        log_output += f"\n\nVoice sample regenerated for: {character_name}"
        return log_output, pipeline_state, wav_path

    except Exception as e:
        log_output += f"\nError regenerating voice sample: {str(e)}"
        return log_output, pipeline_state, None


# ============================================================================
# Stage 6: Full Audiobook Generation - Broken into steps
# ============================================================================

def generate_tts_audio(pipeline_state, log_output, max_chapters=None, progress=gr.Progress()):
    """Stage 5.1: Generate TTS audio for each line/voice."""
    # Check dependencies: Labels and character descriptions must exist
    log_output += "\n\n=== Stage 5.1: Generating TTS Audio ==="

    try:
        # Check if chapter map files exist
        chapters_dir = get_chapters_dir()
        map_files = glob.glob(str(chapters_dir / "*.map.json"))
        if not map_files:
            log_output += "\nNo .map.json files found. Please run Stage 2 (Label Speakers) first."
            return log_output, pipeline_state

        # Check if characters_descriptions.json exists
        if not os.path.exists(str(SCRIPT_DIR / "characters_descriptions.json")):
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
        cmd = [sys.executable, str(SCRIPT_DIR / "parse_epub.py"), epub_path, "--resume", "--output-dir", str(chapters_dir)]
        if max_chapters:
            cmd.extend(["--max-chapters", str(max_chapters)])

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(SCRIPT_DIR)
        )

        # Track TTS generation progress
        completed_lines = 0
        current_chapter = 0

        for line in iter(process.stdout.readline, ''):
            if line:
                line_stripped = line.strip()
                log_output += f"\n{line_stripped}"

                # Parse chapter start
                chapter_match = re.search(r'\[CHAPTER_START\]\s+Chapter\s+(\d+)', line_stripped)
                if chapter_match:
                    current_chapter = int(chapter_match.group(1))
                    completed_lines = 0  # Reset line count per chapter

                # Parse line progress
                line_match = re.search(r'\[LINE_PROGRESS\]\s+Chapter\s+(\d+),\s+Line\s+(\d+)', line_stripped)
                if line_match:
                    completed_lines += 1
                    progress(min((current_chapter + completed_lines / 100.0) / num_chapters, 1.0),
                            desc=f"Chapter {current_chapter + 1}/{num_chapters} - Generating audio")

        process.stdout.close()
        process.wait()

        progress(1, desc="TTS Audio Complete")
        log_output += "\n\nStage 5.1 (TTS Audio) complete!"
        return log_output, pipeline_state

    except Exception as e:
        log_output += f"\nError generating TTS audio: {str(e)}"
        return log_output, pipeline_state


def assemble_chapter_audiobooks(pipeline_state, log_output, progress=gr.Progress()):
    """Stage 5.3: Assemble final chapter MP3 files from WAV files."""
    log_output += "\n\n=== Stage 5.3: Assembling Chapter MP3 Files ==="

    try:
        chapters_dir = get_chapters_dir()
        chapter_files = sorted(glob.glob(str(chapters_dir / "chapter_*.txt")))
        if not chapter_files:
            log_output += "\nNo chapter text files found. Please run Stage 5.1 first."
            return log_output, pipeline_state

        num_chapters = len(chapter_files)
        log_output += f"\nFound {num_chapters} chapters to assemble."

        import pydub

        def get_non_silent_audio_from_wavs(wav_filepath_list, min_silence_len=1250, silence_thresh=-60):
            """Remove silent audio from list of wave filepaths."""
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

        # Process each chapter
        progress(0, desc="Assembling chapters...")

        for i in range(len(chapter_files)):
            progress((i + 1) / num_chapters, desc=f"Assembling chapter {i + 1}/{num_chapters}")

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

        progress(1, desc="Assembly Complete")
        log_output += "\n\nStage 5.3 (Assembly) complete!"
        return log_output, pipeline_state

    except Exception as e:
        log_output += f"\nError assembling audiobooks: {str(e)}"
        return log_output, pipeline_state


def generate_full_audiobook(pipeline_state, log_output, max_chapters=None, progress=gr.Progress()):
    """Stage 5: Generate full audiobook - runs TTS generation and assembly."""
    log_output += "\n\n=== Stage 5: Full Audiobook Generation ==="

    # Step 5.1: Generate TTS audio (with embedded validation)
    log_output, pipeline_state = generate_tts_audio(pipeline_state, log_output, max_chapters, progress)

    # Step 5.2: Assemble chapter MP3s
    log_output, pipeline_state = assemble_chapter_audiobooks(pipeline_state, log_output, progress)

    # Update state to audiobook complete
    new_state = PipelineState.AUDIOBOOK_COMPLETE
    log_output += f"\n\n=== Stage 5 Complete: Full Audiobook Generation === State: {new_state}"
    return log_output, new_state


def update_character_table(characters_state):
    """
    Update the character table based on stored character state.
    Uses gr.State to track characters after Stage 4.

    Args:
        characters_state: gr.State containing dict of character_name -> description

    Returns:
        Updated Dataframe component with character data (2 columns: name, description)
    """
    descriptions_file = get_character_descriptions_file()

    if not os.path.exists(str(descriptions_file)) or characters_state is None:
        return gr.Dataframe(value=[])

    try:
        with open(descriptions_file, "r", encoding="utf-8") as f:
            descriptions = json.load(f)

        # Build table data with only character name and description
        table_data = []
        for char_name, char_desc in descriptions.items():
            table_data.append([char_name, char_desc])

        return gr.Dataframe(
            headers=["Character", "Description"],
            datatype=["str", "str"],
            value=table_data,
            wrap=True
        )

    except Exception as e:
        return gr.Dataframe(value=[])






# ============================================================================
# Gradio Interface with State Machine
# ============================================================================

def create_interface(api_key_default="lm-studio", port_default="1234", num_attempts_default=10, epub_path_default=None, max_chapters_default=10):
    """Create the Gradio interface with all stages using a state machine pattern."""

    with gr.Blocks() as demo:
        gr.Markdown("# Audiobook Pipeline")

        # Progress bar at top
        progress_bar = gr.Progress()

        # State display
        state_display = gr.Label(label="State", value="Ready", show_label=True)

        # Settings - compact
        with gr.Row():
            api_key_input = gr.Textbox(label="API", value=api_key_default, scale=2)
            port_input = gr.Textbox(label="Port", value=port_default, scale=1)
            num_attempts_input = gr.Slider(minimum=1, maximum=50, value=num_attempts_default,
                                           step=1, label="LLM Attempts", scale=2)
            max_chapters_slider = gr.Slider(minimum=1, maximum=100, value=max_chapters_default,
                                            step=1, label="Max", scale=2)

        # EPUB upload
        epub_upload = gr.File(label="EPUB", file_types=[".epub"], value=epub_path_default)

        # Main buttons - single row
        with gr.Row():
            parse_btn = gr.Button("1. Parse", variant="primary", scale=1)
            label_btn = gr.Button("2. Label", variant="secondary", scale=1)
            describe_btn = gr.Button("3. Describe", variant="secondary", scale=1)

        # Voice samples - single row
        with gr.Row():
            voice_samples_btn = gr.Button("4. Voices", variant="secondary", scale=1)
            generate_char_btn = gr.Button("5. Regen", variant="secondary", scale=1)
            tts_btn = gr.Button("6. Audiobook", variant="primary", scale=2)

        # Log output
        log_output = gr.Textbox(label="Log", lines=4, max_lines=6)

        # Character info
        character_table = gr.Dataframe(
            headers=["Character", "Description"],
            datatype=["str", "str"],
            wrap=True,
            max_height=100
        )

        # Audio player (hidden until needed)
        character_audio = gr.Audio(label="", type="filepath", visible=False)

        # State to track characters
        characters_state = gr.State(None)

        # State to track pipeline state
        pipeline_state = gr.State(None)

        # ============================================================================
        # STATE TRANSITION HANDLERS - Define which buttons are enabled based on state
        # ============================================================================

        def update_state_display(state):
            """Update state display based on pipeline state."""
            state_labels = {
                None: "Ready",
                PipelineState.EPUB_PARSED: "EPUB Parsed",
                PipelineState.LABELS_COMPLETE: "Speakers Labeled",
                PipelineState.CHARACTERS_DESCRIBED: "Characters Described",
                PipelineState.VOICE_SAMPLES_COMPLETE: "Voice Samples Ready",
                PipelineState.AUDIOBOOK_COMPLETE: "Audiobook Complete"
            }
            return (gr.update(value=state_labels.get(state, "Unknown")),)

        def update_button_visibility(state):
            """
            Update button enabled state based on pipeline state.
            Returns tuple of (parse_update, label_update, describe_update, voice_update, gen_char_update, tts_update)
            using gr.update() to properly update button states.
            """
            if state is None:
                # Initial state: only Parse EPUB is enabled
                return (gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False),
                        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False))
            elif state == PipelineState.EPUB_PARSED:
                # EPUB parsed: can do Label Speakers
                return (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False),
                        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False))
            elif state == PipelineState.LABELS_COMPLETE:
                # Labels done: can describe characters
                return (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
                        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False))
            elif state == PipelineState.CHARACTERS_DESCRIBED:
                # Characters described: can generate voice samples
                return (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
                        gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False))
            elif state == PipelineState.VOICE_SAMPLES_COMPLETE:
                # Voice samples done: can generate full audiobook
                return (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
                        gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True))
            else:  # AUDIOBOOK_COMPLETE or beyond
                return (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
                        gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True))

        # ============================================================================
        # EVENT HANDLERS
        # ============================================================================

        # Parse EPUB - Stage 1
        parse_btn.click(
            fn=parse_epub_to_file,
            inputs=[epub_upload, max_chapters_slider],
            outputs=[log_output, log_output, log_output]  # dummy outputs to satisfy function signature
        ).then(
            # Reset log for new session
            fn=lambda: "=== Stage 1: EPUB Parsing Complete ===\n",
            outputs=log_output
        ).then(
            fn=lambda: PipelineState.EPUB_PARSED,
            outputs=pipeline_state
        ).then(
            fn=lambda s: update_button_visibility(s) + update_state_display(s),
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn, state_display]
        )

        # Label Speakers - Stage 2
        label_btn.click(
            fn=process_chapters_for_labels,
            inputs=[api_key_input, port_input, num_attempts_input, pipeline_state, log_output],
            outputs=[log_output, pipeline_state]
        ).then(
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn]
        )

        # Describe Characters - Stage 3
        describe_btn.click(
            fn=describe_characters,
            inputs=[api_key_input, port_input, pipeline_state, log_output],
            outputs=[log_output, pipeline_state, characters_state]
        ).then(
            fn=update_character_table,
            inputs=characters_state,
            outputs=character_table
        ).then(
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn]
        )

        # Generate All Voice Samples - Stage 4
        voice_samples_btn.click(
            fn=generate_voice_samples,
            inputs=[pipeline_state, log_output],
            outputs=[log_output, pipeline_state]
        ).then(
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn]
        )

        # Handle row selection in character table - show audio when character selected
        def on_character_select_simple(evt: gr.SelectData, _characters_state):
            """Handle row selection in the character table."""
            if evt is None or evt.index is None:
                return gr.update(visible=False)

            character_name = evt.row_value[0] if evt.row_value and len(evt.row_value) > 0 else None

            if not character_name:
                return gr.update(visible=False)

            chapters_dir = get_chapters_dir()
            wav_path = get_character_wav_file(character_name, chapters_dir)

            if wav_path and os.path.exists(wav_path):
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)

        character_table.select(
            fn=on_character_select_simple,
            inputs=[characters_state],
            outputs=character_audio
        )

        # Generate Full Audiobook - Stage 5
        tts_btn.click(
            fn=generate_full_audiobook,
            inputs=[pipeline_state, log_output, max_chapters_slider],
            outputs=[log_output, pipeline_state]
        ).then(
            fn=update_button_visibility,
            inputs=pipeline_state,
            outputs=[parse_btn, label_btn, describe_btn, voice_samples_btn, generate_char_btn, tts_btn]
        )

        
    return demo


def cleanup_temp_dir():
    """Clean up the temporary directory when done."""
    global TEMP_DIR
    if TEMP_DIR and os.path.exists(TEMP_DIR):
        import shutil
        shutil.rmtree(TEMP_DIR)
        TEMP_DIR = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audiobook Pipeline Gradio Interface")
    parser.add_argument("--api_key", type=str, default="lm-studio", help="LLM API Key to pre-fill")
    parser.add_argument("--port", type=str, default="1234", help="LLM Port to pre-fill")
    parser.add_argument("--num_llm_attempts", type=int, default=10, help="Number of LLM attempts to pre-fill")
    parser.add_argument("--epub", type=str, help="EPUB file path to pre-load")
    parser.add_argument("--max_chapters", type=int, default=10, help="Max chapters to pre-fill")
    args = parser.parse_args()

    demo = create_interface(
        api_key_default=args.api_key,
        port_default=args.port,
        num_attempts_default=args.num_llm_attempts,
        epub_path_default=args.epub,
        max_chapters_default=args.max_chapters
    )

    try:
        demo.launch(share=False, theme=gr.themes.Soft())
    finally:
        cleanup_temp_dir()