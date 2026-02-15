#!/usr/bin/env python3
"""
Gradio Interface for Audiobook Pipeline

A unified web interface for the 6-stage audiobook creation pipeline:
1. EPUB Parsing
2. LLM Speaker Labeling
3. Chapter Analysis
4. Character Description
5. Voice Sample Generation
6. Full Audiobook Generation
"""

import gradio as gr
import os
import sys
import json
import glob
import subprocess
import tempfile
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

def process_chapters_for_labels(api_key, port, num_attempts, log_output, progress=gr.Progress()):
    """Stage 2: Run LLM to label speakers in all chapters."""
    # Get list of chapter files
    chapters_dir = get_chapters_dir()
    chapter_files = sorted(glob.glob(str(chapters_dir / "chapter_*.txt")))

    if not chapter_files:
        log_output += "\nNo chapter files found. Please run Stage 1 first."
        return log_output

    # Process all chapters (chapters were selected in Stage 1 via max_chapters_slider)
    selected_chapters = chapter_files

    num_chapters = len(selected_chapters)
    log_output += f"\nProcessing {num_chapters} chapters with LLM..."

    for i, chapter_file in enumerate(selected_chapters):
        # Update progress before processing each chapter
        # i represents the number of completed chapters (0-indexed)
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

    # Final progress: all chapters completed
    progress(1, desc="Chapter 100%")
    log_output += "\n\nStage 2 complete!"
    return log_output


# ============================================================================
# Stage 3: Chapter Analysis
# ============================================================================

def analyze_chapters(log_output, progress=gr.Progress()):
    """Stage 3: Analyze chapter map files and generate statistics."""
    log_output += "\n\nRunning chapter analysis..."

    try:
        # Check if map files exist
        chapters_dir = get_chapters_dir()
        map_files = glob.glob(str(chapters_dir / "*.map.json"))
        if not map_files:
            log_output += "\nNo .map.json files found. Please run Stage 2 first."
            return log_output

        # Count map files for progress
        num_map_files = len(map_files)
        progress(0, desc="Analyzing chapters...")

        # Run analyze_chapters.py
        cmd = [sys.executable, str(SCRIPT_DIR / "analyze_chapters.py"), str(chapters_dir), "--json-output", "--verbose"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(SCRIPT_DIR))

        # Read output line by line for progress feedback
        # Count lines with chapter progress markers
        lines_with_chapter = 0
        for line in iter(process.stdout.readline, ''):
            if line:
                log_output += f"\n{line.strip()}"
                # Count lines that contain chapter info
                if "Chapter" in line or "Loaded" in line or "Analyzing" in line:
                    lines_with_chapter += 1
                    progress(min(lines_with_chapter / (num_map_files + 1), 1.0), desc="Analyzing...")

        process.stdout.close()
        process.wait()

        progress(1, desc="Stage 3 Complete")
        log_output += "\n\nStage 3 complete!"
        return log_output

    except Exception as e:
        log_output += f"\nError analyzing chapters: {str(e)}"
        return log_output


# ============================================================================
# Stage 4: Character Descriptions
# ============================================================================

def describe_characters(api_key, port, log_output, progress=gr.Progress()):
    """Stage 4: Use LLM to describe characters."""
    log_output += "\n\nGenerating character descriptions..."

    try:
        # Check if characters.json exists
        chapters_dir = get_chapters_dir()
        characters_file = chapters_dir / "characters.json"
        if not os.path.exists(str(characters_file)):
            log_output += "\ncharacters.json not found. Please run Stage 3 first."
            return log_output

        # Load characters to get count for progress
        with open(characters_file, "r", encoding="utf-8") as f:
            characters_data = json.load(f)
        num_characters = len(characters_data.get("characters", []))

        progress(0, desc=f"Character 0/{num_characters}")

        # Run llm_describe_character.py with progress tracking
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "llm_describe_character.py"),
            str(characters_file),
            str(chapters_dir),
            "--api_key", api_key,
            "--port", port,
            "--verbose"
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

        progress(1, desc="Stage 4 Complete")
        log_output += "\n\nStage 4 complete!"
        return log_output

    except Exception as e:
        log_output += f"\nError describing characters: {str(e)}"
        return log_output


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


def generate_voice_samples(log_output, progress=gr.Progress()):
    """Stage 5: Generate voice samples for each character."""
    log_output += "\n\nGenerating voice samples..."

    try:
        # Check if characters_descriptions.json exists
        chapters_dir = get_chapters_dir()
        descriptions_file = get_character_descriptions_file()
        if not os.path.exists(str(descriptions_file)):
            log_output += "\ncharacters_descriptions.json not found. Please run Stage 4 first."
            return log_output

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
        import re
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

        progress(1, desc="Stage 5 Complete")
        log_output += "\n\nStage 5 complete!"
        return log_output

    except Exception as e:
        log_output += f"\nError generating voice samples: {str(e)}"
        return log_output


def regenerate_voice_sample(character_name, api_key, port, log_output, progress=gr.Progress()):
    """Regenerate a single voice sample for a character."""
    log_output += f"\n\nRegenerating voice sample for: {character_name}"

    try:
        chapters_dir = get_chapters_dir()
        descriptions_file = get_character_descriptions_file()

        if not os.path.exists(str(descriptions_file)):
            log_output += "\ncharacters_descriptions.json not found. Please run Stage 4 first."
            return log_output, None

        # Load the specific character's description
        with open(descriptions_file, "r", encoding="utf-8") as f:
            descriptions = json.load(f)

        if character_name not in descriptions:
            log_output += f"\nCharacter '{character_name}' not found in descriptions."
            return log_output, None

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
            return log_output, None

        # Return the path to the regenerated file
        wav_path = get_character_wav_file(character_name, chapters_dir)
        progress(1, desc="Done!")
        log_output += f"\n\nVoice sample regenerated for: {character_name}"
        return log_output, wav_path

    except Exception as e:
        log_output += f"\nError regenerating voice sample: {str(e)}"
        return log_output, None


# ============================================================================
# Stage 6: Full Audiobook Generation - Broken into steps
# ============================================================================

def generate_tts_audio(log_output, max_chapters=None, progress=gr.Progress()):
    """Stage 6.1: Generate TTS audio for each line/voice."""
    log_output += "\n\n=== Stage 6.1: Generating TTS Audio ==="

    try:
        # Check if chapter map files exist
        chapters_dir = get_chapters_dir()
        map_files = glob.glob(str(chapters_dir / "*.map.json"))
        if not map_files:
            log_output += "\nNo .map.json files found. Please run Stage 2 first."
            return log_output

        # Check if characters_descriptions.json exists
        if not os.path.exists(str(SCRIPT_DIR / "characters_descriptions.json")):
            log_output += "\ncharacters_descriptions.json not found. Please run Stage 4 first."
            return log_output

        # Check if uploaded EPUB exists in temp directory
        epub_path = str(chapters_dir / "uploaded.epub")
        if not os.path.exists(epub_path):
            log_output += "\nUploaded EPUB file not found. Please run Stage 1 first."
            return log_output

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

        import re
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
        log_output += "\n\nStage 6.1 (TTS Audio) complete!"
        return log_output

    except Exception as e:
        log_output += f"\nError generating TTS audio: {str(e)}"
        return log_output


def validate_and_clip_audio(log_output, max_chapters=None, progress=gr.Progress()):
    """Stage 6.2: Validate and clip audio with Whisper."""
    log_output += "\n\n=== Stage 6.2: Validating and Clipping Audio ==="

    try:
        # Check if temporary audio files exist
        chapters_dir = get_chapters_dir()
        tmp_wavs = glob.glob(str(chapters_dir / "*.tmp.wav"))
        if not tmp_wavs:
            log_output += "\nNo temporary audio files found. Please run Stage 6.1 first."
            return log_output

        num_tmp_files = len(tmp_wavs)
        log_output += f"\nFound {num_tmp_files} temporary audio files to validate..."

        chapter_files = sorted(glob.glob(str(chapters_dir / "chapter_*.txt")))
        num_chapters = len(chapter_files)
        if max_chapters:
            num_chapters = min(num_chapters, int(max_chapters))

        # Get the list of already generated valid .wav files
        valid_wavs = glob.glob(str(chapters_dir / "*.wav"))
        valid_count = len(valid_wavs)

        log_output += f"\nValidating audio files... (already validated: {valid_count})"

        # The validation happens DURING generation in parse_epub.py
        # We run parse_epub with --resume to continue the process
        epub_path = str(chapters_dir / "uploaded.epub")
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

        processed_lines = 0
        total_estimate = num_chapters * 100  # Rough estimate

        for line in iter(process.stdout.readline, ''):
            if line:
                line_stripped = line.strip()
                log_output += f"\n{line_stripped}"

                # Parse line progress for validation
                if "[LINE_PROGRESS]" in line_stripped:
                    processed_lines += 1
                    progress(processed_lines / total_estimate, desc="Validating audio")

        process.stdout.close()
        process.wait()

        progress(1, desc="Validation Complete")
        log_output += "\n\nStage 6.2 (Validation) complete!"
        return log_output

    except Exception as e:
        log_output += f"\nError validating audio: {str(e)}"
        return log_output


def assemble_chapter_audiobooks(log_output, progress=gr.Progress()):
    """Stage 6.3: Assemble final chapter MP3 files from WAV files."""
    log_output += "\n\n=== Stage 6.3: Assembling Chapter MP3 Files ==="

    try:
        chapters_dir = get_chapters_dir()
        chapter_files = sorted(glob.glob(str(chapters_dir / "chapter_*.txt")))
        if not chapter_files:
            log_output += "\nNo chapter text files found. Please run Stage 6.1 first."
            return log_output

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
        log_output += "\n\nStage 6.3 (Assembly) complete!"
        return log_output

    except Exception as e:
        log_output += f"\nError assembling audiobooks: {str(e)}"
        return log_output


def generate_full_audiobook(log_output, max_chapters=None, progress=gr.Progress()):
    """Stage 6: Generate full audiobook - runs all 6.x steps in sequence."""
    log_output += "\n\n=== Stage 6: Full Audiobook Generation ==="

    # Step 6.1: Generate TTS audio
    log_output = generate_tts_audio(log_output, max_chapters, progress)

    # Step 6.2: Validate and clip audio
    log_output = validate_and_clip_audio(log_output, max_chapters, progress)

    # Step 6.3: Assemble chapter MP3s
    log_output = assemble_chapter_audiobooks(log_output, progress)

    log_output += "\n\n=== Stage 6 Complete: Full Audiobook Generation ==="
    return log_output


def generate_character_table():
    """Generate a table showing character descriptions with audio players and redo buttons."""
    chapters_dir = get_chapters_dir()
    descriptions_file = get_character_descriptions_file()

    if not os.path.exists(str(descriptions_file)):
        return gr.Dataframe(value=[])

    try:
        with open(descriptions_file, "r", encoding="utf-8") as f:
            descriptions = json.load(f)

        # Get all generated wav files
        wav_files = get_all_character_wav_files(chapters_dir)

        # Build table data
        table_data = []
        for char_name, char_desc in descriptions.items():
            # Check if wav file exists for this character
            wav_path = wav_files.get(char_name)
            if wav_path and os.path.exists(wav_path):
                # Create audio player HTML with fixed height for audio controls
                audio_html = f'<audio controls src="{wav_path}" style="width: 100%; height: 32px;"></audio>'
                # Create redo button HTML
                redo_btn = f'<button class="redo-btn" data-char="{char_name}">Redo</button>'
                table_data.append([char_name, char_desc, audio_html, redo_btn])
            else:
                table_data.append([char_name, char_desc, "", "<button class=\"generate-btn\" data-char=\"{}\">Generate</button>".format(char_name)])

        # Create dataframe with HTML datatype for audio and buttons
        return gr.Dataframe(
            headers=["Character", "Description", "Audio", "Action"],
            datatype=["str", "str", "HTML", "HTML"],
            value=table_data,
            wrap=True
        )

    except Exception as e:
        return gr.Dataframe(value=[])


def on_character_table_select(evt: gr.SelectData, table_state, api_key, port, log_output):
    """Handle button clicks in the character table."""
    import re

    # evt.value contains the cell value (the button HTML)
    button_html = evt.value if evt.value else ""
    character_name = evt.row[0] if evt.row and len(evt.row) > 0 else None

    if not character_name:
        return gr.State(None), log_output

    # Check if it's a redo button
    if "Redo" in button_html or "redo-btn" in button_html:
        # Extract character name from data attribute
        match = re.search(r'data-char="([^"]+)"', button_html)
        if match:
            char_name = match.group(1)
            # Call the regenerate function
            new_log, wav_path = regenerate_voice_sample(char_name, api_key, port, log_output)
            # Refresh the table
            new_table = generate_character_table()
            return gr.State(new_table), new_log

    # Check if it's a generate button
    elif "Generate" in button_html or "generate-btn" in button_html:
        # Extract character name from data attribute
        match = re.search(r'data-char="([^"]+)"', button_html)
        if match:
            char_name = match.group(1)
            log_output += f"\n\nWould regenerate voice sample for: {char_name}"
            # Call the regenerate function anyway
            new_log, wav_path = regenerate_voice_sample(char_name, api_key, port, log_output)
            new_table = generate_character_table()
            return gr.State(new_table), new_log

    return gr.State(None), log_output


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface(api_key_default="lm-studio", port_default="1234", num_attempts_default=10, epub_path_default=None, max_chapters_default=10):
    """Create the Gradio interface with all stages."""

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Audiobook Pipeline Interface")

        # Configuration and EPUB Parsing (required first step)
        with gr.Accordion("Configuration & EPUB Parsing", open=True):
            gr.Markdown("### Pipeline Settings")
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="LLM API Key",
                    value=api_key_default,
                    placeholder="Enter your API key (default: lm-studio)"
                )
                port_input = gr.Textbox(
                    label="LLM Port",
                    value=port_default,
                    placeholder="Port for LLM inference"
                )
                num_attempts_input = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=num_attempts_default,
                    step=1,
                    label="Number of LLM Attempts"
                )
            max_chapters_slider = gr.Slider(
                minimum=1,
                maximum=100,
                value=max_chapters_default,
                step=1,
                label="Max Chapters to Generate"
            )

            gr.Markdown("### EPUB File")
            epub_upload = gr.File(label="Upload EPUB File", file_types=[".epub"], value=epub_path_default)
            parse_btn = gr.Button("Parse EPUB", variant="primary")
            parse_output = gr.Textbox(label="Status")
            chapter_count_display = gr.Number(label="Chapters Created", precision=0)

        # Stage 2: LLM Speaker Labeling
        with gr.Accordion("Stage 2: Label Speakers", open=False) as stage2:
            gr.Markdown("Use LLM to identify speakers and attribute dialogue lines.")
            label_btn = gr.Button("Label All Chapters", variant="primary")
            log_output = gr.Textbox(label="Progress Log", lines=10, max_lines=20)

        # Stage 3: Chapter Analysis
        with gr.Accordion("Stage 3: Analyze Chapters", open=False) as stage3:
            gr.Markdown("Analyze chapter map files and generate statistics.")
            analyze_btn = gr.Button("Analyze Chapters", variant="secondary")
            analysis_output = gr.Textbox(label="Analysis Results")

        # Stage 4: Character Descriptions
        with gr.Accordion("Stage 4: Describe Characters", open=False) as stage4:
            gr.Markdown("Generate voice profiles for each character using LLM.")
            describe_btn = gr.Button("Describe Characters", variant="secondary")
            descriptions_output = gr.Textbox(label="Character Descriptions")

        # Character Descriptions Table (after Stage 4)
        with gr.Accordion("Character Descriptions Table", open=False) as char_table_accordion:
            character_table = gr.Dataframe(
                headers=["Character", "Description", "Audio", "Action"],
                datatype=["str", "str", "HTML", "HTML"],
                label="Characters",
                wrap=True
            )

        # Stage 5: Voice Samples
        with gr.Accordion("Stage 5: Voice Samples", open=False) as stage5:
            gr.Markdown("Generate voice samples for each character using TTS.")
            voice_samples_btn = gr.Button("Generate Voice Samples", variant="secondary")
            voice_samples_output = gr.Textbox(label="Voice Sample Results")

        # Stage 6: Full Audiobook (3-step process)
        with gr.Accordion("Stage 6: Generate Audiobook", open=False) as stage6:
            gr.Markdown("Generate the complete audiobook with all voices applied.")
            gr.Markdown("### Step 6.1: Generate TTS Audio")
            tts_btn = gr.Button("Generate TTS Audio", variant="secondary")
            tts_output = gr.Textbox(label="TTS Audio Generation Results")

            gr.Markdown("### Step 6.2: Validate and Clip Audio")
            validate_btn = gr.Button("Validate and Clip Audio", variant="secondary")
            validate_output = gr.Textbox(label="Validation Results")

            gr.Markdown("### Step 6.3: Assemble Chapter MP3s")
            assemble_btn = gr.Button("Assemble Chapter MP3s", variant="secondary")
            assemble_output = gr.Textbox(label="Assembly Results")

            gr.Markdown("### Full Pipeline")
            audiobook_btn = gr.Button("Generate Full Audiobook (All Steps)", variant="primary")
            audiobook_output = gr.Textbox(label="Full Audiobook Results")

        # Output file display
        with gr.Accordion("Generated Files", open=False):
            files_output = gr.Textbox(label="Chapter Files")

        # Hidden state for character table updates
        table_state = gr.State(None)

        # Event handlers
        parse_btn.click(
            fn=parse_epub_to_file,
            inputs=[epub_upload, max_chapters_slider],
            outputs=[parse_output, chapter_count_display, gr.Textbox(visible=False)]
        )

        label_btn.click(
            fn=process_chapters_for_labels,
            inputs=[api_key_input, port_input, num_attempts_input, log_output],
            outputs=log_output
        )

        analyze_btn.click(
            fn=analyze_chapters,
            inputs=log_output,
            outputs=analysis_output
        )

        describe_btn.click(
            fn=describe_characters,
            inputs=[api_key_input, port_input, log_output],
            outputs=descriptions_output
        ).then(
            fn=generate_character_table,
            outputs=character_table
        )

        voice_samples_btn.click(
            fn=generate_voice_samples,
            inputs=voice_samples_output,
            outputs=voice_samples_output
        ).then(
            fn=generate_character_table,
            outputs=character_table
        )

        # Handle button clicks in character table
        character_table.select(
            fn=on_character_table_select,
            inputs=[character_table, table_state, api_key_input, port_input, log_output],
            outputs=[table_state, log_output]
        )

        tts_btn.click(
            fn=generate_tts_audio,
            inputs=[tts_output, max_chapters_slider],
            outputs=tts_output
        )

        validate_btn.click(
            fn=validate_and_clip_audio,
            inputs=[validate_output, max_chapters_slider],
            outputs=validate_output
        )

        assemble_btn.click(
            fn=assemble_chapter_audiobooks,
            inputs=[assemble_output],
            outputs=assemble_output
        )

        audiobook_btn.click(
            fn=generate_full_audiobook,
            inputs=[audiobook_output, max_chapters_slider],
            outputs=audiobook_output
        )

        # Update files list periodically
        gr.on(
            triggers=[parse_btn.click, label_btn.click, analyze_btn.click],
            fn=lambda: "\n".join(glob.glob(str(get_chapters_dir() / "*"))),
            outputs=files_output
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
        demo.launch(share=False)
    finally:
        cleanup_temp_dir()