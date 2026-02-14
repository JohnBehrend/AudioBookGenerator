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

def generate_voice_samples(log_output, progress=gr.Progress()):
    """Stage 5: Generate voice samples for each character."""
    log_output += "\n\nGenerating voice samples..."

    try:
        # Check if characters_descriptions.json exists
        chapters_dir = get_chapters_dir()
        descriptions_file = SCRIPT_DIR / "characters_descriptions.json"
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
        processed_chars = 0
        for line in iter(process.stdout.readline, ''):
            if line:
                log_output += f"\n{line.strip()}"
                # Parse progress from output format: [1/10] character_name
                import re
                match = re.search(r'\[(\d+)/(\d+)\]', line)
                if match:
                    processed_chars = int(match.group(1))
                    total_chars = int(match.group(2))
                    progress(processed_chars / total_chars, desc=f"Character {processed_chars}/{total_chars}")

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


# ============================================================================
# Stage 6: Full Audiobook Generation
# ============================================================================

def generate_full_audiobook(log_output, max_chapters=None, progress=gr.Progress()):
    """Stage 6: Generate full audiobook with all chapters."""
    log_output += "\n\nGenerating full audiobook..."

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

        # Count chapters for progress tracking and pass to parse_epub.py
        chapter_files = sorted(glob.glob(str(chapters_dir / "chapter_*.txt")))
        num_chapters = len(chapter_files)
        log_output += f"\nGenerating audiobook for {num_chapters} chapters..."

        # Run parse_epub.py with the temp EPUB file, --resume, and --max-chapters
        # Use chapters_dir as output directory to avoid nested folders
        cmd = [sys.executable, str(SCRIPT_DIR / "parse_epub.py"), epub_path, "--resume", "--output-dir", str(chapters_dir)]
        if max_chapters:
            cmd.extend(["--max-chapters", str(max_chapters)])

        # Use subprocess with progress tracking (no cwd needed since we pass --output-dir)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(SCRIPT_DIR)
        )

        # Track progress based on output patterns
        import re
        completed_chapters = 0
        current_chapter = -1
        current_voice = ""
        current_line = -1
        current_retry = 0
        total_estimate = num_chapters * 10  # Rough estimate of total work units

        for line in iter(process.stdout.readline, ''):
            if line:
                line_stripped = line.strip()
                log_output += f"\n{line_stripped}"

                # Parse [CHAPTER_START] Chapter X of Y
                chapter_match = re.search(r'\[CHAPTER_START\]\s+Chapter\s+(\d+)', line_stripped)
                if chapter_match:
                    current_chapter = int(chapter_match.group(1))
                    log_output += f"\n[INFO] Starting chapter {current_chapter}"

                # Parse [VOICE_START] Processing voice N/M: name
                voice_match = re.search(r'\[VOICE_START\]\s+Processing\s+voice\s+(\d+)/(\d+):\s+(.+)', line_stripped)
                if voice_match:
                    voice_idx = int(voice_match.group(1))
                    total_voices = int(voice_match.group(2))
                    current_voice = voice_match.group(3)
                    log_output += f"\n[INFO] Voice {voice_idx}/{total_voices}: {current_voice}"

                # Parse [GENERATE_START] Chapter X, Line Y, Voice 'name'
                gen_match = re.search(r'\[GENERATE_START\]\s+Chapter\s+(\d+),\s+Line\s+(\d+)', line_stripped)
                if gen_match:
                    gen_chapter = int(gen_match.group(1))
                    gen_line = int(gen_match.group(2))
                    current_line = gen_line
                    log_output += f"\n[INFO] Generating line {gen_line} of chapter {gen_chapter}"

                # Parse [RETRY_START] Attempt N/5
                retry_match = re.search(r'\[RETRY_START\]\s+Attempt\s+(\d+)/5', line_stripped)
                if retry_match:
                    current_retry = int(retry_match.group(1))
                    log_output += f"\n[INFO] Retry attempt {current_retry}/5"

                # Parse [TTS_GENERATE] or [WHISPER_TRANSCRIBE] for more granular progress
                if "[TTS_GENERATE]" in line_stripped:
                    log_output += f"\n[INFO] TTS Generation in progress..."
                if "[TTS_COMPLETE]" in line_stripped:
                    log_output += f"\n[INFO] TTS Generation complete"
                if "[WHISPER_TRANSCRIBE]" in line_stripped:
                    log_output += f"\n[INFO] Whisper transcription in progress..."
                if "[WHISPER_ALIGN]" in line_stripped:
                    log_output += f"\n[INFO] Whisper alignment in progress..."
                if "[VALIDATION_COMPLETE]" in line_stripped:
                    log_output += f"\n[INFO] Validation complete"
                if "[CLIP_POSTFIX]" in line_stripped:
                    log_output += f"\n[INFO] Clipping postfix..."
                if "[CLIP_VALID]" in line_stripped:
                    log_output += f"\n[INFO] Clipping to valid token..."
                if "[SAVE_AUDIO_FINAL]" in line_stripped:
                    log_output += f"\n[INFO] Saving final audio..."
                if "[SAVE_AUDIO_FINAL_COMPLETE]" in line_stripped:
                    log_output += f"\n[INFO] Final audio saved"

                # Update progress based on current stage
                if "[CHAPTER_COMPLETE]" in line_stripped:
                    completed_chapters += 1
                    progress(completed_chapters / num_chapters, desc=f"Chapter {completed_chapters}/{num_chapters}")
                elif current_chapter >= 0:
                    # Estimate progress within a chapter based on line number
                    progress_estimate = completed_chapters + (current_line + 1) / 20.0
                    progress(min(progress_estimate / num_chapters, 1.0), desc=f"Chapter {completed_chapters + 1}/{num_chapters}")

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            if process.stderr:
                log_output += f"\nErrors: {process.stderr.read()}"
            log_output += "\nWarning: Process exited with non-zero code."

        progress(1, desc="Stage 6 Complete")
        log_output += "\n\nStage 6 complete!"
        return log_output

    except Exception as e:
        log_output += f"\nError generating audiobook: {str(e)}"
        return log_output


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create the Gradio interface with all stages."""

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Audiobook Pipeline Interface")

        # API Configuration
        with gr.Accordion("API Configuration", open=False):
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="LLM API Key",
                    value="lm-studio",
                    placeholder="Enter your API key (default: lm-studio)"
                )
                port_input = gr.Textbox(
                    label="LLM Port",
                    value="1234",
                    placeholder="Port for LLM inference"
                )
                num_attempts_input = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Number of LLM Attempts"
                )

        # Stage 1: EPUB Parsing
        with gr.Accordion("Stage 1: EPUB Parsing", open=True) as stage1:
            gr.Markdown("Upload an EPUB file and parse it into chapter text files.")
            epub_upload = gr.File(label="Upload EPUB File", file_types=[".epub"])
            parse_btn = gr.Button("Parse EPUB", variant="primary")
            parse_output = gr.Textbox(label="Status")
            chapter_count_display = gr.Number(label="Chapters Created", precision=0)
            max_chapters_slider = gr.Slider(
                minimum=1,
                maximum=100,
                value=10,
                step=1,
                label="Max Chapters to Generate"
            )

        # Stage 2: LLM Speaker Labeling
        with gr.Accordion("Stage 2: LLM Speaker Labeling", open=False) as stage2:
            gr.Markdown("Use LLM to identify speakers and attribute dialogue lines.")
            label_btn = gr.Button("Label All Chapters", variant="primary")
            log_output = gr.Textbox(label="Progress Log", lines=10, max_lines=20)

        # Stage 3: Chapter Analysis
        with gr.Accordion("Stage 3: Chapter Analysis", open=False) as stage3:
            gr.Markdown("Analyze chapter map files and generate statistics.")
            analyze_btn = gr.Button("Analyze Chapters", variant="secondary")
            analysis_output = gr.Textbox(label="Analysis Results")

        # Stage 4: Character Descriptions
        with gr.Accordion("Stage 4: Character Descriptions", open=False) as stage4:
            gr.Markdown("Generate voice profiles for each character using LLM.")
            describe_btn = gr.Button("Describe Characters", variant="secondary")
            descriptions_output = gr.Textbox(label="Character Descriptions")

        # Stage 5: Voice Samples
        with gr.Accordion("Stage 5: Voice Sample Generation", open=False) as stage5:
            gr.Markdown("Generate voice samples for each character using TTS.")
            voice_samples_btn = gr.Button("Generate Voice Samples", variant="secondary")
            voice_samples_output = gr.Textbox(label="Voice Sample Results")

        # Stage 6: Full Audiobook
        with gr.Accordion("Stage 6: Full Audiobook Generation", open=False) as stage6:
            gr.Markdown("Generate the complete audiobook with all voices applied.")
            audiobook_btn = gr.Button("Generate Full Audiobook", variant="primary")
            audiobook_output = gr.Textbox(label="Audiobook Generation Results")

        # Output file display
        with gr.Accordion("Generated Files", open=False):
            files_output = gr.Textbox(label="Chapter Files")

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
        )

        voice_samples_btn.click(
            fn=generate_voice_samples,
            inputs=voice_samples_output,
            outputs=voice_samples_output
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
    demo = create_interface()
    try:
        demo.launch(share=False)
    finally:
        cleanup_temp_dir()