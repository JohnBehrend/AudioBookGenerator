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

def parse_epub_to_file(epub_file):
    """Stage 1: Parse EPUB file into chapter text files."""
    if epub_file is None:
        return "Error: No EPUB file uploaded.", 0, []

    try:
        # Import parse_chapter module
        from parse_chapter import parse_epub_to_chapters

        # Get the temp chapters directory
        chapters_dir = get_chapters_dir()

        # Parse the EPUB
        chapters = parse_epub_to_chapters(epub_file.name)

        if not chapters:
            return "Error: No chapters found in EPUB file.", 0, []

        # Save each chapter as a text file
        chapter_files = []
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
# Stage 2: LLM Speaker Labeling
# ============================================================================

def process_chapters_for_labels(api_key, port, num_attempts, use_all_chapters, chapter_range, log_output):
    """Stage 2: Run LLM to label speakers in selected chapters."""
    # Get list of chapter files
    chapters_dir = get_chapters_dir()
    chapter_files = sorted(glob.glob(str(chapters_dir / "chapter_*.txt")))

    if not chapter_files:
        log_output += "\nNo chapter files found. Please run Stage 1 first."
        return log_output

    # Determine which chapters to process
    if use_all_chapters:
        selected_chapters = chapter_files
    else:
        # Handle both list (range) and single int values
        if isinstance(chapter_range, (list, tuple)) and len(chapter_range) >= 2:
            start_chapter, end_chapter = chapter_range
        else:
            # Fallback: use single value as both start and end
            start_chapter = int(chapter_range) if chapter_range else 0
            end_chapter = start_chapter
        selected_chapters = [
            str(chapters_dir / f"chapter_{i}.txt")
            for i in range(start_chapter, end_chapter + 1)
            if os.path.exists(str(chapters_dir / f"chapter_{i}.txt"))
        ]

    if not selected_chapters:
        log_output += "\nNo chapters selected for processing."
        return log_output

    log_output += f"\nProcessing {len(selected_chapters)} chapters with LLM..."

    for chapter_file in selected_chapters:
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
                capture_output=True,
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

    log_output += "\n\nStage 2 complete!"
    return log_output


# ============================================================================
# Stage 3: Chapter Analysis
# ============================================================================

def analyze_chapters(log_output):
    """Stage 3: Analyze chapter map files and generate statistics."""
    log_output += "\n\nRunning chapter analysis..."

    try:
        # Check if map files exist
        chapters_dir = get_chapters_dir()
        map_files = glob.glob(str(chapters_dir / "*.map.json"))
        if not map_files:
            log_output += "\nNo .map.json files found. Please run Stage 2 first."
            return log_output

        # Run analyze_chapters.py
        cmd = [sys.executable, str(SCRIPT_DIR / "analyze_chapters.py"), str(chapters_dir), "--json-output", "--verbose"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR))

        log_output += result.stdout
        if result.stderr:
            log_output += f"\nErrors: {result.stderr}"

        log_output += "\n\nStage 3 complete!"
        return log_output

    except Exception as e:
        log_output += f"\nError analyzing chapters: {str(e)}"
        return log_output


# ============================================================================
# Stage 4: Character Descriptions
# ============================================================================

def describe_characters(api_key, port, log_output):
    """Stage 4: Use LLM to describe characters."""
    log_output += "\n\nGenerating character descriptions..."

    try:
        # Check if characters.json exists
        chapters_dir = get_chapters_dir()
        if not os.path.exists(str(chapters_dir / "characters.json")):
            log_output += "\ncharacters.json not found. Please run Stage 3 first."
            return log_output

        # Run llm_describe_character.py
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "llm_describe_character.py"),
            str(chapters_dir / "characters.json"),
            str(chapters_dir),
            "--api_key", api_key,
            "--port", port,
            "--verbose"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
        log_output += result.stdout
        if result.stderr:
            log_output += f"\nErrors: {result.stderr}"

        log_output += "\n\nStage 4 complete!"
        return log_output

    except Exception as e:
        log_output += f"\nError describing characters: {str(e)}"
        return log_output


# ============================================================================
# Stage 5: Voice Sample Generation
# ============================================================================

def generate_voice_samples(log_output):
    """Stage 5: Generate voice samples for each character."""
    log_output += "\n\nGenerating voice samples..."

    try:
        # Check if characters_descriptions.json exists
        chapters_dir = get_chapters_dir()
        if not os.path.exists(str(SCRIPT_DIR / "characters_descriptions.json")):
            log_output += "\ncharacters_descriptions.json not found. Please run Stage 4 first."
            return log_output

        # Run generate_voice_samples.py
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "generate_voice_samples.py"),
            "--descriptions", str(SCRIPT_DIR / "characters_descriptions.json"),
            "--output-dir", str(chapters_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
        log_output += result.stdout
        if result.stderr:
            log_output += f"\nErrors: {result.stderr}"

        log_output += "\n\nStage 5 complete!"
        return log_output

    except Exception as e:
        log_output += f"\nError generating voice samples: {str(e)}"
        return log_output


# ============================================================================
# Stage 6: Full Audiobook Generation
# ============================================================================

def generate_full_audiobook(log_output):
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

        # Run parse_epub.py
        cmd = [sys.executable, str(SCRIPT_DIR / "parse_epub.py"), "--resume"]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
        log_output += result.stdout
        if result.stderr:
            log_output += f"\nErrors: {result.stderr}"

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

        # Stage 2: LLM Speaker Labeling
        with gr.Accordion("Stage 2: LLM Speaker Labeling", open=False) as stage2:
            gr.Markdown("Use LLM to identify speakers and attribute dialogue lines.")
            all_chapters_checkbox = gr.Checkbox(
                label="Process All Chapters",
                value=True
            )
            chapter_range_slider = gr.Slider(
                minimum=0,
                maximum=10,
                value=[0, 5],
                step=1,
                label="Chapter Range (disable when All Chapters is checked)",
                interactive=False
            )

            # Update slider interactivity when checkbox changes
            all_chapters_checkbox.change(
                lambda x: gr.update(interactive=not x),
                inputs=all_chapters_checkbox,
                outputs=chapter_range_slider
            )

            label_btn = gr.Button("Label Selected Chapters", variant="primary")
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
            inputs=epub_upload,
            outputs=[parse_output, chapter_count_display, gr.Textbox(visible=False)]
        )

        # Update chapter range slider when chapters are created
        parse_btn.click(
            fn=lambda count: gr.update(maximum=max(0, count - 1) if count else 10),
            inputs=chapter_count_display,
            outputs=chapter_range_slider
        )

        label_btn.click(
            fn=process_chapters_for_labels,
            inputs=[api_key_input, port_input, num_attempts_input, all_chapters_checkbox, chapter_range_slider, log_output],
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
            inputs=audiobook_output,
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