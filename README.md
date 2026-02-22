# Audiobook Pipeline

A tool for creating synthesized audiobooks with distinct voices for different characters. Processes EPUB files through a 5-stage pipeline with both CLI and Gradio interfaces.

## Quick Start

### CLI Mode (Default)

Run the full pipeline from EPUB to audiobook in one command:

```bash
uv run python audiobook_generator.py <epub_file> [--verbose]
```

### Gradio Web Interface

Launch the interactive web interface:

```bash
uv run python audiobook_generator.py --gradio
```

Then open the provided URL (typically http://127.0.0.1:7860) in your browser.

## Pipeline Stages

1. **EPUB Parsing** - Parse an EPUB file into chapter text files
2. **LLM Speaker Labeling** - Use an LLM to identify speakers and attribute dialogue lines
3. **Character Descriptions** - Generate voice profiles for each character
4. **Voice Sample Generation** - Generate TTS voice samples for each character
5. **Full Audiobook Generation** - Generate complete audiobook MP3 files

## Usage

### CLI

```bash
uv run python audiobook_generator.py <epub_file> [OPTIONS]

# Options:
# --output-dir DIR     Output directory (default: chapters)
# --max-chapters N     Maximum number of chapters to process
# --verbose, -v        Print verbose output
# --api-key KEY        LLM API key
# --port PORT          LLM port
# --tts-engine ENGINE  TTS engine: kugelaudio (default) or vibevoice
# --device DEVICE      CUDA device (default: cuda)
# --num-llm-attempts N Number of LLM attempts (default: 2)

# Launch Gradio interface
uv run python audiobook_generator.py --gradio
```

### Individual Stages (Advanced)

Each pipeline stage can also be run independently:

```bash
# Stage 1: Parse EPUB (handled automatically by CLI)
# Uses: parse_chapter.parse_epub_to_chapters()

# Stage 2: Label Speakers
uv run python llm_label_speakers.py -txt_file chapters/chapter_0.txt

# Stage 3: Describe Characters
uv run python llm_describe_character.py chapters

# Stage 4: Generate Voice Samples
uv run python generate_voice_samples.py

# Stage 5: Generate Audiobook (handled automatically by CLI)
# Uses: audiobook_generator.generate_audiobook_from_chapters()
```

## Requirements

Install all dependencies:

```bash
uv sync
```

### Core Dependencies

- **Python 3.8+**
- **PyTorch** (with CUDA support optional: `torch`, `torchaudio`, `torchvision`)
- **Transformers** and **tokenizers** (HuggingFace)
- **Gradio** for the web interface
- **WhisperX** / **faster-whisper** for STT validation
- **OpenAI API** client for LLM operations

### TTS Engine: KugelAudio

The default TTS engine is KugelAudio, installed as a Python package dependency from GitHub:

- **Installation**: Automatically installed via `uv sync` (uses `pyproject.toml`)
- **Version**: Pinned to commit `c2047edda01aa31e9472d29eac498881e907d628`

The package is installed directly from the GitHub repository and requires no manual setup.

### Audio Processing

- **pydub** for audio manipulation (requires system ffmpeg)
- **scipy** for signal processing
- **pandas** for data handling

### System Dependencies

- **FFmpeg** - Required for audio processing via pydub

  **Windows (using winget):**
  ```bash
  winget install Gyan.FFmpeg
  ```

  **macOS (using Homebrew):**
  ```bash
  brew install ffmpeg
  ```

  **Linux (using apt):**
  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```

  Ensure ffmpeg is on your system PATH. Verify with `ffmpeg -version`.

### EPUB Processing

- **ebooklib** for EPUB parsing
- **beautifulsoup4** for HTML parsing


### LLM Settings

The pipeline uses an OpenAI-compatible API (LM Studio by default):
- **Host**: localhost
- **Port**: 1234 (default)
- **API Key**: lm-studio (default, can be any string)

### GPU Settings

- Default: `cuda:0`
- Alternate GPU: Use `--device cuda:1` flag

## Output Files

- `chapters/chapter_*.txt` - Parsed chapter text files
- `chapters/chapter_*.map.json` - Speaker attribution maps
- `chapters/chapter_*.wav` - Individual voice samples
- `chapters/chapter_*.mp3` - Final audiobook chapters
- `characters_descriptions.json` - Character voice descriptions

## File Structure

| File | Purpose |
|------|---------|
| `audiobook_generator.py` | Main entry point - CLI pipeline + TTS generation + Gradio launcher |
| `audiobook_gradio_ui.py` | Gradio UI components and event handlers |
| `parse_chapter.py` | EPUB parsing + `write_chapters_to_txt()` helper |
| `llm_label_speakers.py` | Stage 2 - LLM speaker labeling |
| `llm_describe_character.py` | Stage 3 - Character descriptions |
| `generate_voice_samples.py` | Stage 4 - Voice sample generation |
| `config.py` | Shared configuration |

## Notes

- All generated audio files are in the `chapters/` directory (gitignored)
- Linux line endings (LF) are used throughout the repository
- Character names are normalized (lowercase, simplified)
