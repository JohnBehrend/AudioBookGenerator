# Audiobook Pipeline

A tool for creating synthesized audiobooks with distinct voices for different characters. Processes EPUB files through a 6-stage pipeline.

## Quick Start

```bash
# Start the Gradio web interface
python gradio_audiobook_interface.py
```

Then open the provided URL (typically http://127.0.0.1:7860) in your browser.

## Pipeline Stages

1. **EPUB Parsing** - Upload an EPUB file and parse it into chapter text files
2. **LLM Speaker Labeling** - Use an LLM to identify speakers and attribute dialogue lines
3. **Character Descriptions** - Generate voice profiles for each character
4. **Voice Sample Generation** - Generate TTS voice samples for each character
5. **Full Audiobook Generation** - Generate complete audiobook MP3 files

## Individual Commands

The Gradio interface wraps these individual commands:

```bash
# Parse EPUB and generate chapter text files
python parse_epub.py <epub_file> [-voices_map voices_map.json] [--resume] [--alt_gpu]

# Label speakers in a chapter using LLM
python llm_label_speakers.py -txt_file chapters/chapter_0.txt [--skip_llm] [--old_format]

# Describe characters using LLM
python llm_describe_character.py chapters [--api_key] [--port]

# Generate voice samples
python generate_voice_samples.py [--descriptions] [--output-dir]
```

## Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
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
- Alternate GPU: Use `--alt_gpu` flag in `parse_epub.py`

## Output Files

- `chapters/chapter_*.txt` - Parsed chapter text files
- `chapters/chapter_*.map.json` - Speaker attribution maps
- `chapters/chapter_*.wav` - Individual voice samples
- `chapters/chapter_*.mp3` - Final audiobook chapters
- `characters_descriptions.json` - Character voice descriptions

## Notes

- All generated audio files are in the `chapters/` directory (gitignored)
- Linux line endings (LF) are used throughout the repository
- Character names are normalized (lowercase, simplified)