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
# --whisper-cpu        Run Whisper validation on CPU
# --whisper-concurrency N  Number of concurrent Whisper models for validation (default: 1)
# --whisper-fast       Use faster Whisper settings (medium model, beam_size=3)
# --gpus GPU [GPU ...] GPU devices to use (e.g., --gpus cuda:0 cuda:1)
# --use-chunkformer    Enable ChunkFormer voice validation (gender/emotion/dialect/age classification)

# Launch Gradio interface
uv run python audiobook_generator.py --gradio
```

### Individual Stages (Advanced)

The pipeline runs all 5 stages automatically. For fine-grained control, use CLI options:

```bash
# Run stages 1-3 only (parsing, labeling, character descriptions)
uv run python audiobook_generator.py <epub_file> --max-chapters 3

# Resume from a specific stage by passing existing output files
uv run python audiobook_generator.py --skip-existing
```

**Internal Stage Files:**
| Stage | File | Function |
|-------|------|----------|
| 1 | `audiobook_generator/parse_chapter.py` | `parse_epub_to_chapters()` |
| 2 | `audiobook_generator/llm_label_speakers.py` | `label_speakers()` |
| 3 | `audiobook_generator/llm_describe_character.py` | `describe_characters()` |
| 4 | `audiobook_generator/generate_voice_samples.py` | `generate_voice_samples()` |
| 5 | `audiobook_generator/audiobook_generator.py` | `generate_audiobook_from_chapters()` |

## Requirements

Install all dependencies:

```bash
uv sync
```

### Core Dependencies

- **Python 3.11+** (required for ChunkFormer voice validation)
- **PyTorch** (with CUDA support optional: `torch`, `torchaudio`, `torchvision`)
- **Transformers** and **tokenizers** (HuggingFace)
- **Gradio** for the web interface
- **WhisperX** / **faster-whisper** for STT validation
- **OpenAI API** client for LLM operations
- **ChunkFormer** for voice classification validation

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

### Voice Validation

The pipeline includes multiple layers of audio validation:

1. **Whisper Transcription Validation**: TTS output is transcribed with Whisper and compared to the expected text using string similarity. Audio is clipped at valid word boundaries and retries are performed if quality is below threshold (0.85). Controlled by `--whisper-cpu`, `--whisper-fast`, and `--whisper-concurrency` flags.

2. **ChunkFormer Voice Validation**: When enabled (`--use-chunkformer`), the ChunkFormer model classifies generated voice samples by gender, emotion, dialect, and age, then compares the classification against the character description to ensure gender match. Uses the `khanhld/chunkformer-gender-emotion-dialect-age-classification` model. Requires Python >= 3.11.


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

```
AudioBookGenerator/
├── audiobook_generator.py              # Main CLI entry point
├── audiobook_generator/
│   ├── __init__.py                     # Package initialization
│   ├── audiobook_generator.py          # Stage 5: Full audiobook TTS generation
│   ├── config.py                       # Shared configuration
│   ├── engines/                        # TTS engine abstraction
│   ├── generate_voice_samples.py       # Stage 4: Voice sample generation
│   ├── gradio_ui.py                    # Gradio web interface
│   ├── llm_describe_character.py       # Stage 3: Character descriptions
│   ├── llm_label_speakers.py           # Stage 2: Speaker labeling
│   ├── parse_chapter.py                # Stage 1: EPUB parsing
│   ├── pipeline.py                     # Pure functions for TTS logic
│   ├── testing.py                      # Test utilities (MockLLMClient, etc.)
│   ├── utils.py                        # Utility functions
│   └── voice_mapper.py                 # Voice mapping logic
└── tests/                              # Test suite (227 tests)
```

## Testing

Run the test suite:

```bash
uv run pytest tests/ -v
```

Run specific test files:

```bash
uv run pytest tests/test_pipeline.py -v
```

The test suite includes:
- EPUB parsing tests
- LLM speaker labeling tests
- Character description tests
- Voice mapper tests
- Utility function tests
- Pipeline pure function tests

## Notes

- All generated audio files are in the `chapters/` directory (gitignored)
- Linux line endings (LF) are used throughout the repository
- Character names are normalized (lowercase, simplified)
- Pure functions in `pipeline.py` are designed for unit testing without GPU