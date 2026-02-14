# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **audiobook creation pipeline** that processes EPUB files into synthesized audiobooks with distinct voices for different characters. The workflow is orchestrated through a Gradio web interface.

## Architecture

### Data Flow

```
EPUB File → gradio_audiobook_interface.py → chapters/*.txt
                              ↓
                    llm_label_speakers.py → chapters/*.map.json
                              ↓
                    analyze_chapters.py → chapter_analysis.json, characters.json
                              ↓
                    llm_describe_character.py → characters_descriptions.json
                              ↓
                    generate_voice_samples.py → chapters/*.wav
                              ↓
                    parse_epub.py → chapters/*.mp3 (audiobook)
```

### Map JSON Format

Each `chapter_X.map.json` contains a two-element array:
1. **character_map**: `{"1": "narrator", "2": "character_name", ...}` (keys are strings)
2. **line_map**: `{"line_number": speaker_key, ...}` (keys are strings, values are ints)

## Gradio Interface

Run the Gradio interface with:

```bash
python gradio_audiobook_interface.py
```

The interface provides a 6-stage workflow:
1. **EPUB Parsing** - Upload an EPUB file and parse it into chapter text files
2. **LLM Speaker Labeling** - Use LLM to identify speakers and attribute dialogue lines
3. **Chapter Analysis** - Generate statistics from chapter map files
4. **Character Descriptions** - Generate voice profiles for each character
5. **Voice Sample Generation** - Generate TTS voice samples for each character
6. **Full Audiobook Generation** - Generate complete audiobook MP3 files

## Key Dependencies

- **TTS**: VibeVoice (HuggingFace Transformers, PyTorch)
- **STT**: WhisperX for validation
- **LLM**: OpenAI-compatible API (LM Studio) for speaker labeling
- **Audio processing**: pydub, scipy
- **Web UI**: Gradio

## Commands

```bash
# Start Gradio web interface
python gradio_audiobook_interface.py

# Individual stage commands (also available via Gradio):
python parse_epub.py <epub_file> [-voices_map voices_map.json] [--resume] [--alt_gpu]
python llm_label_speakers.py -txt_file chapters/chapter_0.txt [--skip_llm] [--old_format]
python analyze_chapters.py chapters [--json-output] [--csv-output] [--verbose]
python llm_describe_character.py characters.json chapters [--api_key] [--port]
python generate_voice_samples.py [--descriptions] [--output-dir]
```

## Important Notes

- The main LLM prompt is in `llm_label_speakers.py` (lines 244-282) - it requires **every** quoted line (lines that both start and end with `"`) to be attributed
- Character names are normalized (lowercase, simplified)
- Multiple LLM attempts are merged, with duplicate characters consolidated by line mapping overlap
- Audio generation uses a postfix for validation and clips it from final output
- GPU selection: `--alt_gpu` uses cuda:1, default is cuda:0
- Generated audio files are in `chapters/` (gitignored)
- Linux line endings (LF) are used throughout the repository

## Before Making Changes

**Always verify syntax first:** Run `python -m py_compile <file.py>` before assuming there are syntax errors. Python's syntax checker is reliable - if it compiles, the syntax is correct.

**Check git diff after fixes:** If you make changes to fix reported errors, verify with `git diff` that actual code was changed. If no diff appears, the reported error may not exist.

**Prefer running tests:** Run the test suite (`python -m pytest tests/`) to identify real issues before assuming syntax errors.