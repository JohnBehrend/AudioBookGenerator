# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **audiobook creation pipeline** that processes EPUB files into synthesized audiobooks with distinct voices for different characters. The workflow involves:

1. **parse_epub.py** - Parses EPUB files into chapter text files
2. **llm_label_speakers.py** - Uses an LLM (via LM Studio API) to identify speakers and attribute dialogue lines
3. **analyze_chapters.py** - Analyzes chapter map files and generates statistics
4. **list_chapters.py** - Lists chapter map files in a directory

## Architecture

### Data Flow

```
EPUB File → parse_epub.py → chapters/*.txt
                    ↓
llm_label_speakers.py → chapters/*.map.json (character_map, line_map)
                    ↓
parse_epub.py (continued) → chapters/*.mp3 (audiobook)
```

### Map JSON Format

Each `chapter_X.map.json` contains a two-element array:
1. **character_map**: `{"1": "narrator", "2": "character_name", ...}` (keys are strings)
2. **line_map**: `{"line_number": speaker_key, ...}` (keys are strings, values are ints)

Note: Some map files (e.g., chapter_1.map.json) may have inconsistent formatting - keys in character_map may be unquoted.

### Key Dependencies

- **TTS**: VibeVoice (HuggingFace Transformers, PyTorch)
- **STT**: WhisperX for validation
- **LLM**: OpenAI-compatible API (LM Studio) for speaker labeling
- **Audio processing**: pydub, scipy

## Commands

```bash
# Label speakers in a chapter using LLM
python llm_label_speakers.py -txt_file chapters/chapter_0.txt [--skip_llm] [--old_format]

# Parse EPUB and generate audiobook
python parse_epub.py <epub_file> [-voices_map voices_map.json] [--resume] [--alt_gpu]

# Analyze chapter maps
python analyze_chapters.py chapters [--json-output] [--csv-output] [--verbose]

# List chapter files
python list_chapters.py chapters
```

## Important Notes

- The main LLM prompt is in `llm_label_speakers.py` (lines 244-282) - it requires **every** quoted line (lines that both start and end with `"`) to be attributed
- Character names are normalized (lowercase, simplified)
- Multiple LLM attempts are merged, with duplicate characters consolidated by line mapping overlap
- Audio generation uses a postfix for validation and clips it from final output
- GPU selection: `--alt_gpu` uses cuda:1, default is cuda:0
- Generated audio files are in `chapters/` (gitignored)

## Known Issues

- **Missing dependency**: `parse_epub.py` imports `parse_chapter` module which doesn't exist in the repository
- **Syntax errors**: `analyze_chapters.py` and `list_chapters.py` have mismatched parentheses that prevent execution
- **Inconsistent map.json format**: Some map files (e.g., chapter_1.map.json) have unquoted keys in character_map