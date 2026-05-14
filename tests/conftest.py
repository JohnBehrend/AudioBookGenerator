"""Shared pytest fixtures for audiobook_generator tests."""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_addoption(parser):
    """Add --run-slow CLI option for real engine tests."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow integration tests that use real TTS models",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow (requires --run-slow)")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is passed."""
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="requires --run-slow to run")
    for item in items:
        if item.fspath.basename == "test_real_engines.py":
            item.add_marker(skip_slow)

from audiobook_generator.testing import MockLLMClient, MockTTSEngine
from audiobook_generator.parse_chapter import ChapterObj, get_chapter_objs


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory(prefix="abg_test_") as d:
        yield Path(d)


@pytest.fixture
def sample_chapter_text():
    """Sample chapter text with dialogue and narration."""
    return '''Mr. Bennet was among the earliest of her neighbours in calling
upon Mrs. Bennet, and he entered the room with an air which decidedly
marked either his delight in the discovery of his wife in one of her
littleoramuseurs, or his wish to have theacolade in the greatest
perfection. "I beg you would not go," said she. "You had much rather have
the honour of it, I assure you." "Indeed, Mrs. Bennet, well I know and
have also experienced your hospitality, for my part I do not go to these
places." She left the room in great agitation. "I hope," said Mr. Bennet,
when they were alone, "that you may have had a pleasant ride."'''


@pytest.fixture
def sample_chapter_objs(sample_chapter_text):
    """ChapterObj list parsed from sample_chapter_text."""
    return get_chapter_objs(sample_chapter_text)


@pytest.fixture
def sample_quoted_only_text():
    """Text where every paragraph is quoted dialogue."""
    return '''"I cannot believe it," she said.
"This is absolutely wonderful news for us."
"But what about the others?" he asked.
"We must consider all possibilities," she replied.'''


@pytest.fixture
def sample_mixed_text():
    """Text with narration paragraphs and quoted dialogue paragraphs."""
    return '''Mr. Bennet was among the earliest of her neighbours in calling
upon Mrs. Bennet, and he entered the room with an air which decidedly
marked either his delight.

"I beg you would not go," said she.

He nodded in response. The room was silent for a moment.

"We must consider all possibilities," she replied.'''


@pytest.fixture
def sample_dialogue_text():
    """Dialogue-heavy text for speaker labeling tests."""
    return '''"I cannot go to London," said Jane.
"But mother insists you must visit," Elizabeth replied.
"Then I shall go, though I dread the journey," Jane said.
Elizabeth smiled at her sister. "It will be fine."
"It will not be fine," Jane said.'''


@pytest.fixture
def sample_map_json():
    """Sample character_map and line_map in list format."""
    return [
        {"1": "narrator", "2": "jane", "3": "elizabeth"},
        {"2": 2, "4": 2, "6": 3, "8": 3, "10": 2}
    ]


@pytest.fixture
def sample_map_dict():
    """Sample map in dict format."""
    return {
        "character_map": {"1": "narrator", "2": "jane", "3": "elizabeth"},
        "line_map": {"2": 2, "4": 2, "6": 3, "8": 3, "10": 2}
    }


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing without a running LLM server."""
    return MockLLMClient()


@pytest.fixture
def mock_tts_engine():
    """Mock TTS engine that generates silence audio files."""
    return MockTTSEngine()


@pytest.fixture
def mock_tts_engine_failure():
    """Mock TTS engine configured to fail generation."""
    return MockTTSEngine(generate_success=False, generate_voice_success=False)


@pytest.fixture
def sample_character_descriptions():
    """Sample character descriptions dict."""
    return {
        "narrator": "A calm, clear female narrator with a pleasant tone.",
        "jane": "A gentle, refined female voice with an elegant and reserved quality.",
        "elizabeth": "An intelligent, witty female voice with spirit and determination.",
    }


@pytest.fixture
def sample_voice_map():
    """Sample voice map mapping characters to voice file paths."""
    return {
        "narrator": "narrator.wav",
        "jane": "jane.wav",
        "elizabeth": "elizabeth.wav",
    }


@pytest.fixture
def sample_chapters_with_maps(temp_dir, sample_chapter_objs, sample_map_json):
    """Create sample chapter files with map files in temp directory."""
    chapters_dir = temp_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    chapters = [
        get_chapter_objs('''"I cannot believe it," said Jane.
"This is wonderful news," she added.
Elizabeth smiled at her sister.'''),
        get_chapter_objs('''"We must go to London," said Jane.
"It will be a grand adventure," Elizabeth replied.'''),
    ]

    for i, chapter in enumerate(chapters):
        chapter_file = chapters_dir / f"chapter_{i}.txt"
        with open(chapter_file, "w", encoding="utf-8") as f:
            for cobj in chapter:
                f.write(f"Line {cobj.line_num}: ")
                if cobj.has_quotes:
                    f.write('"')
                f.write(cobj.text)
                if cobj.has_quotes:
                    f.write('"')
                f.write("\n")

    map_data = [
        {"1": "narrator", "2": "jane", "3": "elizabeth"},
        {"2": 2, "4": 2, "6": 3}
    ]
    map_file = chapters_dir / "chapter_0.map.json"
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(map_data, f)

    return {
        "dir": chapters_dir,
        "chapters": chapters,
        "map_data": map_data,
    }


@pytest.fixture
def sample_epub_path():
    """Path to the sample EPUB file in voice_test directory."""
    path = Path(__file__).resolve().parent.parent / "voice_test" / "test_pride_and_prejudice.epub"
    if path.exists():
        return str(path)
    pytest.skip("Sample EPUB file not found")


@pytest.fixture
def voices_map_with_files(temp_dir, mock_tts_engine):
    """Create sample voice files and return a voice map."""
    voices = {
        "narrator": "narrator.wav",
        "jane": "jane.wav",
        "elizabeth": "elizabeth.wav",
    }

    for voice_file in voices.values():
        voice_path = temp_dir / voice_file
        import numpy as np
        import torch
        audio = np.zeros(int(mock_tts_engine.sample_rate * 1.0), dtype=np.float32)
        import torchaudio
        torchaudio.save(str(voice_path), torch.from_numpy(audio), mock_tts_engine.sample_rate)

    return {
        "dir": temp_dir,
        "map": voices,
    }