"""Shared pytest fixtures for audiobook_generator tests."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_addoption(parser):
    """Add --run-slow and --run-generate CLI options for real engine tests."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow integration tests that use real TTS models",
    )
    parser.addoption(
        "--run-generate",
        action="store_true",
        default=False,
        help="Run slow TTS generation tests (requires --run-slow too)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: slow integration tests requiring real models/GPU (skip unless --run-slow)",
    )
    config.addinivalue_line(
        "markers",
        "generate: tests that generate real audio (skip unless --run-generate)",
    )


def pytest_collection_modifyitems(config, items):
    """Gate slow/generate tests by marker rather than by filename.

    This is the single source of truth for which tests need a live TTS model:
    any test marked ``slow`` is skipped unless ``--run-slow`` is passed, and any
    test marked ``generate`` is skipped unless ``--run-generate`` is passed.
    Gating on markers (instead of hardcoding a filename) keeps the policy in one
    place and prevents new real-engine tests from silently running by default.
    """
    run_slow = config.getoption("--run-slow")
    run_generate = config.getoption("--run-generate")

    skip_slow = pytest.mark.skip(reason="requires --run-slow to run")
    skip_gen = pytest.mark.skip(reason="requires --run-generate to run")

    for item in items:
        if "generate" in item.keywords and not run_generate:
            item.add_marker(skip_gen)
        elif "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)

from audiobook_generator.testing import MockLLMClient, MockTTSEngine
from audiobook_generator.parse_chapter import ChapterObj, get_chapter_objs


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory(prefix="abg_test_") as d:
        yield Path(d)


@pytest.fixture(autouse=True)
def _isolated_celebrity_archive(monkeypatch, tmp_path_factory):
    """Redirect the celebrity voice archive to a temp dir for test isolation.

    Prevents tests from reading/writing the real repo-level archive, which would
    otherwise cause test pollution (e.g. a cached "test_celebrity.wav" making
    build_celebrity_voice return early and never invoke the mocked pipeline).
    """
    archive_dir = tmp_path_factory.mktemp("celebrity_archive")
    monkeypatch.setattr(
        "audiobook_generator.celebrity_voices._archive_dir",
        lambda: str(archive_dir),
    )


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
def sample_epub_path():
    """Path to the sample EPUB file in voice_test directory."""
    path = Path(__file__).resolve().parent.parent / "voice_test" / "test_pride_and_prejudice.epub"
    if path.exists():
        return str(path)
    pytest.skip("Sample EPUB file not found")


@pytest.fixture
def sample_voices_map():
    """Sample voices map mapping characters to voice file paths."""
    return {
        "narrator": "narrator.wav",
        "jane": "jane.wav",
        "elizabeth": "elizabeth.wav",
    }


@pytest.fixture
def sample_chapters():
    """Sample chapter objects used by audiobook pipeline tests."""
    return [
        [
            ChapterObj(False, "Narrator text", 1),
            ChapterObj(True, '"Hello there," said Jane.', 2),
            ChapterObj(False, "Narrator continues.", 3),
        ],
        [
            ChapterObj(True, '"Good morning," Elizabeth replied.', 1),
            ChapterObj(False, "The room was silent.", 2),
        ],
    ]


@pytest.fixture
def sample_chapter_maps():
    """Sample chapter speaker/line maps used by audiobook pipeline tests."""
    return {
        0: ({"1": "narrator", "2": "jane"}, {"2": 2}),
        1: ({"1": "elizabeth"}, {"1": 1}),
    }


@pytest.fixture
def mock_voice_mapper():
    """VoiceMapper whose engine and voice paths resolve without a real TTS engine."""
    from unittest.mock import MagicMock
    from audiobook_generator.testing import MockTTSEngine

    mapper = MagicMock()
    mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
    mapper.get_engine.return_value = MockTTSEngine()
    return mapper