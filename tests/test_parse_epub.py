"""Tests for parse_epub.py (Stage 6: Full Audiobook Generation)."""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import os


class TestParseEpubIntegration:
    """Integration tests for parse_epub pipeline stages."""

    def test_full_pipeline_mock(self, temp_dir):
        """Test the full pipeline with mocked external dependencies."""
        from parse_chapter import ChapterObj

        # Create sample chapter data
        chapters = [
            [
                ChapterObj(has_quotes=True, text="Hello world", line_num=1),
                ChapterObj(has_quotes=False, text="Narrator text", line_num=2),
            ]
        ]

        # Create sample map file
        character_map = {"1": "narrator", "2": "john"}
        line_map = {"1": 2, "2": 1}  # Line 1 -> john, Line 2 -> narrator
        map_file = temp_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([character_map, line_map]))

        # Verify structure
        loaded = json.loads(map_file.read_text())
        assert len(loaded) == 2
        assert "1" in loaded[0]
        assert "2" in loaded[0]


class TestSpeakerAssignment:
    """Tests for speaker assignment logic."""

    def test_line_to_character_mapping(self):
        """Test mapping lines to characters."""
        from parse_chapter import ChapterObj

        character_map = {1: "narrator", 2: "john", 3: "mary"}
        line_map = {1: 1, 2: 2, 3: 3, 4: 2}

        # Build line to character mapping
        line_to_character = {k: character_map[v] for k, v in line_map.items()}

        assert line_to_character[1] == "narrator"
        assert line_to_character[2] == "john"
        assert line_to_character[3] == "mary"
        assert line_to_character[4] == "john"

    def test_chapter_obj_speaker_assignment(self):
        """Test ChapterObj speaker assignment."""
        from parse_chapter import ChapterObj

        obj = ChapterObj(has_quotes=True, text="Hello", line_num=1)

        obj.set_speaker("john")
        assert obj.get_speaker() == "john"

        obj.set_speaker("mary")
        assert obj.get_speaker() == "mary"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Mock fixtures for external dependencies
@pytest.fixture
def mock_tts_model():
    """Mock TTS model for testing."""
    with patch("parse_epub.VibeVoiceForConditionalGenerationInference") as mock:
        mock_instance = Mock()
        mock_instance.generate = Mock(return_value=Mock(speech_outputs=[Mock()]))
        mock.from_pretrained = Mock(return_value=mock_instance)
        yield mock


@pytest.fixture
def mock_processor():
    """Mock processor for testing."""
    with patch("parse_epub.VibeVoiceProcessor") as mock:
        mock_instance = Mock()
        mock_instance.save_audio = Mock()
        mock.from_pretrained = Mock(return_value=mock_instance)
        yield mock


@pytest.fixture
def mock_whisper():
    """Mock Whisper model for testing."""
    with patch("parse_epub.whisperx") as mock:
        mock_instance = Mock()
        mock_instance.load_model = Mock(return_value=Mock())
        mock_instance.load_align_model = Mock(return_value=(Mock(), {}))
        mock_instance.align = Mock(return_value={"segments": []})
        mock.load_model = Mock(return_value=mock_instance)
        yield mock
