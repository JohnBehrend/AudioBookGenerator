"""Tests for parse_epub.py (Stage 6: Full Audiobook Generation)."""
import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Test EPUB path
TEST_EPUB_PATH = Path(__file__).parent.parent / "test_data" / "pride_and_prejudice.epub"


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


class TestFullPipelineIntegration:
    """Integration test for full pipeline using test EPUB file."""

    def test_full_pipeline_first_chapter(self, temp_dir):
        """Test full pipeline on first chapter with mocked TTS/STT models."""
        from parse_chapter import parse_epub_to_chapters
        from parse_epub import generate_audiobook_from_chapters

        # Parse just the first chapter for a quick test
        chapters = parse_epub_to_chapters(str(TEST_EPUB_PATH), max_chapters=1)

        assert len(chapters) >= 1, "Should have at least one chapter"
        assert len(chapters[0]) > 0, "Chapter should have content"

        # Create a simple map file for the first chapter
        chapter = chapters[0]
        character_map = {}
        line_map = {}

        # Assign first quoted line to "elizabeth", others to narrator
        line_idx = 0
        character_idx = 1
        for cobj in chapter:
            if cobj.has_quotes:
                if line_idx not in line_map:
                    line_map[str(line_idx)] = character_idx
                    character_map[str(character_idx)] = "elizabeth"
                    character_idx += 1
            else:
                if line_idx not in line_map:
                    line_map[str(line_idx)] = 0
                    character_map["0"] = "narrator"
            line_idx += 1

        # Ensure we have at least narrator
        if "0" not in character_map:
            character_map["0"] = "narrator"

        # Convert to proper dict format with int keys
        character_map_int = {int(k): v for k, v in character_map.items()}
        line_map_int = {int(k): v for k, v in line_map.items()}

        # Create voices_map with a fake voice file path (will be mocked)
        voices_map = {"narrator": "narrator.wav", "elizabeth": "elizabeth.wav"}

        # Mock generate_tts_for_line and validation model setup to return success without running actual TTS
        with patch("parse_epub.generate_tts_for_line") as mock_generate_tts, \
             patch("parse_epub.setup_validation_model") as mock_setup_val:

            # Setup the TTS generation mock to return success
            mock_generate_tts.return_value = (True, 0.95)

            # Setup validation model mock
            mock_val_model = Mock()
            mock_setup_val.return_value = mock_val_model

            # Run the pipeline - chapter_maps expects tuple of dicts with int keys
            result_msg, chapters_processed = generate_audiobook_from_chapters(
                chapters=chapters,
                chapter_maps={0: (character_map_int, line_map_int)},
                voices_map=voices_map,
                output_dir=str(temp_dir),
                device="cpu",  # Use CPU for test
                tts_engine="kugelaudio",
                cfg_scale=1.30,
                max_chapters=1,
                verbose=True
            )

            # Verify results
            assert chapters_processed == 1, "Should process 1 chapter"
            assert "successfully" in result_msg.lower(), "Should report success"

            # Verify TTS was called for at least one line
            assert mock_generate_tts.called, "TTS generation should have been called"


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
