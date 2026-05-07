"""Integration tests for the full audiobook pipeline with mocks.

These tests verify that the pipeline stages work together correctly:
- EPUB parsing → LLM speaker labeling → Character descriptions → Voice sample generation → TTS generation
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from audiobook_generator.testing import MockTTSEngine, MockLLMClient
from audiobook_generator.parse_chapter import ChapterObj, get_chapter_objs


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_chapters():
    """Create sample chapter objects."""
    chapter1 = [
        ChapterObj(False, "The story begins.", 1),
        ChapterObj(True, '"Hello," said Jane.', 2),
        ChapterObj(False, "Jane continued walking.", 3),
        ChapterObj(True, '"Goodbye," Elizabeth replied.', 4),
    ]

    chapter2 = [
        ChapterObj(True, '"We must go now," said Jane.', 1),
        ChapterObj(False, "They left the room.", 2),
        ChapterObj(True, '"Farewell," Elizabeth said.', 3),
    ]

    return [chapter1, chapter2]


@pytest.fixture
def sample_chapter_maps():
    """Create sample chapter maps."""
    return {
        0: ({"1": "narrator", "2": "jane", "3": "narrator", "4": "elizabeth"}, {"2": 2, "4": 4}),
        1: ({"1": "jane", "2": "narrator", "3": "elizabeth"}, {"1": 1, "3": 3}),
    }


@pytest.fixture
def sample_voices_map():
    """Create sample voices map."""
    return {
        "narrator": "narrator.wav",
        "jane": "jane.wav",
        "elizabeth": "elizabeth.wav",
    }


@pytest.fixture
def mock_tts_engine():
    """Create mock TTS engine."""
    return MockTTSEngine()


# ============================================================================
# TESTS
# ============================================================================

class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_with_mocks(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Test full pipeline from chapters to audiobook generation."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with patch("audiobook_generator.audiobook_generator.setup_validation_model") as mock_validation:
            mock_validation.return_value = MagicMock()
            with patch("audiobook_generator.audiobook_generator.get_validation_client"):
                with patch("audiobook_generator.audiobook_generator.VoiceMapper") as mock_mapper:
                    mock_mapper.return_value = MagicMock()
                    mock_mapper.return_value.add_voice_path.return_value = None
                    with patch("audiobook_generator.audiobook_generator.generate_tts_for_line") as mock_tts:
                        mock_tts.return_value = (True, 0.95)
                        with patch("audiobook_generator.audiobook_generator.get_non_silent_audio_from_wavs") as mock_wavs:
                            mock_audio = MagicMock()
                            mock_wavs.return_value = mock_audio
                            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                                with patch("audiobook_generator.audiobook_generator.glob.glob", return_value=["/tmp/chapter_00.0002.wav"]):
                                    with patch("audiobook_generator.audiobook_generator.ProgressHandler"):
                                        with patch("audiobook_generator.audiobook_generator.os.makedirs"):
                                            with patch("audiobook_generator.audiobook_generator.os.unlink"):
                                                with patch("audiobook_generator.audiobook_generator.os.path.join", side_effect=os.path.join):
                                                    with patch("audiobook_generator.audiobook_generator.gc.collect"):
                                                        result = generate_audiobook_from_chapters(
                                                            chapters=sample_chapters,
                                                            chapter_maps=sample_chapter_maps,
                                                            voices_map=sample_voices_map,
                                                            output_dir=str(temp_dir),
                                                        )

        # Verify the result is correct
        assert isinstance(result, tuple)
        assert result[1] == 2
        assert "Generated 2 chapters" in result[0]

        # Verify TTS was called for each line
        assert mock_tts.call_count > 0

    def test_pipeline_with_empty_chapters(self, temp_dir):
        """Test pipeline with empty chapters list."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with patch("audiobook_generator.audiobook_generator.setup_validation_model") as mock_validation:
            mock_validation.return_value = MagicMock()
            with patch("audiobook_generator.audiobook_generator.get_validation_client"):
                with patch("audiobook_generator.audiobook_generator.VoiceMapper"):
                    with patch("audiobook_generator.audiobook_generator.generate_tts_for_line"):
                        with patch("audiobook_generator.audiobook_generator.get_non_silent_audio_from_wavs"):
                            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                                with patch("audiobook_generator.audiobook_generator.glob.glob", return_value=[]):
                                    with patch("audiobook_generator.audiobook_generator.ProgressHandler"):
                                        with patch("audiobook_generator.audiobook_generator.os.makedirs"):
                                            with patch("audiobook_generator.audiobook_generator.os.path.join", side_effect=os.path.join):
                                                with patch("audiobook_generator.audiobook_generator.gc.collect"):
                                                    result = generate_audiobook_from_chapters(
                                                        chapters=[],
                                                        chapter_maps={},
                                                        voices_map={},
                                                        output_dir=str(temp_dir),
                                                    )

        assert result == ("Generated 0 chapters successfully.", 0)

    def test_pipeline_with_max_chapters(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Test pipeline with max_chapters limit."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with patch("audiobook_generator.audiobook_generator.setup_validation_model") as mock_validation:
            mock_validation.return_value = MagicMock()
            with patch("audiobook_generator.audiobook_generator.get_validation_client"):
                with patch("audiobook_generator.audiobook_generator.VoiceMapper") as mock_mapper:
                    mock_mapper.return_value = MagicMock()
                    mock_mapper.return_value.add_voice_path.return_value = None
                    with patch("audiobook_generator.audiobook_generator.generate_tts_for_line") as mock_tts:
                        mock_tts.return_value = (True, 0.95)
                        with patch("audiobook_generator.audiobook_generator.get_non_silent_audio_from_wavs") as mock_wavs:
                            mock_audio = MagicMock()
                            mock_wavs.return_value = mock_audio
                            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                                with patch("audiobook_generator.audiobook_generator.glob.glob", return_value=["/tmp/chapter_00.0002.wav"]):
                                    with patch("audiobook_generator.audiobook_generator.ProgressHandler"):
                                        with patch("audiobook_generator.audiobook_generator.os.makedirs"):
                                            with patch("audiobook_generator.audiobook_generator.os.unlink"):
                                                with patch("audiobook_generator.audiobook_generator.os.path.join", side_effect=os.path.join):
                                                    with patch("audiobook_generator.audiobook_generator.gc.collect"):
                                                        result = generate_audiobook_from_chapters(
                                                            chapters=sample_chapters,
                                                            chapter_maps=sample_chapter_maps,
                                                            voices_map=sample_voices_map,
                                                            output_dir=str(temp_dir),
                                                            max_chapters=1,
                                                        )

        # Only 1 chapter should be processed
        assert result[1] == 1

    def test_pipeline_skip_existing(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Test pipeline skips chapters that already have MP3 files."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with patch("audiobook_generator.audiobook_generator.setup_validation_model") as mock_validation:
            mock_validation.return_value = MagicMock()
            with patch("audiobook_generator.audiobook_generator.get_validation_client"):
                with patch("audiobook_generator.audiobook_generator.VoiceMapper"):
                    with patch("audiobook_generator.audiobook_generator.generate_tts_for_line"):
                        with patch("audiobook_generator.audiobook_generator.get_non_silent_audio_from_wavs"):
                            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=True):
                                with patch("audiobook_generator.audiobook_generator.glob.glob", return_value=[]):
                                    with patch("audiobook_generator.audiobook_generator.ProgressHandler"):
                                        with patch("audiobook_generator.audiobook_generator.os.makedirs"):
                                            with patch("audiobook_generator.audiobook_generator.os.path.join", side_effect=os.path.join):
                                                with patch("audiobook_generator.audiobook_generator.gc.collect"):
                                                    result = generate_audiobook_from_chapters(
                                                        chapters=sample_chapters,
                                                        chapter_maps=sample_chapter_maps,
                                                        voices_map=sample_voices_map,
                                                        output_dir=str(temp_dir),
                                                    )

        # Both chapters should be skipped (MP3 exists)
        assert result == ("Generated 2 chapters successfully.", 2)

    def test_pipeline_debug_tts_mode(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Test pipeline in debug_tts mode doesn't generate audio."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with patch("audiobook_generator.audiobook_generator.setup_validation_model") as mock_validation:
            mock_validation.return_value = MagicMock()
            with patch("audiobook_generator.audiobook_generator.get_validation_client"):
                with patch("audiobook_generator.audiobook_generator.VoiceMapper"):
                    with patch("audiobook_generator.audiobook_generator.generate_tts_for_line") as mock_tts:
                        with patch("audiobook_generator.audiobook_generator.get_non_silent_audio_from_wavs"):
                            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                                with patch("audiobook_generator.audiobook_generator.glob.glob", return_value=[]):
                                    with patch("audiobook_generator.audiobook_generator.ProgressHandler"):
                                        with patch("audiobook_generator.audiobook_generator.os.makedirs"):
                                            with patch("audiobook_generator.audiobook_generator.os.path.join", side_effect=os.path.join):
                                                with patch("audiobook_generator.audiobook_generator.gc.collect"):
                                                    result = generate_audiobook_from_chapters(
                                                        chapters=sample_chapters,
                                                        chapter_maps=sample_chapter_maps,
                                                        voices_map=sample_voices_map,
                                                        output_dir=str(temp_dir),
                                                        debug_tts=True,
                                                    )

        # TTS should not be called in debug mode
        mock_tts.assert_not_called()


class TestPipelineStageIntegration:
    """Tests for pipeline stage interactions."""

    def test_chapter_parsing_to_tts(self):
        """Test that chapter parsing produces correct ChapterObj objects."""
        text = '"Hello," said Jane. "Goodbye," Elizabeth replied.'
        chapters = get_chapter_objs(text)

        assert len(chapters) == 4
        assert chapters[0].has_quotes is False
        assert chapters[1].has_quotes is True
        assert chapters[2].has_quotes is False
        assert chapters[3].has_quotes is True

    def test_voice_mapping_integration(self, temp_dir):
        """Test that VoiceMapper correctly maps characters to voices."""
        from audiobook_generator.voice_mapper import VoiceMapper

        with patch("audiobook_generator.voice_mapper.get_engine"):
            mapper = VoiceMapper(output_dir=str(temp_dir), device="cpu")
            mapper.add_voice_path("narrator", "/tmp/narrator.wav")
            mapper.add_voice_path("jane", "/tmp/jane.wav")

            assert mapper.get_voice_path("narrator") == "/tmp/narrator.wav"
            assert mapper.get_voice_path("jane") == "/tmp/jane.wav"

    def test_mock_tts_engine_generation(self, temp_dir):
        """Test that MockTTSEngine generates audio files."""
        engine = MockTTSEngine()
        output_path = str(temp_dir / "test_output.wav")

        success = engine.generate_line(
            text="Hello world",
            voice_path="/tmp/voice.wav",
            output_path=output_path,
            device="cpu",
            validation_model=None,
        )

        assert success is True
        assert os.path.exists(output_path)
