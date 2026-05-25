"""Tests for generate_audiobook_from_chapters() in audiobook_generator.py."""

import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from audiobook_generator.parse_chapter import ChapterObj


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_chapters():
    """Create sample chapter objects for testing."""
    chapter1 = [
        ChapterObj(False, "Narrator text", 1),
        ChapterObj(True, '"Hello there," said Jane.', 2),
        ChapterObj(False, "Narrator continues.", 3),
    ]

    chapter2 = [
        ChapterObj(True, '"Good morning," Elizabeth replied.', 1),
        ChapterObj(False, "The room was silent.", 2),
    ]

    return [chapter1, chapter2]


@pytest.fixture
def sample_chapter_maps():
    """Create sample chapter maps."""
    return {
        0: ({"1": "narrator", "2": "jane"}, {"2": 2}),
        1: ({"1": "elizabeth"}, {"1": 1}),
    }


@pytest.fixture
def sample_voices_map():
    """Create sample voices map."""
    return {
        "narrator": "narrator.wav",
        "jane": "jane.wav",
        "elizabeth": "elizabeth.wav",
    }


def _patch_all(temp_dir):
    """Return a context manager that patches all dependencies."""
    from contextlib import contextmanager

    @contextmanager
    def _patches():
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
                                                        yield mock_tts
    return _patches()


# ============================================================================
# TESTS
# ============================================================================

class TestGenerateAudiobookFromChaptersBasic:
    """Tests for basic functionality."""

    def test_returns_tuple(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Should return (status_message, chapters_processed) tuple."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            result = generate_audiobook_from_chapters(
                chapters=sample_chapters,
                chapter_maps=sample_chapter_maps,
                voices_map=sample_voices_map,
                output_dir=str(temp_dir),
            )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], int)

    def test_empty_chapters_returns_zero(self, temp_dir):
        """Empty chapters list should return (message, 0)."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            result = generate_audiobook_from_chapters(
                chapters=[],
                chapter_maps={},
                voices_map={},
                output_dir=str(temp_dir),
            )

        assert result == ("Generated 0 chapters successfully.", 0)

    def test_max_chapters_limit(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """max_chapters should limit the number of chapters processed."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            result = generate_audiobook_from_chapters(
                chapters=sample_chapters,
                chapter_maps=sample_chapter_maps,
                voices_map=sample_voices_map,
                output_dir=str(temp_dir),
                max_chapters=1,
            )

        assert result[1] == 1

    def test_skip_existing_mp3(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Should skip chapters that already have MP3 files."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=True):
                result = generate_audiobook_from_chapters(
                    chapters=sample_chapters,
                    chapter_maps=sample_chapter_maps,
                    voices_map=sample_voices_map,
                    output_dir=str(temp_dir),
                )

        # Both chapters should be skipped (MP3 already exists)
        assert result == ("Generated 2 chapters successfully.", 2)


class TestGenerateAudiobookFromChaptersVoiceResolution:
    """Tests for voice path resolution."""

    def test_no_voice_path_skips_lines(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Lines without voice files should be skipped with a warning."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            with patch("audiobook_generator.audiobook_generator.VoiceMapper") as mock_mapper:
                mock_mapper.return_value = MagicMock()
                mock_mapper.return_value.add_voice_path.return_value = None
                mock_mapper.return_value.get_voice_path.return_value = None
                result = generate_audiobook_from_chapters(
                    chapters=sample_chapters,
                    chapter_maps=sample_chapter_maps,
                    voices_map=sample_voices_map,
                    output_dir=str(temp_dir),
                )

        # Should complete without error, skipping lines without voice files
        assert isinstance(result, tuple)
        assert result[1] == 2
        mock_tts.assert_not_called()


class TestGenerateAudiobookFromChaptersDebugTTS:
    """Tests for debug TTS mode."""

    def test_debug_tts_does_not_generate(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """debug_tts should print instead of generate."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            result = generate_audiobook_from_chapters(
                chapters=sample_chapters,
                chapter_maps=sample_chapter_maps,
                voices_map=sample_voices_map,
                output_dir=str(temp_dir),
                debug_tts=True,
            )

        # When debug_tts is True, generate_tts_for_line should not be called
        mock_tts.assert_not_called()
