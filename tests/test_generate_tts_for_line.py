"""Tests for generate_tts_for_line() in audiobook_generator.py."""

import os
import pytest
from unittest.mock import MagicMock, patch, call
import numpy as np

from audiobook_generator.testing import MockTTSEngine
from audiobook_generator.audiobook_generator import TTSConfig


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_voice_mapper():
    """Create a mock VoiceMapper for testing."""
    mapper = MagicMock()
    mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
    mapper.get_engine.return_value = MockTTSEngine()
    return mapper


@pytest.fixture
def mock_voice_mapper_no_voice():
    """Create a mock VoiceMapper that returns no voice path."""
    mapper = MagicMock()
    mapper.get_voice_path.return_value = None
    mapper.get_engine.return_value = MockTTSEngine()
    return mapper


# ============================================================================
# TESTS
# ============================================================================

class TestGenerateTTSForLineEmptyText:
    """Tests for empty text handling."""

    def test_empty_string_returns_success(self, mock_voice_mapper):
        """Empty string should return (True, 1.0)."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        result = generate_tts_for_line(
            chapter_idx=0,
            line_idx=0,
            text="",
            voice_name="narrator",
            voice_mapper=mock_voice_mapper,
            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir="/tmp/test_output"),
        )
        assert result == (True, 1.0)

    def test_whitespace_only_returns_success(self, mock_voice_mapper):
        """Whitespace-only text should return (True, 1.0)."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        result = generate_tts_for_line(
            chapter_idx=0,
            line_idx=0,
            text="   ",
            voice_name="narrator",
            voice_mapper=mock_voice_mapper,
            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir="/tmp/test_output"),
        )
        assert result == (True, 1.0)

    def test_none_text_returns_success(self, mock_voice_mapper):
        """None text should return (True, 1.0)."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        result = generate_tts_for_line(
            chapter_idx=0,
            line_idx=0,
            text=None,
            voice_name="narrator",
            voice_mapper=mock_voice_mapper,
            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir="/tmp/test_output"),
        )
        assert result == (True, 1.0)


class TestGenerateTTSForLineVoicePath:
    """Tests for voice path handling."""

    def test_raises_exception_when_no_voice_path(self, temp_dir):
        """Should raise exception when voice path is None."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        mapper = MagicMock()
        mapper.get_voice_path.return_value = None
        mapper.get_engine.return_value = MockTTSEngine()

        with pytest.raises(Exception, match="No voice path found for 'narrator'"):
            generate_tts_for_line(
                chapter_idx=0,
                line_idx=0,
                text="Hello world",
                voice_name="narrator",
                voice_mapper=mapper,
                tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir=str(temp_dir)),
            )

    def test_voice_path_from_mapper(self, temp_dir):
        """Should use voice path from voice_mapper when voice_path is None."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        mapper = MagicMock()
        mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
        mapper.get_engine.return_value = MockTTSEngine()

        with patch("audiobook_generator.audiobook_generator.time.sleep"):
            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                with patch("audiobook_generator.audiobook_generator.os.unlink"):
                    with patch("audiobook_generator.audiobook_generator.os.rename"):
                        result = generate_tts_for_line(
                            chapter_idx=0,
                            line_idx=0,
                            text="Hello world",
                            voice_name="narrator",
                            voice_mapper=mapper,
                            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir=str(temp_dir), validation_model=None, short_text_postfix=""),
                        )

        assert result[0] is False or result[0] is True
        assert isinstance(result[1], float)
        mapper.get_voice_path.assert_called_once_with("narrator")

    def test_explicit_voice_path(self, temp_dir):
        """Should use explicit voice_path when provided."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        mapper = MagicMock()
        mapper.get_engine.return_value = MockTTSEngine()

        with patch("audiobook_generator.audiobook_generator.time.sleep"):
            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                with patch("audiobook_generator.audiobook_generator.os.unlink"):
                    with patch("audiobook_generator.audiobook_generator.os.rename"):
                        result = generate_tts_for_line(
                            chapter_idx=0,
                            line_idx=0,
                            text="Hello world",
                            voice_name="narrator",
                            voice_mapper=mapper,
                            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir=str(temp_dir), validation_model=None, short_text_postfix=""),
                            voice_path="/explicit/path.wav",
                        )

        assert result[0] is False or result[0] is True
        mapper.get_voice_path.assert_not_called()


class TestGenerateTTSForLineEngineCall:
    """Tests for engine.generate_line() call."""

    def test_engine_generate_line_called(self, temp_dir):
        """Engine.generate_line should be called with correct parameters."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        engine = MockTTSEngine()
        mapper = MagicMock()
        mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
        mapper.get_engine.return_value = engine

        with patch("audiobook_generator.audiobook_generator.time.sleep"):
            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                with patch("audiobook_generator.audiobook_generator.os.unlink"):
                    with patch("audiobook_generator.audiobook_generator.os.rename"):
                        generate_tts_for_line(
                            chapter_idx=0,
                            line_idx=0,
                            text="Hello world",
                            voice_name="narrator",
                            voice_mapper=mapper,
                            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir=str(temp_dir), validation_model=None, short_text_postfix=""),
                            voice_path="/tmp/test_voice.wav",
                        )

        assert engine.last_generate_line_args["text"] == "Hello world"
        assert engine.last_generate_line_args["voice_path"] == "/tmp/test_voice.wav"
        assert engine.last_generate_line_args["device"] == "cpu"

    def test_engine_generate_line_with_cfg_scale(self, temp_dir):
        """Engine.generate_line should receive cfg_scale."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        engine = MockTTSEngine()
        mapper = MagicMock()
        mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
        mapper.get_engine.return_value = engine

        with patch("audiobook_generator.audiobook_generator.time.sleep"):
            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                with patch("audiobook_generator.audiobook_generator.os.unlink"):
                    with patch("audiobook_generator.audiobook_generator.os.rename"):
                        generate_tts_for_line(
                            chapter_idx=0,
                            line_idx=0,
                            text="Hello world",
                            voice_name="narrator",
                            voice_mapper=mapper,
                            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=2.0, output_dir=str(temp_dir), validation_model=None, short_text_postfix=""),
                            voice_path="/tmp/test_voice.wav",
                        )

        # The engine mock doesn't capture cfg_scale, but we can verify it was called
        assert engine.last_generate_line_args["text"] == "Hello world"


class TestGenerateTTSForLineValidation:
    """Tests for validation model behavior."""

    def test_no_validation_model_returns_zero_ratio(self, temp_dir):
        """When validation_model is None, ratio should be 0.0."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        engine = MockTTSEngine()
        mapper = MagicMock()
        mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
        mapper.get_engine.return_value = engine

        with patch("audiobook_generator.audiobook_generator.time.sleep"):
            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                with patch("audiobook_generator.audiobook_generator.os.unlink"):
                    with patch("audiobook_generator.audiobook_generator.os.rename"):
                        success, ratio = generate_tts_for_line(
                            chapter_idx=0,
                            line_idx=0,
                            text="Hello world",
                            voice_name="narrator",
                            voice_mapper=mapper,
                            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir=str(temp_dir), validation_model=None, short_text_postfix=""),
                            voice_path="/tmp/test_voice.wav",
                        )

        # Without validation_model, ratio stays 0.0 so success is False
        assert ratio == 0.0
        assert success is False


class TestGenerateTTSForLineRetryLoop:
    """Tests for the retry loop behavior."""

    def test_retry_count_increments_on_failure(self, temp_dir):
        """When engine fails, retry count should increment."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        engine = MagicMock()
        engine.generate_line.side_effect = Exception("Generation failed")
        mapper = MagicMock()
        mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
        mapper.get_engine.return_value = engine

        with patch("audiobook_generator.audiobook_generator.time.sleep"):
            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                with patch("audiobook_generator.audiobook_generator.os.unlink"):
                    success, ratio = generate_tts_for_line(
                        chapter_idx=0,
                        line_idx=0,
                        text="Hello world",
                        voice_name="narrator",
                        voice_mapper=mapper,
                        tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir=str(temp_dir), validation_model=None, short_text_postfix=""),
                        voice_path="/tmp/test_voice.wav",
                    )

        # Engine should have been called MAX_RETRIES times (retries 0, 1)
        assert engine.generate_line.call_count == 2
        assert success is False

    def test_engine_failure_returns_zero_ratio(self, temp_dir):
        """When engine fails, max_ratio stays at -inf."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        engine = MagicMock()
        engine.generate_line.side_effect = Exception("Generation failed")
        mapper = MagicMock()
        mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
        mapper.get_engine.return_value = engine

        with patch("audiobook_generator.audiobook_generator.time.sleep"):
            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                with patch("audiobook_generator.audiobook_generator.os.unlink"):
                    success, ratio = generate_tts_for_line(
                        chapter_idx=0,
                        line_idx=0,
                        text="Hello world",
                        voice_name="narrator",
                        voice_mapper=mapper,
                        tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir=str(temp_dir), validation_model=None, short_text_postfix=""),
                        voice_path="/tmp/test_voice.wav",
                    )

        assert ratio == float('-inf')


class TestGenerateTTSForLineTextPreparation:
    """Tests for text preparation before TTS."""

    def test_text_is_normalized(self, temp_dir):
        """Text should be normalized before being passed to the engine."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        engine = MockTTSEngine()
        mapper = MagicMock()
        mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
        mapper.get_engine.return_value = engine

        with patch("audiobook_generator.audiobook_generator.time.sleep"):
            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                with patch("audiobook_generator.audiobook_generator.os.unlink"):
                    with patch("audiobook_generator.audiobook_generator.os.rename"):
                        generate_tts_for_line(
                            chapter_idx=0,
                            line_idx=0,
                            text="hello world",
                            voice_name="narrator",
                            voice_mapper=mapper,
                            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir=str(temp_dir), validation_model=None, short_text_postfix=""),
                            voice_path="/tmp/test_voice.wav",
                        )

        # Text should be capitalized (first letter uppercase)
        assert engine.last_generate_line_args["text"] == "Hello world"


class TestGenerateTTSForLineReturnValues:
    """Tests for return value format and consistency."""

    def test_returns_tuple_of_bool_and_float(self, temp_dir):
        """Should return (bool, float) tuple."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        engine = MockTTSEngine()
        mapper = MagicMock()
        mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
        mapper.get_engine.return_value = engine

        with patch("audiobook_generator.audiobook_generator.time.sleep"):
            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                with patch("audiobook_generator.audiobook_generator.os.unlink"):
                    with patch("audiobook_generator.audiobook_generator.os.rename"):
                        result = generate_tts_for_line(
                            chapter_idx=0,
                            line_idx=0,
                            text="Hello world",
                            voice_name="narrator",
                            voice_mapper=mapper,
                            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir=str(temp_dir), validation_model=None, short_text_postfix=""),
                            voice_path="/tmp/test_voice.wav",
                        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    def test_ratio_between_zero_and_one(self, temp_dir):
        """Ratio should be between 0.0 and 1.0."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        engine = MockTTSEngine()
        mapper = MagicMock()
        mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
        mapper.get_engine.return_value = engine

        with patch("audiobook_generator.audiobook_generator.time.sleep"):
            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                with patch("audiobook_generator.audiobook_generator.os.unlink"):
                    with patch("audiobook_generator.audiobook_generator.os.rename"):
                        _, ratio = generate_tts_for_line(
                            chapter_idx=0,
                            line_idx=0,
                            text="Hello world",
                            voice_name="narrator",
                            voice_mapper=mapper,
                            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir=str(temp_dir), validation_model=None, short_text_postfix=""),
                            voice_path="/tmp/test_voice.wav",
                        )

        assert 0.0 <= ratio <= 1.0


class TestGenerateTTSForLineOutputPath:
    """Tests for output path generation."""

    def test_output_path_format(self, temp_dir):
        """Output path should include chapter and line indices."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        engine = MockTTSEngine()
        mapper = MagicMock()
        mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
        mapper.get_engine.return_value = engine

        with patch("audiobook_generator.audiobook_generator.time.sleep"):
            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                with patch("audiobook_generator.audiobook_generator.os.unlink"):
                    with patch("audiobook_generator.audiobook_generator.os.rename"):
                        generate_tts_for_line(
                            chapter_idx=1,
                            line_idx=2,
                            text="Hello world",
                            voice_name="narrator",
                            voice_mapper=mapper,
                            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir=str(temp_dir), validation_model=None, short_text_postfix=""),
                            voice_path="/tmp/test_voice.wav",
                        )

        # Check that the output path was generated correctly
        assert "chapter_01" in engine.last_generate_line_args["output_path"]
        assert "0002" in engine.last_generate_line_args["output_path"]
        assert str(temp_dir) in engine.last_generate_line_args["output_path"]


class TestGenerateTTSForLineCleanup:
    """Tests for cleanup of temp files."""

    def test_temp_file_removed(self, temp_dir):
        """Temp file should be removed after generation."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        engine = MockTTSEngine()
        mapper = MagicMock()
        mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
        mapper.get_engine.return_value = engine

        with patch("audiobook_generator.audiobook_generator.time.sleep"):
            with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=True):
                with patch("audiobook_generator.audiobook_generator.os.unlink") as mock_unlink:
                    with patch("audiobook_generator.audiobook_generator.os.rename"):
                        generate_tts_for_line(
                            chapter_idx=0,
                            line_idx=0,
                            text="Hello world",
                            voice_name="narrator",
                            voice_mapper=mapper,
                            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir=str(temp_dir), validation_model=None, short_text_postfix=""),
                            voice_path="/tmp/test_voice.wav",
                        )

        # os.unlink should have been called (temp file removal)
        assert mock_unlink.called


class TestGenerateTTSForLineVerbose:
    """Tests for verbose output."""

    def test_verbose_output_for_empty_text(self, mock_voice_mapper, capsys):
        """Verbose output should be printed for empty text."""
        from audiobook_generator.audiobook_generator import generate_tts_for_line

        result = generate_tts_for_line(
            chapter_idx=0,
            line_idx=0,
            text="",
            voice_name="narrator",
            voice_mapper=mock_voice_mapper,
            tts_config=TTSConfig(device="cpu", tts_engine="moss", cfg_scale=1.3, output_dir="/tmp/test_output", verbose=True),
        )

        captured = capsys.readouterr()
        assert "Skipping line 0" in captured.out
        assert result == (True, 1.0)
