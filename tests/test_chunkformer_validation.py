"""Tests for ChunkFormer voice validation."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audiobook_generator.generate_voice_samples import _validate_with_chunkformer


@pytest.fixture
def mock_chunkformer_model():
    """Mock ChunkFormer model for testing."""
    mock = MagicMock()
    return mock


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing."""
    audio_path = temp_dir / "test_voice.wav"
    # Create a minimal WAV file
    import wave
    with wave.open(str(audio_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000)  # 1 second of silence
    return str(audio_path)


class TestValidateWithChunkFormer:
    """Tests for _validate_with_chunkformer function."""

    def test_gender_match_passes(self, sample_audio_file, mock_chunkformer_model):
        """Test that gender match results in valid output."""
        mock_chunkformer_model.classify_audio.return_value = {
            "gender": {"label": "female", "prob": 0.9},
            "emotion": {"label": "neutral", "prob": 0.8},
            "age": {"label": "young", "prob": 0.7},
            "dialect": {"label": "american", "prob": 0.6},
        }

        is_valid, msg = _validate_with_chunkformer(
            sample_audio_file,
            "A gentle, refined female voice.",
            mock_chunkformer_model,
            verbose=False
        )

        assert is_valid is True
        result = json.loads(msg)
        assert result["gender_ok"] is True
        assert result["classification"]["gender"] == "female"

    def test_gender_mismatch_fails(self, sample_audio_file, mock_chunkformer_model):
        """Test that gender mismatch results in invalid output."""
        mock_chunkformer_model.classify_audio.return_value = {
            "gender": {"label": "male", "prob": 0.9},
            "emotion": {"label": "neutral", "prob": 0.8},
            "age": {"label": "young", "prob": 0.7},
            "dialect": {"label": "american", "prob": 0.6},
        }

        is_valid, msg = _validate_with_chunkformer(
            sample_audio_file,
            "A gentle, refined female voice.",
            mock_chunkformer_model,
            verbose=False
        )

        assert is_valid is False
        result = json.loads(msg)
        assert result["gender_ok"] is False
        assert "gender mismatch" in result["reasons"][0]

    def test_unknown_gender_passes(self, sample_audio_file, mock_chunkformer_model):
        """Test that unknown expected gender results in valid output."""
        mock_chunkformer_model.classify_audio.return_value = {
            "gender": {"label": "male", "prob": 0.9},
            "emotion": {"label": "neutral", "prob": 0.8},
            "age": {"label": "young", "prob": 0.7},
            "dialect": {"label": "american", "prob": 0.6},
        }

        is_valid, msg = _validate_with_chunkformer(
            sample_audio_file,
            "A calm, clear voice.",
            mock_chunkformer_model,
            verbose=False
        )

        assert is_valid is True

    def test_male_expected_female_actual_fails(self, sample_audio_file, mock_chunkformer_model):
        """Test that male expected but female detected fails."""
        mock_chunkformer_model.classify_audio.return_value = {
            "gender": {"label": "female", "prob": 0.9},
            "emotion": {"label": "neutral", "prob": 0.8},
            "age": {"label": "young", "prob": 0.7},
            "dialect": {"label": "american", "prob": 0.6},
        }

        is_valid, msg = _validate_with_chunkformer(
            sample_audio_file,
            "A strong, authoritative male voice.",
            mock_chunkformer_model,
            verbose=False
        )

        assert is_valid is False
        result = json.loads(msg)
        assert result["gender_ok"] is False

    def test_handles_exception(self, sample_audio_file, mock_chunkformer_model):
        """Test that exceptions are handled gracefully."""
        mock_chunkformer_model.classify_audio.side_effect = Exception("Model error")

        is_valid, msg = _validate_with_chunkformer(
            sample_audio_file,
            "A gentle voice.",
            mock_chunkformer_model,
            verbose=False
        )

        assert is_valid is True  # Graceful fallback

    def test_verbose_output(self, sample_audio_file, mock_chunkformer_model):
        """Test that verbose flag enables debug output."""
        mock_chunkformer_model.classify_audio.return_value = {
            "gender": {"label": "female", "prob": 0.9},
            "emotion": {"label": "neutral", "prob": 0.8},
            "age": {"label": "young", "prob": 0.7},
            "dialect": {"label": "american", "prob": 0.6},
        }

        with patch("builtins.print") as mock_print:
            _validate_with_chunkformer(
                sample_audio_file,
                "A gentle, refined female voice.",
                mock_chunkformer_model,
                verbose=True
            )
            mock_print.assert_called()

    def test_log_file_created(self, sample_audio_file, mock_chunkformer_model, temp_dir):
        """Test that validation results are logged to JSON file."""
        mock_chunkformer_model.classify_audio.return_value = {
            "gender": {"label": "female", "prob": 0.9},
            "emotion": {"label": "neutral", "prob": 0.8},
            "age": {"label": "young", "prob": 0.7},
            "dialect": {"label": "american", "prob": 0.6},
        }

        # Create a directory structure similar to what the function expects
        voice_dir = temp_dir / "voices"
        voice_dir.mkdir()
        voice_path = voice_dir / "test_voice.wav"
        import wave
        with wave.open(str(voice_path), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)

        _validate_with_chunkformer(
            str(voice_path),
            "A gentle, refined female voice.",
            mock_chunkformer_model,
            verbose=False
        )

        log_path = voice_dir / ".chunkformer_validation.json"
        assert log_path.exists()

        with open(log_path, "r") as f:
            lines = f.readlines()
            assert len(lines) > 0
            data = json.loads(lines[0])
            assert "gender_ok" in data
            assert "classification" in data
