"""Tests for Whisper validation model integration, covering both openai-whisper and faster_whisper return formats."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import wave
import struct

from audiobook_generator.audiobook_generator import (
    _validate_and_clip_audio,
    TTSConfig,
)
from audiobook_generator.pipeline import collect_transcription_segments


def _create_test_wav(path, duration_ms=1000):
    """Create a minimal valid WAV file for testing."""
    sample_rate = 24000
    num_samples = int(sample_rate * duration_ms / 1000)
    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for _ in range(num_samples):
            wf.writeframes(struct.pack('<h', 0))


def _make_segment(text, start, end, use_dict=False):
    """Create a mock segment with word-level data."""
    words = text.split()
    word_segments = []
    word_start = start
    word_duration = (end - start) / max(len(words), 1)
    for w in words:
        if use_dict:
            word_segments.append({"word": w, "start": word_start, "end": word_start + word_duration})
        else:
            ws = MagicMock()
            ws.word = w
            ws.start = word_start
            ws.end = word_start + word_duration
            word_segments.append(ws)
        word_start += word_duration

    if use_dict:
        return {"words": word_segments}
    else:
        seg = MagicMock()
        seg.words = word_segments
        return seg


class TestValidateAndClipTupleReturn:
    """Test that _validate_and_clip_audio handles faster_whisper tuple return format."""

    def test_tuple_return_extracts_generator(self):
        """faster_whisper returns (generator, info) tuple - should extract result[0]."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        _create_test_wav(wav_path)

        mock_segment = _make_segment("hello world", 0.0, 1.0, use_dict=False)

        # faster_whisper returns (generator, TranscriptionInfo) tuple
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_segment]), MagicMock())

        tts_config = TTSConfig(
            device="cpu",
            tts_engine="test",
            output_dir="/tmp",
            short_text_postfix="and also with you",
            validation_model=mock_model,
            verbose=False,
        )

        ratio, last_token = _validate_and_clip_audio("hello world. and also with you", wav_path, tts_config)

        # Ratio should not be 0.0 (which means no segments were extracted)
        # Negative ratio is expected since mock audio doesn't contain postfix
        assert ratio != 0.0, f"Expected non-zero ratio for tuple return, got {ratio}"
        mock_model.transcribe.assert_called_once()

    def test_dict_return_uses_segments_key(self):
        """openai-whisper returns dict - should use result.get('segments', [])."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        _create_test_wav(wav_path)

        mock_segment = _make_segment("hello world", 0.0, 1.0, use_dict=True)

        # openai-whisper returns dict
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [mock_segment],
            "text": "hello world",
            "language": "en",
        }

        tts_config = TTSConfig(
            device="cpu",
            tts_engine="test",
            output_dir="/tmp",
            short_text_postfix="and also with you",
            validation_model=mock_model,
            verbose=False,
        )

        ratio, last_token = _validate_and_clip_audio("hello world. and also with you", wav_path, tts_config)

        assert ratio != 0.0, f"Expected non-zero ratio for dict return, got {ratio}"
        mock_model.transcribe.assert_called_once()

    def test_tuple_return_does_not_iterate_info(self):
        """Tuple return should NOT iterate over the TranscriptionInfo object."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        _create_test_wav(wav_path)

        mock_segment = _make_segment("hello world", 0.0, 1.0, use_dict=False)

        mock_info = MagicMock()
        # If the code incorrectly iterates the tuple, it would try to process this as segments
        mock_info.__iter__ = MagicMock(side_effect=RuntimeError("Should not iterate TranscriptionInfo"))

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        tts_config = TTSConfig(
            device="cpu",
            tts_engine="test",
            output_dir="/tmp",
            short_text_postfix="and also with you",
            validation_model=mock_model,
            verbose=False,
        )

        # Should not raise - the info object should never be iterated
        ratio, last_token = _validate_and_clip_audio("hello world. and also with you", wav_path, tts_config)
        assert ratio != 0.0, f"Expected non-zero ratio, got {ratio}"


class TestValidateAndClipWithLock:
    """Test that validation works correctly with whisper_lock (threaded path)."""

    def test_tuple_return_with_lock(self):
        """Tuple return should work with whisper_lock path."""
        import threading

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        _create_test_wav(wav_path)

        mock_segment = _make_segment("hello world", 0.0, 1.0, use_dict=False)

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_segment]), MagicMock())

        tts_config = TTSConfig(
            device="cpu",
            tts_engine="test",
            output_dir="/tmp",
            short_text_postfix="and also with you",
            validation_model=mock_model,
            whisper_lock=threading.Lock(),
            verbose=False,
        )

        ratio, last_token = _validate_and_clip_audio("hello world. and also with you", wav_path, tts_config)
        assert ratio != 0.0, f"Expected non-zero ratio with lock, got {ratio}"


class TestValidateAndClipWithPool:
    """Test that validation works correctly with whisper_pool (concurrent path)."""

    def test_tuple_return_with_pool(self):
        """Tuple return should work with whisper_pool path."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        _create_test_wav(wav_path)

        mock_segment = _make_segment("hello world", 0.0, 1.0, use_dict=False)

        mock_pool = MagicMock()
        mock_pool.transcribe.return_value = (iter([mock_segment]), MagicMock())

        tts_config = TTSConfig(
            device="cpu",
            tts_engine="test",
            output_dir="/tmp",
            short_text_postfix="and also with you",
            validation_model=mock_pool,
            whisper_pool=mock_pool,
            verbose=False,
        )

        ratio, last_token = _validate_and_clip_audio("hello world. and also with you", wav_path, tts_config)
        assert ratio != 0.0, f"Expected non-zero ratio with pool, got {ratio}"
        mock_pool.transcribe.assert_called_once()


class TestValidateAndClipNoModel:
    """Test that validation gracefully skips when no model is available."""

    def test_no_validation_model_returns_zero_ratio(self):
        """When validation_model is None, should return ratio=0.0 without error."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        _create_test_wav(wav_path)

        tts_config = TTSConfig(
            device="cpu",
            tts_engine="test",
            output_dir="/tmp",
            short_text_postfix="and also with you",
            validation_model=None,
            verbose=False,
        )

        ratio, last_token = _validate_and_clip_audio("hello world. and also with you", wav_path, tts_config)
        assert ratio == 0.0
        assert last_token is None


class TestCollectTranscriptionSegments:
    """Test collect_transcription_segments handles both dict and object segments."""

    def test_object_based_segments(self):
        """Handle faster_whisper object-based segments (with .words attribute)."""
        mock_segment = _make_segment("hello world", 0.0, 1.0, use_dict=False)

        segments, starts, ends = collect_transcription_segments([mock_segment])
        assert len(segments) == 2
        assert segments[0] == "hello"
        assert segments[1] == "world"
        assert starts[0] == 0.0
        assert ends[1] == 1.0

    def test_dict_based_segments(self):
        """Handle openai-whisper dict-based segments."""
        dict_segment = _make_segment("hello world", 0.0, 1.0, use_dict=True)

        segments, starts, ends = collect_transcription_segments([dict_segment])
        assert len(segments) == 2
        assert segments[0] == "hello"
        assert segments[1] == "world"
        assert starts[0] == 0.0
        assert ends[1] == 1.0

    def test_empty_segments(self):
        """Handle empty segment list gracefully."""
        segments, starts, ends = collect_transcription_segments([])
        assert segments == []
        assert starts == []
        assert ends == []

    def test_mixed_segments(self):
        """Handle mixed dict and object segments in same list."""
        obj_segment = _make_segment("hello", 0.0, 0.5, use_dict=False)
        dict_segment = _make_segment("world", 0.5, 1.0, use_dict=True)

        segments, starts, ends = collect_transcription_segments([obj_segment, dict_segment])
        assert len(segments) == 2
        assert segments[0] == "hello"
        assert segments[1] == "world"
