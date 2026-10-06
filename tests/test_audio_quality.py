"""Tests for audio quality improvements: Whisper transcription and clipping.

Pure-function scoring/normalization tests live in tests/test_pipeline.py and
tests/test_postfix_clipping.py; this file focuses on transcription, cropping,
and text-cleaning behavior.
"""

from unittest.mock import MagicMock

from audiobook_generator.pipeline import (
    score_strings_pop,
    calculate_clip_points,
    clean_text_for_tts,
)
from audiobook_generator.utils import (
    distill_string,
    transcribe_audio_for_ref_text,
)
from audiobook_generator.testing import write_silence_wav


def _write_tone_wav(path, sample_rate: int = 22050, duration: float = 2.0):
    """Write a non-silent WAV (440 Hz tone) so the crop has speech to keep."""
    import numpy as np
    import torch
    import torchaudio

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    torchaudio.save(str(path), torch.from_numpy(audio).unsqueeze(0), sample_rate)
    return path


class TestIntegrationScenarios:
    """Integration tests for audio quality improvements."""

    def test_full_scoring_pipeline(self):
        """Test full scoring pipeline with realistic data."""
        # Simulate a TTS output with extra text (postfix not clipped)
        input_text = "The quick brown fox jumps over the lazy dog"
        detected_text = "The quick brown fox jumps over the lazy dog and also with you"

        score, last_token = score_strings_pop(
            distill_string(input_text),
            distill_string(detected_text),
            postfix=distill_string("and also with you")
        )
        # Score should be high because all tokens match
        assert score > 0.7
        assert last_token == "dog"

    def test_clipping_with_extra_text(self):
        """Test clipping when audio contains extra text after main content."""
        # Simulate word-level timestamps from Whisper
        segments = [
            "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
            "and", "also", "with", "you"
        ]
        start_times = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6]
        end_times = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9]

        # Test clipping to remove postfix
        result = calculate_clip_points(
            segments, start_times, end_times,
            postfix_detect_token="and",
            last_valid_token="dog",
            verbose=True
        )

        assert result is not None
        start_clip, end_clip = result
        # Should clip before "and" (index 9) starts
        assert start_clip == 0  # No start clipping needed
        assert end_clip < 3900  # Should clip before end of audio
        assert end_clip > 2000  # Should include "dog" (2.7s)

    def test_scoring_with_misspelled_words(self):
        """Test scoring when Whisper transcribes words incorrectly."""
        input_text = "The quick brown fox jumps"
        # Simulate Whisper mishearing some words
        detected_text = "The quick brown fox jumps"

        score, last_token = score_strings_pop(
            distill_string(input_text),
            distill_string(detected_text),
            postfix=""
        )
        assert score >= 0.5
        assert last_token == "jumps"

    def test_scoring_with_missing_words(self):
        """Test scoring when some words are missing from transcription."""
        input_text = "The quick brown fox jumps over the lazy dog"
        # Simulate missing words
        detected_text = "The quick fox over the dog"

        score, last_token = score_strings_pop(
            distill_string(input_text),
            distill_string(detected_text),
            postfix=""
        )
        assert score > 0.0
        assert last_token == "dog"

    def test_scoring_with_extra_words(self):
        """Test scoring when extra words are in transcription."""
        input_text = "The quick brown fox"
        # Simulate extra words in transcription
        detected_text = "The quick brown fox jumps over the lazy dog"

        score, last_token = score_strings_pop(
            distill_string(input_text),
            distill_string(detected_text),
            postfix=""
        )
        assert score > 0.0
        assert last_token == "fox"

    def test_clipping_with_only_postfix(self):
        """Test clipping when only postfix is detected (TTS failure)."""
        segments = ["and", "also", "with", "you"]
        start_times = [0.0, 0.3, 0.6, 0.9]
        end_times = [0.3, 0.6, 0.9, 1.2]

        result = calculate_clip_points(
            segments, start_times, end_times,
            postfix_detect_token="and",
            last_valid_token=None,
            verbose=True
        )

        # When only postfix is detected and clip_start (0) >= clip_end (0),
        # the guard returns None since there's no valid content to keep
        assert result is None


class TestWhisperTranscriptionAccuracy:
    """Test Whisper transcription accuracy for reference text."""

    def test_distill_string_normalization(self):
        """Test that distill_string properly normalizes text for comparison."""
        test_cases = [
            ("Hello, World!", "hello world"),
            ("It's a test.", "it's a test"),  # Apostrophe preserved
            ("One two-three four", "one twothree four"),
            ("Multiple   spaces", "multiple spaces"),
        ]
        for input_str, expected in test_cases:
            assert distill_string(input_str) == expected

    def test_distill_string_small_numbers(self):
        """Standalone 0-99 integers normalize to spoken words so that ASR
        backends with different numeral conventions (whisper '1' vs parakeet
        'one') score identically. Larger integers (years) are left alone."""
        assert distill_string("Chapter 1") == "chapter one"
        assert distill_string("chapter one") == "chapter one"
        assert distill_string("50 men") == "fifty men"
        assert distill_string("21 guns") == "twenty one guns"
        assert distill_string("Copyright 1894") == "copyright 1894"
        assert distill_string("Chapter 100") == "chapter 100"

    def test_score_strings_pop_with_postfix(self):
        """Test scoring when postfix is present."""
        input_str = "hello world and also with you"
        detected_str = "hello world and also with you"

        score, last_token = score_strings_pop(
            distill_string(input_str),
            distill_string(detected_str),
            postfix=distill_string("and also with you")
        )
        assert score > 0.5
        # Last token is "you" because it's the last token in the input string
        assert last_token == "you"

    def test_score_strings_pop_without_postfix(self):
        """Test scoring when postfix is absent."""
        input_str = "hello world"
        detected_str = "hello world"

        score, last_token = score_strings_pop(
            distill_string(input_str),
            distill_string(detected_str),
            postfix=""
        )
        # Score is 0.5 because no postfix is present (penalty applied)
        assert score >= 0.5
        assert last_token == "world"


class TestTranscribeAudioForRefText:
    """Test transcribe_audio_for_ref_text function for getting reference text."""

    def test_returns_raw_text_not_distilled(self):
        """Test that raw text is returned, not distilled."""
        mock_model = MagicMock()
        mock_word = MagicMock()
        mock_word.word = " Hello "
        mock_word.start = 0.0
        mock_word.end = 0.5

        mock_segment = MagicMock()
        mock_segment.words = [mock_word]

        mock_model.transcribe.return_value = ([mock_segment], {})

        result = transcribe_audio_for_ref_text(mock_model, "/fake/path.wav", verbose=False)
        assert result == "Hello"
        # Should preserve capitalization and punctuation
        assert result[0].isupper()

    def test_returns_none_on_empty_transcription(self):
        """Test that None is returned when transcription is empty."""
        mock_model = MagicMock()
        mock_word = MagicMock()
        mock_word.word = "   "  # Only whitespace
        mock_word.start = 0.0
        mock_word.end = 0.5

        mock_segment = MagicMock()
        mock_segment.words = [mock_word]

        mock_model.transcribe.return_value = ([mock_segment], {})

        result = transcribe_audio_for_ref_text(mock_model, "/fake/path.wav", verbose=False)
        assert result is None

    def test_returns_none_on_exception(self):
        """Test that None is returned when transcription fails."""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Transcription failed")

        result = transcribe_audio_for_ref_text(mock_model, "/fake/path.wav", verbose=False)
        assert result is None

    def test_multiple_words_joined(self):
        """Test that multiple words are joined with spaces."""
        mock_model = MagicMock()
        words = []
        for i, w in enumerate(["Hello", "world", "test"]):
            mock_word = MagicMock()
            mock_word.word = f" {w} "
            mock_word.start = float(i)
            mock_word.end = float(i) + 0.5
            words.append(mock_word)

        mock_segment = MagicMock()
        mock_segment.words = words

        mock_model.transcribe.return_value = ([mock_segment], {})

        result = transcribe_audio_for_ref_text(mock_model, "/fake/path.wav", verbose=False)
        assert result == "Hello world test"

    def test_multiple_segments_combined(self):
        """Test that words from multiple segments are combined."""
        mock_model = MagicMock()
        words1 = []
        words2 = []

        for i, w in enumerate(["Hello", "world"]):
            mock_word = MagicMock()
            mock_word.word = f" {w} "
            mock_word.start = float(i)
            mock_word.end = float(i) + 0.5
            words1.append(mock_word)

        for i, w in enumerate(["test", "case"]):
            mock_word = MagicMock()
            mock_word.word = f" {w} "
            mock_word.start = float(i) + 2.0
            mock_word.end = float(i) + 2.5
            words2.append(mock_word)

        mock_segment1 = MagicMock()
        mock_segment1.words = words1
        mock_segment2 = MagicMock()
        mock_segment2.words = words2

        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], {})

        result = transcribe_audio_for_ref_text(mock_model, "/fake/path.wav", verbose=False)
        assert result == "Hello world test case"


class TestCropToRefText:
    """Test crop_to_ref_text function for voice sample cropping."""

    def test_keeps_full_speech_span(self):
        """Crop keeps the full spoken span (trims only leading/trailing silence)."""
        import tempfile
        import os

        from audiobook_generator.audio import crop_to_ref_text

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "test.wav")
            output_path = os.path.join(tmpdir, "cropped.wav")

            _write_tone_wav(audio_path, duration=3.0)

            transcribed_words = ["garbage", "noise", "after", "all", "these", "years"]
            start_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
            end_times = [0.4, 0.9, 1.4, 1.9, 2.4, 2.9]
            ref_words = ["after", "all", "these", "years", "its", "finally"]

            result = crop_to_ref_text(
                audio_path, output_path,
                ref_words, transcribed_words, start_times, end_times,
                verbose=True
            )

            assert result is True
            assert os.path.exists(output_path)

            # The whole tone (3s) is non-silent, so the crop must keep ~3s,
            # NOT crop down to the matched-reference-word span.
            import pydub
            seg = pydub.AudioSegment.from_wav(output_path)
            assert len(seg) > 2500  # ~3s, not ~1s

    def test_keeps_full_duration_no_aggressive_trim(self):
        """Crop preserves the full clip; it does not trim to a small ref span."""
        import tempfile
        import os

        from audiobook_generator.audio import crop_to_ref_text

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "test.wav")
            output_path = os.path.join(tmpdir, "cropped.wav")

            _write_tone_wav(audio_path, duration=2.0)

            transcribed_words = ["after", "all", "these", "years"]
            start_times = [1.0, 1.5, 2.0, 2.5]
            end_times = [1.4, 1.9, 2.4, 2.9]
            ref_words = ["after", "all", "these", "years"]

            result = crop_to_ref_text(
                audio_path, output_path,
                ref_words, transcribed_words, start_times, end_times,
                verbose=True
            )

            assert result is True
            assert os.path.exists(output_path)

            import pydub
            seg = pydub.AudioSegment.from_wav(output_path)
            # Full 2s tone preserved (not trimmed to ~0.8s ref span).
            assert len(seg) > 1500

    def test_returns_false_for_insufficient_matches(self):
        """Test that crop returns False when not enough ref words match."""
        import tempfile
        import os

        from audiobook_generator.audio import crop_to_ref_text

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "test.wav")
            output_path = os.path.join(tmpdir, "cropped.wav")

            sample_rate = 22050
            duration = 5.0
            write_silence_wav(audio_path, sample_rate, duration)

            # Only 2 ref words match (need at least 3)
            transcribed_words = ["garbage", "noise", "after", "all", "garbage2"]
            start_times = [0.0, 0.5, 1.0, 1.5, 2.0]
            end_times = [0.4, 0.9, 1.4, 1.9, 2.4]
            ref_words = ["after", "all", "these", "years"]

            result = crop_to_ref_text(
                audio_path, output_path,
                ref_words, transcribed_words, start_times, end_times,
                verbose=True
            )

            assert result is False

    def test_keeps_all_speech_not_just_ref_span(self):
        """Crop keeps the entire spoken clip, not only the ref-word window."""
        import tempfile
        import os

        from audiobook_generator.audio import crop_to_ref_text

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "test.wav")
            output_path = os.path.join(tmpdir, "cropped.wav")

            _write_tone_wav(audio_path, duration=3.0)

            transcribed_words = [
                "speaker", "one",  # prefix garbage
                "after", "all", "these", "years", "its", "finally", "here",  # ref words
                "blah", "blah"  # trailing garbage
            ]
            start_times = [float(i) * 0.5 for i in range(len(transcribed_words))]
            end_times = [float(i) * 0.5 + 0.4 for i in range(len(transcribed_words))]
            ref_words = ["after", "all", "these", "years", "its", "finally", "here"]

            result = crop_to_ref_text(
                audio_path, output_path,
                ref_words, transcribed_words, start_times, end_times,
                verbose=True
            )

            assert result is True
            assert os.path.exists(output_path)

            import pydub
            seg = pydub.AudioSegment.from_wav(output_path)
            # Full 3s tone kept despite ref words occupying only the middle.
            assert len(seg) > 2500


class TestCleanTextForTts:
    """Test clean_text_for_tts function for text preparation before TTS."""

    def test_removes_parenthetical_annotations(self):
        """Test that parenthetical annotations are removed."""
        text = "Hello (sighing) world"
        result = clean_text_for_tts(text)
        assert "(sighing)" not in result
        assert "Hello" in result
        assert "world" in result

    def test_removes_bracket_annotations(self):
        """Test that bracket annotations are removed."""
        text = "Hello [whispering] world"
        result = clean_text_for_tts(text)
        assert "[whispering]" not in result
        assert "Hello" in result
        assert "world" in result

    def test_removes_asterisks(self):
        """Test that asterisks are removed."""
        text = "Hello *shouting* world"
        result = clean_text_for_tts(text)
        assert "*" not in result
        assert "Hello" in result
        assert "world" in result

    def test_preserves_quotes(self):
        """Test that quotes are preserved."""
        text = 'He said "Hello world"'
        result = clean_text_for_tts(text)
        assert '"' in result
        assert "Hello world" in result

    def test_normalizes_whitespace(self):
        """Test that whitespace is normalized."""
        text = "Hello   world"
        result = clean_text_for_tts(text)
        assert "  " not in result
        assert "Hello world" in result

    def test_empty_string(self):
        """Test empty string handling."""
        result = clean_text_for_tts("")
        assert result == ""

    def test_whitespace_only(self):
        """Test whitespace-only string."""
        result = clean_text_for_tts("   ")
        assert result == ""

    def test_combined_cleaning(self):
        """Test combined cleaning operations."""
        text = "Hello (sighing) [whispering] *shouting* world"
        result = clean_text_for_tts(text)
        assert "(sighing)" not in result
        assert "[whispering]" not in result
        assert "*" not in result
        assert "Hello" in result
        assert "world" in result

    def test_preserves_dialogue(self):
        """Test that dialogue markers are preserved."""
        text = '"Hello world," she said'
        result = clean_text_for_tts(text)
        assert '"' in result
        assert "Hello world" in result

    def test_removes_stage_directions(self):
        """Test that stage directions are removed."""
        text = "Hello (enter stage left) world"
        result = clean_text_for_tts(text)
        assert "(enter stage left)" not in result
        assert "Hello" in result
        assert "world" in result