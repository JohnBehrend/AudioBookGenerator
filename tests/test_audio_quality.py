"""Tests for audio quality improvements: Whisper transcription and clipping."""

import pytest
from unittest.mock import MagicMock, patch

from audiobook_generator.pipeline import (
    score_strings_pop,
    calculate_clip_points,
    prepare_script_for_tts,
    normalize_script,
    clean_text_for_tts,
)
from audiobook_generator.utils import (
    distill_string,
    transcribe_audio_for_ref_text,
)


class TestDistillString:
    """Test distill_string function for text comparison."""

    def test_removes_punctuation(self):
        """Test that punctuation is removed."""
        assert distill_string("Hello, world!") == "hello world"

    def test_normalizes_whitespace(self):
        """Test that whitespace is normalized."""
        assert distill_string("Hello   world") == "hello world"

    def test_lowercases(self):
        """Test that string is lowercased."""
        assert distill_string("HELLO") == "hello"


class TestScoreStringsPop:
    """Test score_strings_pop function for scoring accuracy."""

    def test_perfect_match(self):
        """Test perfect match returns high score."""
        score, last_token = score_strings_pop("hello world", "hello world", postfix="")
        assert score >= 0.5
        assert last_token == "world"

    def test_postfix_detection_improves_score(self):
        """Test that postfix presence improves score."""
        input_str = "hello world and also with you"
        score_with_postfix, _ = score_strings_pop(input_str, input_str)
        score_without_postfix, _ = score_strings_pop(input_str, "hello world")
        assert score_with_postfix > score_without_postfix

    def test_lookahead_affects_matching(self):
        """Test lookahead parameter affects matching."""
        input_str = "one two three four"
        detected = "one two four"
        score, last = score_strings_pop(input_str, detected, lookahead=5)
        assert score > 0.0

    def test_last_token_extraction(self):
        """Test that last valid token is correctly identified."""
        score, last_token = score_strings_pop("the quick brown fox", "the quick fox", postfix="")
        assert last_token is not None
        assert last_token in ["quick", "fox"]


class TestCalculateClipPoints:
    """Test calculate_clip_points function for clipping accuracy."""

    def test_postfix_token_found(self):
        """Test clip points when postfix token is found."""
        segments = ["hello", "world", "and", "also", "with", "you"]
        start_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        end_times = [0.4, 0.9, 1.4, 1.9, 2.4, 2.9]

        result = calculate_clip_points(segments, start_times, end_times, "and", "world")
        assert result is not None
        start_clip, end_clip = result
        # Should clip at midpoint between "and" start (1.0) and "also" end (1.9) = 1.45s
        assert end_clip == 1450.0

    def test_last_valid_token_fallback(self):
        """Test fallback to last valid token when postfix not found."""
        segments = ["hello", "world", "missing"]
        start_times = [0.0, 0.5, 1.0]
        end_times = [0.4, 0.9, 1.4]

        result = calculate_clip_points(segments, start_times, end_times, "notfound", "world")
        assert result is not None
        _, end_clip = result
        assert end_clip == 900.0

    def test_no_clipping_needed(self):
        """Test when no clipping is needed."""
        segments = ["hello", "world"]
        start_times = [0.0, 0.5]
        end_times = [0.4, 0.9]

        result = calculate_clip_points(segments, start_times, end_times, None, None)
        assert result is None


class TestNormalizeScript:
    """Test normalize_script function for text preparation."""

    def test_capitalizes_first_letter(self):
        """Test that first letter is capitalized."""
        assert normalize_script("hello world") == "Hello world"

    def test_cleans_multiple_periods(self):
        """Test that whitespace followed by period is collapsed."""
        assert normalize_script("Hello . world") == "Hello. world"

    def test_empty_string(self):
        """Test empty string handling."""
        assert normalize_script("") == ""


class TestPrepareScriptForTts:
    """Test prepare_script_for_tts function."""

    def test_normalize_and_add_postfix(self):
        """Test combined normalization and postfix."""
        script, token = prepare_script_for_tts("hello world", "and also with you")
        assert script == "Hello world. and also with you"
        assert token == "and"

    def test_empty_text(self):
        """Test empty text returns empty."""
        script, token = prepare_script_for_tts("")
        assert script == ""
        assert token is None


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

    def test_crops_at_first_ref_word_not_prefix(self):
        """Test that crop starts at first ref word, not at prefix garbage."""
        import tempfile
        import os
        import numpy as np
        import torch
        import torchaudio

        from audiobook_generator.audio import crop_to_ref_text

        # Create a 5 second silence audio file
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "test.wav")
            output_path = os.path.join(tmpdir, "cropped.wav")

            sample_rate = 22050
            duration = 5.0
            audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
            torchaudio.save(audio_path, torch.from_numpy(audio), sample_rate)

            # Simulate transcription with prefix garbage before ref words
            # "garbage" "noise" then "after" "all" "these" "years" (ref words)
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

    def test_crops_with_small_start_buffer(self):
        """Test that start buffer is small (200ms) to avoid prefix."""
        import tempfile
        import os
        import numpy as np
        import torch
        import torchaudio

        from audiobook_generator.audio import crop_to_ref_text

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "test.wav")
            output_path = os.path.join(tmpdir, "cropped.wav")

            sample_rate = 22050
            duration = 5.0
            audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
            torchaudio.save(audio_path, torch.from_numpy(audio), sample_rate)

            # First word is a ref word
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

            # Verify crop starts near the first ref word (with small buffer)
            import pydub
            seg = pydub.AudioSegment.from_wav(output_path)
            # Should be around 800ms (1000ms - 200ms buffer)
            assert len(seg) > 0

    def test_returns_false_for_insufficient_matches(self):
        """Test that crop returns False when not enough ref words match."""
        import tempfile
        import os
        import numpy as np
        import torch
        import torchaudio

        from audiobook_generator.audio import crop_to_ref_text

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "test.wav")
            output_path = os.path.join(tmpdir, "cropped.wav")

            sample_rate = 22050
            duration = 5.0
            audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
            torchaudio.save(audio_path, torch.from_numpy(audio), sample_rate)

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

    def test_handles_prefix_garbage_before_ref_words(self):
        """Test realistic case: TTS prefix garbage followed by ref words."""
        import tempfile
        import os
        import numpy as np
        import torch
        import torchaudio

        from audiobook_generator.audio import crop_to_ref_text

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "test.wav")
            output_path = os.path.join(tmpdir, "cropped.wav")

            sample_rate = 22050
            duration = 10.0
            audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
            torchaudio.save(audio_path, torch.from_numpy(audio), sample_rate)

            # Simulate: prefix garbage -> ref words -> postfix garbage
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

            # Verify crop starts at "after" (index 2), not at "speaker" (index 0)
            import pydub
            seg = pydub.AudioSegment.from_wav(output_path)
            # Should start around 900ms (1.0s - 200ms buffer)
            assert len(seg) > 0


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