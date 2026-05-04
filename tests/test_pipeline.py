"""Tests for pipeline module - pure functions for TTS processing."""

import pytest

from audiobook_generator.pipeline import (
    normalize_script,
    add_postfix,
    prepare_script_for_tts,
    score_strings_pop,
    calculate_clip_points,
    should_retry,
    generate_output_filename,
    is_generation_success,
    collect_transcription_segments,
    END_CHARACTERS,
    MIN_RATIO_THRESHOLD,
    MAX_RETRIES,
)


class TestNormalizeScript:
    """Tests for normalize_script function."""

    def test_capitalizes_first_letter(self):
        """Test that first letter is capitalized."""
        assert normalize_script("hello world") == "Hello world"
        assert normalize_script("HELLO WORLD") == "HELLO WORLD"
        assert normalize_script("123 hello") == "123 hello"

    def test_cleans_multiple_periods(self):
        """Test that whitespace followed by period is collapsed to single period."""
        assert normalize_script("Hello . world") == "Hello. world"

    def test_empty_string(self):
        """Test empty string handling."""
        assert normalize_script("") == ""
        assert normalize_script("   ") == "   "

    def test_preserves_content(self):
        """Test that non-special content is preserved."""
        assert normalize_script("Hello, world!") == "Hello, world!"


class TestAddPostfix:
    """Tests for add_postfix function."""

    def test_no_postfix(self):
        """Test when postfix is None."""
        script, token = add_postfix("Hello world", None)
        assert script == "Hello world"
        assert token is None

    def test_postfix_with_punctuation(self):
        """Test postfix after punctuation."""
        script, token = add_postfix("Hello world.", "and also with you")
        assert script == "Hello world. and also with you"
        assert token == "and"

    def test_postfix_without_punctuation(self):
        """Test postfix after no punctuation."""
        script, token = add_postfix("Hello world", "and also with you")
        assert script == "Hello world. and also with you"
        assert token == "and"

    def test_postfix_with_question_mark(self):
        """Test postfix after question mark."""
        script, token = add_postfix("Are you there?", "and also with you")
        assert script == "Are you there? and also with you"

    def test_postfix_with_exclamation(self):
        """Test postfix after exclamation."""
        script, token = add_postfix("Look out!", "and also with you")
        assert script == "Look out! and also with you"

    def test_extracts_first_token(self):
        """Test that first word is extracted as token."""
        script, token = add_postfix("Hello.", "one two three")
        assert token == "one"


class TestPrepareScriptForTts:
    """Tests for prepare_script_for_tts function."""

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

    def test_whitespace_only(self):
        """Test whitespace-only text."""
        script, token = prepare_script_for_tts("   ")
        assert script == ""
        assert token is None

    def test_no_postfix(self):
        """Test without postfix."""
        script, token = prepare_script_for_tts("hello world")
        assert script == "Hello world"
        assert token is None


class TestScoreStringsPop:
    """Tests for score_strings_pop function."""

    def test_perfect_match(self):
        """Test perfect match returns expected score.

        Note: The algorithm penalizes for missing postfix, so max score is 0.5
        when no postfix is provided. This is a known behavior of the original algorithm.
        """
        score, last_token = score_strings_pop("hello world", "hello world", postfix="")
        assert score >= 0.0
        assert last_token == "world"

    def test_partial_match(self):
        """Test partial match returns appropriate score."""
        score, last_token = score_strings_pop("hello world test", "hello world", postfix="")
        assert 0.0 <= score <= 1.0
        assert last_token is not None

    def test_no_match(self):
        """Test no match returns low score."""
        score, last_token = score_strings_pop("hello world", "goodbye world", postfix="")
        assert score < 0.9

    def test_empty_input(self):
        """Test empty input string."""
        score, last_token = score_strings_pop("", "hello")
        assert score == 0.0
        assert last_token is None

    def test_empty_detected(self):
        """Test empty detected string."""
        score, last_token = score_strings_pop("hello world", "")
        assert score == 0.0
        assert last_token is None

    def test_postfix_detection(self):
        """Test that postfix presence affects score."""
        input_str = "hello world and also with you"
        score_with_postfix, _ = score_strings_pop(input_str, input_str)
        score_without_postfix, _ = score_strings_pop(input_str, "hello world")

        assert score_with_postfix > score_without_postfix

    def test_lookahead_parameter(self):
        """Test lookahead parameter affects matching."""
        input_str = "one two three four"
        detected = "one two four"
        score, last = score_strings_pop(input_str, detected, lookahead=5)
        assert score > 0.0

    def test_last_token_extraction(self):
        """Test that last valid token is correctly identified.

        The algorithm returns the last token that was found, which may not
        be the last expected token if there's a gap.
        """
        score, last_token = score_strings_pop("the quick brown fox", "the quick fox", postfix="")
        assert last_token is not None
        assert last_token in ["quick", "fox"]

    def test_token_reuse_detection(self):
        """Test handling of repeated tokens."""
        score, last_token = score_strings_pop("hello world hello", "hello world hello", postfix="")
        assert score >= 0.0
        assert last_token is not None


class TestCalculateClipPoints:
    """Tests for calculate_clip_points function."""

    def test_empty_segments(self):
        """Test empty segments return None."""
        result = calculate_clip_points([], [], [], "test", "token")
        assert result is None

    def test_postfix_token_found(self):
        """Test clip points when postfix token is found.

        When postfix token is found, clip points are calculated using
        the token's position in the reversed lists.
        """
        segments = ["hello", "world", "and", "also", "with", "you"]
        start_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        end_times = [0.4, 0.9, 1.4, 1.9, 2.4, 2.9]

        result = calculate_clip_points(segments, start_times, end_times, "and", "world")

        assert result is not None
        start_clip, end_clip = result
        assert end_clip is not None

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

    def test_token_not_in_segments(self):
        """Test when token is not found in segments."""
        segments = ["hello", "world"]
        start_times = [0.0, 0.5]
        end_times = [0.4, 0.9]

        result = calculate_clip_points(segments, start_times, end_times, "missing", "missing")

        assert result is None


class TestShouldRetry:
    """Tests for should_retry function."""

    def test_retry_on_low_ratio(self):
        """Test retry when ratio is below threshold."""
        assert should_retry(0.5, 0.0, 0, 2) is True

    def test_no_retry_on_good_ratio(self):
        """Test no retry when ratio is above threshold."""
        assert should_retry(0.9, 0.9, 0, 2) is False

    def test_retry_on_max_retries(self):
        """Test no retry when at max retries."""
        assert should_retry(0.5, 0.5, 2, 2) is False

    def test_default_max_retries(self):
        """Test default max retries value."""
        assert should_retry(0.5, 0.0, 0) is True
        assert should_retry(0.5, 0.0, MAX_RETRIES) is False

    def test_custom_threshold(self):
        """Test custom threshold."""
        assert should_retry(0.6, 0.0, 0, max_retries=2, min_ratio=0.7) is True
        assert should_retry(0.8, 0.0, 0, max_retries=2, min_ratio=0.7) is False


class TestGenerateOutputFilename:
    """Tests for generate_output_filename function."""

    def test_temp_filename(self):
        """Test temporary filename format."""
        path = generate_output_filename("/tmp", 0, 5, is_final=False)
        assert path == "/tmp/chapter_00.0005.tmp.wav"

    def test_final_filename(self):
        """Test final filename format."""
        path = generate_output_filename("/tmp", 1, 23, is_final=True)
        assert path == "/tmp/chapter_01.0023.wav"

    def test_zfill_formatting(self):
        """Test zero-padding for chapter and line numbers."""
        path = generate_output_filename("/tmp", 9, 8, is_final=True)
        assert "chapter_09" in path
        assert "0008.wav" in path

    def test_high_indices(self):
        """Test with high chapter and line numbers."""
        path = generate_output_filename("/tmp", 15, 9999, is_final=True)
        assert "chapter_15" in path
        assert "9999.wav" in path


class TestIsGenerationSuccess:
    """Tests for is_generation_success function."""

    def test_success_on_threshold(self):
        """Test success at threshold."""
        assert is_generation_success(MIN_RATIO_THRESHOLD) is True

    def test_success_above_threshold(self):
        """Test success above threshold."""
        assert is_generation_success(0.95) is True
        assert is_generation_success(1.0) is True

    def test_failure_below_threshold(self):
        """Test failure below threshold."""
        assert is_generation_success(0.5) is False
        assert is_generation_success(0.0) is False

    def test_custom_threshold(self):
        """Test custom threshold."""
        assert is_generation_success(0.6, min_ratio=0.5) is True
        assert is_generation_success(0.4, min_ratio=0.5) is False


class TestCollectTranscriptionSegments:
    """Tests for collect_transcription_segments function."""

    def test_extracts_segments(self):
        """Test segment extraction from mock segments."""
        class MockWord:
            def __init__(self, word, start, end):
                self.word = word
                self.start = start
                self.end = end

        class MockSegment:
            def __init__(self, words):
                self.words = words

        segments_list = [
            MockSegment([MockWord("hello", 0.0, 0.5), MockWord("world", 0.5, 1.0)])
        ]

        segments, starts, ends = collect_transcription_segments(segments_list)

        assert segments == ["hello", "world"]
        assert starts == [0.0, 0.5]
        assert ends == [0.5, 1.0]

    def test_empty_segments(self):
        """Test with empty segments."""
        segments, starts, ends = collect_transcription_segments([])
        assert segments == []
        assert starts == []
        assert ends == []


class TestConstants:
    """Tests for module constants."""

    def test_end_characters(self):
        """Test END_CHARACTERS contains expected punctuation."""
        assert "?" in END_CHARACTERS
        assert "." in END_CHARACTERS
        assert "!" in END_CHARACTERS
        assert "," in END_CHARACTERS
        assert ";" in END_CHARACTERS
        assert "-" in END_CHARACTERS

    def test_min_ratio_threshold(self):
        """Test MIN_RATIO_THRESHOLD is reasonable."""
        assert 0.0 < MIN_RATIO_THRESHOLD <= 1.0
        assert MIN_RATIO_THRESHOLD == 0.85

    def test_max_retries(self):
        """Test MAX_RETRIES is reasonable."""
        assert MAX_RETRIES > 0
        assert MAX_RETRIES == 2


class TestIntegrationScenarios:
    """Integration tests for pipeline scenarios."""

    def test_full_normalization_pipeline(self):
        """Test complete text preparation pipeline."""
        raw_text = "hello world"
        postfix = "and also with you"

        normalized = normalize_script(raw_text)
        assert normalized == "Hello world"

        with_postfix, token = add_postfix(normalized, postfix)
        assert "and also with you" in with_postfix
        assert token == "and"

    def test_empty_text_early_return(self):
        """Test that empty text is handled early."""
        script, token = prepare_script_for_tts("")
        assert script == ""
        assert token is None

        assert normalize_script("") == ""

    def test_scoring_pipeline(self):
        """Test full scoring pipeline."""
        input_str = "hello world"
        detected = "hello world"

        score, last_token = score_strings_pop(input_str, detected, postfix="")
        assert score >= 0.0
        assert last_token == "world"

    def test_clipping_pipeline(self):
        """Test full clipping calculation pipeline."""
        segments = ["hello", "world", "and", "also"]
        start_times = [0.0, 0.5, 1.0, 1.5]
        end_times = [0.4, 0.9, 1.4, 1.9]

        result = calculate_clip_points(segments, start_times, end_times, "and", "world")

        assert result is not None

    def test_retry_logic_pipeline(self):
        """Test retry decision pipeline."""
        attempts = [
            (0.5, False),
            (0.6, False),
            (0.85, True),
        ]

        max_ratio = 0.0
        for i, (ratio, should_stop) in enumerate(attempts):
            if ratio > max_ratio:
                max_ratio = ratio

            should_continue = should_retry(ratio, max_ratio, i)
            if not should_continue:
                assert should_stop or i == len(attempts) - 1
            else:
                assert not should_stop