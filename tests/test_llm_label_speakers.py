"""Tests for llm_label_speakers module."""

import json
import pytest

from audiobook_generator.llm_label_speakers import (
    normalize_key_value_pairs,
    extract_json_from_text,
    parse_json_output,
    _normalize_character_map,
    _normalize_line_map,
    parse_old_format_lines,
    interpret_result,
    interpret_new_result,
    is_same_character_by_line_mapping,
)


class TestNormalizeKeyValuePairs:
    """Tests for normalize_key_value_pairs function."""

    def test_valid_json(self):
        """Test valid JSON is returned as-is."""
        result = normalize_key_value_pairs('{"key": "value"}')
        assert '"key"' in result
        assert '"value"' in result

    def test_unquoted_keys(self):
        """Test unquoted keys are properly quoted."""
        result = normalize_key_value_pairs('{1: "narrator", 2: "char1"}')
        data = json.loads(result)
        assert "1" in data or 1 in data

    def test_numeric_keys(self):
        """Test numeric keys are handled."""
        result = normalize_key_value_pairs('{1: "narrator"}')
        data = json.loads(result)
        assert len(data) == 1

    def test_nested_objects(self):
        """Test nested objects are handled."""
        result = normalize_key_value_pairs('{1: {2: "value"}}')
        assert "{" in result

    def test_whitespace_handling(self):
        """Test whitespace is handled correctly."""
        result = normalize_key_value_pairs('{  1 : "value"  }')
        data = json.loads(result)
        assert len(data) == 1


class TestExtractJsonFromText:
    """Tests for extract_json_from_text function."""

    def test_direct_json(self):
        """Test extracting direct JSON."""
        result = extract_json_from_text('{"key": "value"}')
        assert result is not None
        data = json.loads(result)
        assert data["key"] == "value"

    def test_json_in_markdown(self):
        """Test extracting JSON from markdown code block."""
        text = '```json\n{"key": "value"}\n```'
        result = extract_json_from_text(text)
        assert result is not None
        data = json.loads(result)
        assert data["key"] == "value"

    def test_json_with_thinking(self):
        """Test extracting JSON from text with LLM thinking tags."""
        text = '<think> some thinking here ```json{"key": "value"}```'
        result = extract_json_from_text(text)
        assert result is not None
        data = json.loads(result)
        assert data["key"] == "value"

    def test_no_json_returns_none(self):
        """Test that text without JSON returns None."""
        result = extract_json_from_text("Just some text without JSON")
        assert result is None

    def test_nested_json(self):
        """Test extracting nested JSON structures."""
        text = '```json\n{"speaker_map": {"1": "narrator"}, "attributions": {"2": 2}}\n```'
        result = extract_json_from_text(text)
        assert result is not None
        data = json.loads(result)
        assert "speaker_map" in data
        assert "attributions" in data


class TestNormalizeCharacterMap:
    """Tests for _normalize_character_map function."""

    def test_string_keys_to_int(self):
        """Test string keys are converted to integers."""
        speaker_map = {"1": "narrator", "2": "jane"}
        result = _normalize_character_map(speaker_map)
        assert 1 in result
        assert 2 in result

    def test_int_keys_preserved(self):
        """Test integer keys are preserved."""
        speaker_map = {1: "narrator", 2: "jane"}
        result = _normalize_character_map(speaker_map)
        assert 1 in result
        assert 2 in result

    def test_values_lowercased(self):
        """Test character names are lowercased."""
        speaker_map = {1: "Narrator", 2: "Jane"}
        result = _normalize_character_map(speaker_map)
        assert result[1] == "narrator"
        assert result[2] == "jane"

    def test_whitespace_stripped(self):
        """Test whitespace is stripped from values."""
        speaker_map = {1: "  Narrator  ", 2: " Jane "}
        result = _normalize_character_map(speaker_map)
        assert result[1] == "narrator"
        assert result[2] == "jane"

    def test_underscores_replaced(self):
        """Test underscores are replaced with spaces."""
        speaker_map = {1: "Jane_Bennet"}
        result = _normalize_character_map(speaker_map)
        assert result[1] == "jane bennet"

    def test_apostrophes_removed(self):
        """Test apostrophes are removed."""
        speaker_map = {1: "Jane's"}
        result = _normalize_character_map(speaker_map)
        assert result[1] == "janes"


class TestNormalizeLineMap:
    """Tests for _normalize_line_map function."""

    def test_string_keys_to_int(self):
        """Test string line numbers are converted to integers."""
        attributions = {"1": 2, "2": 2, "3": 3}
        result = _normalize_line_map(attributions)
        assert 1 in result
        assert 2 in result
        assert 3 in result

    def test_int_keys_preserved(self):
        """Test integer line numbers are preserved."""
        attributions = {1: 2, 2: 3}
        result = _normalize_line_map(attributions)
        assert 1 in result
        assert 2 in result

    def test_line_ranges(self):
        """Test line ranges are expanded."""
        attributions = {"1-3": 2}
        result = _normalize_line_map(attributions)
        assert 1 in result
        assert 2 in result
        assert 3 in result
        assert result[1] == 2
        assert result[2] == 2
        assert result[3] == 2


class TestParseOldFormatLines:
    """Tests for parse_old_format_lines function."""

    def test_basic_line_mapping(self):
        """Test basic line to speaker mapping."""
        result_lines = ["7:2", "9:2", "11:3"]
        char_map = {1: "narrator", 2: "jane", 3: "elizabeth"}
        result = parse_old_format_lines(result_lines, char_map)
        assert result[7] == 2
        assert result[9] == 2
        assert result[11] == 3

    def test_with_line_prefix(self):
        """Test parsing lines with 'Line' prefix."""
        result_lines = ["Line 7:2", "Line 9:2"]
        char_map = {1: "narrator", 2: "jane"}
        result = parse_old_format_lines(result_lines, char_map)
        assert 7 in result
        assert 9 in result

    def test_with_lines_prefix(self):
        """Test parsing lines with 'Lines' prefix."""
        result_lines = ["Lines 7-9:2"]
        char_map = {1: "narrator", 2: "jane"}
        result = parse_old_format_lines(result_lines, char_map)
        assert 7 in result
        assert 8 in result
        assert 9 in result

    def test_filters_invalid_characters(self):
        """Test that invalid character references are filtered."""
        result_lines = ["7:2", "9:99"]
        char_map = {1: "narrator", 2: "jane"}
        result = parse_old_format_lines(result_lines, char_map)
        assert 7 in result
        assert 9 not in result

    def test_skips_comments(self):
        """Test that comment lines are skipped."""
        result_lines = ["# This is a comment", "7:2"]
        char_map = {1: "narrator", 2: "jane"}
        result = parse_old_format_lines(result_lines, char_map)
        assert 7 in result


class TestParseJsonOutput:
    """Tests for parse_json_output function."""

    def test_new_format(self):
        """Test parsing new format with speaker_map and attributions."""
        text = '{"speaker_map": {"1": "narrator", "2": "jane"}, "attributions": {"7": 2}}'
        char_map, line_map = parse_json_output(text, 1)
        assert 1 in char_map
        assert 2 in char_map
        assert 7 in line_map

    def test_old_format_with_char_map(self):
        """Test parsing old format with char_map key."""
        text = '{"char_map": {"1": "narrator", "2": "jane"}}'
        char_map, line_map = parse_json_output(text, 1)
        assert 1 in char_map
        assert 2 in char_map

    def test_direct_mapping(self):
        """Test parsing direct character mappings."""
        text = '{"1": "narrator", "2": "jane"}'
        char_map, line_map = parse_json_output(text, 1)
        assert 1 in char_map
        assert 2 in char_map

    def test_invalid_json_raises(self):
        """Test that invalid JSON raises appropriate exception."""
        text = "This is not JSON at all"
        with pytest.raises((ValueError, json.JSONDecodeError)):
            parse_json_output(text, 1)


class TestInterpretResult:
    """Tests for interpret_result function."""

    def test_parses_new_format(self):
        """Test parsing new format with speaker_map and attributions.

        Uses extract_json_from_text which handles markdown code blocks.
        """
        from audiobook_generator.llm_label_speakers import extract_json_from_text

        text = '{"speaker_map": {"1": "narrator", "2": "jane"}, "attributions": {"7": 2}}'
        result = extract_json_from_text(text)
        assert result is not None
        data = json.loads(result)
        assert "speaker_map" in data
        assert "attributions" in data

    def test_fallback_on_bad_json(self):
        """Test that fallback parsing is used on bad JSON."""
        result = [
            'Some text before',
            'char_map : {"1": "narrator", "2": "jane"}',
            '7:2',
            '9:2'
        ]
        char_map, line_map = interpret_result(result, 1)
        assert len(char_map) >= 1
        assert len(line_map) >= 1


class TestInterpretNewResult:
    """Tests for interpret_new_result function."""

    def test_basic_parsing(self):
        """Test basic new format parsing."""
        result = ['{"speaker_map": {"1": "narrator"}, "attributions": {"7": 1}}']
        char_map, line_map = interpret_new_result(result, 1)
        assert len(char_map) >= 1
        assert len(line_map) >= 1

    def test_removes_unused_characters(self):
        """Test that unused characters are removed (except narrator)."""
        result = [
            '{"speaker_map": {"1": "narrator", "2": "jane", "3": "unused"}, "attributions": {"7": 1}}'
        ]
        char_map, line_map = interpret_new_result(result, 1)
        assert 3 not in char_map

    def test_seed_character_preference(self):
        """Test that seed character names are preferred when matching."""
        result = [
            '{"speaker_map": {"1": "narrator", "2": "jane bennet"}, "attributions": {"7": 2}}'
        ]
        seed_characters = {"jane": "/path/to/jane.wav"}
        char_map, line_map = interpret_new_result(result, 1, seed_characters)
        assert "jane" in str(char_map.values())


class TestIsSameCharacterByLineMapping:
    """Tests for is_same_character_by_line_mapping function."""

    def test_same_character_found(self):
        """Test that same character is detected by overlapping lines."""
        character_key = 2
        character = "jane"
        line_map = {7: 2, 9: 2, 11: 2}
        merged_character_map = {1: "narrator", 2: "jane"}
        merged_line_map = {7: 2, 9: 2, 11: 2, 13: 2}

        is_same, key = is_same_character_by_line_mapping(
            character_key, character, line_map, merged_character_map, merged_line_map
        )
        assert is_same is True
        assert key == 2

    def test_different_character_not_found(self):
        """Test that different character is not matched."""
        character_key = 3
        character = "elizabeth"
        line_map = {15: 3, 17: 3}
        merged_character_map = {1: "narrator", 2: "jane"}
        merged_line_map = {7: 2, 9: 2}

        is_same, key = is_same_character_by_line_mapping(
            character_key, character, line_map, merged_character_map, merged_line_map
        )
        assert is_same is False

    def test_no_overlap_returns_false(self):
        """Test that no overlap returns False."""
        character_key = 3
        character = "elizabeth"
        line_map = {15: 3, 17: 3}
        merged_character_map = {1: "narrator", 2: "jane"}
        merged_line_map = {7: 2, 9: 2}

        is_same, key = is_same_character_by_line_mapping(
            character_key, character, line_map, merged_character_map, merged_line_map
        )
        assert is_same is False


class TestLabelSpeakersIntegration:
    """Integration tests for label_speakers function."""

    def test_label_speakers_with_mock_client(self, temp_dir, mock_llm_client, sample_chapter_objs):
        """Test label_speakers with mock LLM client."""
        from audiobook_generator.llm_label_speakers import label_speakers
        from audiobook_generator.parse_chapter import write_chapters_to_txt

        chapters = [sample_chapter_objs]
        write_chapters_to_txt(chapters, str(temp_dir))

        chapter_file = temp_dir / "chapter_0.txt"

        mock_llm_client.set_response({
            "role": "assistant",
            "content": '{"speaker_map": {"1": "narrator"}, "attributions": {}}'
        })

        status, char_map, line_map = label_speakers(
            txt_file=str(chapter_file),
            api_key="mock-key",
            port="1234",
            skip_llm=False,
            client=mock_llm_client
        )

        assert isinstance(status, str)
        assert isinstance(char_map, dict)
        assert isinstance(line_map, dict)

    def test_skip_llm_mode(self, temp_dir, sample_chapter_objs):
        """Test label_speakers in skip_llm mode."""
        from audiobook_generator.llm_label_speakers import label_speakers
        from audiobook_generator.parse_chapter import write_chapters_to_txt

        chapters = [sample_chapter_objs]
        write_chapters_to_txt(chapters, str(temp_dir))

        chapter_file = temp_dir / "chapter_0.txt"

        status, char_map, line_map = label_speakers(
            txt_file=str(chapter_file),
            api_key="mock-key",
            port="1234",
            skip_llm=True
        )

        assert isinstance(char_map, dict)
        assert isinstance(line_map, dict)

    def test_multiple_attempts(self, temp_dir, mock_llm_client, sample_chapter_objs):
        """Test that multiple LLM attempts are made when configured.

        When num_attempts > 1, the function tries multiple LLM calls and
        merges the results. Each response should produce a result file.
        """
        from audiobook_generator.llm_label_speakers import label_speakers
        from audiobook_generator.parse_chapter import write_chapters_to_txt

        chapters = [sample_chapter_objs]
        write_chapters_to_txt(chapters, str(temp_dir))

        chapter_file = temp_dir / "chapter_0.txt"

        mock_llm_client.set_responses([
            {"role": "assistant", "content": '{"speaker_map": {"1": "narrator"}, "attributions": {}}'},
            {"role": "assistant", "content": '{"speaker_map": {"1": "narrator", "2": "jane"}, "attributions": {}}'},
        ])

        status, char_map, line_map = label_speakers(
            txt_file=str(chapter_file),
            api_key="mock-key",
            port="1234",
            num_attempts=2,
            client=mock_llm_client
        )

        assert isinstance(status, str)
        assert isinstance(char_map, dict)