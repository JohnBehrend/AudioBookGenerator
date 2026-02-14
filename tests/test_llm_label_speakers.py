"""Tests for llm_label_speakers.py (Stage 2: LLM Speaker Labeling)."""
import pytest
import json
import tempfile
from pathlib import Path

# Import functions to test
from llm_label_speakers import (
    add_quotes_around_keys,
    interpret_new_result,
    interpret_result,
    merge_line_maps,
    compare_characters,
    is_same_character_by_line_mapping
)


class TestAddQuotesAroundKeys:
    """Tests for add_quotes_around_keys function."""

    def test_basic_json_with_unquoted_keys(self):
        """Test basic JSON with unquoted keys."""
        input_json = '{1: "narrator", 2: "john", 3: "mary"}'
        result = add_quotes_around_keys(input_json)

        # Keys should now be quoted
        assert '"1"' in result
        assert '"2"' in result
        assert '"3"' in result

    def test_json_with_already_quoted_keys(self):
        """Test JSON that already has quoted keys."""
        input_json = '{"1": "narrator", "2": "john"}'
        result = add_quotes_around_keys(input_json)

        assert '"1"' in result
        assert '"2"' in result

    def test_json_with_unquoted_values(self):
        """Test JSON with unquoted values."""
        input_json = '{1: narrator, 2: john}'
        result = add_quotes_around_keys(input_json)

        # Values should be quoted
        assert '"narrator"' in result
        assert '"john"' in result


class TestInterpretNewResult:
    """Tests for interpret_new_result function."""

    def test_valid_json_result(self):
        """Test parsing a valid new-format result."""
        result = [
            '{',
            '  "speaker_map": {',
            '    "1": "narrator",',
            '    "2": "John",',
            '    "3": "Mary"',
            '  },',
            '  "attributions": {',
            '    "2": 2,',
            '    "4": 3',
            '  }',
            '}'
        ]

        char_map, line_map = interpret_new_result(result, 1)

        assert 1 in char_map
        assert char_map[1] == "narrator"
        assert 2 in char_map
        assert 3 in char_map

        assert 2 in line_map
        assert line_map[2] == 2
        assert 4 in line_map
        assert line_map[4] == 3

    def test_character_name_normalization(self):
        """Test that character names are normalized (lowercase, simplified)."""
        result = [
            '{',
            '  "speaker_map": {',
            '    "1": "Narrator",',
            '    "2": "John Smith"',
            '  },',
            '  "attributions": {',
            '    "2": 2',
            '  }',
            '}'
        ]

        char_map, line_map = interpret_new_result(result, 1)

        assert char_map[2] == "john smith"  # Lowercase

    def test_handles_name_variants(self):
        """Test handling of name variants with slashes."""
        result = [
            '{',
            '  "speaker_map": {',
            '    "1": "narrator",',
            '    "2": "John/Sir John"',
            '  },',
            '  "attributions": {',
            '    "2": 2',
            '  }',
            '}'
        ]

        char_map, line_map = interpret_new_result(result, 1)

        # Should take first part before slash
        assert char_map[2] == "john"


class TestInterpretResult:
    """Tests for interpret_result function (old format)."""

    def test_old_format_with_char_map(self):
        """Test parsing old format with char_map."""
        result = [
            "-----",
            'char_map : {"1": "narrator", "2": "john", "3": "mary"}',
            "2:2",
            "4:3",
            "6:2",
            "-----"
        ]

        char_map, line_map = interpret_result(result, 1)

        assert "1" in char_map or 1 in char_map
        assert 2 in line_map
        assert line_map[2] == 2

    def test_old_format_with_line_ranges(self):
        """Test old format with line ranges."""
        result = [
            "-----",
            'char_map : {"1": "narrator", "2": "john"}',
            "2-5:2",
            "-----"
        ]

        char_map, line_map = interpret_result(result, 1)

        # Range should expand to individual lines
        assert 2 in line_map
        assert 3 in line_map
        assert 4 in line_map
        assert 5 in line_map


class TestMergeLineMaps:
    """Tests for merge_line_maps function."""

    def test_merge_single_map(self):
        """Test merging a single line map."""
        line_maps = [
            {1: 1, 2: 2, 3: 2}
        ]

        result = merge_line_maps(line_maps)

        assert result == {1: 1, 2: 2, 3: 2}

    def test_merge_multiple_maps_same_values(self):
        """Test merging multiple maps with same values."""
        line_maps = [
            {1: 1, 2: 2},
            {1: 1, 2: 2},
            {1: 1, 2: 2}
        ]

        result = merge_line_maps(line_maps)

        assert result == {1: 1, 2: 2}

    def test_merge_multiple_maps_majority_wins(self):
        """Test that majority wins in merging."""
        line_maps = [
            {1: 1, 2: 2},
            {1: 1, 2: 3},
            {1: 1, 2: 2}  # 2 appears twice
        ]

        result = merge_line_maps(line_maps)

        assert result[2] == 2  # Majority vote

    def test_empty_line_maps(self):
        """Test with empty line maps list."""
        result = merge_line_maps([])

        assert result == {}


class TestCompareCharacters:
    """Tests for compare_characters function."""

    def test_exact_match(self):
        """Test exact character name match."""
        assert compare_characters("john", "john") is True

    def test_substring_match(self):
        """Test substring character name match."""
        assert compare_characters("john", "john smith") is True
        assert compare_characters("smith", "john smith") is True

    def test_different_names(self):
        """Test different character names."""
        assert compare_characters("john", "mary") is False
        assert compare_characters("john", "bob") is False


class TestIsSameCharacterByLineMapping:
    """Tests for is_same_character_by_line_mapping function."""

    def test_same_character_by_line_mapping(self):
        """Test detection of same character by line mapping overlap."""
        character_key = 2
        character = "john"
        line_map = {1: 2, 2: 2, 3: 2}
        merged_character_map = {1: "narrator", 3: "john"}
        merged_line_map = {1: 3, 2: 3, 3: 3}

        # john in line_map maps to key 2, but in merged it maps to key 3 (john)
        is_same, match_key = is_same_character_by_line_mapping(
            character_key, character, line_map,
            merged_character_map, merged_line_map
        )

        assert is_same is True
        assert match_key == 3

    def test_different_characters(self):
        """Test detection of different characters - no overlapping lines."""
        character_key = 2
        character = "john"
        # john speaks on lines 1,2 (key 2)
        line_map = {1: 2, 2: 2}
        # mary speaks on lines 3,4 (key 3) - NO overlap with john
        merged_character_map = {1: "narrator", 3: "mary"}
        merged_line_map = {3: 3, 4: 3}  # mary's lines are 3,4 - no overlap

        is_same, match_key = is_same_character_by_line_mapping(
            character_key, character, line_map,
            merged_character_map, merged_line_map
        )

        assert is_same is False
        assert match_key is None


@pytest.fixture
def sample_result_file(temp_dir):
    """Create a sample result file for testing."""
    def _create_result(attempt_num, content):
        result_file = temp_dir / f"chapter_0.result.{attempt_num}.txt"
        result_file.write_text(content)
        return result_file

    return _create_result


@pytest.fixture
def sample_chapter_text_file(temp_dir):
    """Create a sample chapter text file."""
    chapter_file = temp_dir / "chapter_0.txt"
    chapter_file.write_text("""The narrator speaks here.
"Hello there," said John.
"I am fine," replied Mary.
The story continues.
""")
    return chapter_file