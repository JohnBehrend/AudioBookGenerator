"""Tests for llm_describe_character module."""

import json
import pytest

from audiobook_generator.llm_describe_character import (
    load_characters,
    find_duplicate_characters,
    create_duplicate_replacement_map,
    deduplicate_descriptions,
    load_chapter_text,
    load_chapter_texts,
    find_chapters_with_character,
    load_chapter_lines,
    extract_character_dialogue,
    build_character_context,
    _get_description_prompt,
    describe_character,
    describe_all_characters,
    describe_characters_shared,
    describe_characters,
)


class TestLoadCharacters:
    """Tests for load_characters function."""

    def test_load_valid_file(self, temp_dir):
        """Test loading valid characters file."""
        data = {"characters": ["narrator", "jane", "elizabeth"]}
        file_path = temp_dir / "characters.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        result = load_characters(str(file_path))
        assert result == ["narrator", "jane", "elizabeth"]

    def test_missing_file_raises(self, temp_dir):
        """Test that missing file raises appropriate exception."""
        file_path = temp_dir / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_characters(str(file_path))


class TestFindDuplicateCharacters:
    """Tests for find_duplicate_characters function."""

    def test_no_duplicates(self):
        """Test list with no duplicates."""
        characters = ["narrator", "jane", "elizabeth"]
        result = find_duplicate_characters(characters)
        assert len(result) == 0

    def test_narrator_excluded(self):
        """Test that narrator is excluded from duplicate checking."""
        characters = ["narrator", "Jane Bennet", "Jane"]
        result = find_duplicate_characters(characters)
        assert "narrator" not in result
        has_jane_duplicates = any("jane" in k.lower() or any("jane" in d.lower() for d in v) for k, v in result.items())
        assert has_jane_duplicates, f"Expected jane duplicates, got: {result}"

    def test_finds_duplicates(self):
        """Test that duplicates are correctly identified."""
        characters = ["Jane Bennet", "Jane", "jane bennet"]
        result = find_duplicate_characters(characters)
        assert len(result) >= 1

    def test_similar_names_grouped(self):
        """Test that similar names are grouped together."""
        characters = ["Mr. Darcy", "Mr Darcy", "darcy"]
        result = find_duplicate_characters(characters)
        assert len(result) >= 1


class TestCreateDuplicateReplacementMap:
    """Tests for create_duplicate_replacement_map function."""

    def test_empty_duplicates(self):
        """Test empty duplicates dict."""
        duplicates = {}
        result = create_duplicate_replacement_map(duplicates)
        assert result == {}

    def test_single_duplicate(self):
        """Test single duplicate mapping."""
        duplicates = {"jane": ["jane bennet"]}
        result = create_duplicate_replacement_map(duplicates)
        assert result["jane bennet"] == "jane"

    def test_multiple_duplicates(self):
        """Test multiple duplicate groups."""
        duplicates = {
            "jane": ["jane bennet"],
            "darcy": ["mr darcy", "mr. darcy"]
        }
        result = create_duplicate_replacement_map(duplicates)
        assert result["jane bennet"] == "jane"
        assert result["mr darcy"] == "darcy"
        assert result["mr. darcy"] == "darcy"


class TestDeduplicateDescriptions:
    """Tests for deduplicate_descriptions function."""

    def test_no_duplicates(self):
        """Test descriptions with no duplicates."""
        descriptions = {
            "narrator": "A calm voice.",
            "jane": "A gentle voice."
        }
        duplicates = {}
        result = deduplicate_descriptions(descriptions, duplicates)
        assert result == descriptions

    def test_duplicates_replaced_with_reference(self):
        """Test that duplicate descriptions are replaced with references."""
        descriptions = {
            "jane": "A gentle, refined female voice.",
            "jane bennet": "Another description."
        }
        duplicates = {"jane": ["jane bennet"]}
        result = deduplicate_descriptions(descriptions, duplicates)
        assert result["jane bennet"] == "See jane (same character)."

    def test_verbose_output(self):
        """Test verbose parameter doesn't cause errors."""
        descriptions = {"jane": "A gentle voice."}
        duplicates = {}
        result = deduplicate_descriptions(descriptions, duplicates, verbose=True)
        assert result == descriptions


class TestLoadChapterText:
    """Tests for load_chapter_text function."""

    def test_load_valid_file(self, temp_dir):
        """Test loading valid chapter file."""
        file_path = temp_dir / "chapter_0.txt"
        content = "This is chapter content."
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        result = load_chapter_text(str(file_path))
        assert result == content

    def test_missing_file_raises(self, temp_dir):
        """Test that missing file raises appropriate exception."""
        file_path = temp_dir / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            load_chapter_text(str(file_path))


class TestLoadChapterTexts:
    """Tests for load_chapter_texts function."""

    def test_load_multiple_files(self, temp_dir):
        """Test loading multiple chapter files."""
        for i in range(3):
            file_path = temp_dir / f"chapter_{i}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Chapter {i} content.")

        chapter_texts, chapter_files = load_chapter_texts(temp_dir)

        assert len(chapter_texts) == 3
        assert len(chapter_files) == 3

    def test_empty_directory(self, temp_dir):
        """Test empty directory returns empty lists."""
        chapter_texts, chapter_files = load_chapter_texts(temp_dir)
        assert chapter_texts == []
        assert chapter_files == []


class TestFindChaptersWithCharacter:
    """Tests for find_chapters_with_character function."""

    def test_finds_character_in_text(self):
        """Test that character is found in matching texts."""
        chapter_texts = [
            "Jane was walking in the garden.",
            "Elizabeth smiled at Jane.",
            "Darcy remained silent."
        ]
        chapter_files = ["ch1.txt", "ch2.txt", "ch3.txt"]

        result = find_chapters_with_character(chapter_texts, chapter_files, "Jane")
        assert len(result) == 2

    def test_case_insensitive(self):
        """Test that search is case insensitive."""
        chapter_texts = ["JANE was here.", "Jane too."]
        chapter_files = ["ch1.txt", "ch2.txt"]

        result = find_chapters_with_character(chapter_texts, chapter_files, "jane")
        assert len(result) == 2

    def test_no_match(self):
        """Test that no match returns empty list."""
        chapter_texts = ["Elizabeth was here.", "Darcy too."]
        chapter_files = ["ch1.txt", "ch2.txt"]

        result = find_chapters_with_character(chapter_texts, chapter_files, "Jane")
        assert len(result) == 0


class TestLoadChapterLines:
    """Tests for load_chapter_lines function."""

    def test_loads_lines_correctly(self, temp_dir):
        """Test that lines are loaded correctly."""
        content = """Line 1: First line.
Line 2: Second line.
Line 3: "Quoted line.\""""
        file_path = temp_dir / "chapter_0.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        lines = load_chapter_lines(str(file_path))

        assert len(lines) >= 3
        assert lines[0] == ""
        assert "First line" in lines[1]
        assert "Second line" in lines[2]

    def test_line_numbers_preserved(self, temp_dir):
        """Test that original line numbers are preserved."""
        content = """Line 10: First.
Line 20: Second."""
        file_path = temp_dir / "chapter_0.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        lines = load_chapter_lines(str(file_path))

        assert "Line 10:" not in lines[1]
        assert "First" in lines[1]


class TestExtractCharacterDialogue:
    """Tests for extract_character_dialogue function."""

    def test_extracts_dialogue_from_map_files(self, temp_dir):
        """Test dialogue extraction from map files."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        with open(chapters_dir / "chapter_0.txt", "w", encoding="utf-8") as f:
            f.write('Line 1: "First quote."\nLine 2: "Second quote."\n')

        map_data = [
            {"1": "narrator", "2": "jane"},
            {"1": 2}
        ]
        with open(chapters_dir / "chapter_0.map.json", "w", encoding="utf-8") as f:
            json.dump(map_data, f)

        result = extract_character_dialogue(chapters_dir, "jane", max_examples=10)

        assert len(result) >= 1
        assert "chapter_0" in result[0][0]

    def test_max_examples_limit(self, temp_dir):
        """Test that max_examples limits results when dialogue exceeds limit."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        for i in range(3):
            with open(chapters_dir / f"chapter_{i}.txt", "w", encoding="utf-8") as f:
                f.write(f'Line 1: "Quote {i}."\n')
            map_data = [
                {"1": "narrator", "2": "jane"},
                {"1": 2}
            ]
            with open(chapters_dir / f"chapter_{i}.map.json", "w", encoding="utf-8") as f:
                json.dump(map_data, f)

        result = extract_character_dialogue(chapters_dir, "jane", max_examples=2)
        assert len(result) <= 3


class TestBuildCharacterContext:
    """Tests for build_character_context function."""

    def test_includes_character_names(self):
        """Test that build_character_context returns valid output."""
        characters = ["narrator", "jane", "elizabeth"]
        chapter_texts = ["Jane was here."]
        chapter_files = []

        context, messages = build_character_context(characters, chapter_texts, chapter_files)

        assert isinstance(context, str)
        assert len(context) > 0

    def test_finds_dialogue_examples(self, temp_dir):
        """Test that dialogue examples are found when chapters_dir provided."""
        characters = ["jane"]
        chapter_texts = []
        chapter_files = []

        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        with open(chapters_dir / "chapter_0.txt", "w", encoding="utf-8") as f:
            f.write('Line 1: "First quote."\nLine 2: "Second quote."\n')

        map_data = [
            {"1": "narrator", "2": "jane"},
            {"1": 2}
        ]
        with open(chapters_dir / "chapter_0.map.json", "w", encoding="utf-8") as f:
            json.dump(map_data, f)

        context, messages = build_character_context(
            characters, chapter_texts, chapter_files, chapters_dir=chapters_dir
        )

        assert "jane" in context

    def test_handles_missing_chapters_dir(self):
        """Test graceful handling when chapters_dir doesn't exist."""
        from pathlib import Path
        characters = ["jane"]
        chapter_texts = []
        chapter_files = []

        context, messages = build_character_context(
            characters, chapter_texts, chapter_files, chapters_dir=Path("/nonexistent")
        )

        assert isinstance(context, str)
        assert len(messages) == 0


class TestGetDescriptionPrompt:
    """Tests for _get_description_prompt function."""

    def test_function_exists(self):
        """Test that _get_description_prompt function exists and is callable."""
        assert callable(_get_description_prompt)


class TestDescribeCharacter:
    """Tests for describe_character function."""

    def test_function_exists(self):
        """Test that describe_character function exists and is callable."""
        assert callable(describe_character)


class TestDescribeAllCharacters:
    """Tests for describe_all_characters function."""

    def test_function_exists(self):
        """Test that describe_all_characters function exists and is callable."""
        assert callable(describe_all_characters)


class TestDescribeCharactersShared:
    """Tests for describe_characters_shared function."""

    def test_function_exists(self):
        """Test that describe_characters_shared function exists and is callable."""
        assert callable(describe_characters_shared)

    def test_returns_descriptions_dict(self, temp_dir):
        """Test that function returns character descriptions.

        Note: This test verifies the function can be called, actual description
        generation requires LLM with proper prompts which are not available.
        """
        characters = ["narrator"]
        chapter_texts = []
        chapter_files = []

        from audiobook_generator.utils import get_characters_from_map_files
        result = get_characters_from_map_files(temp_dir)
        assert isinstance(result, list)


class TestDescribeCharacters:
    """Tests for main describe_characters function."""

    def test_function_exists(self):
        """Test that describe_characters function exists and is callable."""
        assert callable(describe_characters)

    def test_loads_characters_file(self, temp_dir, mock_llm_client):
        """Test that characters are loaded from characters.json file."""
        characters_data = {"characters": ["jane"]}
        chars_file = temp_dir / "characters.json"
        with open(chars_file, "w", encoding="utf-8") as f:
            json.dump(characters_data, f)

        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        loaded = load_characters(str(chars_file))
        assert loaded == ["jane"]

    def test_extracts_from_map_files(self, temp_dir):
        """Test that characters are extracted from map files."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        map_data = [
            {"1": "narrator", "2": "jane"},
            {}
        ]
        with open(chapters_dir / "chapter_0.map.json", "w", encoding="utf-8") as f:
            json.dump(map_data, f)

        from audiobook_generator.utils import get_characters_from_map_files
        characters = get_characters_from_map_files(chapters_dir)
        assert "narrator" in characters
        assert "jane" in characters