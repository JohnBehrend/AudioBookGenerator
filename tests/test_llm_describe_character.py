"""Tests for llm_describe_character.py (Stage 4: Character Descriptions)."""
import pytest
import json
import tempfile
from pathlib import Path

from llm_describe_character import (
    load_characters,
    compare_characters,
    find_duplicate_characters,
    create_duplicate_replacement_map,
    deduplicate_descriptions,
    load_chapter_text,
    find_chapters_with_character,
    build_character_context
)


class TestLoadCharacters:
    """Tests for load_characters function."""

    def test_load_valid_characters(self, temp_dir):
        """Test loading valid characters JSON."""
        characters_file = temp_dir / "characters.json"
        characters_file.write_text(json.dumps({
            "characters": ["narrator", "john", "mary"]
        }))

        result = load_characters(str(characters_file))

        assert result == ["narrator", "john", "mary"]

    def test_load_characters_with_extra_fields(self, temp_dir):
        """Test loading characters JSON with extra fields."""
        characters_file = temp_dir / "characters.json"
        characters_file.write_text(json.dumps({
            "characters": ["narrator", "john"],
            "metadata": {"source": "test"}
        }))

        result = load_characters(str(characters_file))

        assert result == ["narrator", "john"]

    def test_load_empty_characters(self, temp_dir):
        """Test loading empty characters list."""
        characters_file = temp_dir / "characters.json"
        characters_file.write_text(json.dumps({"characters": []}))

        result = load_characters(str(characters_file))

        assert result == []


class TestCompareCharacters:
    """Tests for compare_characters function."""

    def test_exact_match(self):
        """Test exact character name match."""
        assert compare_characters("john", "john") is True

    def test_substring_full_word_match(self):
        """Test substring match with word boundaries."""
        assert compare_characters("john", "john smith") is True
        assert compare_characters("smith", "john smith") is True

    def test_no_partial_match(self):
        """Test that partial substring doesn't match."""
        assert compare_characters("ara", "sarah") is False
        assert compare_characters("jo", "john") is False

    def test_different_names(self):
        """Test different character names."""
        assert compare_characters("john", "mary") is False
        assert compare_characters("bob", "alice") is False


class TestFindDuplicateCharacters:
    """Tests for find_duplicate_characters function."""

    def test_no_duplicates(self):
        """Test with no duplicate characters."""
        characters = ["narrator", "john", "mary", "bob"]
        duplicates = find_duplicate_characters(characters)
        assert duplicates == {}

    def test_simple_duplicate(self):
        """Test with simple duplicate."""
        characters = ["narrator", "john smith", "john"]
        duplicates = find_duplicate_characters(characters)
        assert "john smith" in duplicates
        assert "john" in duplicates["john smith"]

    def test_narrator_excluded(self):
        """Test that narrator is excluded from duplicates."""
        characters = ["narrator", "narrator voice"]
        duplicates = find_duplicate_characters(characters)
        assert duplicates == {}

    def test_multiple_duplicates(self):
        """Test with multiple duplicate groups."""
        characters = ["john", "john smith", "mary", "mary jane", "bob"]
        duplicates = find_duplicate_characters(characters)
        # canonical -> duplicates mapping
        assert "john" in duplicates
        assert "mary" in duplicates
        assert "bob" not in duplicates


class TestCreateDuplicateReplacementMap:
    """Tests for create_duplicate_replacement_map function."""

    def test_create_replacement_map(self):
        """Test creating replacement map."""
        duplicates = {
            "john smith": ["john"],
            "mary jane": ["mary", "maryjane"]
        }
        replacement_map = create_duplicate_replacement_map(duplicates)
        assert replacement_map == {
            "john": "john smith",
            "mary": "mary jane",
            "maryjane": "mary jane"
        }


class TestDeduplicateDescriptions:
    """Tests for deduplicate_descriptions function."""

    def test_deduplicate_with_references(self):
        """Test deduplication with reference comments."""
        descriptions = {
            "narrator": "Male, middle-aged, calm voice.",
            "john": "Male, young, enthusiastic voice.",
            "john smith": "Male, old, gruff voice."
        }
        duplicates = {"john smith": ["john"]}
        deduped = deduplicate_descriptions(descriptions, duplicates)
        assert "john" in deduped
        assert "See john smith" in deduped["john"]

    def test_preserve_canonical(self):
        """Test that canonical descriptions are preserved."""
        descriptions = {
            "narrator": "Male, middle-aged, calm voice.",
            "john smith": "Male, old, gruff voice."
        }
        duplicates = {}
        deduped = deduplicate_descriptions(descriptions, duplicates)
        assert deduped["john smith"] == "Male, old, gruff voice."


class TestLoadChapterText:
    """Tests for load_chapter_text function."""

    def test_load_existing_chapter(self, temp_dir):
        """Test loading an existing chapter file."""
        chapter_file = temp_dir / "chapter_0.txt"
        chapter_file.write_text("This is chapter text.")
        result = load_chapter_text(str(chapter_file))
        assert result == "This is chapter text."

    def test_load_missing_chapter(self, temp_dir):
        """Test loading a non-existent chapter file."""
        with pytest.raises(FileNotFoundError):
            load_chapter_text(str(temp_dir / "nonexistent.txt"))


class TestFindChaptersWithCharacter:
    """Tests for find_chapters_with_character function."""

    def test_find_character_in_chapters(self, temp_dir):
        """Test finding a character in chapter texts."""
        chapter_texts = ["John appears here", "No one here", "Mary and John"]
        chapter_files = [temp_dir / "c1.txt", temp_dir / "c2.txt", temp_dir / "c3.txt"]
        result = find_chapters_with_character(chapter_texts, chapter_files, "john")
        assert len(result) == 2
        assert result[0][0] == chapter_files[0]
        assert result[1][0] == chapter_files[2]

    def test_case_insensitive_search(self, temp_dir):
        """Test case-insensitive character search."""
        chapter_texts = ["JOHN WAS HERE", "john again", "No one"]
        chapter_files = [temp_dir / "c1.txt", temp_dir / "c2.txt", temp_dir / "c3.txt"]
        result = find_chapters_with_character(chapter_texts, chapter_files, "John")
        assert len(result) == 2


class TestBuildCharacterContext:
    """Tests for build_character_context function."""

    def test_build_basic_context(self):
        """Test building basic character context."""
        characters = ["narrator", "john", "mary"]
        chapter_texts = ["Some text here", "More text", "Final text"]
        context = build_character_context(characters, chapter_texts)
        assert "Characters to describe:" in context
        assert "- narrator" in context
        assert "- john" in context
        assert "- mary" in context

    def test_build_context_with_chapter_files(self, temp_dir):
        """Test building context with chapter files."""
        characters = ["john"]
        chapter_texts = ["John speaks here"]
        chapter_files = [temp_dir / "chapter_0.txt"]
        context = build_character_context(characters, chapter_texts, chapter_files)
        assert "Relevant dialogue and context" in context

    def test_build_context_with_wiki_template(self):
        """Test building context with wiki URL template."""
        characters = ["john"]
        chapter_texts = ["Some text"]
        context = build_character_context(
            characters, chapter_texts,
            wiki_url_template="https://en.wikipedia.org/wiki/{name}"
        )
        assert "Additional context from wiki" in context


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)