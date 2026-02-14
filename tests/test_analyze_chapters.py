"""Tests for analyze_chapters.py (Stage 3: Chapter Analysis)."""
import pytest
import json
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from analyze_chapters import (
    Chapter,
    ChapterAnalyzer,
    load_chapter_map,
    find_chapter_files
)


class TestChapter:
    """Tests for the Chapter class."""

    @pytest.fixture
    def sample_character_map(self):
        return {"1": "narrator", "2": "john", "3": "mary"}

    @pytest.fixture
    def sample_line_map(self):
        return {"1": 1, "2": 2, "3": 2, "4": 3, "5": 1}

    def test_initialization(self, sample_character_map, sample_line_map):
        """Test Chapter initialization."""
        chapter = Chapter(0, sample_character_map, sample_line_map)

        assert chapter.chapter_num == 0
        assert chapter.character_map == sample_character_map
        assert chapter.line_map == sample_line_map

    def test_total_lines(self, sample_line_map):
        """Test total_lines property."""
        chapter = Chapter(0, {}, sample_line_map)
        assert chapter.total_lines == 5

    def test_narrator_lines(self, sample_line_map):
        """Test narrator_lines property."""
        chapter = Chapter(0, {}, sample_line_map)
        # Lines 1 and 5 have speaker 1 (narrator)
        assert chapter.narrator_lines == 2

    def test_dialogue_lines(self, sample_line_map):
        """Test dialogue_lines property."""
        chapter = Chapter(0, {}, sample_line_map)
        # 5 total - 2 narrator = 3 dialogue
        assert chapter.dialogue_lines == 3

    def test_unique_speakers(self, sample_character_map, sample_line_map):
        """Test unique_speakers property."""
        chapter = Chapter(0, sample_character_map, sample_line_map)
        speakers = chapter.unique_speakers

        assert "narrator" in speakers
        assert "john" in speakers
        assert "mary" in speakers
        assert len(speakers) == 3

    def test_speaker_counts(self, sample_line_map):
        """Test speaker_counts property."""
        chapter = Chapter(0, {"1": "narrator", "2": "john", "3": "mary"}, sample_line_map)
        counts = chapter.speaker_counts

        assert counts["narrator"] == 2  # Lines 1, 5
        assert counts["john"] == 2      # Lines 2, 3
        assert counts["mary"] == 1      # Line 4

    def test_get_speaker_name_narrator(self):
        """Test _get_speaker_name for narrator (key 1)."""
        chapter = Chapter(0, {"1": "narrator"}, {})
        assert chapter._get_speaker_name(1) == "narrator"

    def test_get_speaker_name_known(self):
        """Test _get_speaker_name for known character."""
        chapter = Chapter(0, {"2": "john"}, {})
        assert chapter._get_speaker_name(2) == "john"

    def test_get_speaker_name_unknown(self):
        """Test _get_speaker_name for unknown character."""
        chapter = Chapter(0, {"2": "john"}, {})
        assert chapter._get_speaker_name(99) == "Unknown_99"


class TestChapterAnalyzer:
    """Tests for the ChapterAnalyzer class."""

    @pytest.fixture
    def temp_chapters_dir(self, temp_dir):
        """Create a temporary chapters directory with sample map files."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir()

        # Create sample map files
        for i in range(3):
            map_file = chapters_dir / f"chapter_{i}.map.json"
            character_map = {"1": "narrator", "2": f"char_{i}"}
            line_map = {str(j): (j % 2) + 1 for j in range(10)}
            map_file.write_text(json.dumps([character_map, line_map]))

        return chapters_dir

    def test_initialization_with_valid_directory(self, temp_chapters_dir):
        """Test ChapterAnalyzer initialization with valid directory."""
        analyzer = ChapterAnalyzer(str(temp_chapters_dir), verbose=False)

        assert len(analyzer.chapters) == 3
        assert analyzer.directory == temp_chapters_dir

    def test_get_overall_statistics(self, temp_chapters_dir):
        """Test overall statistics calculation."""
        analyzer = ChapterAnalyzer(str(temp_chapters_dir), verbose=False)
        stats = analyzer.get_overall_statistics()

        assert stats["total_chapters"] == 3
        assert stats["total_lines"] == 30  # 10 lines * 3 chapters
        assert "speaker_counts" in stats
        assert "narrator_lines" in stats

    def test_get_chapter_stats(self, temp_chapters_dir):
        """Test per-chapter statistics."""
        analyzer = ChapterAnalyzer(str(temp_chapters_dir), verbose=False)
        chapter_stats = analyzer.get_chapter_stats()

        assert len(chapter_stats) == 3
        assert chapter_stats[0]["chapter"] == 0
        assert "total_lines" in chapter_stats[0]
        assert "unique_speakers" in chapter_stats[0]

    def test_get_top_speakers(self, temp_chapters_dir):
        """Test top speakers ranking."""
        analyzer = ChapterAnalyzer(str(temp_chapters_dir), verbose=False)
        top_speakers = analyzer.get_top_speakers(n=5)

        assert len(top_speakers) > 0
        # Should be sorted by count (descending)
        if len(top_speakers) > 1:
            assert top_speakers[0][1] >= top_speakers[1][1]


class TestLoadChapterMap:
    """Tests for load_chapter_map function."""

    def test_load_valid_map(self, temp_dir):
        """Test loading a valid chapter map."""
        map_file = temp_dir / "chapter_0.map.json"
        character_map = {"1": "narrator", "2": "john"}
        line_map = {"1": 1, "2": 2}
        map_file.write_text(json.dumps([character_map, line_map]))

        loaded_map, loaded_lines = load_chapter_map(str(map_file))

        assert loaded_map == character_map
        assert loaded_lines == line_map

    def test_load_invalid_file(self, temp_dir):
        """Test loading an invalid JSON file."""
        map_file = temp_dir / "chapter_0.map.json"
        map_file.write_text("invalid json")

        result = load_chapter_map(str(map_file))

        assert result == ({}, {})

    def test_load_missing_file(self):
        """Test loading a non-existent file."""
        result = load_chapter_map("/nonexistent/file.map.json")

        assert result == ({}, {})


class TestFindChapterFiles:
    """Tests for find_chapter_files function (imported from list_chapters)."""

    def test_find_chapter_files_in_directory(self, temp_dir):
        """Test finding chapter files in directory."""
        from list_chapters import find_chapter_files

        # Create sample files
        (temp_dir / "chapter_0.map.json").write_text("{}")
        (temp_dir / "chapter_1.map.json").write_text("{}")
        (temp_dir / "chapter_2.txt").write_text("{}")  # Should not be found
        (temp_dir / "other.txt").write_text("{}")  # Should not be found

        files = find_chapter_files(str(temp_dir), verbose=False)

        assert len(files) == 2
        assert any("chapter_0" in str(f) for f in files)
        assert any("chapter_1" in str(f) for f in files)

    def test_filter_by_chapter_number(self, temp_dir):
        """Test filtering files by chapter number."""
        from list_chapters import find_chapter_files

        # Note: the function uses :02d format, so chapter 1 becomes "01"
        (temp_dir / "chapter_00.map.json").write_text("{}")
        (temp_dir / "chapter_01.map.json").write_text("{}")
        (temp_dir / "chapter_02.map.json").write_text("{}")

        files = find_chapter_files(str(temp_dir), chapter_filter=1, verbose=False)

        assert len(files) == 1
        assert "chapter_01" in str(files[0])


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)