"""Tests for parse_chapter.py (Stage 1: EPUB Parsing)."""
import pytest
from parse_chapter import ChapterObj, get_chapter_objs, cleanup_text


class TestChapterObj:
    """Tests for the ChapterObj class."""

    def test_initialization(self):
        """Test ChapterObj initialization with required properties."""
        obj = ChapterObj(has_quotes=True, text="Hello world", line_num=1)

        assert obj.has_quotes is True
        assert obj.text == "Hello world"
        assert obj.line_num == 1
        assert obj.speaker == "narrator"  # default value

    def test_get_speaker_default(self):
        """Test get_speaker returns default narrator."""
        obj = ChapterObj(has_quotes=False, text="Some text", line_num=1)
        assert obj.get_speaker() == "narrator"

    def test_set_speaker(self):
        """Test set_speaker method."""
        obj = ChapterObj(has_quotes=True, text="Hello", line_num=1)
        obj.set_speaker("john")
        assert obj.get_speaker() == "john"

    def test_str_representation(self):
        """Test string representation."""
        obj = ChapterObj(has_quotes=True, text="Hello world", line_num=5)
        obj.set_speaker("john")
        expected = "Line 5: Speaker john: Hello world"
        assert str(obj) == expected


class TestCleanupText:
    """Tests for the cleanup_text function."""

    def test_reduces_multiple_spaces(self):
        """Test that multiple spaces are reduced."""
        assert cleanup_text("hello    world") == "hello world"
        assert cleanup_text("hello     world") == "hello world"

    def test_reduces_double_spaces(self):
        """Test that double spaces are reduced."""
        assert cleanup_text("hello  world") == "hello world"

    def test_preserves_single_spaces(self):
        """Test that single spaces are preserved."""
        assert cleanup_text("hello world") == "hello world"

    def test_combined_cleaning(self):
        """Test combined space cleaning."""
        text = "hello   world   test"
        assert cleanup_text(text) == "hello world test"


class TestGetChapterObjs:
    """Tests for the get_chapter_objs function."""

    def test_empty_text(self):
        """Test with empty text."""
        result = get_chapter_objs("")
        assert result == []

    def test_whitespace_only(self):
        """Test with whitespace-only text."""
        result = get_chapter_objs("   \n   \n   ")
        assert result == []

    def test_single_paragraph_no_quotes(self):
        """Test single paragraph without quotes."""
        text = "This is a simple paragraph."
        result = get_chapter_objs(text)

        assert len(result) == 1
        assert result[0].has_quotes is False
        assert result[0].text == "This is a simple paragraph."
        assert result[0].line_num == 1

    def test_single_paragraph_with_quotes(self):
        """Test single paragraph with quotes."""
        text = 'He said "hello" to me.'
        result = get_chapter_objs(text)

        assert len(result) == 3  # Before quote, quote, after quote
        assert result[0].has_quotes is False
        assert result[1].has_quotes is True
        assert result[2].has_quotes is False

    def test_multiple_paragraphs(self):
        """Test multiple paragraphs separated by newlines."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = get_chapter_objs(text)

        assert len(result) == 3
        assert result[0].text == "First paragraph."
        assert result[1].text == "Second paragraph."
        assert result[2].text == "Third paragraph."

    def test_line_numbering(self):
        """Test that line numbers increment correctly."""
        text = "Line one.\n\nLine two.\n\nLine three."
        result = get_chapter_objs(text)

        assert result[0].line_num == 1
        assert result[1].line_num == 2
        assert result[2].line_num == 3

    def test_quoted_lines_at_start(self):
        """Test quoted text at the start of paragraph."""
        text = '"Hello," he said. "How are you?"'
        result = get_chapter_objs(text)

        assert len(result) >= 2
        # First quoted line
        assert result[0].has_quotes is True
        assert '"Hello,' in result[0].text

    def test_text_cleanup_in_paragraphs(self):
        """Test that text is cleaned up in paragraphs."""
        text = "Hello   world"
        result = get_chapter_objs(text)

        assert len(result) == 1
        assert result[0].text == "Hello world"  # Double space reduced

    def test_mixed_content(self):
        """Test mixed quoted and unquoted content."""
        text = "Narrator text.\n\n\"Quote one.\"\n\nMore narrator.\n\n\"Quote two.\""
        result = get_chapter_objs(text)

        # Should have: narrator, quote, narrator, quote
        quote_count = sum(1 for r in result if r.has_quotes)
        non_quote_count = sum(1 for r in result if not r.has_quotes)

        assert quote_count >= 2
        assert non_quote_count >= 2


@pytest.fixture
def simple_chapter_text():
    """Simple chapter text for integration testing."""
    return """The old man sat by the fire.

"Tell me about the sea," the boy requested.

"It holds many secrets," the old man replied.

"Tell me one," the boy insisted."""