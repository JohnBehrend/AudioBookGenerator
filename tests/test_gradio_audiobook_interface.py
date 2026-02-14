"""Tests for gradio_audiobook_interface.py (Stages 1-6)."""
import pytest
import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add project root to path
import sys
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from gradio_audiobook_interface import (
    get_chapters_dir,
    parse_epub_to_file,
    process_chapters_for_labels,
    analyze_chapters,
    describe_characters,
    generate_voice_samples,
    generate_full_audiobook,
    cleanup_temp_dir,
    create_interface,
    SCRIPT_DIR,
    TEMP_DIR,
    CHAPTERS_DIR
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_epub_file(temp_dir):
    """Create a mock EPUB file for testing."""
    epub_path = temp_dir / "test_book.epub"
    epub_path.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
    return epub_path


@pytest.fixture
def mock_parsed_chapter(temp_dir):
    """Create a mock chapter text file."""
    chapters_dir = temp_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    chapter_file = chapters_dir / "chapter_0.txt"
    chapter_file.write_text("""Chapter One

The morning sun shone brightly.

"Good morning, everyone!" shouted the captain.
"Is everyone ready?" he asked.

" Yes, sir!" came the reply.
The ship set sail.
""")
    return str(chapter_file)


@pytest.fixture
def mock_map_file(temp_dir):
    """Create a mock chapter map file."""
    chapters_dir = temp_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    map_file = chapters_dir / "chapter_0.map.json"
    content = [
        {"1": "narrator", "2": "captain"},
        {"2": 2, "4": 2, "6": 1, "8": 1}
    ]
    map_file.write_text(json.dumps(content))
    return str(map_file)


@pytest.fixture
def mock_characters_json(temp_dir):
    """Create a mock characters.json file."""
    chapters_dir = temp_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    characters_file = chapters_dir / "characters.json"
    content = {"characters": ["narrator", "captain", "first_mate"]}
    characters_file.write_text(json.dumps(content))
    return str(characters_file)


@pytest.fixture
def mock_characters_descriptions_json(temp_dir):
    """Create a mock characters_descriptions.json file."""
    characters_file = temp_dir / "characters_descriptions.json"
    content = {
        "narrator": "Male, middle-aged, neutral accent.",
        "captain": "Male, older, authoritative voice.",
        "first_mate": "Male, young adult, nervous tone."
    }
    characters_file.write_text(json.dumps(content, indent=2))
    return str(characters_file)


@pytest.fixture
def reset_temp_globals():
    """Reset temporary directory globals before each test."""
    global TEMP_DIR, CHAPTERS_DIR
    old_temp = TEMP_DIR
    old_chapters = CHAPTERS_DIR
    TEMP_DIR = None
    CHAPTERS_DIR = None
    yield
    TEMP_DIR = old_temp
    CHAPTERS_DIR = old_chapters


# ============================================================================
# Tests for get_chapters_dir()
# ============================================================================

class TestGetChaptersDir:
    """Tests for get_chapters_dir function."""

    def test_creates_directory_if_none_exists(self, monkeypatch):
        """Test that get_chapters_dir creates a directory when none exists."""
        monkeypatch.setattr("gradio_audiobook_interface.TEMP_DIR", None)
        monkeypatch.setattr("gradio_audiobook_interface.CHAPTERS_DIR", None)

        result = get_chapters_dir()

        assert result.exists()
        assert result.is_dir()

    def test_returns_existing_directory(self, monkeypatch):
        """Test that get_chapters_dir returns existing directory."""
        monkeypatch.setattr("gradio_audiobook_interface.TEMP_DIR", None)
        monkeypatch.setattr("gradio_audiobook_interface.CHAPTERS_DIR", None)

        result1 = get_chapters_dir()
        result2 = get_chapters_dir()

        assert result1 == result2

    def test_creates_unique_directory_per_session(self, monkeypatch):
        """Test that each call creates a unique temp directory."""
        monkeypatch.setattr("gradio_audiobook_interface.TEMP_DIR", None)
        monkeypatch.setattr("gradio_audiobook_interface.CHAPTERS_DIR", None)

        result1 = get_chapters_dir()
        monkeypatch.setattr("gradio_audiobook_interface.TEMP_DIR", None)
        monkeypatch.setattr("gradio_audiobook_interface.CHAPTERS_DIR", None)
        result2 = get_chapters_dir()

        assert result1 != result2


# ============================================================================
# Tests for parse_epub_to_file()
# ============================================================================

class TestParseEpubToFile:
    """Tests for parse_epub_to_file function (Stage 1)."""

    def test_returns_error_for_no_file(self, reset_temp_globals):
        """Test error handling when no file is provided."""
        result = parse_epub_to_file(None)

        assert result[0].startswith("Error:")
        assert result[1] == 0
        assert result[2] == []

    def test_parses_epub_successfully(self, temp_dir, reset_temp_globals):
        """Test successful EPUB parsing."""
        mock_chapter_obj = MagicMock()
        mock_chapter_obj.line_num = 1
        mock_chapter_obj.text = "Test line"
        mock_chapter_obj.has_quotes = False

        mock_parse_module = MagicMock()
        mock_parse_module.parse_epub_to_chapters.return_value = [
            [mock_chapter_obj, mock_chapter_obj],
            [mock_chapter_obj]
        ]

        epub_path = temp_dir / "test.epub"
        epub_path.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        mock_file = MagicMock()
        mock_file.name = str(epub_path)

        with patch.dict('sys.modules', {'parse_chapter': mock_parse_module}):
            result = parse_epub_to_file(mock_file)

        assert "Successfully parsed" in result[0]
        assert result[1] == 2
        assert len(result[2]) == 2

        for chapter_file in result[2]:
            assert os.path.exists(chapter_file)

    def test_returns_error_for_no_chapters(self, temp_dir, reset_temp_globals):
        """Test error handling when no chapters are found."""
        mock_parse_module = MagicMock()
        mock_parse_module.parse_epub_to_chapters.return_value = []

        epub_path = temp_dir / "test.epub"
        epub_path.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        mock_file = MagicMock()
        mock_file.name = str(epub_path)

        with patch.dict('sys.modules', {'parse_chapter': mock_parse_module}):
            result = parse_epub_to_file(mock_file)

        assert "Error: No chapters found" in result[0]
        assert result[1] == 0

    def test_copies_epub_to_temp_dir(self, temp_dir, reset_temp_globals):
        """Test that EPUB file is copied to temp directory."""
        mock_chapter_obj = MagicMock()
        mock_chapter_obj.line_num = 1
        mock_chapter_obj.text = "Test"
        mock_chapter_obj.has_quotes = False

        mock_parse_module = MagicMock()
        mock_parse_module.parse_epub_to_chapters.return_value = [[mock_chapter_obj]]

        epub_path = temp_dir / "test.epub"
        epub_path.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        mock_file = MagicMock()
        mock_file.name = str(epub_path)

        with patch.dict('sys.modules', {'parse_chapter': mock_parse_module}):
            parse_epub_to_file(mock_file)

        chapters_dir = get_chapters_dir()
        assert (chapters_dir / "uploaded.epub").exists()

    def test_handles_parse_error(self, temp_dir, reset_temp_globals):
        """Test error handling when EPUB parsing fails."""
        mock_parse_module = MagicMock()
        mock_parse_module.parse_epub_to_chapters.side_effect = Exception("Invalid EPUB")

        epub_path = temp_dir / "test.epub"
        epub_path.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        mock_file = MagicMock()
        mock_file.name = str(epub_path)

        with patch.dict('sys.modules', {'parse_chapter': mock_parse_module}):
            result = parse_epub_to_file(mock_file)

        assert "Error parsing EPUB" in result[0]
        assert result[1] == 0


# ============================================================================
# Tests for process_chapters_for_labels()
# ============================================================================

class TestProcessChaptersForLabels:
    """Tests for process_chapters_for_labels function (Stage 2)."""

    def test_returns_error_when_no_chapters(self, reset_temp_globals):
        """Test error handling when no chapter files exist."""
        log_output = ""
        result = process_chapters_for_labels(
            api_key="test_key",
            port="1234",
            num_attempts=10,
            use_all_chapters=True,
            chapter_range=[0, 5],
            log_output=log_output
        )

        assert "No chapter files found" in result

    def test_processes_all_chapters_when_flag_set(self, temp_dir, reset_temp_globals):
        """Test processing all chapters when use_all_chapters is True."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (chapters_dir / f"chapter_{i}.txt").write_text(f"Chapter {i} content")

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="Processing complete", stderr="")

                result = process_chapters_for_labels(
                    api_key="test_key",
                    port="1234",
                    num_attempts=10,
                    use_all_chapters=True,
                    chapter_range=[0, 5],
                    log_output=log_output
                )

                assert mock_run.call_count == 3

    def test_processes_selected_chapter_range(self, temp_dir, reset_temp_globals):
        """Test processing only selected chapter range."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            (chapters_dir / f"chapter_{i}.txt").write_text(f"Chapter {i} content")

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="OK", stderr="")

                result = process_chapters_for_labels(
                    api_key="test_key",
                    port="1234",
                    num_attempts=10,
                    use_all_chapters=False,
                    chapter_range=[1, 3],
                    log_output=log_output
                )

                assert mock_run.call_count == 3

    def test_handles_single_chapter_int_value(self, temp_dir, reset_temp_globals):
        """Test handling of single chapter integer value."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        (chapters_dir / "chapter_0.txt").write_text("Chapter 0 content")

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="OK", stderr="")

                result = process_chapters_for_labels(
                    api_key="test_key",
                    port="1234",
                    num_attempts=10,
                    use_all_chapters=False,
                    chapter_range=0,
                    log_output=log_output
                )

                assert mock_run.call_count == 1

    def test_handles_subprocess_timeout(self, temp_dir, reset_temp_globals):
        """Test timeout handling in subprocess."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        (chapters_dir / "chapter_0.txt").write_text("Chapter 0 content")

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=300)

                result = process_chapters_for_labels(
                    api_key="test_key",
                    port="1234",
                    num_attempts=10,
                    use_all_chapters=True,
                    chapter_range=[0, 5],
                    log_output=log_output
                )

                assert "Timeout" in result

    def test_handles_subprocess_error(self, temp_dir, reset_temp_globals):
        """Test error handling in subprocess."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        (chapters_dir / "chapter_0.txt").write_text("Chapter 0 content")

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.side_effect = Exception("Subprocess error")

                result = process_chapters_for_labels(
                    api_key="test_key",
                    port="1234",
                    num_attempts=10,
                    use_all_chapters=True,
                    chapter_range=[0, 5],
                    log_output=log_output
                )

                assert "Error processing" in result

    def test_logs_stderr_output(self, temp_dir, reset_temp_globals):
        """Test that stderr output is logged."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        (chapters_dir / "chapter_0.txt").write_text("Chapter 0 content")

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="OK", stderr="Error occurred")

                result = process_chapters_for_labels(
                    api_key="test_key",
                    port="1234",
                    num_attempts=10,
                    use_all_chapters=True,
                    chapter_range=[0, 5],
                    log_output=log_output
                )

                assert "Errors: Error occurred" in result

    def test_handles_list_tuple_chapter_range(self, temp_dir, reset_temp_globals):
        """Test handling of list/tuple chapter range."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        (chapters_dir / "chapter_0.txt").write_text("Chapter 0 content")

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="OK", stderr="")

                result = process_chapters_for_labels(
                    api_key="test_key",
                    port="1234",
                    num_attempts=10,
                    use_all_chapters=False,
                    chapter_range=(0, 0),
                    log_output=log_output
                )

                assert mock_run.call_count == 1


# ============================================================================
# Tests for analyze_chapters()
# ============================================================================

class TestAnalyzeChapters:
    """Tests for analyze_chapters function (Stage 3)."""

    def test_returns_error_when_no_map_files(self, reset_temp_globals):
        """Test error handling when no .map.json files exist."""
        log_output = ""
        result = analyze_chapters(log_output)

        assert "No .map.json files found" in result

    def test_successfully_analyzes_chapters(self, temp_dir, reset_temp_globals):
        """Test successful chapter analysis."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        for i in range(2):
            map_file = chapters_dir / f"chapter_{i}.map.json"
            content = [
                {"1": "narrator", "2": "character"},
                {"1": 1, "2": 2}
            ]
            map_file.write_text(json.dumps(content))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="Analysis complete\n", stderr="")

                result = analyze_chapters(log_output)

                assert "Stage 3 complete" in result
                mock_run.assert_called_once()

    def test_logs_stderr_from_analysis(self, temp_dir, reset_temp_globals):
        """Test that stderr from analysis is logged."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        map_file = chapters_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([{"1": "narrator"}, {"1": 1}]))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="Analysis complete\n", stderr="Warning: some issues"
                )

                result = analyze_chapters(log_output)

                assert "Errors: Warning: some issues" in result

    def test_handles_analysis_error(self, temp_dir, reset_temp_globals):
        """Test error handling when analysis fails."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        map_file = chapters_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([{"1": "narrator"}, {"1": 1}]))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.side_effect = Exception("Analysis failed")

                result = analyze_chapters(log_output)

                assert "Error analyzing chapters" in result


# ============================================================================
# Tests for describe_characters()
# ============================================================================

class TestDescribeCharacters:
    """Tests for describe_characters function (Stage 4)."""

    def test_returns_error_when_no_characters_json(self, reset_temp_globals):
        """Test error handling when characters.json doesn't exist."""
        log_output = ""
        result = describe_characters(
            api_key="test_key", port="1234", log_output=log_output
        )

        assert "characters.json not found" in result

    def test_successfully_describes_characters(self, temp_dir, reset_temp_globals):
        """Test successful character description generation."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        characters_file = chapters_dir / "characters.json"
        characters_file.write_text(json.dumps({"characters": ["narrator", "john"]}))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="Character descriptions generated\n", stderr=""
                )

                result = describe_characters(
                    api_key="test_key", port="1234", log_output=log_output
                )

                assert "Stage 4 complete" in result
                mock_run.assert_called_once()

    def test_logs_stderr_from_describe(self, temp_dir, reset_temp_globals):
        """Test that stderr from describe is logged."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        characters_file = chapters_dir / "characters.json"
        characters_file.write_text(json.dumps({"characters": ["narrator"]}))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="OK\n", stderr="Warning: low confidence"
                )

                result = describe_characters(
                    api_key="test_key", port="1234", log_output=log_output
                )

                assert "Errors: Warning: low confidence" in result

    def test_handles_describe_error(self, temp_dir, reset_temp_globals):
        """Test error handling when describe fails."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        characters_file = chapters_dir / "characters.json"
        characters_file.write_text(json.dumps({"characters": ["narrator"]}))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.side_effect = Exception("Describe failed")

                result = describe_characters(
                    api_key="test_key", port="1234", log_output=log_output
                )

                assert "Error describing characters" in result


# ============================================================================
# Tests for generate_voice_samples()
# ============================================================================

class TestGenerateVoiceSamples:
    """Tests for generate_voice_samples function (Stage 5)."""

    def test_returns_error_when_no_descriptions_file(self, temp_dir, reset_temp_globals):
        """Test error handling when characters_descriptions.json doesn't exist."""
        log_output = ""
        with patch("gradio_audiobook_interface.SCRIPT_DIR", temp_dir):
            result = generate_voice_samples(log_output)

        assert "characters_descriptions.json not found" in result

    def test_successfully_generates_voice_samples(self, temp_dir, reset_temp_globals):
        """Test successful voice sample generation."""
        descriptions_file = temp_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({
            "narrator": "Male, middle-aged, neutral accent.",
            "john": "Male, young adult, British accent."
        }))

        log_output = ""
        with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Voice samples generated\n", stderr="")

            with patch("gradio_audiobook_interface.SCRIPT_DIR", temp_dir):
                result = generate_voice_samples(log_output)

                assert "Stage 5 complete" in result
                mock_run.assert_called_once()

    def test_logs_stderr_from_voice_samples(self, temp_dir, reset_temp_globals):
        """Test that stderr from voice samples is logged."""
        descriptions_file = temp_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({"narrator": "Description"}))

        log_output = ""
        with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="OK\n", stderr="Warning: some voices failed"
            )

            with patch("gradio_audiobook_interface.SCRIPT_DIR", temp_dir):
                result = generate_voice_samples(log_output)

                assert "Errors: Warning: some voices failed" in result

    def test_handles_voice_samples_error(self, temp_dir, reset_temp_globals):
        """Test error handling when voice sample generation fails."""
        descriptions_file = temp_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({"narrator": "Description"}))

        log_output = ""
        with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Voice sample generation failed")

            with patch("gradio_audiobook_interface.SCRIPT_DIR", temp_dir):
                result = generate_voice_samples(log_output)

                assert "Error generating voice samples" in result


# ============================================================================
# Tests for generate_full_audiobook()
# ============================================================================

class TestGenerateFullAudiobook:
    """Tests for generate_full_audiobook function (Stage 6)."""

    def test_returns_error_when_no_map_files(self, reset_temp_globals):
        """Test error handling when no .map.json files exist."""
        log_output = ""
        result = generate_full_audiobook(log_output)

        assert "No .map.json files found" in result

    def test_returns_error_when_no_descriptions_file(self, temp_dir, reset_temp_globals):
        """Test error handling when characters_descriptions.json doesn't exist."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        map_file = chapters_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([{"1": "narrator"}, {"1": 1}]))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.SCRIPT_DIR", temp_dir):
                result = generate_full_audiobook(log_output)

        assert "characters_descriptions.json not found" in result

    def test_returns_error_when_no_uploaded_epub(self, temp_dir, reset_temp_globals):
        """Test error handling when uploaded EPUB doesn't exist."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        map_file = chapters_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([{"1": "narrator"}, {"1": 1}]))

        descriptions_file = temp_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({"narrator": "Description"}))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.SCRIPT_DIR", temp_dir):
                result = generate_full_audiobook(log_output)

        assert "Uploaded EPUB file not found" in result

    def test_successfully_generates_audiobook(self, temp_dir, reset_temp_globals):
        """Test successful full audiobook generation."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        map_file = chapters_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([{"1": "narrator"}, {"1": 1}]))

        descriptions_file = temp_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({"narrator": "Description"}))

        epub_file = chapters_dir / "uploaded.epub"
        epub_file.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="Audiobook generated\n", stderr="")

                with patch("gradio_audiobook_interface.SCRIPT_DIR", temp_dir):
                    result = generate_full_audiobook(log_output)

                    assert "Stage 6 complete" in result
                    mock_run.assert_called_once()

    def test_logs_stderr_from_audiobook_generation(self, temp_dir, reset_temp_globals):
        """Test that stderr from audiobook generation is logged."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        map_file = chapters_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([{"1": "narrator"}, {"1": 1}]))

        descriptions_file = temp_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({"narrator": "Description"}))

        epub_file = chapters_dir / "uploaded.epub"
        epub_file.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="OK\n", stderr="Warning: some chapters skipped"
                )

                with patch("gradio_audiobook_interface.SCRIPT_DIR", temp_dir):
                    result = generate_full_audiobook(log_output)

                    assert "Errors: Warning: some chapters skipped" in result

    def test_handles_audiobook_generation_error(self, temp_dir, reset_temp_globals):
        """Test error handling when audiobook generation fails."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        map_file = chapters_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([{"1": "narrator"}, {"1": 1}]))

        descriptions_file = temp_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({"narrator": "Description"}))

        epub_file = chapters_dir / "uploaded.epub"
        epub_file.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.side_effect = Exception("Audiobook generation failed")

                with patch("gradio_audiobook_interface.SCRIPT_DIR", temp_dir):
                    result = generate_full_audiobook(log_output)

                    assert "Error generating audiobook" in result


# ============================================================================
# Tests for cleanup_temp_dir()
# ============================================================================

class TestCleanupTempDir:
    """Tests for cleanup_temp_dir function."""

    def test_cleans_up_temp_directory(self, monkeypatch):
        """Test that temp directory is cleaned up."""
        test_temp_dir = tempfile.mkdtemp()
        test_chapters_dir = Path(test_temp_dir) / "chapters"
        test_chapters_dir.mkdir()

        monkeypatch.setattr("gradio_audiobook_interface.TEMP_DIR", test_temp_dir)
        monkeypatch.setattr("gradio_audiobook_interface.CHAPTERS_DIR", test_chapters_dir)

        cleanup_temp_dir()

        assert not os.path.exists(test_temp_dir)
        assert getattr(__import__('gradio_audiobook_interface'), 'TEMP_DIR') is None

    def test_handles_none_temp_dir(self):
        """Test that None temp directory doesn't cause error."""
        cleanup_temp_dir()

    def test_handles_missing_temp_directory(self, monkeypatch):
        """Test that missing temp directory doesn't cause error."""
        monkeypatch.setattr("gradio_audiobook_interface.TEMP_DIR", "/nonexistent/path")
        monkeypatch.setattr("gradio_audiobook_interface.CHAPTERS_DIR", None)

        cleanup_temp_dir()


# ============================================================================
# Tests for create_interface()
# ============================================================================

class TestCreateInterface:
    """Tests for create_interface function."""

    def test_creates_interface(self):
        """Test that create_interface function exists and returns a result."""
        with patch("gradio_audiobook_interface.gr") as mock_gr:
            mock_blocks_instance = MagicMock()
            mock_accordion_instance = MagicMock()
            mock_accordion_instance.__enter__ = MagicMock(return_value=None)
            mock_accordion_instance.__exit__ = MagicMock(return_value=None)
            mock_row_instance = MagicMock()
            mock_row_instance.__enter__ = MagicMock(return_value=None)
            mock_row_instance.__exit__ = MagicMock(return_value=None)

            mock_gr.Blocks.return_value.__enter__ = MagicMock(return_value=mock_blocks_instance)
            mock_gr.Blocks.return_value.__exit__ = MagicMock(return_value=None)
            mock_gr.Blocks.return_value.launch = MagicMock()
            mock_gr.Accordion.return_value = mock_accordion_instance
            mock_gr.Row.return_value = mock_row_instance
            mock_gr.Textbox.return_value = MagicMock()
            mock_gr.Slider.return_value = MagicMock()
            mock_gr.Checkbox.return_value = MagicMock()
            mock_gr.File.return_value = MagicMock()
            mock_gr.Button.return_value = MagicMock()
            mock_gr.Number.return_value = MagicMock()
            mock_gr.Markdown.return_value = MagicMock()

            result = create_interface()
            assert result is not None

    def test_interface_has_all_stages(self):
        """Test that create_interface function exists and has proper structure."""
        with patch("gradio_audiobook_interface.gr") as mock_gr:
            mock_blocks_instance = MagicMock()
            mock_accordion_instance = MagicMock()
            mock_accordion_instance.__enter__ = MagicMock(return_value=None)
            mock_accordion_instance.__exit__ = MagicMock(return_value=None)
            mock_row_instance = MagicMock()
            mock_row_instance.__enter__ = MagicMock(return_value=None)
            mock_row_instance.__exit__ = MagicMock(return_value=None)

            mock_gr.Blocks.return_value.__enter__ = MagicMock(return_value=mock_blocks_instance)
            mock_gr.Blocks.return_value.__exit__ = MagicMock(return_value=None)
            mock_gr.Blocks.return_value.launch = MagicMock()
            mock_gr.Accordion.return_value = mock_accordion_instance
            mock_gr.Row.return_value = mock_row_instance
            mock_gr.Textbox.return_value = MagicMock()
            mock_gr.Slider.return_value = MagicMock()
            mock_gr.Checkbox.return_value = MagicMock()
            mock_gr.File.return_value = MagicMock()
            mock_gr.Button.return_value = MagicMock()
            mock_gr.Number.return_value = MagicMock()
            mock_gr.Markdown.return_value = MagicMock()

            result = create_interface()
            assert result is not None

    def test_interface_has_api_configuration(self):
        """Test that create_interface function has API config components."""
        with patch("gradio_audiobook_interface.gr") as mock_gr:
            mock_blocks_instance = MagicMock()
            mock_accordion_instance = MagicMock()
            mock_accordion_instance.__enter__ = MagicMock(return_value=None)
            mock_accordion_instance.__exit__ = MagicMock(return_value=None)
            mock_row_instance = MagicMock()
            mock_row_instance.__enter__ = MagicMock(return_value=None)
            mock_row_instance.__exit__ = MagicMock(return_value=None)

            mock_gr.Blocks.return_value.__enter__ = MagicMock(return_value=mock_blocks_instance)
            mock_gr.Blocks.return_value.__exit__ = MagicMock(return_value=None)
            mock_gr.Blocks.return_value.launch = MagicMock()
            mock_gr.Accordion.return_value = mock_accordion_instance
            mock_gr.Row.return_value = mock_row_instance
            mock_gr.Textbox.return_value = MagicMock()
            mock_gr.Slider.return_value = MagicMock()
            mock_gr.Checkbox.return_value = MagicMock()
            mock_gr.File.return_value = MagicMock()
            mock_gr.Button.return_value = MagicMock()
            mock_gr.Number.return_value = MagicMock()
            mock_gr.Markdown.return_value = MagicMock()

            result = create_interface()
            assert result is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_full_workflow_simulation(self, temp_dir):
        """Test the full workflow end-to-end."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        for i in range(2):
            (chapters_dir / f"chapter_{i}.txt").write_text(f"Chapter {i} content")
            map_file = chapters_dir / f"chapter_{i}.map.json"
            map_file.write_text(json.dumps([
                {"1": "narrator", "2": "character"},
                {"1": 1, "2": 2}
            ]))

        (chapters_dir / "characters.json").write_text(
            json.dumps({"characters": ["narrator", "character"]})
        )

        descriptions_file = temp_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({
            "narrator": "Description",
            "character": "Description"
        }))

        (chapters_dir / "uploaded.epub").write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="OK", stderr="")

                with patch("gradio_audiobook_interface.SCRIPT_DIR", temp_dir):
                    log_output = ""

                    log_output = process_chapters_for_labels(
                        api_key="test", port="1234", num_attempts=1,
                        use_all_chapters=True, chapter_range=[0, 1],
                        log_output=log_output
                    )
                    assert "Stage 2 complete" in log_output

                log_output = analyze_chapters(log_output)
                assert "Stage 3 complete" in log_output

                log_output = describe_characters("test", "1234", log_output)
                assert "Stage 4 complete" in log_output

                log_output = generate_voice_samples(log_output)
                assert "Stage 5 complete" in log_output

                log_output = generate_full_audiobook(log_output)
                assert "Stage 6 complete" in log_output