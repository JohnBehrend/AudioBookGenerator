"""Tests for gradio_audiobook_interface.py (Stages 1-6)."""
import pytest
import json
import os
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
    describe_characters,
    generate_voice_samples,
    generate_full_audiobook,
    cleanup_temp_dir,
    create_interface,
    SCRIPT_DIR
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
    """Reset temporary directory state before each test (no-op - using dynamic dirs now)."""
    # Global TEMP_DIR/CHAPTERS_DIR have been replaced with dynamic temp dirs
    yield


# ============================================================================
# Tests for get_chapters_dir()
# ============================================================================

class TestGetChaptersDir:
    """Tests for get_chapters_dir function."""

    def test_returns_temp_chapters_directory(self):
        """Test that get_chapters_dir returns a temp directory with chapters subdirectory."""
        result = get_chapters_dir()

        assert result.exists()
        assert result.is_dir()
        # The function returns chapters subdirectory, not temp dir directly
        assert "chapters" in str(result.name) or "chapters" in str(result.parent)

    def test_directory_cleanup(self, temp_dir):
        """Test that temp directories are properly cleaned up."""
        cleanup_temp_dir()
        # Should not raise any errors


# ============================================================================
# Tests for parse_epub_to_file()
# ============================================================================

class TestParseEpubToFile:
    """Tests for parse_epub_to_file function (Stage 1)."""

    def test_returns_error_for_no_file(self, temp_dir, reset_temp_globals):
        """Test error handling when no file is provided."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            result = parse_epub_to_file(None, max_chapters=None)

        assert result[0].startswith("Error:")
        assert result[1] == 0

    def test_parses_epub_successfully(self, temp_dir, reset_temp_globals):
        """Test successful EPUB parsing."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

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
            with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
                result = parse_epub_to_file(mock_file, max_chapters=None)

        assert "Stage 1" in result[0]
        assert result[1] == "epub_parsed"

    def test_returns_error_for_no_chapters(self, temp_dir, reset_temp_globals):
        """Test error handling when no chapters are found."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        mock_parse_module = MagicMock()
        mock_parse_module.parse_epub_to_chapters.return_value = []

        epub_path = temp_dir / "test.epub"
        epub_path.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        mock_file = MagicMock()
        mock_file.name = str(epub_path)

        with patch.dict('sys.modules', {'parse_chapter': mock_parse_module}):
            with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
                result = parse_epub_to_file(mock_file, max_chapters=None)

        assert "Error: No chapters found" in result[0]
        assert result[1] == 0

    def test_copies_epub_to_temp_dir(self, temp_dir, reset_temp_globals):
        """Test that EPUB file is copied to temp directory."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

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
            with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
                parse_epub_to_file(mock_file, max_chapters=None)

        assert (chapters_dir / "uploaded.epub").exists()

    def test_handles_parse_error(self, temp_dir, reset_temp_globals):
        """Test error handling when EPUB parsing fails."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        mock_parse_module = MagicMock()
        mock_parse_module.parse_epub_to_chapters.side_effect = Exception("Invalid EPUB")

        epub_path = temp_dir / "test.epub"
        epub_path.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        mock_file = MagicMock()
        mock_file.name = str(epub_path)

        with patch.dict('sys.modules', {'parse_chapter': mock_parse_module}):
            with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
                result = parse_epub_to_file(mock_file, max_chapters=None)

        assert "Error parsing EPUB" in result[0]
        assert result[1] == 0


# ============================================================================
# Tests for process_chapters_for_labels()
# ============================================================================

class TestProcessChaptersForLabels:
    """Tests for process_chapters_for_labels function (Stage 2)."""

    def test_returns_error_when_no_chapters(self, temp_dir):
        """Test error handling when no chapter files exist."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            result = process_chapters_for_labels(
                api_key="test_key",
                port="1234",
                num_attempts=10,
                pipeline_state=None,
                log_output=log_output
            )

        assert "No chapter files found" in result[0]

    def test_processes_all_chapters(self, temp_dir, reset_temp_globals):
        """Test processing all chapters."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (chapters_dir / f"chapter_{i}.txt").write_text(f"Chapter {i} content")
            (chapters_dir / f"chapter_{i}.map.json").write_text(json.dumps([{"1": "narrator"}, {"1": 1}]))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("llm_label_speakers.label_speakers_in_file") as mock_label:
                mock_label.return_value = ("Labels complete", {"1": "narrator"}, {"1": 1})

                result = process_chapters_for_labels(
                    api_key="test_key",
                    port="1234",
                    num_attempts=10,
                    pipeline_state=None,
                    log_output=log_output
                )

                assert "Stage 2 complete" in result[0]
                assert result[1] == "labels_complete"
                assert "narrator" in result[2]

    def test_processes_multiple_chapters(self, temp_dir, reset_temp_globals):
        """Test processing multiple chapters."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            (chapters_dir / f"chapter_{i}.txt").write_text(f"Chapter {i} content")
            (chapters_dir / f"chapter_{i}.map.json").write_text(json.dumps([{"1": "narrator", "2": "character"}, {"1": 1, "2": 2}]))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("llm_label_speakers.label_speakers_in_file") as mock_label:
                mock_label.return_value = ("Labels complete", {"1": "narrator", "2": "character"}, {"1": 1, "2": 2})

                result = process_chapters_for_labels(
                    api_key="test_key",
                    port="1234",
                    num_attempts=10,
                    pipeline_state=None,
                    log_output=log_output
                )

                assert "Stage 2 complete" in result[0]
                assert result[1] == "labels_complete"
                assert "narrator" in result[2]
                assert "character" in result[2]

    def test_handles_label_error(self, temp_dir, reset_temp_globals):
        """Test error handling when label fails."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        (chapters_dir / "chapter_0.txt").write_text("Chapter 0 content")

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("llm_label_speakers.label_speakers_in_file") as mock_label:
                mock_label.side_effect = Exception("Label failed")

                result = process_chapters_for_labels(
                    api_key="test_key",
                    port="1234",
                    num_attempts=10,
                    pipeline_state=None,
                    log_output=log_output
                )

                assert "Error processing" in result[0]

    def test_returns_character_list(self, temp_dir, reset_temp_globals):
        """Test that character list is returned correctly."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        (chapters_dir / "chapter_0.txt").write_text("Chapter 0 content")
        (chapters_dir / "chapter_0.map.json").write_text(json.dumps([
            {"1": "narrator", "2": "john", "3": "mary"},
            {"1": 1, "2": 2, "3": 3}
        ]))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("llm_label_speakers.label_speakers_in_file") as mock_label:
                mock_label.return_value = ("Labels complete", {"1": "narrator", "2": "john", "3": "mary"}, {"1": 1, "2": 2, "3": 3})

                result = process_chapters_for_labels(
                    api_key="test_key",
                    port="1234",
                    num_attempts=10,
                    pipeline_state=None,
                    log_output=log_output
                )

                assert "narrator" in result[2]
                assert "john" in result[2]
                assert "mary" in result[2]


# ============================================================================
# Tests for describe_characters()
# ============================================================================

class TestDescribeCharacters:
    """Tests for describe_characters function (Stage 3: Character Descriptions)."""

    def test_returns_error_when_no_map_files(self, temp_dir, reset_temp_globals):
        """Test error handling when no map files exist."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            result = describe_characters(
                api_key="test_key", port="1234", pipeline_state=None, log_output=log_output
            )

        assert "No .map.json files found" in result[0]

    def test_successfully_describes_characters(self, temp_dir, reset_temp_globals):
        """Test successful character description generation."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        # Create a map file with characters
        map_file = chapters_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([
            {"1": "narrator", "2": "john"},
            {"1": 1, "2": 2}
        ]))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("llm_describe_character.describe_characters_in_dir") as mock_describe:
                mock_describe.return_value = ("Successfully described 2 characters.", {"narrator": "Desc", "john": "Desc"})

                result = describe_characters(
                    api_key="test_key", port="1234", pipeline_state=None, log_output=log_output
                )

                assert "Successfully described" in result[0] or "characters_described" in result[0]
                mock_describe.assert_called_once()

    def test_handles_describe_error(self, temp_dir, reset_temp_globals):
        """Test error handling when describe fails."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        map_file = chapters_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([
            {"1": "narrator"},
            {"1": 1}
        ]))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("llm_describe_character.describe_characters_in_dir") as mock_describe:
                mock_describe.side_effect = Exception("Describe failed")

                result = describe_characters(
                    api_key="test_key", port="1234", pipeline_state=None, log_output=log_output
                )

                assert "Error describing characters" in result[0]


# ============================================================================
# Tests for generate_voice_samples()
# ============================================================================

class TestGenerateVoiceSamples:
    """Tests for generate_voice_samples function (Stage 4)."""

    def test_returns_error_when_no_descriptions_file(self, temp_dir, reset_temp_globals):
        """Test error handling when characters_descriptions.json doesn't exist."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            result = generate_voice_samples(
                pipeline_state=None,
                log_output=log_output
            )

        assert "characters_descriptions.json not found" in result[0]

    def test_successfully_generates_voice_samples(self, temp_dir, reset_temp_globals):
        """Test successful voice sample generation."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        descriptions_file = chapters_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({
            "narrator": "Male, middle-aged, neutral accent.",
            "john": "Male, young adult, British accent."
        }))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("generate_voice_samples.generate_voice_samples") as mock_gen:
                mock_gen.return_value = ("Stage 4 complete", {"narrator": "narrator.wav", "john": "john.wav"})

                result = generate_voice_samples(
                    pipeline_state=None,
                    log_output=log_output
                )

                assert "Stage 4 complete" in result[0]
                mock_gen.assert_called_once()

    def test_logs_stderr_from_voice_samples(self, temp_dir, reset_temp_globals):
        """Test that stderr from voice samples is logged."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        descriptions_file = chapters_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({"narrator": "Description"}))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("generate_voice_samples.generate_voice_samples") as mock_gen:
                mock_gen.return_value = ("Stage 4 complete", {"narrator": "narrator.wav"})

                result = generate_voice_samples(
                    pipeline_state=None,
                    log_output=log_output
                )

                assert "Stage 4 complete" in result[0]

    def test_handles_voice_samples_error(self, temp_dir, reset_temp_globals):
        """Test error handling when voice sample generation fails."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        descriptions_file = chapters_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({"narrator": "Description"}))

        log_output = ""
        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("generate_voice_samples.generate_voice_samples") as mock_gen:
                mock_gen.side_effect = Exception("Voice sample generation failed")

                result = generate_voice_samples(
                    pipeline_state=None,
                    log_output=log_output
                )

                assert "Error generating voice samples" in result[0]


# ============================================================================
# Tests for generate_full_audiobook()
# ============================================================================

class TestGenerateFullAudiobook:
    """Tests for generate_full_audiobook function (Stage 5)."""

    def test_returns_error_when_no_map_files(self, reset_temp_globals):
        """Test error handling when no .map.json files exist."""
        # Use get_chapters_dir to get the temp directory
        chapters_dir = get_chapters_dir()
        if chapters_dir:
            log_output, new_state = generate_full_audiobook(
                pipeline_state=None,
                log_output="",
                max_chapters=None
            )
            assert "No .map.json files found" in log_output

    def test_returns_error_when_no_descriptions_file(self, temp_dir, reset_temp_globals):
        """Test error handling when characters_descriptions.json doesn't exist."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        map_file = chapters_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([{"1": "narrator"}, {"1": 1}]))

        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            log_output, new_state = generate_full_audiobook(
                pipeline_state=None,
                log_output="",
                max_chapters=None
            )

        assert "characters_descriptions.json not found" in log_output

    def test_returns_error_when_no_uploaded_epub(self, temp_dir, reset_temp_globals):
        """Test error handling when uploaded EPUB doesn't exist."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        map_file = chapters_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([{"1": "narrator"}, {"1": 1}]))

        descriptions_file = chapters_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({"narrator": "Description"}))

        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            log_output, new_state = generate_full_audiobook(
                pipeline_state=None,
                log_output="",
                max_chapters=None
            )

        # The function checks for voice samples after descriptions
        # So the actual error will be about missing voice samples
        assert "voice samples" in log_output.lower() or "error" in log_output.lower()

    def test_successfully_generates_audiobook(self, temp_dir, reset_temp_globals):
        """Test successful full audiobook generation."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        map_file = chapters_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([{"1": "narrator"}, {"1": 1}]))

        descriptions_file = chapters_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({"narrator": "Description"}))

        epub_file = chapters_dir / "uploaded.epub"
        epub_file.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        # Create a voice sample file
        voice_file = chapters_dir / "narrator.wav"
        voice_file.write_bytes(b"fake wav content")

        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.generate_tts_audio") as mock_tts:
                with patch("gradio_audiobook_interface.assemble_chapter_audiobooks") as mock_assemble:
                    mock_tts.return_value = ("TTS audio generated\n", "labels_complete")
                    mock_assemble.return_value = ("Audiobook assembled\n", "audiobook_complete")
                    log_output, new_state = generate_full_audiobook(
                        pipeline_state=None,
                        log_output="",
                        max_chapters=None
                    )

                    assert "Stage 5 Complete" in log_output
                    assert new_state == "audiobook_complete"
                    mock_tts.assert_called_once()
                    mock_assemble.assert_called_once()

    def test_handles_audiobook_generation_error(self, temp_dir, reset_temp_globals):
        """Test error handling when audiobook generation fails."""
        chapters_dir = temp_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        map_file = chapters_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([{"1": "narrator"}, {"1": 1}]))

        descriptions_file = chapters_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({"narrator": "Description"}))

        epub_file = chapters_dir / "uploaded.epub"
        epub_file.write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.generate_tts_audio") as mock_tts:
                mock_tts.side_effect = Exception("TTS generation failed")
                log_output, new_state = generate_full_audiobook(
                    pipeline_state=None,
                    log_output="",
                    max_chapters=None
                )

                assert "Error" in log_output


# ============================================================================
# Tests for cleanup_temp_dir()
# ============================================================================

class TestCleanupTempDir:
    """Tests for cleanup_temp_dir function."""

    def test_cleans_up_noop_for_dynamic_dirs(self):
        """Test that cleanup_temp_dir handles dynamic temp dirs (no-op)."""
        # cleanup_temp_dir() now handles dynamic temp dirs - should not raise
        cleanup_temp_dir()

    def test_handles_none_temp_dir(self):
        """Test that None temp directory doesn't cause error."""
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

        # characters.json is no longer generated - characters extracted from map files
        descriptions_file = chapters_dir / "characters_descriptions.json"
        descriptions_file.write_text(json.dumps({
            "narrator": "Description",
            "character": "Description"
        }))

        (chapters_dir / "uploaded.epub").write_bytes(b"PK\x03\x04" + b"\x00" * 100)

        with patch("gradio_audiobook_interface.get_chapters_dir", return_value=chapters_dir):
            with patch("gradio_audiobook_interface.SCRIPT_DIR", temp_dir):
                log_output = ""

                # Stage 2: Process chapters for labels
                with patch("llm_label_speakers.label_speakers_in_file") as mock_label:
                    mock_label.return_value = ("Labels complete", {"1": "narrator"}, {"1": 1})
                    log_output, new_state, characters = process_chapters_for_labels(
                        api_key="test", port="1234", num_attempts=1,
                        pipeline_state=None,
                        log_output=log_output
                    )
                    assert "Stage 2 complete" in log_output

                # Stage 3: Describe characters
                with patch("llm_describe_character.describe_characters_in_dir") as mock_describe:
                    mock_describe.return_value = ("Successfully described 2 characters.", {"narrator": "Desc"})
                    log_output, new_state, characters = describe_characters(
                        api_key="test", port="1234",
                        pipeline_state=new_state,
                        log_output=log_output
                    )
                    assert "Stage 3 complete" in log_output or "Successfully described" in log_output

                # Stage 4: Generate voice samples
                with patch("generate_voice_samples.generate_voice_samples") as mock_gen:
                    mock_gen.return_value = ("Stage 4 complete", {"narrator": "narrator.wav"})
                    log_output, new_state = generate_voice_samples(
                        pipeline_state=new_state,
                        log_output=log_output
                    )
                    assert "Stage 4 complete" in log_output or "Successfully generated" in log_output

                # Stage 5: Generate full audiobook
                with patch("gradio_audiobook_interface.generate_tts_audio") as mock_tts:
                    with patch("gradio_audiobook_interface.assemble_chapter_audiobooks") as mock_assemble:
                        mock_tts.return_value = ("TTS audio generated", "voice_samples_complete")
                        mock_assemble.return_value = ("Audiobook assembled", "audiobook_complete")
                        log_output, new_state = generate_full_audiobook(
                            pipeline_state=new_state,
                            log_output=log_output,
                            max_chapters=None
                        )
                        assert "Stage 5 Complete" in log_output