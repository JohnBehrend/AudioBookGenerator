"""Tests for the pure, non-GUI logic in gradio_ui.

gradio_ui is a thin interactive wrapper over the shared workflow functions
(label_speakers, describe_characters, generate_voice_samples,
generate_audiobook_from_chapters), which are tested elsewhere. These tests
cover only the module's own logic: the PipelineState state machine, file
helpers, and the button-gating tables.
"""

import json
from unittest.mock import MagicMock, patch

from audiobook_generator.gradio_ui import (
    PipelineState,
    parse_epub_to_file,
    get_all_character_wav_files,
    create_or_get_pipeline_state,
    BUTTON_STATES,
    STATE_LABELS,
)
from audiobook_generator.parse_chapter import ChapterObj


class TestPipelineStateMachine:
    """The 5-stage state machine driven by files present on disk."""

    def _state(self, tmp):
        return PipelineState(str(tmp), temp_dir=str(tmp))

    def test_initial_when_no_files(self, temp_dir):
        assert self._state(temp_dir).get_pipeline_state() == "initial"

    def test_epub_parsed_when_chapter_texts_exist(self, temp_dir):
        (temp_dir / "chapter_0.txt").touch()
        assert self._state(temp_dir).get_pipeline_state() == "epub_parsed"

    def test_labels_complete_when_maps_exist(self, temp_dir):
        (temp_dir / "chapter_0.txt").touch()
        (temp_dir / "chapter_0.map.json").write_text(json.dumps([{}, {}]))
        assert self._state(temp_dir).get_pipeline_state() == "labels_complete"

    def test_characters_described_when_descriptions_exist(self, temp_dir):
        (temp_dir / "chapter_0.txt").touch()
        (temp_dir / "chapter_0.map.json").write_text(json.dumps([{}, {}]))
        (temp_dir / "characters_descriptions.json").write_text(json.dumps({}))
        assert self._state(temp_dir).get_pipeline_state() == "characters_described"

    def test_voice_samples_complete_when_wavs_exist(self, temp_dir):
        (temp_dir / "chapter_0.txt").touch()
        (temp_dir / "chapter_0.map.json").write_text(json.dumps([{}, {}]))
        (temp_dir / "characters_descriptions.json").write_text(json.dumps({}))
        (temp_dir / "jane.wav").touch()
        assert self._state(temp_dir).get_pipeline_state() == "voice_samples_complete"

    def test_audiobook_complete_when_mp3s_exist(self, temp_dir):
        (temp_dir / "chapter_0.txt").touch()
        (temp_dir / "chapter_0.map.json").write_text(json.dumps([{}, {}]))
        (temp_dir / "characters_descriptions.json").write_text(json.dumps({}))
        (temp_dir / "jane.wav").touch()
        (temp_dir / "chapter_0.mp3").touch()
        assert self._state(temp_dir).get_pipeline_state() == "audiobook_complete"


class TestParseEpubToFile:
    def test_parses_and_writes_chapters_via_shared_writer(self, temp_dir, tmp_path_factory):
        """Stage 1 should write chapters with the shared writer, not hand-rolled code."""
        # The uploaded EPUB lives outside the chapters dir (Stage 1 wipes chapters_dir).
        epub_dir = tmp_path_factory.mktemp("epub_upload")
        epub = epub_dir / "book.epub"
        epub.write_text("fake epub")
        epub_file = MagicMock()
        epub_file.name = str(epub)

        chapters = [
            [ChapterObj(False, "Narrator text", 1),
             ChapterObj(True, '"Hi," said Jane.', 2)],
        ]

        with patch("audiobook_generator.gradio_ui.get_chapters_dir", return_value=temp_dir), \
             patch("audiobook_generator.gradio_ui.parse_chapter.parse_epub_to_chapters", return_value=chapters):
            log, state = parse_epub_to_file(epub_file, None, progress=MagicMock())

        assert state is not None
        assert state.pipeline_state == "epub_parsed"
        chapter_file = temp_dir / "chapter_0.txt"
        assert chapter_file.exists()
        content = chapter_file.read_text()
        assert "Line 1: Narrator text" in content
        assert '"Hi," said Jane.' in content

    def test_returns_error_when_no_epub(self, temp_dir):
        log, state = parse_epub_to_file(None, None, progress=MagicMock())
        assert "Error" in log
        assert state is None


class TestGetAllCharacterWavFiles:
    def test_maps_character_wavs(self, temp_dir):
        (temp_dir / "jane.wav").touch()
        (temp_dir / "elizabeth.wav").touch()
        result = get_all_character_wav_files(temp_dir)
        assert "jane" in result
        assert "elizabeth" in result

    def test_excludes_narrator_and_chapter_files(self, temp_dir):
        (temp_dir / "narrator.wav").touch()
        (temp_dir / "chapter_1.wav").touch()
        (temp_dir / "jane.wav").touch()
        result = get_all_character_wav_files(temp_dir)
        assert "jane" in result
        assert "narrator" not in result
        assert "chapter_1" not in result


class TestCreateOrGetPipelineState:
    def test_uses_provided_output_dir_and_engine(self, temp_dir):
        state = create_or_get_pipeline_state(output_dir=str(temp_dir), voice_engine="omni")
        assert state.output_dir == temp_dir
        assert state.voice_engine == "omni"


class TestButtonGating:
    def test_every_state_label_has_gating(self):
        assert set(BUTTON_STATES) == set(STATE_LABELS)

    def test_gating_is_monotonic(self):
        """Later stages must enable at least as many buttons as earlier ones."""
        order = ["epub_parsed", "labels_complete", "characters_described",
                 "voice_samples_complete", "audiobook_complete"]
        counts = [sum(BUTTON_STATES[s]) for s in order]
        assert counts == sorted(counts)
