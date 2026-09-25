"""Tests for generate_audiobook_from_chapters() in audiobook_generator.py.

Basic happy-path/limit scenarios (empty chapters, max_chapters, skip-existing,
debug_tts) are covered by tests/test_integration_pipeline.py; this file pins
behavior those don't cover.
"""

from audiobook_generator.testing import patch_audiobook_pipeline


class TestGenerateAudiobookFromChaptersBasic:
    """Tests for basic functionality."""

    def test_returns_tuple(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Should return (status_message, chapters_processed) tuple."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with patch_audiobook_pipeline() as mock_tts:
            result = generate_audiobook_from_chapters(
                chapters=sample_chapters,
                chapter_maps=sample_chapter_maps,
                voices_map=sample_voices_map,
                output_dir=str(temp_dir),
            )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], int)


class TestGenerateAudiobookFromChaptersSkipCorrectness:
    """Guards the "stale chapter MP3 falsely satisfies the skip check" failure.

    Previously, bogus ``chapter_XX.mp3`` files copied in from a seed (prior)
    book made stage 5 skip real chapters, silently filling the audiobook with
    the wrong book's audio. These tests pin the two invariants that prevent it.
    """

    def test_generates_when_mp3_missing(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Companion to test_skip_existing_mp3: with NO pre-existing mp3, every
        chapter must actually be generated (not skipped)."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        # Voices "exist" (so TTS runs) but no chapter MP3 exists (so nothing is
        # skipped) — the exact condition that must trigger real generation.
        with patch_audiobook_pipeline(exists=lambda p: not str(p).endswith(".mp3")) as mock_tts:
            result = generate_audiobook_from_chapters(
                chapters=sample_chapters,
                chapter_maps=sample_chapter_maps,
                voices_map=sample_voices_map,
                output_dir=str(temp_dir),
            )

        # Both chapters processed/generated (not skipped) and TTS invoked.
        assert result[1] == 2
        mock_tts.assert_called()

    def test_no_chapter_mp3_planted_by_pipeline_helpers(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Invariant: running the pipeline in a fresh output dir must NOT leave
        chapter_*.mp3 files behind (only the real assembly step writes them, and
        it is mocked here). A stray mp3 here would falsely skip a later run."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with patch_audiobook_pipeline(exists=False) as mock_tts:
            generate_audiobook_from_chapters(
                chapters=sample_chapters,
                chapter_maps=sample_chapter_maps,
                voices_map=sample_voices_map,
                output_dir=str(temp_dir),
            )

        assert list(temp_dir.glob("chapter_*.mp3")) == []


class TestGenerateAudiobookFromChaptersVoiceResolution:
    """Tests for voice path resolution."""

    def test_no_voice_path_skips_lines(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Lines without voice files should be skipped with a warning."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with patch_audiobook_pipeline(voice_path=None) as mock_tts:
            result = generate_audiobook_from_chapters(
                chapters=sample_chapters,
                chapter_maps=sample_chapter_maps,
                voices_map=sample_voices_map,
                output_dir=str(temp_dir),
            )

        # Should complete without error, skipping lines without voice files
        assert isinstance(result, tuple)
        assert result[1] == 2
        mock_tts.assert_not_called()
