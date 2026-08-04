"""Tests for generate_audiobook_from_chapters() in audiobook_generator.py."""

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

    def test_empty_chapters_returns_zero(self, temp_dir):
        """Empty chapters list should return (message, 0)."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with patch_audiobook_pipeline() as mock_tts:
            result = generate_audiobook_from_chapters(
                chapters=[],
                chapter_maps={},
                voices_map={},
                output_dir=str(temp_dir),
            )

        assert result == ("Generated 0 chapters successfully.", 0)

    def test_max_chapters_limit(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """max_chapters should limit the number of chapters processed."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with patch_audiobook_pipeline() as mock_tts:
            result = generate_audiobook_from_chapters(
                chapters=sample_chapters,
                chapter_maps=sample_chapter_maps,
                voices_map=sample_voices_map,
                output_dir=str(temp_dir),
                max_chapters=1,
            )

        assert result[1] == 1

    def test_skip_existing_mp3(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Should skip chapters that already have MP3 files."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with patch_audiobook_pipeline(exists=True) as mock_tts:
            result = generate_audiobook_from_chapters(
                chapters=sample_chapters,
                chapter_maps=sample_chapter_maps,
                voices_map=sample_voices_map,
                output_dir=str(temp_dir),
            )

        # Both chapters should be skipped (MP3 already exists)
        assert result == ("Generated 2 chapters successfully.", 2)


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


class TestGenerateAudiobookFromChaptersDebugTTS:
    """Tests for debug TTS mode."""

    def test_debug_tts_does_not_generate(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """debug_tts should print instead of generate."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with patch_audiobook_pipeline() as mock_tts:
            result = generate_audiobook_from_chapters(
                chapters=sample_chapters,
                chapter_maps=sample_chapter_maps,
                voices_map=sample_voices_map,
                output_dir=str(temp_dir),
                debug_tts=True,
            )

        # When debug_tts is True, generate_tts_for_line should not be called
        mock_tts.assert_not_called()
