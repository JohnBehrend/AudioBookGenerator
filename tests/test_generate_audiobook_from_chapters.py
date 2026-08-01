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
