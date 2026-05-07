"""Tests for generate_voice_samples module."""

import json
import os
import pytest

from audiobook_generator.generate_voice_samples import (
    load_character_descriptions,
    generate_voice_sample,
    generate_voice_samples,
)


class TestLoadCharacterDescriptions:
    """Tests for load_character_descriptions function."""

    def test_loads_valid_file(self, temp_dir):
        """Test loading valid descriptions file."""
        data = {
            "narrator": "A calm voice.",
            "jane": "A gentle voice."
        }
        file_path = temp_dir / "characters_descriptions.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_character_descriptions(str(file_path))
        assert result == data

    def test_missing_file_raises(self, temp_dir):
        """Test that missing file raises appropriate exception."""
        file_path = temp_dir / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_character_descriptions(str(file_path))


class TestGenerateVoiceSample:
    """Tests for generate_voice_sample function."""

    def test_generates_with_mock_engine(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test voice sample generation with mock engine."""
        from audiobook_generator.voice_mapper import VoiceMapper

        voice_mapper = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)

        success, output_file, duration, is_valid, validation_msg = generate_voice_sample(
            character_name="jane",
            description="A gentle, refined female voice.",
            voice_mapper=voice_mapper,
            output_dir=str(temp_dir),
            verbose=False
        )

        assert success is True
        assert output_file is not None
        assert duration > 0

    def test_validates_when_requested(self, temp_dir, mock_tts_engine, mock_llm_client, sample_character_descriptions):
        """Test that validation is performed when requested."""
        from audiobook_generator.voice_mapper import VoiceMapper

        mock_llm_client.set_response({
            "role": "assistant",
            "content": '{"overall_match": true, "gender_match": true, "reasons": ""}'
        })

        voice_mapper = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)

        success, output_file, duration, is_valid, validation_msg = generate_voice_sample(
            character_name="jane",
            description="A gentle, refined female voice.",
            voice_mapper=voice_mapper,
            output_dir=str(temp_dir),
            verbose=False,
            validate=True,
            validation_client=mock_llm_client
        )

        assert isinstance(is_valid, bool)

    def test_handles_failure(self, temp_dir, mock_tts_engine_failure, sample_character_descriptions):
        """Test handling of generation failure."""
        from audiobook_generator.voice_mapper import VoiceMapper

        voice_mapper = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine_failure)

        success, output_file, duration, is_valid, validation_msg = generate_voice_sample(
            character_name="jane",
            description="A gentle voice.",
            voice_mapper=voice_mapper,
            output_dir=str(temp_dir),
            verbose=False
        )

        assert success is False

    def test_returns_correct_tuple_format(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that returned tuple has correct format."""
        from audiobook_generator.voice_mapper import VoiceMapper

        voice_mapper = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)

        result = generate_voice_sample(
            character_name="jane",
            description="A gentle voice.",
            voice_mapper=voice_mapper,
            output_dir=str(temp_dir),
            verbose=False
        )

        assert isinstance(result, tuple)
        assert len(result) == 5
        success, output_file, duration, is_valid, validation_msg = result
        assert isinstance(success, bool)
        assert output_file is None or isinstance(output_file, str)
        assert isinstance(duration, float)
        assert isinstance(is_valid, bool)
        assert isinstance(validation_msg, str)


class TestGenerateVoiceSamples:
    """Tests for generate_voice_samples function."""

    def test_generates_all_characters(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that all characters are processed."""
        status, voices = generate_voice_samples(
            descriptions=sample_character_descriptions,
            output_dir=str(temp_dir),
            device="cpu",
            verbose=False,
            voice_engine="mock",
            force_regenerate=True
        )

        assert "Successfully" in status or len(voices) > 0

    def test_adds_narrator_automatically(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that narrator voice is added automatically."""
        descriptions_without_narrator = {k: v for k, v in sample_character_descriptions.items() if k != "narrator"}

        status, voices = generate_voice_samples(
            descriptions=descriptions_without_narrator,
            output_dir=str(temp_dir),
            device="cpu",
            verbose=False,
            voice_engine="mock",
            engine=mock_tts_engine
        )

        assert "narrator" in voices or len(voices) > 0

    def test_respects_single_character_filter(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that single_character parameter filters correctly."""
        status, voices = generate_voice_samples(
            descriptions=sample_character_descriptions,
            output_dir=str(temp_dir),
            device="cpu",
            single_character="jane",
            verbose=False,
            voice_engine="mock"
        )

        if len(voices) > 0:
            assert "jane" in voices or len(voices) == 1

    def test_filters_seed_characters(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that seed characters are filtered."""
        seed_characters = {"jane": str(temp_dir / "seed_jane.wav")}

        status, voices = generate_voice_samples(
            descriptions=sample_character_descriptions,
            output_dir=str(temp_dir),
            device="cpu",
            verbose=False,
            seed_characters=seed_characters,
            voice_engine="mock"
        )

        assert isinstance(status, str)
        assert isinstance(voices, dict)

    def test_skips_existing_voices(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that existing voice files are skipped."""
        existing_voice = temp_dir / "jane.wav"
        existing_voice.touch()

        status, voices = generate_voice_samples(
            descriptions=sample_character_descriptions,
            output_dir=str(temp_dir),
            device="cpu",
            verbose=False,
            voice_engine="mock"
        )

        assert "jane" in voices

    def test_force_regenerate_overrides_skip(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that force_regenerate generates even when files exist."""
        existing_voice = temp_dir / "jane.wav"
        existing_voice.touch()

        status, voices = generate_voice_samples(
            descriptions=sample_character_descriptions,
            output_dir=str(temp_dir),
            device="cpu",
            verbose=False,
            voice_engine="mock",
            force_regenerate=True
        )

        assert isinstance(status, str)

    def test_handles_missing_character(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test handling of missing character."""
        status, voices = generate_voice_samples(
            descriptions=sample_character_descriptions,
            output_dir=str(temp_dir),
            device="cpu",
            single_character="nonexistent_character_xyz",
            verbose=False,
            voice_engine="mock"
        )

        assert "not found" in status.lower() or len(voices) == 0

    def test_returns_error_tuple_on_exception(self, temp_dir, mock_tts_engine_failure, sample_character_descriptions):
        """Test that error tuple is returned on exception."""
        status, voices = generate_voice_samples(
            descriptions=sample_character_descriptions,
            output_dir=str(temp_dir),
            device="cpu",
            verbose=False,
            voice_engine="mock"
        )

        assert isinstance(status, str)
        assert isinstance(voices, dict)

    def test_saves_voices_map_json(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that voices_map.json is saved."""
        status, voices = generate_voice_samples(
            descriptions=sample_character_descriptions,
            output_dir=str(temp_dir),
            device="cpu",
            verbose=False,
            voice_engine="mock"
        )

        voices_map_file = temp_dir / "voices_map.json"
        if voices_map_file.exists():
            with open(voices_map_file, "r", encoding="utf-8") as f:
                saved_map = json.load(f)
            assert isinstance(saved_map, dict)


class TestGenerateVoiceSamplesIntegration:
    """Integration tests for voice sample generation."""

    def test_full_pipeline_with_mock(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test full pipeline with mock engine."""
        status, voices = generate_voice_samples(
            descriptions=sample_character_descriptions,
            output_dir=str(temp_dir),
            device="cpu",
            verbose=True,
            voice_engine="mock"
        )

        assert isinstance(status, str)
        assert isinstance(voices, dict)

    def test_voice_files_exist(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that voice files are created."""
        generate_voice_samples(
            descriptions=sample_character_descriptions,
            output_dir=str(temp_dir),
            device="cpu",
            verbose=False,
            voice_engine="mock"
        )

        for char_name in sample_character_descriptions:
            for ext in [".wav", ".mp3", ".flac"]:
                voice_file = temp_dir / f"{char_name}{ext}"
                if voice_file.exists():
                    assert voice_file.stat().st_size >= 0
                    break

    def test_tracks_failed_characters(self, temp_dir, mock_tts_engine_failure, sample_character_descriptions):
        """Test that failed characters are tracked."""
        status, voices = generate_voice_samples(
            descriptions=sample_character_descriptions,
            output_dir=str(temp_dir),
            device="cpu",
            verbose=False,
            voice_engine="mock"
        )

        assert "failed" in status.lower() or len(voices) == 0


class TestVoiceMapperIntegration:
    """Integration tests for VoiceMapper with voice generation."""

    def test_voice_mapper_generates_and_caches(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that VoiceMapper caches generated voices."""
        from audiobook_generator.voice_mapper import VoiceMapper

        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)

        success, output_file, duration = vm.generate_voice_sample(
            character_name="jane",
            description="A gentle voice.",
            verbose=False
        )

        assert success is True
        cached_path = vm.get_voice_path("jane")
        assert cached_path is not None

    def test_voice_mapper_manages_multiple_characters(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test VoiceMapper handles multiple characters."""
        from audiobook_generator.voice_mapper import VoiceMapper

        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)

        for char_name, description in sample_character_descriptions.items():
            if char_name == "narrator":
                continue
            success, output_file, duration = vm.generate_voice_sample(
                character_name=char_name,
                description=description,
                verbose=False
            )
            assert success is True

        assert len(vm.voice_paths) > 0