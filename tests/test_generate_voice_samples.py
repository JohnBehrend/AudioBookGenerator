"""Tests for generate_voice_samples module."""

import json
import os
from pathlib import Path

import pytest
from unittest.mock import patch

from audiobook_generator.config import DEFAULTS
from audiobook_generator.generate_voice_samples import (
    load_character_descriptions,
    generate_voice_sample,
    generate_voice_samples,
)


STATIC_VOICE_TEXT = DEFAULTS["static_voice_text"]


def _run_generate_voice_samples(engine, output_dir, descriptions, **kwargs):
    """Run generate_voice_samples with Whisper mocked so samples are accepted.

    The real pipeline scores each generated sample by Whisper-transcribing it and
    matching words against the static voice text; silence never passes, so without
    this the loop rejects every sample. Mocking transcription to return the exact
    static text makes every sample pass the word-match threshold, letting the test
    exercise the real generation/copy/voices_map logic.
    """
    with patch("audiobook_generator.audiobook_generator.setup_validation_model", return_value=object()), \
         patch("audiobook_generator.utils.transcribe_audio_with_whisper",
               return_value=(STATIC_VOICE_TEXT, [], [])), \
         patch("audiobook_generator.utils.crop_to_ref_text", return_value=False), \
         patch("audiobook_generator.voice_mapper.get_engine", return_value=engine):
        return generate_voice_samples(
            descriptions=descriptions,
            output_dir=str(output_dir),
            device="cpu",
            verbose=False,
            voice_engine="mock",
            engine=engine,
            **kwargs,
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

        with patch("tts.voice_sample.generate_voice_sample") as mock_gen:
            mock_gen.return_value = (True, str(temp_dir / "jane.wav"), 1.0)
            success, output_file, duration, is_valid, validation_msg, is_celebrity = generate_voice_sample(
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

        with patch("tts.voice_sample.generate_voice_sample") as mock_gen:
            mock_gen.return_value = (True, str(temp_dir / "jane.wav"), 1.0)
            success, output_file, duration, is_valid, validation_msg, is_celebrity = generate_voice_sample(
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

        with patch("tts.voice_sample.generate_voice_sample") as mock_gen:
            mock_gen.return_value = (False, None, 0)
            success, output_file, duration, is_valid, validation_msg, is_celebrity = generate_voice_sample(
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

        with patch("tts.voice_sample.generate_voice_sample") as mock_gen:
            mock_gen.return_value = (True, str(temp_dir / "jane.wav"), 1.0)
            result = generate_voice_sample(
                character_name="jane",
                description="A gentle voice.",
                voice_mapper=voice_mapper,
                output_dir=str(temp_dir),
                verbose=False
            )

        assert isinstance(result, tuple)
        assert len(result) == 6
        success, output_file, duration, is_valid, validation_msg, is_celebrity = result
        assert isinstance(success, bool)
        assert output_file is None or isinstance(output_file, str)
        assert isinstance(duration, float)
        assert isinstance(is_valid, bool)
        assert isinstance(validation_msg, str)
        assert isinstance(is_celebrity, bool)


class TestGenerateVoiceSamples:
    """Tests for generate_voice_samples function."""

    def test_respects_single_character_filter(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that single_character parameter filters correctly."""
        status, voices = _run_generate_voice_samples(
            mock_tts_engine, temp_dir, sample_character_descriptions, single_character="jane"
        )

        assert "jane" in voices
        assert "elizabeth" not in voices

    def test_filters_seed_characters(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that seed characters are filtered out of generation."""
        seed_characters = {"jane": str(temp_dir / "seed_jane.wav")}

        status, voices = _run_generate_voice_samples(
            mock_tts_engine, temp_dir, sample_character_descriptions, seed_characters=seed_characters
        )

        assert "jane" not in voices
        assert "elizabeth" in voices

    def test_skips_existing_voices(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that existing voice files are skipped."""
        (temp_dir / "jane.wav").touch()

        status, voices = _run_generate_voice_samples(mock_tts_engine, temp_dir, sample_character_descriptions)

        assert "jane" in voices
        assert "elizabeth" in voices

    def test_force_regenerate_overrides_skip(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that force_regenerate generates even when files exist."""
        (temp_dir / "jane.wav").touch()

        status, voices = _run_generate_voice_samples(
            mock_tts_engine, temp_dir, sample_character_descriptions, force_regenerate=True
        )

        assert "jane" in voices
        assert "Successfully generated" in status

    def test_handles_missing_character(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test handling of missing character."""
        status, voices = _run_generate_voice_samples(
            mock_tts_engine, temp_dir, sample_character_descriptions,
            single_character="nonexistent_character_xyz",
        )

        assert status == "Character 'nonexistent_character_xyz' not found in descriptions."
        assert voices == {}

    def test_returns_error_tuple_on_exception(self, temp_dir, mock_tts_engine_failure, sample_character_descriptions):
        """Test that a total generation failure surfaces an error, not an exception."""
        status, voices = _run_generate_voice_samples(
            mock_tts_engine_failure, temp_dir, sample_character_descriptions
        )

        assert "failed for narrator" in status
        assert voices == {}

    def test_saves_voices_map_json(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that voices_map.json is saved with the generated voices."""
        status, voices = _run_generate_voice_samples(mock_tts_engine, temp_dir, sample_character_descriptions)

        voices_map_file = temp_dir / "voices_map.json"
        assert voices_map_file.exists()
        with open(voices_map_file, "r", encoding="utf-8") as f:
            saved_map = json.load(f)
        assert set(saved_map) == set(voices)


class TestGenerateVoiceSamplesIntegration:
    """Integration tests for voice sample generation."""

    def test_full_pipeline_with_mock(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test full pipeline with mock engine."""
        status, voices = _run_generate_voice_samples(mock_tts_engine, temp_dir, sample_character_descriptions)

        assert "Successfully generated" in status
        assert len(voices) >= 2

    def test_voice_files_exist(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that non-empty voice files are created for every character."""
        status, voices = _run_generate_voice_samples(mock_tts_engine, temp_dir, sample_character_descriptions)

        for char_name, voice_path in voices.items():
            assert Path(voice_path).exists()
            assert Path(voice_path).stat().st_size > 0

    def test_tracks_failed_characters(self, temp_dir, mock_tts_engine_failure, sample_character_descriptions):
        """Test that a failure is reported rather than raising."""
        status, voices = _run_generate_voice_samples(
            mock_tts_engine_failure, temp_dir, sample_character_descriptions
        )

        assert "failed" in status.lower()
        assert voices == {}


class TestVoiceMapperIntegration:
    """Integration tests for VoiceMapper with voice generation."""

    def test_voice_mapper_generates_and_caches(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """Test that VoiceMapper caches generated voices."""
        from audiobook_generator.voice_mapper import VoiceMapper

        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)

        with patch("tts.voice_sample.generate_voice_sample") as mock_gen:
            mock_gen.return_value = (True, str(temp_dir / "jane.wav"), 1.0)
            success, output_file, duration, is_celebrity = vm.generate_voice_sample(
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
            with patch("tts.voice_sample.generate_voice_sample") as mock_gen:
                mock_gen.return_value = (True, str(temp_dir / f"{char_name}.wav"), 1.0)
                success, output_file, duration, is_celebrity = vm.generate_voice_sample(
                    character_name=char_name,
                    description=description,
                    verbose=False
                )
            assert success is True

        assert len(vm.voice_paths) > 0


class TestCelebrityTraceability:
    """Pins the celebrity traceability design through the real pipeline.

    Intended behavior: when celebrity voices are enabled, a character's winning
    celebrity voice is stored under the CELEBRITY's name (``{celebrity}_ref.wav``)
    rather than ``{char}.wav``, and ``voices_map[char]`` points at that
    celebrity-named file — so a character is directly traceable to its celebrity
    just by reading ``voices_map.json``.
    """

    def _run_celebrity(self, temp_dir, mock_llm_client, descriptions, celebrities):
        """Run generate_voice_samples with celebrity voices and a mocked
        build_celebrity_voice returning a distinct celebrity-named ref per char."""
        from audiobook_generator.testing import write_silence_wav

        def fake_build(client, model, character, description, output_dir,
                       pre_matched_celebrity=None, whisper_model=None,
                       tts_engine=None, verbose=False, **kwargs):
            # build_celebrity_voice is called with the per-sample name
            # "<char>.sampleN"; the traceability target is the base character.
            base = character.split(".")[0]
            celeb = celebrities[base]
            ref = os.path.join(output_dir, f"{celeb}_ref.wav")
            if not os.path.exists(ref):
                write_silence_wav(ref, 22050, 1)
            return ref, {
                "character": base,
                "celebrity": celeb,
                "reason": "test",
                "search_query": f"{celeb} interview",
                "segment": ref,
                "audio_source": None,
            }

        with patch("audiobook_generator.celebrity_voices.build_celebrity_voice",
                   side_effect=fake_build):
            return _run_generate_voice_samples(
                None, temp_dir, descriptions,
                use_celebrity_voices=True,
                llm_client=mock_llm_client,
            )

    def test_voices_map_points_to_celebrity_named_file(
        self, temp_dir, mock_llm_client, sample_character_descriptions
    ):
        celebrities = {
            "narrator": "morgan_freeman",
            "jane": "emma_watson",
            "elizabeth": "natalie_portman",
        }
        status, voices = self._run_celebrity(
            temp_dir, mock_llm_client, sample_character_descriptions, celebrities
        )

        # voices_map maps each char to the celebrity-named ref, not {char}.wav.
        assert voices["jane"] == os.path.join(str(temp_dir), "emma_watson_ref.wav")
        assert voices["elizabeth"] == os.path.join(str(temp_dir), "natalie_portman_ref.wav")

        with open(temp_dir / "voices_map.json", "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["jane"] == "emma_watson_ref.wav"
        assert saved["elizabeth"] == "natalie_portman_ref.wav"

        # The old {char}.wav rename must NOT be produced for celebrity voices.
        assert not (temp_dir / "jane.wav").exists()
        assert not (temp_dir / "elizabeth.wav").exists()

    def test_celebrity_ref_files_exist_and_are_what_voices_map_references(
        self, temp_dir, mock_llm_client, sample_character_descriptions
    ):
        celebrities = {
            "narrator": "morgan_freeman",
            "jane": "emma_watson",
            "elizabeth": "natalie_portman",
        }
        _, voices = self._run_celebrity(
            temp_dir, mock_llm_client, sample_character_descriptions, celebrities
        )

        for char_name, voice_path in voices.items():
            assert os.path.exists(voice_path)
            assert os.path.basename(voice_path).endswith("_ref.wav")
            # The file is named by the celebrity, so its name IS the trace.
            assert celebrities[char_name] in os.path.basename(voice_path)