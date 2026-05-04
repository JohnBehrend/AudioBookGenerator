"""Tests for voice_mapper module."""

import json
import os
import pytest

from audiobook_generator.voice_mapper import VoiceMapper


class TestVoiceMapperInit:
    """Tests for VoiceMapper initialization."""

    def test_creates_output_dir(self, temp_dir):
        """Test that output directory is created."""
        vm = VoiceMapper(output_dir=str(temp_dir))
        assert temp_dir.exists()
        assert temp_dir.is_dir()

    def test_default_engine(self, temp_dir):
        """Test default TTS engine is set."""
        vm = VoiceMapper(output_dir=str(temp_dir))
        assert vm.tts_engine is not None

    def test_custom_engine(self, temp_dir):
        """Test custom TTS engine can be set."""
        vm = VoiceMapper(output_dir=str(temp_dir), tts_engine="omni")
        assert vm.tts_engine == "omni"

    def test_custom_device(self, temp_dir):
        """Test custom device can be set."""
        vm = VoiceMapper(output_dir=str(temp_dir), device="cuda:1")
        assert vm.device == "cuda:1"

    def test_loads_existing_voice_map(self, temp_dir, mock_tts_engine):
        """Test that existing voices_map.json is loaded."""
        voices_map = {"narrator": "narrator.wav", "jane": "jane.wav"}
        map_file = temp_dir / "voices_map.json"
        with open(map_file, "w", encoding="utf-8") as f:
            json.dump(voices_map, f)

        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        assert "narrator" in vm.voice_paths
        assert "jane" in vm.voice_paths

    def test_injected_engine(self, temp_dir, mock_tts_engine):
        """Test that injected engine is used."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        assert vm._injected_engine is mock_tts_engine


class TestGetVoicePath:
    """Tests for get_voice_path method."""

    def test_returns_cached_path(self, temp_dir, mock_tts_engine):
        """Test that cached voice paths are returned."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        vm.voice_paths["jane"] = "/path/to/jane.wav"
        result = vm.get_voice_path("jane")
        assert result == "/path/to/jane.wav"

    def test_finds_existing_file(self, temp_dir, mock_tts_engine):
        """Test that existing voice files are found."""
        voice_file = temp_dir / "narrator.wav"
        voice_file.touch()

        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        result = vm.get_voice_path("narrator")
        assert result is not None
        assert result.endswith("narrator.wav")

    def test_partial_match_case_insensitive(self, temp_dir, mock_tts_engine):
        """Test partial match is case insensitive."""
        voice_file = temp_dir / "JaneBennet.wav"
        voice_file.touch()

        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        result = vm.get_voice_path("jane")
        assert result is not None

    def test_returns_none_for_missing(self, temp_dir, mock_tts_engine):
        """Test that None is returned for non-existent character."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        result = vm.get_voice_path("nonexistent")
        assert result is None

    def test_duplicate_replacement_map(self, temp_dir, mock_tts_engine):
        """Test that duplicate replacement map is used."""
        voice_file = temp_dir / "jane.wav"
        voice_file.touch()

        vm = VoiceMapper(
            output_dir=str(temp_dir),
            engine=mock_tts_engine,
            duplicate_replacement_map={"jane bennet": "jane"}
        )
        result = vm.get_voice_path("jane bennet")
        assert result is not None


class TestAddVoicePath:
    """Tests for add_voice_path method."""

    def test_adds_voice_path(self, temp_dir, mock_tts_engine):
        """Test that voice path is added correctly."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        vm.add_voice_path("jane", str(temp_dir / "jane.wav"))

        assert "jane" in vm.voice_paths
        assert "jane" in vm._voice_map

    def test_saves_to_voices_map_json(self, temp_dir, mock_tts_engine):
        """Test that voice map is saved to file."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        vm.add_voice_path("jane", str(temp_dir / "jane.wav"))

        map_file = temp_dir / "voices_map.json"
        assert map_file.exists()

        with open(map_file, "r", encoding="utf-8") as f:
            saved_map = json.load(f)
        assert "jane" in saved_map


class TestGetAllVoicePaths:
    """Tests for get_all_voice_paths method."""

    def test_returns_copy(self, temp_dir, mock_tts_engine):
        """Test that returned dict is a copy."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        vm.voice_paths["jane"] = "/path/to/jane.wav"

        result = vm.get_all_voice_paths()
        assert "jane" in result
        result["jane"] = "/changed/path"
        assert vm.voice_paths["jane"] == "/path/to/jane.wav"

    def test_empty_when_no_voices(self, temp_dir, mock_tts_engine):
        """Test empty dict when no voices."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        result = vm.get_all_voice_paths()
        assert result == {}


class TestGetNarratorVoice:
    """Tests for get_narrator_voice method."""

    def test_returns_narrator_voice(self, temp_dir, mock_tts_engine):
        """Test that narrator voice is returned."""
        voice_file = temp_dir / "narrator.wav"
        voice_file.touch()

        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        result = vm.get_narrator_voice()
        assert result is not None
        assert "narrator" in result


class TestSetupTTSEngine:
    """Tests for setup_tts_engine method."""

    def test_caches_engine(self, temp_dir, mock_tts_engine):
        """Test that engine is cached after first call."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        result1 = vm.setup_tts_engine()
        result2 = vm.setup_tts_engine()
        assert result1 == result2


class TestGetEngine:
    """Tests for get_engine method."""

    def test_returns_injected_engine(self, temp_dir, mock_tts_engine):
        """Test that injected engine is returned."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        engine = vm.get_engine()
        assert engine is mock_tts_engine

    def test_returns_cached_engine(self, temp_dir, mock_tts_engine):
        """Test that cached engine is returned."""
        vm = VoiceMapper(output_dir=str(temp_dir))
        engine1 = vm.get_engine()
        engine2 = vm.get_engine()
        assert engine1 is engine2


class TestSetEngine:
    """Tests for set_engine method."""

    def test_sets_engine(self, temp_dir, mock_tts_engine, mock_tts_engine_failure):
        """Test that engine can be set."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        vm.set_engine(mock_tts_engine_failure)
        assert vm._injected_engine is mock_tts_engine_failure


class TestCleanupEngines:
    """Tests for cleanup_engines method."""

    def test_shuts_down_engine(self, temp_dir, mock_tts_engine):
        """Test that engine is shut down."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        vm._cached_engine = mock_tts_engine
        vm.cleanup_engines()
        assert vm._cached_engine is None


class TestCleanupTTSModels:
    """Tests for cleanup_tts_models method."""

    def test_clears_models_cache(self, temp_dir, mock_tts_engine):
        """Test that models cache is cleared."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        vm.tts_models["test_key"] = "test_value"
        vm.cleanup_tts_models()
        assert len(vm.tts_models) == 0


class TestReset:
    """Tests for reset method."""

    def test_clears_all_state(self, temp_dir, mock_tts_engine):
        """Test that all state is cleared."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        vm.voice_paths["jane"] = "/path/to/jane.wav"
        vm.voice_clone_prompts["jane"] = "test_prompt"
        vm._voice_map["jane"] = "jane.wav"
        vm.tts_models["test"] = "value"

        vm.reset()

        assert len(vm.voice_paths) == 0
        assert len(vm.voice_clone_prompts) == 0
        assert len(vm._voice_map) == 0
        assert len(vm.tts_models) == 0


class TestUnloadModel:
    """Tests for unload_model method."""

    def test_unloads_specific_engine(self, temp_dir, mock_tts_engine):
        """Test that specific engine models are unloaded."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        vm.tts_models["moss_turbo_False"] = "model1"
        vm.tts_models["omni_turbo_False"] = "model2"
        vm.tts_models["moss_turbo_True"] = "model3"

        vm.unload_model("moss")

        assert "moss_turbo_False" not in vm.tts_models
        assert "moss_turbo_True" not in vm.tts_models
        assert "omni_turbo_False" in vm.tts_models


class TestGenerateVoiceSample:
    """Tests for generate_voice_sample method."""

    def test_generates_voice_sample(self, temp_dir, mock_tts_engine):
        """Test that voice sample is generated."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        success, output_file, duration = vm.generate_voice_sample(
            character_name="test_char",
            description="A test voice.",
            verbose=False
        )

        assert success is True
        assert output_file is not None
        assert duration > 0

    def test_adds_voice_path_on_success(self, temp_dir, mock_tts_engine):
        """Test that voice path is added on successful generation."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        initial_count = len(vm.voice_paths)

        vm.generate_voice_sample(
            character_name="test_char",
            description="A test voice.",
            verbose=False
        )

        assert len(vm.voice_paths) == initial_count + 1


class TestBuildVoiceClonePrompt:
    """Tests for build_voice_clone_prompt method."""

    def test_requires_voice_file(self, temp_dir, mock_tts_engine):
        """Test that voice file is required."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        with pytest.raises(Exception):
            vm.build_voice_clone_prompt(
                voice_path=str(temp_dir / "nonexistent.wav"),
                ref_text="Test text"
            )


class TestGetVoiceClonePrompt:
    """Tests for get_voice_clone_prompt method."""

    def test_returns_none_for_missing_voice(self, temp_dir, mock_tts_engine):
        """Test that None is returned when voice is missing."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        result = vm.get_voice_clone_prompt("nonexistent", verbose=False)
        assert result is None


class TestGetAllClonePrompts:
    """Tests for get_all_clone_prompts method."""

    def test_returns_empty_for_no_voices(self, temp_dir, mock_tts_engine):
        """Test that empty dict is returned when no voices."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        result = vm.get_all_clone_prompts(verbose=False)
        assert result == {}


class TestLoadVoiceMap:
    """Tests for _load_voice_map method."""

    def test_handles_invalid_json(self, temp_dir, mock_tts_engine):
        """Test that invalid JSON is handled gracefully."""
        map_file = temp_dir / "voices_map.json"
        with open(map_file, "w") as f:
            f.write("not valid json {{{")

        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        assert vm._voice_map == {}


class TestSaveVoiceMap:
    """Tests for _save_voice_map method."""

    def test_saves_voice_map(self, temp_dir, mock_tts_engine):
        """Test that voice map is saved correctly."""
        vm = VoiceMapper(output_dir=str(temp_dir), engine=mock_tts_engine)
        vm._voice_map = {"jane": "jane.wav"}
        vm._save_voice_map()

        map_file = temp_dir / "voices_map.json"
        with open(map_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["jane"] == "jane.wav"


class TestValidateVoiceWithLLM:
    """Tests for validate_voice_with_llm static method."""

    def test_with_mock_client(self, temp_dir, mock_tts_engine, mock_llm_client):
        """Test voice validation with mock client."""
        voice_file = temp_dir / "test_voice.wav"
        import numpy as np
        import torch
        import torchaudio
        audio = np.zeros(22050, dtype=np.float32)
        torchaudio.save(str(voice_file), torch.from_numpy(audio), 22050)

        mock_llm_client.set_response({
            "role": "assistant",
            "content": '{"overall_match": true, "gender_match": true, "reasons": ""}'
        })

        from audiobook_generator.voice_mapper import VoiceMapper
        is_valid, msg = VoiceMapper.validate_voice_with_llm(
            voice_path=str(voice_file),
            description="A gentle voice.",
            sample_text="Hello world.",
            client=mock_llm_client,
            verbose=False
        )

        assert isinstance(is_valid, bool)

    def test_handles_json_error(self, temp_dir, mock_tts_engine, mock_llm_client):
        """Test handling of invalid JSON response."""
        voice_file = temp_dir / "test_voice.wav"
        import numpy as np
        import torch
        import torchaudio
        audio = np.zeros(22050, dtype=np.float32)
        torchaudio.save(str(voice_file), torch.from_numpy(audio), 22050)

        mock_llm_client.set_response({
            "role": "assistant",
            "content": "YES - the voice matches"
        })

        from audiobook_generator.voice_mapper import VoiceMapper
        is_valid, msg = VoiceMapper.validate_voice_with_llm(
            voice_path=str(voice_file),
            description="A gentle voice.",
            sample_text="Hello world.",
            client=mock_llm_client,
            verbose=False
        )

        assert is_valid is True


class TestDuplicateReplacementMap:
    """Tests for duplicate_replacement_map functionality."""

    def test_applies_replacement_for_lookup(self, temp_dir, mock_tts_engine):
        """Test that duplicate replacement map is applied during lookup."""
        voice_file = temp_dir / "jane.wav"
        voice_file.touch()

        vm = VoiceMapper(
            output_dir=str(temp_dir),
            engine=mock_tts_engine,
            duplicate_replacement_map={"jane bennet": "jane"}
        )

        result = vm.get_voice_path("jane bennet")
        assert result is not None
        assert "jane" in result

    def test_prefers_exact_match(self, temp_dir, mock_tts_engine):
        """Test that exact match is preferred over replacement."""
        voice_file = temp_dir / "jane_bennet.wav"
        voice_file.touch()

        vm = VoiceMapper(
            output_dir=str(temp_dir),
            engine=mock_tts_engine,
            duplicate_replacement_map={"jane bennet": "jane"}
        )

        result = vm.get_voice_path("jane bennet")
        assert result is not None
        assert "jane_bennet" in result