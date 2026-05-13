"""Smoke tests for TTS engine registry and base classes."""

import pytest
from unittest.mock import MagicMock, patch

from audiobook_generator.engines import get_engine, list_engines, _ENGINE_REGISTRY
from audiobook_generator.engines.base import TTSEngine


# ============================================================================
# TESTS
# ============================================================================

class TestEngineRegistry:
    """Tests for engine registry and factory."""

    def test_list_engines_returns_all_engines(self):
        """list_engines should return all registered engine names."""
        engines = list_engines()
        assert "moss" in engines
        assert "omni" in engines
        assert "vox" in engines
        assert "vibevoice" in engines
        assert "echo-tts" in engines
        assert "dramabox" in engines
        assert len(engines) == 6

    def test_get_engine_returns_correct_class(self):
        """get_engine should return the correct engine class."""
        engine = get_engine("moss", device="cpu")
        assert isinstance(engine, TTSEngine)
        assert engine.__class__.__name__ == "MossEngine"

    def test_get_engine_raises_for_unknown(self):
        """get_engine should raise ValueError for unknown engine name."""
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            get_engine("unknown_engine")

    def test_get_engine_raises_for_empty_string(self):
        """get_engine should raise ValueError for empty string."""
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            get_engine("")

    def test_engine_registry_mapping(self):
        """Engine registry should map names to correct classes."""
        from audiobook_generator.engines.moss import MossEngine
        from audiobook_generator.engines.omni import OmniEngine
        from audiobook_generator.engines.vox import VoxEngine
        from audiobook_generator.engines.vibevoice import VibeVoiceEngine
        from audiobook_generator.engines.echo_tts import EchoTTSAdapter
        from audiobook_generator.engines.dramabox import DramaboxEngine

        assert _ENGINE_REGISTRY["moss"] is MossEngine
        assert _ENGINE_REGISTRY["omni"] is OmniEngine
        assert _ENGINE_REGISTRY["vox"] is VoxEngine
        assert _ENGINE_REGISTRY["vibevoice"] is VibeVoiceEngine
        assert _ENGINE_REGISTRY["echo-tts"] is EchoTTSAdapter
        assert _ENGINE_REGISTRY["dramabox"] is DramaboxEngine


class TestTTSEngineBase:
    """Tests for TTSEngine base class."""

    def test_base_class_is_abstract(self):
        """TTSEngine should not be instantiable directly."""
        with pytest.raises(TypeError):
            TTSEngine()

    def test_base_class_has_abstract_methods(self):
        """TTSEngine should have abstract methods."""
        assert hasattr(TTSEngine, '_run_worker')
        assert hasattr(TTSEngine, 'setup')
        assert hasattr(TTSEngine, 'generate_voice_sample')
        assert hasattr(TTSEngine, 'generate_line')

    def test_shutdown_worker_noop(self):
        """shutdown_worker should not raise when no worker exists."""
        engine = MagicMock(spec=TTSEngine)
        # shutdown_worker should handle missing worker gracefully
        assert not hasattr(engine, '_worker')


class TestEngineInstances:
    """Tests for engine instance creation."""

    def test_moss_engine_creation(self):
        """MossEngine should be creatable."""
        from audiobook_generator.engines.moss import MossEngine
        engine = MossEngine(device="cpu")
        assert engine._device == "cpu"

    def test_omni_engine_creation(self):
        """OmniEngine should be creatable."""
        from audiobook_generator.engines.omni import OmniEngine
        engine = OmniEngine(device="cpu")
        assert engine._device == "cpu"

    def test_vox_engine_creation(self):
        """VoxEngine should be creatable."""
        from audiobook_generator.engines.vox import VoxEngine
        engine = VoxEngine(device="cpu")
        assert engine._device == "cpu"

    def test_vibevoice_engine_creation(self):
        """VibeVoiceEngine should be creatable."""
        from audiobook_generator.engines.vibevoice import VibeVoiceEngine
        engine = VibeVoiceEngine(device="cpu")
        assert engine._device == "cpu"

    def test_echo_tts_engine_creation(self):
        """EchoTTSAdapter should be creatable."""
        from audiobook_generator.engines.echo_tts import EchoTTSAdapter
        engine = EchoTTSAdapter(device="cpu")
        assert engine._device == "cpu"

    def test_dramabox_engine_creation(self):
        """DramaboxEngine should be creatable."""
        from audiobook_generator.engines.dramabox import DramaboxEngine
        engine = DramaboxEngine(device="cpu")
        assert engine._device == "cpu"
        assert engine.ENV_NAME == "dramabox"


class TestEngineFactory:
    """Tests for get_engine factory with different engines."""

    def test_get_engine_moss(self):
        """get_engine('moss') returns MossEngine."""
        engine = get_engine("moss", device="cpu")
        from audiobook_generator.engines.moss import MossEngine
        assert isinstance(engine, MossEngine)

    def test_get_engine_omni(self):
        """get_engine('omni') returns OmniEngine."""
        engine = get_engine("omni", device="cpu")
        from audiobook_generator.engines.omni import OmniEngine
        assert isinstance(engine, OmniEngine)

    def test_get_engine_vox(self):
        """get_engine('vox') returns VoxEngine."""
        engine = get_engine("vox", device="cpu")
        from audiobook_generator.engines.vox import VoxEngine
        assert isinstance(engine, VoxEngine)

    def test_get_engine_vibevoice(self):
        """get_engine('vibevoice') returns VibeVoiceEngine."""
        engine = get_engine("vibevoice", device="cpu")
        from audiobook_generator.engines.vibevoice import VibeVoiceEngine
        assert isinstance(engine, VibeVoiceEngine)

    def test_get_engine_echo_tts(self):
        """get_engine('echo-tts') returns EchoTTSAdapter."""
        engine = get_engine("echo-tts", device="cpu")
        from audiobook_generator.engines.echo_tts import EchoTTSAdapter
        assert isinstance(engine, EchoTTSAdapter)

    def test_get_engine_dramabox(self):
        """get_engine('dramabox') returns DramaboxEngine."""
        engine = get_engine("dramabox", device="cpu")
        from audiobook_generator.engines.dramabox import DramaboxEngine
        assert isinstance(engine, DramaboxEngine)
