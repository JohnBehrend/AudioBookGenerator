"""Smoke tests for TTS engine registry and base classes."""

import pytest
from unittest.mock import MagicMock, patch

from tts import TTSEngine, list_engines, get_engine
from tts.pool import WorkerPool, WhisperPool


# ============================================================================
# TESTS
# ============================================================================

class TestEngineRegistry:
    """Tests for engine registry and factory."""

    def test_list_engines_returns_nonempty(self):
        """list_engines should return at least one engine name."""
        engines = list_engines()
        assert len(engines) > 0

    def test_list_engines_returns_strings(self):
        """list_engines should return a list of strings."""
        engines = list_engines()
        for e in engines:
            assert isinstance(e, str)

    def test_get_engine_raises_for_unknown(self):
        """get_engine should raise ValueError for unknown engine name."""
        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine("unknown_engine")

    def test_get_engine_raises_for_empty_string(self):
        """get_engine should raise ValueError for empty string."""
        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine("")


class TestTTSEngineBase:
    """Tests for TTSEngine base class."""

    def test_base_class_is_abstract(self):
        """TTSEngine should not be instantiable directly with engine_dir only (missing abstract methods)."""
        with pytest.raises(TypeError):
            TTSEngine()

    def test_base_class_has_abstract_methods(self):
        """TTSEngine should have abstract methods."""
        assert hasattr(TTSEngine, 'generate_voice_sample')
        assert hasattr(TTSEngine, 'generate_line')


class TestWorkerPool:
    """Tests for WorkerPool class."""

    def test_worker_pool_init(self):
        """WorkerPool should accept engine_dir and devices."""
        pool = WorkerPool(engine_dir="/tmp/test", devices=["cuda:0"])
        assert pool.devices == ["cuda:0"]

    def test_whisper_pool_init(self):
        """WhisperPool should accept model_factory and size."""
        mock_factory = MagicMock(return_value=MagicMock())
        pool = WhisperPool(mock_factory, size=2)
        assert pool._size == 2
