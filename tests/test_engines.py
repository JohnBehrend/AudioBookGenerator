"""Smoke tests for TTS engine registry and base classes.

WorkerPool/WhisperPool behavior is covered in tests/test_tts_pool.py; routing
with the audiobook pipeline in tests/test_concurrency.py.
"""

import pytest
from unittest.mock import MagicMock, patch

from tts import TTSEngine, list_engines, get_engine


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

    def test_requires_engine_dir(self):
        """TTSEngine requires an engine_dir (TypeError without it)."""
        with pytest.raises(TypeError):
            TTSEngine()

    def test_base_class_delegates_to_worker(self):
        """TTSEngine exposes the worker-facing API."""
        assert hasattr(TTSEngine, 'generate_voice_sample')
        assert hasattr(TTSEngine, 'generate_line')
        assert hasattr(TTSEngine, 'shutdown_worker')
