"""Unit tests for tts.engine module."""

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestTTSEngine:
    """Test TTSEngine base class."""

    def test_can_instantiate_directly(self):
        """Test that TTSEngine can be instantiated directly (it delegates to worker subprocess)."""
        from tts.engine import TTSEngine

        engine_dir = Path("/tmp/test-engine")
        engine = TTSEngine(engine_dir)

        assert engine.engine_dir == engine_dir
        assert engine._shared is None

    def test_concrete_class(self):
        """Test that a concrete subclass can be created and used."""
        from tts.engine import TTSEngine

        class ConcreteEngine(TTSEngine):
            def generate_line(self, text, voice_path, output_path, verbose=False, ref_text=None):
                return True

            def generate_voice_sample(self, character_name, description, output_dir, verbose=False):
                return (True, "/path/to/sample.wav", 1.5)

        engine_dir = Path("/tmp/test-engine")
        engine = ConcreteEngine(engine_dir, device="cuda:0")

        assert engine.engine_dir == engine_dir
        assert engine.device == "cuda:0"

        # Test generate_line
        result = engine.generate_line("Hello", None, "/output.wav")
        assert result is True

        # Test generate_voice_sample
        success, path, duration = engine.generate_voice_sample("character", "description", Path("/output"))
        assert success is True
        assert path == "/path/to/sample.wav"
        assert duration == 1.5

    def test_clear_cuda_cache(self):
        """Test _clear_cuda_cache method."""
        from tts.engine import TTSEngine
        
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache") as mock_empty_cache, \
             patch("gc.collect") as mock_collect:
            
            TTSEngine._clear_cuda_cache()
            
            mock_collect.assert_called_once()
            mock_empty_cache.assert_called_once()
