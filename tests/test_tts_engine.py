"""Unit tests for tts.engine module."""

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestTTSEngine:
    """Test TTSEngine base class."""

    def test_init(self):
        """Test TTSEngine initialization with concrete subclass."""
        from tts.engine import TTSEngine
        
        class ConcreteEngine(TTSEngine):
            def generate_line(self, text, voice_path, output_path, verbose=False, ref_text=None):
                return True
            def generate_voice_sample(self, character_name, description, output_dir, verbose=False):
                return (True, None, 0)
        
        engine_dir = Path("/tmp/test-engine")
        engine = ConcreteEngine(engine_dir, device="cuda:0")
        
        assert engine.engine_dir == engine_dir
        assert engine.device == "cuda:0"

    def test_can_instantiate_directly(self):
        """Test that TTSEngine can be instantiated directly (it delegates to worker subprocess)."""
        from tts.engine import TTSEngine

        engine_dir = Path("/tmp/test-engine")
        engine = TTSEngine(engine_dir)

        assert engine.engine_dir == engine_dir
        assert engine._shared is None

    def test_concrete_class(self):
        """Test that a concrete class can be created."""
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


class TestVoiceSample:
    """Test tts.voice_sample module."""

    @patch("tts.voice_sample.EngineWorker")
    def test_generate_voice_sample(self, MockWorker):
        """Test generate_voice_sample function."""
        from tts.voice_sample import generate_voice_sample
        
        # Mock the worker
        mock_worker = MagicMock()
        mock_worker.request.return_value = {
            "success": True,
            "output_file": "/path/to/output.wav",
            "duration": 2.5,
        }
        MockWorker.return_value = mock_worker
        
        engine_dir = Path("/tmp/test-engine")
        device = "cuda:0"
        character_name = "test_character"
        description = "male, middle-aged, moderate pitch"
        output_dir = Path("/tmp/output")
        
        success, output_file, duration = generate_voice_sample(
            engine_dir=engine_dir,
            device=device,
            character_name=character_name,
            description=description,
            output_dir=output_dir,
        )
        
        assert success is True
        assert output_file == "/path/to/output.wav"
        assert duration == 2.5
        
        mock_worker.start.assert_called_once()
        mock_worker.request.assert_called_once()
        mock_worker.shutdown.assert_called_once()

    @patch("tts.voice_sample.EngineWorker")
    def test_generate_voice_sample_empty_description(self, MockWorker):
        """Test generate_voice_sample with empty description."""
        from tts.voice_sample import generate_voice_sample
        
        engine_dir = Path("/tmp/test-engine")
        device = "cuda:0"
        character_name = "test_character"
        description = ""
        output_dir = Path("/tmp/output")
        
        success, output_file, duration = generate_voice_sample(
            engine_dir=engine_dir,
            device=device,
            character_name=character_name,
            description=description,
            output_dir=output_dir,
        )
        
        assert success is False
        assert output_file is None
        assert duration == 0

    def test_build_voice_clone_prompt_delegates_to_worker(self):
        """Test that build_voice_clone_prompt requests the prompt from the worker."""
        import sys

        from tts.voice_sample import build_voice_clone_prompt

        mock_worker = MagicMock()
        mock_worker.request.return_value = {"voice_clone_prompt": "PROMPT"}
        mock_sf = MagicMock()
        mock_sf.read.return_value = ("audio", 22050)
        mock_torch = MagicMock()
        mock_torch.from_numpy.return_value = "tensor"

        with patch.dict(sys.modules, {"soundfile": mock_sf, "torch": mock_torch}), \
             patch("tts.voice_sample.EngineWorker", return_value=mock_worker):
            result = build_voice_clone_prompt(
                Path("/tmp/engine"), "cuda:0", "/tmp/voice.wav", ref_text="hello"
            )

        assert result == "PROMPT"
        mock_worker.request.assert_called_once_with(
            "build_voice_clone_prompt",
            voice_path="/tmp/voice.wav",
            ref_text="hello",
            device="cuda:0",
        )
        mock_worker.shutdown.assert_called_once()
