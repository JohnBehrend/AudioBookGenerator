"""Unit tests for tts discovery functions."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetEngineDir:
    """Test get_engine_dir function."""

    def test_get_engine_dir_not_found(self):
        """Test getting engine directory for non-existent engine."""
        from tts import get_engine_dir
        
        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine_dir("nonexistent")


class TestGetEngineCapabilities:
    """Test get_engine_capabilities function."""

    @patch("tts.subprocess.run")
    def test_probe_engine(self, mock_run):
        """Test probing engine capabilities."""
        from tts import get_engine_capabilities
        
        # Mock the subprocess output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "name": "omni",
            "methods": ["generate_line", "generate_voice_sample"],
            "sample_rate": 24000,
        })
        mock_run.return_value = mock_result
        
        with patch("tts.get_engine_dir", return_value=Path("/tmp/omni")):
            caps = get_engine_capabilities("omni")
            
            assert caps["name"] == "omni"
            assert "generate_line" in caps["methods"]
            assert "generate_voice_sample" in caps["methods"]
            assert caps["sample_rate"] == 24000


class TestListVoiceEngines:
    """Test list_voice_engines function."""

    @patch("tts.get_engine_capabilities")
    @patch("tts.list_engines")
    def test_list_voice_engines(self, mock_list_engines, mock_caps):
        """Test listing engines that support voice samples."""
        from tts import list_voice_engines
        
        mock_list_engines.return_value = ["omni", "moss", "vox"]
        mock_caps.side_effect = [
            {"methods": ["generate_line", "generate_voice_sample"]},
            {"methods": ["generate_line"]},
            {"methods": ["generate_line", "generate_voice_sample"]},
        ]
        
        voice_engines = list_voice_engines()
        
        assert "omni" in voice_engines
        assert "vox" in voice_engines
        assert "moss" not in voice_engines


class TestGetEngine:
    """Test get_engine function."""

    @patch("tts.TTSEngine")
    @patch("tts.get_engine_dir")
    def test_get_engine(self, mock_dir, MockEngine):
        """Test getting an engine instance."""
        from tts import get_engine
        
        mock_dir.return_value = Path("/tmp/omni")
        mock_instance = MagicMock()
        MockEngine.return_value = mock_instance
        
        engine = get_engine("omni", device="cuda:0")
        
        assert engine == mock_instance
        mock_dir.assert_called_once_with("omni")
        MockEngine.assert_called_once_with(Path("/tmp/omni"), device="cuda:0")

    @patch("tts.get_engine_dir")
    def test_get_engine_not_found(self, mock_dir):
        """Test getting engine for non-existent engine."""
        from tts import get_engine
        
        mock_dir.side_effect = ValueError("Unknown engine: nonexistent")
        
        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine("nonexistent")
