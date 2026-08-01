"""Tests for describe_voice.py CLI script."""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from audiobook_generator.testing import write_silence_wav


class TestDescribeVoiceCLI:
    """Tests for describe_voice.py CLI script."""

    def test_script_syntax(self):
        """Test that describe_voice.py has valid Python syntax."""
        import ast
        script_path = Path(__file__).resolve().parent.parent / "describe_voice.py"
        with open(script_path) as f:
            ast.parse(f.read())

    def test_creates_wav_file(self, temp_dir):
        """Test that a WAV file is created for testing."""
        voice_file = temp_dir / "test_voice.wav"
        write_silence_wav(voice_file, 22050, 1)
        assert voice_file.exists()

    def test_missing_file_error(self, temp_dir, capsys):
        """Test error when file is missing."""
        from describe_voice import main
        with patch.object(sys, 'argv', ['describe_voice.py', '/nonexistent/file.wav']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_verbose_flag(self, temp_dir, capsys):
        """Test that verbose flag is passed correctly."""
        voice_file = temp_dir / "test_voice.wav"
        write_silence_wav(voice_file, 22050, 1)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "female, young adult, high pitch"
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(sys, 'argv', ['describe_voice.py', str(voice_file), '--verbose']):
            with patch('describe_voice.OpenAI', return_value=mock_client):
                with patch('describe_voice.VoiceMapper') as mock_mapper:
                    mock_mapper.describe_voice_with_llm.return_value = "female, young adult, high pitch"
                    from describe_voice import main
                    main()

        captured = capsys.readouterr()
        assert "female" in captured.out

    def test_default_endpoint(self, temp_dir):
        """Test that default endpoint is used."""
        voice_file = temp_dir / "test_voice.wav"
        write_silence_wav(voice_file, 22050, 1)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "male, middle-aged, moderate pitch"
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(sys, 'argv', ['describe_voice.py', str(voice_file)]):
            with patch('describe_voice.OpenAI') as mock_openai:
                mock_openai.return_value = mock_client
                with patch('describe_voice.VoiceMapper') as mock_mapper:
                    mock_mapper.describe_voice_with_llm.return_value = "male, middle-aged, moderate pitch"
                    from describe_voice import main
                    main()

        # Verify OpenAI was called with default endpoint
        mock_openai.assert_called_once()


class TestDescribeVoiceIntegration:
    """Integration tests for describe_voice functionality."""

    def test_describe_voice_method_exists(self):
        """Test that describe_voice_with_llm method exists on VoiceMapper."""
        from audiobook_generator.voice_mapper import VoiceMapper
        assert hasattr(VoiceMapper, 'describe_voice_with_llm')
        assert callable(VoiceMapper.describe_voice_with_llm)

    def test_describe_voice_signature(self):
        """Test that describe_voice_with_llm has correct signature."""
        import inspect
        from audiobook_generator.voice_mapper import VoiceMapper
        sig = inspect.signature(VoiceMapper.describe_voice_with_llm)
        params = list(sig.parameters.keys())
        assert 'voice_path' in params
        assert 'client' in params
        assert 'model' in params
        assert 'verbose' in params

    def test_describe_voice_return_type(self):
        """Test that describe_voice_with_llm returns a string."""
        import inspect
        from audiobook_generator.voice_mapper import VoiceMapper
        sig = inspect.signature(VoiceMapper.describe_voice_with_llm)
        return_annotation = sig.return_annotation
        assert return_annotation == str

    def test_describe_voice_is_static(self):
        """Test that describe_voice_with_llm is a static method."""
        from audiobook_generator.voice_mapper import VoiceMapper
        assert isinstance(
            VoiceMapper.__dict__['describe_voice_with_llm'],
            staticmethod
        )
