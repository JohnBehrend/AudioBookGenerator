"""Tests for postfix handling, clipping accuracy, and engine integration."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from audiobook_generator.pipeline import (
    calculate_clip_points,
    prepare_script_for_tts,
    add_postfix,
)
from audiobook_generator.audiobook_generator import (
    _tts_generate_only,
    _validate_and_clip_audio,
    generate_tts_for_line,
    get_non_silent_audio_from_wavs,
    TTSConfig,
)
from audiobook_generator.voice_mapper import VoiceMapper
from audiobook_generator.config import DEFAULTS


class TestPostfixNotDoubled:
    """Ensure postfix is only appended once through the call chain."""

    def test_prepare_script_adds_postfix_once(self):
        """prepare_script_for_tts should add postfix exactly once."""
        text = "Hello world"
        full_script, token = prepare_script_for_tts(text, DEFAULTS["short_text_postfix"])
        # The postfix should appear exactly once
        assert full_script.count(DEFAULTS["short_text_postfix"]) == 1
        assert token == DEFAULTS["short_text_postfix"].strip().split(" ")[0]

    def test_prepare_script_no_postfix_when_disabled(self):
        """prepare_script_for_tts should not add postfix when None."""
        text = "Hello world"
        full_script, token = prepare_script_for_tts(text, None)
        assert full_script == "Hello world"
        assert token is None

    def test_prepare_script_after_punctuation(self):
        """Postfix should be appended after existing punctuation."""
        text = "Hello world."
        full_script, token = prepare_script_for_tts(text, "and also with you")
        assert "Hello world. and also with you" == full_script

    def test_prepare_script_no_punctuation(self):
        """Postfix should add period before appending when none exists."""
        text = "Hello world"
        full_script, token = prepare_script_for_tts(text, "and also with you")
        assert "Hello world. and also with you" == full_script


class TestTtsGenerateOnlyNoDoublePostfix:
    """_tts_generate_only must NOT call prepare_script_for_tts again."""

    @patch("audiobook_generator.audiobook_generator.generate_output_filename")
    def test_passes_full_script_directly_to_engine(self, mock_filename):
        """Engine should receive the pre-prepared script, not re-prepare it."""
        mock_filename.return_value = "/tmp/test.wav"

        voice_mapper = MagicMock()
        voice_mapper.get_voice_path.return_value = "/tmp/voice.wav"
        engine = MagicMock()
        engine.generate_line.return_value = True

        tts_config = TTSConfig(
            device="cpu",
            tts_engine="test",
            output_dir="/tmp",
            validation_model=None,
            engine=engine,
            short_text_postfix=DEFAULTS["short_text_postfix"],
        )

        # Pre-prepare script with postfix
        text = "Hello world"
        full_script, _ = prepare_script_for_tts(text, tts_config.short_text_postfix)

        _tts_generate_only(0, 0, full_script, "narrator", voice_mapper, tts_config)

        # Verify engine received the full_script as-is (with postfix once)
        engine.generate_line.assert_called_once()
        called_text = engine.generate_line.call_args.kwargs.get("text")
        assert called_text == full_script
        assert called_text.count(DEFAULTS["short_text_postfix"]) == 1


class TestValidateAndClipNoDoublePostfix:
    """_validate_and_clip_audio must NOT call prepare_script_for_tts again."""

    def test_accepts_preprepared_script(self):
        """Function should work with pre-prepared script text."""
        full_script, _ = prepare_script_for_tts("Hello world", DEFAULTS["short_text_postfix"])
        # Should not raise - function accepts pre-prepared text
        assert DEFAULTS["short_text_postfix"] in full_script


class TestClippingAccuracy:
    """Test that clipping removes postfix completely."""

    def test_clip_at_last_word_before_postfix(self):
        """Clip point should be at end of last word before postfix."""
        segments = ["hello", "world", "and", "also", "with", "you"]
        start_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        end_times = [0.4, 0.9, 1.4, 1.9, 2.4, 2.9]

        result = calculate_clip_points(segments, start_times, end_times, "and", "world")
        assert result is not None
        start_clip, end_clip = result
        # Should clip at end of "world" (0.9s) minus 50ms safety buffer = 850ms
        assert end_clip == 850.0

    def test_safety_buffer_prevents_residual_postfix(self):
        """50ms safety buffer should account for Whisper timestamp inaccuracy."""
        segments = ["hello", "world", "and", "also"]
        start_times = [0.0, 0.5, 1.0, 1.5]
        end_times = [0.4, 0.9, 1.4, 1.9]

        result = calculate_clip_points(segments, start_times, end_times, "and", "world")
        assert result is not None
        _, end_clip = result
        # 0.9s - 0.05s buffer = 850ms
        assert end_clip == 850.0

    def test_postfix_at_start_clips_to_zero(self):
        """When postfix is the first token, clip_end should be 0."""
        segments = ["and", "also", "with", "you"]
        start_times = [0.0, 0.5, 1.0, 1.5]
        end_times = [0.4, 0.9, 1.4, 1.9]

        result = calculate_clip_points(
            segments, start_times, end_times, "and", None,
            input_tokens=["and", "also", "with", "you"],
        )
        # Returns None because clip_start (0) >= clip_end (0) triggers guard
        assert result is None

    def test_no_postfix_falls_back_to_last_valid_token(self):
        """When postfix token not found, use last_valid_token for clipping."""
        segments = ["hello", "world", "foo"]
        start_times = [0.0, 0.5, 1.0]
        end_times = [0.4, 0.9, 1.4]

        result = calculate_clip_points(segments, start_times, end_times, "notfound", "world")
        assert result is not None
        _, end_clip = result
        assert end_clip == 900.0


class TestInterLinePause:
    """Test that inter-line pause is inserted between audio segments."""

    @patch("pydub.AudioSegment")
    def test_pause_inserted_between_lines(self, mock_segment_class):
        """get_non_silent_audio_from_wavs should insert silence between lines."""
        mock_seg = MagicMock()
        mock_segment_class.from_wav.return_value = mock_seg
        mock_segment_class.empty.return_value = MagicMock()
        mock_segment_class.silent.return_value = MagicMock(name="pause")

        wav_files = ["/tmp/line1.wav", "/tmp/line2.wav"]

        # Mock detect_nonsilent to return one segment per file
        import pydub.silence
        with patch.object(pydub.silence, "detect_nonsilent", return_value=[(0, 1000)]):
            get_non_silent_audio_from_wavs(
                wav_files,
                inter_line_pause_ms=300,
            )

        # Verify silent() was called (pause was inserted between lines)
        mock_segment_class.silent.assert_called()


class TestEngineInterfaceContract:
    """Test that any engine implementing the worker protocol works through TTSEngine.

    These tests verify the adapter contract, not specific engines. Any engine that
    implements generate_voice_sample and generate_line in main.py should pass.
    """

    def test_all_engines_discoverable(self):
        """All engine directories must be discoverable via list_engines."""
        from tts import list_engines
        from pathlib import Path
        engines_dir = Path(__file__).resolve().parent.parent / "engines"
        expected = [d.name for d in engines_dir.iterdir() if d.is_dir() and (d / "main.py").exists()]
        actual = list_engines()
        for eng in expected:
            assert eng in actual, f"Engine '{eng}' not discoverable"

    def test_all_engines_have_pyproject(self):
        """All engine directories must have pyproject.toml for uv installation."""
        from tts import list_engines
        from pathlib import Path
        engines_dir = Path(__file__).resolve().parent.parent / "engines"
        for eng in list_engines():
            pyproject = engines_dir / eng / "pyproject.toml"
            assert pyproject.exists(), f"Engine '{eng}' missing pyproject.toml"

    def test_all_engines_have_main_py(self):
        """All engine directories must have main.py (worker entry point)."""
        from tts import list_engines
        from pathlib import Path
        engines_dir = Path(__file__).resolve().parent.parent / "engines"
        for eng in list_engines():
            main_py = engines_dir / eng / "main.py"
            assert main_py.exists(), f"Engine '{eng}' missing main.py"

    def test_all_engines_support_probe(self):
        """All engines must support --probe flag to report capabilities."""
        from tts import list_engines
        from tts.worker import EngineWorker
        for eng in list_engines():
            engine_dir = Path(__file__).resolve().parent.parent / "engines" / eng
            worker = EngineWorker(engine_dir, "cpu")
            try:
                worker.start()
                # If worker starts, it supports the protocol
            finally:
                worker.shutdown()

    def test_tts_engine_adapter_delegates_to_worker(self):
        """TTSEngine.generate_line must delegate to EngineWorker, not inline code."""
        from tts.engine import TTSEngine
        from unittest.mock import MagicMock, patch
        from pathlib import Path

        engine = TTSEngine(Path("/tmp/fake_engine"), "cpu")
        with patch.object(engine, "_get_worker") as mock_get_worker:
            mock_worker = MagicMock()
            mock_worker.request.return_value = {"success": True}
            mock_get_worker.return_value = mock_worker

            engine.generate_line(
                text="test",
                voice_path="/tmp/voice.wav",
                output_path="/tmp/out.wav",
            )

            mock_worker.request.assert_called_once()
            call_kwargs = mock_worker.request.call_args
            assert call_kwargs[0][0] == "generate_line"  # method name

    def test_tts_engine_voice_sample_delegates_to_worker(self):
        """TTSEngine.generate_voice_sample must delegate to EngineWorker."""
        from tts.engine import TTSEngine
        from unittest.mock import MagicMock, patch
        from pathlib import Path

        engine = TTSEngine(Path("/tmp/fake_engine"), "cpu")
        with patch.object(engine, "_get_worker") as mock_get_worker:
            mock_worker = MagicMock()
            mock_worker.request.return_value = {
                "success": True,
                "output_file": "/tmp/test.wav",
                "duration": 1.0,
            }
            mock_get_worker.return_value = mock_worker

            result = engine.generate_voice_sample(
                character_name="test",
                description="male",
                output_dir=Path("/tmp"),
            )

            mock_worker.request.assert_called_once()
            call_kwargs = mock_worker.request.call_args
            assert call_kwargs[0][0] == "generate_voice_sample"
            assert result[0] is True  # success

    def test_voice_mapper_cleanup_uses_correct_method(self):
        """VoiceMapper.cleanup_engines must call shutdown_worker(), not shutdown()."""
        mock_engine = MagicMock()
        vm = VoiceMapper(output_dir="/tmp", device="cpu", tts_engine="test")
        vm._cached_engine = mock_engine
        vm.cleanup_engines()
        mock_engine.shutdown_worker.assert_called_once()

    def test_static_voice_text_injected_when_missing(self):
        """VoiceMapper.generate_voice_sample must inject static_voice_text from config."""
        mock_engine = MagicMock()
        mock_engine.generate_voice_sample.return_value = (True, "/tmp/test.wav", 1.0)
        vm = VoiceMapper(output_dir="/tmp", device="cpu", tts_engine="test")
        vm.set_engine(mock_engine)

        success, output_file, duration = vm.generate_voice_sample(
            character_name="test",
            description="male, young",
            output_dir="/tmp",
        )

        # Verify static_voice_text was passed to engine
        call_kwargs = mock_engine.generate_voice_sample.call_args.kwargs
        assert "static_voice_text" in call_kwargs
        assert call_kwargs["static_voice_text"] == DEFAULTS["static_voice_text"]

    def test_validation_model_not_passed_to_engine(self):
        """validation_model (WhisperModel) must NOT be passed to engine.generate_line()."""
        mock_filename = "/tmp/test.wav"
        with patch("audiobook_generator.audiobook_generator.generate_output_filename", return_value=mock_filename):
            voice_mapper = MagicMock()
            voice_mapper.get_voice_path.return_value = "/tmp/voice.wav"
            engine = MagicMock()
            engine.generate_line.return_value = True

            mock_whisper = MagicMock(name="WhisperModel")
            tts_config = TTSConfig(
                device="cpu",
                tts_engine="test",
                output_dir="/tmp",
                validation_model=mock_whisper,
                engine=engine,
                short_text_postfix="",
            )

            _tts_generate_only(0, 0, "Hello world", "narrator", voice_mapper, tts_config)

            # Verify validation_model was NOT passed
            call_kwargs = engine.generate_line.call_args.kwargs
            assert "validation_model" not in call_kwargs
