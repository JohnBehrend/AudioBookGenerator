"""Tests for postfix handling, clipping accuracy, and engine integration."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from audiobook_generator.pipeline import (
    calculate_clip_points,
    refine_clip_end_with_energy,
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

    def test_clip_at_start_of_postfix(self):
        """Clip point should be at the start of the postfix word.

        Clipping at the start of the postfix (rather than at the end of the
        last content word minus a buffer) preserves the full final content word,
        since Whisper under-reports the final word's end time and would otherwise
        cut off its tail (e.g. the 's' in 'girls').
        """
        segments = ["hello", "world", "and", "also", "with", "you"]
        start_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        end_times = [0.4, 0.9, 1.4, 1.9, 2.4, 2.9]

        result = calculate_clip_points(segments, start_times, end_times, "and", "world")
        assert result is not None
        start_clip, end_clip = result
        # Postfix "and" starts at 1.0s -> clip there, keeping all of "world".
        assert end_clip == 1000.0

    def test_keeps_full_final_content_word_when_postfix_delayed(self):
        """A pause before the postfix keeps the final word's tail intact."""
        # Whisper reports "girls" ending early (1.0s) but the postfix "and"
        # only starts later (1.6s); clipping must reach the postfix start.
        segments = ["our", "girls", "and", "also", "with", "you"]
        start_times = [0.0, 0.3, 1.6, 1.8, 2.0, 2.2]
        end_times = [0.3, 1.0, 1.8, 2.0, 2.2, 2.4]

        result = calculate_clip_points(segments, start_times, end_times, "and", "girls")
        assert result is not None
        _, end_clip = result
        # Old logic: end of "girls" (1.0s) - 0.05 = 950ms (cut the 's').
        # New logic: clip at start of postfix "and" = 1600ms (keeps full "girls").
        assert end_clip == 1600.0

    def test_safety_buffer_prevents_residual_postfix(self):
        """Postfix start boundary should prevent residual postfix audio."""
        segments = ["hello", "world", "and", "also"]
        start_times = [0.0, 0.5, 1.0, 1.5]
        end_times = [0.4, 0.9, 1.4, 1.9]

        result = calculate_clip_points(segments, start_times, end_times, "and", "world")
        assert result is not None
        _, end_clip = result
        # Postfix "and" starts at 1.0s -> clip there.
        assert end_clip == 1000.0

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


class TestRefineClipEndWithEnergy:
    """Tests for energy-based clip-end refinement."""

    @staticmethod
    def _make_wav(path, content_ms, silence_ms, postfix_ms):
        """content tone -> silence -> postfix tone (16-bit mono)."""
        import numpy as np
        import wave
        sr = 16000
        def tone(ms):
            t = np.arange(int(sr * ms / 1000))
            return (0.3 * np.sin(2 * np.pi * 440 * t / sr) * 32767).astype(np.int16)
        def silence(ms):
            return np.zeros(int(sr * ms / 1000), dtype=np.int16)
        data = np.concatenate([tone(content_ms), silence(silence_ms), tone(postfix_ms)])
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(data.tobytes())

    def test_refines_when_clip_end_in_pause(self, tmp_path):
        # content (0-1000ms), silence (1000-1400ms), postfix (1400-2000ms)
        wav = tmp_path / "t.wav"
        self._make_wav(wav, 1000, 400, 600)
        # clip_end 1200 falls inside the pause -> refine to just before postfix onset
        result = refine_clip_end_with_energy(str(wav), 1200)
        # pause ends at 1400; minus 20ms margin = 1380
        assert result == 1380

    def test_no_refine_when_clip_end_in_speech(self, tmp_path):
        # clip_end 500 is inside the content tone (no pause) -> unchanged
        wav = tmp_path / "t.wav"
        self._make_wav(wav, 1000, 400, 600)
        assert refine_clip_end_with_energy(str(wav), 500) == 500

    def test_no_refine_when_clip_end_in_trailing_postfix_speech(self, tmp_path):
        # clip_end far past the pause (inside postfix tone) -> unchanged
        wav = tmp_path / "t.wav"
        self._make_wav(wav, 1000, 400, 600)
        assert refine_clip_end_with_energy(str(wav), 1800) == 1800

    @staticmethod
    def _make_segmented_wav(path, segments):
        """segments: list of (kind, ms) where kind is 'tone' or 'sil'."""
        import numpy as np
        import wave
        sr = 16000
        def tone(ms):
            t = np.arange(int(sr * ms / 1000))
            return (0.3 * np.sin(2 * np.pi * 440 * t / sr) * 32767).astype(np.int16)
        def silence(ms):
            return np.zeros(int(sr * ms / 1000), dtype=np.int16)
        data = np.concatenate([
            (tone if kind == "tone" else silence)(ms) for kind, ms in segments
        ])
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(data.tobytes())

    @staticmethod
    def _postfix_words(offset_ms=0):
        return [
            ("and", offset_ms + 0), ("also", offset_ms + 150),
            ("with", offset_ms + 250), ("you", offset_ms + 350),
        ]

    def test_step2_does_not_overclip_on_leading_silence(self, tmp_path):
        # content flows straight into postfix (no pause). Whisper locates the
        # postfix too early, inside the content. Energy must NOT extend into the
        # leading silence (which would cut the line) -- it stays at the onset.
        wav = tmp_path / "t.wav"
        # leading silence 200 + content 200-1000 + postfix 1000-1600
        self._make_segmented_wav(wav, [("sil", 200), ("tone", 800), ("tone", 600)])
        words = [
            ("what", 200), ("is", 300), ("his", 400), ("name", 500),
        ] + self._postfix_words(1000)
        result = refine_clip_end_with_energy(
            str(wav), 500.0, postfix_tokens=["and", "also", "with", "you"],
            word_starts_ms=words,
        )
        assert result == 500.0

    def test_step2_extends_to_verified_postfix_boundary(self, tmp_path):
        # real pause before the postfix (800-1000). Whisper locates postfix too
        # early (300, inside content). Energy extends to the postfix boundary so
        # content is not cut.
        wav = tmp_path / "t.wav"
        # content 0-800 + pause 800-1000 + postfix 1000-1600
        self._make_segmented_wav(wav, [("tone", 800), ("sil", 200), ("tone", 600)])
        words = [
            ("what", 0), ("is", 100), ("his", 200), ("name", 300),
        ] + self._postfix_words(1000)
        result = refine_clip_end_with_energy(
            str(wav), 300.0, postfix_tokens=["and", "also", "with", "you"],
            word_starts_ms=words,
        )
        assert result == 980.0


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


class TestRealVoiceGeneration:
    """Integration tests that actually generate voice samples with real engines."""

    @pytest.mark.slow
    @pytest.mark.generate
    @pytest.mark.skipif(
        not any(Path(__file__).resolve().parent.parent / "engines" / eng / "main.py"
                 for eng in ["omni", "dramabox"]),
        reason="No voice generation engines available"
    )
    def test_omni_generates_valid_wav(self, tmp_path):
        """Omni engine should produce a valid WAV file from a voice description."""
        from audiobook_generator.voice_mapper import VoiceMapper

        output_dir = tmp_path / "voices"
        output_dir.mkdir()

        vm = VoiceMapper(output_dir=str(output_dir), device="cuda:0", tts_engine="omni")
        success, output_file, duration = vm.generate_voice_sample(
            character_name="test_narrator",
            description="male, middle-aged, moderate pitch",
            output_dir=str(output_dir),
            verbose=True,
        )
        vm.cleanup_engines()

        assert success, "Voice generation should succeed"
        assert output_file is not None, "Should return output file path"
        assert Path(output_file).exists(), f"Output file should exist: {output_file}"
        assert duration > 0, f"Duration should be positive, got {duration}"

        # Verify WAV file properties
        import soundfile as sf
        info = sf.info(output_file)
        assert info.samplerate == 24000, f"Expected 24kHz sample rate, got {info.samplerate}"
        assert info.frames > 0, "WAV should have audio frames"

    @pytest.mark.slow
    @pytest.mark.generate
    @pytest.mark.skipif(
        not Path(__file__).resolve().parent.parent / "engines" / "dramabox" / "main.py",
        reason="Dramabox engine not available"
    )
    def test_dramabox_generates_valid_wav(self, tmp_path):
        """Dramabox engine should produce a valid WAV file from a voice description."""
        from audiobook_generator.voice_mapper import VoiceMapper

        output_dir = tmp_path / "voices"
        output_dir.mkdir()

        vm = VoiceMapper(output_dir=str(output_dir), device="cuda:0", tts_engine="dramabox")
        success, output_file, duration = vm.generate_voice_sample(
            character_name="test_narrator",
            description="male, middle-aged, moderate pitch",
            output_dir=str(output_dir),
            verbose=True,
        )
        vm.cleanup_engines()

        assert success, "Voice generation should succeed"
        assert output_file is not None, "Should return output file path"
        assert Path(output_file).exists(), f"Output file should exist: {output_file}"
        assert duration > 0, f"Duration should be positive, got {duration}"

        # Verify WAV file properties
        import soundfile as sf
        info = sf.info(output_file)
        assert info.samplerate == 24000, f"Expected 24kHz sample rate, got {info.samplerate}"
        assert info.frames > 0, "WAV should have audio frames"
