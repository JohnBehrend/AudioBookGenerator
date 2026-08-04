"""Tests for celebrity_voices module - per-video validation flow."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from audiobook_generator.celebrity_voices import (
    _extract_segment_from_audio,
    find_and_extract_video_segment,
    validate_celebrity_segment,
    generate_celebrity_reference,
    build_celebrity_voice,
    match_celebrity,
    save_celebrity_voice_as,
    _retry_llm_call,
)
from audiobook_generator.testing import write_silence_wav


class TestExtractSegmentFromAudio:
    """Tests for _extract_segment_from_audio helper."""

    def test_returns_false_for_missing_input(self, temp_dir):
        """Test that missing input file returns False."""
        result = _extract_segment_from_audio(
            str(temp_dir / "nonexistent.wav"),
            0.0,
            5.0,
            str(temp_dir / "output.wav"),
        )
        assert result is False

    def test_returns_false_for_missing_ffmpeg(self, temp_dir):
        """Test that missing ffmpeg returns False."""
        input_path = temp_dir / "input.wav"
        write_silence_wav(input_path, 22050, 1)

        with patch("audiobook_generator.celebrity_voices.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ffmpeg not found")
            result = _extract_segment_from_audio(
                str(input_path),
                0.0,
                5.0,
                str(temp_dir / "output.wav"),
            )
        assert result is False

    def test_returns_true_on_success(self, temp_dir):
        """Test that successful extraction returns True."""
        input_path = temp_dir / "input.wav"
        write_silence_wav(input_path, 22050, 10)

        with patch("audiobook_generator.celebrity_voices.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            output_path = temp_dir / "output.wav"
            output_path.touch()
            result = _extract_segment_from_audio(
                str(input_path),
                0.0,
                5.0,
                str(output_path),
            )
        assert result is True


class TestFindAndExtractVideoSegment:
    """Tests for find_and_extract_video_segment."""

    def test_returns_none_when_download_fails(self, temp_dir, mock_llm_client):
        """Test that download failure returns (None, None)."""
        with patch("audiobook_generator.celebrity_voices.download_celebrity_audio") as mock_dl:
            mock_dl.return_value = None
            seg, audio = find_and_extract_video_segment(
                client=mock_llm_client,
                model="test",
                search_query="test celebrity interview",
                celebrity="Test Celebrity",
                description='{"gender": "male", "style": "deep"}',
                output_dir=str(temp_dir),
                file_prefix="test_0",
                verbose=False,
            )
        assert seg is None
        assert audio is None

    def test_calls_download_celebrity_audio(self, temp_dir, mock_llm_client):
        """Test that download_celebrity_audio is called with correct args."""
        with patch("audiobook_generator.celebrity_voices.download_celebrity_audio") as mock_dl:
            mock_dl.return_value = None
            find_and_extract_video_segment(
                client=mock_llm_client,
                model="test",
                search_query="test celebrity interview",
                celebrity="Test Celebrity",
                description='{"gender": "male", "style": "deep"}',
                output_dir=str(temp_dir),
                file_prefix="test_0",
                verbose=False,
            )
            mock_dl.assert_called_once()
            call_args = mock_dl.call_args
            assert call_args[1]["search_query"] == "test celebrity interview"
            assert call_args[1]["file_prefix"] == "test_0"

    def test_falls_back_to_silence_detection(self, temp_dir, mock_llm_client):
        """Test that silence detection is used when Whisper+LLM fails."""
        audio_path = temp_dir / "downloaded.wav"
        write_silence_wav(audio_path, 22050, 10)

        with patch("audiobook_generator.celebrity_voices.download_celebrity_audio") as mock_dl, \
             patch("audiobook_generator.celebrity_voices.identify_celebrity_segments") as mock_id, \
             patch("audiobook_generator.celebrity_voices.extract_speech_segments") as mock_extract:
            mock_dl.return_value = str(audio_path)
            mock_id.return_value = []
            seg_path = temp_dir / "test_0_segment_0.wav"
            seg_path.touch()
            mock_extract.return_value = [str(seg_path)]

            seg, audio = find_and_extract_video_segment(
                client=mock_llm_client,
                model="test",
                search_query="test",
                celebrity="Test Celebrity",
                description='{"gender": "male"}',
                output_dir=str(temp_dir),
                file_prefix="test_0",
                whisper_model=MagicMock(),
                verbose=False,
            )

        assert seg is not None
        assert audio is not None


class TestValidateCelebritySegment:
    """Tests for validate_celebrity_segment."""

    def test_returns_false_for_missing_file(self, temp_dir):
        """Test that missing file returns False."""
        is_valid, reason = validate_celebrity_segment(
            str(temp_dir / "nonexistent.wav"),
            "male voice",
        )
        assert is_valid is False
        assert "does not exist" in reason

    def test_returns_false_for_tiny_file(self, temp_dir):
        """Test that tiny file returns False."""
        tiny = temp_dir / "tiny.wav"
        tiny.write_bytes(b"tiny")
        is_valid, reason = validate_celebrity_segment(
            str(tiny),
            "male voice",
        )
        assert is_valid is False
        assert "too small" in reason

    def test_passes_without_chunkformer(self, temp_dir):
        """Test that validation passes when no ChunkFormer is provided."""
        seg_path = temp_dir / "segment.wav"
        write_silence_wav(seg_path, 22050, 3)

        is_valid, reason = validate_celebrity_segment(
            str(seg_path),
            "male voice",
            chunkformer_model=None,
        )
        assert is_valid is True

    def test_chunkformer_gender_mismatch(self, temp_dir):
        """Test that gender mismatch is detected by ChunkFormer."""
        seg_path = temp_dir / "segment.wav"
        write_silence_wav(seg_path, 22050, 3)

        mock_cf = MagicMock()
        mock_cf.classify_audio.return_value = {
            "gender": {"label": "male", "prob": 0.95},
            "age": {"label": "middle age", "prob": 0.8},
            "emotion": {"label": "neutral", "prob": 0.7},
        }

        is_valid, reason = validate_celebrity_segment(
            str(seg_path),
            "female voice",
            chunkformer_model=mock_cf,
        )
        assert is_valid is False
        assert "gender mismatch" in reason.lower()

    def test_chunkformer_low_confidence_ignored(self, temp_dir):
        """Test that low-confidence mismatch is ignored."""
        seg_path = temp_dir / "segment.wav"
        write_silence_wav(seg_path, 22050, 3)

        mock_cf = MagicMock()
        mock_cf.classify_audio.return_value = {
            "gender": {"label": "male", "prob": 0.5},
            "age": {"label": "middle age", "prob": 0.8},
            "emotion": {"label": "neutral", "prob": 0.7},
        }

        is_valid, reason = validate_celebrity_segment(
            str(seg_path),
            "female voice",
            chunkformer_model=mock_cf,
        )
        assert is_valid is True

    def test_chunkformer_match_passes(self, temp_dir):
        """Test that matching gender passes validation."""
        seg_path = temp_dir / "segment.wav"
        write_silence_wav(seg_path, 22050, 3)

        mock_cf = MagicMock()
        mock_cf.classify_audio.return_value = {
            "gender": {"label": "female", "prob": 0.95},
            "age": {"label": "middle age", "prob": 0.8},
            "emotion": {"label": "neutral", "prob": 0.7},
        }

        is_valid, reason = validate_celebrity_segment(
            str(seg_path),
            "female voice",
            chunkformer_model=mock_cf,
        )
        assert is_valid is True


class TestGenerateCelebrityReference:
    """Tests for generate_celebrity_reference."""

    def test_returns_none_on_engine_failure(self, temp_dir, mock_tts_engine_failure):
        """Test that engine failure returns (None, 0.0)."""
        seg_path = temp_dir / "segment.wav"
        write_silence_wav(seg_path, 22050, 3)

        ref, dur = generate_celebrity_reference(
            segment_path=str(seg_path),
            character="test_char",
            output_dir=str(temp_dir),
            engine=mock_tts_engine_failure,
            static_text="test text",
        )
        assert ref is None
        assert dur == 0.0

    def test_returns_path_on_success(self, temp_dir):
        """Test that successful generation returns path and duration."""
        from audiobook_generator.testing import MockTTSEngine
        seg_path = temp_dir / "segment.wav"
        write_silence_wav(seg_path, 22050, 3)

        # Use 5s duration so output file is large enough to pass 2s threshold
        long_engine = MockTTSEngine(duration=5.0, sample_rate=22050)
        ref, dur = generate_celebrity_reference(
            segment_path=str(seg_path),
            character="test_char",
            output_dir=str(temp_dir),
            engine=long_engine,
            static_text="test text",
        )
        assert ref is not None
        assert dur > 0

    def test_returns_none_for_too_short_reference(self, temp_dir):
        """Test that too-short reference returns (None, 0.0)."""
        seg_path = temp_dir / "segment.wav"
        write_silence_wav(seg_path, 22050, 3)

        short_engine = MagicMock()
        short_engine.generate_line.return_value = True
        short_engine.generate_line.side_effect = lambda **kw: (
            Path(kw["output_path"]).touch() or True
        )
        tiny_path = temp_dir / "test_char_ref.wav"
        tiny_path.write_bytes(b"tiny")

        def mock_gen(**kw):
            Path(kw["output_path"]).write_bytes(b"tiny")
            return True

        short_engine.generate_line = mock_gen

        ref, dur = generate_celebrity_reference(
            segment_path=str(seg_path),
            character="test_char",
            output_dir=str(temp_dir),
            engine=short_engine,
            static_text="test text",
        )
        assert ref is None


class TestSaveCelebrityVoiceAs:
    """Tests for save_celebrity_voice_as — the traceability writer.

    Intended behavior: a celebrity's winning voice is saved under the CELEBRITY's
    name (``{celebrity}_ref.wav``), not the character's name, so that the
    ``voices_map`` entry for a character is directly traceable to the celebrity.
    """

    def _src(self, temp_dir):
        src = temp_dir / "src.wav"
        write_silence_wav(src, 22050, 1)
        return src

    def test_writes_celebrity_named_ref_file(self, temp_dir):
        src = self._src(temp_dir)
        dest = save_celebrity_voice_as("Johnny Depp", str(src), str(temp_dir))
        assert os.path.basename(dest) == "johnny_depp_ref.wav"
        assert os.path.exists(dest)
        assert os.path.getsize(dest) == os.path.getsize(src)

    def test_normalizes_whitespace_and_case(self, temp_dir):
        src = self._src(temp_dir)
        dest = save_celebrity_voice_as("  IAN MCKELLEN  ", str(src), str(temp_dir))
        assert os.path.basename(dest) == "ian_mckellen_ref.wav"

    def test_strips_punctuation_and_non_ascii(self, temp_dir):
        src = self._src(temp_dir)
        dest = save_celebrity_voice_as("Máry-Ann O'Brien", str(src), str(temp_dir))
        assert os.path.basename(dest) == "mry-ann_obrien_ref.wav"

    def test_returns_absolute_path_inside_output_dir(self, temp_dir):
        src = self._src(temp_dir)
        dest = save_celebrity_voice_as("Test Celebrity", str(src), str(temp_dir))
        assert os.path.dirname(dest) == str(temp_dir)


class TestBuildCelebrityVoice:
    """Tests for build_celebrity_voice with per-video validation."""

    def test_returns_none_when_no_celebrity_match(self, temp_dir, mock_llm_client):
        """Test that LLM match failure returns (None, None)."""
        with patch("audiobook_generator.celebrity_voices.match_celebrity") as mock_match:
            mock_match.return_value = None
            ref, meta = build_celebrity_voice(
                client=mock_llm_client,
                model="test",
                character="test_char",
                description='{"gender": "male"}',
                output_dir=str(temp_dir),
            )
        assert ref is None
        assert meta is None

    def test_uses_pre_matched_celebrity(self, temp_dir, mock_llm_client):
        """Test that pre-matched celebrity skips LLM matching."""
        from audiobook_generator.testing import MockTTSEngine

        seg_path = temp_dir / "test_char_v0_segment.wav"
        write_silence_wav(seg_path, 22050, 3)

        long_engine = MockTTSEngine(duration=5.0, sample_rate=22050)
        with patch("audiobook_generator.celebrity_voices.find_and_extract_video_segment") as mock_find, \
             patch("audiobook_generator.celebrity_voices.match_celebrity") as mock_match:
            mock_find.return_value = (str(seg_path), str(seg_path))
            ref, meta = build_celebrity_voice(
                client=mock_llm_client,
                model="test",
                character="test_char",
                description='{"gender": "male", "style": "deep"}',
                output_dir=str(temp_dir),
                pre_matched_celebrity="Test Celebrity",
                tts_engine=long_engine,
                verbose=False,
            )
            mock_match.assert_not_called()

        assert meta is not None
        assert meta["celebrity"] == "Test Celebrity"

    def test_exits_early_on_first_valid_video(self, temp_dir, mock_llm_client, mock_tts_engine):
        """Test that valid first video returns immediately."""
        seg_path = temp_dir / "test_char_v0_segment.wav"
        write_silence_wav(seg_path, 22050, 3)

        with patch("audiobook_generator.celebrity_voices.match_celebrity") as mock_match, \
             patch("audiobook_generator.celebrity_voices.find_and_extract_video_segment") as mock_find, \
             patch("audiobook_generator.celebrity_voices.validate_celebrity_segment") as mock_val, \
             patch("audiobook_generator.celebrity_voices.generate_celebrity_reference") as mock_gen:
            mock_match.return_value = {
                "celebrity": "Test Celebrity",
                "reason": "test",
                "search_query": "Test Celebrity interview",
            }
            mock_find.return_value = (str(seg_path), str(seg_path))
            mock_val.return_value = (True, "passed")
            ref_path = temp_dir / "test_char_v0_ref.wav"
            ref_path.touch()
            mock_gen.return_value = (str(ref_path), 5.0)

            ref, meta = build_celebrity_voice(
                client=mock_llm_client,
                model="test",
                character="test_char",
                description='{"gender": "male"}',
                output_dir=str(temp_dir),
                tts_engine=mock_tts_engine,
                verbose=False,
            )

        assert mock_find.call_count == 1
        assert ref is not None

    def test_falls_back_to_next_video_on_validation_failure(self, temp_dir, mock_llm_client, mock_tts_engine):
        """Test that failed validation tries next video."""
        seg1 = temp_dir / "test_char_v0_segment.wav"
        seg2 = temp_dir / "test_char_v1_segment.wav"
        write_silence_wav(seg1, 22050, 3)
        write_silence_wav(seg2, 22050, 3)

        with patch("audiobook_generator.celebrity_voices.match_celebrity") as mock_match, \
             patch("audiobook_generator.celebrity_voices.find_and_extract_video_segment") as mock_find, \
             patch("audiobook_generator.celebrity_voices.validate_celebrity_segment") as mock_val, \
             patch("audiobook_generator.celebrity_voices.generate_celebrity_reference") as mock_gen:
            mock_match.return_value = {
                "celebrity": "Test Celebrity",
                "reason": "test",
                "search_query": "Test Celebrity interview",
            }
            mock_find.side_effect = [
                (str(seg1), str(seg1)),
                (str(seg2), str(seg2)),
            ]
            mock_val.side_effect = [
                (False, "gender mismatch"),
                (True, "passed"),
            ]
            ref_path = temp_dir / "test_char_v1_ref.wav"
            ref_path.touch()
            mock_gen.return_value = (str(ref_path), 5.0)

            ref, meta = build_celebrity_voice(
                client=mock_llm_client,
                model="test",
                character="test_char",
                description='{"gender": "male"}',
                output_dir=str(temp_dir),
                max_videos=3,
                tts_engine=mock_tts_engine,
                verbose=False,
            )

        assert mock_find.call_count == 2
        assert ref is not None

    def test_falls_back_to_first_when_all_fail(self, temp_dir, mock_llm_client, mock_tts_engine):
        """Test that all failures returns first segment."""
        seg1 = temp_dir / "test_char_v0_segment.wav"
        seg2 = temp_dir / "test_char_v1_segment.wav"
        write_silence_wav(seg1, 22050, 3)
        write_silence_wav(seg2, 22050, 3)

        with patch("audiobook_generator.celebrity_voices.match_celebrity") as mock_match, \
             patch("audiobook_generator.celebrity_voices.find_and_extract_video_segment") as mock_find, \
             patch("audiobook_generator.celebrity_voices.validate_celebrity_segment") as mock_val, \
             patch("audiobook_generator.celebrity_voices.generate_celebrity_reference") as mock_gen:
            mock_match.return_value = {
                "celebrity": "Test Celebrity",
                "reason": "test",
                "search_query": "Test Celebrity interview",
            }
            mock_find.side_effect = [
                (str(seg1), str(seg1)),
                (str(seg2), str(seg2)),
            ]
            mock_val.return_value = (False, "gender mismatch")
            mock_gen.return_value = (None, 0.0)

            ref, meta = build_celebrity_voice(
                client=mock_llm_client,
                model="test",
                character="test_char",
                description='{"gender": "male"}',
                output_dir=str(temp_dir),
                max_videos=2,
                tts_engine=mock_tts_engine,
                verbose=False,
            )

        assert mock_find.call_count == 2
        assert ref is not None

    def test_returns_none_when_all_downloads_fail(self, temp_dir, mock_llm_client, mock_tts_engine):
        """Test that all download failures returns (None, None)."""
        with patch("audiobook_generator.celebrity_voices.match_celebrity") as mock_match, \
             patch("audiobook_generator.celebrity_voices.find_and_extract_video_segment") as mock_find:
            mock_match.return_value = {
                "celebrity": "Test Celebrity",
                "reason": "test",
                "search_query": "Test Celebrity interview",
            }
            mock_find.return_value = (None, None)

            ref, meta = build_celebrity_voice(
                client=mock_llm_client,
                model="test",
                character="test_char",
                description='{"gender": "male"}',
                output_dir=str(temp_dir),
                max_videos=2,
                tts_engine=mock_tts_engine,
                verbose=False,
            )

        assert ref is None
        assert meta is None

    def test_returns_segment_directly_without_tts_engine(self, temp_dir, mock_llm_client):
        """Test that segment is returned directly when no TTS engine."""
        seg_path = temp_dir / "test_char_v0_segment.wav"
        write_silence_wav(seg_path, 22050, 3)

        with patch("audiobook_generator.celebrity_voices.match_celebrity") as mock_match, \
             patch("audiobook_generator.celebrity_voices.find_and_extract_video_segment") as mock_find, \
             patch("audiobook_generator.celebrity_voices.validate_celebrity_segment") as mock_val:
            mock_match.return_value = {
                "celebrity": "Test Celebrity",
                "reason": "test",
                "search_query": "Test Celebrity interview",
            }
            mock_find.return_value = (str(seg_path), str(seg_path))
            mock_val.return_value = (True, "passed")

            ref, meta = build_celebrity_voice(
                client=mock_llm_client,
                model="test",
                character="test_char",
                description='{"gender": "male"}',
                output_dir=str(temp_dir),
                tts_engine=None,
                verbose=False,
            )

        assert ref is not None
        assert meta is not None
        # Intended behavior: the returned voice is named by the CELEBRITY (not the
        # character), so voices_map is directly traceable to the celebrity.
        assert os.path.basename(ref) == "test_celebrity_ref.wav"
        assert os.path.exists(ref)

    def test_returns_celebrity_named_ref_with_tts_engine(self, temp_dir, mock_llm_client, mock_tts_engine):
        """TTS-engine path also returns a celebrity-named reference."""
        seg_path = temp_dir / "test_char_v0_segment.wav"
        write_silence_wav(seg_path, 22050, 3)

        with patch("audiobook_generator.celebrity_voices.match_celebrity") as mock_match, \
             patch("audiobook_generator.celebrity_voices.find_and_extract_video_segment") as mock_find, \
             patch("audiobook_generator.celebrity_voices.validate_celebrity_segment") as mock_val, \
             patch("audiobook_generator.celebrity_voices.generate_celebrity_reference") as mock_gen:
            mock_match.return_value = {
                "celebrity": "Test Celebrity",
                "reason": "test",
                "search_query": "Test Celebrity interview",
            }
            mock_find.return_value = (str(seg_path), str(seg_path))
            mock_val.return_value = (True, "passed")
            ref_path = temp_dir / "test_char_v0_ref.wav"
            ref_path.touch()
            mock_gen.return_value = (str(ref_path), 5.0)

            ref, meta = build_celebrity_voice(
                client=mock_llm_client,
                model="test",
                character="test_char",
                description='{"gender": "male"}',
                output_dir=str(temp_dir),
                tts_engine=mock_tts_engine,
                verbose=False,
            )

        assert ref is not None
        assert meta is not None
        assert os.path.basename(ref) == "test_celebrity_ref.wav"
        assert os.path.exists(ref)

    def test_metadata_contains_required_fields(self, temp_dir, mock_llm_client, mock_tts_engine):
        """Test that metadata dict contains all required fields."""
        seg_path = temp_dir / "test_char_v0_segment.wav"
        write_silence_wav(seg_path, 22050, 3)

        with patch("audiobook_generator.celebrity_voices.match_celebrity") as mock_match, \
             patch("audiobook_generator.celebrity_voices.find_and_extract_video_segment") as mock_find, \
             patch("audiobook_generator.celebrity_voices.validate_celebrity_segment") as mock_val, \
             patch("audiobook_generator.celebrity_voices.generate_celebrity_reference") as mock_gen:
            mock_match.return_value = {
                "celebrity": "Test Celebrity",
                "reason": "test reason",
                "search_query": "Test Celebrity interview",
            }
            mock_find.return_value = (str(seg_path), str(seg_path))
            mock_val.return_value = (True, "passed")
            ref_path = temp_dir / "test_char_v0_ref.wav"
            ref_path.touch()
            mock_gen.return_value = (str(ref_path), 5.0)

            ref, meta = build_celebrity_voice(
                client=mock_llm_client,
                model="test",
                character="test_char",
                description='{"gender": "male"}',
                output_dir=str(temp_dir),
                tts_engine=mock_tts_engine,
                verbose=False,
            )

        assert "character" in meta
        assert "celebrity" in meta
        assert "reason" in meta
        assert "search_query" in meta
        assert "segment" in meta
        assert "audio_source" in meta
        # Intended behavior: the returned voice file is named by the celebrity.
        assert os.path.basename(ref) == "test_celebrity_ref.wav"


class TestRetryLLMCall:
    """Tests for _retry_llm_call helper."""

    def test_returns_result_on_success(self):
        """Test that successful call returns result immediately."""
        def success_func():
            return "result"

        result = _retry_llm_call(success_func, max_retries=3)
        assert result == "result"

    def test_retries_on_connection_error(self):
        """Test that connection errors trigger retries."""
        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("timeout")
            return "result"

        result = _retry_llm_call(flaky_func, max_retries=3, backoff=0.01)
        assert result == "result"
        assert call_count == 3

    def test_returns_none_after_max_retries(self):
        """Test that max retries returns None."""
        def always_fail():
            raise ConnectionError("timeout")

        result = _retry_llm_call(always_fail, max_retries=3, backoff=0.01)
        assert result is None

    def test_raises_non_connection_error(self):
        """Test that non-connection errors are raised immediately."""
        def value_error_func():
            raise ValueError("bad input")

        with pytest.raises(ValueError):
            _retry_llm_call(value_error_func, max_retries=3)


class TestMatchCelebrity:
    """Tests for match_celebrity function."""

    def test_parses_valid_json_response(self, mock_llm_client):
        """Test that valid JSON response is parsed correctly."""
        mock_llm_client.set_response({
            "role": "assistant",
            "content": '{"celebrity": "Ryan Reynolds", "reason": "matches", "search_query": "Ryan Reynolds interview"}'
        })

        result = match_celebrity(
            client=mock_llm_client,
            model="test",
            character="test_char",
            description="male voice",
        )

        assert result is not None
        assert result["celebrity"] == "Ryan Reynolds"
        assert "search_query" in result

    def test_handles_json_in_markdown(self, mock_llm_client):
        """Test that JSON wrapped in markdown is extracted."""
        mock_llm_client.set_response({
            "role": "assistant",
            "content": "Here's the result:\n```json\n{\"celebrity\": \"Test\", \"reason\": \"r\", \"search_query\": \"Test interview\"}\n```"
        })

        result = match_celebrity(
            client=mock_llm_client,
            model="test",
            character="test_char",
            description="male voice",
        )

        assert result is not None
        assert result["celebrity"] == "Test"

    def test_returns_none_on_invalid_response(self, mock_llm_client):
        """Test that invalid response returns None after retries."""
        mock_llm_client.set_response({
            "role": "assistant",
            "content": "not valid json at all"
        })

        result = match_celebrity(
            client=mock_llm_client,
            model="test",
            character="test_char",
            description="male voice",
            max_retries=2,
        )

        assert result is None

    def test_handles_dict_description(self, mock_llm_client):
        """Test that dict description is serialized to JSON."""
        mock_llm_client.set_response({
            "role": "assistant",
            "content": '{"celebrity": "Test", "reason": "r", "search_query": "Test interview"}'
        })

        result = match_celebrity(
            client=mock_llm_client,
            model="test",
            character="test_char",
            description={"gender": "male", "style": "deep"},
        )

        assert result is not None
