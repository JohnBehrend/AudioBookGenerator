"""Tests for the Parakeet ASR backend adapter (no NeMo / model required)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from audiobook_generator import asr  # noqa: E402
from audiobook_generator.pipeline import collect_transcription_segments  # noqa: E402
from audiobook_generator.config import DEFAULTS  # noqa: E402


class TestBackendDispatch:
    def test_default_backend_is_parakeet(self):
        """DEFAULTS routes setup_validation_model (backend=None) to parakeet."""
        from unittest.mock import patch
        from audiobook_generator.audiobook_generator import setup_validation_model

        with patch("audiobook_generator.asr.load_parakeet_validation_model") as m:
            setup_validation_model("cuda:0", backend=None)
        m.assert_called_once()
        model_name = m.call_args.args[0]
        assert model_name == DEFAULTS["parakeet_model_name"]
        assert "parakeet" in model_name.lower()

    def test_load_light_asr_model_dispatches_on_backend(self):
        from unittest.mock import patch
        from audiobook_generator.asr import load_light_asr_model

        with patch("audiobook_generator.asr.ParakeetModel") as m:
            load_light_asr_model("cuda:0")
        m.assert_called_once()
        assert "parakeet" in m.call_args.args[0].lower()

        with patch.dict(DEFAULTS, {"validation_backend": "whisper"}):
            with patch("faster_whisper.WhisperModel") as wm:
                load_light_asr_model("cuda:0")
        wm.assert_called_once()
        assert wm.call_args.args[0] == "base"

    def test_local_nemo_file_uses_restore_from(self):
        """A local .nemo path (e.g. moondream/parakeet-ultra repack) loads via
        EncDecRNNTBPEModel.restore_from, not from_pretrained (which would
        treat the path as an HF repo id)."""
        from unittest.mock import patch, MagicMock
        model = asr.ParakeetModel.__new__(asr.ParakeetModel)
        with patch("nemo.collections.asr.models.EncDecRNNTBPEModel") as cls, \
             patch("torch.device"), \
             patch("torch.cuda.is_available", return_value=False):
            fake = MagicMock()
            fake.cfg.preprocessor.window_stride = 0.01
            cls.restore_from.return_value = fake
            model.__init__("/models/parakeet-ultra.nemo", device="cuda", cpu=True)
            cls.restore_from.assert_called_once()
        assert model.model is fake


class TestParseWordEntry:
    def test_seconds_dict(self):
        w = asr.parse_word_entry({"word": "Hello", "start": 0.5, "end": 0.8}, 0.01)
        assert w is not None
        assert w.word == "Hello "
        assert w.start == 0.5
        assert w.end == 0.8

    def test_offset_dict_uses_frame_stride(self):
        w = asr.parse_word_entry({"word": "world", "start_offset": 10, "end_offset": 30}, 0.01)
        assert w is not None
        assert w.word == "world "
        assert w.start == pytest.approx(0.1)
        assert w.end == pytest.approx(0.3)

    def test_object_style(self):
        class Entry:
            word = "hi"
            start = 1.0
            end = 1.5

        w = asr.parse_word_entry(Entry(), 0.01)
        assert w is not None
        assert w.word == "hi "
        assert w.start == 1.0

    def test_empty_word_rejected(self):
        assert asr.parse_word_entry({"word": "", "start": 0.0, "end": 0.1}, 0.01) is None

    def test_missing_times_rejected(self):
        assert asr.parse_word_entry({"word": "hi"}, 0.01) is None


class TestSegmentsFromTimestamps:
    def test_word_list(self):
        entries = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
        ]
        segs = asr.segments_from_timestamps(entries, 0.01, "Hello world")
        assert len(segs) == 1
        assert segs[0].start == 0.0
        assert segs[0].end == 1.0
        assert [w.word for w in segs[0].words] == ["Hello ", "world "]

    def test_full_timestamps_mapping(self):
        mapping = {"word": [{"word": "one", "start_offset": 0, "end_offset": 10}]}
        segs = asr.segments_from_timestamps(mapping, 0.02)
        assert segs[0].words[0].start == 0.0
        assert segs[0].words[0].end == pytest.approx(0.2)

    def test_empty(self):
        assert asr.segments_from_timestamps([], 0.01) == []
        assert asr.segments_from_timestamps(None, 0.01) == []


class TestWhisperCompat:
    def test_transcribe_shape_matches_faster_whisper(self):
        """ParakeetModel.transcribe returns (segments, info) usable by
        collect_transcription_segments without touching NeMo."""
        model = asr.ParakeetModel.__new__(asr.ParakeetModel)
        model._frame_stride = 0.01

        class FakeHyp:
            text = "Hello world"
            timestamps = {"word": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.6, "end": 1.0},
            ]}

        model.model = type("M", (), {"transcribe": staticmethod(lambda paths, timestamps: [FakeHyp()])})()

        # soundfile import inside transcribe needs a real file; stub duration via monkeypatch
        import unittest.mock as mock
        with mock.patch("soundfile.info") as sf_info:
            sf_info.return_value.duration = 1.2
            segments, info = model.transcribe("/fake/line.wav", beam_size=5, word_timestamps=True)

        assert isinstance(segments, list) and len(segments) == 1
        assert info.duration == pytest.approx(1.2)
        words, starts, ends = collect_transcription_segments(segments)
        assert words == ["Hello", "world"]
        assert starts == [0.0, 0.6]
        assert ends == [0.5, 1.0]
