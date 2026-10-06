"""ASR backends for audio validation and karaoke alignment.

Whisper (faster-whisper) is the historical validation backend. This module
adds Parakeet (NVIDIA TDT/FastConformer) as an alternative. Parakeet is a
transducer, so word timestamps are produced natively during decoding (no
alignment pass, no beam-search word timestamps) and it decodes long-form
audio roughly an order of magnitude faster than whisper on the same GPU.

The ParakeetModel wrapper deliberately mimics faster-whisper's return shape:
``model.transcribe(path)`` yields ``(segments, info)`` where each segment is
an object exposing ``.words`` (each word with ``.word``/``.start``/``.end``).
All existing call sites (utils.transcribe_audio_with_whisper,
transcribe_audio_for_ref_text, tts.WhisperPool, pipeline validation) work
unchanged against either backend.

Requires: nemo_toolkit[asr] (only imported when the parakeet backend is
actually used, so whisper-only installs stay lean).
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Optional, Tuple


@dataclasses.dataclass
class Word:
    """Mimics faster_whisper.transcribe.Word."""

    word: str
    start: float
    end: float


@dataclasses.dataclass
class Segment:
    """Mimics faster_whisper.transcribe.Segment (word-timestamp subset)."""

    start: float
    end: float
    text: str
    words: List[Word]


@dataclasses.dataclass
class Info:
    """Mimics faster_whisper.transcribe.TranscriptionInfo (subset)."""

    duration: float = 0.0
    language: str = ""
    language_probability: float = 0.0


def parse_word_entry(entry: Any, frame_stride: float) -> Optional[Word]:
    """Convert a NeMo word-timestamp entry to a Word.

    NeMo has used several entry shapes across versions:
      * dict with 'word'/'start'/'end' in seconds
      * dict with 'word'/'start_offset'/'end_offset' as frame indices
      * objects with the same fields as attributes
    """
    def get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    word = get(entry, "word")
    if not word:
        return None
    start = get(entry, "start")
    end = get(entry, "end")
    if start is None or end is None:
        so = get(entry, "start_offset")
        eo = get(entry, "end_offset")
        if so is None or eo is None:
            return None
        start, end = so * frame_stride, eo * frame_stride
    return Word(word=str(word).strip() + " ", start=float(start), end=float(end))


def segments_from_timestamps(word_entries: Any, frame_stride: float,
                             text: str = "") -> List[Segment]:
    """Build whisper-shaped segments from NeMo word timestamp entries.

    ``word_entries`` may be the full ``hyp.timestamps`` mapping or just its
    'word' list. Callers in this codebase only read per-word data
    (collect_transcription_segments), so the grouping is cosmetic: one
    segment holding all words.
    """
    if isinstance(word_entries, dict):
        word_entries = word_entries.get("word", [])

    words = [w for w in (parse_word_entry(e, frame_stride) for e in word_entries or []) if w]
    if not words:
        return []
    return [Segment(start=words[0].start, end=words[-1].end, text=text, words=words)]


class ParakeetModel:
    """NeMo Parakeet ASR wrapped to the faster-whisper call signature.

    ``transcribe(audio_path, beam_size=..., word_timestamps=True)`` returns
    ``(segments, info)``; whisper-only kwargs (beam_size, word_timestamps,
    language, ...) are accepted and ignored.
    """

    backend = "parakeet"

    def __init__(self, model_name: str, device: str = "cuda", cpu: bool = False):
        import torch
        from nemo.collections.asr.models import ASRModel

        self.model_name = model_name
        # Word timestamps arrive as frame indices; parakeet's preprocessor
        # strides by 10ms, but read the configured value rather than assume.
        self._frame_stride = 0.01

        # NeMo logs per-file "Timestamps requested..." info lines through its
        # own 'NeMo' logger; silence it for batch-of-one production calls.
        import logging
        logging.getLogger("NeMo").setLevel(logging.WARNING)

        if model_name.endswith(".nemo"):
            from nemo.collections.asr.models import EncDecRNNTBPEModel

            self.model = EncDecRNNTBPEModel.restore_from(model_name, map_location="cpu")
        else:
            self.model = ASRModel.from_pretrained(model_name=model_name, map_location="cpu")
        self.model.eval()
        if cpu or not str(device).startswith("cuda"):
            self.device = "cpu"
            self.model.to(torch.device("cpu"))
        else:
            # torch accepts "cuda:N" directly; bare "cuda" means the default
            # device (respects CUDA_VISIBLE_DEVICES pinning like the whisper
            # path does).
            self.device = "cuda:0" if device == "cuda" else device
            self.model.to(torch.device(self.device))
        try:
            self._frame_stride = float(self.model.cfg.preprocessor.window_stride)
        except Exception:
            pass

    def transcribe(self, audio_path: str, beam_size: int = 5,
                   word_timestamps: bool = True, **kwargs) -> Tuple[List[Segment], Info]:
        import soundfile as sf

        hyps = self.model.transcribe([audio_path], timestamps=True)
        hyp = hyps[0]

        # NeMo >= 2.x exposes word timestamps on hyp.timestamp (dict with
        # 'word'/'segment'/'char' keys); older versions used hyp.timestamps.
        timestamps = getattr(hyp, "timestamp", None)
        if timestamps is None:
            timestamps = getattr(hyp, "timestamps", None) or {}
        hyp_text = getattr(hyp, "text", "") or ""
        segments = segments_from_timestamps(timestamps.get("word", []),
                                            self._frame_stride, hyp_text)

        try:
            duration = float(sf.info(audio_path).duration)
        except Exception:
            duration = segments[0].end if segments else 0.0
        return segments, Info(duration=duration)


def load_parakeet_validation_model(model_name: str, device: str = "cuda",
                                   cpu: bool = False) -> ParakeetModel:
    """Load the Parakeet validation model (parity with setup_validation_model)."""
    return ParakeetModel(model_name, device=device, cpu=cpu)


def load_light_asr_model(device: str = "cuda") -> Any:
    """Cheap ASR model for auxiliary transcription (celebrity clip search).

    Dispatches on DEFAULTS["validation_backend"] like setup_validation_model,
    but always uses the lightest model for the chosen backend: whisper 'base'
    or the Parakeet validation model (which is already small at 0.6B).
    """
    from .config import DEFAULTS

    backend = (DEFAULTS.get("validation_backend") or "whisper").lower()
    if backend == "parakeet":
        model_name = DEFAULTS.get("parakeet_model_name") or "nvidia/parakeet-tdt-0.6b-v3"
        return ParakeetModel(model_name, device=device)
    from faster_whisper import WhisperModel
    return WhisperModel("base", device=device, compute_type="float16")
