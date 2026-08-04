"""Tests for the OmniVoice worker's lazy ASR loading (engines/omni/main.py).

Guards the regression where the ASR model was loaded *unconditionally* on the
celebrity (voice **design**) path — even though that path never needs ASR —
wasting several GB on the shared 24GB GPU and causing OOM when Whisper
validation also needed memory.

Invariant under test: the ASR model may only be loaded through the voice
**clone** prompt path (``generate_line``), never during ``generate_voice_sample``
(design). These tests drive the real worker loop with a fake model and assert
``load_asr_model`` is not invoked on the design request but is on the clone one.
"""
import importlib.util
import io
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_OMNI_MAIN = Path(__file__).resolve().parent.parent / "engines" / "omni" / "main.py"


class FakeOmniVoice:
    """Stand-in for ``omnivoice.OmniVoice`` with recorded ASR/clone activity."""

    instances = []

    def __init__(self):
        self.asr_loaded = 0
        self.clone_prompt_calls = 0
        FakeOmniVoice.instances.append(self)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def load_asr_model(self):
        self.asr_loaded += 1

    def create_voice_clone_prompt(self, **kwargs):
        self.clone_prompt_calls += 1
        return object()

    def generate(self, **kwargs):
        # Returned audio: tuple where [0] is a plain numpy array (no .numel/.cpu).
        return (np.zeros(16000, dtype=np.float32),)


def _load_main():
    spec = importlib.util.spec_from_file_location("omni_main", _OMNI_MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _drive(requests):
    """Run the worker loop against the given JSON requests, return (responses, model)."""
    FakeOmniVoice.instances = []
    main = _load_main()

    fake_omnivoice = types.ModuleType("omnivoice")
    fake_omnivoice.OmniVoice = FakeOmniVoice

    payload = "\n".join(json.dumps(r) for r in requests) + "\n"
    stdin = io.StringIO(payload)
    stdout = io.StringIO()

    with patch.dict(sys.modules, {"omnivoice": fake_omnivoice}), \
         patch.object(sys, "stdin", stdin), \
         patch.object(sys, "stdout", stdout):
        main.run_worker("cpu")

    responses = [json.loads(l) for l in stdout.getvalue().strip().splitlines() if l.strip()]
    model = FakeOmniVoice.instances[0] if FakeOmniVoice.instances else None
    return responses, model


def _by_id(responses):
    return {r.get("id"): r for r in responses if "id" in r}


class TestAsrLazyLoading:
    def test_design_path_does_not_load_asr(self, tmp_path):
        """generate_voice_sample (celebrity/design) must never load the ASR model."""
        requests = [
            {"type": "request", "id": 1, "method": "generate_voice_sample",
             "kwargs": {"character_name": "elayne", "description": "a cool voice",
                        "output_dir": str(tmp_path), "static_voice_text": "hello"}},
            {"type": "shutdown"},
        ]
        responses, model = _drive(requests)
        assert model is not None
        by_id = _by_id(responses)
        assert by_id[1]["success"] is True
        assert model.asr_loaded == 0

    def test_clone_path_loads_asr_lazily_and_caches(self, tmp_path):
        """generate_line (clone) loads ASR once; repeat calls reuse the cache."""
        voice = str(tmp_path / "ref.wav")
        requests = [
            {"type": "request", "id": 1, "method": "generate_line",
             "kwargs": {"text": "hello", "voice_path": voice, "output_path": str(tmp_path / "o1.wav")}},
            {"type": "request", "id": 2, "method": "generate_line",
             "kwargs": {"text": "again", "voice_path": voice, "output_path": str(tmp_path / "o2.wav")}},
            {"type": "shutdown"},
        ]
        responses, model = _drive(requests)
        assert model is not None
        by_id = _by_id(responses)
        assert by_id[1]["success"] is True
        assert by_id[2]["success"] is True
        # ASR loaded once (first distinct voice); clone prompt computed once and cached.
        assert model.asr_loaded == 1
        assert model.clone_prompt_calls == 1

    def test_design_then_clone_loads_asr_once_total(self, tmp_path):
        """ASR stays unloaded across a design request, then loads for the first
        clone request — i.e. design never triggers it."""
        voice = str(tmp_path / "ref.wav")
        requests = [
            {"type": "request", "id": 1, "method": "generate_voice_sample",
             "kwargs": {"character_name": "eline", "description": "warm",
                        "output_dir": str(tmp_path), "static_voice_text": "hi"}},
            {"type": "request", "id": 2, "method": "generate_line",
             "kwargs": {"text": "hi", "voice_path": voice, "output_path": str(tmp_path / "o.wav")}},
            {"type": "shutdown"},
        ]
        responses, model = _drive(requests)
        by_id = _by_id(responses)
        assert by_id[1]["success"] is True
        assert by_id[2]["success"] is True
        # Only the clone request caused the ASR load.
        assert model.asr_loaded == 1
