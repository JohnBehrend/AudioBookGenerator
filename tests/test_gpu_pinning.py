"""Tests for the GPU pinning / device-routing critical infrastructure.

These tests guard against the class of production failure where a TTS engine
worker (or Whisper validation model) silently targets the WRONG GPU, causing
CUDA out-of-memory errors and forcing the pipeline onto low-quality fallback
engines (e.g. zonos2) for celebrity voice references.

Key invariant under test: every torch-backed component (engine workers and the
Whisper validation model) must be routed to the exact device the pipeline was
configured with. When the device enumeration used by an engine venv differs
from the host's ``nvidia-smi`` order, the process must be pinned via
``CUDA_VISIBLE_DEVICES`` so that torch sees the intended GPU as ``cuda:0``.
"""

import json
from unittest.mock import patch, MagicMock, call

import pytest

from audiobook_generator.audiobook_generator import setup_validation_model
from audiobook_generator.config import DEFAULTS
from audiobook_generator.voice_mapper import VoiceMapper


# ============================================================================
# Whisper validation model device routing (the "30x faster on GPU" guarantee)
# ============================================================================

class TestWhisperDeviceRouting:
    """Whisper validation model must run on GPU by default, CPU only when asked.

    faster_whisper/ctranslate2 only accepts bare device strings ("cuda"/"cpu"),
    NOT "cuda:N". The GPU index is selected via CUDA_VISIBLE_DEVICES pinning, so
    a cuda device is normalized to bare "cuda" (which then maps to the pinned
    GPU). Passing "cuda:N" would raise ValueError: unsupported device.
    """

    def test_gpu_path_uses_cuda_and_float16(self):
        """cpu=False must load Whisper on bare 'cuda' in float16 (GPU path)."""
        with patch("faster_whisper.WhisperModel") as mock_whisper:
            setup_validation_model(device="cuda:0", cpu=False, fast=False)
        mock_whisper.assert_called_once_with(
            DEFAULTS["validation_model_name"],
            device="cuda",
            compute_type="float16",
        )

    def test_cuda_index_normalized_to_bare_cuda(self):
        """'cuda:N' must be normalized to bare 'cuda' (faster_whisper limitation)."""
        with patch("faster_whisper.WhisperModel") as mock_whisper:
            setup_validation_model(device="cuda:2", cpu=False, fast=False)
        mock_whisper.assert_called_once_with(
            DEFAULTS["validation_model_name"],
            device="cuda",
            compute_type="float16",
        )

    def test_fast_uses_smaller_model(self):
        """fast=True must select the faster (smaller) model, still on GPU float16."""
        with patch("faster_whisper.WhisperModel") as mock_whisper:
            setup_validation_model(device="cuda:1", cpu=False, fast=True)
        mock_whisper.assert_called_once_with(
            DEFAULTS["validation_model_name_fast"],
            device="cuda",
            compute_type="float16",
        )

    def test_cpu_path_uses_cpu_and_float32(self):
        """cpu=True must force CPU + float32 (the slow ~30x path)."""
        with patch("faster_whisper.WhisperModel") as mock_whisper:
            setup_validation_model(device="cuda:0", cpu=True, fast=False)
        mock_whisper.assert_called_once_with(
            DEFAULTS["validation_model_name"],
            device="cpu",
            compute_type="float32",
        )

    def test_gpu_device_normalized_to_bare_cuda(self):
        """A specific 'cuda:N' device must be normalized to bare 'cuda', not kept."""
        with patch("faster_whisper.WhisperModel") as mock_whisper:
            setup_validation_model(device="cuda:3", cpu=False, fast=False)
        # faster_whisper would reject "cuda:3"; it must be normalized to "cuda".
        args, kwargs = mock_whisper.call_args
        assert kwargs["device"] == "cuda"

    def test_generate_voice_samples_defaults_to_gpu_whisper(self):
        """Stage-4 voice sample generation must default to GPU (not CPU) Whisper."""
        import importlib
        import inspect
        mod = importlib.import_module("audiobook_generator.generate_voice_samples")
        default = inspect.signature(mod.generate_voice_samples).parameters["whisper_cpu"].default
        # Must be False (GPU), not True (CPU), so stage 4 isn't 30x slower.
        assert default is False

    def test_run_full_pipeline_forwards_whisper_cpu_to_stage4(self):
        """run_full_pipeline must pass whisper_cpu through to generate_voice_samples."""
        src = open(
            "audiobook_generator/audiobook_generator.py",
            encoding="utf-8",
        ).read()
        # The gen_voice_samples call site must forward whisper_cpu.
        assert "whisper_cpu=whisper_cpu" in src


# ============================================================================
# Engine worker device propagation (engine -> subprocess -> torch)
# ============================================================================

class TestEngineWorkerDevicePropagation:
    """The engine worker must hand the configured device to the subprocess."""

    def test_worker_spawns_with_device_flag(self):
        """EngineWorker.start() must pass '--device <device>' to the engine main.py."""
        from pathlib import Path
        from tts.worker import EngineWorker

        worker = EngineWorker(engine_dir=Path("/tmp/fake_engine"), device="cuda:0")

        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.stdout.readline.return_value = (
            json.dumps({"type": "ready"}) + "\n"
        )

        with patch.object(worker, "_find_python", return_value="/tmp/fake_engine/.venv/bin/python"), \
             patch("subprocess.Popen", return_value=fake_proc) as mock_popen:
            worker.start()

        mock_popen.assert_called_once()
        argv = mock_popen.call_args.args[0]
        assert argv == [
            "/tmp/fake_engine/.venv/bin/python",
            "/tmp/fake_engine/main.py",
            "--device",
            "cuda:0",
        ]

    def test_worker_preserves_device_index(self):
        """A 'cuda:2' device must not be collapsed to 'cuda' when spawning."""
        from pathlib import Path
        from tts.worker import EngineWorker

        worker = EngineWorker(engine_dir=Path("/tmp/fake_engine"), device="cuda:2")
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.stdout.readline.return_value = (
            json.dumps({"type": "ready"}) + "\n"
        )

        with patch.object(worker, "_find_python", return_value="/tmp/fake_engine/.venv/bin/python"), \
             patch("subprocess.Popen", return_value=fake_proc) as mock_popen:
            worker.start()

        argv = mock_popen.call_args.args[0]
        assert argv[argv.index("--device") + 1] == "cuda:2"


# ============================================================================
# VoiceMapper -> engine device routing
# ============================================================================

class TestVoiceMapperDeviceRouting:
    """VoiceMapper must forward its device to the TTS engine factory."""

    def test_get_engine_uses_voice_mapper_device(self):
        """get_engine() must be called with the VoiceMapper's configured device."""
        with patch("audiobook_generator.voice_mapper.get_engine") as mock_get_engine:
            mapper = VoiceMapper(output_dir="/tmp/out", device="cuda:2", tts_engine="omni")
            mapper.get_engine()
        mock_get_engine.assert_called_once_with("omni", device="cuda:2")

    def test_get_engine_uses_voice_mapper_device_no_index_collapse(self):
        """The configured 'cuda:2' must reach the factory unchanged."""
        with patch("audiobook_generator.voice_mapper.get_engine") as mock_get_engine:
            mapper = VoiceMapper(output_dir="/tmp/out", device="cuda:2", tts_engine="omni")
            mapper.get_engine()
        args, kwargs = mock_get_engine.call_args
        assert kwargs["device"] == "cuda:2"


# ============================================================================
# AUDIO_DEVICE env override (primary mechanism for selecting the engine GPU)
# ============================================================================

class TestAudioDeviceEnvOverride:
    """AUDIO_DEVICE must override the configured default device (config.py:101)."""

    def test_audio_device_overrides_default(self):
        """Config must apply AUDIO_DEVICE on top of the default device."""
        import importlib
        import audiobook_generator.config as config

        with patch.dict("os.environ", {"AUDIO_DEVICE": "cuda:0"}):
            # Re-evaluate the module body to pick up the env override.
            config = importlib.reload(config)

        assert config.AUDIO_SETTINGS["default_device"] == "cuda:0"
        # Restore so later tests aren't affected by the pinned value.
        importlib.reload(config)
