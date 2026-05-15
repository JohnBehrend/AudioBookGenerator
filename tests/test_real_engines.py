"""Real TTS engine integration tests.

Two tiers:
  - TestEngineStartup (fast, <60s): verifies each engine can initialize and its
    worker responds. Runs by default when --run-slow is passed.
  - TestRealGeneration (slow): generates actual audio. Requires --run-generate.

Run fast tier:
    pytest tests/test_real_engines.py --run-slow

Run both tiers:
    pytest tests/test_real_engines.py --run-slow --run-generate

Requirements:
    - CUDA GPU available
    - Model weights downloaded (or accessible via HuggingFace)
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from audiobook_generator.engines import get_engine, list_engines


def load_audio(path: str) -> tuple[int, np.ndarray]:
    """Load audio file and return (sample_rate, waveform_array)."""
    data, sr = sf.read(path)
    return sr, data


# Test fixtures
TEST_DESCRIPTION = "A calm, clear female narrator with a pleasant tone."

TEST_TEXT = "Hello, world."

# Engines that require special setup or large models
OPTIONAL_ENGINES = {"echo-tts", "vibevoice"}


@pytest.fixture(scope="session")
def output_dir():
    """Temporary output directory for test audio files."""
    with tempfile.TemporaryDirectory(prefix="real_tts_test_") as d:
        yield Path(d)


@pytest.fixture(scope="session")
def device():
    """Get available device (cuda if available, else skip)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda:0"


@pytest.fixture(scope="session")
def available_engines(device: str):
    """Load each engine once per session, skip on failure."""
    engines = {}
    for engine_name in list_engines():
        if engine_name in OPTIONAL_ENGINES:
            continue
        try:
            engines[engine_name] = get_engine(engine_name, device=device)
        except Exception:
            pass
    return engines


@pytest.fixture(scope="session")
def voice_refs(available_engines: dict, output_dir: Path, device: str):
    """Generate one voice reference per engine, reuse for all tests."""
    refs = {}
    for engine_name, engine in available_engines.items():
        try:
            success, ref_path, _ = engine.generate_voice_sample(
                character_name="narrator",
                description=TEST_DESCRIPTION,
                output_dir=output_dir / engine_name,
                device=device,
                verbose=False,
            )
            if success:
                refs[engine_name] = ref_path
        except Exception:
            pass
    return refs


# ============================================================================
# FAST TIER: startup + worker readiness (< 60s)
# ============================================================================


class TestEngineStartup:
    """Verify each engine can initialize and its worker responds."""

    @pytest.mark.parametrize("engine_name", list_engines())
    def test_engine_initializes(self, engine_name: str, available_engines: dict):
        """Engine must appear in available_engines (worker sent 'ready')."""
        if engine_name in OPTIONAL_ENGINES:
            pytest.skip(f"Optional engine {engine_name} requires special setup")
        assert engine_name in available_engines, (
            f"{engine_name} failed to initialize or worker never sent 'ready'"
        )


# ============================================================================
# SLOW TIER: actual audio generation (needs --run-generate)
# ============================================================================


class TestRealGeneration:
    """Generate actual audio and verify output."""

    @pytest.mark.parametrize("engine_name", list_engines())
    def test_generate_voice_sample(self, engine_name: str, available_engines: dict, device: str, output_dir: Path):
        """Generate a voice sample and verify valid audio output."""
        if engine_name in OPTIONAL_ENGINES:
            pytest.skip(f"Optional engine {engine_name} requires special setup")
        if engine_name not in available_engines:
            pytest.skip(f"Failed to initialize {engine_name}")

        engine = available_engines[engine_name]
        success, output_file, duration = engine.generate_voice_sample(
            character_name="narrator",
            description=TEST_DESCRIPTION,
            output_dir=output_dir / engine_name,
            device=device,
            verbose=False,
        )

        assert success, f"{engine_name} failed to generate voice sample"
        assert output_file is not None, f"{engine_name} returned no output file"

        output_path = Path(output_file)
        assert output_path.exists(), f"{engine_name} output file not found: {output_file}"
        assert output_path.stat().st_size > 0, f"{engine_name} output file is empty"
        assert duration > 0, f"{engine_name} reported zero duration"

        sr, waveform = load_audio(output_file)
        assert waveform.size > 0, f"{engine_name} audio has no samples"
        assert sr > 0, f"{engine_name} invalid sample rate"

    @pytest.mark.parametrize("engine_name", list_engines())
    def test_generate_line_with_voice_ref(self, engine_name: str, available_engines: dict, voice_refs: dict, device: str, output_dir: Path):
        """Generate a line of audio using a voice reference."""
        if engine_name in OPTIONAL_ENGINES:
            pytest.skip(f"Optional engine {engine_name} requires special setup")
        if engine_name not in available_engines:
            pytest.skip(f"Failed to initialize {engine_name}")
        if engine_name not in voice_refs:
            pytest.skip(f"No voice reference generated for {engine_name}")

        engine = available_engines[engine_name]
        ref_path = voice_refs[engine_name]

        output_path = str(output_dir / engine_name / "line_test.wav")
        success = engine.generate_line(
            text=TEST_TEXT,
            voice_path=ref_path,
            output_path=output_path,
            device=device,
            validation_model=None,
            verbose=False,
        )

        assert success, f"{engine_name} failed to generate line audio"
        assert Path(output_path).exists(), f"{engine_name} line output not found"
        assert Path(output_path).stat().st_size > 0, f"{engine_name} line output is empty"

        sr, waveform = load_audio(output_path)
        assert waveform.size > 0, f"{engine_name} line audio has no samples"

    @pytest.mark.parametrize("engine_name", list_engines())
    def test_multiple_generations_same_engine(self, engine_name: str, available_engines: dict, device: str, output_dir: Path):
        """Verify engine can generate multiple samples without re-initialization."""
        if engine_name in OPTIONAL_ENGINES:
            pytest.skip(f"Optional engine {engine_name} requires special setup")
        if engine_name not in available_engines:
            pytest.skip(f"Failed to initialize {engine_name}")

        engine = available_engines[engine_name]
        success, output_file, _ = engine.generate_voice_sample(
            character_name="hero",
            description="A brave, deep male voice with authority and warmth.",
            output_dir=output_dir / engine_name,
            device=device,
            verbose=False,
        )
        assert success, f"{engine_name} batch gen failed for hero"
        assert Path(output_file).exists(), f"{engine_name} batch output missing for hero"

    @pytest.mark.parametrize("engine_name", list_engines())
    def test_audio_not_silent(self, engine_name: str, available_engines: dict, voice_refs: dict):
        """Generated audio should not be completely silent."""
        if engine_name in OPTIONAL_ENGINES:
            pytest.skip(f"Optional engine {engine_name} requires special setup")
        if engine_name not in available_engines:
            pytest.skip(f"Failed to initialize {engine_name}")
        if engine_name not in voice_refs:
            pytest.skip(f"No voice reference generated for {engine_name}")

        sr, waveform = load_audio(voice_refs[engine_name])
        rms = np.abs(waveform).mean()
        assert rms > 0.001, (
            f"{engine_name} generated audio is nearly silent (RMS={rms:.6f}). "
            "Model may not have loaded correctly."
        )


@pytest.fixture(scope="session", autouse=True)
def shutdown_engines(available_engines: dict):
    """Shutdown all engine workers at the end of the test session."""
    yield
    for engine in available_engines.values():
        try:
            engine.shutdown_worker()
        except Exception:
            pass
