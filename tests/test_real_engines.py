"""Real TTS engine integration tests.

These tests generate actual .wav files using real TTS models.
They are skipped by default and require --run-slow to execute.

Run with:
    pytest tests/test_real_engines.py --run-slow

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
TEST_DESCRIPTIONS = {
    "narrator": "A calm, clear female narrator with a pleasant tone.",
    "hero": "A brave, deep male voice with authority and warmth.",
}

TEST_TEXT = "Hello, this is a test of the text to speech system."

# Engines that require special setup or large models
OPTIONAL_ENGINES = {"echo-tts", "vibevoice"}


@pytest.fixture
def output_dir():
    """Temporary output directory for test audio files."""
    with tempfile.TemporaryDirectory(prefix="real_tts_test_") as d:
        yield Path(d)


@pytest.fixture
def device():
    """Get available device (cuda if available, else skip)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda:0"


class TestRealVoiceSamples:
    """Test voice sample generation with real TTS engines."""

    @pytest.mark.parametrize("engine_name", list_engines())
    def test_generate_voice_sample(self, engine_name: str, device: str, output_dir: Path):
        """Generate a voice sample for each engine and verify output."""
        # Skip optional engines if not configured
        if engine_name in OPTIONAL_ENGINES:
            pytest.skip(f"Optional engine {engine_name} requires special setup")

        try:
            engine = get_engine(engine_name, device=device)
        except Exception as e:
            pytest.skip(f"Failed to initialize {engine_name}: {e}")

        for char_name, description in TEST_DESCRIPTIONS.items():
            success, output_file, duration = engine.generate_voice_sample(
                character_name=char_name,
                description=description,
                output_dir=output_dir,
                device=device,
                verbose=False,
            )

            assert success, f"{engine_name} failed to generate voice sample for {char_name}"
            assert output_file is not None, f"{engine_name} returned no output file for {char_name}"

            output_path = Path(output_file)
            assert output_path.exists(), f"{engine_name} output file not found: {output_file}"
            assert output_path.stat().st_size > 0, f"{engine_name} output file is empty: {output_file}"
            assert duration > 0, f"{engine_name} reported zero duration for {char_name}"

            # Verify valid audio
            sr, waveform = load_audio(output_file)
            assert waveform.size > 0, f"{engine_name} audio has no samples for {char_name}"
            assert sr > 0, f"{engine_name} invalid sample rate for {char_name}"

        engine.shutdown_worker()


class TestRealLineGeneration:
    """Test line-by-line TTS generation with real engines."""

    @pytest.mark.parametrize("engine_name", list_engines())
    def test_generate_line_with_voice_ref(self, engine_name: str, device: str, output_dir: Path):
        """Generate a line of audio using a voice reference for each engine."""
        if engine_name in OPTIONAL_ENGINES:
            pytest.skip(f"Optional engine {engine_name} requires special setup")

        try:
            engine = get_engine(engine_name, device=device)
        except Exception as e:
            pytest.skip(f"Failed to initialize {engine_name}: {e}")

        # First generate a voice sample to use as reference
        voice_ref = output_dir / "ref_narrator.wav"
        success, ref_path, _ = engine.generate_voice_sample(
            character_name="narrator",
            description=TEST_DESCRIPTIONS["narrator"],
            output_dir=output_dir,
            device=device,
            verbose=False,
        )

        assert success, f"{engine_name} failed to generate reference voice"
        assert Path(ref_path).exists(), f"{engine_name} reference voice not found"

        # Now generate a line using that reference
        output_path = str(output_dir / "line_test.wav")
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

        out_stat = Path(output_path).stat()
        assert out_stat.st_size > 0, f"{engine_name} line output is empty"

        sr, waveform = load_audio(output_path)
        assert waveform.size > 0, f"{engine_name} line audio has no samples"

        engine.shutdown_worker()


class TestRealEngineLifecycle:
    """Test engine initialization, shutdown, and re-use."""

    @pytest.mark.parametrize("engine_name", list_engines())
    def test_engine_setup_and_shutdown(self, engine_name: str, device: str):
        """Verify engine can be created and shut down cleanly."""
        if engine_name in OPTIONAL_ENGINES:
            pytest.skip(f"Optional engine {engine_name} requires special setup")

        try:
            engine = get_engine(engine_name, device=device)
        except Exception as e:
            pytest.skip(f"Failed to initialize {engine_name}: {e}")

        engine.shutdown_worker()

    @pytest.mark.parametrize("engine_name", list_engines())
    def test_multiple_generations_same_engine(self, engine_name: str, device: str, output_dir: Path):
        """Verify engine can generate multiple samples without re-initialization."""
        if engine_name in OPTIONAL_ENGINES:
            pytest.skip(f"Optional engine {engine_name} requires special setup")

        try:
            engine = get_engine(engine_name, device=device)
        except Exception as e:
            pytest.skip(f"Failed to initialize {engine_name}: {e}")

        results = []
        for char_name, description in TEST_DESCRIPTIONS.items():
            success, output_file, duration = engine.generate_voice_sample(
                character_name=char_name,
                description=description,
                output_dir=output_dir,
                device=device,
                verbose=False,
            )
            results.append((char_name, success, output_file))

        engine.shutdown_worker()

        for char_name, success, output_file in results:
            assert success, f"{engine_name} batch gen failed for {char_name}"
            assert Path(output_file).exists(), f"{engine_name} batch output missing for {char_name}"


class TestRealAudioQuality:
    """Basic audio quality checks on generated files."""

    @pytest.mark.parametrize("engine_name", list_engines())
    def test_audio_not_silent(self, engine_name: str, device: str, output_dir: Path):
        """Generated audio should not be completely silent."""
        if engine_name in OPTIONAL_ENGINES:
            pytest.skip(f"Optional engine {engine_name} requires special setup")

        try:
            engine = get_engine(engine_name, device=device)
        except Exception as e:
            pytest.skip(f"Failed to initialize {engine_name}: {e}")

        success, output_file, _ = engine.generate_voice_sample(
            character_name="narrator",
            description=TEST_DESCRIPTIONS["narrator"],
            output_dir=output_dir,
            device=device,
            verbose=False,
        )

        engine.shutdown_worker()

        assert success, f"{engine_name} generation failed"

        sr, waveform = load_audio(output_file)
        rms = np.abs(waveform).mean()
        assert rms > 0.001, (
            f"{engine_name} generated audio is nearly silent (RMS={rms:.6f}). "
            "Model may not have loaded correctly."
        )
