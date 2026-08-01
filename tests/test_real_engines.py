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
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

from tts import get_engine, list_engines


def dbg(msg: str) -> None:
    """Print debug output that survives pytest capture."""
    print(f"  [DEBUG] {msg}", flush=True)


def load_audio(path: str) -> tuple[int, np.ndarray]:
    """Load audio file and return (sample_rate, waveform_array)."""
    data, sr = sf.read(path)
    return sr, data


def generate_test_voice(output_path: Path, sample_rate: int = 24000, duration_s: float = 2.0) -> Path:
    """Generate a simple sine wave audio file for clone-only engine testing."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    # Mix of two tones to create something that looks like voice data
    waveform = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 300 * t)
    sf.write(str(output_path), waveform, sample_rate)
    return output_path


# Test fixtures (Dramabox-style verbose descriptions for best voice diversity)
TEST_DESCRIPTIONS = {
    "narrator": "A middle-aged woman with a warm, resonant contralto voice that carries the quiet authority of a seasoned storyteller. Her speech is measured and articulate, with a gentle British upper-class accent that lends refinement to every word. There's a calm, inviting quality to her tone, like a trusted companion reading by firelight.",
    "hero": "A young man in his late twenties with a deep, gravelly baritone voice full of quiet strength and determination. He speaks at a deliberate pace with a slight rasp that hints at hardship, his words carrying an underlying intensity. His tone is earnest and resolute, like someone who has faced danger and emerged unbroken.",
}

TEST_TEXT = "Hello, world."

# Engines that require special setup or large models
OPTIONAL_ENGINES = set()

# Engines that are clone-only (no voice sample generation from description)
CLONE_ONLY_ENGINES = {"echo-tts", "miso-tts"}

# Persistent output directory for generated test voices
_TEST_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "voice_test" / "test_voices"


@pytest.fixture(scope="session")
def output_dir():
    """Persistent output directory for test audio files."""
    _TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield _TEST_OUTPUT_DIR


@pytest.fixture(scope="session")
def device():
    """Get available device (cuda if available, else skip)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda:0"


@pytest.fixture(scope="session")
def available_engines(device: str, pytestconfig):
    """Load each engine once per session, skip on failure.

    When a -k filter is used, only loads engines matching that filter
    to avoid wasting time loading unrelated engines.
    """
    keyword = str(pytestconfig.getoption("keyword", default=""))
    engines = {}
    for engine_name in list_engines():
        if engine_name in OPTIONAL_ENGINES:
            continue
        if keyword and keyword not in engine_name:
            continue
        dbg(f"Loading engine: {engine_name}")
        t0 = time.monotonic()
        try:
            engines[engine_name] = get_engine(engine_name, device=device)
            dbg(f"Engine {engine_name} ready in {time.monotonic()-t0:.1f}s")
        except Exception as e:
            dbg(f"Engine {engine_name} failed: {e}")
    return engines


@pytest.fixture(scope="session")
def all_engines(device: str):
    """Load every engine once per session, including optional ones."""
    engines = {}
    for engine_name in list_engines():
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
            if engine_name in CLONE_ONLY_ENGINES:
                # Generate a synthetic voice reference for clone-only engines
                ref_path = output_dir / engine_name / "test_voice.wav"
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                generate_test_voice(ref_path)
                refs[engine_name] = str(ref_path)
            else:
                success, ref_path, _ = engine.generate_voice_sample(
                    character_name="narrator",
                    description=TEST_DESCRIPTIONS["narrator"],
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

    @pytest.mark.slow
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

    @pytest.mark.slow
    @pytest.mark.generate
    @pytest.mark.parametrize("engine_name", list_engines())
    def test_generate_voice_sample(self, engine_name: str, available_engines: dict, device: str, output_dir: Path):
        """Generate a voice sample and verify valid audio output."""
        if engine_name in OPTIONAL_ENGINES:
            pytest.skip(f"Optional engine {engine_name} requires special setup")
        if engine_name in CLONE_ONLY_ENGINES:
            pytest.skip(f"Clone-only engine {engine_name} does not support voice sample generation")
        if engine_name not in available_engines:
            pytest.skip(f"Failed to initialize {engine_name}")

        engine = available_engines[engine_name]
        success, output_file, duration = engine.generate_voice_sample(
            character_name="narrator",
            description=TEST_DESCRIPTIONS["narrator"],
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

    @pytest.mark.slow
    @pytest.mark.generate
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

        dbg(f"[{engine_name}] voice_ref = {ref_path}")
        dbg(f"[{engine_name}] GPU mem before = {torch.cuda.memory_allocated(device) // 1024**2} MiB")

        output_path = str(output_dir / engine_name / "line_test.wav")
        t0 = time.monotonic()
        dbg(f"[{engine_name}] calling generate_line...")
        success = engine.generate_line(
            text=TEST_TEXT,
            voice_path=ref_path,
            output_path=output_path,
            device=device,
            validation_model=None,
            verbose=False,
        )
        dbg(f"[{engine_name}] generate_line returned in {time.monotonic()-t0:.1f}s, success={success}")
        dbg(f"[{engine_name}] GPU mem after = {torch.cuda.memory_allocated(device) // 1024**2} MiB")

        assert success, f"{engine_name} failed to generate line audio"
        assert Path(output_path).exists(), f"{engine_name} line output not found"
        assert Path(output_path).stat().st_size > 0, f"{engine_name} line output is empty"

        sr, waveform = load_audio(output_path)
        assert waveform.size > 0, f"{engine_name} line audio has no samples"

    @pytest.mark.slow
    @pytest.mark.generate
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

    @pytest.mark.slow
    @pytest.mark.generate
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


class TestRealWhisperValidation:
    """Touch real generators for the validation and improvement loops.

    Unlike the mocked Whisper tests elsewhere, these load the actual Whisper
    model and transcribe genuinely generated TTS audio, then exercise the full
    generate -> validate -> retry improvement loop end-to-end.

    Requires --run-slow --run-generate and a CUDA GPU.
    """

    @pytest.fixture(scope="session")
    def whisper_model(self, device: str):
        """Load real Whisper on GPU, falling back to CPU when VRAM is exhausted.

        Real TTS engines often consume most of the GPU, so the validation model
        may not fit alongside them. CPU fallback keeps the real-inference tests
        runnable without requiring dedicated VRAM for Whisper.
        """
        from audiobook_generator.audiobook_generator import setup_validation_model

        try:
            return setup_validation_model(device, fast=True)
        except Exception as gpu_err:
            dbg(f"Whisper GPU load failed ({gpu_err}), falling back to CPU")
            try:
                return setup_validation_model(device, cpu=True, fast=True)
            except Exception as cpu_err:
                pytest.skip(f"Could not load real Whisper model on GPU or CPU: {cpu_err}")

    @pytest.mark.slow
    @pytest.mark.generate
    @pytest.mark.parametrize("engine_name", list_engines())
    def test_real_whisper_transcribes_generated_line(
        self, engine_name, available_engines, voice_refs, device, output_dir, whisper_model
    ):
        """Real Whisper must produce word-level transcription from real TTS audio.

        This proves the real validation path (real TTS generation -> real Whisper
        transcription with word timestamps) works end-to-end. Correctness of the
        recovered words is enforced by the improvement-loop test below, since some
        engines do not faithfully reproduce the requested text.
        """
        if engine_name in OPTIONAL_ENGINES:
            pytest.skip(f"Optional engine {engine_name} requires special setup")
        if engine_name not in available_engines:
            pytest.skip(f"Failed to initialize {engine_name}")
        if engine_name not in voice_refs:
            pytest.skip(f"No voice reference generated for {engine_name}")

        from audiobook_generator.utils import distill_string, transcribe_audio_with_whisper

        engine = available_engines[engine_name]
        output_path = str(output_dir / engine_name / "whisper_validation.wav")
        success = engine.generate_line(
            text=TEST_TEXT,
            voice_path=voice_refs[engine_name],
            output_path=output_path,
            device=device,
            validation_model=None,
            verbose=False,
        )
        assert success, f"{engine_name} failed to generate line audio for whisper validation"

        detected, start_times, end_times = transcribe_audio_with_whisper(whisper_model, output_path)
        assert distill_string(detected), f"{engine_name}: real Whisper returned empty transcription"
        assert len(start_times) == len(end_times) > 0, (
            f"{engine_name}: real Whisper produced no word-level timestamps"
        )

    @pytest.mark.slow
    @pytest.mark.generate
    @pytest.mark.parametrize("engine_name", list_engines())
    def test_improvement_loop_with_real_generators(
        self, engine_name, available_engines, voice_refs, device, output_dir, whisper_model
    ):
        """Full generate->validate->retry loop with real TTS and real Whisper."""
        if engine_name in OPTIONAL_ENGINES:
            pytest.skip(f"Optional engine {engine_name} requires special setup")
        if engine_name not in available_engines:
            pytest.skip(f"Failed to initialize {engine_name}")
        if engine_name not in voice_refs:
            pytest.skip(f"No voice reference generated for {engine_name}")

        from audiobook_generator.audiobook_generator import TTSConfig, generate_tts_for_line
        from audiobook_generator.pipeline import generate_output_filename
        from audiobook_generator.voice_mapper import VoiceMapper

        loop_dir = output_dir / engine_name / "loop"
        loop_dir.mkdir(parents=True, exist_ok=True)
        mapper = VoiceMapper(output_dir=str(loop_dir), device=device, engine=available_engines[engine_name])
        tts_config = TTSConfig(
            device=device,
            tts_engine=engine_name,
            output_dir=str(loop_dir),
            short_text_postfix="and also with you",
            validation_model=whisper_model,
            engine=available_engines[engine_name],
            verbose=False,
        )
        chapter_idx, line_idx = 0, 1
        success, ratio = generate_tts_for_line(
            chapter_idx=chapter_idx,
            line_idx=line_idx,
            text=TEST_TEXT,
            voice_name="narrator",
            voice_mapper=mapper,
            tts_config=tts_config,
            voice_path=voice_refs[engine_name],
        )
        assert success, (
            f"{engine_name}: real improvement loop failed to produce acceptable audio (ratio={ratio})"
        )

        final_path = generate_output_filename(tts_config.output_dir, chapter_idx, line_idx, is_final=True)
        assert os.path.exists(final_path), f"{engine_name}: final audio not produced at {final_path}"
        sr, waveform = load_audio(final_path)
        assert waveform.size > 0, f"{engine_name}: improvement loop produced empty audio"
        assert float(np.abs(waveform).mean()) > 0.001, f"{engine_name}: improvement loop produced silent audio"


@pytest.fixture(scope="session", autouse=True)
def shutdown_engines(available_engines: dict):
    """Shutdown all engine workers at the end of the test session."""
    yield
    for engine in available_engines.values():
        try:
            engine.shutdown_worker()
        except Exception:
            pass
