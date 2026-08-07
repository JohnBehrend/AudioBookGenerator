"""Heavy MiniMax H3 integration test: generate -> free VRAM -> Whisper validate.

This proves the full real pipeline that H3 is used for in production:

  1. Generate a voice sample with the real minimax_h3 engine (via ComfyUI).
  2. Confirm the engine freed ComfyUI's VRAM afterward (H3_FREE_VRAM=1), so
     the GPU is available for a downstream engine (e.g. omni) or Whisper.
  3. Load the real Whisper model on the same GPU and transcribe the generated
     audio, verifying H3 actually spoke the reference text.

This is intentionally *heavy*: it loads the 30B-class H3 model through a live
ComfyUI instance and generates real audio. It only runs with both
``--run-slow`` and ``--run-generate``.

Prerequisites:
    - CUDA GPU available
    - A ComfyUI server running on the same GPU, with the MiniMax H3 models
      installed. Point the test at it via COMFYUI_URL (default
      http://127.0.0.1:8188). Recommended ComfyUI launch:
        CUDA_VISIBLE_DEVICES=<4090 cuda idx> python main.py --listen 127.0.0.1 \
            --port 8188 --disable-dynamic-vram

Run:
    pytest tests/test_minimax_h3_heavy.py --run-slow --run-generate -v
"""

import difflib
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from tts import get_engine
from audiobook_generator.config import DEFAULTS

# Env inherited by the H3 engine subprocess. H3_FREE_VRAM=1 makes the engine
# call ComfyUI's /free after every generation (the behavior under test).
# H3_LENGTH=auto sizes each clip to fit the FULL voice-sample reference text
# (previously a short fixed length truncated the tail, ending mid-sentence).
H3_ENV = {
    "H3_FREE_VRAM": "1",
    "COMFYUI_URL": os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"),
    "H3_WIDTH": "64",
    "H3_HEIGHT": "64",
    "H3_LENGTH": "auto",
    "H3_STEPS": "20",
    "H3_CFG": "1.0",
}

DEVICE = "cuda:0"

# Rand from The Eye of the World / The Dragon Reborn (universal-JSON format).
RAND_DESC = (
    '{"gender": "male", "age": "young adult", "pitch": "low", '
    '"accent": "british", "style": ["authoritative", "weary", "determined"], '
    '"description": "Rand speaks with a low, resonant voice that carries the weight of destiny."}'
)
HERO_DESC = "A brave, deep male voice with authority and warmth."
NARRATOR_DESC = "A calm, clear female narrator with a warm, measured tone."


def dbg(msg: str) -> None:
    print(f"  [DEBUG] {msg}", flush=True)


def _comfy_reachable(url: str) -> bool:
    import requests

    try:
        requests.get(f"{url}/system_stats", timeout=5)
        return True
    except Exception:
        return False


def _free_vram_gb(device: str) -> float:
    """Free VRAM (GB) on the CUDA device ComfyUI shares."""
    free, total = torch.cuda.mem_get_info(device)
    return free / 1024**3


def _transcribe_whisper(wav_path: str) -> str:
    """Load real Whisper on GPU and transcribe; assert it fit in VRAM."""
    from audiobook_generator.audiobook_generator import setup_validation_model
    from audiobook_generator.utils import transcribe_audio_with_whisper

    vm = setup_validation_model(DEVICE, cpu=False, fast=True)
    detected, _, _ = transcribe_audio_with_whisper(vm, wav_path)
    return detected


def _ref_match(detected: str) -> dict:
    from audiobook_generator.utils import distill_string

    ref_words = distill_string(DEFAULTS["static_voice_text"]).split()
    det_words = distill_string(detected).split()
    sm = difflib.SequenceMatcher(None, ref_words, det_words)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return {
        "det_words": len(det_words),
        "ref_words": len(ref_words),
        "matched": matched,
        "match_rate": round(matched / max(len(ref_words), 1), 3),
    }


def _tail_coverage(detected: str) -> float:
    """Fraction of the reference text reached by the detected speech.

    Guards against truncated clips: a clip that stops mid-sentence (e.g. at
    "but we've") scores low (~0.2), whereas a full-length clip scores high.
    """
    from audiobook_generator.utils import distill_string

    ref_words = distill_string(DEFAULTS["static_voice_text"]).split()
    det_words = distill_string(detected).split()
    best = 0
    for w in det_words:
        if w in ref_words:
            best = max(best, ref_words.index(w) + 1)
    return best / len(ref_words)


@pytest.fixture(scope="module")
def h3_engine():
    """Set H3 env, require a reachable ComfyUI, and provide the real engine."""
    for k, v in H3_ENV.items():
        os.environ[k] = v

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not _comfy_reachable(H3_ENV["COMFYUI_URL"]):
        pytest.skip(
            f"ComfyUI not reachable at {H3_ENV['COMFYUI_URL']}. "
            "Start it on the same GPU before running this heavy test."
        )

    dbg(f"free VRAM before generation: {_free_vram_gb(DEVICE):.1f} GB")
    engine = get_engine("minimax_h3", device=DEVICE)
    yield engine
    engine.shutdown_worker()


@pytest.fixture(scope="module")
def whisper_model():
    """Load Whisper on GPU; skip if it cannot fit even with VRAM freed."""
    from audiobook_generator.audiobook_generator import setup_validation_model

    try:
        return setup_validation_model(DEVICE, cpu=False, fast=True)
    except Exception as e:
        pytest.skip(f"Could not load Whisper on GPU: {e}")


@pytest.mark.slow
@pytest.mark.generate
def test_generate_frees_vram_and_whisper_validates(h3_engine, whisper_model, tmp_path):
    """Generate Rand's voice, confirm VRAM freed, then Whisper-validate it."""
    from audiobook_generator.utils import transcribe_audio_with_whisper

    t0 = __import__("time").monotonic()
    success, output_file, duration = h3_engine.generate_voice_sample(
        character_name="rand",
        description=RAND_DESC,
        output_dir=tmp_path,
        device=DEVICE,
        verbose=False,
        static_voice_text=DEFAULTS["static_voice_text"],
    )
    dbg(f"generate took {__import__('time').monotonic()-t0:.1f}s")

    assert success, "minimax_h3 failed to generate Rand's voice sample"
    assert output_file and Path(output_file).exists(), "output file missing"
    assert duration > 0, "reported zero duration"

    data, sr = sf.read(output_file)
    assert data.size > 0 and sr > 0, "invalid audio output"
    assert float(np.abs(data).mean()) > 0.001, "generated audio is silent"

    # H3_FREE_VRAM=1 must have unloaded the H3 model from the shared GPU, so
    # Whisper (loaded next) fits. Assert most of VRAM is free right after gen.
    free_gb = _free_vram_gb(DEVICE)
    dbg(f"free VRAM after generation: {free_gb:.1f} GB")
    assert free_gb > 10.0, (
        f"H3 model still resident in VRAM after generation ({free_gb:.1f} GB free); "
        "H3_FREE_VRAM did not free ComfyUI memory. Whisper would OOM."
    )

    detected, starts, ends = transcribe_audio_with_whisper(whisper_model, output_file)
    dbg(f"Whisper transcript: {detected}")
    assert detected.strip(), "Whisper returned empty transcription"
    assert len(starts) == len(ends) > 0, "no word-level timestamps"

    m = _ref_match(detected)
    dbg(f"match: {m}")
    assert m["matched"] >= 5, (
        f"H3 speech barely overlaps reference text (matched={m['matched']}). "
        f"Transcript: {detected!r}"
    )

    cov = _tail_coverage(detected)
    dbg(f"tail coverage: {cov:.2f} (last detected word at {cov*100:.0f}% of ref)")
    assert cov >= 0.5, (
        f"Clip appears truncated mid-sentence: only reached {cov*100:.0f}% of the "
        f"reference. Expected a ~15s clip covering most of it. "
        f"Transcript: {detected!r}"
    )


@pytest.mark.slow
@pytest.mark.generate
def test_multiple_sequential_voices_then_whisper(h3_engine, whisper_model, tmp_path):
    """Generate several H3 voices back-to-back, then Whisper-validate one."""
    from audiobook_generator.utils import transcribe_audio_with_whisper

    outputs = {}
    for name, desc in [("narrator", NARRATOR_DESC), ("hero", HERO_DESC)]:
        success, output_file, _ = h3_engine.generate_voice_sample(
            character_name=name,
            description=desc,
            output_dir=tmp_path,
            device=DEVICE,
            verbose=False,
            static_voice_text=DEFAULTS["static_voice_text"],
        )
        assert success, f"sequential generation failed for {name}"
        assert Path(output_file).exists()
        outputs[name] = output_file

    free_gb = _free_vram_gb(DEVICE)
    dbg(f"free VRAM after {len(outputs)} sequential gens: {free_gb:.1f} GB")
    assert free_gb > 10.0, "VRAM not freed after sequential generations"

    detected, _, _ = transcribe_audio_with_whisper(whisper_model, outputs["narrator"])
    dbg(f"Whisper transcript (narrator): {detected}")
    assert detected.strip(), "sequential run: Whisper returned empty transcription"
    assert _ref_match(detected)["matched"] >= 5, "sequential run: speech not on-reference"
