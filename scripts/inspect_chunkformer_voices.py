#!/usr/bin/env python3
"""
Inspect ChunkFormer voice classification with known voice samples.

Tests seed voices (known good) vs cloned voices to identify exactly
where classification fails.

Voice lists and sample dirs come from a JSON map (default
voice_test/chunkformer_inspect.json, gitignored — sample names and
locations are personal data):

    {"seed_dir": "...", "cloned_dir": "...",
     "seed_voices": [{"name", "gender", "age"}, ...],
     "cloned_voices": [...]}

Run manually: uv run python scripts/inspect_chunkformer_voices.py
(Diagnostic script, not part of the pytest suite.)
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

DEFAULT_MAP = Path(__file__).parent.parent / "voice_test" / "chunkformer_inspect.json"

_model = None


def _load_map(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"no voice map at {path}; create it with 'seed_dir', 'cloned_dir', "
            "'seed_voices' and 'cloned_voices' lists (see module docstring)")
    return json.loads(path.read_text(encoding="utf-8"))


def _get_model():
    """Load the ChunkFormer model lazily, using the id from config."""
    global _model
    if _model is None:
        from audiobook_generator.config import CHUNKFORMER_VALIDATION
        from chunkformer import ChunkFormerModel

        _model = ChunkFormerModel.from_pretrained(CHUNKFORMER_VALIDATION["model_id"])
    return _model


def classify_voice(audio_path: str) -> dict:
    """Classify voice using ChunkFormer model."""
    result = _get_model().classify_audio(audio_path=audio_path)
    return {
        "gender": result["gender"]["label"],
        "age_group": result["age"]["label"],
        "emotion": result["emotion"]["label"],
        "dialect": result["dialect"]["label"],
    }


def voice_stats(path: str) -> dict:
    """Compute basic audio stats for debugging."""
    import soundfile as sf
    import numpy as np

    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()[:8]
    zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / len(audio)
    # Estimate pitch via autocorrelation
    window = audio[:sr]  # 1 second
    corr = np.correlate(window - np.mean(window), window - np.mean(window), mode="full")
    corr = corr[corr.size // 2:]
    if len(corr) > 1:
        peak = np.argmax(corr[1:]) + 1
        pitch = sr / peak if peak > 0 else 0
    else:
        pitch = 0
    return {
        "duration": f"{dur:.1f}s",
        "md5": md5,
        "zcr": f"{zcr:.4f}",
        "pitch_hz": f"{pitch:.0f}Hz",
    }


def test_voice(label: str, path: str, expected_gender: str = None, expected_age: str = None):
    """Test a single voice file."""
    if not os.path.exists(path):
        print(f"  {label}: MISSING")
        return

    result = classify_voice(path)
    gender = result.get("gender", "?")
    age = result.get("age_group", "?")
    emotion = result.get("emotion", "?")
    dialect = result.get("dialect", "?")

    # Check against expectations
    ok = "✓"
    if expected_gender and gender != expected_gender:
        ok = "✗ GENDER MISMATCH"
    if expected_age and age != expected_age:
        ok = ok + ("✗ AGE MISMATCH" if ok == "✓" else "; AGE MISMATCH")

    print(f"  {label}: {gender} / {age} / {emotion} / {dialect}  {ok}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", type=Path, default=DEFAULT_MAP,
                    help="JSON voice map (see module docstring)")
    args = ap.parse_args()
    voice_map = _load_map(args.map)

    print("=" * 70)
    print("ChunkFormer Voice Classification Test")
    print("=" * 70)

    seed_dir = voice_map["seed_dir"]
    cloned_dir = voice_map["cloned_dir"]

    print("\n--- SEED VOICES (known good) ---")
    for v in voice_map.get("seed_voices", []):
        path = os.path.join(seed_dir, f"{v['name']}.wav")
        test_voice(v["name"], path, v.get("gender"), v.get("age"))

    print("\n--- CLONED VOICES ---")
    for v in voice_map.get("cloned_voices", []):
        path = os.path.join(cloned_dir, f"{v['name']}.wav")
        test_voice(v["name"], path, v.get("gender"), v.get("age"))

    print("\n--- CLONED VOICES (from description) ---")
    for v in voice_map.get("desc_voices", []):
        path = os.path.join(cloned_dir, f"{v['name']}.wav")
        test_voice(v["name"], path, v.get("gender"), v.get("age"))

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
