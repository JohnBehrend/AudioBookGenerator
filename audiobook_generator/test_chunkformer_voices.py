#!/usr/bin/env python3
"""
Test ChunkFormer voice classification with known voice samples.

Tests seed voices (known good) vs Dramabox voices (suspected broken)
to identify exactly where classification fails.
"""
import os
import json
import hashlib
from chunkformer import ChunkFormerModel

SEED_DIR = "/home/johnbehrend/pydev/voices_archive"
DRAMABOX_DIR = "/home/johnbehrend/pydev/AudioBookGenerator/voice_test/eye_of_the_world"

model = ChunkFormerModel.from_pretrained("khanhld/chunkformer-gender-emotion-dialect-age-classification")


def classify_voice(audio_path: str) -> dict:
    """Classify voice using ChunkFormer model."""
    result = model.classify_audio(audio_path=audio_path)
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
    print("=" * 70)
    print("ChunkFormer Voice Classification Test")
    print("=" * 70)

    print("\n--- SEED VOICES (known good) ---")
    seed_tests = [
        ("baalzamon", "male", "old"),
        ("egwene", "female", "young"),
        ("bornhald", "male", "middle age"),
        ("narrator", "male", "middle age"),
        ("rand", "male", "young"),
        ("moiraine", "female", "young"),
        ("elaida", "female", "old"),
        ("elayne", "female", "young"),
        ("nynaeve", "female", "young"),
        ("generic man", "male", "middle age"),
    ]
    for name, exp_gender, exp_age in seed_tests:
        path = os.path.join(SEED_DIR, f"{name}.wav")
        test_voice(name, path, exp_gender, exp_age)

    print("\n--- SEED VOICES (Eye of the World output, cloned through omni) ---")
    seed_eotw = [
        ("rand", "male", "young"),
        ("mat", "male", "young"),
        ("perrin", "male", "young"),
        ("egwene", "female", "young"),
        ("nynaeve", "female", "young"),
        ("moiraine", "female", "young"),
        ("narrator", "male", "young"),
        ("baalzamon", "male", "young"),
        ("bornhald", "male", "young"),
    ]
    for name, exp_gender, exp_age in seed_eotw:
        path = os.path.join(DRAMABOX_DIR, f"{name}.wav")
        test_voice(name, path, exp_gender, exp_age)

    print("\n--- DRAMABOX VOICES (generated from description) ---")
    dramabox_tests = [
        ("aginor", "male", "young"),
        ("ara", "female", "young"),
        ("byar", "male", "young"),
        ("cenn", "male", "young"),
        ("darl", "male", "middle age"),
        ("elyas", "male", "young"),
        ("gawyn", "male", "middle age"),
        ("ila", "female", "young"),
        ("jon", "male", "middle age"),
        ("mordeth", "male", "young"),
        ("suian", "male", "young"),
    ]
    for name, exp_gender, exp_age in dramabox_tests:
        path = os.path.join(DRAMABOX_DIR, f"{name}.wav")
        test_voice(name, path, exp_gender, exp_age)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
