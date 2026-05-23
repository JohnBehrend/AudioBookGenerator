#!/usr/bin/env python3
"""
Test Nemotron voice classification with known voice samples.

Tests seed voices (known good) vs Dramabox voices (suspected broken)
to identify exactly where classification fails.
"""
import soundfile as sf
import tempfile
import os
import librosa
import json
import hashlib
import numpy as np
from pathlib import Path
from openai import OpenAI

NEMOTRON_URL = "http://localhost:8082/v1"
SEED_DIR = "/home/johnbehrend/pydev/voices_archive"
DRAMABOX_DIR = "/home/johnbehrend/pydev/AudioBookGenerator/voice_test/eye_of_the_world"

client = OpenAI(base_url=NEMOTRON_URL, api_key="EMPTY")


def classify_voice(audio_path: str) -> dict:
    """Resample to 16kHz and send to Nemotron for classification."""
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, audio, 16000, subtype="PCM_16")
    file_url = Path(tmp).resolve().as_uri()

    resp = client.chat.completions.create(
        model="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
        messages=[{
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": file_url}},
                {"type": "text", "text": (
                    "Listen to this voice sample. Analyze the voice carefully and output ONLY valid JSON:\n"
                    '{"gender": "male"|"female", "age_group": "young"|"middle-aged"|"old", '
                    '"pitch": "very low"|"low"|"moderate"|"high"|"very high", '
                    '"tone_keywords": "up to 3 words"}\n'
                    "Return ONLY the JSON object."
                )},
            ],
        }],
        max_tokens=256,
        temperature=0.2,
        extra_body={"top_k": 1, "chat_template_kwargs": {"enable_thinking": False}},
    )
    os.remove(tmp)
    raw = resp.choices[0].message.content.strip()
    # Extract JSON
    if "{" in raw:
        raw = raw[raw.index("{"):raw.rindex("}") + 1]
        return json.loads(raw)
    return {"error": "No JSON in response", "raw": raw[:200]}


def voice_stats(path: str) -> dict:
    """Compute basic audio stats for debugging."""
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

    stats = voice_stats(path)
    print(f"  {label}: {stats['duration']} pitch={stats['pitch_hz']} zcr={stats['zcr']} md5={stats['md5']}")

    result = classify_voice(path)
    gender = result.get("gender", "?")
    age = result.get("age_group", "?")
    pitch = result.get("pitch", "?")
    tone = result.get("tone_keywords", "?")

    # Check against expectations
    ok = "✓"
    if expected_gender and gender != expected_gender:
        ok = "✗ GENDER MISMATCH"
    if expected_age and age != expected_age:
        ok = ok + ("✗ AGE MISMATCH" if ok == "✓" else "; AGE MISMATCH")

    print(f"    → {gender} / {age} / {pitch} / {tone}  {ok}")


def main():
    print("=" * 70)
    print("Nemotron Voice Classification Test")
    print("=" * 70)

    print("\n--- SEED VOICES (known good) ---")
    seed_tests = [
        ("baalzamon", "male", "old"),
        ("egwene", "female", "young"),
        ("bornhald", "male", "middle-aged"),
        ("narrator", "male", "middle-aged"),
        ("rand", "male", "young"),
        ("moiraine", "female", "middle-aged"),
        ("elaida", "female", "young"),
        ("elayne", "female", "young"),
        ("nynaeve", "female", "young"),
        ("generic man", "male", "middle-aged"),
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
        ("moiraine", "female", "middle-aged"),
        ("narrator", "male", "middle-aged"),
        ("baalzamon", "male", "old"),
        ("bornhald", "male", "middle-aged"),
    ]
    for name, exp_gender, exp_age in seed_eotw:
        path = os.path.join(DRAMABOX_DIR, f"{name}.wav")
        test_voice(name, path, exp_gender, exp_age)

    print("\n--- DRAMABOX VOICES (generated from description) ---")
    dramabox_tests = [
        ("aginor", "male", "old"),
        ("ara", "female", "young"),
        ("byar", "male", "young"),
        ("cenn", "male", "old"),
        ("darl", "male", "young"),
        ("elyas", "male", "old"),
        ("gawyn", "male", "young"),
        ("ila", "female", "old"),
        ("jon", "male", "young"),
        ("mordeth", "male", "old"),
        ("suian", "female", "young"),
    ]
    for name, exp_gender, exp_age in dramabox_tests:
        path = os.path.join(DRAMABOX_DIR, f"{name}.wav")
        test_voice(name, path, exp_gender, exp_age)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
