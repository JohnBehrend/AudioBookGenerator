"""Audio processing and voice validation utilities.

Contains audio cropping, ChunkFormer voice validation, and quality-validation
helpers used by the TTS pipeline. These were extracted from utils.py for
clarity.
"""

import json
import os
from typing import Optional, Tuple, List, Dict, Any

from openai import OpenAI

from .config import LLM_NO_THINKING_EXTRA_BODY

# A classification mismatch only fails validation when the model is at least
# this confident; below it the mismatch is treated as noise.
CHUNKFORMER_CONFIDENCE_THRESHOLD = 0.7

_GENDER_FEMALE_KEYWORDS = ("female", "woman", "women", "girl")
_GENDER_MALE_KEYWORDS = ("male", "man", "men", "boy")
_AGE_YOUNG_KEYWORDS = ("young", "youth", "teen", "teenager", "child")
_AGE_OLD_KEYWORDS = ("old", "elder", "elderly", "senior", "ancient")
_AGE_MIDDLE_KEYWORDS = ("middle", "mature", "adult", "forty", "fifty", "thirty")


def _expected_gender_from_description(desc_lower: str) -> Optional[str]:
    if any(w in desc_lower for w in _GENDER_FEMALE_KEYWORDS):
        return "female"
    if any(w in desc_lower for w in _GENDER_MALE_KEYWORDS):
        return "male"
    return None


def _expected_age_from_description(desc_lower: str) -> Optional[str]:
    if any(w in desc_lower for w in _AGE_YOUNG_KEYWORDS):
        return "young"
    if any(w in desc_lower for w in _AGE_OLD_KEYWORDS):
        return "old"
    if any(w in desc_lower for w in _AGE_MIDDLE_KEYWORDS):
        return "middle age"
    return None


def validate_voice_with_chunkformer(
    audio_path: str,
    description: str,
    chunkformer_model: Any,
    check_age: bool = True,
    verbose: bool = False,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate a voice sample against its description using ChunkFormer.

    Classifies the sample (gender/age/emotion/dialect) and compares gender
    (and optionally age) to what the description asks for. A mismatch only
    fails validation when the classifier is confident (>= 0.7).

    Shared by synthetic voice-sample validation and celebrity-clip validation
    (which passes ``check_age=False`` since celebrity ages vary in-character).

    Args:
        audio_path: Path to the voice sample WAV
        description: Voice description from the LLM
        chunkformer_model: Loaded ChunkFormer model
        check_age: Also validate age when the description mentions one
        verbose: Print debug output

    Returns:
        Tuple of (is_valid, reason, classification) where classification is a
        dict with predicted/expected labels and probabilities for logging.
        On model errors, validation passes (is_valid=True) with the error as
        reason — validation is best-effort and must not block generation.
    """
    try:
        result = chunkformer_model.classify_audio(audio_path=audio_path)

        predicted_gender = result["gender"]["label"]
        predicted_age = result["age"]["label"]
        gender_prob = result["gender"]["prob"]
        age_prob = result["age"]["prob"]

        desc_lower = description.lower().strip()
        expected_gender = _expected_gender_from_description(desc_lower)
        expected_age = _expected_age_from_description(desc_lower)

        is_valid = True
        reasons = []

        if expected_gender is not None and predicted_gender != expected_gender:
            if gender_prob >= CHUNKFORMER_CONFIDENCE_THRESHOLD:
                is_valid = False
                reasons.append(
                    f"gender mismatch: expected {expected_gender}, got {predicted_gender} (conf: {gender_prob:.2f})"
                )
            elif verbose:
                print(f"      [INFO] Gender mismatch ignored (conf: {gender_prob:.2f} < {CHUNKFORMER_CONFIDENCE_THRESHOLD})")

        if check_age and expected_age is not None and predicted_age != expected_age:
            if age_prob >= CHUNKFORMER_CONFIDENCE_THRESHOLD:
                is_valid = False
                reasons.append(
                    f"age mismatch: expected {expected_age}, got {predicted_age} (conf: {age_prob:.2f})"
                )
            elif verbose:
                print(f"      [INFO] Age mismatch ignored (conf: {age_prob:.2f} < {CHUNKFORMER_CONFIDENCE_THRESHOLD})")

        if verbose:
            print(f"      Description: {description[:80]}")
            print(f"      Classified: {predicted_gender} / {predicted_age} / {result['emotion']['label']}")
            print(f"      Expected: gender={expected_gender}, age={expected_age if check_age else '(skipped)'}")
            print(f"      Overall: {'PASS' if is_valid else 'FAIL'}")
            if reasons:
                print(f"      Reasons: {'; '.join(reasons)}")

        classification = {
            "classification": {
                "gender": {"label": predicted_gender, "prob": result["gender"]["prob"]},
                "emotion": {"label": result["emotion"]["label"], "prob": result["emotion"]["prob"]},
                "age": {"label": predicted_age, "prob": result["age"]["prob"]},
            },
            "expected": {
                "gender": expected_gender,
                "age": expected_age if check_age else None,
            },
            "gender_ok": expected_gender is None or predicted_gender == expected_gender,
            "age_ok": (expected_age is None or not check_age or predicted_age == expected_age),
            "is_valid": is_valid,
            "reasons": reasons,
        }
        reason = "; ".join(reasons) if reasons else "Validation passed"
        return is_valid, reason, classification

    except Exception as e:
        if verbose:
            print(f"    ChunkFormer validation error: {e}")
        return True, str(e), {}


def crop_to_ref_text(audio_path: str, output_path: str, ref_words: List[str], transcribed_words: List[str], start_times: List[float], end_times: List[float], verbose: bool = False) -> bool:
    """Crop audio to its full spoken span, trimming only leading/trailing silence.

    Keeps all real speech (preserving natural pacing between words) and just
    removes silent padding at the start/end of the clip. The caller gates the
    sample on a high reference word-match first, so the clip is essentially the
    reference speech and there is no garbled prefix to excise.

    Args:
        audio_path: Path to the source audio file (.wav)
        output_path: Path to write the cropped audio (.wav)
        ref_words: List of words from the reference text (lowercased, no punctuation)
        transcribed_words: List of transcribed words from Whisper
        start_times: Start time for each transcribed word (seconds)
        end_times: End time for each transcribed word (seconds)
        verbose: Print verbose output

    Returns:
        True if cropping was successful, False otherwise
    """
    import pydub

    if len(ref_words) < 3 or len(transcribed_words) == 0:
        return False

    try:
        seg = pydub.AudioSegment.from_wav(audio_path)
    except Exception:
        return False

    # The caller has already gated this sample on >=80% word match, so the clip
    # is essentially the reference speech. Crop to the FULL speech span: trim
    # only leading/trailing silence (preserving natural pacing and all spoken
    # content) rather than the matched-reference-word span, which could discard
    # large amounts of valid speech (e.g. H3's ~14s talking-head clips were
    # being cut down to ~6s).
    from pydub.silence import detect_leading_silence

    def _trim_edges(s: "pydub.AudioSegment") -> "pydub.AudioSegment":
        start_ms = detect_leading_silence(s, silence_threshold=-40)
        end_ms = detect_leading_silence(s.reverse(), silence_threshold=-40)
        # Keep a small buffer so the first/last phoneme isn't clipped.
        start_ms = max(0, start_ms - 150)
        end_ms = max(0, end_ms - 150)
        return s[start_ms: len(s) - end_ms]

    try:
        cropped = _trim_edges(seg)
    except Exception:
        return False

    if len(cropped) < 1000:  # <1s of speech
        return False

    cropped.export(output_path, format="wav")
    return True


def validate_audio_clean(audio_path: str, client: Optional[OpenAI] = None, verbose: bool = False) -> Tuple[bool, str]:
    """Validate that audio contains only clean speech without music or background effects.

    Uses the LLM at the configured validation endpoint to analyze audio quality.

    Args:
        audio_path: Path to the audio file (.wav)
        client: OpenAI client for the validation LLM (created if None)
        verbose: Print verbose output

    Returns:
        Tuple of (is_clean, validation_message)
        - is_clean: True if audio contains only clean speech
        - validation_message: Description of what was detected
    """
    from .config import VOICE_VALIDATION

    model = VOICE_VALIDATION.get("model", "default-model")

    abs_audio_path = os.path.abspath(audio_path)
    file_url = f"file://{abs_audio_path}"

    clean_audio_prompt = """You are an audio quality analyzer. Listen to the audio file and determine if it contains ONLY clean speech.

Respond with a JSON object in this exact format:
{{
    "is_clean": true/false,
    "detected_issues": ["list of any issues detected, empty if clean"],
    "description": "brief description of what you heard"
}}

Consider audio CLEAN if it contains:
- Only human speech/voice
- Normal speech pauses and breathing

Consider audio NOT CLEAN if it contains:
- Music or musical sounds
- Sound effects (doors, footsteps, etc.)
- Background noise or ambience
- Non-speech audio elements
- Distortion or artifacts that aren't natural speech

Respond with ONLY the JSON object, no other text."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": file_url}},
                        {"type": "text", "text": clean_audio_prompt}
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=300,
            extra_body=LLM_NO_THINKING_EXTRA_BODY,
        )

        response_text = response.choices[0].message.content.strip()

        import json as json_module
        try:
            result = json_module.loads(response_text)
            is_clean = result.get("is_clean", False)
            detected_issues = result.get("detected_issues", [])
            description = result.get("description", "")

            if verbose:
                print(f"    [Clean Check] Is clean: {is_clean}")
                if detected_issues:
                    print(f"    [Clean Check] Issues: {', '.join(detected_issues)}")
                print(f"    [Clean Check] Description: {description}")

            if not is_clean and detected_issues:
                return False, f"Audio contains: {', '.join(detected_issues)}"
            elif not is_clean:
                return False, description if description else "Audio is not clean speech"
            return True, "Clean speech detected"

        except json_module.JSONDecodeError:
            if verbose:
                print(f"    [Clean Check] Failed to parse LLM response: {response_text}")
            return False, f"Validation error: could not parse response"

    except Exception as e:
        if verbose:
            print(f"    [Clean Check] Error during validation: {e}")
        return False, f"Validation error: {str(e)}"
