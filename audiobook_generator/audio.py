"""Audio processing, gender detection, and voice validation utilities.

This module contains functions for audio analysis, pitch-based gender detection,
and audio quality validation. These were extracted from utils.py for clarity.
"""

import os
from typing import Any, Optional, Tuple, List

from openai import OpenAI


# ============================================================================
# GENDER DETECTION FROM DESCRIPTIONS
# ============================================================================


def extract_gender_from_description(description: str) -> Optional[str]:
    """Extract gender from a voice description string.

    Args:
        description: Voice description (e.g., "A calm, commanding male narrator")

    Returns:
        "male" or "female" if found, None otherwise
    """
    desc_lower = description.lower()
    # Check for "female" first to avoid matching "male" in "female"
    if "female" in desc_lower:
        return "female"
    if "woman" in desc_lower:
        return "female"
    if "male" in desc_lower:
        return "male"
    if "man" in desc_lower:
        return "male"
    return None


# ============================================================================
# PITCH ANALYSIS & GENDER CLASSIFICATION
# ============================================================================


def classify_gender_statistical(
    voiced_f0: "np.ndarray",
    male_ref_mean: float = 122.5,
    female_ref_mean: float = 210.0,
    alpha: float = 0.05,
    verbose: bool = False
) -> Tuple[str, float, str]:
    """Classify gender using one-sample t-test against reference distributions.

    Performs two one-sample t-tests to determine if the pitch distribution
    is statistically consistent with male or female reference distributions.

    Reference distributions based on physiological ranges:
    - Male: 90-155 Hz (mean ~122.5 Hz)
    - Female: 165-255 Hz (mean ~210 Hz)

    Args:
        voiced_f0: Array of voiced pitch values in Hz (from librosa.pyin)
        male_ref_mean: Reference mean pitch for male voices (default: 122.5 Hz)
        female_ref_mean: Reference mean pitch for female voices (default: 210 Hz)
        alpha: Significance level for t-test (default: 0.05)
        verbose: Print detailed analysis

    Returns:
        Tuple of (gender, confidence, reason)
        - gender: 'male' or 'female'
        - confidence: 0.0-1.0 confidence score (higher = more certain)
        - reason: Human-readable explanation of classification
    """
    import numpy as np
    from scipy import stats

    sample_mean = np.mean(voiced_f0)
    sample_std = np.std(voiced_f0, ddof=1) if len(voiced_f0) > 1 else 0
    sample_size = len(voiced_f0)

    if verbose:
        print(f"    Pitch distribution: n={sample_size}, mean={sample_mean:.1f}Hz, std={sample_std:.1f}Hz")
        print(f"    Reference: male={male_ref_mean}Hz, female={female_ref_mean}Hz")

    t_stat_male, p_value_male = stats.ttest_1samp(voiced_f0, male_ref_mean)
    t_stat_female, p_value_female = stats.ttest_1samp(voiced_f0, female_ref_mean)

    if verbose:
        print(f"    T-test vs male ref: t={t_stat_male:.3f}, p={p_value_male:.4f}")
        print(f"    T-test vs female ref: t={t_stat_female:.3f}, p={p_value_female:.4f}")

    is_male = p_value_male > alpha
    is_female = p_value_female > alpha

    if is_male and not is_female:
        confidence = min(1.0, p_value_male)
        return "male", confidence, f"Statistically consistent with male distribution (p={p_value_male:.3f})"

    if is_female and not is_male:
        confidence = min(1.0, p_value_female)
        return "female", confidence, f"Statistically consistent with female distribution (p={p_value_female:.3f})"

    dist_to_male = abs(sample_mean - male_ref_mean)
    dist_to_female = abs(sample_mean - female_ref_mean)

    if dist_to_male < dist_to_female:
        confidence = 0.5 + (dist_to_female - dist_to_male) / (dist_to_female + dist_to_male) * 0.3
        reason = "ambiguous" if is_male and is_female else "unusual distribution"
        return "male", min(1.0, confidence), f"{reason.capitalize()} - closer to male (mean={sample_mean:.1f}Hz)"
    else:
        confidence = 0.5 + (dist_to_male - dist_to_female) / (dist_to_male + dist_to_female) * 0.3
        reason = "ambiguous" if is_male and is_female else "unusual distribution"
        return "female", min(1.0, confidence), f"{reason.capitalize()} - closer to female (mean={sample_mean:.1f}Hz)"


def plot_pitch_histogram(
    voiced_f0: "np.ndarray",
    detected_gender: str,
    output_path: str,
    male_ref_mean: float = 122.5,
    female_ref_mean: float = 210.0,
    ref_std: float = 30.0,
    confidence: float = None,
    reason: str = None
):
    """Generate histogram of pitch distribution with reference overlays.

    Creates a visualization showing:
    - Histogram of detected pitch values
    - Overlaid male and female reference distributions
    - Sample mean and reference means marked

    Args:
        voiced_f0: Array of voiced pitch values in Hz
        detected_gender: 'male', 'female', or 'ambiguous'
        output_path: Path to save the plot (PNG or PDF)
        male_ref_mean: Reference mean for male distribution
        female_ref_mean: Reference mean for female distribution
        ref_std: Standard deviation for reference distributions
        confidence: Confidence score (0-1) if available
        reason: Classification reason text if available
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats as scipy_stats

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(voiced_f0, bins=30, alpha=0.7, color='steelblue', edgecolor='black',
                label=f'Detected pitches (n={len(voiced_f0)}, mean={np.mean(voiced_f0):.1f}Hz)')

        x = np.linspace(60, 280, 500)

        male_pdf = scipy_stats.norm.pdf(x, male_ref_mean, ref_std)
        ax.plot(x, male_pdf * len(voiced_f0) * 0.3, 'r--', linewidth=2,
                label=f'Male ref (μ={male_ref_mean}Hz)')
        ax.axvline(male_ref_mean, color='red', linestyle='--', alpha=0.5, linewidth=1)

        female_pdf = scipy_stats.norm.pdf(x, female_ref_mean, ref_std)
        ax.plot(x, female_pdf * len(voiced_f0) * 0.3, 'm--', linewidth=2,
                label=f'Female ref (μ={female_ref_mean}Hz)')
        ax.axvline(female_ref_mean, color='magenta', linestyle='--', alpha=0.5, linewidth=1)

        sample_mean = np.mean(voiced_f0)
        ax.axvline(sample_mean, color='green', linestyle='-', linewidth=2,
                   label=f'Sample mean ({sample_mean:.1f}Hz)')

        threshold = (male_ref_mean + female_ref_mean) / 2
        ax.axvline(threshold, color='gray', linestyle=':', linewidth=1.5,
                   label=f'Threshold ({threshold:.1f}Hz)')

        ax.set_xlabel('Pitch (Hz)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        title = f'Pitch Distribution - Detected: {detected_gender.upper()}'
        if confidence is not None:
            title += f' (confidence: {confidence:.2f})'
        ax.set_title(title, fontsize=14)

        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(60, 280)
        ax.grid(True, alpha=0.3)

        if reason:
            ax.text(0.02, 0.02, reason, transform=ax.transAxes, fontsize=9,
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    except ImportError:
        print("    matplotlib not available - skipping histogram generation")
    except Exception as e:
        print(f"    Histogram generation error: {e}")


def extract_pitch_from_audio(audio_path: str) -> Tuple["np.ndarray", "np.ndarray"]:
    """Load audio and extract pitch using librosa.pyin.

    Args:
        audio_path: Path to the audio file

    Returns:
        Tuple of (f0, voiced_f0) where:
        - f0: Full pitch contour (includes 0 for unvoiced frames)
        - voiced_f0: Only voiced pitch values (f0 > 0)

    Raises:
        Exception: If audio cannot be loaded or pitch cannot be estimated
    """
    import warnings

    # audioread (librosa dependency) imports deprecated aifc/sunau at module load time.
    # These are removed in Python 3.13. Suppress until audioread is fixed upstream.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'aifc' is deprecated", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message="'sunau' is deprecated", category=DeprecationWarning)
        import librosa

    import numpy as np

    y, sr = librosa.load(audio_path, sr=None)
    pyin_result = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    f0 = pyin_result[0]
    voiced_f0 = f0[f0 > 0]
    return f0, voiced_f0


def detect_gender_from_audio(audio_path: str, threshold_hz: float = 160.0, use_ttest: bool = True,
                             male_ref_mean: float = 122.5, female_ref_mean: float = 210.0,
                             alpha: float = 0.05, verbose: bool = False) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Detect gender from audio file using pitch analysis.

    Uses librosa.pyin for pitch estimation. By default, uses statistical t-test
    for robust classification against reference distributions. Falls back to
    simple threshold comparison if t-test is disabled.

    Args:
        audio_path: Path to the audio file
        threshold_hz: Pitch threshold in Hz for simple comparison (default: 160Hz)
        use_ttest: Use statistical t-test instead of simple threshold (default: True)
        male_ref_mean: Reference mean for male distribution (default: 122.5 Hz)
        female_ref_mean: Reference mean for female distribution (default: 210 Hz)
        alpha: Significance level for t-test (default: 0.05)
        verbose: Print detailed analysis

    Returns:
        Tuple of (gender, confidence, reason)
        - gender: "male" or "female" based on pitch, None if detection fails
        - confidence: 0.0-1.0 confidence score (None if using simple threshold)
        - reason: Human-readable explanation (None if using simple threshold)
    """
    try:
        import numpy as np

        f0, voiced_f0 = extract_pitch_from_audio(audio_path)

        if len(voiced_f0) == 0:
            return None, None, None

        if use_ttest and len(voiced_f0) >= 3:
            gender, confidence, reason = classify_gender_statistical(
                voiced_f0, male_ref_mean, female_ref_mean, alpha, verbose
            )
            return gender, confidence, reason
        else:
            avg_pitch = np.mean(voiced_f0)
            if verbose:
                print(f"    Using threshold method: avg_pitch={avg_pitch:.1f}Hz, threshold={threshold_hz}Hz")
            return ("female", None, None) if avg_pitch > threshold_hz else ("male", None, None)

    except Exception as e:
        if verbose:
            print(f"    Gender detection error: {e}")
        return None, None, None


def correct_voice_gender(
    audio_path: str,
    description: str,
    threshold_hz: float = 160.0,
    male_target_pitch_hz: float = 130.0,
    female_target_pitch_hz: float = 220.0,
    verbose: bool = False,
    use_ttest: bool = True,
    alpha: float = 0.05,
    male_ref_mean: float = 122.5,
    female_ref_mean: float = 210.0,
    plot_histogram: bool = False,
    histogram_dir: str = None
) -> Tuple[bool, str, Optional[str], Optional[float], Optional[float]]:
    """Correct voice gender by pitch shifting if needed.

    Detects current gender from audio pitch and compares to target gender
    extracted from description. If they don't match, applies pitch shift
    using TD-PSOLA algorithm to move toward the target gender's typical pitch range.

    Args:
        audio_path: Path to the audio file (will be overwritten if correction applied)
        description: Voice description containing target gender (e.g., "male voice")
        threshold_hz: Pitch threshold for simple gender detection (default: 160Hz)
        male_target_pitch_hz: Target average pitch for male voices (default: 130Hz)
        female_target_pitch_hz: Target average pitch for female voices (default: 220Hz)
        verbose: Print verbose output
        use_ttest: Use statistical t-test for gender classification (default: True)
        alpha: Significance level for t-test (default: 0.05)
        male_ref_mean: Reference mean for male distribution (default: 122.5 Hz)
        female_ref_mean: Reference mean for female distribution (default: 210 Hz)
        plot_histogram: Generate pitch distribution histogram (default: False)
        histogram_dir: Directory to save histograms (default: same as audio file)

    Returns:
        Tuple of (success, message, final_gender, final_pitch_hz, confidence)
        - success: True if correction was applied (or not needed), False if failed
        - message: Description of what was done
        - final_gender: "male" or "female" based on final pitch, None if detection failed
        - final_pitch_hz: Average pitch in Hz after any correction, None if detection failed
        - confidence: 0.0-1.0 confidence score from t-test, None if using threshold method
    """
    try:
        import warnings

        # audioread (librosa dependency) imports deprecated aifc/sunau at module load time.
        # These are removed in Python 3.13. Suppress until audioread is fixed upstream.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="'aifc' is deprecated", category=DeprecationWarning)
            warnings.filterwarnings("ignore", message="'sunau' is deprecated", category=DeprecationWarning)
            import librosa

        from psola import vocode
        import numpy as np
        from pathlib import Path

        target_gender = extract_gender_from_description(description)
        if target_gender is None:
            return False, "Could not extract target gender from description", None, None, None

        y, sr = librosa.load(audio_path, sr=None)
        f0, voiced_f0 = extract_pitch_from_audio(audio_path)

        if len(voiced_f0) == 0:
            return False, "Could not detect pitch in audio", None, None, None

        current_avg_pitch = np.mean(voiced_f0)

        if use_ttest and len(voiced_f0) >= 3:
            current_gender, confidence, reason = classify_gender_statistical(
                voiced_f0, male_ref_mean, female_ref_mean, alpha, verbose
            )
            if verbose:
                print(f"    Target gender: {target_gender}, Detected gender: {current_gender} "
                      f"({current_avg_pitch:.1f}Hz, confidence: {confidence:.2f})")
                print(f"    Reason: {reason}")
        else:
            current_gender = "female" if current_avg_pitch > threshold_hz else "male"
            confidence = None
            if verbose:
                print(f"    Target gender: {target_gender}, Detected gender: {current_gender} ({current_avg_pitch:.1f}Hz)")

        if plot_histogram:
            hist_dir = histogram_dir if histogram_dir else str(Path(audio_path).parent)
            hist_path = str(Path(hist_dir) / f"{Path(audio_path).stem}_pitch_histogram.png")
            plot_pitch_histogram(
                voiced_f0, current_gender, hist_path,
                male_ref_mean, female_ref_mean, 30.0, confidence, reason if use_ttest else None
            )
            if verbose:
                print(f"    Saved histogram to: {hist_path}")

        if current_gender == target_gender:
            return True, f"Gender already correct ({current_gender})", current_gender, current_avg_pitch, confidence

        target_pitch = female_target_pitch_hz if target_gender == "female" else male_target_pitch_hz
        shift_ratio = target_pitch / current_avg_pitch

        if shift_ratio < 0.5 or shift_ratio > 2.0:
            return False, f"Extreme pitch ({current_avg_pitch:.1f}Hz) beyond correction range - needs regeneration", current_gender, current_avg_pitch, confidence

        shift_percent = (shift_ratio - 1.0) * 100

        if verbose:
            print(f"    Applying pitch shift: {shift_percent:+.1f}% ({current_gender} → {target_gender})")
            print(f"    Current: {current_avg_pitch:.1f}Hz → Target: {target_pitch:.1f}Hz (ratio: {shift_ratio:.2f}x)")

        target_f0 = f0 * shift_ratio
        y_shifted = vocode(y, sr, target_pitch=target_f0)

        import soundfile as sf
        sf.write(audio_path, y_shifted, sr)

        final_pitch = current_avg_pitch * shift_ratio
        final_gender = target_gender
        final_confidence = confidence

        return True, f"Applied pitch shift of {shift_percent:+.1f}% ({current_gender} → {target_gender})", final_gender, final_pitch, final_confidence

    except ImportError as e:
        return False, f"Missing dependency: {e}. Install librosa and psola.", None, None, None
    except Exception as e:
        return False, f"Gender correction error: {e}", None, None, None


# ============================================================================
# AUDIO QUALITY VALIDATION
# ============================================================================


def crop_to_ref_text(audio_path: str, output_path: str, ref_words: List[str], transcribed_words: List[str], start_times: List[float], end_times: List[float], verbose: bool = False) -> bool:
    """Crop audio to the span where reference words are spoken.

    Uses Whisper word-level timestamps to find the contiguous span of
    reference words in the transcription, then crops the audio to that range.
    Ensures the crop starts at a valid reference word, not at garbled prefix.

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

    ref_set = set(ref_words)
    best_start = 0
    best_end = 0
    best_len = 0
    max_gap = 1

    for i in range(len(transcribed_words)):
        gap = 0
        match_count = 0
        for j in range(i, len(transcribed_words)):
            if transcribed_words[j] in ref_set:
                match_count += 1
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
        if match_count > best_len:
            best_len = match_count
            best_start = i
            best_end = j + 1

    if best_len < 3:
        return False

    # Find the first word in the best span that is actually a reference word
    # This ensures we don't start at garbled prefix text
    actual_start = best_start
    for k in range(best_start, best_end):
        if transcribed_words[k] in ref_set:
            actual_start = k
            break

    # Small buffer at start (200ms) to avoid clipping the beginning of the word
    # Large buffer at end (1000ms) to ensure we capture the full word
    start_buffer_ms = 200
    end_buffer_ms = 1000

    crop_start_ms = max(0, int(start_times[actual_start] * 1000) - start_buffer_ms)
    crop_end_ms = min(len(seg), int(end_times[best_end - 1] * 1000) + end_buffer_ms)

    if verbose:
        print(f"  [Crop] Start at '{transcribed_words[actual_start]}' ({start_times[actual_start]:.2f}s), end at '{transcribed_words[best_end-1]}' ({end_times[best_end-1]:.2f}s)")

    cropped = seg[crop_start_ms:crop_end_ms]
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
            max_tokens=300
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
