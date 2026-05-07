"""Pure functions for TTS pipeline processing.

This module contains stateless, side-effect-free functions extracted from
audiobook_generator.py for testability. These functions handle:
- Text normalization and preparation
- Postfix handling for validation
- Scoring and ratio calculation
- Clip point calculation for audio trimming
"""

import re
from typing import Any, List, Tuple, Optional


END_CHARACTERS = ["?", ".", "-", ";", ",", "!"]

MIN_RATIO_THRESHOLD = 0.85
MAX_RETRIES = 2


def normalize_script(text: str) -> str:
    """Normalize text for TTS generation.

    Args:
        text: Raw input text

    Returns:
        Normalized text with capitalized first letter and cleaned spacing
    """
    if not text:
        return ""

    full_script = str(text[0].upper() + text[1:])
    full_script = re.sub(r"(\s\.)+", r".", full_script)
    return full_script


def add_postfix(script: str, postfix: Optional[str]) -> Tuple[str, Optional[str]]:
    """Add postfix to script for validation detection.

    Args:
        script: The normalized script text
        postfix: Optional postfix string to append

    Returns:
        Tuple of (modified_script, postfix_detect_token)
        postfix_detect_token is None if no postfix was added
    """
    if not postfix:
        return script, None

    end_characters = END_CHARACTERS
    postfix_detect_token = postfix.strip().split(" ")[0]

    if script[-1] in end_characters:
        modified_script = script + " " + postfix
    else:
        modified_script = script + ". " + postfix

    return modified_script, postfix_detect_token


def prepare_script_for_tts(
    text: str,
    short_text_postfix: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """Prepare a script for TTS generation.

    Combines normalize_script and add_postfix into a single operation.

    Args:
        text: Raw input text
        short_text_postfix: Optional postfix for validation

    Returns:
        Tuple of (prepared_script, postfix_detect_token)
    """
    if not text or not text.strip():
        return "", None

    normalized = normalize_script(text)
    return add_postfix(normalized, short_text_postfix)


def score_strings_pop(
    input_string: str,
    detected_string: str,
    lookahead: int = 5,
    postfix: str = "and also with you"
) -> Tuple[float, Optional[str]]:
    """Score how well detected string matches input string.

    Uses a lookahead-based algorithm to find the last valid token
    position in the detected string and calculates a ratio.

    Args:
        input_string: Distilled input text
        detected_string: Distilled transcribed text
        lookahead: Number of tokens to look ahead for matching
        postfix: Postfix string to check for (reduces score if missing)

    Returns:
        Tuple of (score_ratio, last_valid_token)
        score_ratio is between 0.0 and 1.0
        last_valid_token is the last token found in both strings
    """
    import pandas as pd

    lookahead = max(0, lookahead)
    prev_undetected = False
    results = []
    input_tokens = input_string.split(" ")
    detected_tokens = detected_string.split(" ")
    diff_list = []

    for i, i_tok in enumerate(input_tokens):
        if i_tok in diff_list:
            detected = True
            this_idx = diff_list.index(i_tok)
            detected_tokens = diff_list[this_idx + 1:] + detected_tokens
            diff_list = diff_list[:this_idx]
        else:
            detected = False
            if prev_undetected and len(diff_list) > 0:
                diff_list.pop(0)
            else:
                diff_list = []

            if detected_tokens:
                n = max(min(lookahead, len(detected_tokens) - len(diff_list)), 0)
                for j in range(n):
                    d_tok = detected_tokens.pop(0)
                    diff_list.append(d_tok)
                    if i_tok in diff_list:
                        detected = True
                        break
                if not detected:
                    prev_undetected = True

        diff_str = " ".join(diff_list)
        results.append((i, i_tok, diff_str, detected, " ".join(detected_tokens[:lookahead])))

    df_temp = pd.DataFrame(results, columns=["i", "i_tok", "diff", "found", "next_tokens"])
    last_valid_token_index = df_temp[df_temp["found"] == True]["i"].max()
    last_valid_token = df_temp[df_temp["i"] == last_valid_token_index]["i_tok"]

    if len(last_valid_token.values) == 0:
        return 0.0, None

    postfix_present = postfix and len(postfix) > 0 and postfix in detected_string[-len(postfix):]
    score = float(df_temp["found"].mean()) - 0.5 * (not postfix_present)

    return score, last_valid_token.values[0]


def calculate_clip_points(
    segments: List[str],
    start_times: List[float],
    end_times: List[float],
    postfix_detect_token: Optional[str],
    last_valid_token: Optional[str],
    verbose: bool = False
) -> Optional[Tuple[float, float]]:
    """Calculate audio clip points based on detected tokens.

    Args:
        segments: List of distilled word tokens from STT
        start_times: List of start times for each segment
        end_times: List of end times for each segment
        postfix_detect_token: Token to use as postfix marker (None to skip)
        last_valid_token: Last valid token for fallback clipping
        verbose: Enable debug output

    Returns:
        Tuple of (clip_start_ms, clip_end_ms) or None if clipping not needed
    """
    if not segments or not start_times or not end_times:
        return None

    if postfix_detect_token and postfix_detect_token in segments:
        try:
            postfix_start_index = len(segments) - 1 - segments[::-1].index(postfix_detect_token)

            if len(end_times) > postfix_start_index + 1:
                clip_end_s = end_times[::-1][postfix_start_index + 1]
            else:
                clip_end_s = end_times[-1]

            clip_start_s = start_times[::-1][postfix_start_index]
            clip_start_ms = (clip_start_s + clip_end_s) * 500

            if verbose:
                print(f"POSTFIX DETECTED CLIPPING to {clip_start_s} - {clip_end_s}")

            return None, clip_start_ms
        except (ValueError, IndexError):
            pass

    if last_valid_token and last_valid_token in segments:
        try:
            lastvalid_index = len(segments) - 1 - segments[::-1].index(last_valid_token)
            clip_end_s = end_times[::-1][lastvalid_index]
            clip_end_ms = clip_end_s * 1000

            if verbose:
                print(f"POSTFIX UN-DETECTED LAST VALID CLIPPING TO {last_valid_token} {clip_end_s}")

            return None, clip_end_ms
        except (ValueError, IndexError):
            pass

    if verbose:
        print("No clipping needed")

    return None


def should_retry(
    ratio: float,
    max_ratio: float,
    retries: int,
    max_retries: int = MAX_RETRIES,
    min_ratio: float = MIN_RATIO_THRESHOLD
) -> bool:
    """Determine if TTS generation should retry.

    Args:
        ratio: Current attempt's ratio
        max_ratio: Best ratio achieved so far
        retries: Current retry count
        max_retries: Maximum retry attempts allowed
        min_ratio: Minimum ratio threshold for success

    Returns:
        True if another attempt should be made
    """
    return ratio < min_ratio and retries < max_retries


def generate_output_filename(
    output_dir: str,
    chapter_idx: int,
    line_idx: int,
    is_final: bool = False
) -> str:
    """Generate output filename for TTS audio.

    Args:
        output_dir: Output directory path
        chapter_idx: Chapter index
        line_idx: Line index
        is_final: If True, use .wav extension, else .tmp.wav

    Returns:
        Full path to output file
    """
    import os
    suffix = ".wav" if is_final else ".tmp.wav"
    return os.path.join(
        output_dir,
        f"chapter_{str(chapter_idx).zfill(2)}.{str(line_idx).zfill(4)}{suffix}"
    )


def is_generation_success(
    ratio: float,
    min_ratio: float = MIN_RATIO_THRESHOLD
) -> bool:
    """Check if generation ratio indicates success.

    Args:
        ratio: The ratio score from scoring function
        min_ratio: Minimum threshold for success

    Returns:
        True if ratio meets or exceeds threshold
    """
    return ratio >= min_ratio


def collect_transcription_segments(
    segments_list: Any,
) -> Tuple[List[str], List[float], List[float]]:
    """Collect word segments and timestamps from Whisper transcription.

    Args:
        segments_list: Iterable of segments from Whisper transcription

    Returns:
        Tuple of (segments, start_times, end_times) lists
    """
    from .utils import distill_string

    segments = []
    start_times = []
    end_times = []

    for segment in segments_list:
        for word in segment.words:
            segments.append(distill_string(word.word.strip()))
            start_times.append(word.start)
            end_times.append(word.end)

    return segments, start_times, end_times