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


def clean_text_for_tts(text: str) -> str:
    """Clean text before sending to TTS engine.

    Removes annotations, stage directions, and other non-speech content
    that could cause extra text or unclear audio in the output.

    Args:
        text: Raw input text from EPUB parsing

    Returns:
        Cleaned text suitable for TTS generation
    """
    if not text or not text.strip():
        return ""

    # Remove parenthetical annotations (e.g., "(sighing)", "(whispering)")
    # Use a loop to handle nested parentheses
    while '(' in text and ')' in text:
        new_text = re.sub(r'\([^()]*\)', '', text)
        if new_text == text:
            break  # No more non-nested matches, stop to avoid infinite loop
        text = new_text

    # Remove bracket annotations (e.g., "[whispering]", "[stage direction]")
    while '[' in text and ']' in text:
        new_text = re.sub(r'\[[^\[\]]*\]', '', text)
        if new_text == text:
            break
        text = new_text

    # Remove asterisk-based emphasis/directions (e.g., "*shouting*"), handling nesting
    while '*' in text:
        new_text = re.sub(r'\*[^*]*\*', '', text)
        if new_text == text:
            # Orphan asterisks that don't form pairs, just remove them
            text = text.replace('*', '')
            break
        text = new_text

    # Remove spaces before punctuation (left behind by annotation removal)
    text = re.sub(r' +([?.!,;:])', r'\1', text)

    # Normalize whitespace (collapse multiple spaces)
    text = re.sub(r'  +', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


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

    Cleans text, normalizes, and adds postfix for validation.

    Args:
        text: Raw input text
        short_text_postfix: Optional postfix for validation

    Returns:
        Tuple of (prepared_script, postfix_detect_token)
    """
    if not text or not text.strip():
        return "", None

    cleaned = clean_text_for_tts(text)
    if not cleaned:
        return "", None

    normalized = normalize_script(cleaned)
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

        results.append((i, i_tok, detected))

    # Compute metrics without pandas
    found_count = sum(1 for _, _, found in results if found)
    total_count = len(results)
    mean_score = found_count / total_count if total_count > 0 else 0.0

    # Find last valid token (highest i where found == True)
    last_valid_token_index = None
    for i, i_tok, found in results:
        if found:
            last_valid_token_index = i

    if last_valid_token_index is None:
        return 0.0, None

    # Get the token at that index
    last_valid_token = None
    for i, i_tok, found in results:
        if i == last_valid_token_index:
            last_valid_token = i_tok
            break

    postfix_present = postfix and len(postfix) > 0 and postfix in detected_string[-len(postfix):]
    score = float(mean_score) - 0.5 * (not postfix_present)

    return score, last_valid_token


def calculate_clip_points(
    segments: List[str],
    start_times: List[float],
    end_times: List[float],
    postfix_detect_token: Optional[str],
    last_valid_token: Optional[str],
    input_tokens: Optional[List[str]] = None,
    verbose: bool = False
) -> Optional[Tuple[float, float]]:
    """Calculate audio clip points based on detected tokens.

    Clips both start (prefix garbage) and end (postfix) of the audio.

    Args:
        segments: List of distilled word tokens from STT
        start_times: List of start times for each segment
        end_times: List of end times for each segment
        postfix_detect_token: Token to use as postfix marker (None to skip)
        last_valid_token: Last valid token for fallback clipping
        input_tokens: Expected input tokens for finding first valid word
        verbose: Enable debug output

    Returns:
        Tuple of (clip_start_ms, clip_end_ms) or None if clipping not needed.
        Both values are in milliseconds relative to audio start.
    """
    if not segments or not start_times or not end_times:
        return None

    # Find start clip point: match a sequence of input tokens to avoid false matches
    clip_start_ms = 0
    start_found = False
    if input_tokens and len(input_tokens) >= 2:
        # Find the first single-token match of any input word in the transcription.
        # This anchors the search so we don't accept windows that start too late.
        first_single_match = None
        for i, seg in enumerate(segments):
            if seg in input_tokens[:5]:
                first_single_match = i
                break

        # Try to find a sequence of 3 input tokens in the transcription
        # Try different starting positions in input to handle first-word mismatches
        match_length = min(3, len(input_tokens))
        best_match_start = None

        for skip in range(min(3, len(input_tokens) - match_length + 1)):
            target = input_tokens[skip:skip + match_length]
            # Limit search to windows that start at or before the first single match
            search_limit = (first_single_match + 1) if first_single_match is not None else len(segments) - match_length + 1
            for i in range(min(search_limit, len(segments) - match_length + 1)):
                window = segments[i:i + match_length]
                matches = sum(1 for t, s in zip(target, window) if t == s)
                if matches >= match_length - 1:
                    best_match_start = i
                    break
            if best_match_start is not None:
                break

        if best_match_start is not None:
            clip_start_ms = max(0, int(start_times[best_match_start] * 1000) - 200)
            start_found = True
            if verbose:
                print(f"PREFIX DETECTED CLIPPING at '{segments[best_match_start]}' ({start_times[best_match_start]:.2f}s)")

        # Fallback: if sequence not found, match any of the first 5 input tokens
        if not start_found:
            fallback_tokens = input_tokens[:5]
            for i, seg in enumerate(segments):
                if seg in fallback_tokens:
                    clip_start_ms = max(0, int(start_times[i] * 1000) - 200)
                    start_found = True
                    if verbose:
                        print(f"PREFIX FALLBACK CLIPPING at '{seg}' ({start_times[i]:.2f}s)")
                    break

    # Find end clip point: before postfix or at last valid token
    clip_end_ms = None

    if postfix_detect_token and postfix_detect_token in segments:
        try:
            # Find last occurrence of postfix token
            postfix_start_index = len(segments) - 1 - segments[::-1].index(postfix_detect_token)

            # Clip before the postfix starts with a small buffer
            # Use midpoint between postfix start and next word's end for safety
            if postfix_start_index == 0:
                # No content before postfix, clip to 0 so guard catches it
                clip_end_s = 0.0
            elif postfix_start_index + 1 < len(segments):
                clip_end_s = (start_times[postfix_start_index] + end_times[postfix_start_index + 1]) / 2
            else:
                clip_end_s = start_times[postfix_start_index]
            clip_end_ms = clip_end_s * 1000

            if verbose:
                print(f"POSTFIX DETECTED CLIPPING at {clip_end_s}s ({clip_end_ms}ms)")
        except (ValueError, IndexError):
            pass

    if clip_end_ms is None and last_valid_token and last_valid_token in segments:
        try:
            lastvalid_index = len(segments) - 1 - segments[::-1].index(last_valid_token)
            clip_end_s = end_times[lastvalid_index]
            clip_end_ms = clip_end_s * 1000

            if verbose:
                print(f"POSTFIX UN-DETECTED LAST VALID CLIPPING TO {last_valid_token} {clip_end_s}")
        except (ValueError, IndexError):
            pass

    # If neither start nor end clipping is needed
    if not start_found and clip_end_ms is None:
        if verbose:
            print("No clipping needed")
        return None

    # If only start clipping is needed, use full audio length
    if clip_end_ms is None:
        clip_end_ms = end_times[-1] * 1000

    # Guard: if start >= end, the content is empty or garbled
    if clip_start_ms >= clip_end_ms:
        if verbose:
            print(f"CLIP START ({clip_start_ms}ms) >= END ({clip_end_ms}ms), skipping")
        return None

    return clip_start_ms, clip_end_ms


def apply_audio_clipping(
    audio_path: str,
    clip_points: Tuple[float, float],
    verbose: bool = False,
) -> bool:
    """Apply audio clipping to a WAV file using calculated clip points.

    Args:
        audio_path: Path to the WAV file to clip (modified in-place)
        clip_points: Tuple of (start_ms, end_ms) from calculate_clip_points
        verbose: Print verbose output

    Returns:
        True if clipping was applied successfully, False otherwise
    """
    import pydub

    clip_start_ms, clip_end_ms = clip_points
    try:
        audio = pydub.AudioSegment.from_wav(audio_path)
        trimmed_audio = audio[int(clip_start_ms):int(clip_end_ms)]
        trimmed_audio.export(audio_path, format="wav")
        return True
    except Exception as e:
        if verbose:
            print(f"Audio clipping failed: {e}")
        return False


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
    is_final: bool = False,
    thread_id: Optional[int] = None,
) -> str:
    """Generate output filename for TTS audio.

    Args:
        output_dir: Output directory path
        chapter_idx: Chapter index
        line_idx: Line index
        is_final: If True, use .wav extension, else .tmp.wav
        thread_id: Optional thread ID for unique temp filenames in parallel mode

    Returns:
        Full path to output file
    """
    import os
    if is_final:
        suffix = ".wav"
    else:
        thread_suffix = f".t{thread_id}" if thread_id is not None else ""
        suffix = f"{thread_suffix}.tmp.wav"
    return os.path.join(
        output_dir,
        f"chapter_{str(chapter_idx).zfill(2)}.{str(line_idx).zfill(4)}{suffix}"
    )


def get_temp_filenames(
    output_dir: str,
    chapter_idx: int,
    line_idx: int,
) -> List[str]:
    """Get all temp filenames for a given chapter+line (across all threads)."""
    import os
    import glob as glob_mod
    pattern = os.path.join(
        output_dir,
        f"chapter_{str(chapter_idx).zfill(2)}.{str(line_idx).zfill(4)}.t*.tmp.wav"
    )
    return glob_mod.glob(pattern)


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