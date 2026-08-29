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


def _normalize_clip_token(token: str) -> str:
    """Lowercase a transcription token and strip punctuation for matching.

    Whisper preserves case and punctuation (e.g. "And", "you.", "Bingley")
    while input/postfix tokens are distilled (lowercase, punctuation removed).
    This makes token matching in calculate_clip_points case/punctuation
    insensitive so a capitalized or punctuated word isn't missed.
    """
    t = token.lower()
    for ch in END_CHARACTERS:
        t = t.replace(ch, "")
    return t



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

    if postfix:
        postfix_present = postfix in detected_string[-len(postfix):]
        score = float(mean_score) - 0.5 * (not postfix_present)
    else:
        score = float(mean_score)

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

    # Whisper preserves case and punctuation (e.g. "And", "you.", "Bingley")
    # while the input/postfix tokens are distilled (lowercase, punctuation
    # removed). Normalize the transcription tokens the same way so matching
    # doesn't fail on case/punctuation and clip the wrong part of the line.
    norm_segments = [_normalize_clip_token(s) for s in segments]

    # Find start clip point: match a sequence of input tokens to avoid false matches
    clip_start_ms = 0
    start_found = False
    if input_tokens and len(input_tokens) >= 2:
        # Find the first single-token match of any input word in the transcription.
        # This anchors the search so we don't accept windows that start too late.
        first_single_match = None
        for i, seg in enumerate(norm_segments):
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
            search_limit = (first_single_match + 1) if first_single_match is not None else len(norm_segments) - match_length + 1
            for i in range(min(search_limit, len(norm_segments) - match_length + 1)):
                window = norm_segments[i:i + match_length]
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
            for i, seg in enumerate(norm_segments):
                if seg in fallback_tokens:
                    clip_start_ms = max(0, int(start_times[i] * 1000) - 200)
                    start_found = True
                    if verbose:
                        print(f"PREFIX FALLBACK CLIPPING at '{seg}' ({start_times[i]:.2f}s)")
                    break

    # Find end clip point: clip at the postfix onset so the final content word
    # is preserved. If the postfix is NOT detected, do NOT clip the end at all:
    # clipping at Whisper's (under-reported) end time / last-valid-token cuts
    # off the final word's tail (e.g. "tank" -> "t-"). Keep the full audio end
    # in that case (the caller handles an end of None by using full duration).
    clip_end_ms = None

    if postfix_detect_token and postfix_detect_token in norm_segments:
        try:
            # Find last occurrence of postfix token
            postfix_start_index = len(norm_segments) - 1 - norm_segments[::-1].index(postfix_detect_token)

            # Clip before the postfix starts
            if postfix_start_index == 0:
                # No content before postfix, clip to 0 so guard catches it
                clip_end_s = 0.0
            else:
                # Clip at the START of the postfix word. Whisper tends to
                # UNDER-report the end time of the final content word, so
                # clipping at (end - buffer) cut off that word's tail (e.g.
                # "girls" losing its final 's'). The postfix word's start is a
                # cleaner boundary (there is typically a pause before it) and
                # keeps the full last content word.
                clip_end_s = start_times[postfix_start_index]
            clip_end_ms = max(0, clip_end_s * 1000)

            if verbose:
                print(f"POSTFIX DETECTED CLIPPING at {clip_end_s}s ({clip_end_ms}ms)")
        except (ValueError, IndexError):
            pass

    # If neither start nor end clipping is needed
    if not start_found and clip_end_ms is None:
        if verbose:
            print("No clipping needed")
        return None

    # Postfix not detected -> keep the full audio end (never cut content).
    if clip_end_ms is None:
        if verbose:
            print("POSTFIX UN-DETECTED: keeping full audio end (no end clip)")
        return clip_start_ms, None

    # Guard: if start >= end, the content is empty or garbled
    if clip_start_ms >= clip_end_ms:
        if verbose:
            print(f"CLIP START ({clip_start_ms}ms) >= END ({clip_end_ms}ms), skipping")
        return None

    return clip_start_ms, clip_end_ms


def _distill_tail_words(word_starts_ms: List[float], threshold_ms: float):
    """Return the Whisper words whose start time is >= threshold_ms.

    Returns a tuple ``(tail_distilled, tail_word_count)`` describing the speech
    that would be *removed* by clipping at ``threshold_ms``. Used to confirm
    that the trailing speech after a candidate boundary is the postfix rather
    than more content.
    """
    from audiobook_generator.utils import distill_string

    tail = [
        w for w, start in word_starts_ms
        if start is not None and start >= threshold_ms
    ]
    return distill_string(" ".join(tail)), len(tail)


def _is_postfix_tail(
    tail_distilled: str,
    postfix_distilled: str,
    tail_word_count: int,
    postfix_word_count: int,
    slack: int = 2,
) -> bool:
    """True if the trailing speech is the postfix (not content + postfix).

    After a true postfix boundary the trailing speech is essentially just the
    postfix (short). If the tail is much longer than the postfix it contains
    content, which means the candidate gap was a content-internal pause or the
    leading silence -- clipping there would cut content. We reject such long
    tails.

    Crucially we do NOT require Whisper to read the content's final word
    correctly -- we only inspect the short trailing region after the boundary.
    """
    if not tail_distilled or not postfix_distilled:
        return False
    if tail_word_count > postfix_word_count + slack:
        return False
    if postfix_distilled in tail_distilled:
        return True
    if tail_distilled in postfix_distilled:
        return True
    return tail_distilled.startswith(postfix_distilled[: max(1, len(postfix_distilled) // 2)])


def refine_clip_end_with_energy(
    audio_path: str,
    clip_end_ms: float,
    postfix_tokens: Optional[List[str]] = None,
    word_starts_ms: Optional[List[float]] = None,
    silence_thresh_db: int = -40,
    min_silence_ms: int = 100,
    margin_ms: int = 20,
) -> float:
    """Refine the end clip point to the start of the postfix speech using energy.

    The postfix is spoken after a brief pause following the last content word.
    This anchors the clip boundary to the ENERGY GAP (the pause before the
    postfix) rather than to Whisper's word timestamps, and uses Whisper only to
    *confirm* that the trailing speech after the candidate gap is the postfix.

    Two refinement steps:

    1. Conservative: if the Whisper-derived clip end (the postfix word's start
       time) falls inside a detected silence gap, clip at the END of that gap
       (minus a small margin) -- i.e. at the start of the speech right after the
       silence. This only ever EXTENDS the clip (keeps more audio), so it can
       never over-clip content.

    2. Robust (only when postfix_tokens + word_starts_ms are supplied): from the
       last gap backward, pick the first gap whose trailing speech matches the
       postfix, and clip at that gap's end. This keeps the clip correct even if
       Whisper misread or mislocated the content ending, because the boundary is
       decided by the energy gap and verified against the postfix string.

    It never collapses the clip (it cannot grab leading silence).

    Args:
        audio_path: Path to the WAV file (must exist)
        clip_end_ms: Rough end clip point (ms) from Whisper-based logic
        postfix_tokens: Postfix word tokens (used for robust gap anchoring)
        word_starts_ms: Whisper word start times, in order, as (word, start_ms)
        silence_thresh_db: Silence threshold in dBFS
        min_silence_ms: Minimum silence length (ms) to consider
        margin_ms: Margin (ms) to stay before the postfix onset

    Returns:
        Refined clip end in ms, or the input clip_end_ms if no suitable pause
        is found (or the clip would otherwise not improve).
    """
    import pydub
    from audiobook_generator.utils import distill_string

    try:
        audio = pydub.AudioSegment.from_wav(audio_path)
    except Exception:
        return clip_end_ms

    total_ms = audio.duration_seconds * 1000
    if clip_end_ms <= 0 or clip_end_ms >= total_ms:
        return clip_end_ms

    silences = pydub.silence.detect_silence(
        audio, min_silence_len=min_silence_ms, silence_thresh=silence_thresh_db
    )

    # Step 1: conservative -- clip_end inside a pause -> extend to pause end.
    for start, end in silences:
        if start < clip_end_ms <= end:
            refined = int(end) - margin_ms
            if refined > clip_end_ms:
                return refined

    # Step 2: robust -- anchor to the last gap whose trailing speech is the
    # postfix. Decouples the boundary from Whisper's (possibly wrong) word read.
    if postfix_tokens and word_starts_ms is not None:
        postfix_distilled = distill_string(" ".join(postfix_tokens))
        postfix_word_count = len(postfix_tokens)
        for start, end in reversed(silences):
            tail, tail_count = _distill_tail_words(word_starts_ms, float(end))
            if _is_postfix_tail(tail, postfix_distilled, tail_count, postfix_word_count):
                refined = int(end) - margin_ms
                # Only ever EXTEND the clip (keep more audio) toward the verified
                # postfix onset. Never move earlier than Whisper's onset, so this
                # can never over-clip content.
                if refined > clip_end_ms:
                    return refined
                break
    return clip_end_ms


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


def detect_onset_tail_penalty(
    input_string: str,
    detected_string: str,
    postfix: str = "",
) -> float:
    """Return a ratio penalty for garbled line onset or truncated tail.

    The TTS clone frequently (a) garbles the FIRST word at line onset (e.g.
    "Rael" spoken as "rail"/"raul", or an extra "yet" prepended) and (b)
    truncates / has its final word clipped off. These problems barely lower the
    word-match ratio, so the line is accepted as-is. This function returns a
    penalty large enough to push the ratio below ``MIN_RATIO_THRESHOLD`` so the
    caller's retry loop regenerates the line with a fresh seed (keeping the best
    attempt). Fuzzy matching avoids false alarms from Whisper mis-transcription.

    Args:
        input_string: Distilled expected script (with postfix, if any).
        detected_string: Distilled Whisper transcription.
        postfix: The distilled postfix string, or "" if none.

    Returns:
        A penalty in [0, 1] to subtract from the ratio.
    """
    from difflib import SequenceMatcher
    from audiobook_generator.utils import distill_string

    def words(s: str) -> List[str]:
        return [w for w in distill_string(s).split() if w]

    def fuzzy(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    exp = words(input_string)
    det = words(detected_string)
    if postfix:
        pf = words(postfix)
        if pf and exp[-len(pf):] == pf:
            exp = exp[:-len(pf)]
        if pf and det[-len(pf):] == pf:
            det = det[:-len(pf)]

    if not exp or not det:
        return 0.0

    penalty = 0.0

    # Truncated tail: the final expected content word must appear near the end.
    # (The garbled-line-onset case is deliberately NOT penalized here: the TTS
    # clone mispronounces the first word consistently, so regenerating with a
    # fresh seed does not fix it and only wastes GPU. See detect_audio_glitches.)
    last_exp = exp[-1]
    tail_win = det[-min(6, len(det)):]
    if not any(fuzzy(w, last_exp) >= 0.8 for w in tail_win):
        penalty += 0.5

    return penalty


def uncover_speech_stats(
    audio_path: str,
    starts: List[float],
    ends: List[float],
    min_silence_len: int = 60,
    silence_thresh: int = -32,
) -> Tuple[int, Optional[int]]:
    """Measure "spoken but no words detected" audio in a line.

    Whisper occasionally fails to turn real speech into words (a garbled "teh"
    fragment, a cut-off word, a residual postfix). Such audio is non-silent but
    is NOT covered by any transcribed word interval. This finds those regions.

    A stricter ``silence_thresh`` (-32 dB) is used so light ambient/background
    noise (wind, room tone) is treated as silence rather than false "speech".

    Args:
        audio_path: Path to the WAV file.
        starts: Whisper word start times (seconds).
        ends: Whisper word end times (seconds).
        min_silence_len: Minimum silence (ms) to split non-silent regions.
        silence_thresh: Silence threshold in dBFS.

    Returns:
        Tuple of:
            uncovered_ms: total ms of non-silent audio with no covering word.
            last_covered_end_ms: end (ms) of the last non-silent region that IS
                covered by a word (the true tail of the final spoken word, by
                energy rather than Whisper's under-reported end time), or None.
    """
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    if not starts:
        return 0, None

    audio = AudioSegment.from_wav(audio_path)
    regions = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    uncovered_ms = 0
    last_covered_end_ms = None
    for s, e in regions:  # s, e in milliseconds
        s_s, e_s = s / 1000.0, e / 1000.0
        covered = any(s_s < end and e_s > start for start, end in zip(starts, ends))
        if covered:
            if last_covered_end_ms is None or e > last_covered_end_ms:
                last_covered_end_ms = e
        else:
            uncovered_ms += int(e - s)
    return uncovered_ms, last_covered_end_ms


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
    segments = []
    start_times = []
    end_times = []

    for segment in segments_list:
        # Handle both dict and object-based segment structures
        if isinstance(segment, dict):
            words = segment.get("words", [])
        else:
            words = getattr(segment, "words", [])

        for word in words:
            if isinstance(word, dict):
                segments.append(word.get("word", "").strip())
                start_times.append(word.get("start", 0.0))
                end_times.append(word.get("end", 0.0))
            else:
                segments.append(word.word.strip())
                start_times.append(word.start)
                end_times.append(word.end)

    return segments, start_times, end_times