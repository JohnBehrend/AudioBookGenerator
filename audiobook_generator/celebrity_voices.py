#!/usr/bin/env python3
"""Celebrity voice matching and downloading for audiobook voice references.

Uses LLM to match characters to celebrities with similar voices,
then downloads and preprocesses audio clips for use as voice references.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yt_dlp

from .config import LLM_SETTINGS


CELEBRITY_MATCHING_PROMPT = """You are a voice matching expert. Given a character description, suggest a celebrity whose voice best matches.

Output ONLY a JSON object with these keys:
- "celebrity": Full name of the celebrity
- "reason": Brief explanation of why this celebrity matches
- "search_query": YouTube search query to find their voice (e.g. "celebrity name interview")

Rules:
- Choose living or recently deceased celebrities with plenty of audio available
- Match voice qualities: gender, age, accent, pitch, tone
- Prefer celebrities known for voice work, interviews, or audiobooks
- Avoid celebrities with very distinctive voices that don't match the character

Example:
{{"celebrity": "Benedict Cumberbatch", "reason": "Deep, resonant British voice matches the aristocratic, authoritative character", "search_query": "Benedict Cumberbatch interview"}}
"""


def match_celebrity(
    client: Any,
    model: str,
    character: str,
    description: str,
    max_retries: int = 2,
) -> Optional[Dict[str, str]]:
    """Use LLM to match a character to a celebrity voice.

    Args:
        client: OpenAI client instance
        model: Model name
        character: Character name
        description: Character voice description (JSON string or dict)
        max_retries: Max retry attempts

    Returns:
        Dict with celebrity, reason, search_query or None on failure
    """
    if isinstance(description, dict):
        desc_str = json.dumps(description)
    else:
        desc_str = description

    messages = [
        {"role": "system", "content": CELEBRITY_MATCHING_PROMPT},
        {"role": "user", "content": f"Match this character to a celebrity voice:\n\nCharacter: {character}\nDescription: {desc_str}"},
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            raw = response.choices[0].message.content
            if not raw:
                messages.append({"role": "assistant", "content": ""})
                messages.append({"role": "user", "content": "Please return valid JSON."})
                continue

            # Extract JSON from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                obj = json.loads(raw[start:end])
                if "celebrity" in obj and "search_query" in obj:
                    return obj

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Please return valid JSON with celebrity and search_query keys."})
        except Exception:
            if attempt == max_retries - 1:
                return None

    return None


def download_celebrity_audio(
    search_query: str,
    output_dir: str,
    max_duration: int = 30,
    file_prefix: str = "",
) -> Optional[str]:
    """Download audio clip of celebrity using yt-dlp.

    Searches YouTube for the query, downloads the best match,
    and extracts a clean audio segment.

    Args:
        search_query: YouTube search query (e.g. "Benedict Cumberbatch interview")
        output_dir: Directory to save audio file
        max_duration: Maximum duration in seconds for the clip
        file_prefix: Prefix for output filename

    Returns:
        Path to downloaded WAV file or None on failure
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prefix = f"{file_prefix}_" if file_prefix else ""
    output_file = out_path / f"{prefix}celebrity_voice.wav"

    # Search YouTube and download
    ydl_opts = {
        "format": "bestaudio/best",
        "extractaudio": True,
        "audioformat": "mp3",
        "audioquality": 5,
        "outtmpl": str(output_file.with_suffix(".mp3")),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 2,
    }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Search YouTube for short clips
                search_url = f"ytsearch1:{search_query} under:1"
                info = ydl.extract_info(search_url, download=False)

                if not info or "title" not in info:
                    # Fallback: try without duration filter
                    search_url = f"ytsearch1:{search_query}"
                    info = ydl.extract_info(search_url, download=False)

                if not info or "title" not in info:
                    return None

                # Download
                ydl.download([search_url])

                mp3_file = output_file.with_suffix(".mp3")
                if not mp3_file.exists():
                    return None

                # Convert to WAV
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(mp3_file), "-q:a", "0", str(output_file)],
                    capture_output=True, timeout=30,
                )
                mp3_file.unlink(missing_ok=True)

                if not output_file.exists():
                    return None

                # Trim to max_duration if needed
                trim_audio(str(output_file), max_duration)

            return str(output_file)

    except Exception as e:
        print(f"  Failed to download audio: {e}")
        if output_file.exists():
            output_file.unlink()
        return None


def trim_audio(audio_path: str, max_duration: int) -> None:
    """Trim audio file to max_duration using ffmpeg.

    Args:
        audio_path: Path to audio file
        max_duration: Maximum duration in seconds
    """
    if not Path(audio_path).exists():
        return

    # Get duration
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=10,
        )
        duration = float(result.stdout.strip())
        if duration <= max_duration:
            return
    except Exception:
        return

    # Trim
    temp_path = audio_path + ".tmp"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-t", str(max_duration),
             "-q:a", "0", temp_path],
            capture_output=True, timeout=30, check=True,
        )
        shutil.move(temp_path, audio_path)
    except Exception:
        if Path(temp_path).exists():
            temp_path.unlink()


def extract_speech_segments(
    audio_path: str,
    output_dir: str,
    file_prefix: str = "",
    min_duration: float = 2.0,
    max_segments: int = 3,
) -> List[str]:
    """Extract clean speech segments from audio file.

    Uses silence detection to split audio into speech segments,
    then selects the best ones.

    Args:
        audio_path: Path to audio file
        output_dir: Directory to save segments
        file_prefix: Prefix for output filenames
        min_duration: Minimum segment duration in seconds
        max_segments: Maximum number of segments to extract

    Returns:
        List of paths to extracted WAV segments
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prefix = f"{file_prefix}_" if file_prefix else ""

    # Use ffmpeg to detect silence and split
    silence_filter = f"silencedetect=noise=-30dB:d=0.5"
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-af", silence_filter, "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )

        # Parse silence start/end from stderr
        silence_starts = []
        silence_ends = []
        for line in result.stderr.split("\n"):
            if "silence_start:" in line:
                try:
                    val = float(line.split("silence_start:")[1].strip())
                    silence_starts.append(val)
                except ValueError:
                    pass
            elif "silence_end:" in line:
                try:
                    val = float(line.split("silence_end:")[1].strip())
                    silence_ends.append(val)
                except ValueError:
                    pass

        if not silence_starts or not silence_ends:
            # No silence detected, use the whole file
            segment_path = out_path / f"{prefix}segment_0.wav"
            shutil.copy2(audio_path, segment_path)
            return [str(segment_path)]

        # Extract speech segments (between silence gaps)
        segments = []
        for i, (start, end) in enumerate(zip(silence_ends, silence_starts)):
            duration = end - start
            if duration >= min_duration:
                segment_path = out_path / f"{prefix}segment_{i}.wav"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", audio_path, "-ss", str(start),
                     "-to", str(end), "-q:a", "0", str(segment_path)],
                    capture_output=True, timeout=30,
                )
                if segment_path.exists():
                    segments.append(str(segment_path))
                if len(segments) >= max_segments:
                    break

        # If no segments found, just use first chunk
        if not segments:
            segment_path = out_path / f"{prefix}segment_0.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-t", str(min_duration + 1),
                 "-q:a", "0", str(segment_path)],
                capture_output=True, timeout=30,
            )
            if segment_path.exists():
                segments.append(str(segment_path))

        return segments

    except Exception:
        # Fallback: just use the whole file
        segment_path = out_path / f"{prefix}segment_0.wav"
        shutil.copy2(audio_path, segment_path)
        return [str(segment_path)]


def build_celebrity_voice(
    client: Any,
    model: str,
    character: str,
    description: str,
    output_dir: str,
    max_duration: int = 30,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Full pipeline: match celebrity, download audio, extract segments.

    Args:
        client: OpenAI client instance
        model: Model name
        character: Character name
        description: Character voice description
        output_dir: Directory to save voice files
        max_duration: Max duration for downloaded clip

    Returns:
        Tuple of (best_voice_path, metadata) or (None, None) on failure
    """
    # Match celebrity
    match = match_celebrity(client, model, character, description)
    if not match:
        return None, None

    celebrity = match["celebrity"]
    search_query = match["search_query"]

    # Download audio
    audio_path = download_celebrity_audio(
        search_query=search_query,
        output_dir=output_dir,
        max_duration=max_duration,
        file_prefix=character,
    )
    if not audio_path:
        return None, None

    # Extract speech segments
    segments = extract_speech_segments(
        audio_path=audio_path,
        output_dir=output_dir,
        file_prefix=character,
    )
    if not segments:
        return None, None

    # Use longest segment as voice reference
    best_segment = max(segments, key=lambda p: Path(p).stat().st_size)

    metadata = {
        "character": character,
        "celebrity": celebrity,
        "reason": match.get("reason", ""),
        "search_query": search_query,
        "audio_source": audio_path,
        "segments": segments,
        "best_segment": best_segment,
    }

    return best_segment, metadata


def match_all_celebrities(
    client: Any,
    model: str,
    characters: Dict[str, str],
    output_dir: str,
    max_duration: int = 30,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Match celebrities for all characters.

    Args:
        client: OpenAI client instance
        model: Model name
        characters: Dict of character_name -> description
        output_dir: Directory to save voice files
        max_duration: Max duration for downloaded clips
        verbose: Print progress

    Returns:
        Dict of character_name -> metadata
    """
    results = {}
    total = len(characters)

    for i, (char, desc) in enumerate(characters.items(), 1):
        if char == "narrator":
            if verbose:
                print(f"  [{i}/{total}] Skipping narrator (no celebrity match)")
            continue

        if verbose:
            print(f"  [{i}/{total}] Matching celebrity for: {char}")

        voice_path, metadata = build_celebrity_voice(
            client=client,
            model=model,
            character=char,
            description=desc,
            output_dir=output_dir,
            max_duration=max_duration,
        )

        if metadata:
            results[char] = metadata
            if verbose:
                print(f"    Matched: {metadata['celebrity']}")
                print(f"    Voice: {voice_path}")
        else:
            if verbose:
                print(f"    Failed to match celebrity")

    return results
