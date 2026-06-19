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
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yt_dlp

from .config import DEFAULTS, LLM_SETTINGS


# Module-level cache for celebrity audio downloads to avoid duplicate YouTube requests
_celebrity_audio_cache: Dict[str, str] = {}


def _retry_llm_call(func: Callable, max_retries: int = 3, backoff: float = 1.0, verbose: bool = False) -> Any:
    """Retry an LLM call with exponential backoff on connection errors.

    Args:
        func: Function to call (no arguments)
        max_retries: Maximum number of attempts
        backoff: Initial backoff in seconds
        verbose: Print debug output

    Returns:
        Result of func() or None on failure
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            err_str = str(e).lower()
            if "connection" in err_str or "timeout" in err_str or "reset" in err_str:
                wait_time = backoff * (2 ** attempt)
                if verbose:
                    print(f"      [DEBUG] LLM connection error (attempt {attempt+1}/{max_retries}): {e}, retrying in {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                raise
    return None


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
    client: Any = None,
    model: str = "",
    celebrity: str = "",
    description: str = "",
    verbose: bool = False,
) -> Optional[str]:
    """Download audio clip of celebrity using yt-dlp.

    First tries to find the best video by analyzing subtitles,
    then downloads audio only from the selected video.

    Falls back to direct download if subtitle analysis fails.

    Args:
        search_query: YouTube search query (e.g. "Benedict Cumberbatch interview")
        output_dir: Directory to save audio file
        max_duration: Maximum duration in seconds for the clip
        file_prefix: Prefix for output filename
        client: OpenAI client for LLM-based video selection
        model: LLM model name
        celebrity: Celebrity name
        description: Character description
        verbose: Print debug output

    Returns:
        Path to downloaded WAV file or None on failure
    """
    # Check cache first - avoid downloading the same video multiple times
    cache_key = f"{search_query}|{output_dir}"
    if cache_key in _celebrity_audio_cache:
        cached_path = _celebrity_audio_cache[cache_key]
        if os.path.exists(cached_path):
            print(f"    [DEBUG] Using cached celebrity audio: {cached_path}")
            return cached_path
        else:
            # Cache entry is stale, remove it
            del _celebrity_audio_cache[cache_key]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prefix = f"{file_prefix}_" if file_prefix else ""
    output_file = out_path / f"{prefix}celebrity_voice.wav"

    # Try subtitle-based video selection first
    selected_url = None
    best_segment = None
    existing_audio = None
    all_rejected = False
    if client and model and celebrity:
        if verbose:
            print(f"    [DEBUG] Trying subtitle-based video selection...")
        selected_url, best_segment, existing_audio = find_best_celebrity_video(
            client=client,
            model=model,
            search_query=search_query,
            celebrity=celebrity,
            description=description,
            output_dir=output_dir,
            max_results=5,
            verbose=verbose,
        )
        if selected_url:
            if verbose:
                print(f"    [DEBUG] Selected video via subtitles: {selected_url}")
            if best_segment:
                print(f"    [DEBUG] Best segment: [{best_segment['start']:.1f}s-{best_segment['end']:.1f}s]")
            if existing_audio:
                if verbose:
                    print(f"    [DEBUG] Reusing existing audio: {existing_audio}")
        else:
            if verbose:
                print(f"    [DEBUG] All videos rejected by LLM, not falling back to blind download")
            all_rejected = True

    # Use existing audio from video selection, or download fresh
    if existing_audio and os.path.exists(existing_audio):
        # Reuse the audio file from find_best_celebrity_video
        shutil.copy2(existing_audio, str(output_file))
        try:
            os.unlink(existing_audio)
        except Exception:
            pass
    elif all_rejected:
        # All videos were rejected by LLM — don't fall back to blind download
        return None
    else:
        # Download audio (either from selected URL or direct search)
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
                if selected_url:
                    # Download the selected video directly
                    ydl.download([selected_url])
                else:
                    # Search YouTube for short clips
                    search_url = f"ytsearch1:{search_query} short"
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
        except Exception as e:
            print(f"  Failed to download audio: {e}")
            if output_file.exists():
                output_file.unlink()
            return None

    # Trim to max_duration
    trim_audio(str(output_file), max_duration)

    # Cache the result
    _celebrity_audio_cache[cache_key] = str(output_file)
    print(f"    [DEBUG] Cached celebrity audio: {cache_key} -> {output_file}")

    return str(output_file)


def find_best_celebrity_video(
    client: Any,
    model: str,
    search_query: str,
    celebrity: str,
    description: str,
    output_dir: str = ".",
    max_results: int = 5,
    verbose: bool = False,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """Find the best YouTube video for a celebrity using Whisper transcription.

    Searches YouTube, downloads audio, transcribes with Whisper,
    uses LLM to determine which video likely has the celebrity speaking most clearly,
    then returns the video URL, best segment timestamps, and the downloaded audio path.

    Args:
        client: OpenAI client instance
        model: LLM model name
        search_query: YouTube search query
        celebrity: Celebrity name
        description: Character description
        output_dir: Directory for temp files
        max_results: Max number of videos to check
        verbose: Print debug output

    Returns:
        Tuple of (url, best_segment_info, audio_path) where best_segment_info contains
        'start', 'end', 'text' keys, and audio_path is the downloaded WAV file for reuse.
        Returns (None, None, None) on failure.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"    [DEBUG] Searching YouTube for {max_results} videos with query: '{search_query}'")

    # Search YouTube for multiple results - use browser cookies to avoid rate limiting
    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 2,
        # Don't download anything yet, just get info
        "skip_download": True,
        # Use browser cookies to avoid rate limiting
        "cookiesfrombrowser": ("firefox",),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Search and get multiple results
            search_url = f"ytsearch{max_results}:{search_query}"
            info_dict = ydl.extract_info(search_url, download=False)

            if not info_dict:
                if verbose:
                    print(f"    [DEBUG] No results found for '{search_query}'")
                return None, None, None

            # Extract entries from search result dict
            info_list = []
            if isinstance(info_dict, dict):
                if 'entries' in info_dict:
                    info_list = info_dict['entries']
                elif 'url' in info_dict or 'title' in info_dict:
                    info_list = [info_dict]
            elif isinstance(info_dict, list):
                info_list = info_dict

            if not info_list:
                if verbose:
                    print(f"    [DEBUG] No results found for '{search_query}'")
                return None, None, None

            # Filter: celebrity name must appear in the video title
            celeb_name = celebrity.split()[0].lower()  # First word for matching
            filtered = [
                info for info in info_list
                if celeb_name in info.get('title', '').lower()
            ]
            if filtered:
                if verbose:
                    print(f"    [DEBUG] Filtered to {len(filtered)} videos with '{celebrity}' in title (from {len(info_list)})")
                info_list = filtered
            else:
                if verbose:
                    print(f"    [DEBUG] No videos with '{celebrity}' in title, returning None to try next query")
                return None, None, None

            if verbose:
                print(f"    [DEBUG] Found {len(info_list)} videos")

            # Download audio and transcribe with Whisper, stopping early when LLM approves
            candidate_videos = []
            rejected_videos = set()
            best_audio_path = None
            for idx, info in enumerate(info_list):
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                url = info.get('webpage_url', '')

                if verbose:
                    print(f"    [DEBUG] Video {idx+1}: {title[:60]} ({duration}s) - {url}")

                # Download audio for this video
                audio_path = _download_audio_for_transcription(
                    info=info,
                    output_dir=out_path,
                    file_prefix=f"audio_{idx}",
                )
                
                if audio_path:
                    # Transcribe full audio with Whisper (reuse for segment extraction later)
                    whisper_text = _transcribe_with_whisper(
                        audio_path=audio_path,
                        whisper_model=None,
                        max_duration=0,  # Transcribe full audio
                    )
                    
                    if whisper_text:
                        candidate_videos.append({
                            'index': idx,
                            'title': title,
                            'duration': duration,
                            'url': url,
                            'subtitle_text': whisper_text,
                        })
                        if verbose:
                            print(f"    [DEBUG] Whisper transcription found for video {idx+1}: {len(whisper_text)} chars")
                        
                        # Stop early - ask LLM if this video is good enough
                        if len(whisper_text) >= 200:
                            approved, best_segment = _evaluate_single_video_with_llm(
                                client=client,
                                model=model,
                                celebrity=celebrity,
                                description=description,
                                video=candidate_videos[-1],
                                verbose=verbose,
                            )
                            if approved:
                                # Store the best segment info for later extraction
                                if best_segment:
                                    candidate_videos[-1]['best_segment'] = best_segment
                                if verbose:
                                    print(f"    [DEBUG] LLM approved this video, stopping early")
                                best_audio_path = audio_path
                                return candidate_videos[-1]['url'], best_segment, best_audio_path
                            else:
                                rejected_videos.add(candidate_videos[-1]['index'])
                
                # Clean up audio file after transcription (only for non-selected videos)
                if audio_path and Path(audio_path).exists():
                    try:
                        os.unlink(audio_path)
                    except Exception:
                        pass

            if not candidate_videos:
                if verbose:
                    print(f"    [DEBUG] No videos with transcriptions found")
                return None, None, None

            # Filter out rejected videos
            viable = [v for v in candidate_videos if v['index'] not in rejected_videos]
            if not viable:
                if verbose:
                    print(f"    [DEBUG] All {len(candidate_videos)} videos rejected by LLM, returning None")
                return None, None, None

            # If no video was approved early, use LLM to select the best from viable candidates
            if verbose:
                print(f"    [DEBUG] No video approved early, selecting best from {len(viable)} viable candidates")
            best_video = _select_best_video_with_llm(
                client=client,
                model=model,
                celebrity=celebrity,
                description=description,
                candidate_videos=viable,
                verbose=verbose,
            )

            if best_video:
                if verbose:
                    print(f"    [DEBUG] Best video selected: {best_video['title'][:60]} - {best_video['url']}")
                return best_video['url'], best_video.get('best_segment'), best_audio_path
            else:
                if verbose:
                    print(f"    [DEBUG] LLM failed to select best video, returning first viable")
                first = viable[0] if viable else None
                return (first['url'], first.get('best_segment'), best_audio_path) if first else (None, None, None)

    except Exception as e:
        if verbose:
            print(f"    [DEBUG] Error finding best video: {e}")
        return None, None, None


def _download_and_parse_subtitles(
    info: Dict[str, Any],
    output_dir: Path,
    file_prefix: str = "",
) -> Optional[str]:
    """Download and parse subtitles from a YouTube video using yt-dlp.

    Uses yt-dlp to download subtitles directly (handles rate limiting),
    then parses the content.

    Args:
        info: Video info dict from yt-dlp extract_info
        output_dir: Directory to save temporary subtitle files
        file_prefix: Prefix for subtitle files

    Returns:
        Parsed subtitle text or None if no subtitles available
    """
    try:
        # Check for subtitles (manual or auto-generated)
        subs = info.get('subtitles') or {}
        auto_subs = info.get('automatic_captions') or {}

        # Combine both sources
        all_subs = {}
        if subs:
            all_subs.update(subs)
        if auto_subs:
            all_subs.update(auto_subs)

        if not all_subs:
            return None

        # Find English subtitles
        lang_keys = ['en', 'en-US', 'en-GB']
        sub_lang = None
        for lang in lang_keys:
            if lang in all_subs:
                sub_lang = lang
                break

        if not sub_lang:
            # Try first available language
            for lang in all_subs.keys():
                sub_lang = lang
                break

        if not sub_lang:
            return None

        # Get the webpage URL for this video
        video_url = info.get('webpage_url') or info.get('url')
        if not video_url:
            return None

        # Create temp directory for subtitle download
        import tempfile
        with tempfile.TemporaryDirectory(prefix=f"sub_{file_prefix}_") as tmp_dir:
            # Use yt-dlp to download subtitles with browser cookies
            ydl_opts = {
                "format": "bestaudio/best",
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": False,
                "subtitleslangs": [sub_lang],
                "subtitlesformat": "vtt",
                "outtmpl": os.path.join(tmp_dir, "%(id)s"),
                # Use browser cookies to avoid rate limiting
                "cookiesfrombrowser": ("firefox",),
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
            except Exception:
                pass

            # Look for downloaded subtitle files
            for f in Path(tmp_dir).glob("*.vtt"):
                content = f.read_text(encoding='utf-8')
                parsed = _parse_vtt_content(content)
                if parsed:
                    return parsed

            # Try SRT format
            for f in Path(tmp_dir).glob("*.srt"):
                content = f.read_text(encoding='utf-8')
                parsed = _parse_srt_content(content)
                if parsed:
                    return parsed

            # Try JSON3 format
            for f in Path(tmp_dir).glob("*.json3"):
                content = f.read_text(encoding='utf-8')
                parsed = _parse_json3_content(content)
                if parsed:
                    return parsed

        return None

    except Exception as e:
        return None


def _download_audio_for_transcription(
    info: Dict[str, Any],
    output_dir: Path,
    file_prefix: str = "",
    max_duration: int = 300,
) -> Optional[str]:
    """Download audio from a YouTube video for transcription.

    Args:
        info: Video info dict from yt-dlp extract_info
        output_dir: Directory to save audio file
        file_prefix: Prefix for output filename
        max_duration: Maximum duration in seconds

    Returns:
        Path to downloaded WAV file or None on failure
    """
    try:
        video_url = info.get('webpage_url') or info.get('url')
        if not video_url:
            return None

        output_file = output_dir / f"{file_prefix}_whisper_input.wav"

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

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        mp3_file = output_file.with_suffix(".mp3")
        if not mp3_file.exists():
            print(f"    [DEBUG] Audio download failed: {mp3_file} does not exist")
            return None

        # Convert to WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_file), "-q:a", "0", str(output_file)],
            capture_output=True, timeout=30,
        )
        mp3_file.unlink(missing_ok=True)

        if not output_file.exists():
            print(f"    [DEBUG] WAV conversion failed: {output_file} does not exist")
            return None

        # Trim to max_duration
        trim_audio(str(output_file), max_duration)

        return str(output_file)

    except Exception as e:
        print(f"    [DEBUG] Error downloading audio for transcription: {e}")
        return None


def _transcribe_with_whisper(
    audio_path: str,
    whisper_model: Any = None,
    max_duration: int = 30,
) -> Optional[str]:
    """Transcribe audio using Whisper.

    Args:
        audio_path: Path to audio file
        whisper_model: WhisperModel instance (if None, will be created)
        max_duration: Maximum duration to transcribe

    Returns:
        Transcribed text or None on failure
    """
    try:
        from .utils import transcribe_audio_with_whisper

        if not Path(audio_path).exists():
            print(f"    [DEBUG] Whisper input file missing: {audio_path}")
            return None

        # Create Whisper model if needed (using faster_whisper)
        if whisper_model is None:
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("base", device="cuda", compute_type="float16")

        # Transcribe the audio
        transcribed, start_times, end_times = transcribe_audio_with_whisper(whisper_model, audio_path)

        if not transcribed:
            print(f"    [DEBUG] Whisper transcription returned empty for {audio_path}")
            return None

        print(f"    [DEBUG] Whisper transcription succeeded: {len(transcribed)} chars")
        return transcribed

    except Exception as e:
        print(f"    [DEBUG] Error in Whisper transcription: {e}")
        return None


def _download_subtitle_from_url(url: str, output_path: Path) -> bool:
    """Download subtitle content from URL with retry logic.

    Args:
        url: Subtitle URL
        output_path: Where to save the subtitle file

    Returns:
        True if successful, False otherwise
    """
    import time
    import requests

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                output_path.write_text(resp.text, encoding='utf-8')
                return True
            elif resp.status_code == 429:
                # Rate limited, wait and retry
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                return False
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return False

    return False


def _parse_json3_content(content: str) -> Optional[str]:
    """Parse JSON3 subtitle content and extract text with timestamps.

    Args:
        content: Raw JSON3 content string

    Returns:
        Formatted text with timestamps or None if parsing fails
    """
    try:
        data = json.loads(content)
        events = data.get('events', [])
        if not events:
            return None

        result_lines = []
        for event in events:
            start_ms = event.get('tStartMs', 0)
            duration_ms = event.get('dDurationMs', 0)
            end_ms = start_ms + duration_ms

            # Get text segments
            segs = event.get('segs', [])
            text = ''.join(seg.get('utf8', '') for seg in segs).strip()

            # Skip empty or noise text
            if not text or text in ['\n', '-']:
                continue

            # Clean up text (remove newlines within lines)
            text = text.replace('\n', ' ').strip()

            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0

            result_lines.append(f"[{start_sec:.1f}s-{end_sec:.1f}s] {text}")

        return '\n'.join(result_lines) if result_lines else None

    except Exception:
        return None


def _parse_vtt_content(content: str) -> Optional[str]:
    """Parse VTT subtitle content and extract text with timestamps.

    Args:
        content: Raw VTT content string

    Returns:
        Formatted text with timestamps or None if parsing fails
    """
    lines = content.split('\n')
    entries = []
    current_start = None
    current_end = None
    current_text = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip WEBVTT header
        if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
            continue

        # Skip cue identifier numbers
        if line.isdigit():
            continue

        # Check for timestamp line (e.g., "00:00:00.000 --> 00:00:05.000")
        if '-->' in line:
            # Save previous entry
            if current_start and current_text:
                text = ' '.join(current_text).strip()
                if text and text not in ['[Music]', '[Applause]', '[Laughter]', '(music)', '(applause)', '(laughter)']:
                    entries.append({
                        'start': current_start,
                        'end': current_end,
                        'text': text,
                    })
            current_text = []
            parts = line.split('-->')
            if len(parts) == 2:
                current_start = _parse_vtt_time(parts[0].strip())
                current_end = _parse_vtt_time(parts[1].strip())
        else:
            # This is subtitle text
            current_text.append(line)

    # Save last entry
    if current_start and current_text:
        text = ' '.join(current_text).strip()
        if text and text not in ['[Music]', '[Applause]', '[Laughter]', '(music)', '(applause)', '(laughter)']:
            entries.append({
                'start': current_start,
                'end': current_end,
                'text': text,
            })

    if not entries:
        return None

    # Build formatted text
    result_lines = []
    for entry in entries:
        start_str = f"{entry['start']:.1f}"
        end_str = f"{entry['end']:.1f}"
        result_lines.append(f"[{start_str}s-{end_str}s] {entry['text']}")

    return '\n'.join(result_lines)


def _parse_srt_content(content: str) -> Optional[str]:
    """Parse SRT subtitle content and extract text with timestamps.

    Args:
        content: Raw SRT content string

    Returns:
        Formatted text with timestamps or None if parsing fails
    """
    lines = content.split('\n')
    entries = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Check if this is a sequence number
        if line.isdigit():
            i += 1
            if i >= len(lines):
                break

            # Parse timestamp line (e.g., "00:00:00,000 --> 00:00:05,000")
            time_line = lines[i].strip()
            if '-->' in time_line:
                parts = time_line.split('-->')
                if len(parts) == 2:
                    start = _parse_srt_time(parts[0].strip())
                    end = _parse_srt_time(parts[1].strip())

                    # Collect text lines until next entry
                    text_lines = []
                    i += 1
                    while i < len(lines):
                        text_line = lines[i].strip()
                        if not text_line:
                            break
                        # Stop if next line is a number (sequence number)
                        if text_line.isdigit():
                            break
                        text_lines.append(text_line)
                        i += 1

                    text = ' '.join(text_lines).strip()
                    if text:
                        entries.append({
                            'start': start,
                            'end': end,
                            'text': text,
                        })
                    continue

            i += 1
        else:
            i += 1

    if not entries:
        return None

    # Build formatted text
    result_lines = []
    for entry in entries:
        result_lines.append(f"[{entry['start']:.1f}s-{entry['end']:.1f}s] {entry['text']}")

    return '\n'.join(result_lines)


def _parse_srt_time(time_str: str) -> float:
    """Parse SRT time string (e.g., "00:00:05,000") to seconds.

    Args:
        time_str: SRT time string

    Returns:
        Time in seconds as float
    """
    try:
        time_str = time_str.strip()
        # Handle format "HH:MM:SS,mmm"
        if ',' in time_str:
            parts = time_str.split(',')
            time_part = parts[0]
            ms = int(parts[1]) if len(parts) > 1 else 0
        else:
            time_part = time_str
            ms = 0

        if ':' in time_part:
            time_parts = time_part.split(':')
            hours = int(time_parts[0]) if len(time_parts) > 2 else 0
            minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
            seconds = int(time_parts[2]) if len(time_parts) > 0 else 0
            return hours * 3600 + minutes * 60 + seconds + ms / 1000.0

        return float(time_str)
    except (ValueError, IndexError):
        return 0.0


def _parse_vtt_time(time_str: str) -> float:
    """Parse VTT time string (e.g., "00:00:05.000") to seconds.

    Args:
        time_str: VTT time string

    Returns:
        Time in seconds as float
    """
    try:
        # Remove any extra spaces
        time_str = time_str.strip()

        # Handle format "HH:MM:SS.mmm" or "MM:SS.mmm"
        if '.' in time_str:
            parts = time_str.split('.')
            if ':' in parts[0]:
                time_parts = parts[0].split(':')
                hours = int(time_parts[0]) if len(time_parts) > 2 else 0
                minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
                seconds = int(time_parts[2]) if len(time_parts) > 0 else 0
                return hours * 3600 + minutes * 60 + seconds + float(f"0.{parts[1]}")
            else:
                return float(time_str)

        # Handle format without milliseconds
        if ':' in time_str:
            time_parts = time_str.split(':')
            hours = int(time_parts[0]) if len(time_parts) > 2 else 0
            minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
            seconds = int(time_parts[2]) if len(time_parts) > 0 else 0
            return hours * 3600 + minutes * 60 + seconds

        return float(time_str)
    except (ValueError, IndexError):
        return 0.0


def _select_best_video_with_llm(
    client: Any,
    model: str,
    celebrity: str,
    description: str,
    candidate_videos: List[Dict[str, Any]],
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Use LLM to select the best video based on subtitles.

    Args:
        client: OpenAI client instance
        model: LLM model name
        celebrity: Celebrity name
        description: Character description
        candidate_videos: List of candidate videos with subtitles
        verbose: Print debug output

    Returns:
        Best video dict or None on failure
    """
    if not candidate_videos:
        return None

    # Build prompt for LLM
    videos_summary = []
    for i, vid in enumerate(candidate_videos):
        # Truncate subtitle text to keep prompt manageable
        subtitle_text = vid.get('subtitle_text', '')[:2000]
        videos_summary.append(
            f"Video {i+1}: {vid['title'][:80]}\n"
            f"Duration: {vid['duration']}s\n"
            f"Subtitles:\n{subtitle_text}\n"
        )

    videos_text = "\n---\n".join(videos_summary)

    prompt = f"""You are analyzing YouTube videos to find one that contains speech segments suitable for extracting {celebrity}'s voice.

Given character description: {description}

These videos may contain multiple speakers (interviews, podcasts, etc.). Your task is to find which video has the BEST segments where {celebrity} speaks clearly enough to extract their voice for voice cloning.

Consider:
1. Does the video contain any segments where {celebrity} is actually speaking (first-person, personal experiences)?
2. Which video has the most clear, isolated segments of {celebrity} speaking?
3. Are there usable segments even if other speakers are present?

Return ONLY a JSON object with:
- "best_index": 0-based index of the best video (integer)
- "reason": Brief explanation why this video was chosen

Example:
{{"best_index": 2, "reason": "This interview has Awkwafina answering questions directly with clear speech segments suitable for voice extraction"}}
"""

    def _llm_call():
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Select the best video for {celebrity} from these {len(candidate_videos)} options:\n\n{videos_text}"},
            ],
        )
        return response.choices[0].message.content

    raw = _retry_llm_call(_llm_call, max_retries=3, backoff=2.0, verbose=verbose)
    if not raw:
        if verbose:
            print(f"    [DEBUG] Error selecting best video: Connection error (all retries failed)")
        return None

    # Parse JSON
    start_idx = raw.find("{")
    end_idx = raw.rfind("}") + 1
    if start_idx >= 0 and end_idx > start_idx:
        result = json.loads(raw[start_idx:end_idx])
        best_idx = result.get('best_index', 0)
        reason = result.get('reason', 'N/A')

        if verbose:
            print(f"    [DEBUG] LLM selected video {best_idx}: {reason}")

        if 0 <= best_idx < len(candidate_videos):
            return candidate_videos[best_idx]

    return None


def _evaluate_single_video_with_llm(
    client: Any,
    model: str,
    celebrity: str,
    description: str,
    video: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Ask LLM if a single video is good enough for celebrity voice.

    Returns tuple of (approved, best_segment_info) where best_segment_info
    contains 'start', 'end', 'text' if approved.
    """
    subtitle_text = video.get('subtitle_text', '')[:2000]

    prompt = f"""You are evaluating a YouTube video to determine if it contains speech segments suitable for extracting the celebrity {celebrity}'s voice.

Character description: {description}

Video: {video['title']}
Duration: {video['duration']}s
Transcription excerpt:
{subtitle_text}

This video may contain multiple speakers (interviews, podcasts, conversations). Your task is to identify the BEST segment where {celebrity} speaks clearly enough to extract their voice.

CRITICAL REQUIREMENTS:
- The selected segment MUST be at least 10 seconds long
- The segment should be between 10 and 30 seconds
- Choose a continuous portion of speech, not fragmented fragments

Consider:
1. Does {celebrity} speak at all in this video? Look for first-person statements, personal experiences, or direct dialogue.
2. Are there any clear, isolated segments of {celebrity} speaking (at least 10 seconds)?
3. Is the audio quality acceptable for voice extraction?

Return ONLY a JSON object with:
- "approved": true or false (boolean)
- "reason": Brief explanation
- "best_start": Start time in seconds of the best segment (float), e.g. 5.2
- "best_end": End time in seconds of the best segment (float), e.g. 15.8 (must be at least 10 seconds after start)
- "best_text": The transcribed text of the best segment

If not approved, set best_start/best_end to null.

Example:
{{"approved": true, "reason": "Ryan Reynolds speaks directly in several interview segments", "best_start": 5.2, "best_end": 15.8, "best_text": "I think the key is to stay focused and keep pushing forward"}}
"""

    def _llm_call():
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Is this video good enough for {celebrity}'s voice? Find the best segment."},
            ],
        )
        return response.choices[0].message.content

    raw = _retry_llm_call(_llm_call, max_retries=3, backoff=2.0, verbose=verbose)
    if not raw:
        if verbose:
            print(f"    [DEBUG] Error evaluating video: Connection error (all retries failed)")
        return False, None

    # Parse JSON
    start_idx = raw.find("{")
    end_idx = raw.rfind("}") + 1
    if start_idx >= 0 and end_idx > start_idx:
        result = json.loads(raw[start_idx:end_idx])
        approved = result.get('approved', False)
        reason = result.get('reason', 'N/A')
        best_start = result.get('best_start')
        best_end = result.get('best_end')
        best_text = result.get('best_text')

        if verbose:
            print(f"    [DEBUG] LLM evaluated video: {'APPROVED' if approved else 'REJECTED'} - {reason}")

        best_segment = None
        if approved and best_start is not None and best_end is not None:
            seg_start = float(best_start)
            seg_end = float(best_end)
            seg_duration = seg_end - seg_start

            # Enforce minimum segment length (10 seconds)
            if seg_duration < 10.0:
                if verbose:
                    print(f"    [DEBUG] Segment too short ({seg_duration:.1f}s), rejecting video")
                return False, None

            best_segment = {
                'start': seg_start,
                'end': seg_end,
                'text': best_text or '',
            }
            if verbose:
                print(f"    [DEBUG] Best segment: [{seg_start:.1f}s-{seg_end:.1f}s] ({seg_duration:.1f}s) {best_text[:80]}...")

        return approved, best_segment

    return False, None


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
    temp_path = audio_path + ".tmp.wav"
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
        # silence_ends[i] to silence_starts[i+1] = speech region
        segments = []
        for i, (start, end) in enumerate(zip(silence_ends, silence_starts[1:])):
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


CELEBRITY_SPEECH_SEGMENT_PROMPT = """You are analyzing a transcribed audio clip that may contain multiple speakers. You must identify which speaker is the celebrity and select their speech segments.

Given:
- Celebrity name: {celebrity}
- Character description: {description}
- Transcription with timestamps: {transcription}

CRITICAL INSTRUCTIONS:
1. FIRST identify all speakers in the audio. Look for patterns like:
   - Host/interviewer asking questions (often shorter, directed at someone)
   - Guest/celebrity answering (longer, more personal responses)
   - Third parties talking ABOUT the celebrity (mentions by name, "she said", "he mentioned")
   - Multiple people talking at once or overlapping

2. Then select segments where the celebrity {celebrity} is speaking. A segment should NOT be selected if:
   - Someone else is talking about the celebrity
   - The speaker is clearly an interviewer/host asking questions
   - The content suggests a third party describing events
   - The speaker is introducing or talking about the celebrity rather than being the celebrity

3. The celebrity's speech should:
   - Match the gender described in the character description (if female, look for female speech; if male, look for male speech)
   - Use first-person language ("I", "my", "we") rather than third-person references to the celebrity
   - Be the actual voice of the celebrity, not someone else talking about them

4. If there are multiple speakers and you're unsure which is the celebrity, select segments from BOTH speakers and add them all to the same array so the pipeline can try each voice.

Output ONLY a JSON array of objects, each with:
- "start": start time in seconds (float)
- "end": end time in seconds (float)
- "text": the transcribed text for this segment

Select up to 6 segments total (including alternatives if unsure).

Example:
[
  {{"start": 5.2, "end": 12.8, "text": "I think the key is..."}},
  {{"start": 18.0, "end": 25.5, "text": "Every challenge is..."}}
]

IMPORTANT: Segments must be at least 2 seconds long. If you cannot find any usable segments, return an empty array [].
"""


def identify_celebrity_segments(
    client: Any,
    model: str,
    celebrity: str,
    description: str,
    audio_path: str,
    whisper_model: Any,
    max_segments: int = 3,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Use Whisper + LLM to identify which segments contain celebrity speech.

    Args:
        client: OpenAI client instance
        model: LLM model name
        celebrity: Celebrity name
        description: Character description (JSON string)
        audio_path: Path to the audio file
        whisper_model: WhisperModel instance (if None, loads base model)
        max_segments: Maximum number of segments to return
        verbose: Print debug output

    Returns:
        List of dicts with 'start', 'end', 'text' keys, sorted by relevance
    """
    try:
        from .utils import transcribe_audio_with_whisper

        if verbose:
            print(f"      [DEBUG] Transcribing celebrity audio with Whisper...")

        # Load whisper model if not provided (using faster_whisper)
        if whisper_model is None:
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("base", device="cuda", compute_type="float16")

        # Transcribe the full audio with word-level timestamps
        transcribed, start_times, end_times = transcribe_audio_with_whisper(whisper_model, audio_path)

        if not transcribed or not start_times:
            if verbose:
                print(f"      [DEBUG] No transcription result for {audio_path}")
            return []

        if verbose:
            print(f"      [DEBUG] Transcription: {transcribed[:200]}...")

        # Build a readable transcription with timestamps
        # Group words into sentences (approximate: split on punctuation)
        sentences = []
        current_sentence_words = []
        current_start = None
        current_end = None

        for i, (word, start, end) in enumerate(zip(transcribed.split(), start_times, end_times)):
            if current_start is None:
                current_start = start
            current_end = end
            current_sentence_words.append(word)

            # Split on sentence-ending punctuation
            if word.endswith(('.', '!', '?', ';')) or i == len(start_times) - 1:
                sentences.append({
                    'start': round(current_start, 2),
                    'end': round(current_end, 2),
                    'text': ' '.join(current_sentence_words),
                })
                current_sentence_words = []
                current_start = None
                current_end = None

        if not sentences:
            if verbose:
                print(f"      [DEBUG] No sentences found in transcription")
            return []

        # Build timestamped transcription string for the LLM
        timestamped_lines = []
        for sent in sentences:
            timestamped_lines.append(f"[{sent['start']:.1f}s-{sent['end']:.1f}s] {sent['text']}")
        timestamped_text = '\n'.join(timestamped_lines)

        if verbose:
            print(f"      [DEBUG] Timestamped transcription ({len(sentences)} sentences):")
            for line in timestamped_lines[:10]:
                print(f"        {line}")

        # Use LLM to identify celebrity speech segments
        prompt = CELEBRITY_SPEECH_SEGMENT_PROMPT.format(
            celebrity=celebrity,
            description=description,
            transcription=timestamped_text,
        )

        def _llm_call():
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Analyze the transcription and identify which segments are spoken by the celebrity."},
                ],
            )
            return response.choices[0].message.content

        raw = _retry_llm_call(_llm_call, max_retries=3, backoff=2.0, verbose=verbose)
        if not raw:
            if verbose:
                print(f"      [DEBUG] Error identifying celebrity segments: Connection error (all retries failed)")
            return []

        # Parse JSON array from response
        start_idx = raw.find("[")
        end_idx = raw.rfind("]") + 1
        if start_idx >= 0 and end_idx > start_idx:
            try:
                segments = json.loads(raw[start_idx:end_idx])
            except json.JSONDecodeError:
                # Try to clean up the JSON (remove extra commas, etc.)
                cleaned = raw[start_idx:end_idx]
                cleaned = re.sub(r',\s*]', ']', cleaned)
                cleaned = re.sub(r'\[\s*,', '[', cleaned)
                try:
                    segments = json.loads(cleaned)
                except json.JSONDecodeError:
                    if verbose:
                        print(f"      [DEBUG] Failed to parse JSON from LLM response: {raw[start_idx:end_idx][:200]}")
                    return []
            # Validate and filter
            valid_segments = []
            for seg in segments:
                if isinstance(seg, dict) and all(k in seg for k in ['start', 'end', 'text']):
                    duration = seg['end'] - seg['start']
                    if duration >= 2.0:
                        valid_segments.append(seg)
                        if verbose:
                            print(f"      [DEBUG] Selected segment: [{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['text'][:80]}...")
            return valid_segments[:max_segments]

        return []

    except Exception as e:
        if verbose:
            print(f"      [DEBUG] Error identifying celebrity segments: {e}")
        return []


def _extract_segment_from_audio(
    audio_path: str,
    start: float,
    end: float,
    output_path: str,
) -> bool:
    """Extract a segment from an audio file using ffmpeg.

    Args:
        audio_path: Path to source audio file
        start: Start time in seconds
        end: End time in seconds
        output_path: Path to write extracted segment

    Returns:
        True if extraction succeeded, False otherwise
    """
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ss", str(start),
             "-to", str(end), "-q:a", "0", output_path],
            capture_output=True, timeout=30,
        )
        return Path(output_path).exists()
    except Exception:
        return False


def find_and_extract_video_segment(
    client: Any,
    model: str,
    search_query: str,
    celebrity: str,
    description: str,
    output_dir: str,
    file_prefix: str,
    max_duration: int = 300,
    whisper_model: Any = None,
    max_segments: int = 3,
    verbose: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Download one video, transcribe once, LLM evaluates and picks segments, extract to WAV.

    Single transcription is used for both LLM evaluation and segment extraction.

    Args:
        client: OpenAI client instance
        model: Model name
        search_query: YouTube search query
        celebrity: Celebrity name
        description: Character voice description
        output_dir: Directory to save files
        file_prefix: Prefix for output filenames
        max_duration: Max duration for downloaded clip
        whisper_model: WhisperModel for transcription
        max_segments: Maximum number of segments to extract from this video
        verbose: Print debug output

    Returns:
        Tuple of (segment_path, audio_source_path) or (None, None) on failure
    """
    audio_path = download_celebrity_audio(
        search_query=search_query,
        output_dir=output_dir,
        max_duration=max_duration,
        file_prefix=file_prefix,
        client=client,
        model=model,
        celebrity=celebrity,
        description=description,
        verbose=verbose,
    )

    if not audio_path:
        return None, None

    # Transcribe once, reuse for both LLM evaluation and segment extraction
    if whisper_model:
        try:
            from .utils import transcribe_audio_with_whisper
            transcribed_text, start_times, end_times = transcribe_audio_with_whisper(whisper_model, audio_path)
        except Exception as e:
            if verbose:
                print(f"    [DEBUG] Transcription failed: {e}")
            transcribed_text = None

        if transcribed_text:
            if verbose:
                print(f"    [DEBUG] Transcribed {len(transcribed_text)} chars (single pass)")

            # Build timestamped transcription for LLM
            sentences = []
            current_words = []
            current_start = None
            current_end = None
            for i, (word, start, end) in enumerate(zip(transcribed_text.split(), start_times, end_times)):
                if current_start is None:
                    current_start = start
                current_end = end
                current_words.append(word)
                if word.endswith(('.', '!', '?', ';')) or i == len(start_times) - 1:
                    sentences.append({
                        'start': round(current_start, 2),
                        'end': round(current_end, 2),
                        'text': ' '.join(current_words),
                    })
                    current_words = []
                    current_start = None
                    current_end = None

            timestamped_lines = [f"[{s['start']:.1f}s-{s['end']:.1f}s] {s['text']}" for s in sentences]
            timestamped_text = '\n'.join(timestamped_lines)

            # Ask LLM to identify the best segment
            prompt = CELEBRITY_SPEECH_SEGMENT_PROMPT.format(
                celebrity=celebrity,
                description=description,
                transcription=timestamped_text,
            )

            def _llm_call():
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Analyze the transcription and identify the best segment spoken by the celebrity."},
                    ],
                )
                return response.choices[0].message.content

            raw = _retry_llm_call(_llm_call, max_retries=3, backoff=2.0, verbose=verbose)
            if raw:
                start_idx = raw.find("[")
                end_idx = raw.rfind("]") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    try:
                        segments = json.loads(raw[start_idx:end_idx])
                    except json.JSONDecodeError:
                        cleaned = re.sub(r',\s*]', ']', raw[start_idx:end_idx])
                        try:
                            segments = json.loads(cleaned)
                        except json.JSONDecodeError:
                            segments = []

                    valid_segments = [
                        seg for seg in segments
                        if isinstance(seg, dict) and all(k in seg for k in ['start', 'end', 'text'])
                        and (seg['end'] - seg['start']) >= 2.0
                    ][:max_segments]

                    if valid_segments:
                        for seg_idx, seg in enumerate(valid_segments):
                            seg_path = Path(output_dir) / f"{file_prefix}_segment{seg_idx}.wav"
                            if _extract_segment_from_audio(audio_path, seg['start'], seg['end'], str(seg_path)):
                                if verbose:
                                    print(f"    [DEBUG] Extracted segment {seg_idx}: {seg_path}")
                        return str(Path(output_dir) / f"{file_prefix}_segment0.wav"), audio_path

    # Fallback: silence detection
    segments = extract_speech_segments(
        audio_path=audio_path,
        output_dir=output_dir,
        file_prefix=file_prefix,
        max_segments=1,
    )

    if segments:
        seg_path = Path(output_dir) / f"{file_prefix}_segment.wav"
        shutil.copy2(segments[0], seg_path)
        for s in segments:
            try:
                os.unlink(s)
            except OSError:
                pass
        return str(seg_path), audio_path

    return None, None


def validate_celebrity_segment(
    segment_path: str,
    description: str,
    chunkformer_model: Any = None,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """Validate a celebrity voice segment against the character description.

    Uses ChunkFormer for gender/age validation.

    Args:
        segment_path: Path to the segment WAV file
        description: Character voice description
        chunkformer_model: Loaded ChunkFormer model (optional)
        verbose: Print debug output

    Returns:
        Tuple of (is_valid, reason)
    """
    if not Path(segment_path).exists():
        return False, "Segment file does not exist"

    file_size = Path(segment_path).stat().st_size
    if file_size < 1000:
        return False, f"File too small ({file_size} bytes)"

    if chunkformer_model:
        try:
            result = chunkformer_model.classify_audio(audio_path=segment_path)
            predicted_gender = result["gender"]["label"]
            gender_prob = result["gender"]["prob"]

            desc_lower = description.lower()
            expected_gender = "female" if any(w in desc_lower for w in ["female", "woman", "women", "girl"]) else ("male" if any(w in desc_lower for w in ["male", "man", "men", "boy"]) else None)

            GENDER_CONFIDENCE_THRESHOLD = 0.7
            if expected_gender is not None and predicted_gender != expected_gender:
                if gender_prob >= GENDER_CONFIDENCE_THRESHOLD:
                    return False, f"Gender mismatch: expected {expected_gender}, got {predicted_gender} (conf: {gender_prob:.2f})"
                elif verbose:
                    print(f"    [DEBUG] Gender mismatch ignored (conf: {gender_prob:.2f} < {GENDER_CONFIDENCE_THRESHOLD})")
        except Exception as e:
            if verbose:
                print(f"    [DEBUG] ChunkFormer validation error: {e}")

    return True, "Validation passed"


def generate_celebrity_reference(
    segment_path: str,
    character: str,
    output_dir: str,
    engine: Any,
    static_text: str,
    verbose: bool = False,
) -> Tuple[Optional[str], float]:
    """Generate a TTS reference WAV using a celebrity segment as voice reference.

    Args:
        segment_path: Path to celebrity segment WAV
        character: Character name
        output_dir: Directory to save output
        engine: TTS engine instance
        static_text: Text to synthesize
        verbose: Print debug output

    Returns:
        Tuple of (reference_path, duration_seconds) or (None, 0.0) on failure
    """
    ref_path = Path(output_dir) / f"{character}_ref.wav"

    try:
        success = engine.generate_line(
            text=static_text,
            voice_path=segment_path,
            output_path=str(ref_path),
            verbose=verbose,
            ref_text="",
        )

        if not success or not ref_path.exists():
            return None, 0.0

        file_size = ref_path.stat().st_size
        duration = file_size / (24000 * 2)

        if duration < 2.0 or duration > 60.0:
            if verbose:
                print(f"    [DEBUG] Reference duration {duration:.1f}s outside range")
            return None, 0.0

        return str(ref_path), duration
    except Exception as e:
        if verbose:
            print(f"    [DEBUG] Reference generation error: {e}")
        return None, 0.0


def build_celebrity_voice(
    client: Any,
    model: str,
    character: str,
    description: str,
    output_dir: str,
    max_duration: int = 300,
    pre_matched_celebrity: Optional[str] = None,
    max_videos: int = 3,
    whisper_model: Any = None,
    chunkformer_model: Any = None,
    tts_engine: Any = None,
    verbose: bool = False,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Full pipeline: match celebrity, download videos, validate per-video, early exit.

    Flow:
    1. Match celebrity (or use pre-matched)
    2. For each video (up to max_videos):
       a. Download video, extract best segment
       b. Validate segment with ChunkFormer
       c. Generate TTS reference from segment
       d. If valid, return immediately
    3. If all fail, return first generated reference anyway

    Args:
        client: OpenAI client instance
        model: Model name
        character: Character name
        description: Character voice description
        output_dir: Directory to save voice files
        max_duration: Max duration for downloaded clip
        pre_matched_celebrity: Optional pre-matched celebrity name
        max_videos: Maximum number of videos to try
        whisper_model: WhisperModel for transcription
        chunkformer_model: ChunkFormer model for validation
        tts_engine: TTS engine for reference generation
        verbose: Print debug output

    Returns:
        Tuple of (best_reference_path, metadata) or (None, None) on failure
    """
    base_character = re.sub(r'\.sample\d+$', '', character)

    # Match celebrity
    if pre_matched_celebrity:
        if verbose:
            print(f"    [DEBUG] Using pre-matched celebrity: {pre_matched_celebrity}")
        try:
            desc_obj = json.loads(description) if isinstance(description, str) else description
            style = desc_obj.get("style", "") if isinstance(desc_obj, dict) else ""
            gender = desc_obj.get("gender", "") if isinstance(desc_obj, dict) else ""
        except (json.JSONDecodeError, AttributeError):
            style = ""
            gender = ""
        match = {
            "celebrity": pre_matched_celebrity,
            "reason": "Pre-matched from character description",
            "search_query": f"{pre_matched_celebrity} {style} dialogue",
        }
    else:
        if verbose:
            print(f"    [DEBUG] Calling LLM to match celebrity for '{character}'")
        match = match_celebrity(client, model, character, description)
        if not match:
            return None, None

    celebrity = match["celebrity"]
    base_query = match["search_query"]

    if verbose:
        print(f"    [DEBUG] Celebrity matched: {celebrity}")

    # Build diverse search queries
    try:
        desc_obj = json.loads(description) if isinstance(description, str) else description
        style = desc_obj.get("style", "") if isinstance(desc_obj, dict) else ""
        gender = desc_obj.get("gender", "") if isinstance(desc_obj, dict) else ""
    except (json.JSONDecodeError, AttributeError):
        style = ""
        gender = ""

    search_queries = [
        f"{celebrity} {style} dialogue",
        f"{celebrity} emotional scene",
        f"{celebrity} {gender} speech",
    ]

    static_text = DEFAULTS.get("static_voice_text", "")
    all_segments = []

    for vid_idx in range(max_videos):
        query = search_queries[vid_idx % len(search_queries)]
        file_prefix = f"{base_character}_v{vid_idx}"

        if verbose:
            print(f"    [DEBUG] Video {vid_idx+1}/{max_videos}: query='{query}'")

        # Step 1: Download and extract segments
        segment_path, audio_source = find_and_extract_video_segment(
            client=client,
            model=model,
            search_query=query,
            celebrity=celebrity,
            description=description,
            output_dir=output_dir,
            file_prefix=file_prefix,
            max_duration=max_duration,
            whisper_model=whisper_model,
            verbose=verbose,
        )

        if not segment_path:
            if verbose:
                print(f"    [DEBUG] Failed to extract segment for video {vid_idx+1}")
            continue

        # Collect all segments from this video
        video_segments = []
        seg_dir = Path(output_dir)
        for seg_file in sorted(seg_dir.glob(f"{file_prefix}_segment*.wav")):
            if seg_file.exists():
                video_segments.append(str(seg_file))

        if not video_segments:
            video_segments = [segment_path]

        if verbose:
            print(f"    [DEBUG] Found {len(video_segments)} segments from video {vid_idx+1}")

        # Try each segment: validate, generate reference, return on first pass
        for seg_idx, seg_path in enumerate(video_segments):
            # Step 2: Validate segment
            is_valid, reason = validate_celebrity_segment(
                segment_path=seg_path,
                description=description,
                chunkformer_model=chunkformer_model,
                verbose=verbose,
            )

            if not is_valid:
                if verbose:
                    print(f"    [DEBUG] Segment {seg_idx} validation failed: {reason}")
                all_segments.append(seg_path)
                continue

            # Step 3: Generate reference
            if tts_engine:
                ref_path, duration = generate_celebrity_reference(
                    segment_path=seg_path,
                    character=f"{base_character}_v{vid_idx}_s{seg_idx}",
                    output_dir=output_dir,
                    engine=tts_engine,
                    static_text=static_text,
                    verbose=verbose,
                )

                if ref_path:
                    if verbose:
                        print(f"    [DEBUG] Video {vid_idx+1} segment {seg_idx}: reference generated, returning early")
                    metadata = {
                        "character": character,
                        "celebrity": celebrity,
                        "reason": match.get("reason", ""),
                        "search_query": query,
                        "segment": seg_path,
                        "audio_source": audio_source,
                    }
                    return ref_path, metadata
                else:
                    if verbose:
                        print(f"    [DEBUG] Reference generation failed for video {vid_idx+1} segment {seg_idx}")
                    all_segments.append(seg_path)
                    continue
            else:
                # No TTS engine, return segment directly
                if verbose:
                    print(f"    [DEBUG] No TTS engine, returning segment directly")
                metadata = {
                    "character": character,
                    "celebrity": celebrity,
                    "reason": match.get("reason", ""),
                    "search_query": query,
                    "segment": seg_path,
                    "audio_source": audio_source,
                }
                return seg_path, metadata

            all_segments.append(seg_path)

    # All videos failed — return first segment if available
    if all_segments:
        seg_path = all_segments[0]
        audio_src = None
        if tts_engine:
            ref_path, duration = generate_celebrity_reference(
                segment_path=seg_path,
                character=f"{base_character}_fallback",
                output_dir=output_dir,
                engine=tts_engine,
                static_text=static_text,
                verbose=verbose,
            )
            if ref_path:
                metadata = {
                    "character": character,
                    "celebrity": celebrity,
                    "reason": match.get("reason", ""),
                    "search_query": search_queries[0],
                    "segment": seg_path,
                    "audio_source": audio_src,
                }
                return ref_path, metadata

        metadata = {
            "character": character,
            "celebrity": celebrity,
            "reason": match.get("reason", ""),
            "search_query": search_queries[0],
            "segment": seg_path,
            "audio_source": audio_src,
        }
        return seg_path, metadata

    return None, None


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
