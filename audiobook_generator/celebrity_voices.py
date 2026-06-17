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


# Module-level cache for celebrity audio downloads to avoid duplicate YouTube requests
_celebrity_audio_cache: Dict[str, str] = {}


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

            # Trim to max_duration if needed
            trim_audio(str(output_file), max_duration)

            # Cache the result
            _celebrity_audio_cache[cache_key] = str(output_file)
            print(f"    [DEBUG] Cached celebrity audio: {cache_key} -> {output_file}")

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


CELEBRITY_SPEECH_SEGMENT_PROMPT = """You are analyzing a transcribed audio clip of a celebrity. The audio may contain multiple speakers (e.g., an interview with a host and guest).

Given:
- Celebrity name: {celebrity}
- Character description: {description}
- Transcription with timestamps: {transcription}

Identify which portions of the audio are most likely spoken by the celebrity {celebrity}. The celebrity's speech should match the voice qualities described in the character description.

Output ONLY a JSON array of objects, each with:
- "start": start time in seconds (float)
- "end": end time in seconds (float)
- "text": the transcribed text for this segment

Select up to 3 best segments that:
1. Are most likely spoken by the celebrity (not the host/interviewer)
2. Have clear speech (no music, crowd noise, or overlapping voices)
3. Are at least 2 seconds long
4. Best represent the voice quality described

Example output:
[
  {{"start": 5.2, "end": 12.8, "text": "I think the key is to stay focused and keep pushing forward"}},
  {{"start": 18.0, "end": 25.5, "text": "Every challenge is an opportunity to grow"}}
]
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
        whisper_model: WhisperModel instance
        max_segments: Maximum number of segments to return
        verbose: Print debug output

    Returns:
        List of dicts with 'start', 'end', 'text' keys, sorted by relevance
    """
    try:
        from .utils import transcribe_audio_with_whisper

        if verbose:
            print(f"      [DEBUG] Transcribing celebrity audio with Whisper...")

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

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Analyze the transcription and identify which segments are spoken by the celebrity."},
            ],
        )

        raw = response.choices[0].message.content
        if not raw:
            return []

        # Parse JSON array from response
        start_idx = raw.find("[")
        end_idx = raw.rfind("]") + 1
        if start_idx >= 0 and end_idx > start_idx:
            segments = json.loads(raw[start_idx:end_idx])
            # Validate and filter
            valid_segments = []
            for seg in segments:
                if all(k in seg for k in ['start', 'end', 'text']):
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


def build_celebrity_voice(
    client: Any,
    model: str,
    character: str,
    description: str,
    output_dir: str,
    max_duration: int = 30,
    pre_matched_celebrity: Optional[str] = None,
    num_samples: int = 3,
    segments_per_sample: int = 3,
    whisper_model: Any = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Full pipeline: match celebrity, download audio, extract segments.

    Args:
        client: OpenAI client instance
        model: Model name
        character: Character name
        description: Character voice description
        output_dir: Directory to save voice files
        max_duration: Max duration for downloaded clip
        pre_matched_celebrity: Optional pre-matched celebrity name (skips LLM matching)
        num_samples: Number of different audio samples to download
        segments_per_sample: Number of segments to extract from each sample
        whisper_model: Optional WhisperModel for celebrity speech identification

    Returns:
        Tuple of (best_voice_path, metadata) or (None, None) on failure
    """
    # Strip .sampleN suffix from character name for search queries and file names
    base_character = re.sub(r'\.sample\d+$', '', character)

    # Match celebrity - use pre-matched if available
    if pre_matched_celebrity:
        print(f"    [DEBUG] Using pre-matched celebrity: {pre_matched_celebrity}")
        # Generate a character-specific search query to avoid downloading the same video
        character_key = base_character.replace(" ", "_").lower()
        match = {
            "celebrity": pre_matched_celebrity,
            "reason": "Pre-matched from character description",
            "search_query": f"{pre_matched_celebrity} {character_key} voice",
        }
    else:
        print(f"    [DEBUG] No pre-matched celebrity - calling LLM to match for '{character}'")
        match = match_celebrity(client, model, character, description)
        if not match:
            return None, None

    celebrity = match["celebrity"]
    search_query = match["search_query"]
    print(f"    [DEBUG] Celebrity matched: {celebrity}")
    print(f"    [DEBUG] Search query: {search_query}")

    # Generate alternative search queries for diversity
    search_queries = [search_query]
    character_key = base_character.replace(" ", "_").lower()
    search_queries.append(f"{celebrity} interview")
    search_queries.append(f"{celebrity} speech")

    # Download audio samples and extract segments
    all_segments = []
    best_segment = None
    audio_sources = []
    
    for sample_idx in range(num_samples):
        query = search_queries[sample_idx % len(search_queries)]
        file_prefix = f"{base_character}_{sample_idx}"
        
        print(f"    [DEBUG] Sample {sample_idx + 1}/{num_samples}: Downloading with query '{query}'")
        
        audio_path = download_celebrity_audio(
            search_query=query,
            output_dir=output_dir,
            max_duration=max_duration,
            file_prefix=file_prefix,
        )
        
        if not audio_path:
            print(f"    [DEBUG] Failed to download audio for sample {sample_idx + 1}")
            continue
        
        audio_sources.append(audio_path)
        
        # Use Whisper+LLM to identify celebrity speech segments if whisper_model is available
        if whisper_model:
            print(f"    [DEBUG] Using Whisper+LLM to identify celebrity speech segments...")
            llm_segments = identify_celebrity_segments(
                client=client,
                model=model,
                celebrity=celebrity,
                description=description,
                audio_path=audio_path,
                whisper_model=whisper_model,
                max_segments=segments_per_sample,
                verbose=True,
            )
            
            if llm_segments:
                print(f"    [DEBUG] LLM identified {len(llm_segments)} celebrity speech segments")
                # Extract LLM-identified segments using ffmpeg
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                
                for seg_idx, seg in enumerate(llm_segments):
                    seg_path = out_path / f"{file_prefix}_llm_segment_{seg_idx}.wav"
                    start = seg['start']
                    end = seg['end']
                    
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", audio_path, "-ss", str(start),
                             "-to", str(end), "-q:a", "0", str(seg_path)],
                            capture_output=True, timeout=30,
                        )
                        if seg_path.exists():
                            all_segments.append(str(seg_path))
                            if best_segment is None or Path(seg_path).stat().st_size > Path(best_segment).stat().st_size:
                                best_segment = str(seg_path)
                    except Exception as e:
                        print(f"      [DEBUG] Error extracting LLM segment: {e}")
            else:
                print(f"    [DEBUG] LLM failed to identify celebrity speech segments, falling back to silence detection")
        
        # Also extract speech segments using silence detection (for fallback)
        segments = extract_speech_segments(
            audio_path=audio_path,
            output_dir=output_dir,
            file_prefix=file_prefix,
        )
        
        if not segments:
            print(f"    [DEBUG] No segments extracted for sample {sample_idx + 1}")
            continue
        
        print(f"    [DEBUG] Extracted {len(segments)} segments from sample {sample_idx + 1}")
        
        # Add to all segments
        all_segments.extend(segments)
        
        # Update best segment (longest by file size)
        for seg in segments:
            if best_segment is None or Path(seg).stat().st_size > Path(best_segment).stat().st_size:
                best_segment = seg
    
    if not all_segments:
        return None, None

    print(f"    [DEBUG] Total segments collected: {len(all_segments)}")

    metadata = {
        "character": character,
        "celebrity": celebrity,
        "reason": match.get("reason", ""),
        "search_query": search_query,
        "audio_sources": audio_sources,
        "segments": all_segments,
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
