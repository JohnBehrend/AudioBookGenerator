#!/usr/bin/env python3
"""
Utility functions shared across multiple modules.
"""

import json
import os
import re
import tempfile
import atexit
from scipy import stats
import numpy as np
import glob
import shutil
import zipfile
import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, Optional, Tuple, List, Union
from openai import OpenAI

from .config import LLM_SETTINGS


def _get_attn_implementation() -> Optional[str]:
    """Return flash_attention_2 if available, otherwise None."""
    try:
        import flash_attn
        return "flash_attention_2"
    except ImportError:
        return None


class ProgressHandler:
    """Unified progress handler for both Gradio and CLI.

    This class provides a single interface for progress tracking that works
    with both Gradio's gr.Progress() and CLI's tqdm/verbose output.

    Args:
        progress: Gradio progress callback or None for CLI mode
        use_tqdm: If True, use tqdm for CLI progress (default: True)
        total: Total iterations for progress calculation
        desc: Default description for progress updates
    """

    def __init__(self, progress=None, use_tqdm: bool = True, total: int = None, desc: str = ""):
        self.progress = progress
        self.use_tqdm = use_tqdm
        self.total = total
        self.desc = desc
        self.last_ratio = 0

        # CLI progress with tqdm
        self._tqdm = None
        if use_tqdm and progress is None:
            try:
                from tqdm import tqdm
                if total:
                    self._tqdm = tqdm(total=total, desc=desc)
                else:
                    self._tqdm = tqdm(desc=desc)
            except ImportError:
                self.use_tqdm = False

    def update(self, ratio: float = None, desc: str = None) -> None:
        """Update progress bar.

        Args:
            ratio: Progress ratio (0.0 to 1.0) or None if not applicable
            desc: Description for this progress update
        """
        # Gradio progress
        if self.progress is not None:
            self.progress(ratio, desc=desc or self.desc)
            return

        # CLI progress
        desc = desc or self.desc

        if self._tqdm is not None:
            # tqdm progress
            if ratio is not None and self.total:
                self._tqdm.n = int(ratio * self.total)
                self._tqdm.refresh()
        elif ratio is not None:
            # Verbose mode fallback - print percentage
            current_pct = int(ratio * 100)
            if current_pct >= self.last_ratio + 10 or ratio == 1.0:
                self.last_ratio = current_pct
                print(f"{desc}: {current_pct}%")

    def set_total(self, total: int) -> None:
        """Update total for tqdm progress bar."""
        if self._tqdm is not None:
            self._tqdm.total = total
            self._tqdm.refresh()

    def close(self) -> None:
        """Close the progress bar."""
        if self._tqdm is not None:
            self._tqdm.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _reset_chapters_dir_internal() -> None:
    """Internal function to reset cached state from get_chapters_dir.

    Clears all cached attributes and cleans up the temp context.
    This is an internal helper used by get_chapters_dir_from_saved and load_temp_dir.
    """
    if hasattr(get_chapters_dir, "_temp_context") and get_chapters_dir._temp_context:
        try:
            get_chapters_dir._temp_context.cleanup()
        except Exception:
            pass
    for attr in ["_temp_dir", "_chapters_dir", "_temp_context"]:
        if hasattr(get_chapters_dir, attr):
            delattr(get_chapters_dir, attr)


def get_chapters_dir(saved_temp_dir: Optional[str] = None) -> Path:
    """Get or create a temporary chapters directory for this session.

    This provides a consistent way for both CLI and Gradio to use temporary
    directories that auto-clean on program exit.

    Args:
        saved_temp_dir: Optional path to a saved temp directory to restore from.
                       If provided, uses this directory instead of creating a new one.

    Returns:
        Path to the chapters directory (temp_dir / "chapters")
    """
    if saved_temp_dir:
        return get_chapters_dir_from_saved(saved_temp_dir)

    if not hasattr(get_chapters_dir, "_temp_dir"):
        # Use TemporaryDirectory which auto-cleans on program exit
        get_chapters_dir._temp_context = tempfile.TemporaryDirectory(prefix="jbab_chapters_")
        get_chapters_dir._temp_dir = get_chapters_dir._temp_context.name
        # Register cleanup on normal exit
        atexit.register(cleanup_temp_dir)
    if not hasattr(get_chapters_dir, "_chapters_dir"):
        get_chapters_dir._chapters_dir = Path(get_chapters_dir._temp_dir) / "chapters"
        get_chapters_dir._chapters_dir.mkdir(parents=True, exist_ok=True)
    return get_chapters_dir._chapters_dir


def get_temp_dir() -> str:
    """Get the temporary directory path for display purposes."""
    if hasattr(get_chapters_dir, "_temp_dir") and get_chapters_dir._temp_dir:
        return get_chapters_dir._temp_dir
    return ""


def cleanup_temp_dir() -> None:
    """Clean up the temporary directory created by get_chapters_dir.

    First copies any MP3 files to ./chapters/ directory to preserve them.
    """
    if hasattr(get_chapters_dir, "_temp_dir") and get_chapters_dir._temp_dir:
        temp_chapters_dir = get_chapters_dir._temp_dir
        # Find and copy any MP3 files from temp directory to ./chapters/
        mp3_files = sorted(glob.glob(os.path.join(temp_chapters_dir, "chapter_*.mp3")), key=natural_sort_key)
        if mp3_files:
            os.makedirs("chapters", exist_ok=True)
            for mp3_path in mp3_files:
                filename = os.path.basename(mp3_path)
                dest_path = os.path.join("chapters", filename)
                shutil.copy2(mp3_path, dest_path)
                print(f"Copied {filename} to chapters/")

    if hasattr(get_chapters_dir, "_temp_context") and get_chapters_dir._temp_context:
        try:
            get_chapters_dir._temp_context.cleanup()
        except Exception:
            pass
        get_chapters_dir._temp_dir = None
        get_chapters_dir._chapters_dir = None
        get_chapters_dir._temp_context = None


def reset_chapters_dir() -> None:
    """Reset the chapters directory cache (for testing).

    This clears all cached state from get_chapters_dir() to allow
    fresh directory creation in tests.
    """
    _reset_chapters_dir_internal()


# ============================================================================
# SAVE/RESUME TEMP FOLDER FUNCTIONALITY
# ============================================================================

def get_saved_audiobooks_dir() -> Path:
    """Get the directory where saved audiobook archives are stored."""
    return Path.home() / ".audiobook_generator" / "saved_audiobooks"


def get_latest_saved_file() -> Path:
    """Get the path to the latest saved audiobook info file."""
    return Path.home() / ".audiobook_generator" / "latest_saved_audiobook.json"


def get_loaded_temp_file() -> Path:
    """Get the path to the loaded temp directory info file."""
    return Path.home() / ".audiobook_generator" / "loaded_temp.json"


def save_temp_dir(temp_dir: str) -> str:
    """Save the temp directory as a zip archive for later recovery.

    Args:
        temp_dir: Path to the temp directory to save

    Returns:
        Path to the saved archive
    """
    temp_path = Path(temp_dir)
    if not temp_path.exists():
        raise ValueError(f"Temp directory does not exist: {temp_dir}")

    # Create saved archives directory
    saved_dir = get_saved_audiobooks_dir()
    saved_dir.mkdir(parents=True, exist_ok=True)

    # Use timestamp in archive name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"audiobook_{timestamp}"
    archive_path = saved_dir / f"{archive_name}.zip"

    print(f"Creating zip archive: {archive_path}")
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for item in temp_path.rglob('*'):
            if item.is_file():
                arcname = item.relative_to(temp_path)
                zip_file.write(item, arcname)

    # Save metadata
    state_file = get_latest_saved_file()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "temp_dir": str(temp_dir),
        "archive": str(archive_path),
        "timestamp": timestamp,
        "name": archive_name,
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print(f"Saved temp dir to archive: {archive_path}")
    print(f"Metadata saved to: {state_file}")
    return str(archive_path)


def load_temp_dir(archive_path: Optional[str] = None) -> Optional[str]:
    """Load from a saved audiobook archive and extract to a new temp directory.

    Args:
        archive_path: Path to the archive to load. If None, loads the latest archive.

    Returns:
        Path to the extracted temp directory, or None if not found
    """
    import tempfile

    if archive_path:
        archive_file = Path(archive_path)
        if not archive_file.exists():
            print(f"Error: Archive not found: {archive_path}")
            return None
    else:
        # Load latest archive
        state_file = get_latest_saved_file()
        if not state_file.exists():
            print("No saved audiobook found.")
            return None
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        archive_file = Path(state.get("archive"))
        if not archive_file.exists():
            print(f"Error: Archive not found: {archive_file}")
            return None

    print(f"Loading archive: {archive_file}")

    # Create a new temp directory for extraction
    temp_context = tempfile.TemporaryDirectory(prefix="jbab_resumed_")
    extract_dir = Path(temp_context.name)

    # Extract the archive
    with zipfile.ZipFile(archive_file, 'r') as zip_file:
        zip_file.extractall(extract_dir)

    chapters_dir = extract_dir / "chapters"
    if not chapters_dir.exists():
        chapters_dir = extract_dir

    print(f"Extracted to: {extract_dir}")
    print(f"Chapters directory: {chapters_dir}")

    # Save the new temp dir path for get_chapters_dir to use
    _reset_chapters_dir_internal()

    # Set up the extracted directory as the temp directory
    get_chapters_dir._temp_dir = str(extract_dir)
    get_chapters_dir._chapters_dir = chapters_dir
    get_chapters_dir._temp_context = temp_context  # Register cleanup on exit

    # Save loaded temp dir path to file for persistence across page refreshes
    loaded_file = get_loaded_temp_file()
    loaded_file.parent.mkdir(parents=True, exist_ok=True)
    with open(loaded_file, "w", encoding="utf-8") as f:
        json.dump({"temp_dir": str(extract_dir)}, f, indent=2)

    # Register cleanup on normal exit
    atexit.register(cleanup_temp_dir)

    return str(extract_dir)


def get_loaded_temp_dir() -> Optional[str]:
    """Get the loaded temp directory path from file (if any).

    Returns:
        Path to the loaded temp directory, or None if not loaded
    """
    loaded_file = get_loaded_temp_file()
    if loaded_file.exists():
        try:
            with open(loaded_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            temp_dir = data.get("temp_dir")
            if temp_dir and Path(temp_dir).exists():
                return temp_dir
        except Exception:
            pass
    return None


def clear_loaded_temp_dir() -> None:
    """Clear the loaded temp directory file."""
    loaded_file = get_loaded_temp_file()
    if loaded_file.exists():
        try:
            loaded_file.unlink()
        except Exception:
            pass


def get_available_saved_audiobooks() -> List[Dict[str, str]]:
    """Get list of all available saved audiobook archives.

    Returns:
        List of dicts with 'name', 'timestamp', 'archive', and 'temp_dir' keys
    """
    saved_dir = get_saved_audiobooks_dir()
    if not saved_dir.exists():
        return []

    # Get the latest saved audiobook metadata (stored in latest_saved_audiobook.json)
    latest_metadata = {}
    state_file = get_latest_saved_file()
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            latest_metadata = json.load(f)

    archives = []
    # Pre-compute mtime to avoid repeated stat calls
    archives_data = [(f, f.stat().st_mtime) for f in saved_dir.glob("*.zip")]
    for archive_file, _ in sorted(archives_data, key=lambda x: x[1], reverse=True):
        # Use metadata from latest_saved_audiobook.json if it matches this archive
        if latest_metadata.get("archive") == str(archive_file):
            meta = latest_metadata
        else:
            # Fall back to archive filename for archives without separate metadata
            meta = {"name": archive_file.stem, "timestamp": archive_file.stat().st_mtime}

        archives.append({
            "name": meta.get("name", archive_file.stem),
            "timestamp": meta.get("timestamp", ""),
            "archive": str(archive_file),
            "temp_dir": meta.get("temp_dir", ""),
        })
    return archives


def cleanup_saved_temp_dir() -> None:
    """Remove the latest saved audiobook state file."""
    try:
        state_file = get_latest_saved_file()
        if state_file.exists():
            state_file.unlink()
    except Exception:
        pass


def get_chapters_dir_from_saved(saved_temp_dir: str) -> Path:
    """Get the chapters directory from a saved temp directory path.

    This points to an existing saved temp directory without creating a new context.

    Args:
        saved_temp_dir: Path to the saved temp directory

    Returns:
        Path to the chapters subdirectory within the saved temp dir
    """
    # Check if we're restoring from a directory that's already set up
    existing_temp = getattr(get_chapters_dir, "_temp_dir", None)
    if existing_temp == saved_temp_dir:
        # Already set up - just return the existing chapters path
        if hasattr(get_chapters_dir, "_chapters_dir"):
            return get_chapters_dir._chapters_dir

    # Set up the saved directory as the temp directory
    saved_path = Path(saved_temp_dir)
    chapters_path = saved_path / "chapters"

    # Create the chapters directory if it doesn't exist
    chapters_path.mkdir(parents=True, exist_ok=True)

    get_chapters_dir._temp_dir = str(saved_path)
    get_chapters_dir._chapters_dir = chapters_path
    get_chapters_dir._temp_context = None  # No context - we're using an existing directory

    return chapters_path


def get_characters_from_map_files(chapters_dir: Path) -> list:
    """Extract unique character names from map.json files.

    Args:
        chapters_dir: Path to the directory containing chapter map files

    Returns:
        Sorted list of unique character names
    """
    characters = set()

    map_files = sorted([f for f in chapters_dir.glob("*.map.json")
                       if re.match(r"^chapter_\d+\.map\.json$", f.name)],
                      key=natural_sort_key)
    for map_file in map_files:
        try:
            with open(map_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Handle both list format [char_map, line_map] and dict format
            if isinstance(data, list) and len(data) >= 1:
                character_map = data[0]
            elif isinstance(data, dict):
                character_map = data.get("character_map", {})
            else:
                continue

            if isinstance(character_map, dict):
                for char_name in character_map.values():
                    if isinstance(char_name, str):
                        characters.add(char_name)
        except Exception:
            pass

    return sorted(list(characters))


def compare_characters(character_name: str, other_character: str) -> bool:
    """Check if two characters are likely the same based on name similarity.

    Uses simple substring matching to detect if one name contains the other
    or they are identical.

    Args:
        character_name: First character name to compare
        other_character: Second character name to compare

    Returns:
        True if the characters are likely the same, False otherwise
    """
    if character_name == other_character:
        return True

    lower1, lower2 = character_name.lower(), other_character.lower()

    # Check if one contains the other as a substring
    if lower1 in lower2 or lower2 in lower1:
        return True

    return False


def merge_line_maps(line_maps: list, verbose: bool = False) -> dict:
    """Take multiple line maps and determine the most common mapping for each line.

    If there is only one value for a line, we will pick that value.
    If there are two values for a line, we'll pick the first.
    If there are more than two values for a line, pick the majority. If all different pick first.

    Args:
        line_maps: List of line maps (each map is dict mapping line number to speaker key)
        verbose: Print verbose output

    Returns:
        Merged line map with most common speaker for each line
    """
    merged_line_map = {}
    if len(line_maps) > 0:
        for line_map in line_maps:
            for line, speaker_num in line_map.items():
                if not (line in merged_line_map.keys()):
                    merged_line_map[line] = [speaker_num]
                else:
                    merged_line_map[line].append(speaker_num)

    if verbose:
        print("Merged Line Map:")
        print(merged_line_map)

    return {k: Counter(v).most_common(1)[0][0] for k, v in merged_line_map.items()}


def get_llm_client(api_key: str, port: str) -> OpenAI:
    """Create and return an OpenAI client for LM Studio.

    Args:
        api_key: API key for the LLM (can be any string for LM Studio)
        port: Port for the LLM inference

    Returns:
        OpenAI client configured for LM Studio
    """
    return OpenAI(base_url=f"http://localhost:{port}/v1", api_key=api_key)


def get_validation_client() -> OpenAI:
    """Create an OpenAI client for voice validation using validation settings.

    Returns:
        OpenAI client configured for validation LLM
    """
    from config import VOICE_VALIDATION
    return OpenAI(base_url=VOICE_VALIDATION["endpoint"], api_key="EMPTY")


# ============================================================================
# FILE OPERATIONS
# ============================================================================


def load_json_file(filepath: str) -> Optional[dict]:
    """Load JSON file if it exists, return None otherwise.

    Args:
        filepath: Path to the JSON file

    Returns:
        Parsed JSON data or None if file doesn't exist
    """
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def copy_mp3_files_to_chapters(source_dir: str) -> int:
    """Copy MP3 files from source_dir to ./chapters/ directory.

    Args:
        source_dir: Source directory containing chapter MP3 files

    Returns:
        Number of files copied
    """
    mp3_files = sorted(glob.glob(os.path.join(source_dir, "chapter_*.mp3")), key=natural_sort_key)

    if not mp3_files:
        return 0

    os.makedirs("chapters", exist_ok=True)

    for mp3_path in mp3_files:
        filename = os.path.basename(mp3_path)
        dest_path = os.path.join("chapters", filename)
        shutil.copy2(mp3_path, dest_path)
        print(f"Copied {filename} to chapters/")

    return len(mp3_files)


def get_character_wav_file(character_name: str, chapters_dir: Path) -> Optional[str]:
    """Get the path to a character's generated WAV file.

    Searches in both chapters_dir and SCRIPT_DIR for the file.

    Args:
        character_name: Name of the character
        chapters_dir: Path to the chapters directory

    Returns:
        Path to the WAV file or None if not found
    """
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent

    for base_dir in [chapters_dir, script_dir]:
        wav_path = base_dir / f"{character_name}.wav"
        if wav_path.exists():
            return str(wav_path)
    return None


def load_seed_characters(seed_voice_map: Optional[Union[str, Dict]]) -> Optional[Dict[str, str]]:
    """Load seed characters from a voices_map.json file.

    Args:
        seed_voice_map: Path to the seed voices_map.json file, or a dict/FileData from Gradio

    Returns:
        Dict mapping character names to voice file paths, or None if file not found
    """
    if not seed_voice_map:
        return None

    # Handle Gradio File component output (can be dict with 'name' key or path string)
    if isinstance(seed_voice_map, dict):
        seed_voice_map = seed_voice_map.get('name')

    if not seed_voice_map or not os.path.exists(seed_voice_map):
        return None

    try:
        with open(seed_voice_map, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def normalize_character_name(name: str) -> str:
    """Normalize a character name for comparison.

    Args:
        name: Character name to normalize

    Returns:
        Normalized name (lowercase, spaces instead of underscores/apostrophes)
    """
    return name.lower().strip().replace("_", " ").replace("'", " ")


def distill_string(input_str: str) -> str:
    """Remove punctuation and convert to lowercase for string comparison.

    Args:
        input_str: Input string to distill

    Returns:
        Lowercase string with punctuation removed (?, ., -, ;, ,, !) and normalized whitespace
    """
    import re
    return re.sub(r'\s+', ' ', (input_str.lower()
            .replace("?", "")
            .replace(".", "")
            .replace("-", "")
            .replace(";", "")
            .replace(",", "")
            .replace("!", ""))).strip()


def transcribe_audio_with_whisper(validation_model, audio_path: str) -> Tuple[str, list, list]:
    """Transcribe audio using Whisper with word-level timestamps.

    This function reuses the transcription logic from audiobook_generator.py
    to avoid code duplication.

    Args:
        validation_model: The WhisperModel instance
        audio_path: Path to the audio file (.wav)

    Returns:
        Tuple of (detected_string, start_times, end_times)
        - detected_string: The transcribed text (distilled)
        - start_times: List of start times for each word
        - end_times: List of end times for each word
    """
    segments_list, info = validation_model.transcribe(audio_path, beam_size=5, word_timestamps=True)

    # Collect segments (words) and timestamps
    segments = []
    start_times = []
    end_times = []
    for segment in segments_list:
        for word in segment.words:
            segments.append(distill_string(word.word.strip()))
            start_times.append(word.start)
            end_times.append(word.end)

    detected_string = distill_string(" ".join(segments))
    return detected_string, start_times, end_times


def validate_audio_clean(audio_path: str, client: Optional[OpenAI] = None, verbose: bool = False) -> Tuple[bool, str]:
    """Validate that audio contains only clean speech without music or background effects.

    Uses the LLM at port 8081 to analyze the audio and determine if it contains
    only clean speech or if there are unwanted sounds like music, sound effects,
    or background noise.

    Args:
        audio_path: Path to the audio file (.wav)
        client: OpenAI client for the validation LLM (created if None)
        verbose: Print verbose output

    Returns:
        Tuple of (is_clean, validation_message)
        - is_clean: True if audio contains only clean speech
        - validation_message: Description of what was detected
    """
    from config import VOICE_VALIDATION

    model = VOICE_VALIDATION.get("model", "default-model")

    # Convert to absolute path for file:// URL
    abs_audio_path = os.path.abspath(audio_path)
    file_url = f"file://{abs_audio_path}"

    # Prompt to check for clean speech
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

        # Parse JSON response
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


# ============================================================================
# MAP FILE PARSING
# ============================================================================


def parse_map_file(map_file: Path) -> Optional[Tuple[Dict[int, str], Dict[int, int]]]:
    """Parse a chapter map JSON file.

    Handles both list format [char_map, line_map] and dict format.

    Args:
        map_file: Path to the map file

    Returns:
        Tuple of (character_map, line_map) with integer keys, or None if parsing fails
    """
    try:
        with open(map_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) >= 2:
            char_map = data[0]
            line_map = data[1]
        elif isinstance(data, dict):
            char_map = data.get("character_map", {})
            line_map = data.get("line_map", {})
        else:
            return None

        char_map = {int(k): v for k, v in char_map.items()}
        line_map = {int(k): v for k, v in line_map.items()}

        return char_map, line_map
    except Exception:
        return None


def get_chapter_map_files(chapters_dir: Path) -> list:
    """Get sorted list of chapter map files.

    Args:
        chapters_dir: Path to the chapters directory

    Returns:
        Sorted list of map file paths (naturally sorted)
    """
    return sorted([f for f in chapters_dir.glob("*.map.json")
                   if re.match(r"^chapter_\d+\.map\.json$", f.name)],
                  key=natural_sort_key)


def extract_characters_from_maps(chapters_dir: Path) -> list:
    """Extract unique character names from all map files in a directory.

    Args:
        chapters_dir: Path to the chapters directory

    Returns:
        Sorted list of unique character names
    """
    characters = set()
    map_files = get_chapter_map_files(chapters_dir)

    for map_file in map_files:
        result = parse_map_file(map_file)
        if result:
            char_map, _ = result
            for char_name in char_map.values():
                if isinstance(char_name, str):
                    characters.add(char_name)

    return sorted(list(characters))


def count_lines_per_character(chapters_dir: Path) -> Dict[str, int]:
    """Count lines spoken per character from map.json files.

    Args:
        chapters_dir: Path to the chapters directory

    Returns:
        Dict mapping character name to line count (including narrator for unlabeled lines)
    """
    character_lines = {}

    for map_file in get_chapter_map_files(chapters_dir):
        result = parse_map_file(map_file)
        if not result:
            continue

        char_map, line_map = result
        char_by_id = char_map

        # Count labeled lines for each character
        labeled_count = 0
        for line_num, char_num in line_map.items():
            char_name = char_by_id.get(char_num)
            if char_name:
                character_lines[char_name] = character_lines.get(char_name, 0) + 1
                labeled_count += 1

        # Count total spoken lines from the corresponding txt file
        txt_file = map_file.parent / map_file.name.replace(".map.json", ".txt")
        spoken_lines = set()
        if txt_file.exists():
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Line ") and ": " in line:
                        try:
                            line_num_str = line.split(": ", 1)[0].replace("Line ", "")
                            line_num = int(line_num_str)
                            content = line.split(": ", 1)[1] if ": " in line else ""
                            if content.strip():
                                spoken_lines.add(line_num)
                        except (ValueError, IndexError):
                            pass

        # Add unlabeled spoken lines to narrator
        narrator_lines = len(spoken_lines) - labeled_count
        if narrator_lines > 0:
            character_lines["narrator"] = character_lines.get("narrator", 0) + narrator_lines

    return character_lines


def natural_sort_key(filename: str):
    """Generate a sort key for natural (human) sorting of filenames.

    This ensures that chapter_10.txt comes after chapter_2.txt (not before).
    Used for sorting chapter files, map files, etc.

    Args:
        filename: The filename to generate a sort key for

    Returns:
        A tuple of (prefix, number, suffix) where number is an integer,
        allowing proper numerical sorting of filenames like chapter_1.txt,
        chapter_2.txt, chapter_10.txt, etc.
    """
    # Match pattern like "chapter_123.txt" -> ("chapter_", 123, ".txt")
    match = re.match(r"^(.*?)(\d+)(.*?)$", os.path.basename(filename))
    if match:
        prefix, num, suffix = match.groups()
        return (prefix, int(num), suffix)
    # If no number found, just return the filename
    return (filename, 0, "")


# ============================================================================
# VOICE GENDER CORRECTION
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
    sample_mean = np.mean(voiced_f0)
    sample_std = np.std(voiced_f0, ddof=1) if len(voiced_f0) > 1 else 0
    sample_size = len(voiced_f0)

    if verbose:
        print(f"    Pitch distribution: n={sample_size}, mean={sample_mean:.1f}Hz, std={sample_std:.1f}Hz")
        print(f"    Reference: male={male_ref_mean}Hz, female={female_ref_mean}Hz")

    # Perform one-sample t-tests against each reference
    t_stat_male, p_value_male = stats.ttest_1samp(voiced_f0, male_ref_mean)
    t_stat_female, p_value_female = stats.ttest_1samp(voiced_f0, female_ref_mean)

    if verbose:
        print(f"    T-test vs male ref: t={t_stat_male:.3f}, p={p_value_male:.4f}")
        print(f"    T-test vs female ref: t={t_stat_female:.3f}, p={p_value_female:.4f}")

    # Classification logic:
    # - If p_male > alpha (consistent with male) AND p_female < alpha (not female) -> male
    # - If p_female > alpha (consistent with female) AND p_male < alpha (not male) -> female
    # - Otherwise -> use distance-based classification

    is_male = p_value_male > alpha
    is_female = p_value_female > alpha

    if is_male and not is_female:
        confidence = min(1.0, p_value_male)
        return "male", confidence, f"Statistically consistent with male distribution (p={p_value_male:.3f})"

    if is_female and not is_male:
        confidence = min(1.0, p_value_female)
        return "female", confidence, f"Statistically consistent with female distribution (p={p_value_female:.3f})"

    # Both tests pass or both fail - use distance-based classification
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

        # Plot histogram of detected pitches
        ax.hist(voiced_f0, bins=30, alpha=0.7, color='steelblue', edgecolor='black',
                label=f'Detected pitches (n={len(voiced_f0)}, mean={np.mean(voiced_f0):.1f}Hz)')

        # Create x-axis range for reference distributions
        x = np.linspace(60, 280, 500)

        # Plot male reference distribution
        male_pdf = scipy_stats.norm.pdf(x, male_ref_mean, ref_std)
        ax.plot(x, male_pdf * len(voiced_f0) * 0.3, 'r--', linewidth=2,
                label=f'Male ref (μ={male_ref_mean}Hz)')
        ax.axvline(male_ref_mean, color='red', linestyle='--', alpha=0.5, linewidth=1)

        # Plot female reference distribution
        female_pdf = scipy_stats.norm.pdf(x, female_ref_mean, ref_std)
        ax.plot(x, female_pdf * len(voiced_f0) * 0.3, 'm--', linewidth=2,
                label=f'Female ref (μ={female_ref_mean}Hz)')
        ax.axvline(female_ref_mean, color='magenta', linestyle='--', alpha=0.5, linewidth=1)

        # Mark sample mean
        sample_mean = np.mean(voiced_f0)
        ax.axvline(sample_mean, color='green', linestyle='-', linewidth=2,
                   label=f'Sample mean ({sample_mean:.1f}Hz)')

        # Threshold line
        threshold = (male_ref_mean + female_ref_mean) / 2
        ax.axvline(threshold, color='gray', linestyle=':', linewidth=1.5,
                   label=f'Threshold ({threshold:.1f}Hz)')

        # Labels and title
        ax.set_xlabel('Pitch (Hz)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        title = f'Pitch Distribution - Detected: {detected_gender.upper()}'
        if confidence is not None:
            title += f' (confidence: {confidence:.2f})'
        ax.set_title(title, fontsize=14)

        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(60, 280)
        ax.grid(True, alpha=0.3)

        # Add reason text if provided
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

    By default, uses statistical t-test for robust gender classification against
    reference distributions. Falls back to simple threshold comparison if t-test
    is disabled or sample size is too small.

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
        import librosa
        from psola import vocode
        import numpy as np
        from pathlib import Path

        # Extract target gender from description
        target_gender = extract_gender_from_description(description)
        if target_gender is None:
            return False, "Could not extract target gender from description", None, None, None

        # Load audio and estimate pitch using shared helper
        y, sr = librosa.load(audio_path, sr=None)
        f0, voiced_f0 = extract_pitch_from_audio(audio_path)

        if len(voiced_f0) == 0:
            return False, "Could not detect pitch in audio", None, None, None

        current_avg_pitch = np.mean(voiced_f0)

        # Determine current gender using statistical method or threshold
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

        # Generate histogram if requested
        if plot_histogram:
            hist_dir = histogram_dir if histogram_dir else str(Path(audio_path).parent)
            hist_path = str(Path(hist_dir) / f"{Path(audio_path).stem}_pitch_histogram.png")
            plot_pitch_histogram(
                voiced_f0, current_gender, hist_path,
                male_ref_mean, female_ref_mean, 30.0, confidence, reason if use_ttest else None
            )
            if verbose:
                print(f"    Saved histogram to: {hist_path}")

        # If genders match, no correction needed
        if current_gender == target_gender:
            return True, f"Gender already correct ({current_gender})", current_gender, current_avg_pitch, confidence

        # Determine target pitch based on desired gender
        target_pitch = female_target_pitch_hz if target_gender == "female" else male_target_pitch_hz

        # Calculate shift ratio to reach target pitch
        shift_ratio = target_pitch / current_avg_pitch

        # Check if shift is beyond reasonable bounds (PSOLA works best within ~0.5-2.0x)
        # If beyond bounds, return failure so the voice will be regenerated instead
        if shift_ratio < 0.5 or shift_ratio > 2.0:
            return False, f"Extreme pitch ({current_avg_pitch:.1f}Hz) beyond correction range - needs regeneration", current_gender, current_avg_pitch, confidence

        # Calculate percentage for reporting
        shift_percent = (shift_ratio - 1.0) * 100

        if verbose:
            print(f"    Applying pitch shift: {shift_percent:+.1f}% ({current_gender} → {target_gender})")
            print(f"    Current: {current_avg_pitch:.1f}Hz → Target: {target_pitch:.1f}Hz (ratio: {shift_ratio:.2f}x)")

        # Apply pitch shift using PSOLA vocode
        target_f0 = f0 * shift_ratio
        y_shifted = vocode(y, sr, target_pitch=target_f0)

        # Save corrected audio back to the same file
        import soundfile as sf
        sf.write(audio_path, y_shifted, sr)

        # After pitch shift, final gender should match target (that's the point of correction)
        final_pitch = current_avg_pitch * shift_ratio
        final_gender = target_gender
        final_confidence = confidence  # Confidence remains similar after deterministic shift

        return True, f"Applied pitch shift of {shift_percent:+.1f}% ({current_gender} → {target_gender})", final_gender, final_pitch, final_confidence

    except ImportError as e:
        return False, f"Missing dependency: {e}. Install librosa and psola.", None, None, None
    except Exception as e:
        return False, f"Gender correction error: {e}", None, None, None
