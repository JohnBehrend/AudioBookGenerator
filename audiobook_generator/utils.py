#!/usr/bin/env python3
"""
Utility functions shared across multiple modules.
"""

import json
import os
import re
import tempfile
import atexit
import glob
import shutil
import zipfile
import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, Optional, Tuple, List, Union
from openai import OpenAI

from config import LLM_SETTINGS


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
        mp3_files = sorted(glob.glob(os.path.join(temp_chapters_dir, "chapter_*.mp3")))
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
    return Path.home() / ".claude" / "saved_audiobooks"


def get_latest_saved_file() -> Path:
    """Get the path to the latest saved audiobook info file."""
    return Path.home() / ".claude" / "latest_saved_audiobook.json"


def get_loaded_temp_file() -> Path:
    """Get the path to the loaded temp directory info file."""
    return Path.home() / ".claude" / "loaded_temp.json"


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
                       if re.match(r"^chapter_\d+\.map\.json$", f.name)])
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
    mp3_files = sorted(glob.glob(os.path.join(source_dir, "chapter_*.mp3")))

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
        Lowercase string with punctuation removed (?, ., -, ;, ,, !)
    """
    return (input_str.lower()
            .replace("?", "")
            .replace(".", "")
            .replace("-", "")
            .replace(";", "")
            .replace(",", "")
            .replace("!", ""))


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
        Sorted list of map file paths
    """
    return sorted([f for f in chapters_dir.glob("*.map.json")
                   if re.match(r"^chapter_\d+\.map\.json$", f.name)])


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
        for char_num in line_map.values():
            char_name = char_by_id.get(char_num)
            if char_name:
                character_lines[char_name] = character_lines.get(char_name, 0) + 1

        # Count total spoken lines from the corresponding txt file
        txt_file = map_file.with_suffix(".txt")
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
        labeled_count = len(line_map)
        narrator_lines = len(spoken_lines) - labeled_count
        if narrator_lines > 0:
            character_lines["narrator"] = character_lines.get("narrator", 0) + narrator_lines

    return character_lines
