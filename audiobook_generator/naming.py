"""Canonical chapter file naming, shared across the pipeline.

Chapter artifacts are named ``chapter_<n><ext>`` (e.g. ``chapter_0.txt``,
``chapter_0.map.json``, ``chapter_0.mp3``). These helpers keep the glob and
regex in one place so every stage agrees on what counts as a chapter file.
"""

import re
from pathlib import Path
from typing import List

# Bare ``chapter_<n>`` stem (e.g. used to skip chapter artifacts among voice files).
CHAPTER_STEM_RE = re.compile(r"^chapter_(\d+)$", re.IGNORECASE)

_NAME_RES: dict = {}


def chapter_name_re(ext: str) -> "re.Pattern[str]":
    """Regex matching canonical ``chapter_<n><ext>`` file names (anchored)."""
    pattern = _NAME_RES.get(ext)
    if pattern is None:
        pattern = re.compile(r"^chapter_\d+" + re.escape(ext) + r"$", re.IGNORECASE)
        _NAME_RES[ext] = pattern
    return pattern


def is_chapter_name(name: str, ext: str = ".txt") -> bool:
    """Return True if ``name`` is a canonical ``chapter_<n><ext>`` file name."""
    return bool(chapter_name_re(ext).match(name))


def sorted_chapter_files(chapters_dir, ext: str = ".txt") -> List[Path]:
    """All canonical ``chapter_<n><ext>`` files in a directory, naturally sorted.

    Args:
        chapters_dir: Directory holding the chapter files (Path or str)
        ext: File-name suffix (e.g. ".txt", ".map.json", ".mp3")

    Returns:
        Naturally sorted list of matching paths
    """
    from .utils import natural_sort_key

    pattern = chapter_name_re(ext)
    files = [
        f for f in Path(chapters_dir).glob(f"chapter_*{ext}")
        if pattern.match(f.name)
    ]
    return sorted(files, key=natural_sort_key)
