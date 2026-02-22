"""
Audiobook Generator Package

A LLM-based EPUB to audiobook generator with character voice synthesis.
"""

from .config import DEFAULTS, LLM_SETTINGS, AUDIO_SETTINGS, VOICE_SAMPLES_DIR
from .parse_chapter import parse_epub_to_chapters, write_chapters_to_txt
from .llm_label_speakers import label_speakers_in_file
from .llm_describe_character import describe_characters_in_dir
from .generate_voice_samples import generate_voice_samples
from .audiobook_generator import (
    PipelineState,
    run_full_pipeline,
    generate_audiobook_from_chapters,
)

__all__ = [
    "DEFAULTS",
    "LLM_SETTINGS",
    "AUDIO_SETTINGS",
    "VOICE_SAMPLES_DIR",
    "parse_epub_to_chapters",
    "write_chapters_to_txt",
    "label_speakers_in_file",
    "describe_characters_in_dir",
    "generate_voice_samples",
    "PipelineState",
    "run_full_pipeline",
    "generate_audiobook_from_chapters",
]
