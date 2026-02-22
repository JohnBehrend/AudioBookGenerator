"""
Audiobook Generator Package

A LLM-based EPUB to audiobook generator with character voice synthesis.

Core Pipeline Stages:
    Stage 1: parse_chapter.parse_epub_to_chapters() - EPUB -> ChapterObj list
    Stage 2: llm_label_speakers.label_speakers() - Text -> character_map, line_map
    Stage 3: llm_describe_character.describe_characters() - Characters -> Descriptions
    Stage 4: generate_voice_samples.generate_voice_samples() - Descriptions -> Voice samples
    Stage 5: audiobook_generator.generate_audiobook_from_chapters() - All -> Audiobook MP3s
"""

from .config import DEFAULTS, LLM_SETTINGS, AUDIO_SETTINGS, VOICE_SAMPLES_DIR
from .llm_label_speakers import label_speakers
from .llm_describe_character import describe_characters
from .generate_voice_samples import generate_voice_samples
from .audiobook_generator import (
    PipelineState,
    run_full_pipeline,
    generate_audiobook_from_chapters,
    VoiceMapper,
    get_non_silent_audio_from_wavs,
    main,
)
from .gradio_ui import (
    create_interface,
    get_pipeline_state,
    get_characters_from_map_files,
    update_character_table,
    update_button_visibility,
    cleanup_temp_dir,
)

__all__ = [
    "DEFAULTS",
    "LLM_SETTINGS",
    "AUDIO_SETTINGS",
    "VOICE_SAMPLES_DIR",
    # Stage functions (public interfaces)
    "label_speakers",
    "describe_characters",
    "generate_voice_samples",
    "generate_audiobook_from_chapters",
    "PipelineState",
    "run_full_pipeline",
    "VoiceMapper",
    "get_non_silent_audio_from_wavs",
    "main",
    "create_interface",
    "get_pipeline_state",
    "get_characters_from_map_files",
    "update_character_table",
    "update_button_visibility",
    "cleanup_temp_dir",
]
