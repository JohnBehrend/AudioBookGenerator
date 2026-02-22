"""
Audiobook Generator Package

A LLM-based EPUB to audiobook generator with character voice synthesis.
"""

from .config import DEFAULTS, LLM_SETTINGS, AUDIO_SETTINGS, VOICE_SAMPLES_DIR
from .llm_label_speakers import label_speakers_in_file
from .llm_describe_character import describe_characters_in_dir
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
    "label_speakers_in_file",
    "describe_characters_in_dir",
    "generate_voice_samples",
    "PipelineState",
    "run_full_pipeline",
    "generate_audiobook_from_chapters",
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
