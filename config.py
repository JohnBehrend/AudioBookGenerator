"""Centralized configuration for the AudioBook Generator.

This module provides a single source of truth for all configuration settings,
including LLM settings, device selection, defaults, and file paths.

Usage:
    from config import LLM_SETTINGS, AUDIO_SETTINGS, DEFAULTS

    # Access settings
    port = LLM_SETTINGS["port"]
    device = AUDIO_SETTINGS["default_device"]
"""

import os
from pathlib import Path

# ============================================================================
# LLM SETTINGS
# ============================================================================

LLM_SETTINGS = {
    "endpoint": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "default_port": "1234",
    "default_api_key": "lm-studio",
    "default_model": "local-model",
}

# Environment variable overrides
if os.environ.get("LLM_ENDPOINT"):
    LLM_SETTINGS["endpoint"] = os.environ["LLM_ENDPOINT"]
if os.environ.get("LLM_API_KEY"):
    LLM_SETTINGS["api_key"] = os.environ["LLM_API_KEY"]


# ============================================================================
# AUDIO SETTINGS
# ============================================================================

AUDIO_SETTINGS = {
    "default_device": "cuda:0",
    "alt_device": "cuda:1",
    "default_tts_engine": "kugelaudio",
    "voice_samples_dir": "character_voice_samples",
    "supported_audio_extensions": [".wav", ".mp3", ".flac"],
}

# Environment variable overrides
if os.environ.get("TTS_ENGINE"):
    AUDIO_SETTINGS["default_tts_engine"] = os.environ["TTS_ENGINE"]
if os.environ.get("AUDIO_DEVICE"):
    AUDIO_SETTINGS["default_device"] = os.environ["AUDIO_DEVICE"]


# ============================================================================
# DEFAULT VALUES
# ============================================================================

DEFAULTS = {
    "num_llm_attempts": 10,
    "max_chapters": 10,
    "max_new_tokens": 512,
    "sample_text_length": 150,
    "description_length": 400,
}


# ============================================================================
# FILE PATHS
# ============================================================================

# Output directories
OUTPUT_DIR = Path("chapters")
VOICE_SAMPLES_DIR = Path("character_voice_samples")

# File patterns
CHAPTER_PATTERN = "chapter_*.txt"
MAP_PATTERN = "*.map.json"
WAV_PATTERN = "*.wav"
MP3_PATTERN = "*.mp3"

# JSON files
CHARACTERS_FILE = "characters.json"
CHARACTER_DESCRIPTIONS_FILE = "characters_descriptions.json"


# ============================================================================
# VALIDATION
# ============================================================================

def validate() -> list[str]:
    """Validate configuration and return list of warnings."""
    warnings = []

    # Check if voice samples directory exists
    if not VOICE_SAMPLES_DIR.exists():
        warnings.append(f"Voice samples directory not found: {VOICE_SAMPLES_DIR}")

    # Check TTS engine is valid
    if AUDIO_SETTINGS["default_tts_engine"] not in ["kugelaudio", "vibevoice"]:
        warnings.append(
            f"Unknown TTS engine: {AUDIO_SETTINGS['default_tts_engine']}. "
            "Using 'kugelaudio' or 'vibevoice' is recommended."
        )

    return warnings


def print_config() -> None:
    """Print current configuration for debugging."""
    print("=" * 60)
    print("AUDIOBOOK GENERATOR CONFIGURATION")
    print("=" * 60)
    print("\nLLM SETTINGS:")
    for key, value in LLM_SETTINGS.items():
        print(f"  {key}: {value}")
    print("\nAUDIO SETTINGS:")
    for key, value in AUDIO_SETTINGS.items():
        print(f"  {key}: {value}")
    print("\nDEFAULTS:")
    for key, value in DEFAULTS.items():
        print(f"  {key}: {value}")
    print("\nFILE PATHS:")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  VOICE_SAMPLES_DIR: {VOICE_SAMPLES_DIR}")
    print("=" * 60)
