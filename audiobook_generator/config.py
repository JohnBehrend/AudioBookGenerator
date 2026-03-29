"""Centralized configuration for the AudioBook Generator.

This module provides a single source of truth for all configuration settings,
including LLM settings, device selection, defaults, and file paths.

Usage:
    from config import LLM_SETTINGS, AUDIO_SETTINGS, DEFAULTS, DEFAULT_EPUB_FILE

    # Access settings
    port = LLM_SETTINGS["port"]
    device = AUDIO_SETTINGS["default_device"]
    default_epub = DEFAULT_EPUB_FILE
"""

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================================
# LLM SETTINGS
# ============================================================================

LLM_SETTINGS = {
    "endpoint": "http://localhost:8080/v1",
    "port": 8080,
    "api_key": "lm-studio",
    "default_model": "qwen3-model",
}

# ============================================================================
# AUDIO SETTINGS
# ============================================================================

AUDIO_SETTINGS = {
    "default_device": "cuda:0",
    "alt_device": "cuda:1",
    "default_tts_engine": "kugelaudio",
    "supported_audio_extensions": [".wav", ".mp3", ".flac"],
    "gradio_port": 7860,
}

# TTS Model paths for each engine
TTS_MODEL_PATHS = {
    "kugelaudio": {
        "base": "kugelaudio/kugelaudio-0-open",
        "turbo": "kugel-1-turbo",
    },
    "vibevoice": "Jmica/VibeVoice7B",
    "moss": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
}

# Voice samples directory
VOICE_SAMPLES_DIR = Path("character_voice_samples")

# Environment variable overrides for LLM settings
if os.environ.get("LLM_ENDPOINT"):
    LLM_SETTINGS["endpoint"] = os.environ["LLM_ENDPOINT"]
if os.environ.get("LLM_API_KEY"):
    LLM_SETTINGS["api_key"] = os.environ["LLM_API_KEY"]
if os.environ.get("LLM_PORT"):
    LLM_SETTINGS["port"] = int(os.environ["LLM_PORT"])
if os.environ.get("LLM_MODEL"):
    LLM_SETTINGS["default_model"] = os.environ["LLM_MODEL"]

# Environment variable overrides for AUDIO settings

if os.environ.get("TTS_ENGINE"):
    AUDIO_SETTINGS["default_tts_engine"] = os.environ["TTS_ENGINE"]
if os.environ.get("AUDIO_DEVICE"):
    AUDIO_SETTINGS["default_device"] = os.environ["AUDIO_DEVICE"]
if os.environ.get("GRADIO_PORT"):
    AUDIO_SETTINGS["gradio_port"] = int(os.environ["GRADIO_PORT"])


# ============================================================================
# DEFAULT VALUES
# ============================================================================

DEFAULTS = {
    "num_llm_attempts": 1,
    "max_chapters": 1,
    "max_new_tokens": 384,
    "sample_text_length": 150,
    "description_length": 400,
    # Audio generation defaults
    "cfg_scale": 1.30,
    "short_text_postfix": "and also with you?",
    # Static text for voice generation - 10 seconds, high emotional range
    "static_voice_text": "I cannot believe it! After all these years, it's finally here for us to bask in the warm glow of the future.",
    "validation_model_name": "large-v2",
    "min_silence_len": 1250,
    "silence_thresh": -60,
    # Reference text for TTS voice cloning
    # MOSS-VoiceGenerator hyperparameters (per model card recommendations) for Stage 1 voice design
    "moss_voicegen_temperature": 1.5,
    "moss_voicegen_top_p": 0.6,
    "moss_voicegen_top_k": 50,
    "moss_voicegen_repetition_penalty": 1.1,
    # MOSS-TTS hyperparameters (per model card recommendations) for Stage 2 cloning
    "moss_audio_temperature": 1.0,
    "moss_audio_top_p": 0.95,
    "moss_audio_top_k": 50,
    "moss_audio_repetition_penalty": 1.1,
}


# ============================================================================
# FILE PATHS
# ============================================================================

# Output directories
OUTPUT_DIR = Path("chapters")
# VOICE_SAMPLES_DIR is already defined above in AUDIO_SETTINGS section

# Default EPUB file for Gradio interface
DEFAULT_EPUB_FILE = SCRIPT_DIR.parent / "voice_test" / "test_pride_and_prejudice.epub"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_llm_port() -> int:
    """Get the LLM port from settings.

    Returns:
        LLM port number (default: 8080)
    """
    return LLM_SETTINGS["port"]

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
    if AUDIO_SETTINGS["default_tts_engine"] not in ["kugelaudio", "vibevoice", "moss"]:
        warnings.append(
            f"Unknown TTS engine: {AUDIO_SETTINGS['default_tts_engine']}. "
            "Using 'kugelaudio', 'vibevoice', or 'moss' is recommended."
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
