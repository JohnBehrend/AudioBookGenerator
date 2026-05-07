#!/usr/bin/env python3
"""
Module to generate voice samples for audiobook characters.
"""
import json
import os
import sys
import shutil
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple
from openai import OpenAI

# Import config for default values
from .config import DEFAULTS, AUDIO_SETTINGS, VOICE_SAMPLES_DIR
from .utils import get_validation_client

# Import VoiceMapper for centralized TTS management
from .voice_mapper import VoiceMapper


def load_character_descriptions(descriptions_file: str) -> Dict[str, str]:
    """Load character descriptions from JSON file."""
    with open(descriptions_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_voice_sample(character_name: str, description: str, voice_mapper: VoiceMapper,
                          output_dir: str, max_new_tokens: Optional[int] = None, verbose: bool = False,
                          validate: bool = False, validation_client: Optional[OpenAI] = None) -> Tuple[bool, Optional[str], float, bool, str]:
    """Generate a short voice sample for a character using VoiceDesign model via VoiceMapper.

    Uses voice design with an instruct prompt to generate speech
    in the designed voice style.

    Args:
        character_name: Name of the character
        description: Voice description from LLM
        voice_mapper: Shared VoiceMapper instance (prevents repeated model loading)
        output_dir: Directory to save voice samples
        max_new_tokens: Max tokens for generation
        verbose: Print verbose output
        validate: If True, validate the generated voice with LLM
        validation_client: OpenAI client for validation (created if None)

    Returns:
        Tuple of (success, output_file_path, duration_seconds, is_valid, validation_msg)
        When validate=True, is_valid indicates if the voice passed validation.
    """
    if max_new_tokens is None:
        max_new_tokens = DEFAULTS["max_new_tokens"]

    try:
        success, output_file, duration = voice_mapper.generate_voice_sample(
            character_name=character_name,
            description=description,
            output_dir=output_dir,
            max_new_tokens=max_new_tokens,
            verbose=verbose
        )

        is_valid = True  # Default: no validation = accepted
        validation_msg = ""

        if success and output_file and validate:
            if verbose:
                print(f"    Validating voice sample...")

            sample_text = DEFAULTS.get("static_voice_text", "")

            if validation_client is None:
                validation_client = get_validation_client()

            is_valid, validation_msg = VoiceMapper.validate_voice_with_llm(
                voice_path=output_file,
                description=description,
                sample_text=sample_text,
                client=validation_client,
                verbose=verbose
            )

            if not is_valid:
                print(f"    Warning: Voice validation failed: {validation_msg}")

        return success, output_file, duration, is_valid, validation_msg

    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)
        print(f"    Exception type: {type(e).__name__}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False, None, 0, False, "Exception during generation"


from typing import Dict, Tuple


def generate_voice_samples(
    descriptions: Dict[str, str],
    output_dir: str,
    device: str = "cuda:0",
    max_tokens: int = DEFAULTS["max_new_tokens"],
    single_character: Optional[str] = None,
    verbose: bool = False,
    progress=None,
    seed_characters: Dict[str, str] = None,
    voice_engine: str = "moss",
    force_regenerate: bool = False,
    validate: bool = False,
    engine=None
) -> Tuple[str, Dict[str, str]]:
    """Generate voice samples for characters via VoiceMapper.

    Args:
        descriptions: Dict mapping character names to their descriptions
        output_dir: Directory to save voice samples
        device: CUDA device (e.g., cuda:0, cpu)
        max_tokens: Max tokens for generation
        single_character: Generate only one character
        verbose: Print verbose output
        progress: Gradio progress bar to update during generation
        seed_characters: Dict mapping character names to existing voice paths from seed voices_map
        voice_engine: TTS engine for voice generation ('moss', 'omni', 'vox')
        force_regenerate: If True, regenerate voices even if they already exist
        validate: Deprecated - ignored

    Returns:
        Tuple of (status_message, character_voice_paths)
    """
    try:
        if single_character:
            if single_character not in descriptions:
                return f"Character '{single_character}' not found in descriptions.", {}
            descriptions = {single_character: descriptions[single_character]}
            if verbose:
                print(f"Generating voice for single character: {single_character}")
        else:
            if verbose:
                print(f"Found {len(descriptions)} characters")

        # Ensure narrator voice is included (it's always needed as fallback)
        # Add narrator with a default description if not already present
        if "narrator" not in descriptions:
            descriptions["narrator"] = "A neutral, clear narrator voice suitable for audiobook narration"
            if verbose:
                print("Added narrator voice to generation list (fallback voice)")

        # Filter out seed characters
        if seed_characters:
            initial_count = len(descriptions)
            descriptions = {k: v for k, v in descriptions.items() if k not in seed_characters}
            if verbose:
                print(f"Filtered out {initial_count - len(descriptions)} seeded characters, {len(descriptions)} remaining to generate")

            # Copy seed voice files to output directory
            if verbose:
                print(f"Copying seed voice files to {output_dir}...")
            for char_name, voice_path in seed_characters.items():
                if os.path.exists(voice_path):
                    dest_path = os.path.join(output_dir, os.path.basename(voice_path))
                    shutil.copy2(voice_path, dest_path)
                    if verbose:
                        print(f"  Copied {voice_path} -> {dest_path}")
                else:
                    if verbose:
                        print(f"  Warning: Seed voice file not found: {voice_path}")

        os.makedirs(output_dir, exist_ok=True)

        generated = {}
        failed = []
        total_chars = len(descriptions)

        # Note: Descriptions are no longer used for voice generation
        # Voices use static text from config instead
        if verbose:
            print("\n" + "=" * 60)
            print("NOTE: Character descriptions are no longer used for voice generation.")
            print("      Voices use a static string from config instead.")
            print("=" * 60 + "\n")

        # Create VoiceMapper once to cache the TTS model across all characters
        voice_mapper = VoiceMapper(output_dir=output_dir, device=device, tts_engine=voice_engine, engine=engine)

        try:
            for i, (char_name, char_desc) in enumerate(descriptions.items()):
                if verbose:
                    print(f"[{i+1}/{total_chars}] {char_name}")

                # Update progress bar for each character
                if progress is not None:
                    progress((i + 1) / total_chars, desc=f"Generating voice for '{char_name}'...")

                # Check if voice sample already exists (resume mode)
                if char_name in generated:
                    if verbose:
                        print(f"    Skipped - already generated")
                    continue

                # Check for existing voice file in output_dir (skip if exists, unless force_regenerate)
                if not force_regenerate:
                    voice_found = False
                    for ext in [".wav", ".mp3", ".flac"]:
                        test_path = os.path.join(output_dir, f"{char_name}{ext}")
                        if os.path.exists(test_path):
                            generated[char_name] = test_path
                            if verbose:
                                print(f"    Found existing voice: {test_path}")
                            voice_found = True
                            break

                    if voice_found:
                        continue

                # Generate voice (pass shared voice_mapper to avoid repeated model loading)
                success, output_file, duration, is_valid, validation_msg = generate_voice_sample(
                    character_name=char_name,
                    description=char_desc,
                    voice_mapper=voice_mapper,
                    output_dir=output_dir,
                    max_new_tokens=max_tokens,
                    verbose=verbose,
                    validate=False,
                    validation_client=None
                )

                if success:
                    generated[char_name] = output_file
                    if verbose:
                        print(f"    Generated: {output_file}")
                else:
                    failed.append(char_name)
                    if verbose:
                        print(f"    Failed to generate voice")

            # Clean up TTS models after all characters are processed
            voice_mapper.cleanup_tts_models()
        except Exception as e:
            return f"Error generating voices: {str(e)}\n{traceback.format_exc()}", {}

        if verbose:
            print("\n" + "=" * 60)
            print(f"Summary: {len(generated)} generated, {len(failed)} failed")
            print("=" * 60)

        # VoiceMapper automatically saves voices_map.json when voice paths are added
        if verbose:
            print(f"\nGenerated voices_map.json automatically by VoiceMapper")

        return f"Successfully generated {len(generated)} voice sample(s).", generated

    except Exception as e:
        error_msg = f"Error generating voice samples: {str(e)}\n{traceback.format_exc()}"
        if verbose:
            print(error_msg)
        return error_msg, {}
