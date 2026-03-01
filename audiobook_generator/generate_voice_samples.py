#!/usr/bin/env python3
"""
Voice Sample Generator for Audiobook Characters

Generates short voice sample .wav files for each character based on their
description. Uses TTS engine from AUDIO_SETTINGS (default: kugelaudio).
"""

import argparse
import json
import os
import sys
import shutil
from typing import Optional

# Import config for default values
from config import DEFAULTS, AUDIO_SETTINGS, VOICE_SAMPLES_DIR

# Import VoiceMapper for centralized TTS management
from voice_mapper import VoiceMapper


def load_character_descriptions(descriptions_file):
    """Load character descriptions from JSON file."""
    with open(descriptions_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_voice_sample(character_name: str, description: str, output_dir: str,
                          device: str = None, max_new_tokens: int = None, verbose: bool = False) -> tuple:
    """Generate a short voice sample for a character using VoiceDesign model via VoiceMapper.

    Uses voice design with an instruct prompt to generate speech
    in the designed voice style.

    Args:
        character_name: Name of the character
        description: Voice description from LLM
        output_dir: Directory to save voice samples
        device: CUDA device (uses config default if not specified)
        max_new_tokens: Max tokens for generation
        verbose: Print verbose output

    Returns:
        Tuple of (success, output_file_path, duration_seconds)
    """
    if max_new_tokens is None:
        max_new_tokens = DEFAULTS["max_new_tokens"]
    if device is None:
        device = AUDIO_SETTINGS["default_device"]

    # Validate description
    if not description or not description.strip():
        if verbose:
            print(f"    ERROR: Skipping '{character_name}' due to empty description")
        return False, None, 0

    # Create VoiceMapper and generate voice sample
    voice_mapper = VoiceMapper(output_dir=output_dir, device=device, tts_engine=AUDIO_SETTINGS.get("default_tts_engine", "kugelaudio"))

    try:
        success, output_file, duration = voice_mapper.generate_voice_sample(
            character_name=character_name,
            description=description,
            output_dir=output_dir,
            max_new_tokens=max_new_tokens,
            verbose=verbose
        )

        # Cleanup model from GPU memory
        voice_mapper.cleanup_tts_models()

        return success, output_file, duration

    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)
        return False, None, 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate voice samples for audiobook characters"
    )
    parser.add_argument(
        "--descriptions",
        default="characters_descriptions.json",
        help="Path to characters_descriptions.json"
    )
    parser.add_argument(
        "--output-dir",
        default=VOICE_SAMPLES_DIR,
        help="Directory to save voice samples"
    )
    parser.add_argument(
        "--device",
        default=AUDIO_SETTINGS["default_device"],
        help="CUDA device (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULTS["max_new_tokens"],
        help="Max tokens for generation"
    )
    parser.add_argument(
        "--single-character",
        help="Generate only one character"
    )
    parser.add_argument(
        "--seed-voice-map",
        help="Path to existing voices_map.json to seed voices"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose printing for debug."
    )

    args = parser.parse_args()

    if not os.path.exists(args.descriptions):
        print(f"Error: Descriptions file not found: {args.descriptions}", file=sys.stderr)
        sys.exit(1)

    descriptions = load_character_descriptions(args.descriptions)

    if args.single_character:
        if args.single_character not in descriptions:
            print(f"Error: Character '{args.single_character}' not found.", file=sys.stderr)
            sys.exit(1)
        descriptions = {args.single_character: descriptions[args.single_character]}

    # Load seed voices if provided
    seed_characters = None
    if args.seed_voice_map:
        if os.path.exists(args.seed_voice_map):
            with open(args.seed_voice_map, 'r', encoding='utf-8') as f:
                seed_characters = json.load(f)

    status, generated = generate_voice_samples(
        descriptions=descriptions,
        output_dir=args.output_dir,
        device=args.device,
        max_tokens=args.max_tokens,
        single_character=args.single_character,
        verbose=args.verbose,
        seed_characters=seed_characters
    )

    print(status)


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================
# generate_voice_samples is the public function used by both CLI and Gradio UI.
# The CLI main() function above calls this function to do the actual work.
# This provides a single, consistent interface for all callers.

from typing import Dict, Tuple


def generate_voice_samples(
    descriptions: Dict[str, str],
    output_dir: str,
    device: str = "cuda:0",
    max_tokens: int = DEFAULTS["max_new_tokens"],
    single_character: Optional[str] = None,
    verbose: bool = False,
    progress=None,
    seed_characters: Dict[str, str] = None
) -> Tuple[str, Dict[str, str]]:
    """Generate voice samples for characters via VoiceMapper.

    This is a simplified interface for calling voice sample generation directly
    from the Gradio UI without subprocess.

    Args:
        descriptions: Dict mapping character names to their descriptions
        output_dir: Directory to save voice samples
        device: CUDA device (e.g., cuda:0, cpu)
        max_tokens: Max tokens for generation
        single_character: Generate only one character
        verbose: Print verbose output
        progress: Gradio progress bar to update during generation
        seed_characters: Dict mapping character names to existing voice paths from seed voices_map

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

        # Validate and print descriptions summary
        if verbose:
            print("\n" + "=" * 60)
            print("Character Descriptions Summary:")
            print("=" * 60)
            for char_name, char_desc in descriptions.items():
                desc_preview = char_desc.strip()[:100].replace("\n", " ")
                print(f"  {char_name}:")
                print(f"    Description: {desc_preview}...")
                if not char_desc or not char_desc.strip():
                    print(f"    WARNING: Empty or whitespace-only description!")
            print("=" * 60 + "\n")

        try:
            for i, (char_name, char_desc) in enumerate(descriptions.items()):
                if verbose:
                    print(f"[{i+1}/{total_chars}] {char_name}")

                # Update progress bar for each character
                if progress is not None:
                    progress((i + 1) / total_chars, desc=f"Generating voice for '{char_name}'...")

                success, output_file, duration = generate_voice_sample(
                    character_name=char_name,
                    description=char_desc,
                    output_dir=output_dir,
                    device=device,
                    max_new_tokens=max_tokens,
                    verbose=verbose
                )

                if success:
                    generated[char_name] = output_file
                    if verbose:
                        print(f"    Generated: {duration:.2f}s -> {output_file}")
                else:
                    failed.append(char_name)
                    if verbose:
                        print(f"    Failed")
        except Exception as e:
            import traceback
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
        import traceback
        error_msg = f"Error generating voice samples: {str(e)}\n{traceback.format_exc()}"
        if verbose:
            print(error_msg)
        return error_msg, {}