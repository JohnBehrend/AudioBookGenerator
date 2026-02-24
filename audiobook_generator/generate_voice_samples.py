#!/usr/bin/env python3
"""
Voice Sample Generator for Audiobook Characters

Generates short voice sample .wav files for each character based on their
description. Uses Qwen3-TTS with a generic interface.

Minimal imports - only uses qwen_tts package.
"""

import argparse
import json
import os
import sys
import torch
from typing import Optional

# Import config for default values
from config import DEFAULTS, AUDIO_SETTINGS, VOICE_SAMPLES_DIR

# Helper to check if flash-attn is available
def _get_attn_implementation() -> Optional[str]:
    """Return flash_attention_2 if available, otherwise None."""
    try:
        import flash_attn
        return "flash_attention_2"
    except ImportError:
        return None


try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("Error: qwen_tts package not found. Make sure it's installed.", file=sys.stderr)
    print("Try: python -m pip install qwen-tts", file=sys.stderr)
    sys.exit(1)


def load_character_descriptions(descriptions_file):
    """Load character descriptions from JSON file."""
    with open(descriptions_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_voice_sample(tts_model, character_name, description, output_dir, max_new_tokens=None):
    """
    Generate a short voice sample for a character using VoiceDesign model.

    Uses voice design with an instruct prompt to generate speech
    in the designed voice style.
    """
    if max_new_tokens is None:
        max_new_tokens = DEFAULTS["max_new_tokens"]
    sample_text = DEFAULTS["qwen3_ref_text"]
    instruct = f"Voice design for {character_name}: {description[:DEFAULTS['description_length']]}"

    try:
        wavs, sr = tts_model.generate_voice_design(
            text=sample_text,
            language="Auto",
            instruct=instruct,
            max_new_tokens=max_new_tokens,
        )

        if not wavs or len(wavs) == 0:
            return False, None, 0

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{character_name}.wav")

        # Save using soundfile (already available in venv)
        import soundfile as sf
        sf.write(output_file, wavs[0], sr)

        duration = len(wavs[0]) / sr
        return True, output_file, duration

    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)
        return False, None, 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate voice samples for audiobook characters using Qwen3-TTS"
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
        "--model-path",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="HuggingFace model ID"
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
        model_path=args.model_path,
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
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device: str = "cuda:0",
    max_tokens: int = DEFAULTS["max_new_tokens"],
    single_character: Optional[str] = None,
    verbose: bool = False,
    progress=None,
    seed_characters: Dict[str, str] = None
) -> Tuple[str, Dict[str, str]]:
    """Generate voice samples for characters using Qwen3-TTS.

    This is a simplified interface for calling voice sample generation directly
    from the Gradio UI without subprocess.

    Args:
        descriptions: Dict mapping character names to their descriptions
        output_dir: Directory to save voice samples
        model_path: HuggingFace model ID
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
        import time
        import soundfile as sf

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

        if verbose:
            print(f"\nLoading model: {model_path}")
            print("  (First run downloads weights from HuggingFace)")
        start_load = time.time()

        try:
            attn_impl = _get_attn_implementation()
            attn_kwargs = {"attn_implementation": attn_impl} if attn_impl else {}
            tts_model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=device,
                dtype=torch.bfloat16,
                **attn_kwargs
            )
        except Exception as e:
            return f"Error loading model: {e}", {}

        load_time = time.time() - start_load
        if verbose:
            print(f"Model loaded in {load_time:.1f}s")

        os.makedirs(output_dir, exist_ok=True)

        generated = {}
        failed = []
        total_chars = len(descriptions)

        try:
            for i, (char_name, char_desc) in enumerate(descriptions.items()):
                if verbose:
                    print(f"[{i+1}/{total_chars}] {char_name}")

                # Update progress bar for each character
                if progress is not None:
                    progress((i + 1) / total_chars, desc=f"Generating voice for '{char_name}'...")

                success, output_file, duration = generate_voice_sample(
                    tts_model, char_name, char_desc, output_dir, max_new_tokens=max_tokens
                )

                if success:
                    generated[char_name] = output_file
                    if verbose:
                        print(f"    Generated: {duration:.2f}s -> {output_file}")
                else:
                    failed.append(char_name)
                    if verbose:
                        print(f"    Failed")
        finally:
            # Clean up model from GPU memory after generation
            del tts_model
            import gc
            gc.collect()
            torch.cuda.synchronize()  # Wait for pending GPU operations to complete
            torch.cuda.empty_cache()  # Release cached memory

        if verbose:
            print("\n" + "=" * 60)
            print(f"Summary: {len(generated)} generated, {len(failed)} failed")
            print("=" * 60)

        # Build final voices_map
        if verbose:
            print("\n" + "=" * 60)
            print(f"Summary: {len(generated)} generated, {len(failed)} failed")
            print("=" * 60)

        if generated or seed_characters:
            # Generate voices_map.json for use in audiobook generation
            voices_map = {}
            voices_map["narrator"] = "narrator.wav"

            # Start with seed characters (from voices_map.json)
            if seed_characters:
                for char_name, voice_path in seed_characters.items():
                    # Extract just the filename if full path provided
                    voice_file = os.path.basename(voice_path)
                    voices_map[char_name] = voice_file
                if verbose:
                    print(f"Added {len(seed_characters)} seeded characters to voices_map")

            # Add newly generated voices
            for char_name, path in generated.items():
                voice_file = os.path.basename(path)
                voices_map[char_name] = voice_file

            voices_map_path = os.path.join(output_dir, "voices_map.json")
            with open(voices_map_path, "w", encoding="utf-8") as f:
                json.dump(voices_map, f, indent=2)
            if verbose:
                print(f"\nGenerated voices_map.json: {voices_map_path}")

        return f"Successfully generated {len(generated)} voice sample(s).", generated

    except Exception as e:
        import traceback
        error_msg = f"Error generating voice samples: {str(e)}\n{traceback.format_exc()}"
        if verbose:
            print(error_msg)
        return error_msg, {}