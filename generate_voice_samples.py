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
import time

# Add parent directory to path to import qwen_tts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


def create_sample_text_from_description(description):
    """Create a short sample text from the character description."""
    cleaned = description.replace('|', ', ')
    sentences = cleaned.split('.')
    if sentences:
        return sentences[0][:150].strip()
    return cleaned[:150].strip()


def generate_voice_sample(tts_model, character_name, description, output_dir, max_new_tokens=512):
    """
    Generate a short voice sample for a character using VoiceDesign model.
    """
    sample_text = create_sample_text_from_description(description)
    instruct = f"Voice design for {character_name}: {description[:400]}"

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
        default="character_voice_samples",
        help="Directory to save voice samples"
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens for generation"
    )
    parser.add_argument(
        "--single-character",
        help="Generate only one character"
    )

    args = parser.parse_args()

    if not os.path.exists(args.descriptions):
        print(f"Error: Descriptions file not found: {args.descriptions}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading character descriptions from: {args.descriptions}")
    descriptions = load_character_descriptions(args.descriptions)

    if args.single_character:
        if args.single_character not in descriptions:
            print(f"Error: Character '{args.single_character}' not found.", file=sys.stderr)
            sys.exit(1)
        descriptions = {args.single_character: descriptions[args.single_character]}
        print(f"Generating voice for single character: {args.single_character}")
    else:
        print(f"Found {len(descriptions)} characters")

    print(f"\nLoading model: {args.model_path}")
    print("  (First run downloads weights from HuggingFace)")
    start_load = time.time()

    try:
        tts_model = Qwen3TTSModel.from_pretrained(
            args.model_path,
            device_map=args.device,
            dtype="bfloat16",
            attn_implementation=None,
        )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        print("\nNote: You may need to manually download the model first.", file=sys.stderr)
        sys.exit(1)

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.1f}s")

    output_dir = args.output_dir
    print(f"\nGenerating voice samples to: {output_dir}\n")

    generated = []
    failed = []

    for i, (char_name, char_desc) in enumerate(descriptions.items()):
        print(f"[{i+1}/{len(descriptions)}] {char_name}")

        success, output_file, duration = generate_voice_sample(
            tts_model, char_name, char_desc, output_dir,
            max_new_tokens=args.max_tokens
        )

        if success:
            generated.append((char_name, output_file, duration))
            print(f"    Generated: {duration:.2f}s -> {output_file}")
        else:
            failed.append((char_name, None))
            print(f"    Failed")

    print("\n" + "=" * 60)
    print(f"Summary: {len(generated)} generated, {len(failed)} failed")
    print("=" * 60)

    if generated:
        print("\nGenerated voice samples:")
        for char_name, path, duration in generated:
            print(f"  {char_name}: {path} ({duration:.2f}s)")

        # Generate voices_map.json for use in audiobook generation
        voices_map = {}
        # Add narrator as default (will be used for non-character speech)
        voices_map["narrator"] = "narrator.wav"
        # Map character names to their voice samples
        for char_name, path, duration in generated:
            # Extract just the filename from the path
            voice_file = os.path.basename(path)
            voices_map[char_name] = voice_file

        voices_map_path = os.path.join(output_dir, "voices_map.json")
        with open(voices_map_path, "w", encoding="utf-8") as f:
            json.dump(voices_map, f, indent=2)
        print(f"\nGenerated voices_map.json: {voices_map_path}")
        print("  (This file maps characters to their voice samples for audiobook generation)")


# ============================================================================
# MODULE FUNCTIONS FOR GRADIO INTERFACE
# ============================================================================

from typing import Dict, Optional, Tuple


def generate_voice_samples(
    descriptions: Dict[str, str],
    output_dir: str,
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device: str = "cuda:0",
    max_tokens: int = 512,
    single_character: Optional[str] = None,
    verbose: bool = False
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

        if verbose:
            print(f"\nLoading model: {model_path}")
            print("  (First run downloads weights from HuggingFace)")
        start_load = time.time()

        try:
            tts_model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=device,
                dtype="bfloat16",
                attn_implementation=None,
            )
        except Exception as e:
            return f"Error loading model: {e}", {}

        load_time = time.time() - start_load
        if verbose:
            print(f"Model loaded in {load_time:.1f}s")

        os.makedirs(output_dir, exist_ok=True)

        generated = {}
        failed = []

        for i, (char_name, char_desc) in enumerate(descriptions.items()):
            if verbose:
                print(f"[{i+1}/{len(descriptions)}] {char_name}")

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

        if verbose:
            print("\n" + "=" * 60)
            print(f"Summary: {len(generated)} generated, {len(failed)} failed")
            print("=" * 60)

        if generated:
            # Generate voices_map.json for use in audiobook generation
            voices_map = {}
            voices_map["narrator"] = "narrator.wav"
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


if __name__ == "__main__":
    main()