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


if __name__ == "__main__":
    main()