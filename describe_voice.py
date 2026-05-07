#!/usr/bin/env python3
"""
Describe a voice WAV file using an OpenAI-compatible API.

Usage:
    python describe_voice.py <wav_file> [--endpoint URL] [--model MODEL] [--api-key KEY] [--verbose]

Example:
    python describe_voice.py my_voice.wav
    python describe_voice.py my_voice.wav --endpoint http://localhost:8081/v1 --model gemma4-model
"""
import argparse
import sys
from pathlib import Path

from openai import OpenAI

from audiobook_generator.config import LLM_SETTINGS, VOICE_VALIDATION
from audiobook_generator.voice_mapper import VoiceMapper


def main():
    parser = argparse.ArgumentParser(description="Describe a voice WAV file using an OpenAI-compatible API")
    parser.add_argument("wav_file", help="Path to the WAV file to describe")
    parser.add_argument("--endpoint", default=None,
                        help=f"API endpoint (default: {VOICE_VALIDATION['endpoint']})")
    parser.add_argument("--model", default=None,
                        help=f"Model name (default: {VOICE_VALIDATION['model']})")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: from config)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")

    args = parser.parse_args()

    # Validate input file
    wav_path = Path(args.wav_file)
    if not wav_path.exists():
        print(f"Error: File not found: {args.wav_file}", file=sys.stderr)
        sys.exit(1)
    if wav_path.suffix.lower() not in [".wav", ".mp3", ".flac"]:
        print(f"Warning: File extension '{wav_path.suffix}' may not be supported. Expected .wav, .mp3, or .flac", file=sys.stderr)

    # Create client
    endpoint = args.endpoint or VOICE_VALIDATION["endpoint"]
    api_key = args.api_key or LLM_SETTINGS["api_key"]
    model = args.model or VOICE_VALIDATION["model"]

    client = OpenAI(base_url=endpoint, api_key=api_key)

    if args.verbose:
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model}")
        print(f"Analyzing: {args.wav_file}")
        print()

    # Call the description function
    description = VoiceMapper.describe_voice_with_llm(
        voice_path=args.wav_file,
        client=client,
        model=model,
        verbose=args.verbose
    )

    if description:
        print(description)
    else:
        print("Error: Failed to get voice description", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
