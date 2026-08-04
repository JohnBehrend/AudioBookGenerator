#!/usr/bin/env python3
"""Clone Arnold Schwarzenegger celebrity voice and save to arnold.wav.

Uses the existing celebrity voice pipeline to:
1. Download Arnold Schwarzenegger audio from YouTube
2. Extract clean speech segments
3. Generate voice reference using TTS engine
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from audiobook_generator.config import LLM_SETTINGS, DEFAULTS, AUDIO_SETTINGS
from audiobook_generator.celebrity_voices import build_celebrity_voice
from tts import get_engine


def main():
    output_file = Path(__file__).parent / "arnold.wav"
    output_dir = str(output_file.parent)
    device = AUDIO_SETTINGS.get("default_device", "cuda:0")
    celebrity = "Arnold Schwarzenegger"

    print(f"Cloning {celebrity} voice...")
    print(f"Output: {output_file}")
    print(f"Device: {device}")

    # Create LLM client for celebrity matching and video selection
    client = OpenAI(
        base_url=LLM_SETTINGS["endpoint"],
        api_key=LLM_SETTINGS["api_key"],
    )
    model = LLM_SETTINGS["default_model"]

    # Character description for Arnold's voice
    description = json.dumps({
        "gender": "male",
        "age": "elderly",
        "pitch": "low pitch",
        "accent": "austrian accent",
        "style": "authoritative",
        "celebrity_voice": celebrity,
    })

    # Get TTS engine for voice cloning
    print(f"\nLoading TTS engine: {AUDIO_SETTINGS['default_tts_engine']}")
    engine = get_engine(AUDIO_SETTINGS["default_tts_engine"], device=device)

    # Build celebrity voice - downloads audio, extracts segments, generates reference
    print(f"\nSearching YouTube for {celebrity} audio...")
    voice_path, metadata = build_celebrity_voice(
        client=client,
        model=model,
        character="arnold",
        description=description,
        output_dir=output_dir,
        pre_matched_celebrity=celebrity,
        max_videos=3,
        tts_engine=engine,
        verbose=True,
    )

    if voice_path:
        # Copy/move to final output location
        import shutil
        final_path = Path(output_dir) / "arnold.wav"
        if voice_path != str(final_path):
            shutil.copy2(voice_path, str(final_path))

        print(f"\nSuccess! Voice saved to: {final_path}")
        print(f"File size: {final_path.stat().st_size / 1024:.1f} KB")

        if metadata:
            print(f"\nMetadata:")
            print(f"  Celebrity: {metadata.get('celebrity', 'N/A')}")
            print(f"  Source: {metadata.get('source', 'N/A')}")
            if 'search_query' in metadata:
                print(f"  Search query: {metadata['search_query']}")

        # Cleanup engine
        engine.shutdown_worker()
        return True
    else:
        print(f"\nFailed to clone {celebrity} voice")
        engine.shutdown_worker()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
