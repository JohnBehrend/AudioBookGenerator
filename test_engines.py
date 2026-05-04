#!/usr/bin/env python3
"""Test TTS engines one at a time."""

import sys
import os
import gc
import torch
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path

def test_engine(engine_name: str, description: str = None) -> bool:
    gc.collect()
    torch.cuda.empty_cache()

    output_dir = Path("voice_test")
    output_dir.mkdir(exist_ok=True)

    if description is None:
        description = "male, middle-aged, british accent"

    print(f"\n{'='*60}")
    print(f"Testing {engine_name.upper()}")
    print(f"{'='*60}")

    try:
        from audiobook_generator.engines import get_engine
        engine = get_engine(engine_name, device="cuda:0")

        result = engine.generate_voice_sample(
            character_name=f"test_{engine_name}",
            description=description,
            output_dir=output_dir,
            device="cuda:0",
            verbose=True,
        )

        success, output_file, duration = result

        if success:
            print(f"SUCCESS: {output_file} ({duration:.2f}s)")
            return True
        else:
            print("FAILED: no output")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    engines = [
        ("moss", "a calm male voice with a smooth tone"),
        ("omni", "male, middle-aged, british accent"),
        ("vox", "male, middle aged, american accent"),
        ("vibevoice", "young female voice, cheerful and energetic"),
    ]

    results = {}
    for engine_name, description in engines:
        success = test_engine(engine_name, description)
        results[engine_name] = success
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for engine_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {engine_name}: {status}")

if __name__ == "__main__":
    main()