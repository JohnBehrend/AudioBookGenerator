#!/usr/bin/env python3
"""Test all TTS engines with a simple voice sample generation."""

import sys
import os
import torch
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from audiobook_generator.engines import get_engine

def test_engine(engine_name: str, description: str = None) -> bool:
    output_dir = Path("voice_test")
    output_dir.mkdir(exist_ok=True)

    if description is None:
        description = "male, middle-aged, british accent"

    print(f"\n{'='*60}")
    print(f"Testing {engine_name.upper()} engine...")
    print(f"Description: {description}")
    print(f"{'='*60}")

    try:
        torch.cuda.empty_cache()
        engine = get_engine(engine_name, device="cuda:0")

        result = engine.generate_voice_sample(
            character_name=f"test_{engine_name}_voice",
            description=description,
            output_dir=output_dir,
            device="cuda:0",
            verbose=True,
        )

        success, output_file, duration = result

        if success:
            print(f"SUCCESS! Generated: {output_file}")
            print(f"Duration: {duration:.2f}s")
            return True
        else:
            print("FAILED - engine returned no output")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    engines = [
        ("moss", "a calm male voice with a smooth tone"),
        ("omni", "male, middle-aged, british accent"),
        ("vox", "male, middle aged, american accent"),
        ("kugelaudio", "a warm female voice with a gentle accent"),
        ("vibevoice", "young female voice, cheerful and energetic"),
        ("dramabox", "A middle-aged man speaks calmly, his voice steady and measured."),
    ]

    results = {}
    for engine_name, description in engines:
        results[engine_name] = test_engine(engine_name, description)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for engine_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {engine_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

if __name__ == "__main__":
    main()