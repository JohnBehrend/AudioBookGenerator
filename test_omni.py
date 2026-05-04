#!/usr/bin/env python3
"""Test Omni engine with a simple voice sample generation."""

import sys
import os
import torch
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from audiobook_generator.engines import get_engine
from audiobook_generator.config import DEFAULTS

def main():
    torch.cuda.empty_cache()
    print("GPU Memory before:", torch.cuda.memory_allocated() / 1e9, "GB")

    output_dir = Path("voice_test")
    output_dir.mkdir(exist_ok=True)

    print("Setting up Omni engine...")
    engine = get_engine("omni", device="cuda:0")

    description = "male, middle-aged, british accent"
    print(f"Generating voice sample for: {description}")

    success, output_file, duration = engine.generate_voice_sample(
        character_name="test_omni_voice",
        description=description,
        output_dir=output_dir,
        device="cuda:0",
        verbose=True,
    )

    if success:
        print(f"Success! Generated: {output_file}")
        print(f"Duration: {duration:.2f}s")
    else:
        print("Failed to generate voice sample")

if __name__ == "__main__":
    main()