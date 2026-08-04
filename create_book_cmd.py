#!/usr/bin/env python3
"""
Create a run command file for a new audiobook.

Usage:
    python3 create_book_cmd.py RA4 /path/to/RA4.epub

Creates ra4_cmd.txt that:
- Seeds voices from all previous books (RA1, RA2, RA3, etc.)
- Seeds celebrity voices from celebrity_voices_archive/
- Uses celebrity voice matching for new characters
- Normalizes loudness to -23 LUFS
- Disables postfix
"""

import json
import os
import glob
import sys

output_dir = "voice_test"

# Find existing book output directories
existing_books = []
for d in sorted(glob.glob(os.path.join(output_dir, "RA*_output"))):
    voices_map = os.path.join(d, "voices_map.json")
    if os.path.exists(voices_map):
        existing_books.append(d)

# Merge all existing voices maps
merged = {}
for book_dir in existing_books:
    with open(os.path.join(book_dir, "voices_map.json")) as f:
        voices = json.load(f)
    book_name = os.path.basename(book_dir).replace("_output", "")
    for char, wav in voices.items():
        # Use relative path from voice_test/
        key = char
        if key not in merged:  # First occurrence wins
            merged[key] = f"{book_name}_output/{wav}"

# Save merged map
merged_path = os.path.join(output_dir, "all_voices_map.json")
with open(merged_path, "w") as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(merged)} voices from {len(existing_books)} books -> {merged_path}")

if len(sys.argv) >= 3:
    book_name = sys.argv[1]
    epub_path = sys.argv[2]
else:
    book_name = input("Book name (e.g., RA4): ").strip()
    epub_path = input("EPUB path: ").strip()

# Create cmd file
cmd_path = f"{book_name.lower()}_cmd.txt"
cmd = f'''from audiobook_generator.audiobook_generator import run_full_pipeline

result = run_full_pipeline(
    epub_path='{epub_path}',
    output_dir='{output_dir}/{book_name}_output',
    max_chapters=None,
    verbose=True,
    api_key=None,
    llm_port='2136',
    voice_engine='dramabox',
    tts_engine='omni',
    device='cuda:0',
    seed_voice_map='{merged_path}',
    num_llm_attempts=2,
    resume=True,
    whisper_cpu=False,
    concurrency=1,
    gpus=['cuda:0'],
    whisper_concurrency=1,
    whisper_fast=False,
    use_chunkformer=True,
    celebrity_voices=True,
    llm_model='coder-model',
    desc_concurrency=1,
    enable_postfix=False,
)
print('RESULT:', result)
'''

with open(cmd_path, "w") as f:
    f.write(cmd)

print(f"Created {cmd_path}")
print(f"\nTo run: setsid nohup uv run --python 3.12 python3 {cmd_path} > /tmp/{book_name.lower()}_resume.out 2>&1 &")
