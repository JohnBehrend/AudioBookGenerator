"""Voice sample generation for TTS engines.

This module provides functions for generating voice samples using TTS engines.
Moved from voice_mapper.py to separate TTS concerns from audiobook concerns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from .worker import EngineWorker


def generate_voice_sample(
    engine_dir: Path,
    device: str,
    character_name: str,
    description: str,
    output_dir: Path,
    verbose: bool = False,
) -> Tuple[bool, Optional[str], float]:
    """Generate a voice sample for a character using the configured TTS engine.

    Args:
        engine_dir: Path to the engine directory
        device: CUDA device string
        character_name: Name of the character
        description: Voice description from LLM
        output_dir: Output directory for the voice sample
        verbose: Print verbose output

    Returns:
        Tuple of (success, output_file_path, duration_seconds)
    """
    if not description or not description.strip():
        if verbose:
            print(f"  ERROR: Skipping '{character_name}' due to empty description")
        return False, None, 0

    worker = EngineWorker(engine_dir, device)
    try:
        worker.start()
        resp = worker.request(
            "generate_voice_sample",
            character_name=character_name,
            description=description,
            output_dir=str(output_dir),
            device=device,
        )
        success = resp.get("success", False)
        return success, resp.get("output_file"), resp.get("duration", 0)
    finally:
        worker.shutdown()


def build_voice_clone_prompt(
    engine_dir: Path,
    device: str,
    voice_path: str,
    ref_text: Optional[str] = None,
    auto_transcribe: bool = False,
    verbose: bool = False,
) -> Any:
    """Build a voice_clone_prompt for voice cloning.

    Args:
        engine_dir: Path to the engine directory
        device: CUDA device string
        voice_path: Path to the voice sample file
        ref_text: Reference text for voice cloning
        auto_transcribe: If True, transcribe the audio to get ref_text
        verbose: Print verbose output

    Returns:
        A voice_clone_prompt that can be reused for generate_line calls
    """
    if ref_text is None:
        ref_text = ""

    # Auto-transcribe if requested
    if auto_transcribe:
        try:
            from ..utils import transcribe_audio_with_whisper
            actual_ref_text, _, _ = transcribe_audio_with_whisper(None, voice_path)
            if verbose:
                print(f"  Transcribed ref_text: {actual_ref_text}")
            ref_text = actual_ref_text
        except Exception as e:
            if verbose:
                print(f"  Warning: auto_transcribe failed: {e}")

    import soundfile as sf
    import torch

    voice_audio, sr = sf.read(voice_path)
    voice_audio = torch.from_numpy(voice_audio)

    worker = EngineWorker(engine_dir, device)
    try:
        worker.start()
        resp = worker.request(
            "build_voice_clone_prompt",
            voice_path=voice_path,
            ref_text=ref_text,
            device=device,
        )
        return resp.get("voice_clone_prompt")
    finally:
        worker.shutdown()
