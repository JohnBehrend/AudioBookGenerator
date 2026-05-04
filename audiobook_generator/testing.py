#!/usr/bin/env python3
"""Mock TTS engine for unit testing."""

import os
import numpy as np
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torchaudio


class MockTTSEngine:
    """Mock TTS engine for testing without GPU/heavy dependencies.

    This engine simulates TTS generation by writing silence audio files.
    It can be configured to return success or failure for testing.

    Usage:
        from audiobook_generator.testing import MockTTSEngine

        engine = MockTTSEngine()
        voice_mapper = VoiceMapper(output_dir="/tmp/test", engine=engine)
    """

    def __init__(
        self,
        generate_success: bool = True,
        generate_voice_success: bool = True,
        duration: float = 1.0,
        sample_rate: int = 22050,
    ):
        """Initialize mock engine.

        Args:
            generate_success: Whether generate_line should return success
            generate_voice_success: Whether generate_voice_sample should return success
            duration: Duration of generated audio in seconds
            sample_rate: Sample rate for generated audio
        """
        self.generate_success = generate_success
        self.generate_voice_success = generate_voice_success
        self.duration = duration
        self.sample_rate = sample_rate
        self._device = "cpu"
        self._worker = None

        self.last_generate_line_args: dict = {}
        self.last_generate_voice_args: dict = {}

    def setup(self, device: str, turbo: bool = False) -> Tuple[Any, Optional[Any]]:
        """Mock setup - returns dummy values."""
        self._device = device
        return None, None

    def generate_line(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        device: str,
        validation_model: Any,
        cfg_scale: float = 1.3,
        max_new_tokens: int = 19200,
        verbose: bool = False,
    ) -> bool:
        """Mock line generation - writes silence audio.

        Args:
            text: Text that would be synthesized
            voice_path: Path to voice reference (ignored)
            output_path: Where to write the output audio
            device: Device string (ignored)
            validation_model: Validation model (ignored)
            cfg_scale: CFG scale (ignored)
            max_new_tokens: Max tokens (ignored)
            verbose: Print verbose output

        Returns:
            generate_success value
        """
        self.last_generate_line_args = {
            "text": text,
            "voice_path": voice_path,
            "output_path": output_path,
            "device": device,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        num_samples = int(self.sample_rate * self.duration)
        audio = np.zeros(num_samples, dtype=np.float32)

        torchaudio.save(output_path, torch.from_numpy(audio), self.sample_rate)

        if verbose:
            print(f"[MockTTS] Generated {output_path} ({self.duration}s silence)")

        return self.generate_success

    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: Path,
        device: str,
        verbose: bool = False,
    ) -> Tuple[bool, Optional[str], float]:
        """Mock voice sample generation.

        Args:
            character_name: Character name
            description: Voice description (ignored)
            output_dir: Directory to save sample
            device: Device string (ignored)
            verbose: Print verbose output

        Returns:
            Tuple of (generate_voice_success, output_path, duration)
        """
        self.last_generate_voice_args = {
            "character_name": character_name,
            "description": description,
            "output_dir": str(output_dir),
            "device": device,
        }

        output_path = output_dir / f"{character_name}.wav"
        os.makedirs(output_dir, exist_ok=True)

        num_samples = int(self.sample_rate * self.duration)
        audio = np.zeros(num_samples, dtype=np.float32)

        torchaudio.save(str(output_path), torch.from_numpy(audio), self.sample_rate)

        if verbose:
            print(f"[MockTTS] Generated voice sample {output_path}")

        return self.generate_voice_success, str(output_path), self.duration

    def shutdown_worker(self) -> None:
        """Mock shutdown - no-op."""
        pass

    def reset(self) -> None:
        """Reset captured arguments."""
        self.last_generate_line_args = {}
        self.last_generate_voice_args = {}