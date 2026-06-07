"""Base class for TTS engines."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple

import torch


class TTSEngine(ABC):
    """Abstract base class for TTS engines.

    Each engine implementation handles:
    - Loading its model(s) from HuggingFace or local sources
    - Generating voice samples from character descriptions (Stage 4)
    - Generating audio lines from text + voice reference (Stage 5)

    Engines run TTS inference in isolated subprocess workers. The adapter
    methods delegate to EngineWorker, while _run_worker() handles the actual
    model loading and inference in the subprocess.
    """

    def __init__(self, engine_dir: Path, device: str = "cuda"):
        self.engine_dir = engine_dir
        self.device = device

    @abstractmethod
    def generate_line(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        verbose: bool = False,
        ref_text: Optional[str] = None,
    ) -> bool:
        """Generate audio for a single line (Stage 5)."""
        ...

    @abstractmethod
    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: Path,
        verbose: bool = False,
    ) -> Tuple[bool, Optional[str], float]:
        """Generate a voice sample for a character (Stage 4)."""
        ...

    @staticmethod
    def _clear_cuda_cache() -> None:
        """Clear CUDA memory after worker calls."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
