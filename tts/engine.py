"""Base class for TTS engines."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from .worker import EngineWorker


class TTSEngine:
    """Adapter for TTS engines running in isolated subprocess workers.

    Each engine implementation handles:
    - Loading its model(s) from HuggingFace or local sources
    - Generating voice samples from character descriptions (Stage 4)
    - Generating audio lines from text + voice reference (Stage 5)

    Engines run TTS inference in isolated subprocess workers. The adapter
    methods delegate to EngineWorker, while the engine's main.py handles the actual
    model loading and inference in the subprocess.
    """

    def __init__(self, engine_dir: Path, device: str = "cuda"):
        self.engine_dir = engine_dir
        self.device = device
        self._worker: Optional[EngineWorker] = None

    def _get_worker(self) -> EngineWorker:
        """Get or create the worker subprocess."""
        if self._worker is None:
            self._worker = EngineWorker(self.engine_dir, device=self.device)
            self._worker.start()
        return self._worker

    def generate_line(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        verbose: bool = False,
        ref_text: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """Generate audio for a single line (Stage 5)."""
        worker = self._get_worker()
        resp = worker.request(
            "generate_line",
            text=text,
            voice_path=voice_path,
            output_path=output_path,
            verbose=verbose,
            ref_text=ref_text,
            **kwargs,
        )
        if "error" in resp:
            print(f"    [EngineError] generate_line failed: {resp['error']}")
            if resp.get("traceback"):
                print(f"    {resp['traceback']}")
        elif not resp.get("success", False):
            print(f"    [EngineError] generate_line returned success=False (no error details)")
        return resp.get("success", False)

    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: Path,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tuple[bool, Optional[str], float]:
        """Generate a voice sample for a character (Stage 4)."""
        worker = self._get_worker()
        resp = worker.request(
            "generate_voice_sample",
            character_name=character_name,
            description=description,
            output_dir=str(output_dir),
            verbose=verbose,
            **kwargs,
        )
        if not resp.get("success", False):
            return (False, None, 0.0)
        return (True, resp.get("output_file"), resp.get("duration", 0.0))

    def shutdown_worker(self) -> None:
        """Shutdown the worker subprocess."""
        if self._worker is not None:
            self._worker.shutdown()
            self._worker = None

    @staticmethod
    def _clear_cuda_cache() -> None:
        """Clear CUDA memory after worker calls."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
