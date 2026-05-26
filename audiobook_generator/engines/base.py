"""Base class for TTS engines."""

from __future__ import annotations

import gc
import json
from abc import ABC, abstractmethod
from multiprocessing import Queue
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .worker import EngineWorker


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

    # Name matching the per-engine environment directory
    ENV_NAME: str = ""

    @classmethod
    @abstractmethod
    def _run_worker(cls, request_queue: Queue, response_queue: Queue) -> None:
        """Run the TTS worker loop in the subprocess.

        Loads the model once, then processes requests from the queue.
        Must send {"type": "ready"} to response_queue before the loop.
        """

    @abstractmethod
    def setup(self, device: str, turbo: bool = False) -> Tuple[Any, Optional[Any]]:
        """Load the model(s) for this engine (in-process, for non-worker usage).

        Subclasses should override this only for in-process usage.
        In worker mode, model loading happens in _run_worker().
        """

    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: Path,
        device: str,
        verbose: bool = False,
    ) -> Tuple[bool, Optional[str], float]:
        """Generate a voice sample for a character (Stage 4).

        Default implementation: checks for empty description, delegates to
        worker via _worker_request, clears CUDA cache.

        Override for engines with special handling needs.
        """
        if not description or not description.strip():
            if verbose:
                print(f"  ERROR: Skipping '{character_name}' due to empty description")
            return False, None, 0

        resp = self._worker_request(
            "generate_voice_sample",
            character_name=character_name,
            description=description,
            output_dir=str(output_dir),
        )
        self._clear_cuda_cache()
        success = resp.get("success", False)
        return success, resp.get("output_file"), resp.get("duration", 0)

    def generate_line(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        device: str,
        validation_model: Optional[Any] = None,
        cfg_scale: float = 1.3,
        max_new_tokens: int = 19200,
        verbose: bool = False,
    ) -> bool:
        """Generate audio for a single line (Stage 5).

        Default implementation: delegates to worker via _worker_request,
        clears CUDA cache.

        Override for engines with extra parameters (e.g., ref_text).
        """
        resp = self._worker_request(
            "generate_line",
            text=text,
            voice_path=voice_path,
            output_path=output_path,
        )
        self._clear_cuda_cache()
        return resp.get("success", False)

    def _get_worker(self) -> "EngineWorker":
        """Get or create the EngineWorker for this engine."""
        if not hasattr(self, "_worker"):
            from .worker import EngineWorker
            self._worker = EngineWorker(self.ENV_NAME, self.__class__.__name__)
            self._worker.start()
        return self._worker

    def shutdown_worker(self) -> None:
        """Shutdown the worker subprocess."""
        if hasattr(self, "_worker"):
            self._worker.shutdown()
            del self._worker

    def _worker_request(self, method: str, **kwargs: Any) -> dict[str, Any]:
        """Send a request to the worker and return the response."""
        worker = self._get_worker()
        resp = worker.request(method, device=self._device, **kwargs)
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp

    @staticmethod
    def _clear_cuda_cache() -> None:
        """Clear CUDA memory after worker calls."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
