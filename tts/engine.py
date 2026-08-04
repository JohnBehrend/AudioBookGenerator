"""Base class for TTS engines."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from .worker import (
    EngineWorker,
    SharedEngineWorker,
    acquire_shared_worker,
    release_shared_worker,
)


class TTSEngine:
    """Adapter for TTS engines running in isolated subprocess workers.

    Each engine implementation handles:
    - Loading its model(s) from HuggingFace or local sources
    - Generating voice samples from character descriptions (Stage 4)
    - Generating audio lines from text + voice reference (Stage 5)

    Engines run TTS inference in isolated subprocess workers. The adapter
    methods delegate to a *shared* worker (see SharedEngineWorker): every
    TTSEngine instance for the same (engine_dir, device) reuses one worker
    subprocess so a single model copy is loaded per GPU/engine, avoiding
    duplicate full-model loads (and OOM) from separate VoiceMappers.
    """

    def __init__(self, engine_dir: Path, device: str = "cuda"):
        self.engine_dir = engine_dir
        self.device = device
        self._shared: Optional[SharedEngineWorker] = None

    def _get_shared(self) -> SharedEngineWorker:
        """Get (and start) the shared worker, acquiring a reference on first use."""
        if self._shared is None:
            self._shared = acquire_shared_worker(self.engine_dir, self.device)
            self._shared.start()
        return self._shared

    def _get_worker(self) -> EngineWorker:
        """Return the underlying shared EngineWorker (kept for compatibility)."""
        return self._get_shared()._worker

    def _request(self, method: str, **kwargs: Any) -> Any:
        """Send a request through the shared worker (serialized)."""
        return self._get_shared().request(method, **kwargs)

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
        resp = self._request(
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
        resp = self._request(
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
        """Release this engine's reference to the shared worker.

        The worker subprocess is only actually terminated when the last
        reference (from any TTSEngine) is released.
        """
        if self._shared is not None:
            release_shared_worker(self.engine_dir, self.device)
            self._shared = None

    @staticmethod
    def _clear_cuda_cache() -> None:
        """Clear CUDA memory after worker calls."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
