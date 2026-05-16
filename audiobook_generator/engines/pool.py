"""Multi-GPU worker pool for TTS engines.

A WorkerPool manages one EngineWorker subprocess per GPU and distributes
requests round-robin across them. Each worker loads its own model on its
own GPU, providing transparent multi-GPU scaling.
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional


class WhisperPool:
    """Pool of Whisper models for parallel validation.

    Each model has its own lock, so N models allow N concurrent transcriptions.
    Requests are distributed round-robin across the pool. Models can be
    distributed across multiple devices for balanced GPU utilization.
    """

    def __init__(self, model_factory, size: int, devices: Optional[List[str]] = None):
        self._size = size
        self._models: List[Any] = []
        self._locks: List[threading.Lock] = []
        self._index = 0
        self._global_lock = threading.Lock()

        for i in range(size):
            if devices:
                device = devices[i % len(devices)]
                model = model_factory(device)
            else:
                model = model_factory()
            self._models.append(model)
            self._locks.append(threading.Lock())

    def transcribe(self, audio_path: str, **kwargs) -> tuple:
        """Transcribe audio, routing to next model in round-robin order."""
        with self._global_lock:
            idx = self._index % self._size
            self._index += 1
        lock = self._locks[idx]
        model = self._models[idx]
        with lock:
            result = model.transcribe(audio_path, **kwargs)
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class WorkerPool:
    """Round-robin pool of TTS engine workers across multiple GPUs.

    Each worker runs its own model on its own GPU. Requests are distributed
    round-robin so all GPUs stay busy. The pool presents the same interface
    as a single TTSEngine for drop-in replacement.
    """

    def __init__(self, engine_name: str, engine_class: str, devices: List[str]):
        self.engine_name = engine_name
        self.engine_class = engine_class
        self.devices = devices
        self._workers: List["_WorkerDevice"] = []
        self._index = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start all worker subprocesses."""
        from .worker import EngineWorker

        for device in self.devices:
            worker = EngineWorker(self.engine_name, self.engine_class)
            worker.start()
            self._workers.append(_WorkerDevice(worker, device))

    def _next_worker(self) -> _WorkerDevice:
        """Get the next worker in round-robin order."""
        with self._lock:
            w = self._workers[self._index % len(self._workers)]
            self._index += 1
            return w

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
        """Generate audio for a single line, routing to next worker.

        Note: validation_model is accepted for API compatibility but NOT
        sent to the subprocess (it can't be pickled). Whisper validation
        runs in the main process after the audio file is written.
        """
        w = self._next_worker()
        resp = w.worker.request(
            "generate_line",
            text=text,
            voice_path=voice_path,
            output_path=output_path,
            device=w.device,
            cfg_scale=cfg_scale,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
        )
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp.get("success", True)

    def shutdown(self) -> None:
        """Shutdown all worker subprocesses."""
        for w in self._workers:
            w.worker.shutdown()
        self._workers.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.shutdown()


class _WorkerDevice:
    """Bundles an EngineWorker with its target GPU device."""

    __slots__ = ("worker", "device")

    def __init__(self, worker: Any, device: str):
        self.worker = worker
        self.device = device
