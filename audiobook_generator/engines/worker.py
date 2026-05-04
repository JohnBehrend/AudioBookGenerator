"""Subprocess worker infrastructure for TTS engines.

Each engine runs TTS inference in an isolated uv environment via a worker
subprocess. This module provides EngineWorker which manages the subprocess
lifecycle and request/response communication via multiprocessing queues.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from multiprocessing import Process, Queue, get_context
from pathlib import Path
from typing import Any, Optional

_ENVIRONMENTS_DIR = Path(__file__).parent / ".environments"

_Request = dict[str, Any]
_Response = dict[str, Any]

_SPAWN_CONTEXT = get_context("spawn")


def _run_cmd(cmd: list[str], cwd: str, env: dict[str, str], label: str, engine_name: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to {label} for {engine_name}: {result.stderr}")


def _ensure_env(engine_name: str) -> str:
    """Ensure the per-engine uv environment exists. Returns the python executable path."""
    env_dir = _ENVIRONMENTS_DIR / engine_name
    venv_dir = env_dir / ".venv"
    python = str(venv_dir / "bin" / "python")

    if venv_dir.exists():
        # Check if main package is installed
        result = subprocess.run(
            [python, "-c", "import audiobook_generator"],
            capture_output=True,
        )
        if result.returncode == 0:
            return python

    project_root = Path(__file__).resolve().parent.parent.parent
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)

    print(f"  Setting up {engine_name} environment...")
    if not venv_dir.exists():
        _run_cmd(["uv", "venv", str(venv_dir)], str(env_dir), env,
                  "create venv", engine_name)

    _run_cmd(["uv", "pip", "install", "-e", "."], str(env_dir), env,
              "install deps", engine_name)
    _run_cmd(["uv", "pip", "install", "-e", str(project_root)], str(env_dir), env,
              "install main package", engine_name)

    print(f"  {engine_name} environment ready.")
    return python


def _worker_entry(engine_name: str, engine_class: str, request_queue: Queue, response_queue: Queue) -> None:
    """Entry point for the worker subprocess.

    Imports the engine class from the main package and runs the worker loop.
    """
    import importlib

    module = importlib.import_module(
        f"audiobook_generator.engines.{engine_name.replace('-', '_')}"
    )
    engine_cls = getattr(module, engine_class)
    engine_cls._run_worker(request_queue, response_queue)


class EngineWorker:
    """Manages a TTS engine worker subprocess."""

    def __init__(self, engine_name: str, engine_class: str, env_dir: Path | None = None):
        self.engine_name = engine_name
        self.engine_class = engine_class
        self._env_dir = env_dir or (_ENVIRONMENTS_DIR / engine_name)
        self._worker: Optional[Process] = None
        self._request_queue: Optional[Queue] = None
        self._response_queue: Optional[Queue] = None
        self._python: str | None = None
        self._next_id = 0

    def start(self) -> None:
        """Start the worker subprocess using spawn context to avoid CUDA fork issues."""
        if self._worker is not None and self._worker.is_alive():
            return

        self._python = _ensure_env(self.engine_name)
        self._request_queue = _SPAWN_CONTEXT.Queue()
        self._response_queue = _SPAWN_CONTEXT.Queue()

        self._worker = _SPAWN_CONTEXT.Process(
            target=_worker_entry,
            args=(self.engine_name, self.engine_class, self._request_queue, self._response_queue),
            daemon=True,
        )
        self._worker.start()

        # Wait for ready signal
        try:
            resp = self._response_queue.get(timeout=30)
            if resp.get("type") != "ready":
                raise RuntimeError(f"Worker did not report ready: {resp}")
        except Exception as e:
            self.shutdown()
            raise RuntimeError(f"Worker failed to start: {e}") from e

    def _next_request_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def request(self, method: str, **kwargs: Any) -> _Response:
        """Send a request to the worker and wait for the response."""
        if self._request_queue is None:
            self.start()

        req_id = self._next_request_id()
        req: _Request = {
            "type": "request",
            "id": req_id,
            "method": method,
            "kwargs": kwargs,
        }
        self._request_queue.put(req)

        deadline = time.monotonic() + 600  # 10 min timeout per request
        while True:
            try:
                resp = self._response_queue.get(timeout=min(5, deadline - time.monotonic()))
            except Exception:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Worker request {req_id} timed out")
                continue

            if resp.get("id") == req_id:
                return resp

    def shutdown(self) -> None:
        """Shutdown the worker subprocess."""
        try:
            if self._request_queue is not None and self._worker and self._worker.is_alive():
                self._request_queue.put({"type": "shutdown"})
                self._worker.join(timeout=10)
                if self._worker.is_alive():
                    self._worker.terminate()
                    self._worker.join(timeout=5)
        except Exception:
            pass
        finally:
            self._worker = None
            self._request_queue = None
            self._response_queue = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.shutdown()
