"""Subprocess worker infrastructure for TTS engines.

Each engine runs TTS inference in an isolated uv environment via a worker
subprocess. This module provides EngineWorker which manages the subprocess
lifecycle and request/response communication via JSON over stdin/stdout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

_ENVIRONMENTS_DIR = Path(__file__).parent / ".environments"

_Request = dict[str, Any]
_Response = dict[str, Any]


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
    _run_cmd(["uv", "pip", "install", "-e", str(project_root), "--no-deps"], str(env_dir), env,
              "install main package", engine_name)

    print(f"  {engine_name} environment ready.")
    return python


def _run_worker_subprocess(engine_name: str, engine_class: str) -> None:
    """Entry point for the worker subprocess.

    Reads JSON requests from stdin, writes JSON responses to stdout.
    """
    import importlib

    module = importlib.import_module(
        f"audiobook_generator.engines.{engine_name.replace('-', '_')}"
    )
    engine_cls = getattr(module, engine_class)

    # Create in-memory queues for the engine's _run_worker
    from multiprocessing import Queue
    req_queue: Any = Queue()
    resp_queue: Any = Queue()

    import threading

    def _forward_responses():
        """Forward responses from engine queue to stdout."""
        while True:
            try:
                resp = resp_queue.get(timeout=1)
                line = json.dumps(resp) + "\n"
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception:
                continue

    def _forward_requests():
        """Forward requests from stdin to engine queue."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                req_queue.put(msg)
            except json.JSONDecodeError:
                continue

    t = threading.Thread(target=_forward_responses, daemon=True)
    t.start()

    engine_cls._run_worker(req_queue, resp_queue)


class EngineWorker:
    """Manages a TTS engine worker subprocess."""

    def __init__(self, engine_dir: Path, device: str = "cuda"):
        self.engine_dir = engine_dir
        self.device = device
        self._process: Optional[subprocess.Popen] = None
        self._python: str | None = None
        self._next_id = 0

    def start(self) -> None:
        """Start the worker subprocess using the engine's isolated Python."""
        if self._process is not None and self._process.poll() is None:
            return

        self._python = _ensure_env(str(self.engine_dir))

        self._process = subprocess.Popen(
            [self._python, str(self.engine_dir / "main.py"), "--device", self.device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Wait for ready signal
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            line = self._process.stdout.readline()
            if not line:
                continue
            try:
                resp = json.loads(line.strip())
                if resp.get("type") == "ready":
                    return
            except json.JSONDecodeError:
                continue

        self.shutdown()
        raise RuntimeError(f"Worker did not report ready within 60s")

    def _next_request_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def request(self, method: str, **kwargs: Any) -> _Response:
        """Send a request to the worker and wait for the response."""
        if self._process is None:
            self.start()

        req_id = self._next_request_id()
        req: _Request = {
            "type": "request",
            "id": req_id,
            "method": method,
            "kwargs": kwargs,
        }

        line = json.dumps(req) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

        deadline = time.monotonic() + 600  # 10 min timeout per request
        while time.monotonic() < deadline:
            line = self._process.stdout.readline()
            if not line:
                if self._process.poll() is not None:
                    stderr = self._process.stderr.read()
                    raise RuntimeError(f"Worker process exited unexpectedly: {stderr}")
                time.sleep(0.1)
                continue
            try:
                resp = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if resp.get("id") == req_id:
                return resp

        raise TimeoutError(f"Worker request {req_id} timed out")

    def shutdown(self) -> None:
        """Shutdown the worker subprocess."""
        try:
            if self._process and self._process.poll() is None:
                try:
                    line = json.dumps({"type": "shutdown"}) + "\n"
                    self._process.stdin.write(line)
                    self._process.stdin.flush()
                except Exception:
                    pass
                self._process.wait(timeout=10)
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
        except Exception:
            pass
        finally:
            self._process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.shutdown()
