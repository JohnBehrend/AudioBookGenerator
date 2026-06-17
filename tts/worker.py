"""Subprocess worker infrastructure for TTS engines.

Each engine runs TTS inference in an isolated subprocess. This module provides
EngineWorker which manages the subprocess lifecycle and request/response
communication via JSON over stdin/stdout.

Engine-agnostic: discovers engines by `main.py`, finds python via `.venv/bin/python`.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

_Request = dict[str, Any]
_Response = dict[str, Any]


class EngineWorker:
    """Manages a TTS engine worker subprocess."""

    def __init__(self, engine_dir: Path, device: str = "cuda"):
        self.engine_dir = engine_dir
        self.device = device
        self._process: Optional[subprocess.Popen] = None
        self._next_id = 0

    def _find_python(self) -> str:
        """Find the python executable for this engine."""
        venv_python = self.engine_dir / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        raise RuntimeError(
            f"Engine {self.engine_dir.name} environment not set up. "
            f"Run: uv run python scripts/setup-engines.py {self.engine_dir.name}"
        )

    def start(self) -> None:
        """Start the worker subprocess."""
        if self._process is not None and self._process.poll() is None:
            return

        python = self._find_python()

        self._process = subprocess.Popen(
            [python, str(self.engine_dir / "main.py"), "--device", self.device],
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
