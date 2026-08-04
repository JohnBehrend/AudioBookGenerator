"""Subprocess worker infrastructure for TTS engines.

Each engine runs TTS inference in an isolated subprocess. This module provides
EngineWorker which manages the subprocess lifecycle and request/response
communication via JSON over stdin/stdout.

Engine-agnostic: discovers engines by `main.py`, finds python via `.venv/bin/python`.

Deadlock avoidance
------------------
The worker subprocess has two output pipes (stdout = JSON protocol, stderr =
diagnostics) that are only 64KB each. If stderr is never drained while the
worker is alive, it fills up and the worker blocks forever writing to stderr —
never producing the stdout response — while the parent blocks reading stdout.
This produced the 7-hour "main on anon_pipe_read / worker on anon_pipe_write"
deadlock. We avoid it two ways:

1. A background daemon thread continuously drains stderr and forwards it, so the
   stderr pipe can never fill.
2. The stdout read loop uses a select() with a hard timeout, so a genuinely hung
   worker raises TimeoutError instead of blocking the parent indefinitely.
"""

from __future__ import annotations

import io
import json
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

_Request = dict[str, Any]
_Response = dict[str, Any]


class EngineWorker:
    """Manages a TTS engine worker subprocess."""

    def __init__(self, engine_dir: Path, device: str = "cuda", request_timeout: float = 600.0):
        self.engine_dir = engine_dir
        self.device = device
        self.request_timeout = request_timeout
        self._process: Optional[subprocess.Popen] = None
        self._next_id = 0
        self._stderr_thread: Optional[threading.Thread] = None
        self._start_lock = threading.Lock()

    def _find_python(self) -> str:
        """Find the python executable for this engine."""
        venv_python = self.engine_dir / ".venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        raise RuntimeError(
            f"Engine {self.engine_dir.name} environment not set up. "
            f"Run: uv run python scripts/setup-engines.py {self.engine_dir.name}"
        )

    def _drain_stderr(self, stream) -> None:
        """Read the worker's stderr pipe line-by-line and forward it.

        Without this, the worker's 64KB stderr pipe fills and the worker blocks
        writing to it forever (see module docstring). Runs in a daemon thread so
        it never blocks shutdown and never prevents the parent from exiting.
        """
        try:
            for line in iter(stream.readline, ""):
                if line:
                    print(f"[{self.engine_dir.name}] {line}", end="", file=sys.stderr, flush=True)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def start(self) -> None:
        """Start the worker subprocess (thread-safe; idempotent)."""
        with self._start_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        """Create the subprocess and wait for the ready signal.

        Assumes ``self._start_lock`` is held. ``start()`` wraps this so concurrent
        callers (e.g. several TTSEngine instances resolving a shared worker) can
        never both pass the not-running check and spawn two subprocesses.
        """
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

        # Drain stderr in the background so the pipe can never fill and deadlock.
        # Only real subprocess pipes (io.TextIOBase) need this; mocked pipes in
        # unit tests are excluded so they keep their blocking readline behavior.
        if isinstance(self._process.stderr, io.TextIOBase):
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(self._process.stderr,),
                daemon=True,
            )
            self._stderr_thread.start()

        # Wait for ready signal
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            line = self._readline(timeout=1.0)
            if line is None:
                if self._process.poll() is not None:
                    self.shutdown()
                    raise RuntimeError(f"Worker exited before reporting ready")
                continue
            try:
                resp = json.loads(line.strip())
                if resp.get("type") == "ready":
                    return
            except json.JSONDecodeError:
                continue

        self.shutdown()
        raise RuntimeError(f"Worker did not report ready within 60s")

    def _readline(self, timeout: float) -> Optional[str]:
        """Read one line from stdout, honoring a timeout.

        Uses select() so a real worker that never writes (e.g. hung or
        deadlocked) is surfaced as a timeout rather than blocking the parent
        forever. Mocked stdout (unit tests) is not select-able, so it falls back
        to a plain blocking readline.
        """
        if self._process is None or self._process.stdout is None:
            return None
        if not isinstance(self._process.stdout, io.TextIOBase):
            return self._process.stdout.readline()
        try:
            r, _, _ = select.select([self._process.stdout], [], [], timeout)
        except (ValueError, OSError):
            # stdout closed underneath us.
            return None
        if not r:
            return None
        line = self._process.stdout.readline()
        return line

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

        deadline = time.monotonic() + self.request_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Worker request {req_id} ({method}) timed out after "
                    f"{self.request_timeout:.0f}s"
                )
            line = self._readline(min(remaining, 1.0))
            if line is None:
                if self._process.poll() is not None:
                    stderr = self._process.stderr.read()
                    raise RuntimeError(f"Worker process exited unexpectedly: {stderr}")
                continue
            try:
                resp = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if resp.get("id") == req_id:
                return resp

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
            self._stderr_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.shutdown()


class SharedEngineWorker:
    """A reference-counted, request-serialized shared EngineWorker.

    Multiple ``TTSEngine`` instances for the same ``(engine_dir, device)`` share
    one worker subprocess so that only a single model copy is loaded per
    GPU/engine. Previously each engine instance spawned its own worker (e.g. the
    seed-clone mapper and the main voice mapper each loaded a full omni model on
    the same GPU, doubling memory and risking OOM).

    Requests are serialized with a lock because the worker's JSON-over-stdin/
    stdout protocol is not safe under concurrent writers — interleaved writes
    would corrupt request/response matching.
    """

    def __init__(self, engine_dir: Path, device: str):
        self.engine_dir = engine_dir
        self.device = device
        self._worker = EngineWorker(engine_dir, device)
        self._lock = threading.Lock()
        self.refcount = 0

    def acquire(self) -> None:
        """Take a reference to this shared worker."""
        self.refcount += 1

    def release(self) -> None:
        """Drop a reference; shut the worker down when the last ref is released."""
        self.refcount -= 1
        if self.refcount <= 0:
            self.shutdown()

    def start(self) -> None:
        """Start the underlying worker if it is not already running."""
        self._worker.start()

    def request(self, method: str, **kwargs: Any) -> _Response:
        """Send a request under the serialization lock."""
        with self._lock:
            return self._worker.request(method, **kwargs)

    def shutdown(self) -> None:
        self._worker.shutdown()

    @property
    def is_alive(self) -> bool:
        proc = self._worker._process
        return proc is not None and proc.poll() is None

    @property
    def pid(self) -> Optional[int]:
        proc = self._worker._process
        return proc.pid if proc is not None else None


# Module-level pool of shared workers keyed by (engine_dir, device). Using the
# *module* registry (not an instance attribute) is what lets separate TTSEngine
# instances — and separate VoiceMappers — share a single worker subprocess.
_shared_pool: dict[tuple[str, str], SharedEngineWorker] = {}
_shared_pool_lock = threading.Lock()


def acquire_shared_worker(engine_dir: Path, device: str) -> SharedEngineWorker:
    """Get a shared worker for (engine_dir, device), incrementing its refcount."""
    key = (str(engine_dir), device)
    with _shared_pool_lock:
        sw = _shared_pool.get(key)
        if sw is None:
            sw = SharedEngineWorker(engine_dir, device)
            _shared_pool[key] = sw
        sw.acquire()
        return sw


def release_shared_worker(engine_dir: Path, device: str) -> None:
    """Release a reference to the shared worker for (engine_dir, device)."""
    key = (str(engine_dir), device)
    with _shared_pool_lock:
        sw = _shared_pool.get(key)
        if sw is None:
            return
        sw.release()
        if sw.refcount <= 0:
            _shared_pool.pop(key, None)


def shared_worker_count(engine_dir: Optional[Path] = None, device: Optional[str] = None) -> int:
    """Number of live shared workers (optionally filtered), for tests/observability."""
    with _shared_pool_lock:
        if engine_dir is None and device is None:
            return len(_shared_pool)
        return sum(
            1
            for (dir_key, dev_key) in _shared_pool
            if (engine_dir is None or dir_key == str(engine_dir))
            and (device is None or dev_key == device)
        )
