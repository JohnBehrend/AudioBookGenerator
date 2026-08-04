"""Unit tests for tts.worker module."""

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_fake_worker(tmp_path, code):
    """Create a fake engine dir with a worker entrypoint that runs the given code.

    The worker is launched with ``sys.executable`` (patched in via _find_python),
    so no engine venv is needed and tests exercise the real wire protocol.
    """
    eng = tmp_path / "fake_engine"
    (eng / ".venv" / "bin").mkdir(parents=True)
    (eng / "main.py").write_text(textwrap.dedent(code))
    return eng


def _echo_worker():
    """A worker that reports ready and echoes each request's id back."""
    return '''
        import sys, json
        print(json.dumps({"type": "ready"}), flush=True)
        for line in sys.stdin:
            req = json.loads(line)
            if req.get("type") == "shutdown":
                break
            if req.get("type") == "request":
                print(json.dumps({"id": req["id"], "success": True,
                                  "method": req["method"],
                                  "kwargs": req.get("kwargs", {})}), flush=True)
    '''


class TestEngineWorker:
    """Test EngineWorker initialization and pure unit behavior."""

    def test_init(self):
        """Test EngineWorker initialization."""
        from tts.worker import EngineWorker

        engine_dir = Path("/tmp/test-engine")
        worker = EngineWorker(engine_dir, device="cuda:0")

        assert worker.engine_dir == engine_dir
        assert worker.device == "cuda:0"
        assert worker._process is None
        assert worker._next_id == 0

    def test_next_request_id(self):
        """Test request ID increment."""
        from tts.worker import EngineWorker

        engine_dir = Path("/tmp/test-engine")
        worker = EngineWorker(engine_dir)

        assert worker._next_request_id() == 1
        assert worker._next_request_id() == 2
        assert worker._next_request_id() == 3

    def test_shutdown(self):
        """Test shutdown of worker process."""
        from tts.worker import EngineWorker

        engine_dir = Path("/tmp/test-engine")
        worker = EngineWorker(engine_dir)

        # Create a mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = [None, None]
        mock_process.stdin.write.return_value = None
        mock_process.stdin.flush.return_value = None

        worker._process = mock_process

        worker.shutdown()

        assert worker._process is None

    def test_context_manager(self):
        """Test context manager usage."""
        from tts.worker import EngineWorker

        engine_dir = Path("/tmp/test-engine")
        worker = EngineWorker(engine_dir)

        # Mock start and shutdown
        with patch.object(worker, 'start') as mock_start, \
             patch.object(worker, 'shutdown') as mock_shutdown:

            with worker as w:
                assert w == worker
                mock_start.assert_called_once()

            mock_shutdown.assert_called_once()


class TestFindPython:
    """Test EngineWorker._find_python method."""

    def test_find_python_without_venv(self):
        """Test raising error when venv doesn't exist."""
        from tts.worker import EngineWorker

        engine_dir = Path("/tmp/test-engine")
        worker = EngineWorker(engine_dir)

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(RuntimeError, match="environment not set up"):
                worker._find_python()


class TestRealProtocol:
    """Tests of the real EngineWorker wire protocol against a live subprocess."""

    def _worker(self, tmp_path, code, request_timeout=600.0):
        from tts.worker import EngineWorker
        eng = _make_fake_worker(tmp_path, code)
        worker = EngineWorker(eng, "cpu", request_timeout=request_timeout)
        with patch.object(EngineWorker, "_find_python", return_value=sys.executable):
            worker.start()
        return worker

    def test_start_launches_real_subprocess(self, tmp_path):
        from tts.worker import EngineWorker
        eng = _make_fake_worker(tmp_path, _echo_worker())
        worker = EngineWorker(eng, "cpu")
        with patch.object(EngineWorker, "_find_python", return_value=sys.executable):
            worker.start()
        try:
            assert worker._process is not None
            assert worker._process.poll() is None  # still alive
            assert worker._stderr_thread is not None  # drainer running
            assert worker._stderr_thread.is_alive()
        finally:
            worker.shutdown()

    def test_request_serializes_json_and_matches_id(self, tmp_path):
        """request() writes JSON, the worker echoes method/kwargs, response matched by id."""
        worker = self._worker(tmp_path, _echo_worker())
        try:
            resp = worker.request("generate_line", text="Hello world", n=42)
            assert resp["id"] == 1
            assert resp["success"] is True
            assert resp["method"] == "generate_line"
            assert resp["kwargs"] == {"text": "Hello world", "n": 42}
            assert worker._next_request_id() == 2
        finally:
            worker.shutdown()

    def test_drains_large_stderr(self, tmp_path):
        """A worker that writes >64KB to stderr must still respond (stderr drained).

        Regression for the 7-hour deadlock where an undrained stderr pipe filled,
        the worker blocked writing to it, and the parent blocked reading stdout.
        """
        code = '''
            import sys, json
            print(json.dumps({"type": "ready"}), flush=True)
            _chunk = "x" * 1024
            for _i in range(300):
                print(_chunk, file=sys.stderr)
            sys.stderr.flush()
            for line in sys.stdin:
                req = json.loads(line)
                if req.get("type") == "shutdown":
                    break
                if req.get("type") == "request":
                    print(json.dumps({"id": req["id"], "success": True}), flush=True)
        '''
        worker = self._worker(tmp_path, code)
        try:
            resp = worker.request("generate_line", text="hi")
            assert resp["success"] is True
        finally:
            worker.shutdown()

    def test_request_times_out_instead_of_hanging(self, tmp_path):
        """A worker that never answers surfaces TimeoutError rather than hanging."""
        code = '''
            import sys, json
            print(json.dumps({"type": "ready"}), flush=True)
            for line in sys.stdin:
                req = json.loads(line)
                if req.get("type") == "shutdown":
                    break
                # never respond to requests
        '''
        worker = self._worker(tmp_path, code, request_timeout=1.0)
        try:
            start = time.monotonic()
            with pytest.raises(TimeoutError):
                worker.request("generate_line", text="hi")
            assert time.monotonic() - start < 5.0
        finally:
            worker.shutdown()

    def test_shutdown_sends_shutdown_message_and_worker_exits(self, tmp_path):
        worker = self._worker(tmp_path, _echo_worker())
        proc = worker._process
        worker.shutdown()
        # Shutdown message honored -> worker process exited cleanly.
        assert proc.poll() is not None
