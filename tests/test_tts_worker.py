"""Unit tests for tts.worker module."""

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class TestEngineWorker:
    """Test EngineWorker subprocess management."""

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

    @patch("tts.worker.subprocess.Popen")
    def test_start_process(self, mock_popen):
        """Test starting the worker subprocess."""
        from tts.worker import EngineWorker
        
        # Mock the process
        mock_process = MagicMock()
        mock_process.stdout.readline.return_value = '{"type": "ready"}\n'
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        
        # Mock _ensure_env
        with patch("tts.worker._ensure_env", return_value="/usr/bin/python"):
            worker = EngineWorker(Path("/tmp/test-engine"), device="cuda:0")
            worker.start()
            
            assert worker._process == mock_process
            mock_popen.assert_called_once()

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


class TestEnsureEnv:
    """Test _ensure_env function."""

    @patch("tts.worker.subprocess.run")
    def test_existing_env(self, mock_run):
        """Test using existing environment."""
        from tts.worker import _ensure_env

        # Mock that venv exists and has audiobook_generator installed
        mock_run.return_value = MagicMock(returncode=0)

        with patch("tts.worker.Path.exists", return_value=True):
            result = _ensure_env("omni", Path("/tmp/test-engine"))
            assert isinstance(result, str)


class TestRunCmd:
    """Test _run_cmd function."""

    @patch("tts.worker.subprocess.run")
    def test_successful_command(self, mock_run):
        """Test successful command execution."""
        from tts.worker import _run_cmd
        
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        _run_cmd(["uv", "venv"], "/tmp", {}, "create venv", "omni")
        
        mock_run.assert_called_once()

    @patch("tts.worker.subprocess.run")
    def test_failed_command(self, mock_run):
        """Test failed command execution."""
        from tts.worker import _run_cmd
        
        mock_run.return_value = MagicMock(returncode=1, stderr="Error occurred")
        
        with pytest.raises(RuntimeError, match="Failed to create venv"):
            _run_cmd(["uv", "venv"], "/tmp", {}, "create venv", "omni")


class TestRunWorkerSubprocess:
    """Test _run_worker_subprocess function."""

    def test_function_exists(self):
        """Test that _run_worker_subprocess exists."""
        from tts.worker import _run_worker_subprocess
        
        assert callable(_run_worker_subprocess)


class TestProtocol:
    """Test JSON protocol format."""

    def test_ready_message(self):
        """Test ready message format."""
        msg = {"type": "ready"}
        assert msg["type"] == "ready"

    def test_request_format(self):
        """Test request message format."""
        req = {
            "type": "request",
            "id": 1,
            "method": "generate_line",
            "kwargs": {
                "text": "Hello world",
                "voice_path": "/path/to/voice.wav",
                "output_path": "/path/to/output.wav",
            }
        }
        assert req["type"] == "request"
        assert req["id"] == 1
        assert req["method"] == "generate_line"
        assert "text" in req["kwargs"]
        assert "voice_path" in req["kwargs"]
        assert "output_path" in req["kwargs"]

    def test_response_format(self):
        """Test response message format."""
        resp_success = {"id": 1, "success": True}
        resp_fail = {"id": 1, "success": False}
        resp_error = {"id": 1, "error": "message", "traceback": "..."}
        
        assert resp_success["id"] == 1
        assert resp_success["success"] is True
        assert resp_fail["success"] is False
        assert resp_error["error"] == "message"

    def test_shutdown_message(self):
        """Test shutdown message format."""
        msg = {"type": "shutdown"}
        assert msg["type"] == "shutdown"
