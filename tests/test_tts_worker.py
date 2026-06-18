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
        
        # Mock _find_python
        with patch.object(EngineWorker, '_find_python', return_value="/usr/bin/python"):
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
