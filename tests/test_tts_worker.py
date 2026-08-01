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
    """Test the real EngineWorker JSON request/response wire protocol."""

    @patch("tts.worker.subprocess.Popen")
    def test_request_serializes_json_on_stdin(self, mock_popen):
        """request() writes a JSON request with incrementing id and matches response by id."""
        from tts.worker import EngineWorker

        mock_process = MagicMock()
        # First line consumed by start() (ready), second by request() (response for id 1)
        mock_process.stdout.readline.side_effect = [
            '{"type": "ready"}\n',
            '{"id": 1, "success": true, "output_file": "/tmp/out.wav"}\n',
        ]
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        with patch.object(EngineWorker, "_find_python", return_value="/usr/bin/python"):
            worker = EngineWorker(Path("/tmp/test-engine"), device="cuda:0")
            resp = worker.request(
                "generate_line",
                text="Hello world",
                voice_path="/tmp/voice.wav",
                output_path="/tmp/out.wav",
            )

        req = json.loads(mock_process.stdin.write.call_args_list[0][0][0])
        assert req["type"] == "request"
        assert req["id"] == 1
        assert req["method"] == "generate_line"
        assert req["kwargs"] == {
            "text": "Hello world",
            "voice_path": "/tmp/voice.wav",
            "output_path": "/tmp/out.wav",
        }

        # Response is matched by request id and returned as-is
        assert resp == {"id": 1, "success": True, "output_file": "/tmp/out.wav"}
        mock_process.stdin.flush.assert_called_once()

    @patch("tts.worker.subprocess.Popen")
    def test_request_ignores_other_ids_and_increments(self, mock_popen):
        """request() ignores responses with a different id and increments the request id."""
        from tts.worker import EngineWorker

        mock_process = MagicMock()
        # ready, then an unrelated response (id=7), then the matching response (id=1)
        mock_process.stdout.readline.side_effect = [
            '{"type": "ready"}\n',
            '{"id": 7, "success": false}\n',
            '{"id": 1, "success": true}\n',
        ]
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        with patch.object(EngineWorker, "_find_python", return_value="/usr/bin/python"):
            worker = EngineWorker(Path("/tmp/test-engine"), device="cuda:0")
            resp = worker.request("generate_line", text="hi")

        assert resp == {"id": 1, "success": True}
        # Two requests would get distinct ids
        assert worker._next_request_id() == 2

    @patch("tts.worker.subprocess.Popen")
    def test_shutdown_sends_shutdown_message(self, mock_popen):
        """shutdown() writes a JSON shutdown message to the worker stdin."""
        from tts.worker import EngineWorker

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        with patch.object(EngineWorker, "_find_python", return_value="/usr/bin/python"):
            worker = EngineWorker(Path("/tmp/test-engine"), device="cuda:0")
            worker._process = mock_process
            worker.shutdown()

        msg = json.loads(mock_process.stdin.write.call_args_list[0][0][0])
        assert msg == {"type": "shutdown"}
