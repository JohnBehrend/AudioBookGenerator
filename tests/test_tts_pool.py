"""Unit tests for tts.pool module."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestWorkerPool:
    """Test WorkerPool round-robin distribution."""

    def test_init(self):
        """Test WorkerPool initialization."""
        from tts.pool import WorkerPool
        
        engine_dir = Path("/tmp/test-engine")
        devices = ["cuda:0", "cuda:1"]
        
        pool = WorkerPool(engine_dir, devices)
        
        assert pool.engine_dir == engine_dir
        assert pool.devices == devices
        assert pool._workers == []
        assert pool._index == 0

    def test_next_worker(self):
        """Test round-robin worker selection."""
        from tts.pool import WorkerPool, _WorkerDevice
        
        engine_dir = Path("/tmp/test-engine")
        devices = ["cuda:0", "cuda:1"]
        
        pool = WorkerPool(engine_dir, devices)
        
        # Create mock workers
        mock_worker1 = MagicMock()
        mock_worker2 = MagicMock()
        pool._workers = [
            _WorkerDevice(mock_worker1, "cuda:0"),
            _WorkerDevice(mock_worker2, "cuda:1"),
        ]
        
        # Test round-robin
        w1 = pool._next_worker()
        w2 = pool._next_worker()
        w3 = pool._next_worker()
        
        assert w1.device == "cuda:0"
        assert w2.device == "cuda:1"
        assert w3.device == "cuda:0"  # Back to first

    def test_shutdown(self):
        """Test shutdown of all workers."""
        from tts.pool import WorkerPool, _WorkerDevice
        
        engine_dir = Path("/tmp/test-engine")
        devices = ["cuda:0", "cuda:1"]
        
        pool = WorkerPool(engine_dir, devices)
        
        # Create mock workers
        mock_worker1 = MagicMock()
        mock_worker2 = MagicMock()
        pool._workers = [
            _WorkerDevice(mock_worker1, "cuda:0"),
            _WorkerDevice(mock_worker2, "cuda:1"),
        ]
        
        pool.shutdown()
        
        mock_worker1.shutdown.assert_called_once()
        mock_worker2.shutdown.assert_called_once()
        assert pool._workers == []

    def test_context_manager(self):
        """Test context manager usage."""
        from tts.pool import WorkerPool
        
        engine_dir = Path("/tmp/test-engine")
        devices = ["cuda:0"]
        
        pool = WorkerPool(engine_dir, devices)
        
        with patch.object(pool, 'start') as mock_start, \
             patch.object(pool, 'shutdown') as mock_shutdown:
            
            with pool as p:
                assert p == pool
                mock_start.assert_called_once()
            
            mock_shutdown.assert_called_once()


class TestWhisperPool:
    """Test WhisperPool for parallel validation."""

    def test_init(self):
        """Test WhisperPool initialization."""
        from tts.pool import WhisperPool
        
        # Mock model factory - must accept optional device arg
        model_factory = lambda device=None: MagicMock()
        
        pool = WhisperPool(model_factory, size=2)
        
        assert pool._size == 2
        assert len(pool._models) == 2
        assert len(pool._locks) == 2

    def test_transcribe_round_robin(self):
        """Test round-robin transcription routing."""
        from tts.pool import WhisperPool
        
        # Create mock models
        mock_model1 = MagicMock()
        mock_model1.transcribe.return_value = ("text1", 0.9, [])
        mock_model2 = MagicMock()
        mock_model2.transcribe.return_value = ("text2", 0.8, [])
        
        model_factory = lambda device=None: [mock_model1, mock_model2][0]
        
        pool = WhisperPool(model_factory, size=2)
        pool._models = [mock_model1, mock_model2]
        
        # Test round-robin
        result1 = pool.transcribe("audio1.wav")
        result2 = pool.transcribe("audio2.wav")
        
        mock_model1.transcribe.assert_called_once()
        mock_model2.transcribe.assert_called_once()

    def test_context_manager(self):
        """Test context manager usage."""
        from tts.pool import WhisperPool
        
        model_factory = lambda device=None: MagicMock()
        pool = WhisperPool(model_factory, size=1)
        
        with pool as p:
            assert p == pool


class TestWorkerDevice:
    """Test _WorkerDevice helper class."""

    def test_init(self):
        """Test _WorkerDevice initialization."""
        from tts.pool import _WorkerDevice
        
        mock_worker = MagicMock()
        device = "cuda:0"
        
        wd = _WorkerDevice(mock_worker, device)
        
        assert wd.worker == mock_worker
        assert wd.device == device

    def test_slots(self):
        """Test that _WorkerDevice uses __slots__."""
        from tts.pool import _WorkerDevice
        
        assert hasattr(_WorkerDevice, '__slots__')
        assert 'worker' in _WorkerDevice.__slots__
        assert 'device' in _WorkerDevice.__slots__
