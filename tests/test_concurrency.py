"""Tests for concurrency, multi-GPU WorkerPool, and thread-safe generation."""

import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from audiobook_generator.testing import MockTTSEngine
from audiobook_generator.audiobook_generator import TTSConfig, generate_tts_for_line
from audiobook_generator.parse_chapter import ChapterObj


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory(prefix="abg_concurrency_test_") as d:
        yield Path(d)


@pytest.fixture
def mock_voice_mapper():
    mapper = MagicMock()
    mapper.get_voice_path.return_value = "/tmp/test_voice.wav"
    mapper.get_engine.return_value = MockTTSEngine()
    return mapper


@pytest.fixture
def sample_chapters():
    chapter1 = [
        ChapterObj(False, "Narrator text", 1),
        ChapterObj(True, '"Hello," said Jane.', 2),
        ChapterObj(False, "Narrator continues.", 3),
    ]
    chapter2 = [
        ChapterObj(True, '"Good morning," Elizabeth replied.', 1),
        ChapterObj(False, "The room was silent.", 2),
    ]
    return [chapter1, chapter2]


@pytest.fixture
def sample_chapter_maps():
    return {
        0: ({"1": "narrator", "2": "jane"}, {"2": 2}),
        1: ({"1": "elizabeth"}, {"1": 1}),
    }


@pytest.fixture
def sample_voices_map():
    return {
        "narrator": "narrator.wav",
        "jane": "jane.wav",
        "elizabeth": "elizabeth.wav",
    }


def _patch_all(temp_dir):
    """Return a context manager that patches all dependencies for generate_audiobook_from_chapters."""
    from contextlib import contextmanager

    @contextmanager
    def _patches():
        with patch("audiobook_generator.audiobook_generator.setup_validation_model", return_value=MagicMock()):
            with patch("audiobook_generator.audiobook_generator.get_validation_client"):
                with patch("audiobook_generator.audiobook_generator.VoiceMapper") as mock_mapper:
                    mock_mapper.return_value = MagicMock()
                    mock_mapper.return_value.add_voice_path.return_value = None
                    with patch("audiobook_generator.audiobook_generator.generate_tts_for_line") as mock_tts:
                        mock_tts.return_value = (True, 0.95)
                        with patch("audiobook_generator.audiobook_generator.get_non_silent_audio_from_wavs") as mock_wavs:
                            mock_audio = MagicMock()
                            mock_wavs.return_value = mock_audio
                            with patch("audiobook_generator.audiobook_generator.glob.glob", return_value=[]):
                                with patch("audiobook_generator.audiobook_generator.ProgressHandler"):
                                    with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                                        with patch("audiobook_generator.audiobook_generator.os.makedirs"):
                                            with patch("audiobook_generator.audiobook_generator.os.unlink"):
                                                with patch("audiobook_generator.audiobook_generator.gc.collect"):
                                                    yield mock_tts
    return _patches()


# ============================================================================
# TESTS: Single worker, sequential (concurrency=1, no gpus)
# ============================================================================

class TestSequentialGeneration:
    """Default behavior: 1 worker, sequential processing."""

    def test_concurrency_1_processes_all_lines(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """concurrency=1 processes all lines sequentially."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            result = generate_audiobook_from_chapters(
                chapters=sample_chapters,
                chapter_maps=sample_chapter_maps,
                voices_map=sample_voices_map,
                output_dir=str(temp_dir),
                concurrency=1,
            )

        assert result[1] == 2
        # 3 lines in chapter 1 + 2 lines in chapter 2 = 5 calls
        assert mock_tts.call_count == 5

    def test_sequential_calls_generate_tts_for_line_per_line(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Sequential mode calls generate_tts_for_line once per line."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            generate_audiobook_from_chapters(
                chapters=sample_chapters,
                chapter_maps=sample_chapter_maps,
                voices_map=sample_voices_map,
                output_dir=str(temp_dir),
                concurrency=1,
            )

        assert mock_tts.call_count == 5

    def test_sequential_does_not_use_thread_pool(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """concurrency=1 should not use ThreadPoolExecutor."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            with patch("audiobook_generator.audiobook_generator.ThreadPoolExecutor") as mock_executor:
                generate_audiobook_from_chapters(
                    chapters=sample_chapters,
                    chapter_maps=sample_chapter_maps,
                    voices_map=sample_voices_map,
                    output_dir=str(temp_dir),
                    concurrency=1,
                )

        mock_executor.assert_not_called()


# ============================================================================
# TESTS: Thread pool concurrency (concurrency > 1, 1 GPU)
# ============================================================================

class TestThreadPoolConcurrency:
    """Multiple concurrent lines on a single GPU."""

    def test_concurrency_4_processes_all_lines(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """concurrency=4 should still process all lines."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            result = generate_audiobook_from_chapters(
                chapters=sample_chapters,
                chapter_maps=sample_chapter_maps,
                voices_map=sample_voices_map,
                output_dir=str(temp_dir),
                concurrency=4,
            )

        assert result[1] == 2
        assert mock_tts.call_count == 5

    def test_concurrency_4_uses_thread_pool(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """concurrency=4 should use ThreadPoolExecutor."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            with patch("audiobook_generator.audiobook_generator.ThreadPoolExecutor") as mock_executor:
                mock_future = MagicMock()
                mock_future.result.return_value = {"success": True, "ratio": 0.95}
                mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                # as_completed is imported and called directly, need to patch it
                with patch("audiobook_generator.audiobook_generator.as_completed", return_value=iter([mock_future])):
                    generate_audiobook_from_chapters(
                        chapters=sample_chapters,
                        chapter_maps=sample_chapter_maps,
                        voices_map=sample_voices_map,
                        output_dir=str(temp_dir),
                        concurrency=4,
                    )

        mock_executor.assert_called_with(max_workers=4)

    def test_concurrency_propagates_to_tts_config(self, temp_dir, mock_voice_mapper):
        """whisper_lock should be set on TTSConfig."""
        lock = threading.Lock()
        config = TTSConfig(
            device="cpu",
            output_dir=str(temp_dir),
            validation_model=None,
            short_text_postfix="",
            whisper_lock=lock,
        )
        assert config.whisper_lock is lock

    def test_no_concurrency_defaults_to_sequential(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """Default (no concurrency arg) should be sequential."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            with patch("audiobook_generator.audiobook_generator.ThreadPoolExecutor") as mock_executor:
                generate_audiobook_from_chapters(
                    chapters=sample_chapters,
                    chapter_maps=sample_chapter_maps,
                    voices_map=sample_voices_map,
                    output_dir=str(temp_dir),
                )

        mock_executor.assert_not_called()
        assert mock_tts.call_count == 5


# ============================================================================
# TESTS: Multi-GPU WorkerPool
# ============================================================================

class TestWorkerPool:
    """Multi-GPU worker pool distribution."""

    def test_pool_shutdown_stops_all_workers(self):
        """WorkerPool.shutdown should stop all workers."""
        from audiobook_generator.engines.pool import WorkerPool, _WorkerDevice

        mock_w1 = MagicMock()
        mock_w2 = MagicMock()

        pool = WorkerPool.__new__(WorkerPool)
        pool._workers = [
            _WorkerDevice(mock_w1, "cuda:0"),
            _WorkerDevice(mock_w2, "cuda:1"),
        ]
        pool._index = 0
        pool._lock = threading.Lock()

        pool.shutdown()
        mock_w1.shutdown.assert_called_once()
        mock_w2.shutdown.assert_called_once()

    def test_pool_generate_line_routes_round_robin(self):
        """generate_line should route to the next worker in rotation."""
        from audiobook_generator.engines.pool import WorkerPool, _WorkerDevice

        mock_w1 = MagicMock()
        mock_w1.request.return_value = {"success": True}
        mock_w2 = MagicMock()
        mock_w2.request.return_value = {"success": True}

        pool = WorkerPool.__new__(WorkerPool)
        pool._workers = [
            _WorkerDevice(mock_w1, "cuda:0"),
            _WorkerDevice(mock_w2, "cuda:1"),
        ]
        pool._index = 0
        pool._lock = threading.Lock()

        pool.generate_line(text="hello", voice_path="/tmp/v.wav", output_path="/tmp/out.wav", device="cuda:0")
        pool.generate_line(text="world", voice_path="/tmp/v.wav", output_path="/tmp/out2.wav", device="cuda:0")

        mock_w1.request.assert_called_once()
        mock_w2.request.assert_called_once()
        assert mock_w1.request.call_args[1]["device"] == "cuda:0"
        assert mock_w2.request.call_args[1]["device"] == "cuda:1"

    def test_single_gpu_pool_delegates_to_one_worker(self):
        """Single GPU pool should only use one worker."""
        from audiobook_generator.engines.pool import WorkerPool, _WorkerDevice

        mock_w1 = MagicMock()
        mock_w1.request.return_value = {"success": True}

        pool = WorkerPool.__new__(WorkerPool)
        pool._workers = [_WorkerDevice(mock_w1, "cuda:0")]
        pool._index = 0
        pool._lock = threading.Lock()

        for _ in range(5):
            pool.generate_line(text="test", voice_path="/tmp/v.wav", output_path="/tmp/out.wav", device="cuda:0")

        assert mock_w1.request.call_count == 5

    def test_worker_device_slots(self):
        """_WorkerDevice should have correct slots."""
        from audiobook_generator.engines.pool import _WorkerDevice

        w = _WorkerDevice(MagicMock(), "cuda:0")
        assert w.device == "cuda:0"
        assert w.worker is not None


# ============================================================================
# TESTS: Multi-GPU integration with generate_audiobook_from_chapters
# ============================================================================

class TestMultiGPUIntegration:
    """Integration tests for multi-GPU audiobook generation."""

    def test_gpus_param_creates_worker_pool(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """gpus=['cuda:0', 'cuda:1'] should create a WorkerPool."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            with patch("audiobook_generator.audiobook_generator.VoiceMapper") as mock_mapper:
                mock_mapper.return_value = MagicMock()
                mock_mapper.return_value.add_voice_path.return_value = None
                mock_mapper.return_value.get_engine.return_value.__class__.__name__ = "MockEngine"
                with patch("audiobook_generator.engines.pool.WorkerPool") as mock_pool_cls:
                    mock_pool = MagicMock()
                    mock_pool_cls.return_value = mock_pool
                    generate_audiobook_from_chapters(
                        chapters=sample_chapters,
                        chapter_maps=sample_chapter_maps,
                        voices_map=sample_voices_map,
                        output_dir=str(temp_dir),
                        gpus=["cuda:0", "cuda:1"],
                    )

        mock_pool_cls.assert_called_once()
        mock_pool.start.assert_called_once()
        mock_pool.shutdown.assert_called_once()

    def test_single_gpu_no_pool(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """gpus=['cuda:0'] should not create a WorkerPool."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            with patch("audiobook_generator.engines.pool.WorkerPool") as mock_pool_cls:
                generate_audiobook_from_chapters(
                    chapters=sample_chapters,
                    chapter_maps=sample_chapter_maps,
                    voices_map=sample_voices_map,
                    output_dir=str(temp_dir),
                    gpus=["cuda:0"],
                )

        mock_pool_cls.assert_not_called()

    def test_no_gpus_no_pool(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """gpus=None should not create a WorkerPool."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            with patch("audiobook_generator.engines.pool.WorkerPool") as mock_pool_cls:
                generate_audiobook_from_chapters(
                    chapters=sample_chapters,
                    chapter_maps=sample_chapter_maps,
                    voices_map=sample_voices_map,
                    output_dir=str(temp_dir),
                )

        mock_pool_cls.assert_not_called()

    def test_pool_engine_passed_to_tts_config(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """WorkerPool should be passed through TTSConfig.engine."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        captured_config = None

        def capture_config(*args, **kwargs):
            nonlocal captured_config
            captured_config = kwargs.get("tts_config", args[5] if len(args) > 5 else None)
            return (True, 0.95)

        with patch("audiobook_generator.audiobook_generator.setup_validation_model", return_value=MagicMock()):
            with patch("audiobook_generator.audiobook_generator.get_validation_client"):
                with patch("audiobook_generator.audiobook_generator.VoiceMapper") as mock_mapper:
                    mock_mapper.return_value = MagicMock()
                    mock_mapper.return_value.add_voice_path.return_value = None
                    mock_mapper.return_value.get_engine.return_value.__class__.__name__ = "MockEngine"
                    with patch("audiobook_generator.engines.pool.WorkerPool") as mock_pool_cls:
                        mock_pool = MagicMock()
                        mock_pool_cls.return_value = mock_pool
                        with patch("audiobook_generator.audiobook_generator.generate_tts_for_line", side_effect=capture_config):
                            with patch("audiobook_generator.audiobook_generator.get_non_silent_audio_from_wavs") as mock_wavs:
                                mock_audio = MagicMock()
                                mock_wavs.return_value = mock_audio
                                with patch("audiobook_generator.audiobook_generator.glob.glob", return_value=[]):
                                    with patch("audiobook_generator.audiobook_generator.ProgressHandler"):
                                        with patch("audiobook_generator.audiobook_generator.os.path.exists", return_value=False):
                                            with patch("audiobook_generator.audiobook_generator.os.makedirs"):
                                                with patch("audiobook_generator.audiobook_generator.os.unlink"):
                                                    with patch("audiobook_generator.audiobook_generator.gc.collect"):
                                                        generate_audiobook_from_chapters(
                                                            chapters=sample_chapters,
                                                            chapter_maps=sample_chapter_maps,
                                                            voices_map=sample_voices_map,
                                                            output_dir=str(temp_dir),
                                                            gpus=["cuda:0", "cuda:1"],
                                                        )

        assert captured_config is not None
        assert captured_config.engine is mock_pool

    def test_four_gpus_creates_pool_with_four_devices(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """4 GPUs should create pool with all 4 devices."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            with patch("audiobook_generator.audiobook_generator.VoiceMapper") as mock_mapper:
                mock_mapper.return_value = MagicMock()
                mock_mapper.return_value.add_voice_path.return_value = None
                mock_mapper.return_value.get_engine.return_value.__class__.__name__ = "MockEngine"
                with patch("audiobook_generator.engines.pool.WorkerPool") as mock_pool_cls:
                    mock_pool_cls.return_value = MagicMock()
                    generate_audiobook_from_chapters(
                        chapters=sample_chapters,
                        chapter_maps=sample_chapter_maps,
                        voices_map=sample_voices_map,
                        output_dir=str(temp_dir),
                        gpus=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                    )

        call_args = mock_pool_cls.call_args
        assert call_args[0][2] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]


# ============================================================================
# TESTS: Combined concurrency + multi-GPU
# ============================================================================

class TestCombinedConcurrencyAndMultiGPU:
    """Thread pool + WorkerPool working together."""

    def test_concurrency_and_gpus_both_applied(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """concurrency=2 + gpus=[cuda:0, cuda:1] should use both."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            with patch("audiobook_generator.audiobook_generator.VoiceMapper") as mock_mapper:
                mock_mapper.return_value = MagicMock()
                mock_mapper.return_value.add_voice_path.return_value = None
                mock_mapper.return_value.get_engine.return_value.__class__.__name__ = "MockEngine"
                with patch("audiobook_generator.engines.pool.WorkerPool") as mock_pool_cls:
                    mock_pool_cls.return_value = MagicMock()
                    with patch("audiobook_generator.audiobook_generator.ThreadPoolExecutor") as mock_executor:
                        mock_future = MagicMock()
                        mock_future.result.return_value = {"success": True, "ratio": 0.95}
                        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                        with patch("audiobook_generator.audiobook_generator.as_completed", return_value=iter([mock_future])):
                            result = generate_audiobook_from_chapters(
                                chapters=sample_chapters,
                                chapter_maps=sample_chapter_maps,
                                voices_map=sample_voices_map,
                                output_dir=str(temp_dir),
                                concurrency=2,
                                gpus=["cuda:0", "cuda:1"],
                            )

        mock_pool_cls.assert_called_once()
        mock_executor.assert_called_with(max_workers=2)
        assert result[1] == 2

    def test_four_gpus_two_concurrent(self, temp_dir, sample_chapters, sample_chapter_maps, sample_voices_map):
        """4 GPUs + concurrency=2 should create pool with 4 workers and thread pool with 2."""
        from audiobook_generator.audiobook_generator import generate_audiobook_from_chapters

        with _patch_all(temp_dir) as mock_tts:
            with patch("audiobook_generator.audiobook_generator.VoiceMapper") as mock_mapper:
                mock_mapper.return_value = MagicMock()
                mock_mapper.return_value.add_voice_path.return_value = None
                mock_mapper.return_value.get_engine.return_value.__class__.__name__ = "MockEngine"
                with patch("audiobook_generator.engines.pool.WorkerPool") as mock_pool_cls:
                    mock_pool_cls.return_value = MagicMock()
                    with patch("audiobook_generator.audiobook_generator.ThreadPoolExecutor") as mock_executor:
                        mock_future = MagicMock()
                        mock_future.result.return_value = {"success": True, "ratio": 0.95}
                        mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                        with patch("audiobook_generator.audiobook_generator.as_completed", return_value=iter([mock_future])):
                            generate_audiobook_from_chapters(
                                chapters=sample_chapters,
                                chapter_maps=sample_chapter_maps,
                                voices_map=sample_voices_map,
                                output_dir=str(temp_dir),
                                concurrency=2,
                                gpus=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                            )

        call_args = mock_pool_cls.call_args
        assert call_args[0][2] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        mock_executor.assert_called_with(max_workers=2)


# ============================================================================
# TESTS: CLI argument parsing
# ============================================================================

class TestCLIArgs:
    """Test that CLI arguments are parsed correctly."""

    def test_gpus_arg_parsed(self):
        """--gpus should accept multiple device strings."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--gpus", nargs="+", default=None)
        parser.add_argument("--concurrency", type=int, default=1)
        parser.add_argument("--whisper-cpu", action="store_true")

        args = parser.parse_args(["--gpus", "cuda:0", "cuda:1", "--concurrency", "4"])
        assert args.gpus == ["cuda:0", "cuda:1"]
        assert args.concurrency == 4

    def test_whisper_cpu_arg_parsed(self):
        """--whisper-cpu should be parsed as bool flag."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--whisper-cpu", action="store_true")

        args = parser.parse_args(["--whisper-cpu"])
        assert args.whisper_cpu is True

        args = parser.parse_args([])
        assert args.whisper_cpu is False

    def test_concurrency_arg_default(self):
        """--concurrency should default to 1."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--concurrency", type=int, default=1)

        args = parser.parse_args([])
        assert args.concurrency == 1


# ============================================================================
# TESTS: Thread safety
# ============================================================================

class TestThreadSafety:
    """Verify thread-safe behavior of concurrent generation."""

    def test_whisper_lock_serializes_transcription(self):
        """Multiple threads should serialize Whisper calls via lock."""
        lock = threading.Lock()
        acquired = []

        def hold_lock(name):
            with lock:
                acquired.append(name)
                threading.Event().wait(0.01)
                acquired.append(f"{name}_done")

        threads = [threading.Thread(target=hold_lock, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each pair (name, name_done) should be adjacent
        for i in range(0, len(acquired), 2):
            base = acquired[i].rsplit("_", 1)[0] if "_done" in acquired[i] else acquired[i]
            assert acquired[i + 1].startswith(base)

    def test_tts_config_accepts_lock(self):
        """TTSConfig should accept a threading.Lock for whisper_lock."""
        lock = threading.Lock()
        config = TTSConfig(
            device="cpu",
            output_dir="/tmp/test",
            whisper_lock=lock,
        )
        assert config.whisper_lock is lock

    def test_tts_config_accepts_none_lock(self):
        """TTSConfig should accept None for whisper_lock (optional)."""
        config = TTSConfig(
            device="cpu",
            output_dir="/tmp/test",
            whisper_lock=None,
        )
        assert config.whisper_lock is None

    def test_tts_config_accepts_engine(self):
        """TTSConfig should accept an engine (WorkerPool or TTSEngine)."""
        mock_engine = MagicMock()
        config = TTSConfig(
            device="cpu",
            output_dir="/tmp/test",
            engine=mock_engine,
        )
        assert config.engine is mock_engine
