"""Tests for the shared worker pool (tts.worker.SharedEngineWorker).

Guards the fix for duplicate engine workers: previously each TTSEngine (and so
each VoiceMapper) spawned its own worker subprocess, so e.g. the seed-clone
mapper and the main voice mapper each loaded a full omni model on the same GPU,
doubling memory and risking OOM. The shared pool makes all TTSEngine instances
for the same (engine_dir, device) reuse a single worker.
"""
import sys
import textwrap
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

_FAKE_MAIN = """
import sys, json, time, os
print(json.dumps({"type": "ready"}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req.get("type") == "shutdown":
        break
    if req.get("type") == "request":
        delay = req.get("kwargs", {}).get("delay", 0)
        if delay:
            time.sleep(delay)
        print(json.dumps({"id": req["id"], "success": True,
                          "pid": os.getpid()}), flush=True)
"""


@pytest.fixture(autouse=True)
def _reset_pool():
    """Reset the module-level shared worker pool between tests for isolation."""
    yield
    from tts.worker import _shared_pool, _shared_pool_lock
    with _shared_pool_lock:
        for sw in list(_shared_pool.values()):
            sw.shutdown()
        _shared_pool.clear()


@pytest.fixture
def run_worker_here():
    """Launch fake workers with the test interpreter (no engine venv needed)."""
    with patch("tts.worker.EngineWorker._find_python", return_value=sys.executable):
        yield


def _make_fake_engine(tmp_path, name="fake_engine"):
    eng = tmp_path / name
    (eng / ".venv" / "bin").mkdir(parents=True)
    (eng / "main.py").write_text(textwrap.dedent(_FAKE_MAIN))
    return eng


class TestSharedEngineWorkerLifecycle:
    """Refcount lifecycle: sharing, release, and shutdown on last release."""

    def test_two_engines_share_one_worker(self, tmp_path, run_worker_here):
        from tts.worker import shared_worker_count
        from tts.engine import TTSEngine

        eng = _make_fake_engine(tmp_path)
        e1 = TTSEngine(eng, "cpu")
        e2 = TTSEngine(eng, "cpu")

        w1 = e1._get_worker()
        w2 = e2._get_worker()

        # Both engines resolved to the SAME underlying worker subprocess.
        assert w1 is w2
        assert e1._get_shared() is e2._get_shared()
        assert shared_worker_count(eng, "cpu") == 1

        e1.shutdown_worker()
        # Still alive: e2 still holds a reference.
        assert e2._get_shared().is_alive
        assert shared_worker_count(eng, "cpu") == 1

        e2.shutdown_worker()
        # Last reference released -> worker shut down.
        assert shared_worker_count(eng, "cpu") == 0

    def test_different_devices_get_separate_workers(self, tmp_path, run_worker_here):
        from tts.worker import shared_worker_count
        from tts.engine import TTSEngine

        eng = _make_fake_engine(tmp_path)
        e1 = TTSEngine(eng, "cuda:0")
        e2 = TTSEngine(eng, "cuda:1")

        w1 = e1._get_worker()
        w2 = e2._get_worker()

        assert w1 is not w2
        assert shared_worker_count(eng, "cuda:0") == 1
        assert shared_worker_count(eng, "cuda:1") == 1
        assert shared_worker_count() == 2

        e1.shutdown_worker()
        e2.shutdown_worker()
        assert shared_worker_count() == 0

    def test_release_below_zero_is_safe(self, tmp_path):
        from tts.worker import release_shared_worker
        eng = _make_fake_engine(tmp_path)
        release_shared_worker(eng, "cpu")  # no-op, never acquired
        assert True


class TestSharedWorkerProtocol:
    """End-to-end: multiple engines use one live subprocess and stay correct."""

    def test_same_pid_across_engines(self, tmp_path, run_worker_here):
        from tts.engine import TTSEngine
        eng = _make_fake_engine(tmp_path)
        e1 = TTSEngine(eng, "cpu")
        e2 = TTSEngine(eng, "cpu")

        r1 = e1.generate_line("hi", None, "/tmp/a.wav")
        r2 = e2.generate_line("yo", None, "/tmp/b.wav")

        assert r1 is True and r2 is True
        pid1 = e1._get_worker()._process.pid
        pid2 = e2._get_worker()._process.pid
        assert pid1 == pid2  # literally the same subprocess

        e1.shutdown_worker()
        # e2 still usable (worker alive via e2's ref).
        r3 = e2.generate_line("still", None, "/tmp/c.wav")
        assert r3 is True
        e2.shutdown_worker()

    def test_concurrent_requests_serialized_correctly(self, tmp_path, run_worker_here):
        """Concurrent requests through two shared engines must all succeed with
        no protocol corruption (serialized by the shared lock)."""
        from tts.engine import TTSEngine
        eng = _make_fake_engine(tmp_path)
        e1 = TTSEngine(eng, "cpu")
        e2 = TTSEngine(eng, "cpu")

        results = []
        errors = []

        def worker_fn(engine, n):
            try:
                ok = engine.generate_line("req", None, f"/tmp/{n}.wav", delay=0.01)
                results.append((n, ok))
            except Exception as e:  # pragma: no cover - fail loudly
                errors.append(e)

        threads = [
            threading.Thread(target=worker_fn, args=(e1 if i % 2 else e2, i))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(results) == 20
        assert all(ok for _, ok in results)

        e1.shutdown_worker()
        e2.shutdown_worker()


class TestWorkerPoolRegistry:
    """The module-level registry is shared across imports (the fix itself)."""

    def test_acquire_release_refcount(self, tmp_path):
        from tts.worker import acquire_shared_worker, release_shared_worker, shared_worker_count
        eng = _make_fake_engine(tmp_path)
        sw = acquire_shared_worker(eng, "cpu")
        assert sw.refcount == 1
        sw2 = acquire_shared_worker(eng, "cpu")
        assert sw2 is sw
        assert sw.refcount == 2
        release_shared_worker(eng, "cpu")
        assert sw.refcount == 1
        release_shared_worker(eng, "cpu")
        assert sw.refcount == 0
        assert shared_worker_count(eng, "cpu") == 0
