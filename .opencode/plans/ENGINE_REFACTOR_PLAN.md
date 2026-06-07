# TTS Engine Isolation - Refactoring Plan

## Goal

Make each TTS engine a fully isolated, standalone package with its own `uv` venv. The main project communicates via JSON-over-stdin/stdout. Create a top-level `tts/` package that abstracts TTS from the audiobook. Omni engine implemented first as canary.

## Current Problems

1. Engine code lives inside `audiobook_generator.engines` — not truly isolated
2. Engine modules import from parent package (`from ..config import DEFAULTS`)
3. Vendored source trees (`dramabox/`, `echo_tts/`, `miso_tts/src/`) are hard to maintain
4. Main `pyproject.toml` lists engine-specific deps (`omnivoice`, `chunkformer`)
5. Registration requires editing `__init__.py` to add new engines
6. `VoiceMapper` has Omni-specific in-process model logic that breaks isolation

## Git Repos

| Engine | Git Repo | HF Model |
|--------|----------|----------|
| omni | `https://github.com/k2-fsa/OmniVoice` | `drbaph/OmniVoice-bf16` |
| moss | `https://github.com/OpenMOSS/MOSS-TTS` | `OpenMOSS-Team/MOSS-TTS-Local-Transformer` |
| vox | `https://github.com/OpenBMB/VoxCPM` | `openbmb/VoxCPM2` |
| echo-tts | `https://github.com/jordandare/echo-tts.git` | `jordand/echo-tts-base` |
| miso-tts | `https://github.com/MisoLabsAI/MisoTTS` | `MisoLabs/MisoTTS` |
| dramabox | `https://github.com/resemble-ai/Dramabox.git` | `ResembleAI/Dramabox` |

## Target Structure

```
AudioBookGenerator/
├── pyproject.toml                    # NO engine-specific deps
├── tts/                              # NEW: Top-level TTS submodule
│   ├── __init__.py                   # Public API: TTSEngine, list_engines, list_voice_engines
│   ├── engine.py                     # Base class for TTS engines
│   ├── worker.py                     # Engine-agnostic subprocess manager
│   ├── pool.py                       # WorkerPool, WhisperPool
│   └── voice_sample.py               # Voice sample generation (moved from voice_mapper.py)
├── audiobook_generator/              # Main audiobook code
│   ├── config.py                     # Pipeline-level config only (static_voice_text stays here)
│   ├── voice_mapper.py               # Updated to use tts/ submodule
│   ├── audiobook_generator.py        # Stage 5 (minimal changes)
│   ├── generate_voice_samples.py     # Stage 4 (minimal changes)
│   ├── parse_chapter.py
│   ├── llm_label_speakers.py
│   ├── llm_describe_character.py
│   ├── pipeline.py
│   ├── audio.py
│   ├── utils.py
│   └── gradio_ui.py
├── engines/                          # Standalone engine packages
│   ├── omni/
│   │   ├── pyproject.toml
│   │   └── main.py
│   ├── moss/                         # (future)
│   ├── vox/                          # (future)
│   ├── echo-tts/                     # (future)
│   ├── miso-tts/                     # (future)
│   └── dramabox/                     # (future)
├── scripts/
│   └── setup-engines.py              # Pre-builds all engine venvs
├── tests/                            # KEPT! Unit tests + new tts tests
│   ├── test_parse_chapter.py         # No engine dependency, kept
│   ├── test_pipeline.py              # No engine dependency, kept
│   ├── test_utils.py                 # No engine dependency, kept
│   ├── test_llm_label_speakers.py    # No engine dependency, kept
│   ├── test_llm_describe_character.py# No engine dependency, kept
│   ├── test_audio_quality.py         # No engine dependency, kept
│   ├── test_generate_audiobook_from_chapters.py # No engine dependency, kept
│   ├── test_voice_mapper.py          # Updated to use tts/, may need updates
│   ├── test_generate_voice_samples.py # Updated to use tts/, may need updates
│   ├── test_engines.py               # Deprecated, replaced by tts/ unit tests
│   ├── test_real_engines.py          # Kept for integration testing
│   └── ...                           # Other tests
└── voice_test/
```

## Universal Protocol

### Wire Format

JSON lines over stdin/stdout.

**Request:**
```json
{"type": "request", "id": N, "method": "generate_line", "kwargs": {"text": "...", "voice_path": "...", "output_path": "..."}}
```

**Response:**
```json
{"id": N, "success": true}
{"id": N, "success": false}
{"id": N, "error": "message", "traceback": "..."}
```

**Startup:** `{"type": "ready"}`
**Shutdown:** `{"type": "shutdown"}`

### Methods

#### `generate_line`
- **Required:** `text`, `voice_path`, `output_path`
- **Optional:** `ref_text` (computed by main process via Whisper, passed to engines that need it)
- **Returns:** `{"id": N, "success": true/false}`

#### `generate_voice_sample`
- **Required:** `character_name`, `description`, `output_dir`
- **Returns:** `{"id": N, "success": true, "output_file": "...", "duration": 1.5}` or `{"id": N, "success": false}`

#### `--probe` (CLI flag)
- Returns engine capabilities as JSON to stdout:
```json
{
  "name": "omni",
  "methods": ["generate_line", "generate_voice_sample"],
  "sample_rate": 24000
}
```

### Engine-Specific Parameters

All engine-specific parameters (temperature, cfg_scale, top_p, top_k, repetition_penalty, etc.) stay **internal to each engine's `main.py`**. The universal API does not surface them. Each engine ships with tuned defaults.

## Implementation Plan

### Phase 1: Infrastructure

**1.1. Create `tts/` directory at repo root**

**1.2. Create `tts/__init__.py`**

Public API package:

```python
"""TTS submodule for AudioBook Generator.

Provides:
- TTSEngine: Base class for TTS engines
- list_engines(): List available engines
- list_voice_engines(): List engines supporting voice design
- get_engine(): Get an engine instance
- WorkerPool: Multi-GPU worker pool
"""

from .engine import TTSEngine
from .worker import EngineWorker
from .pool import WorkerPool
from .voice_sample import generate_voice_sample, build_voice_clone_prompt

import json
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENGINES_DIR = _REPO_ROOT / "engines"

def list_engines() -> list[str]:
    """List available engine names."""
    return sorted(
        d.name for d in _ENGINES_DIR.iterdir()
        if d.is_dir() and (d / "main.py").exists()
    )

def get_engine_dir(name: str) -> Path:
    """Get the directory for a named engine."""
    d = _ENGINES_DIR / name
    if not d.is_dir() or not (d / "main.py").exists():
        raise ValueError(f"Unknown engine: {name}. Available: {list_engines()}")
    return d

def get_engine_capabilities(name: str) -> dict:
    """Probe engine capabilities by running --probe."""
    engine_dir = get_engine_dir(name)
    result = subprocess.run(
        ["python", str(engine_dir / "main.py"), "--probe"],
        capture_output=True, text=True, cwd=str(engine_dir)
    )
    if result.returncode != 0:
        raise RuntimeError(f"Engine {name} --probe failed: {result.stderr}")
    return json.loads(result.stdout)

def list_voice_engines() -> list[str]:
    """List engines that support voice sample generation."""
    return [
        name for name in list_engines()
        if "generate_voice_sample" in get_engine_capabilities(name).get("methods", [])
    ]

def get_engine(engine_name: str, device: str = "cuda") -> TTSEngine:
    """Get a TTS engine instance by name.

    Args:
        engine_name: Engine identifier (e.g., 'moss', 'omni', 'vox', 'echo-tts', 'dramabox')
        device: CUDA device string (e.g., 'cuda:0')

    Returns:
        A TTSEngine instance.

    Raises:
        ValueError: If engine_name is not registered.
    """
    engine_dir = get_engine_dir(engine_name)
    return TTSEngine(engine_dir, device=device)
```

**1.3. Create `tts/engine.py`**

Base class for TTS engines (no engine-specific imports):

```python
"""Base class for TTS engines."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple

import torch


class TTSEngine(ABC):
    """Abstract base class for TTS engines.

    Each engine implementation handles:
    - Loading its model(s) from HuggingFace or local sources
    - Generating voice samples from character descriptions (Stage 4)
    - Generating audio lines from text + voice reference (Stage 5)

    Engines run TTS inference in isolated subprocess workers. The adapter
    methods delegate to EngineWorker, while _run_worker() handles the actual
    model loading and inference in the subprocess.
    """

    def __init__(self, engine_dir: Path, device: str = "cuda"):
        self.engine_dir = engine_dir
        self.device = device

    @abstractmethod
    def generate_line(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        verbose: bool = False,
        ref_text: Optional[str] = None,
    ) -> bool:
        """Generate audio for a single line (Stage 5)."""
        ...

    @abstractmethod
    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: Path,
        verbose: bool = False,
    ) -> Tuple[bool, Optional[str], float]:
        """Generate a voice sample for a character (Stage 4)."""
        ...

    @staticmethod
    def _clear_cuda_cache() -> None:
        """Clear CUDA memory after worker calls."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

**1.4. Rewrite `tts/worker.py`**

Engine-agnostic subprocess manager. No engine class references.

```python
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
```

**1.5. Create `tts/pool.py`**

Multi-GPU worker pool (same as current `pool.py` but imports from `tts`):

```python
"""Multi-GPU worker pool for TTS engines.

A WorkerPool manages one EngineWorker subprocess per GPU and distributes
requests round-robin across them. Each worker loads its own model on its
own GPU, providing transparent multi-GPU scaling.
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional
from pathlib import Path

from .worker import EngineWorker


class WhisperPool:
    """Pool of Whisper models for parallel validation.

    Each model has its own lock, so N models allow N concurrent transcriptions.
    Requests are distributed round-robin across the pool. Models can be
    distributed across multiple devices for balanced GPU utilization.
    """

    def __init__(self, model_factory, size: int, devices: Optional[List[str]] = None):
        self._size = size
        self._models: List[Any] = []
        self._locks: List[threading.Lock] = []
        self._index = 0
        self._global_lock = threading.Lock()

        for i in range(size):
            if devices:
                device = devices[i % len(devices)]
                model = model_factory(device)
            else:
                model = model_factory()
            self._models.append(model)
            self._locks.append(threading.Lock())

    def transcribe(self, audio_path: str, **kwargs) -> tuple:
        """Transcribe audio, routing to next model in round-robin order."""
        with self._global_lock:
            idx = self._index % self._size
            self._index += 1
        lock = self._locks[idx]
        model = self._models[idx]
        with lock:
            result = model.transcribe(audio_path, **kwargs)
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class WorkerPool:
    """Round-robin pool of TTS engine workers across multiple GPUs.

    Each worker runs its own model on its own GPU. Requests are distributed
    round-robin so all GPUs stay busy. The pool presents the same interface
    as a single TTSEngine for drop-in replacement.
    """

    def __init__(self, engine_dir: Path, devices: List[str]):
        self.engine_dir = engine_dir
        self.devices = devices
        self._workers: List["_WorkerDevice"] = []
        self._index = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start all worker subprocesses."""
        for device in self.devices:
            worker = EngineWorker(self.engine_dir, device)
            worker.start()
            self._workers.append(_WorkerDevice(worker, device))

    def _next_worker(self) -> _WorkerDevice:
        """Get the next worker in round-robin order."""
        with self._lock:
            w = self._workers[self._index % len(self._workers)]
            self._index += 1
            return w

    def generate_line(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        verbose: bool = False,
        ref_text: Optional[str] = None,
    ) -> bool:
        """Generate audio for a single line, routing to next worker."""
        w = self._next_worker()
        resp = w.worker.request(
            "generate_line",
            text=text,
            voice_path=voice_path,
            output_path=output_path,
            device=w.device,
        )
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp.get("success", True)

    def shutdown(self) -> None:
        """Shutdown all worker subprocesses."""
        for w in self._workers:
            w.worker.shutdown()
        self._workers.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.shutdown()


class _WorkerDevice:
    """Bundles an EngineWorker with its target GPU device."""

    __slots__ = ("worker", "device")

    def __init__(self, worker: EngineWorker, device: str):
        self.worker = worker
        self.device = device
```

**1.6. Create `tts/voice_sample.py`**

Voice sample generation logic (moved from `voice_mapper.py`):

```python
"""Voice sample generation for TTS engines.

This module provides functions for generating voice samples using TTS engines.
Moved from voice_mapper.py to separate TTS concerns from audiobook concerns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from .worker import EngineWorker


def generate_voice_sample(
    engine_dir: Path,
    device: str,
    character_name: str,
    description: str,
    output_dir: Path,
    verbose: bool = False,
) -> Tuple[bool, Optional[str], float]:
    """Generate a voice sample for a character using the configured TTS engine.

    Args:
        engine_dir: Path to the engine directory
        device: CUDA device string
        character_name: Name of the character
        description: Voice description from LLM
        output_dir: Output directory for the voice sample
        verbose: Print verbose output

    Returns:
        Tuple of (success, output_file_path, duration_seconds)
    """
    if not description or not description.strip():
        if verbose:
            print(f"  ERROR: Skipping '{character_name}' due to empty description")
        return False, None, 0

    worker = EngineWorker(engine_dir, device)
    try:
        worker.start()
        resp = worker.request(
            "generate_voice_sample",
            character_name=character_name,
            description=description,
            output_dir=str(output_dir),
            device=device,
        )
        success = resp.get("success", False)
        return success, resp.get("output_file"), resp.get("duration", 0)
    finally:
        worker.shutdown()


def build_voice_clone_prompt(
    engine_dir: Path,
    device: str,
    voice_path: str,
    ref_text: Optional[str] = None,
    auto_transcribe: bool = False,
    verbose: bool = False,
) -> Any:
    """Build a voice_clone_prompt for voice cloning.

    Args:
        engine_dir: Path to the engine directory
        device: CUDA device string
        voice_path: Path to the voice sample file
        ref_text: Reference text for voice cloning
        auto_transcribe: If True, transcribe the audio to get ref_text
        verbose: Print verbose output

    Returns:
        A voice_clone_prompt that can be reused for generate_line calls
    """
    if ref_text is None:
        ref_text = ""

    # Auto-transcribe if requested
    if auto_transcribe:
        try:
            from ..utils import transcribe_audio_with_whisper
            actual_ref_text, _, _ = transcribe_audio_with_whisper(None, voice_path)
            if verbose:
                print(f"  Transcribed ref_text: {actual_ref_text}")
            ref_text = actual_ref_text
        except Exception as e:
            if verbose:
                print(f"  Warning: auto_transcribe failed: {e}")

    import soundfile as sf
    import torch

    voice_audio, sr = sf.read(voice_path)
    voice_audio = torch.from_numpy(voice_audio)

    worker = EngineWorker(engine_dir, device)
    try:
        worker.start()
        resp = worker.request(
            "build_voice_clone_prompt",
            voice_path=voice_path,
            ref_text=ref_text,
            device=device,
        )
        return resp.get("voice_clone_prompt")
    finally:
        worker.shutdown()
```

### Phase 2: Omni Engine

**2.1. Create `engines/omni/pyproject.toml`**

```toml
[project]
name = "omni-engine"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch>=2.9.1",
    "torchaudio>=2.9.1",
    "transformers>=5.0.0",
    "omnivoice @ git+https://github.com/k2-fsa/OmniVoice.git",
    "soundfile",
]

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
torchaudio = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

**2.2. Create `engines/omni/main.py`**

Self-contained script. No imports from `audiobook_generator`.

Sections:
- `probe()` — prints capabilities JSON
- `convert_description_to_instruct()` — moved from `omni.py`
- `get_fallback_instruct()` — moved from `omni.py`
- `main()` — argparse (`--device`, `--probe`), model loading, JSON request loop

Constants hardcoded:
- `sample_rate = 24000`
- `model_path = "drbaph/OmniVoice-bf16"`
- `num_step = 32`, `class_temperature = 0.5` (voice design)
- `class_temperature_fallback = 3.0` (fallback)

`static_voice_text` is passed as kwarg in request kwargs from main process (global in `config.py`).

Voice clone prompt caching: internal `_voice_clone_prompts` dict, same as current.

### Phase 3: Integration

**3.1. Update `audiobook_generator/voice_mapper.py`**

Remove:
- `setup_tts_engine()` — worker handles model loading
- `build_voice_clone_prompt()` — moved to `tts/voice_sample.py`
- `get_voice_clone_prompt()` — moved to `tts/voice_sample.py`
- `get_all_clone_prompts()` — moved to `tts/voice_sample.py`
- `voice_clone_prompts` cache — moved to `tts/voice_sample.py`
- `_get_model_path()` — uses `TTS_MODEL_PATHS`, dead code
- `unload_model()` — no longer needed
- `cleanup_tts_models()` — no longer needed
- `tts_models` dict — dead code

Update:
- `get_engine()` → creates `TTSEngine(get_engine_dir(self.tts_engine), self.device)`
- `get_pool()` → passes `engine_dir` to `WorkerPool`
- `cleanup_engines()` → calls `worker.shutdown()`

Keep:
- Voice path lookup (`get_voice_path`, `add_voice_path`, etc.)
- Voice sample generation (`generate_voice_sample`) — now delegates to `tts/voice_sample.py`
- LLM validation (`validate_voice_with_llm`, `describe_voice_with_llm`)
- Voice map persistence

**3.2. Update `audiobook_generator/config.py`**

Remove:
- `TTS_MODEL_PATHS` — each engine knows its model
- Engine-specific defaults: `moss_voicegen_temperature`, `moss_audio_temperature`, `moss_voicegen_top_p`, `moss_voicegen_top_k`, `moss_voicegen_repetition_penalty`, `moss_audio_top_p`, `moss_audio_top_k`, `moss_audio_repetition_penalty`
- `DEFAULTS["max_new_tokens"]` — move to each engine's `main.py`
- `DEFAULTS["cfg_scale"]` — move to each engine's `main.py`

Keep:
- `LLM_SETTINGS`, `AUDIO_SETTINGS`, `VOICE_VALIDATION`, `CHUNKFORMER_VALIDATION`, `VOICE_GENDER_CORRECTION`
- `DEFAULTS["num_llm_attempts"]`, `max_chapters`, `sample_text_length`, `description_length`
- `DEFAULTS["static_voice_text"]` — global, passed as kwarg in engine requests
- `DEFAULTS["short_text_postfix"]`, `short_text_prefix_pause_ms`
- `DEFAULTS["min_silence_len"]`, `silence_thresh`
- `DEFAULTS["max_retries"]`, `enable_postfix`
- `DEFAULTS["validation_model_name"]`, `validation_model_name_fast`
- File paths: `OUTPUT_DIR`, `VOICE_SAMPLES_DIR`, `DEFAULT_EPUB_FILE`, patterns, JSON files

Update `validate()`:
- Replace `list(TTS_MODEL_PATHS.keys())` with `list_engines()` from `tts` module

**3.3. Update `audiobook_generator/audiobook_generator.py`**

Changes:
- Line 527-529: `WorkerPool` instantiation — pass `engine_dir` instead of `engine_name` + `engine_cls`
- Line 1702-1704: CLI argument parsing — dynamic choices from `tts.list_engines()` / `tts.list_voice_engines()`
- Everything else unchanged (calls through `VoiceMapper` / `engine.generate_line()`)

**3.4. Update `audiobook_generator/generate_voice_samples.py`**

Changes:
- Remove `DEFAULTS["static_voice_text"]` references (line 316, 380) — engine handles it
- Remove `DEFAULTS["max_new_tokens"]` references — engine handles it
- VoiceMapper usage unchanged (updated in 3.1)
- Fallback engine chain unchanged

**3.5. Delete old `audiobook_generator/engines/` directory**

Delete:
```
audiobook_generator/engines/base.py
audiobook_generator/engines/utils.py
audiobook_generator/engines/omni.py
audiobook_generator/engines/moss.py
audiobook_generator/engines/vox.py
audiobook_generator/engines/echo_tts.py
audiobook_generator/engines/dramabox.py
audiobook_generator/engines/miso_tts/          (entire directory)
audiobook_generator/engines/echo_tts/           (entire directory, vendored)
audiobook_generator/engines/dramabox/           (entire directory, vendored)
audiobook_generator/engines/.environments/      (entire directory)
audiobook_generator/testing.py
```

Keep `audiobook_generator/engines/__init__.py` — replaced with `tts/` imports.

### Phase 4: Cleanup

**4.1. Update `pyproject.toml`**

Remove from dependencies:
- `omnivoice>=0.1.0`
- `chunkformer>=1.2.2`

**4.2. Create `scripts/setup-engines.py`**

Pre-builds all engine venvs. Iterates `engines/*/`, runs `uv venv` + `uv pip install -e .` for each.

**4.3. Add `tts/.gitignore`**

Ignore `.environments/` directory (per-engine virtual environments).

### Phase 5: Testing

**5.1. Keep existing unit tests**

Tests that don't depend on engine internals:
- `test_parse_chapter.py`, `test_pipeline.py`, `test_utils.py`
- `test_llm_label_speakers.py`, `test_llm_describe_character.py`
- `test_audio_quality.py`
- `test_generate_audiobook_from_chapters.py`

Run these after each file change to catch regressions.

**5.2. Add unit tests for `tts/`**

New tests with mocked subprocesses:
- `tests/test_tts_worker.py` — Test `EngineWorker` lifecycle, request/response, error handling
- `tests/test_tts_pool.py` — Test `WorkerPool` round-robin distribution, multi-GPU
- `tests/test_tts_engine.py` — Test `TTSEngine` base class, interface
- `tests/test_tts_discovery.py` — Test `list_engines()`, `list_voice_engines()`
- `tests/test_tts_protocol.py` — Test JSON protocol format, error handling

**5.3. Add integration tests for `tts/`**

Test against real engines (starting with Omni):
- `tests/test_tts_omni_integration.py` — Test Omni engine end-to-end
- `tests/test_tts_moss_integration.py` — Test MOSS engine end-to-end (future)
- etc.

These tests should be marked with `@pytest.mark.integration` and skipped by default.

**5.4. Deprecate broken tests**

If any existing tests break during the refactor:
- Mark them with `@pytest.mark.deprecated` and skip
- Or remove them entirely if they test deprecated functionality

### Phase 6: Verify

1. `python scripts/setup-engines.py` — builds Omni venv
2. `engines/omni/main.py --probe` — verify capabilities
3. Run unit tests: `pytest tests/ -k "not integration"` — verify all unit tests pass
4. Run integration tests: `pytest tests/ -k "integration"` — verify real engine tests pass
5. End-to-end: parse EPUB → label speakers → describe characters → generate voice samples → generate audiobook → Whisper validation
6. Verify `--voice-engine` only lists engines with voice design support
7. Verify `--tts-engine` lists all available engines

## Decisions

- `static_voice_text` is global — kept in `config.py`, passed as kwarg in engine requests
- No streaming progress updates — protocol is request/response only
- `ref_text` is computed by the main codebase where Whisper lives, passed as kwarg to engine
- Tests are kept and expanded with mocked subprocess tests
- API fully switches to `tts/` submodule; broken tests are deprecated
- Voice sample generation moves to `tts/voice_sample.py`
- Voice path lookup, LLM validation, voice map persistence stay in `audiobook_generator/`

## Migration Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Debugging harder (JSON error strings) | Medium | Full traceback in error responses |
| 30s venv setup on first use | Low | Pre-build script |
| No type safety across boundary | Low | Protocol is simple (3 params) |
| `VoiceMapper` dead code removal | Low | Thoroughly audit all callers |
| `config.py` param migration | Low | Move to engine's `main.py` |
| CLI dynamic choices | Low | `list_engines()` + `list_voice_engines()` |
| Tests breaking during refactor | Low | Deprecate broken tests, keep working ones |

## Future: Remaining Engines

For each remaining engine (moss, vox, echo-tts, miso-tts, dramabox):
1. Create `engines/<name>/pyproject.toml` with git dep
2. Create `engines/<name>/main.py` with universal protocol
3. Move engine-specific logic from old `.py` file into `main.py`
4. Test end-to-end
5. Delete old `.py` file and vendored source
