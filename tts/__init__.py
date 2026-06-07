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
from .pool import WorkerPool, WhisperPool
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
