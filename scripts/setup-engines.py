#!/usr/bin/env python3
"""Pre-build all engine virtual environments.

This script iterates over engines/ and creates a uv venv + installs
dependencies for each engine package.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = REPO_ROOT / "engines"


def setup_engine(engine_dir: Path) -> None:
    """Set up a single engine's virtual environment."""
    print(f"\n{'='*60}")
    print(f"Setting up engine: {engine_dir.name}")
    print(f"{'='*60}")

    venv_dir = engine_dir / ".venv"
    python = str(venv_dir / "bin" / "python")

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)

    # Create venv if it doesn't exist
    if not venv_dir.exists():
        print(f"  Creating venv at {venv_dir}...")
        subprocess.run(
            ["uv", "venv", str(venv_dir)],
            cwd=str(engine_dir),
            env=env,
            check=True,
        )

    # Install dependencies
    print(f"  Installing dependencies...")
    subprocess.run(
        ["uv", "pip", "install", "-e", "."],
        cwd=str(engine_dir),
        env=env,
        check=True,
    )

    # Verify installation
    print(f"  Verifying installation...")
    result = subprocess.run(
        [python, "-c", "print('OK')"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"  Engine {engine_dir.name} ready!")
    else:
        print(f"  WARNING: Verification failed: {result.stderr}")


def init_submodules() -> None:
    """Initialize and update git submodules for engine dependencies."""
    print(f"\n{'='*60}")
    print("Initializing engine submodules...")
    print(f"{'='*60}")
    result = subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: Failed to update submodules: {result.stderr}", file=sys.stderr)
    else:
        print("  Submodules initialized successfully.")


def main() -> None:
    """Main entry point."""
    if not ENGINES_DIR.exists():
        print(f"No engines directory found at {ENGINES_DIR}")
        sys.exit(1)

    # Initialize submodules first (e.g., DramaBox source code)
    init_submodules()

    engine_dirs = sorted(
        d for d in ENGINES_DIR.iterdir()
        if d.is_dir() and (d / "pyproject.toml").exists()
    )

    if not engine_dirs:
        print("No engines found. Create engine packages first.")
        sys.exit(1)

    print(f"\nFound {len(engine_dirs)} engines:")
    for d in engine_dirs:
        print(f"  - {d.name}")

    for engine_dir in engine_dirs:
        setup_engine(engine_dir)

    print(f"\n{'='*60}")
    print("All engines set up successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
