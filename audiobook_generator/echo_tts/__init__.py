# Echo TTS integration module
# This module provides lazy loading of echo-tts from the echo-tts-source directory.

import os
import sys
from pathlib import Path

# Get the project root directory
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent  # Go up from echo_tts/ to audiobook_generator/ to project root

# Echo TTS source directory (cloned from GitHub)
ECHO_TTS_CLONE_DIR = PROJECT_ROOT / "echo-tts-source"


def ensure_echo_tts_available():
    """Ensure echo-tts source is available, clone if needed."""
    if not ECHO_TTS_CLONE_DIR.exists():
        print(f"Cloning echo-tts from https://github.com/jordandare/echo-tts.git...")
        import subprocess
        result = subprocess.run(
            ["git", "clone", "https://github.com/jordandare/echo-tts.git", str(ECHO_TTS_CLONE_DIR)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone echo-tts: {result.stderr}")
        print(f"echo-tts cloned to {ECHO_TTS_CLONE_DIR}")


# Lazy import mechanism
__all__ = [
    "EchoTTSLoader",
    "ECHO_TTS_CLONE_DIR",
    "ensure_echo_tts_available",
]

_cached_inference = None


def _get_inference():
    """Get the inference module from echo-tts-source."""
    global _cached_inference
    if _cached_inference is None:
        ensure_echo_tts_available()

        # Add echo-tts-source to path
        source_dir = str(ECHO_TTS_CLONE_DIR)
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)

        # Import inference module
        import importlib
        _cached_inference = importlib.import_module("inference")

    return _cached_inference


class EchoTTSLoader:
    """Lazy loader for Echo TTS models."""

    def __init__(self, device: str, model_path: str):
        self.device = device
        self.model_path = model_path
        self._loaded = False
        self.echo_model = None
        self.fish_ae = None
        self.pca_state = None
        self.sample_fn = None

    def load(self):
        """Load Echo TTS models."""
        if self._loaded:
            return

        import torch
        from functools import partial

        inference = _get_inference()

        print("Loading Echo TTS models...")
        self.echo_model = inference.load_model_from_hf(
            repo_id=self.model_path,
            device=self.device,
            dtype=torch.bfloat16,
            delete_blockwise_modules=True,
        )
        self.fish_ae = inference.load_fish_ae_from_hf(device=self.device, dtype=torch.float32)
        self.pca_state = inference.load_pca_state_from_hf(repo_id=self.model_path, device=self.device)

        # Create the sampler (sample_fn for sample_pipeline)
        # Higher num_steps = better quality, slower generation
        # Higher sequence_length = allows longer audio output
        self.sampler = partial(
            inference.sample_euler_cfg_independent_guidances,
            num_steps=50,  # Increased from 40 for better quality/pacing
            cfg_scale_text=3.0,
            cfg_scale_speaker=8.0,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=None,
            rescale_k=None,
            rescale_sigma=None,
            speaker_kv_scale=None,
            speaker_kv_max_layers=None,
            speaker_kv_min_t=None,
            sequence_length=640,  # Default max (~30 sec) - model limit
        )
        # Store sample_pipeline for direct calling
        self.sample_pipeline = inference.sample_pipeline

        self._loaded = True
        print("Echo TTS models loaded successfully!")

    def get_sample_fn(self):
        """Get the sample function."""
        self.load()
        return self.sample_fn
