"""TTS engine registry and factory."""

from .base import TTSEngine
from .moss import MossEngine
from .omni import OmniEngine
from .vox import VoxEngine
from .kugelaudio import KugelAudioEngine
from .vibevoice import VibeVoiceEngine
from .echo_tts import EchoTTSAdapter


# Engine name -> class mapping
_ENGINE_REGISTRY = {
    "moss": MossEngine,
    "omni": OmniEngine,
    "vox": VoxEngine,
    "kugelaudio": KugelAudioEngine,
    "vibevoice": VibeVoiceEngine,
    "echo-tts": EchoTTSAdapter,
}


def get_engine(engine_name: str, device: str = "cuda", turbo: bool = False) -> TTSEngine:
    """Get a TTS engine instance by name.

    Args:
        engine_name: Engine identifier (e.g., 'moss', 'omni', 'vox', 'kugelaudio', 'vibevoice', 'echo-tts')
        device: CUDA device string (e.g., 'cuda:0')
        turbo: Whether to use turbo variant (engine-specific)

    Returns:
        A TTSEngine instance.

    Raises:
        ValueError: If engine_name is not registered.
    """
    if engine_name not in _ENGINE_REGISTRY:
        raise ValueError(f"Unknown TTS engine: {engine_name}. Available: {list(_ENGINE_REGISTRY.keys())}")
    return _ENGINE_REGISTRY[engine_name](device=device, turbo=turbo)


def list_engines() -> list:
    """Return list of available engine names."""
    return list(_ENGINE_REGISTRY.keys())
