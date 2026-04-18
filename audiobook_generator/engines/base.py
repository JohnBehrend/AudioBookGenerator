"""Base class for TTS engines."""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
from pathlib import Path


class TTSEngine(ABC):
    """Abstract base class for TTS engines.

    Each engine implementation handles:
    - Loading its model(s) from HuggingFace or local sources
    - Generating voice samples from character descriptions (Stage 4)
    - Generating audio lines from text + voice reference (Stage 5)

    Engines are loaded lazily via setup(). Results are cached in
    VoiceMapper.tts_models.
    """

    @abstractmethod
    def setup(self, device: str, turbo: bool = False) -> Tuple[Any, Optional[Any]]:
        """Load the model(s) for this engine.

        Args:
            device: CUDA device string (e.g., 'cuda:0')
            turbo: Whether to use turbo variant (engine-specific)

        Returns:
            Tuple of (model, processor) where processor may be None.
            These are the objects passed to callers in audiobook generation.
        """

    @abstractmethod
    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: Path,
        device: str,
        verbose: bool = False,
    ) -> Tuple[bool, Optional[str], float]:
        """Generate a voice sample for a character (Stage 4).

        Args:
            character_name: Name of the character
            description: Voice description from LLM
            output_dir: Directory to save the .wav file
            device: CUDA device string
            verbose: Print verbose output

        Returns:
            Tuple of (success, output_file_path, duration_seconds)
        """

    @abstractmethod
    def generate_line(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        device: str,
        validation_model,
        cfg_scale: float = 1.3,
        max_new_tokens: int = 19200,
        verbose: bool = False,
    ) -> bool:
        """Generate audio for a single line (Stage 5).

        Args:
            text: Text to synthesize
            voice_path: Path to voice sample for cloning (None for non-cloning engines)
            output_path: Where to save the generated .wav file
            device: CUDA device string
            validation_model: WhisperModel for ref_text transcription
            cfg_scale: CFG scale value
            max_new_tokens: Max tokens for generation
            verbose: Print verbose output

        Returns:
            True if generation succeeded, False otherwise.
        """
