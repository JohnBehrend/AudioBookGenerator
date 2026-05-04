#!/usr/bin/env python3
"""
VoiceMapper Module for Audiobook TTS Pipeline.

This module provides a centralized, stateful VoiceMapper class that:
- Manages voice path lookup and caching
- Handles TTS engine setup and model caching
- Generates voice samples for characters
- Persists and loads voice maps (voices_map.json)
"""

import os
import json
import gc
import traceback
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
from openai import OpenAI

from .config import DEFAULTS, AUDIO_SETTINGS, VOICE_VALIDATION, TTS_MODEL_PATHS

# Import engine registry
from .engines import get_engine

# Import utilities for validation client
from .utils import get_validation_client


class VoiceMapper:
    """Stateful voice mapper for audiobook TTS pipeline.

    This class manages:
    - Voice path lookup and caching
    - TTS engine setup and model caching (lazy loading)
    - Voice sample generation
    - voices_map.json persistence

    Design: Stateful to minimize calls - models and voice paths are cached
    after first load/generation.
    """

    def __init__(
        self,
        output_dir: str,
        device: str = "cuda:0",
        tts_engine: str = None,
        duplicate_replacement_map: Dict[str, str] = None,
        engine: Optional[Any] = None,
    ):
        """Initialize the VoiceMapper.

        Args:
            output_dir: Directory to save/load voice samples and maps
            device: Device to run TTS models ('cuda:0', 'cuda:1', etc.)
            tts_engine: TTS engine to use ('vibevoice', 'moss', 'echo-tts', 'omni', 'vox')
                       Defaults to AUDIO_SETTINGS['default_tts_engine']
            duplicate_replacement_map: Optional dict mapping duplicate character names to canonical names
            engine: Optional pre-created TTS engine instance for injection (for testing/mocking)
        """
        self.output_dir = Path(output_dir)
        self.device = device
        self.tts_engine = tts_engine or AUDIO_SETTINGS["default_tts_engine"]
        self.supported_extensions = AUDIO_SETTINGS.get("supported_audio_extensions", [".wav", ".mp3", ".flac"])
        self.duplicate_replacement_map = duplicate_replacement_map or {}

        # State containers
        self.tts_models: Dict[str, Any] = {}  # Cached TTS models
        self.voice_paths: Dict[str, str] = {}  # Cached voice file paths
        self.voice_clone_prompts: Dict[str, Any] = {}  # Pre-built prompts for voice cloning
        self._cached_engine: Optional[Any] = None  # Cached TTS engine instance

        # Engine injection - allows mocking in tests
        self._injected_engine: Optional[Any] = engine

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing voice map if present
        self._load_voice_map()

    def _load_voice_map(self) -> None:
        """Load voice map from voices_map.json if it exists."""
        voices_map_path = self.output_dir / "voices_map.json"
        if voices_map_path.exists():
            try:
                with open(voices_map_path, "r", encoding="utf-8") as f:
                    loaded_map = json.load(f)
                # Store just the filenames (relative paths)
                for char_name, voice_path in loaded_map.items():
                    # If absolute path, extract just the filename
                    if os.path.isabs(voice_path):
                        voice_file = os.path.basename(voice_path)
                    else:
                        voice_file = voice_path
                    self.voice_paths[char_name] = str(self.output_dir / voice_file)
                self._voice_map = loaded_map
            except Exception as e:
                print(f"Warning: Could not load voices_map.json: {e}")
                self._voice_map = {}
        else:
            self._voice_map = {}

    def _save_voice_map(self) -> None:
        """Save current voice map to voices_map.json."""
        voices_map_path = self.output_dir / "voices_map.json"
        with open(voices_map_path, "w", encoding="utf-8") as f:
            json.dump(self._voice_map, f, indent=2)

    # =========================================================================
    # VOICE PATH LOOKUP
    # =========================================================================

    def get_voice_path(self, character_name: str) -> Optional[str]:
        """Get the path to a voice sample file for the given character.

        Args:
            character_name: Name of the character/voice

        Returns:
            Path to the voice sample file, or None if not found
        """
        # Apply duplicate replacement map if available to find canonical name
        canonical_name = self.duplicate_replacement_map.get(character_name, character_name)

        # Check cached paths first (for both original and canonical name)
        if character_name in self.voice_paths:
            return self.voice_paths[character_name]
        if canonical_name != character_name and canonical_name in self.voice_paths:
            return self.voice_paths[canonical_name]

        # Look for voice files with supported extensions (check canonical name first)
        for ext in self.supported_extensions:
            path = self.output_dir / f"{canonical_name}{ext}"
            if path.exists():
                self.voice_paths[character_name] = str(path)
                return str(path)

        # Try partial match (case-insensitive) on canonical name
        # Only check files with supported audio extensions
        canonical_name_lower = canonical_name.lower()
        for file_path in self.output_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                stem_lower = file_path.stem.lower()
                if canonical_name_lower in stem_lower:
                    self.voice_paths[character_name] = str(file_path)
                    return str(file_path)
                # Bidirectional check: also check if file stem is a substring of canonical name
                if stem_lower in canonical_name_lower:
                    self.voice_paths[character_name] = str(file_path)
                    return str(file_path)

        return None

    def add_voice_path(self, character_name: str, voice_path: str) -> None:
        """Add a voice path to the internal cache and voice map.

        Args:
            character_name: Character name
            voice_path: Absolute path to the voice file
        """
        self.voice_paths[character_name] = voice_path
        # Store relative path in voice map
        voice_file = os.path.basename(voice_path)
        self._voice_map[character_name] = voice_file
        self._save_voice_map()

    def get_all_voice_paths(self) -> Dict[str, str]:
        """Get all cached voice paths.

        Returns:
            Dict mapping character names to voice file paths
        """
        return self.voice_paths.copy()

    def get_narrator_voice(self) -> Optional[str]:
        """Get the narrator voice path.

        Returns:
            Path to narrator.wav or None if not found
        """
        return self.get_voice_path("narrator")

    # =========================================================================
    # TTS ENGINE SETUP
    # =========================================================================

    def setup_tts_engine(self, turbo: bool = False) -> Tuple[Any, Optional[Any], str, Optional[Any]]:
        """Initialize and return the TTS model(s) for the configured engine.

        Models are cached after first load to avoid reloading.

        Args:
            turbo: Reserved for future use

        Returns:
            Tuple of (model, processor, model_path, None) for backward compatibility.
        """
        engine_key = f"{self.tts_engine}_turbo_{turbo}"

        if engine_key in self.tts_models:
            return self.tts_models[engine_key]

        engine = get_engine(self.tts_engine, device=self.device, turbo=turbo)
        model, processor = engine.setup(self.device, turbo=turbo)
        model_path = self._get_model_path()
        result = (model, processor, model_path, None)
        self.tts_models[engine_key] = result
        return result

    def _get_model_path(self) -> str:
        """Get the HuggingFace model path for the current engine."""
        engine_paths = TTS_MODEL_PATHS[self.tts_engine]
        if isinstance(engine_paths, dict):
            return engine_paths.get("base", list(engine_paths.values())[0])
        return engine_paths

    def cleanup_tts_models(self) -> None:
        """Clean up all cached TTS models from GPU memory."""
        self.tts_models.clear()
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    def get_engine(self):
        """Get or create a cached TTS engine instance.

        If an engine was injected via __init__, returns that engine.
        Otherwise creates and caches an engine using get_engine().

        Returns:
            TTSEngine instance.
        """
        if self._injected_engine is not None:
            return self._injected_engine
        if self._cached_engine is None:
            self._cached_engine = get_engine(self.tts_engine, device=self.device)
        return self._cached_engine

    def set_engine(self, engine: Any) -> None:
        """Set a TTS engine instance (for testing/mocking).

        Args:
            engine: TTS engine instance to use
        """
        self._injected_engine = engine

    def cleanup_engines(self) -> None:
        """Release cached engine instance and shutdown worker."""
        if self._cached_engine is not None:
            self._cached_engine.shutdown_worker()
            self._cached_engine = None

    @staticmethod
    def validate_voice_with_llm(
        voice_path: str,
        description: str,
        sample_text: str,
        client: Optional[OpenAI] = None,
        model: str = None,
        threshold: float = None,
        verbose: bool = False
    ) -> Tuple[bool, str]:
        """Validate a generated voice sample using LLM audio analysis.

        Args:
            voice_path: Path to the generated voice .wav file
            description: Voice description (e.g., "male. middle aged. high")
            sample_text: The text that was spoken in the voice sample
            client: OpenAI client for the validation LLM
            model: Model name for validation (defaults to VOICE_VALIDATION["model"])
            threshold: Confidence threshold (YES/NO response interpreted as pass/fail)
            verbose: Print verbose output

        Returns:
            Tuple of (is_valid, validation_message)
        """
        if client is None:
            client = get_validation_client()

        if model is None:
            model = VOICE_VALIDATION["model"]

        if threshold is None:
            threshold = VOICE_VALIDATION["threshold"]

        # Convert to absolute path for file:// URL
        abs_voice_path = os.path.abspath(voice_path)
        file_url = f"file://{abs_voice_path}"
        validation_prompt = VOICE_VALIDATION["prompt"]

        # Format the description for the prompt
        description_text = description.strip() if description else "unknown voice"

        # Format the prompt with sample text and description
        formatted_prompt = validation_prompt.format(
            sample_text=sample_text,
            description=description_text
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {"url": file_url}
                            },
                            {
                                "type": "text",
                                "text": formatted_prompt
                            }
                        ]
                    }
                ],
                max_tokens=512
            )

            result = response.choices[0].message.content.strip()

            # Parse JSON result
            try:
                import json as json_module
                validation_data = json_module.loads(result)

                is_valid = validation_data.get("overall_match", False)

                if verbose:
                    print(f"    Validation Results:")
                    print(f"      Gender match: {validation_data.get('gender_match', 'N/A')}")
                    print(f"      Age match: {validation_data.get('age_match', 'N/A')}")
                    print(f"      Tone match: {validation_data.get('tone_match', 'N/A')}")
                    print(f"      Emotion match: {validation_data.get('emotion_match', 'N/A')}")
                    print(f"      Clarity match: {validation_data.get('clarity_match', 'N/A')}")
                    print(f"      Overall: {'PASS' if is_valid else 'FAIL'}")
                    if validation_data.get('reasons'):
                        print(f"      Reasons: {validation_data['reasons']}")

                return is_valid, result

            except json_module.JSONDecodeError:
                # Fallback: check for YES/NO in plain text response
                is_valid = "YES" in result.upper() or "true" in result.lower()

                if verbose:
                    print(f"    Validation result: {result}")
                    print(f"    Voice {'passed' if is_valid else 'failed'} validation")

                return is_valid, result

        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            if verbose:
                print(f"    {error_msg}")
            # On error, return True to allow generation to continue
            return True, error_msg

    def unload_model(self, engine_name: str) -> None:
        """Unload models for a specific TTS engine.

        Args:
            engine_name: Name of the TTS engine to unload
        """
        keys_to_remove = [k for k in self.tts_models if k.startswith(f"{engine_name}_")]
        for key in keys_to_remove:
            del self.tts_models[key]
        if keys_to_remove:
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    def reset(self) -> None:
        """Reset all internal state (for testing).

        This clears all cached models, voice paths, and prompts to allow
        fresh state in tests.
        """
        self.cleanup_tts_models()
        self.voice_paths.clear()
        self.voice_clone_prompts.clear()
        self._voice_map.clear()

    # =========================================================================
    # VOICE GENERATION
    # =========================================================================

    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: str = None,
        max_new_tokens: int = None,
        verbose: bool = False
    ) -> Tuple[bool, Optional[str], float]:
        """Generate a voice sample for a character using the configured TTS engine.

        Delegates to the registered engine for generation logic.

        Args:
            character_name: Name of the character
            description: Voice description from LLM
            output_dir: Output directory (defaults to self.output_dir)
            max_new_tokens: Max tokens for generation (engine-specific)
            verbose: Print verbose output

        Returns:
            Tuple of (success, output_file_path, duration_seconds)
        """
        if output_dir is None:
            output_dir = self.output_dir
        if max_new_tokens is None:
            max_new_tokens = DEFAULTS["max_new_tokens"]

        engine = get_engine(self.tts_engine, device=self.device)
        success, output_file, duration = engine.generate_voice_sample(
            character_name=character_name,
            description=description,
            output_dir=Path(output_dir),
            device=self.device,
            verbose=verbose,
        )

        if success:
            self.add_voice_path(character_name, output_file)

        engine.shutdown_worker()
        return success, output_file, duration

    def build_voice_clone_prompt(
        self,
        voice_path: str,
        ref_text: str = None,
        validation_model = None,
        auto_transcribe: bool = False,
        verbose: bool = False
    ) -> Any:
        """Build a voice_clone_prompt for voice cloning.

        Prompts are cached to avoid rebuilding for each line.

        Args:
            voice_path: Path to the voice sample file
            ref_text: Reference text for voice cloning
            validation_model: Optional WhisperModel for auto-transcription
            auto_transcribe: If True, transcribe the audio to get ref_text
            verbose: Print verbose output

        Returns:
            A voice_clone_prompt that can be reused for generate_voice_clone calls
        """
        if ref_text is None:
            ref_text = ""

        # Auto-transcribe if requested and validation_model is available
        if auto_transcribe and validation_model is not None:
            try:
                from ..utils import transcribe_audio_with_whisper
                actual_ref_text, _, _ = transcribe_audio_with_whisper(validation_model, voice_path)
                if verbose:
                    print(f"  Transcribed ref_text: {actual_ref_text}")
                ref_text = actual_ref_text
            except Exception as e:
                if verbose:
                    print(f"  Warning: auto_transcribe failed: {e}")

        import soundfile as sf
        import torch

        voice_audio, sr = sf.read(voice_path)
        # Convert numpy array to torch tensor (OmniVoice expects tensor, not numpy array)
        voice_audio = torch.from_numpy(voice_audio)

        # Get Base model (needed to build the prompt)
        _, _, _, base_model = self.setup_tts_engine()

        voice_clone_prompt = base_model.create_voice_clone_prompt(
            ref_audio=(voice_audio, sr),
            ref_text=ref_text,
        )

        return voice_clone_prompt

    # =========================================================================
    # AUDIOBOOK GENERATION HELPERS
    # =========================================================================

    def get_voice_clone_prompt(
        self,
        character_name: str,
        ref_text: str = None,
        validation_model = None,
        auto_transcribe: bool = False,
        verbose: bool = False
    ) -> Optional[Any]:
        """Get or build a cached voice_clone_prompt for a character.

        Args:
            character_name: Name of the character
            ref_text: Reference text for voice cloning
            validation_model: Optional WhisperModel for auto-transcription
            auto_transcribe: If True, transcribe the audio to get ref_text
            verbose: Print verbose output

        Returns:
            Cached or newly built voice_clone_prompt, or None if voice not found
        """
        # Check if we already have the prompt cached
        if character_name in self.voice_clone_prompts:
            return self.voice_clone_prompts[character_name]

        # Get the voice path
        voice_path = self.get_voice_path(character_name)
        if voice_path is None:
            if verbose:
                print(f"  Warning: No voice path found for '{character_name}'")
            return None

        # Build the prompt
        prompt = self.build_voice_clone_prompt(
            voice_path=voice_path,
            ref_text=ref_text,
            validation_model=validation_model,
            auto_transcribe=auto_transcribe,
            verbose=verbose
        )

        # Cache it
        self.voice_clone_prompts[character_name] = prompt
        return prompt

    def get_all_clone_prompts(
        self,
        validation_model = None,
        auto_transcribe: bool = False,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Build and cache voice_clone_prompts for all voices.

        Args:
            validation_model: Optional WhisperModel for auto-transcription
            auto_transcribe: If True, transcribe the audio to get ref_text
            verbose: Print verbose output

        Returns:
            Dict mapping character names to voice_clone_prompts
        """
        prompts = {}
        for character_name in self.voice_paths.keys():
            prompt = self.get_voice_clone_prompt(
                character_name,
                validation_model=validation_model,
                auto_transcribe=auto_transcribe,
                verbose=verbose
            )
            if prompt is not None:
                prompts[character_name] = prompt
        return prompts