#!/usr/bin/env python3
"""
VoiceMapper Module for Audiobook TTS Pipeline.

This module provides a centralized, stateful VoiceMapper class that:
- Manages voice path lookup and caching
- Generates voice samples for characters (via tts/ submodule)
- Persists and loads voice maps (voices_map.json)
"""

import os
import json
import gc
import traceback
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from openai import OpenAI

from .config import DEFAULTS, AUDIO_SETTINGS, VOICE_VALIDATION, LLM_SETTINGS

# Import TTS submodule
from tts import TTSEngine, get_engine_dir, list_engines, get_engine

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
        tts_engine: Optional[str] = None,
        duplicate_replacement_map: Optional[Dict[str, str]] = None,
        engine: Optional[Any] = None,
        use_celebrity_voices: bool = False,
        whisper_model: Any = None,
    ) -> None:
        """Initialize the VoiceMapper.

        Args:
            output_dir: Output directory for voice files
            device: CUDA device string
            tts_engine: TTS engine name (omni, dramabox, etc.)
            duplicate_replacement_map: Map of duplicate names to canonical names
            engine: Optional pre-configured engine instance
            use_celebrity_voices: If True, use celebrity voice references instead of generating samples
            whisper_model: Optional WhisperModel for celebrity speech identification
        """
        self.output_dir = Path(output_dir)
        self.device = device
        self.tts_engine = tts_engine or AUDIO_SETTINGS["default_tts_engine"]
        self.supported_extensions = AUDIO_SETTINGS.get("supported_audio_extensions", [".wav", ".mp3", ".flac"])
        self.duplicate_replacement_map = duplicate_replacement_map or {}
        self._injected_engine = engine
        self._cached_engine = None
        self.use_celebrity_voices = use_celebrity_voices
        self.whisper_model = whisper_model

        # Voice paths cache
        self.voice_paths: Dict[str, str] = {}
        self._voice_map: Dict[str, Any] = {}

        # Load existing voice map if available
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

    def add_voice_path(self, character_name: str, voice_path: str, persist: bool = True) -> None:
        """Add a voice path to the internal cache and voice map.

        Args:
            character_name: Character name
            voice_path: Absolute path to the voice file
            persist: If True, save to voices_map.json. If False, only cache in memory.
        """
        self.voice_paths[character_name] = voice_path
        if persist:
            voice_file = os.path.basename(voice_path)
            self._voice_map[character_name] = voice_file
            self._save_voice_map()
        print(f"    [DEBUG] Added voice path for '{character_name}': {voice_path}")

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

    def get_engine(self):
        """Get or create a cached TTS engine instance.

        If an engine was injected via __init__, returns that engine.
        Otherwise creates and caches an engine using tts.get_engine().

        Returns:
            TTSEngine instance.
        """
        if self._injected_engine is not None:
            return self._injected_engine
        if self._cached_engine is None:
            self._cached_engine = get_engine(self.tts_engine, device=self.device)
        return self._cached_engine

    def get_pool(self, devices: List[str]) -> "WorkerPool":
        """Get a multi-GPU worker pool for the configured engine.

        Args:
            devices: List of CUDA device strings (e.g., ['cuda:0', 'cuda:1'])

        Returns:
            WorkerPool instance ready to distribute requests across GPUs.
        """
        from tts import WorkerPool

        engine_dir = get_engine_dir(self.tts_engine)
        return WorkerPool(engine_dir, devices)

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

        # Convert to absolute path and encode as base64 data URI
        abs_voice_path = os.path.abspath(voice_path)
        import base64
        with open(abs_voice_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        file_url = f"data:audio/wav;base64,{audio_b64}"
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

    @staticmethod
    def describe_voice_with_llm(
        voice_path: str,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        verbose: bool = False
    ) -> str:
        """Describe a voice sample using LLM audio analysis.

        Analyzes a WAV file and returns a voice description in the same format
        used for voice generation (comma-separated attributes: gender, age, pitch, accent).

        Args:
            voice_path: Path to the voice .wav file to describe
            client: OpenAI client for the LLM
            model: Model name for analysis (defaults to VOICE_VALIDATION["model"])
            verbose: Print verbose output

        Returns:
            Voice description string (e.g., "male, middle-aged, moderate pitch, american accent")
        """
        if client is None:
            client = get_validation_client()

        if model is None:
            model = VOICE_VALIDATION["model"]

        abs_voice_path = os.path.abspath(voice_path)
        import base64
        with open(abs_voice_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        file_url = f"data:audio/wav;base64,{audio_b64}"

        prompt = (
            "Analyze this voice sample and describe it using these EXACT attributes:\n\n"
            "GENDER: male OR female\n"
            "AGE: child OR teenager OR young adult OR middle-aged OR elderly\n"
            "PITCH: very low pitch OR low pitch OR moderate pitch OR high pitch OR very high pitch\n"
            "ACCENT: american accent OR british accent OR australian accent OR canadian accent OR indian accent OR chinese accent OR korean accent OR japanese accent OR portuguese accent OR russian accent\n\n"
            "RULES:\n"
            "- Output ONLY comma-separated attributes (no other text)\n"
            "- NO markdown, NO sentences, NO \"a\", NO \"with\", NO \"voice\"\n"
            "- Use ONLY the supported attributes listed above\n"
            "- ONE gender only (male OR female)\n"
            "- ONE age only\n"
            "- ONE pitch only\n"
            "- ONE accent only (or omit if not applicable)\n"
            "- 2-5 attributes max, comma-separated\n\n"
            "Format: <gender>, <age>, <pitch>, <accent>\n\n"
            "Examples:\n"
            "- male, middle-aged, moderate pitch\n"
            "- female, young adult, high pitch, british accent\n"
            "- male, elderly, low pitch\n"
            "- female, young adult, moderate pitch, american accent\n"
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
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=256
            )

            result = response.choices[0].message.content.strip()

            # Clean up the result: remove markdown, quotes, extra whitespace
            result = result.replace('"', '').replace("'", "").strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[-1].rstrip("```").strip()

            if verbose:
                print(f"    Voice description: {result}")

            return result

        except Exception as e:
            error_msg = f"Description error: {str(e)}"
            if verbose:
                print(f"    {error_msg}")
            return ""

    def unload_model(self, engine_name: str) -> None:
        """Unload models for a specific TTS engine.

        Args:
            engine_name: Name of the TTS engine to unload
        """
        pass

    def reset(self) -> None:
        """Reset all internal state (for testing).

        This clears all cached voice paths to allow fresh state in tests.
        """
        self.voice_paths.clear()
        self._voice_map.clear()

    # =========================================================================
    # VOICE GENERATION
    # =========================================================================

    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: Optional[str] = None,
        verbose: bool = False,
        client: Optional[Any] = None,
        model: str = "coder-model",
        **kwargs
    ) -> Tuple[bool, Optional[str], float]:
        """Generate a voice sample for a character using the configured TTS engine.

        Uses the cached engine instance to avoid reloading the model for each call.

        Args:
            character_name: Name of the character
            description: Voice description from LLM
            output_dir: Output directory (defaults to self.output_dir)
            verbose: Print verbose output
            client: Optional OpenAI client for celebrity matching
            model: LLM model name for celebrity matching
            **kwargs: Additional arguments passed to the engine

        Returns:
            Tuple of (success, output_file_path, duration_seconds)
        """
        if output_dir is None:
            output_dir = self.output_dir

        engine = self.get_engine()

        # Use celebrity voice if enabled
        if self.use_celebrity_voices and client:
            from .celebrity_voices import build_celebrity_voice
            # Try to extract celebrity_voice from description first
            pre_matched_celebrity = None
            try:
                desc_obj = json.loads(description) if isinstance(description, str) else description
                if isinstance(desc_obj, dict):
                    pre_matched_celebrity = desc_obj.get("celebrity_voice", "")
            except (json.JSONDecodeError, AttributeError):
                pass

            if verbose:
                print(f"    Celebrity voice enabled for: {character_name}")
                if pre_matched_celebrity:
                    print(f"    Pre-matched celebrity: {pre_matched_celebrity}")
                else:
                    print(f"    No pre-matched celebrity - will use LLM to match")

            voice_path, metadata = build_celebrity_voice(
                client=client,
                model=model,
                character=character_name,
                description=description,
                output_dir=str(output_dir),
                pre_matched_celebrity=pre_matched_celebrity or None,
                whisper_model=self.whisper_model,
                tts_engine=engine,
                verbose=verbose,
            )
            if voice_path:
                if verbose:
                    print(f"    [DEBUG] Celebrity voice path: {voice_path}")
                    if metadata:
                        print(f"    [DEBUG] Celebrity: {metadata.get('celebrity', 'N/A')}")
                        print(f"    [DEBUG] Search query: {metadata.get('search_query', 'N/A')}")

                file_size = os.path.getsize(voice_path)
                duration_seconds = file_size / (24000 * 2)
                self.add_voice_path(character_name, voice_path, persist=False)
                return True, voice_path, duration_seconds
            elif verbose:
                print(f"    [DEBUG] Celebrity voice generation failed for '{character_name}'")

        # Inject static_voice_text from config if not already in kwargs
        if "static_voice_text" not in kwargs:
            kwargs["static_voice_text"] = DEFAULTS.get("static_voice_text", "")
        if verbose:
            print(f"    [DEBUG] Calling TTS engine '{self.tts_engine}' for '{character_name}'")
        success, output_file, duration = engine.generate_voice_sample(
            character_name=character_name,
            description=description,
            output_dir=Path(output_dir),
            device=self.device,
            verbose=verbose,
            **kwargs
        )

        if success:
            if verbose:
                print(f"    [DEBUG] Voice generated successfully: {output_file}")
                if output_file and os.path.exists(output_file):
                    print(f"    [DEBUG] Voice file verified: {output_file} ({os.path.getsize(output_file)} bytes)")
            self.add_voice_path(character_name, output_file, persist=False)
        else:
            if verbose:
                print(f"    [DEBUG] Voice generation failed for '{character_name}'")

        return success, output_file, duration

    # =========================================================================
    # AUDIOBOOK GENERATION HELPERS
    # =========================================================================