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
import torch
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

# Import config for default values
from config import DEFAULTS, AUDIO_SETTINGS

# Helper to check if flash-attn is available
def _get_attn_implementation() -> Optional[str]:
    """Return flash_attention_2 if available, otherwise None."""
    try:
        import flash_attn
        return "flash_attention_2"
    except ImportError:
        return None


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

    def __init__(self, output_dir: str, device: str = "cuda:0", tts_engine: str = None):
        """Initialize the VoiceMapper.

        Args:
            output_dir: Directory to save/load voice samples and maps
            device: Device to run TTS models ('cuda:0', 'cuda:1', etc.)
            tts_engine: TTS engine to use ('kugelaudio', 'vibevoice', 'moss')
                       Defaults to AUDIO_SETTINGS['default_tts_engine']
        """
        self.output_dir = Path(output_dir)
        self.device = device
        self.tts_engine = tts_engine or AUDIO_SETTINGS.get("default_tts_engine", "kugelaudio")
        self.supported_extensions = AUDIO_SETTINGS.get("supported_audio_extensions", [".wav", ".mp3", ".flac"])

        # State containers
        self.tts_models: Dict[str, Any] = {}  # Cached TTS models
        self.voice_paths: Dict[str, str] = {}  # Cached voice file paths
        self.voice_clone_prompts: Dict[str, Any] = {}  # Pre-built prompts for voice cloning

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
        # Check cached paths first
        if character_name in self.voice_paths:
            return self.voice_paths[character_name]

        # Look for voice files with supported extensions
        for ext in self.supported_extensions:
            path = self.output_dir / f"{character_name}{ext}"
            if path.exists():
                self.voice_paths[character_name] = str(path)
                return str(path)

        # Try partial match (case-insensitive)
        character_name_lower = character_name.lower()
        for file_path in self.output_dir.iterdir():
            if file_path.is_file():
                stem_lower = file_path.stem.lower()
                if character_name_lower in stem_lower:
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
            turbo: Use KugelAudio turbo model (ignored for non-kugelaudio engines)

        Returns:
            For kugelaudio/vibevoice: (model, processor, model_path, None)
            For moss: (model, processor, model_path, None)
        """
        engine_key = f"{self.tts_engine}_turbo_{turbo}"

        if engine_key in self.tts_models:
            return self.tts_models[engine_key]

        attn_impl = _get_attn_implementation()
        attn_kwargs = {"attn_implementation": attn_impl} if attn_impl else {}

        if self.tts_engine == "kugelaudio":
            from kugelaudio_open.processors.kugelaudio_processor import KugelAudioProcessor
            from kugelaudio_open.models.kugelaudio_inference import KugelAudioForConditionalGenerationInference

            model_path = "kugel-1-turbo" if turbo else "kugelaudio/kugelaudio-0-open"
            tts_model = KugelAudioForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                **attn_kwargs,
            )
            tts_model.eval()
            processor = KugelAudioProcessor.from_pretrained(model_path)
            result = (tts_model, processor, model_path, None)
            self.tts_models[engine_key] = result
            return result

        elif self.tts_engine == "vibevoice":
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
            from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

            model_path = "Jmica/VibeVoice7B"
            tts_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                **attn_kwargs,
            )
            tts_model.eval()
            processor = VibeVoiceProcessor.from_pretrained(model_path)
            result = (tts_model, processor, model_path, None)
            self.tts_models[engine_key] = result
            return result

        elif self.tts_engine == "moss":
            from transformers import AutoModel, AutoProcessor

            model_path = "OpenMOSS-Team/MOSS-TTS"

            # Initialize model with explicit device placement (not device_map)
            # This avoids meta tensor issues with lazy loading
            tts_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to(self.device).eval()

            # Initialize processor and move audio_tokenizer to device
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            processor.audio_tokenizer = processor.audio_tokenizer.to(self.device)

            result = (tts_model, processor, model_path, None)
            self.tts_models[engine_key] = result
            return result

        else:
            raise ValueError(f"Unknown TTS engine: {self.tts_engine}")

    def cleanup_tts_models(self) -> None:
        """Clean up all cached TTS models from GPU memory."""
        for key, models in list(self.tts_models.items()):
            for model in models:
                if hasattr(model, "to") and hasattr(model, "state_dict"):
                    try:
                        del model
                    except Exception:
                        pass
        self.tts_models.clear()
        gc.collect()
        torch.cuda.empty_cache()

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

        Args:
            character_name: Name of the character
            description: Voice description from LLM
            output_dir: Output directory (defaults to self.output_dir)
            max_new_tokens: Max tokens for generation
            verbose: Print verbose output

        Returns:
            Tuple of (success, output_file_path, duration_seconds)
        """
        if self.tts_engine == "moss":
            return self._generate_voice_sample_moss(
                character_name, description, output_dir, max_new_tokens, verbose
            )
        else:
            # For kugelaudio and vibevoice, generate from a reference text
            return self._generate_voice_sample_generic(
                character_name, description, output_dir, max_new_tokens, verbose
            )

    def _generate_voice_sample_generic(
        self,
        character_name: str,
        description: str,
        output_dir: str = None,
        max_new_tokens: int = None,
        verbose: bool = False
    ) -> Tuple[bool, Optional[str], float]:
        """Generate a voice sample for a character using kugelaudio or vibevoice.

        For these engines, we generate from a reference text directly without
        voice cloning (since we don't have an existing voice to clone yet).

        Args:
            character_name: Name of the character
            description: Voice description from LLM
            output_dir: Output directory (defaults to self.output_dir)
            max_new_tokens: Max tokens for generation
            verbose: Print verbose output

        Returns:
            Tuple of (success, output_file_path, duration_seconds)
        """
        if output_dir is None:
            output_dir = self.output_dir
        if max_new_tokens is None:
            max_new_tokens = DEFAULTS["max_new_tokens"]

        # Reference text for voice generation
        sample_text = "The quick brown fox jumps over the lazy dog."

        # Validate description
        if not description or not description.strip():
            if verbose:
                print(f"  ERROR: Skipping '{character_name}' due to empty description")
            return False, None, 0

        # Get the TTS model and processor
        tts_model, processor, model_path, _ = self.setup_tts_engine()

        try:
            if self.tts_engine == "kugelaudio":
                # KugelAudio: generate from text with voice prompt (using a generic reference)
                # For initial voice sample, we generate from text directly
                inputs = processor(
                    text=sample_text,
                    padding=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    speech_outputs = tts_model.generate(
                        input_ids=inputs["input_ids"].to(self.device),
                        attention_mask=inputs["attention_mask"].to(self.device),
                        max_new_tokens=max_new_tokens,
                    )
                wavs = speech_outputs.float().cpu().numpy()
                sr = tts_model.config.sample_rate
            elif self.tts_engine == "vibevoice":
                # VibeVoice: generate from text
                inputs = processor(
                    text=[sample_text],
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                with torch.no_grad():
                    outputs = tts_model.generate(
                        input_ids=inputs["input_ids"].to(self.device),
                        attention_mask=inputs["attention_mask"].to(self.device),
                        max_new_tokens=max_new_tokens,
                    )
                wavs = outputs.cpu().numpy()
                sr = tts_model.config.sampling_rate
            else:
                return False, None, 0

            if not wavs or len(wavs) == 0:
                return False, None, 0

            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{character_name}.wav")

            import soundfile as sf
            sf.write(output_file, wavs[0], sr)

            duration = len(wavs[0]) / sr
            self.add_voice_path(character_name, output_file)

            if verbose:
                print(f"    Generated: {duration:.2f}s -> {output_file}")

            return True, output_file, duration

        except Exception as e:
            print(f"    Error generating voice: {e}")
            return False, None, 0

    def _generate_voice_sample_moss(
        self,
        character_name: str,
        description: str,
        output_dir: str = None,
        max_new_tokens: int = None,
        verbose: bool = False
    ) -> Tuple[bool, Optional[str], float]:
        """Generate a voice sample for a character using MOSS-TTS.

        MOSS-TTS generates speech directly from text. The character description
        is used as the prompt for voice style, and a sample text is generated.

        Args:
            character_name: Name of the character
            description: Voice description from LLM (used to style the speech)
            output_dir: Output directory (defaults to self.output_dir)
            max_new_tokens: Max tokens for generation
            verbose: Print verbose output

        Returns:
            Tuple of (success, output_file_path, duration_seconds)
        """
        if output_dir is None:
            output_dir = self.output_dir
        if max_new_tokens is None:
            max_new_tokens = DEFAULTS["max_new_tokens"]

        # Use the character description as the speech text
        # This allows the voice to reflect the character's described traits
        sample_text = description.strip() if description else "The quick brown fox jumps over the lazy dog."

        # Validate description
        if not description or not description.strip():
            if verbose:
                print(f"  ERROR: Skipping '{character_name}' due to empty description")
            return False, None, 0

        # Get MOSS model and processor
        model, processor, _, _ = self.setup_tts_engine()

        try:
            # Build conversation with text only - MOSS-TTS generates directly from text
            conversations = [
                processor.build_user_message(text=sample_text)
            ]

            # Prepare batch for generation
            with torch.no_grad():
                batch = processor(conversations, mode="generation")
                outputs = model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    max_new_tokens=max_new_tokens,
                )

                message = processor.decode(outputs)[0]
                audio = message.audio_codes_list[0]
                sr = processor.model_config.sampling_rate

                # MOSS returns CPU tensor directly - no meta tensor handling needed
                # when model is loaded with .to(device)

            if audio is None or (hasattr(audio, 'numel') and audio.numel() == 0):
                return False, None, 0

            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{character_name}.wav")

            # Save audio
            import torchaudio
            torchaudio.save(output_file, audio.unsqueeze(0), sr)

            duration = len(audio) / sr
            self.add_voice_path(character_name, output_file)

            if verbose:
                print(f"    Generated: {duration:.2f}s -> {output_file}")

            return True, output_file, duration

        except Exception as e:
            import traceback
            print(f"    Error generating voice with MOSS-TTS: {e}")
            print(f"    Exception type: {type(e).__name__}")
            traceback.print_exc()
            return False, None, 0

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
                from utils import transcribe_audio_with_whisper
                actual_ref_text, _, _ = transcribe_audio_with_whisper(validation_model, voice_path)
                if verbose:
                    print(f"  Transcribed ref_text: {actual_ref_text}")
                ref_text = actual_ref_text
            except Exception as e:
                if verbose:
                    print(f"  Warning: auto_transcribe failed: {e}")

        import soundfile as sf

        # Load the voice sample
        voice_audio, sr = sf.read(voice_path)

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