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
import torch
import torchaudio
import soundfile as sf
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
from openai import OpenAI

# Import config for default values
from config import DEFAULTS, AUDIO_SETTINGS, TTS_MODEL_PATHS, VOICE_VALIDATION

# Import utilities for validation client and attention implementation
from utils import get_validation_client, _get_attn_implementation


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

    def __init__(self, output_dir: str, device: str = "cuda:0", tts_engine: str = None, duplicate_replacement_map: Dict[str, str] = None):
        """Initialize the VoiceMapper.

        Args:
            output_dir: Directory to save/load voice samples and maps
            device: Device to run TTS models ('cuda:0', 'cuda:1', etc.)
            tts_engine: TTS engine to use ('kugelaudio', 'vibevoice', 'moss', 'echo-tts', 'omni', 'vox')
                       Defaults to AUDIO_SETTINGS['default_tts_engine']
            duplicate_replacement_map: Optional dict mapping duplicate character names to canonical names
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

            model_path = TTS_MODEL_PATHS["kugelaudio"]["turbo"] if turbo else TTS_MODEL_PATHS["kugelaudio"]["base"]
            # Use device_map for KugelAudio (required to avoid meta tensor issues)
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

            model_path = TTS_MODEL_PATHS["vibevoice"]
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

            model_path = TTS_MODEL_PATHS["moss"]

            # Memory-efficient attention settings
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

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

            # Fix: Ensure model.config has num_hidden_layers for DynamicCache compatibility
            # The MOSS model's config class (MossTTSDelayConfig) doesn't expose this
            # so we add it manually if missing
            if not hasattr(tts_model.config, "num_hidden_layers"):
                tts_model.config.num_hidden_layers = getattr(
                    tts_model.config, "num_layers",
                    getattr(tts_model.config, "n_layers", 32)  # Default to 32 if neither exists
                )

            result = (tts_model, processor, model_path, None)
            self.tts_models[engine_key] = result
            return result

        elif self.tts_engine == "echo-tts":
            # Use the EchoTTSLoader from the echo_tts module
            # Import using absolute path to work when running as script
            import importlib.util
            echo_tts_path = Path(__file__).parent / "echo_tts" / "__init__.py"
            spec = importlib.util.spec_from_file_location("echo_tts_module", echo_tts_path)
            echo_tts_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(echo_tts_module)
            EchoTTSLoader = echo_tts_module.EchoTTSLoader

            model_path = TTS_MODEL_PATHS["echo-tts"]
            loader = EchoTTSLoader(self.device, model_path)
            result = (loader, None, model_path, None)
            self.tts_models[engine_key] = result
            return result

        elif self.tts_engine == "omni":
            from omnivoice import OmniVoice

            model_path = TTS_MODEL_PATHS["omni"]

            # Load OmniVoice model (per official docs)
            tts_model = OmniVoice.from_pretrained(
                model_path,
                device_map=self.device,
                dtype=torch.float16,
            )

            # Pre-load ASR model to avoid repeated downloads during generation
            # This is needed when ref_text is not provided (auto-transcription mode)
            try:
                tts_model.load_asr_model()
            except Exception as e:
                print(f"  Warning: Could not pre-load ASR model: {e}")

            # OmniVoice doesn't use a separate processor
            result = (tts_model, None, model_path, None)
            self.tts_models[engine_key] = result
            return result

        elif self.tts_engine == "vox":
            from voxcpm import VoxCPM

            model_path = TTS_MODEL_PATHS["vox"]

            # Load VoxCPM model (disable denoiser if not needed)
            tts_model = VoxCPM.from_pretrained(model_path, load_denoiser=False)

            # VoxCPM doesn't use a separate processor
            result = (tts_model, None, model_path, None)
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

    def unload_model(self, model_name: str) -> None:
        """Unload a specific model by name.

        Args:
            model_name: Name of the model to unload (e.g., 'kugelaudio', 'moss')
        """
        if model_name in self.tts_models:
            del self.tts_models[model_name]
            gc.collect()
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

        Args:
            character_name: Name of the character
            description: Voice description from LLM
            output_dir: Output directory (defaults to self.output_dir)
            max_new_tokens: Max tokens for generation
            verbose: Print verbose output

        Returns:
            Tuple of (success, output_file_path, duration_seconds)
        """
        if self.tts_engine in ("moss", "omni", "vox"):
            # MOSS, OmniVoice, and VoxCPM support voice design from descriptions
            if self.tts_engine == "moss":
                return self._generate_voice_sample_moss(
                    character_name, description, output_dir, max_new_tokens, verbose
                )
            elif self.tts_engine == "omni":
                return self._generate_voice_sample_omni(
                    character_name, description, output_dir, max_new_tokens, verbose
                )
            else:
                return self._generate_voice_sample_vox(
                    character_name, description, output_dir, max_new_tokens, verbose
                )
        else:
            # For kugelaudio, vibevoice, and echo-tts, generate from a reference text
            # Note: echo-tts is only used for voice cloning in stage 5, not for initial voice generation
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

        # Reference text for voice generation - uses static text from config
        sample_text = DEFAULTS["static_voice_text"]

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

    def _generate_voice_sample_echo_tts(
        self,
        character_name: str,
        description: str,
        output_dir: str = None,
        max_new_tokens: int = None,
        verbose: bool = False
    ) -> Tuple[bool, Optional[str], float]:
        """Generate a voice sample for a character using Echo TTS.

        Echo TTS uses speaker reference audio for conditioning. Since we don't
        have an existing voice to clone, we generate from text with the static
        voice text as the prompt.

        Args:
            character_name: Name of the character
            description: Voice description from LLM (not used directly - Echo TTS uses audio reference)
            output_dir: Output directory (defaults to self.output_dir)
            max_new_tokens: Max tokens for generation (ignored for Echo TTS)
            verbose: Print verbose output

        Returns:
            Tuple of (success, output_file_path, duration_seconds)
        """
        if output_dir is None:
            output_dir = self.output_dir
        if max_new_tokens is None:
            max_new_tokens = DEFAULTS["max_new_tokens"]

        # Reference text for voice generation - uses static text from config
        sample_text = DEFAULTS["static_voice_text"]

        # Validate description (kept for API compatibility, though not used)
        if not description or not description.strip():
            if verbose:
                print(f"  ERROR: Skipping '{character_name}' due to empty description")
            return False, None, 0

        # Get Echo TTS models (lazy loader)
        loader, fish_ae, pca_state, sample_fn, model_path = self.setup_tts_engine()

        try:
            # Load echo-tts models (this is where the heavy import happens)
            loader.load()

            # Prepare text prompt - Echo TTS expects speaker tags like [S1]
            # Since we're generating a base voice without reference, we use [S1]
            text_prompt = f"[S1] {sample_text}"

            if verbose:
                print(f"  Generating with text: {text_prompt[:50]}...")

            # Generate audio using Echo TTS pipeline
            # No speaker reference audio (speaker_audio=None) - generates generic voice
            audio_out, _ = loader.sample_fn(
                model=loader.echo_model,
                fish_ae=loader.fish_ae,
                pca_state=loader.pca_state,
                text_prompt=text_prompt,
                speaker_audio=None,  # No speaker reference - generates generic voice
                rng_seed=0,
            )

            if audio_out is None or audio_out.numel() == 0:
                print(f"    ERROR: No audio generated")
                return False, None, 0

            # Save output
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{character_name}.wav")

            # Echo TTS outputs at 44100 Hz
            sr = 44100
            import torchaudio
            torchaudio.save(output_file, audio_out[0].cpu(), sr)

            duration = len(audio_out[0]) / sr
            self.add_voice_path(character_name, output_file)

            if verbose:
                print(f"    Generated: {duration:.2f}s -> {output_file}")

            return True, output_file, duration

        except Exception as e:
            print(f"    Error generating voice with Echo TTS: {e}")
            print(f"    Exception type: {type(e).__name__}")
            traceback.print_exc()
            return False, None, 0

    # Note: _generate_voice_sample_echo_tts removed - echo-tts is only used for
    # voice cloning in stage 5 (audiobook generation), not for initial voice generation.
    # Initial voice samples always use MOSS-TTS.

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

        # Use the static voice text from config for high emotional range
        sample_text = DEFAULTS["static_voice_text"]

        # Validate description
        if not description or not description.strip():
            if verbose:
                print(f"  ERROR: Skipping '{character_name}' due to empty description")
            return False, None, 0

        # Get MOSS model and processor
        model, processor, _, _ = self.setup_tts_engine()

        try:
            # Build structured voice instruction for MOSS-TTS
            # Explicit format improves model's ability to follow voice style directions
            voice_instruction = f"Generate speech with this voice: {description}. Speak the following text in this voice style."

            if verbose:
                print(f"  Character: {character_name}")
                print(f"  Voice instruction: {voice_instruction[:200]}...")
                print(f"  Sample text: {sample_text}")

            conversations = [
                processor.build_user_message(text=sample_text, instruction=voice_instruction)
            ]

            # Prepare batch for generation
            with torch.no_grad():
                batch = processor(conversations, mode="generation")
                outputs = model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    max_new_tokens=max_new_tokens,
                    # MOSS-VoiceGenerator hyperparameters for Stage 1 voice design
                    audio_temperature=DEFAULTS["moss_voicegen_temperature"],
                    audio_top_p=DEFAULTS["moss_voicegen_top_p"],
                    audio_top_k=DEFAULTS["moss_voicegen_top_k"],
                    audio_repetition_penalty=DEFAULTS["moss_voicegen_repetition_penalty"],
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
            print(f"    Error generating voice with MOSS-TTS: {e}")
            print(f"    Exception type: {type(e).__name__}")
            traceback.print_exc()
            return False, None, 0

    def _convert_description_to_omni_instruct(self, description: str, verbose: bool = False) -> str:
        """Convert a character description to OmniVoice instruct format.

        OmniVoice expects comma-separated attributes like:
        "female, young adult, high pitch, british accent"

        Supported OmniVoice attributes (per official docs):
        - Gender: male, female
        - Age: child, teenager, young adult, middle-aged, elderly
        - Pitch: very low pitch, low pitch, moderate pitch, high pitch, very high pitch
        - Style: whisper
        - English Accents: american accent, british accent, australian accent, canadian accent,
          indian accent, chinese accent, korean accent, japanese accent, portuguese accent, russian accent
        - Chinese Dialects: 河南话，陕西话，四川话，贵州话，云南话，桂林话，济南话，石家庄话，
          甘肃话，宁夏话，青岛话，东北话

        Args:
            description: Voice description from LLM (e.g., "male. middle aged. high")

        Returns:
            Formatted instruct string for OmniVoice
        """
        # Normalize: replace periods with commas, strip whitespace, lowercase
        instruct = description.replace(".", ",")
        parts = [p.strip().lower() for p in instruct.split(",") if p.strip()]

        # Mapping from our terms to OmniVoice format
        gender_map = {"male": "male", "female": "female"}
        age_map = {
            "child": "child",
            "young": "young adult",
            "teen": "teenager",
            "teenager": "teenager",
            "young adult": "young adult",
            "middle aged": "middle-aged",
            "middle-aged": "middle-aged",
            "elderly": "elderly",
            "old": "elderly",
        }
        pitch_map = {
            "very low": "very low pitch",
            "low": "low pitch",
            "medium": "moderate pitch",
            "mid": "moderate pitch",
            "moderate": "moderate pitch",
            "high": "high pitch",
            "very high": "very high pitch",
        }
        accent_map = {
            "american": "american accent",
            "british": "british accent",
            "australian": "australian accent",
            "canadian": "canadian accent",
            "indian": "indian accent",
            "chinese": "chinese accent",
            "korean": "korean accent",
            "japanese": "japanese accent",
            "portuguese": "portuguese accent",
            "russian": "russian accent",
        }

        mapped_parts = []
        for part in parts:
            # Gender
            if part in gender_map:
                mapped_parts.append(gender_map[part])
            # Age
            elif part in age_map:
                mapped_parts.append(age_map[part])
            # Pitch
            elif part in pitch_map:
                mapped_parts.append(pitch_map[part])
            # Style
            elif part == "whisper":
                mapped_parts.append("whisper")
            # Accent (with or without "accent" suffix)
            elif part in accent_map:
                mapped_parts.append(accent_map[part])
            elif part.endswith(" accent"):
                mapped_parts.append(part)
            # Chinese dialects (pass through)
            elif any(c in part for c in "河南陕西四川贵云南桂济石甘宁青岛东北话"):
                mapped_parts.append(part)
            # Unknown - skip to avoid invalid attributes
            else:
                if verbose:
                    print(f"    Warning: Skipping unknown attribute '{part}'")

        return ", ".join(mapped_parts)

    def _generate_voice_sample_omni(
        self,
        character_name: str,
        description: str,
        output_dir: str = None,
        max_new_tokens: int = None,
        verbose: bool = False
    ) -> Tuple[bool, Optional[str], float]:
        """Generate a voice sample for a character using OmniVoice voice design.

        OmniVoice supports voice design from text descriptions without reference audio.
        The character description is converted to OmniVoice's instruct format.

        Args:
            character_name: Name of the character
            description: Voice description from LLM (e.g., "male. middle aged. high")
            output_dir: Output directory (defaults to self.output_dir)
            max_new_tokens: Max tokens for generation (not used by OmniVoice)
            verbose: Print verbose output

        Returns:
            Tuple of (success, output_file_path, duration_seconds)
        """
        if output_dir is None:
            output_dir = self.output_dir

        # Use the static voice text from config for high emotional range
        sample_text = DEFAULTS["static_voice_text"]

        # Validate description
        if not description or not description.strip():
            if verbose:
                print(f"  ERROR: Skipping '{character_name}' due to empty description")
            return False, None, 0

        # Get OmniVoice model
        model, _, _, _ = self.setup_tts_engine()

        # Try generation with the original description first
        instruct = self._convert_description_to_omni_instruct(description, verbose)

        if verbose:
            print(f"  Character: {character_name}")
            print(f"  OmniVoice instruct: {instruct}")
            print(f"  Sample text: {sample_text}")

        try:
            # Generate audio using voice design (per OmniVoice docs)
            audio = model.generate(
                text=sample_text,
                num_step=32,  # diffusion steps (or 16 for faster inference)
                class_temperature=0.5, # default is 0, but not enough diversity
                instruct=instruct,
            )

            if audio is None or len(audio) == 0 or audio[0].numel() == 0:
                return False, None, 0

            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{character_name}.wav")

            sr = 24000
            torchaudio.save(output_file, audio[0].cpu(), sr)

            duration = len(audio[0]) / sr
            self.add_voice_path(character_name, output_file)

            if verbose:
                print(f"    Generated: {duration:.2f}s -> {output_file}")

            return True, output_file, duration

        except ValueError as e:
            error_msg = str(e)
            # Check for conflicting instruct items error
            if "Conflicting instruct items" in error_msg or "Each category" in error_msg:
                if verbose:
                    print(f"    Conflict detected in instruct: {error_msg}")
                    print(f"    Retrying with simplified voice description...")

                # Try with a minimal fallback instruct (just gender if possible, otherwise nothing)
                fallback_instruct = self._get_fallback_instruct(description, verbose)
                if fallback_instruct:
                    if verbose:
                        print(f"  Retrying with fallback instruct: {fallback_instruct}")

                    try:
                        audio = model.generate(
                            text=sample_text,
                            num_step=32,
                            class_temperature=3.0,
                            instruct=fallback_instruct,
                        )

                        if audio is None or len(audio) == 0 or audio[0].numel() == 0:
                            return False, None, 0

                        os.makedirs(output_dir, exist_ok=True)
                        output_file = os.path.join(output_dir, f"{character_name}.wav")

                        sr = 24000
                        torchaudio.save(output_file, audio[0].cpu(), sr)

                        duration = len(audio[0]) / sr
                        self.add_voice_path(character_name, output_file)

                        if verbose:
                            print(f"    Generated (fallback): {duration:.2f}s -> {output_file}")

                        return True, output_file, duration

                    except Exception as e2:
                        print(f"    Fallback generation also failed: {e2}")
                        print(f"    Exception type: {type(e2).__name__}")
                        traceback.print_exc()
                        return False, None, 0
                else:
                    if verbose:
                        print(f"    No fallback instruct available, generation failed")
                    return False, None, 0
            else:
                # Different error, re-raise as normal
                return self._handle_voice_generation_error("OmniVoice", e, verbose)

        except Exception as e:
            return self._handle_voice_generation_error("OmniVoice", e, verbose)

    def _generate_voice_sample_vox(
        self,
        character_name: str,
        description: str,
        output_dir: str = None,
        max_new_tokens: int = None,
        verbose: bool = False
    ) -> Tuple[bool, Optional[str], float]:
        """Generate a voice sample for a character using VoxCPM2 voice design.

        Args:
            character_name: Name of the character
            description: Voice description from LLM (e.g., "male. middle aged. high")
            output_dir: Output directory (defaults to self.output_dir)
            max_new_tokens: Max tokens for generation (not used by VoxCPM)
            verbose: Print verbose output

        Returns:
            Tuple of (success, output_file_path, duration_seconds)
        """
        if output_dir is None:
            output_dir = self.output_dir

        # Use the static voice text from config for high emotional range
        sample_text = DEFAULTS["static_voice_text"]

        # Validate description
        if not description or not description.strip():
            if verbose:
                print(f"  ERROR: Skipping '{character_name}' due to empty description")
            return False, None, 0

        # Get VoxCPM model
        model, _, _, _ = self.setup_tts_engine()

        # Wrap description in parentheses for VoxCPM voice style
        voice_style = f"({description})"
        instruct_text = f"{voice_style}{sample_text}"

        if verbose:
            print(f"  Character: {character_name}")
            print(f"  VoxCPM instruct: {instruct_text[:100]}...")
            print(f"  Sample text: {sample_text}")

        try:
            # Generate audio using VoxCPM voice design
            # Based on VoxCPM docs: cfg_value=2.0, inference_timesteps=10
            wav = model.generate(
                text=instruct_text,
                cfg_value=2.0,
                inference_timesteps=10,
            )

            if wav is None or len(wav) == 0:
                return False, None, 0

            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{character_name}.wav")

            # VoxCPM outputs 48kHz audio
            sr = model.tts_model.sample_rate
            sf.write(output_file, wav, sr)

            duration = len(wav) / sr
            self.add_voice_path(character_name, output_file)

            if verbose:
                print(f"    Generated: {duration:.2f}s -> {output_file}")

            return True, output_file, duration

        except Exception as e:
            return self._handle_voice_generation_error("VoxCPM", e, verbose)

    def _handle_voice_generation_error(self, engine_name: str, error: Exception, verbose: bool = False) -> Tuple[bool, None, int]:
        """Centralized error handler for voice generation.

        Args:
            engine_name: Name of the TTS engine for error messages
            error: The exception that occurred
            verbose: Print verbose output

        Returns:
            Failure tuple (False, None, 0)
        """
        if verbose:
            print(f"    Error generating voice with {engine_name}: {error}")
            print(f"    Exception type: {type(error).__name__}")
        traceback.print_exc()
        return False, None, 0

    def _get_fallback_instruct(self, description: str, verbose: bool = False) -> Optional[str]:
        """Extract a minimal, non-conflicting instruct from a description.

        When the original description has conflicting attributes (e.g., multiple
        genders or ages), this extracts just the first valid gender as a fallback.

        Args:
            description: Voice description from LLM
            verbose: Print verbose output

        Returns:
            A minimal instruct string (e.g., "male") or None if nothing valid found
        """
        parts = [p.strip().lower() for p in description.replace(".", ",").split(",") if p.strip()]

        # Gender mapping
        gender_map = {"male": "male", "female": "female"}

        # Find first valid gender
        for part in parts:
            if part in gender_map:
                return gender_map[part]

        # No valid gender found, return None (caller will handle)
        return None

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

        # Load the voice sample
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