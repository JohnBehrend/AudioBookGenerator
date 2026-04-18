"""MOSS-TTS engine implementation."""

from typing import Tuple, Any, Optional
from pathlib import Path

import torch

from config import DEFAULTS, TTS_MODEL_PATHS
from utils import _get_attn_implementation
from .base import TTSEngine


class MossEngine(TTSEngine):
    """MOSS-TTS voice cloning engine."""

    def setup(self, device: str, turbo: bool = False) -> Tuple[Any, Any]:
        from transformers import AutoModel, AutoProcessor

        model_path = TTS_MODEL_PATHS["moss"]
        attn_impl = _get_attn_implementation()
        attn_kwargs = {"attn_implementation": attn_impl} if attn_impl else {}

        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        # Fix: Ensure model.config has num_hidden_layers for DynamicCache compatibility
        if not hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = getattr(
                model.config, "num_layers",
                getattr(model.config, "n_layers", 32)
            )

        return model, processor

    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: Path,
        device: str,
        verbose: bool = False,
    ) -> Tuple[bool, Optional[str], float]:
        from transformers import AutoModel, AutoProcessor
        from config import DEFAULTS, TTS_MODEL_PATHS
        import torchaudio
        import os

        model, processor = self._get_model(device)

        sample_text = DEFAULTS["static_voice_text"]

        if not description or not description.strip():
            if verbose:
                print(f"  ERROR: Skipping '{character_name}' due to empty description")
            return False, None, 0

        voice_instruction = f"Generate speech with this voice: {description}. Speak the following text in this voice style."

        if verbose:
            print(f"  Character: {character_name}")
            print(f"  Voice instruction: {voice_instruction[:200]}...")

        conversations = [
            processor.build_user_message(text=sample_text, instruction=voice_instruction)
        ]

        with torch.no_grad():
            batch = processor(conversations, mode="generation")
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=DEFAULTS["max_new_tokens"],
                audio_temperature=DEFAULTS["moss_voicegen_temperature"],
                audio_top_p=DEFAULTS["moss_voicegen_top_p"],
                audio_top_k=DEFAULTS["moss_voicegen_top_k"],
                audio_repetition_penalty=DEFAULTS["moss_voicegen_repetition_penalty"],
            )

            message = processor.decode(outputs)[0]
            audio = message.audio_codes_list[0]
            sr = processor.model_config.sampling_rate

        if audio is None or (hasattr(audio, "numel") and audio.numel() == 0):
            return False, None, 0

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / f"{character_name}.wav")
        torchaudio.save(output_file, audio.unsqueeze(0), sr)

        duration = len(audio) / sr
        return True, output_file, duration

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
        from transformers import AutoModel, AutoProcessor
        from config import DEFAULTS, TTS_MODEL_PATHS
        import torchaudio

        model, processor = self._get_model(device)

        conversations = [
            processor.build_user_message(text=text, reference=[voice_path])
        ]

        with torch.no_grad():
            batch = processor(conversations, mode="generation")
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=DEFAULTS["max_new_tokens"],
                audio_temperature=DEFAULTS["moss_audio_temperature"],
                audio_top_p=DEFAULTS["moss_audio_top_p"],
                audio_top_k=DEFAULTS["moss_audio_top_k"],
                audio_repetition_penalty=DEFAULTS["moss_audio_repetition_penalty"],
            )

            message = processor.decode(outputs)[0]
            audio = message.audio_codes_list[0]
            sr = processor.model_config.sampling_rate

        if audio is None or audio.numel() == 0:
            return False

        torchaudio.save(output_path, audio.unsqueeze(0), sr)
        return True

    def _get_model(self, device: str):
        """Get or lazily initialize the model."""
        if not hasattr(self, "_cached_model"):
            self._cached_model, self._cached_processor = self.setup(device)
        return self._cached_model, self._cached_processor
