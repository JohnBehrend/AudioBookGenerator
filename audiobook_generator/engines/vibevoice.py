"""VibeVoice engine implementation."""

from typing import Tuple, Any, Optional
from pathlib import Path

import torch

from config import DEFAULTS, TTS_MODEL_PATHS
from utils import _get_attn_implementation
from .base import TTSEngine


class VibeVoiceEngine(TTSEngine):
    """VibeVoice voice cloning engine."""

    def setup(self, device: str, turbo: bool = False) -> Tuple[Any, Any]:
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

        model_path = TTS_MODEL_PATHS["vibevoice"]
        attn_impl = _get_attn_implementation()
        attn_kwargs = {"attn_implementation": attn_impl} if attn_impl else {}

        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            **attn_kwargs,
        )
        model.eval()
        processor = VibeVoiceProcessor.from_pretrained(model_path)
        return model, processor

    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: Path,
        device: str,
        verbose: bool = False,
    ) -> Tuple[bool, Optional[str], float]:
        import torchaudio
        from config import DEFAULTS

        model, processor = self._get_model(device)
        sample_text = DEFAULTS["static_voice_text"]

        if not description or not description.strip():
            if verbose:
                print(f"  ERROR: Skipping '{character_name}' due to empty description")
            return False, None, 0

        inputs = processor(
            text=[sample_text],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                max_new_tokens=DEFAULTS["max_new_tokens"],
            )

        wavs = outputs.cpu().numpy()
        sr = model.config.sampling_rate

        if not wavs or len(wavs) == 0:
            return False, None, 0

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / f"{character_name}.wav")
        import soundfile as sf
        sf.write(output_file, wavs[0], sr)

        duration = len(wavs[0]) / sr
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
        model, processor = self._get_model(device)

        inputs = processor(
            text=["Speaker 1: ? " + text],
            voice_samples=[voice_path],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=DEFAULTS["max_new_tokens"],
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            do_sample=False,
            verbose=False,
        )

        processor.save_audio(
            outputs.speech_outputs[0],
            output_path=output_path,
        )
        return True

    def _get_model(self, device: str):
        if not hasattr(self, "_cached_model"):
            self._cached_model, self._cached_processor = self.setup(device)
        return self._cached_model, self._cached_processor
