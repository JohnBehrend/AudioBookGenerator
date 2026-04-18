"""VoxCPM engine implementation."""

from typing import Tuple, Any, Optional
from pathlib import Path

import torch
import soundfile as sf

from config import DEFAULTS
from .base import TTSEngine


class VoxEngine(TTSEngine):
    """VoxCPM2 voice cloning engine."""

    def setup(self, device: str, turbo: bool = False) -> Tuple[Any, None]:
        from voxcpm import VoxCPM

        model_path = "openbmb/VoxCPM2"
        model = VoxCPM.from_pretrained(model_path, load_denoiser=False)
        return model, None

    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: Path,
        device: str,
        verbose: bool = False,
    ) -> Tuple[bool, Optional[str], float]:
        from config import DEFAULTS

        model, _ = self._get_model(device)
        sample_text = DEFAULTS["static_voice_text"]

        if not description or not description.strip():
            if verbose:
                print(f"  ERROR: Skipping '{character_name}' due to empty description")
            return False, None, 0

        voice_style = f"({description})"
        instruct_text = f"{voice_style}{sample_text}"

        if verbose:
            print(f"  Character: {character_name}")
            print(f"  VoxCPM instruct: {instruct_text[:100]}...")

        try:
            wav = model.generate(
                text=instruct_text,
                cfg_value=2.0,
                inference_timesteps=10,
            )

            if wav is None or len(wav) == 0:
                return False, None, 0

            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(output_dir / f"{character_name}.wav")
            sr = model.tts_model.sample_rate
            sf.write(output_file, wav, sr)

            duration = len(wav) / sr
            return True, output_file, duration

        except Exception as e:
            print(f"    Error generating voice with VoxCPM: {e}")
            return False, None, 0

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
        import torchaudio

        model, _ = self._get_model(device)

        ref_text = self._get_ref_text(voice_path, validation_model, verbose)

        wav = model.generate(
            text=text,
            prompt_wav_path=voice_path,
            prompt_text=ref_text,
            reference_wav_path=voice_path,
            cfg_value=1.5,
        )

        if wav is None or len(wav) == 0:
            return False

        sr = 48000
        torchaudio.save(output_path, torch.from_numpy(wav), sr)
        return True

    def _get_model(self, device: str):
        if not hasattr(self, "_cached_model"):
            self._cached_model, self._cached_processor = self.setup(device)
        return self._cached_model, self._cached_processor

    def _get_ref_text(self, voice_path: str, validation_model, verbose: bool) -> str:
        try:
            from ..utils import transcribe_audio_with_whisper
            ref_text, _, _ = transcribe_audio_with_whisper(validation_model, voice_path)
            return ref_text
        except Exception:
            return DEFAULTS["static_voice_text"]
