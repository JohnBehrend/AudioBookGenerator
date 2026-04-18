"""OmniVoice engine implementation."""

from typing import Tuple, Any, Optional
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

from config import DEFAULTS
from .base import TTSEngine


class OmniEngine(TTSEngine):
    """OmniVoice voice cloning engine."""

    def setup(self, device: str, turbo: bool = False) -> Tuple[Any, None]:
        from omnivoice import OmniVoice

        model_path = "drbaph/OmniVoice-bf16"
        model = OmniVoice.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.float16,
        )

        # Pre-load ASR model to avoid repeated downloads during generation
        try:
            model.load_asr_model()
        except Exception as e:
            print(f"  Warning: Could not pre-load ASR model: {e}")

        return model, None

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

        model, _ = self._get_model(device)
        sample_text = DEFAULTS["static_voice_text"]

        if not description or not description.strip():
            if verbose:
                print(f"  ERROR: Skipping '{character_name}' due to empty description")
            return False, None, 0

        instruct = self._convert_description_to_instruct(description, verbose)

        if verbose:
            print(f"  Character: {character_name}")
            print(f"  OmniVoice instruct: {instruct}")

        try:
            audio = model.generate(
                text=sample_text,
                num_step=32,
                class_temperature=0.5,
                instruct=instruct,
            )

            if audio is None or len(audio) == 0 or audio[0].numel() == 0:
                return False, None, 0

            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(output_dir / f"{character_name}.wav")
            torchaudio.save(output_file, audio[0].cpu(), 24000)

            duration = len(audio[0]) / 24000
            return True, output_file, duration

        except ValueError as e:
            error_msg = str(e)
            if "Conflicting instruct items" in error_msg or "Each category" in error_msg:
                if verbose:
                    print(f"    Conflict detected in instruct: {error_msg}")
                    print(f"    Retrying with simplified voice description...")

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

                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_file = str(output_dir / f"{character_name}.wav")
                        torchaudio.save(output_file, audio[0].cpu(), 24000)

                        duration = len(audio[0]) / 24000
                        return True, output_file, duration

                    except Exception as e2:
                        print(f"    Fallback generation also failed: {e2}")
                        return False, None, 0
                else:
                    return False, None, 0
            else:
                print(f"    Error generating voice with OmniVoice: {e}")
                return False, None, 0

        except Exception as e:
            print(f"    Error generating voice with OmniVoice: {e}")
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
        from config import DEFAULTS

        model, _ = self._get_model(device)

        ref_text = self._get_ref_text(voice_path, validation_model, verbose)
        ref_audio_np, ref_sr = sf.read(voice_path)

        if len(ref_audio_np.shape) > 1:
            ref_audio_np = ref_audio_np.mean(axis=1)

        ref_audio_np = ref_audio_np.astype(np.float32)
        ref_audio = torch.from_numpy(ref_audio_np)

        if ref_audio.numel() == 0:
            return False

        audio = model.generate(
            text=text,
            ref_audio=(ref_audio, ref_sr),
            ref_text=ref_text,
            preprocess_prompt=False,
        )

        if audio is None or len(audio) == 0 or audio[0].numel() == 0:
            return False

        torchaudio.save(output_path, audio[0].cpu(), 24000)
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

    def _convert_description_to_instruct(self, description: str, verbose: bool = False) -> str:
        instruct = description.replace(".", ",")
        parts = [p.strip().lower() for p in instruct.split(",") if p.strip()]

        gender_map = {"male": "male", "female": "female"}
        age_map = {
            "child": "child", "young": "young adult", "teen": "teenager",
            "teenager": "teenager", "young adult": "young adult",
            "middle aged": "middle-aged", "middle-aged": "middle-aged",
            "elderly": "elderly", "old": "elderly",
        }
        pitch_map = {
            "very low": "very low pitch", "low": "low pitch",
            "medium": "moderate pitch", "mid": "moderate pitch",
            "moderate": "moderate pitch", "high": "high pitch",
            "very high": "very high pitch",
        }
        accent_map = {
            "american": "american accent", "british": "british accent",
            "australian": "australian accent", "canadian": "canadian accent",
            "indian": "indian accent", "chinese": "chinese accent",
            "korean": "korean accent", "japanese": "japanese accent",
            "portuguese": "portuguese accent", "russian": "russian accent",
        }

        mapped_parts = []
        for part in parts:
            if part in gender_map:
                mapped_parts.append(gender_map[part])
            elif part in age_map:
                mapped_parts.append(age_map[part])
            elif part in pitch_map:
                mapped_parts.append(pitch_map[part])
            elif part == "whisper":
                mapped_parts.append("whisper")
            elif part in accent_map:
                mapped_parts.append(accent_map[part])
            elif part.endswith(" accent"):
                mapped_parts.append(part)
            elif any(c in part for c in "河南陕西四川贵云南桂济石甘宁青岛东北话"):
                mapped_parts.append(part)
            else:
                if verbose:
                    print(f"    Warning: Skipping unknown attribute '{part}'")

        return ", ".join(mapped_parts)

    def _get_fallback_instruct(self, description: str, verbose: bool = False) -> Optional[str]:
        parts = [p.strip().lower() for p in description.replace(".", ",").split(",") if p.strip()]
        for part in parts:
            if part in ("male", "female"):
                return part
        return None
