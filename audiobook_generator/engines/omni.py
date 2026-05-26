"""OmniVoice engine implementation."""

from __future__ import annotations

from multiprocessing import Queue
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

from ..config import DEFAULTS
from .base import TTSEngine


class OmniEngine(TTSEngine):
    """OmniVoice voice cloning engine."""

    ENV_NAME = "omni"

    def __init__(self, device: str = "cuda", turbo: bool = False):
        self._device = device
        self._turbo = turbo

    @classmethod
    def _run_worker(cls, request_queue: Queue, response_queue: Queue) -> None:
        from omnivoice import OmniVoice
        import torch
        import torchaudio
        import soundfile as sf

        model = None

        def load_model(device: str) -> None:
            nonlocal model
            if model is not None:
                return
            model_path = "drbaph/OmniVoice-bf16"
            model = OmniVoice.from_pretrained(
                model_path,
                device_map=device,
                dtype=torch.float16,
            )
            try:
                model.load_asr_model()
            except Exception as e:
                print(f"  Warning: Could not pre-load ASR model: {e}")

        response_queue.put({"type": "ready"})

        while True:
            try:
                req = request_queue.get(timeout=1)
            except Exception:
                continue

            if req.get("type") == "shutdown":
                break
            if req.get("type") != "request":
                continue

            req_id = req["id"]
            method = req["method"]
            kwargs = req["kwargs"]
            device = kwargs.get("device", "cuda")

            try:
                load_model(device)
                assert model is not None

                if method == "generate_voice_sample":
                    character_name = kwargs["character_name"]
                    description = kwargs["description"]
                    output_dir = kwargs["output_dir"]

                    if not description or not description.strip():
                        response_queue.put({"id": req_id, "success": False})
                        continue

                    sample_text = DEFAULTS["static_voice_text"]
                    instruct = _convert_description_to_instruct(description)

                    try:
                        audio = model.generate(
                            text=sample_text,
                            num_step=32,
                            class_temperature=0.5,
                            instruct=instruct,
                        )
                        if audio is None or len(audio) == 0:
                            response_queue.put({"id": req_id, "success": False})
                            continue
                        audio_arr = audio[0]
                        if hasattr(audio_arr, 'numel'):
                            audio_len = audio_arr.numel()
                        else:
                            audio_len = len(audio_arr)
                        if audio_len == 0:
                            response_queue.put({"id": req_id, "success": False})
                            continue

                        out_dir = Path(output_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        output_file = str(out_dir / f"{character_name}.wav")
                        if hasattr(audio[0], 'cpu'):
                            audio_np = audio[0].cpu().numpy()
                        else:
                            audio_np = audio[0]
                        sf.write(output_file, audio_np, 24000)
                        duration = len(audio[0]) / 24000
                        response_queue.put({"id": req_id, "success": True, "output_file": output_file, "duration": duration})

                    except ValueError as e:
                        error_msg = str(e)
                        if "Conflicting instruct items" in error_msg or "Each category" in error_msg:
                            fallback = _get_fallback_instruct(description)
                            if fallback:
                                try:
                                    audio = model.generate(
                                        text=sample_text,
                                        num_step=32,
                                        class_temperature=3.0,
                                        instruct=fallback,
                                    )
                                    if audio is None or len(audio) == 0:
                                        response_queue.put({"id": req_id, "success": False})
                                        continue
                                    audio_arr = audio[0]
                                    audio_len = audio_arr.numel() if hasattr(audio_arr, 'numel') else len(audio_arr)
                                    if audio_len == 0:
                                        response_queue.put({"id": req_id, "success": False})
                                        continue
                                    out_dir = Path(output_dir)
                                    out_dir.mkdir(parents=True, exist_ok=True)
                                    output_file = str(out_dir / f"{character_name}.wav")
                                    if hasattr(audio[0], 'cpu'):
                                        audio_np = audio[0].cpu().numpy()
                                    else:
                                        audio_np = audio[0]
                                    sf.write(output_file, audio_np, 24000)
                                    duration = len(audio[0]) / 24000
                                    response_queue.put({"id": req_id, "success": True, "output_file": output_file, "duration": duration})
                                except Exception:
                                    response_queue.put({"id": req_id, "success": False})
                            else:
                                response_queue.put({"id": req_id, "success": False})
                        else:
                            response_queue.put({"id": req_id, "success": False})

                elif method == "generate_line":
                    text = kwargs["text"]
                    voice_path = kwargs["voice_path"]
                    output_path = kwargs["output_path"]
                    ref_text = kwargs.get("ref_text", DEFAULTS["static_voice_text"])

                    ref_audio_np, ref_sr = sf.read(voice_path)
                    if len(ref_audio_np.shape) > 1:
                        ref_audio_np = ref_audio_np.mean(axis=1)
                    ref_audio_np = ref_audio_np.astype(np.float32)
                    ref_audio = torch.from_numpy(ref_audio_np)

                    if ref_audio.numel() == 0:
                        response_queue.put({"id": req_id, "success": False})
                        continue

                    audio = model.generate(
                        text=text,
                        ref_audio=(ref_audio, ref_sr),
                        ref_text=ref_text,
                        preprocess_prompt=False,
                    )
                    if audio is None or len(audio) == 0:
                        response_queue.put({"id": req_id, "success": False})
                        continue
                    audio_arr = audio[0]
                    audio_len = audio_arr.numel() if hasattr(audio_arr, 'numel') else len(audio_arr)
                    if audio_len == 0:
                        response_queue.put({"id": req_id, "success": False})
                        continue

                    if hasattr(audio[0], 'cpu'):
                        sf.write(output_path, audio[0].cpu().numpy(), 24000)
                    else:
                        sf.write(output_path, audio[0], 24000)
                    response_queue.put({"id": req_id, "success": True})

                else:
                    response_queue.put({"id": req_id, "error": f"Unknown method: {method}"})

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                response_queue.put({"id": req_id, "error": str(e), "traceback": tb})

    def setup(self, device: str, turbo: bool = False) -> Tuple[Any, None]:
        return None, None

    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: Path,
        device: str,
        verbose: bool = False,
    ) -> Tuple[bool, Optional[str], float]:
        return super().generate_voice_sample(character_name, description, output_dir, device, verbose)

    def generate_line(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        device: str,
        validation_model: Optional[Any] = None,
        cfg_scale: float = 1.3,
        max_new_tokens: int = 19200,
        verbose: bool = False,
        ref_text: Optional[str] = None,
    ) -> bool:
        # Pre-compute ref_text in main process (WhisperModel not serializable)
        if ref_text is None:
            ref_text = self._get_ref_text(voice_path, validation_model, verbose)

        resp = self._worker_request(
            "generate_line",
            text=text,
            voice_path=voice_path,
            output_path=output_path,
            ref_text=ref_text,
        )
        self._clear_cuda_cache()
        return resp.get("success", False)

    def _get_ref_text(self, voice_path: str, validation_model: Optional[Any], verbose: bool) -> str:
        try:
            from ..utils import transcribe_audio_with_whisper
            ref_text, _, _ = transcribe_audio_with_whisper(validation_model, voice_path)
            return ref_text
        except Exception:
            return DEFAULTS["static_voice_text"]


def _convert_description_to_instruct(description: str) -> str:
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

    return ", ".join(mapped_parts)


def _get_fallback_instruct(description: str) -> Optional[str]:
    parts = [p.strip().lower() for p in description.replace(".", ",").split(",") if p.strip()]
    for part in parts:
        if part in ("male", "female"):
            return part
    return None
