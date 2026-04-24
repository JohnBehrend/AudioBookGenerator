"""VoxCPM engine implementation."""

from __future__ import annotations

from multiprocessing import Queue
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

import soundfile as sf

from ..config import DEFAULTS
from .base import TTSEngine


class VoxEngine(TTSEngine):
    """VoxCPM2 voice cloning engine."""

    ENV_NAME = "vox"

    def __init__(self, device: str = "cuda", turbo: bool = False):
        self._device = device
        self._turbo = turbo

    @classmethod
    def _run_worker(cls, request_queue: Queue, response_queue: Queue) -> None:
        from voxcpm import VoxCPM
        import torch
        import torchaudio
        import numpy as np

        model = None

        def load_model(device: str) -> None:
            nonlocal model
            if model is not None:
                return
            model_path = "openbmb/VoxCPM2"
            model = VoxCPM.from_pretrained(model_path, load_denoiser=False)

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
                    voice_style = f"({description})"
                    instruct_text = f"{voice_style}{sample_text}"

                    try:
                        wav = model.generate(
                            text=instruct_text,
                            cfg_value=2.0,
                            inference_timesteps=10,
                        )
                        if wav is None or len(wav) == 0:
                            response_queue.put({"id": req_id, "success": False})
                            continue

                        out_dir = Path(output_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        output_file = str(out_dir / f"{character_name}.wav")
                        sr = model.tts_model.sample_rate
                        sf.write(output_file, wav, sr)
                        duration = len(wav) / sr
                        response_queue.put({"id": req_id, "success": True, "output_file": output_file, "duration": duration})

                    except Exception:
                        response_queue.put({"id": req_id, "success": False})

                elif method == "generate_line":
                    text = kwargs["text"]
                    voice_path = kwargs["voice_path"]
                    output_path = kwargs["output_path"]
                    ref_text = kwargs.get("ref_text", DEFAULTS["static_voice_text"])

                    wav = model.generate(
                        text=text,
                        prompt_wav_path=voice_path,
                        prompt_text=ref_text,
                        reference_wav_path=voice_path,
                        cfg_value=1.5,
                    )
                    if wav is None or len(wav) == 0:
                        response_queue.put({"id": req_id, "success": False})
                        continue

                    sr = 48000
                    torchaudio.save(output_path, torch.from_numpy(wav), sr)
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
        if not description or not description.strip():
            if verbose:
                print(f"  ERROR: Skipping '{character_name}' due to empty description")
            return False, None, 0

        resp = self._worker_request(
            "generate_voice_sample",
            character_name=character_name,
            description=description,
            output_dir=str(output_dir),
        )
        self._clear_cuda_cache()
        success = resp.get("success", False)
        return success, resp.get("output_file"), resp.get("duration", 0)

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
        # Pre-compute ref_text in main process (WhisperModel not serializable)
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

    def _get_ref_text(self, voice_path: str, validation_model, verbose: bool) -> str:
        try:
            from ..utils import transcribe_audio_with_whisper
            ref_text, _, _ = transcribe_audio_with_whisper(validation_model, voice_path)
            return ref_text
        except Exception:
            return DEFAULTS["static_voice_text"]
