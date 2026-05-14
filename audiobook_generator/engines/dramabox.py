"""Dramabox TTS engine adapter."""

from __future__ import annotations

import os
from multiprocessing import Queue
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ..config import DEFAULTS
from .base import TTSEngine


DRAMABOX_DIR = Path(__file__).parent / "dramabox"


class DramaboxEngine(TTSEngine):
    """Dramabox expressive TTS engine with voice cloning."""

    ENV_NAME = "dramabox"

    def __init__(self, device: str = "cuda", turbo: bool = False):
        self._device = device
        self._turbo = turbo

    @classmethod
    def _run_worker(cls, request_queue: Queue, response_queue: Queue) -> None:
        import sys
        import soundfile as sf
        import torch

        sys.path.insert(0, str(DRAMABOX_DIR))
        sys.path.insert(0, str(DRAMABOX_DIR / "ltx2"))
        sys.path.insert(0, str(DRAMABOX_DIR / "src"))

        from inference_server import TTSServer

        server = None

        def load_server(device: str) -> None:
            nonlocal server
            if server is not None:
                return
            checkpoint = os.environ.get(
                "DRAMABOX_CHECKPOINT",
                str(Path.home() / ".cache" / "huggingface" / "hub" / "models--ResembleAI--Dramabox" / "snapshots" / "404f967f653fa1170dc15a9d1ddd3fdb9a0a842d" / "dramabox-dit-v1.safetensors"),
            )
            full_checkpoint = os.environ.get(
                "LTX_FULL_CHECKPOINT",
                str(Path.home() / ".cache" / "huggingface" / "hub" / "models--Lightricks--LTX-2.3" / "snapshots" / "76730e634e70a28f4e8d51f5e29c08e40e2d8e74" / "ltx-2.3-22b-dev.safetensors"),
            )
            gemma_root = os.environ.get(
                "GEMMA_DIR",
                str(Path.home() / ".cache" / "dramabox" / "models--unsloth--gemma-3-12b-it-bnb-4bit" / "snapshots" / "826e729dbaeea4ecb143738eed2bcf3539ebf7bf"),
            )
            server = TTSServer(
                checkpoint=checkpoint,
                full_checkpoint=full_checkpoint,
                gemma_root=gemma_root,
                device=device,
                dtype="bf16",
                compile_model=True,
                bnb_4bit=True,
            )

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
                load_server(device)
                assert server is not None

                if method == "generate_voice_sample":
                    character_name = kwargs["character_name"]
                    description = kwargs["description"]
                    output_dir = kwargs["output_dir"]

                    if not description or not description.strip():
                        response_queue.put({"id": req_id, "success": False})
                        continue

                    prompt = _convert_description_to_prompt(description)
                    sample_text = DEFAULTS["static_voice_text"]
                    full_prompt = f'{prompt} "{sample_text}"'

                    out_dir = Path(output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    output_file = str(out_dir / f"{character_name}.wav")

                    try:
                        server.generate_to_file(
                            prompt=full_prompt,
                            output=output_file,
                            cfg_scale=2.5,
                            stg_scale=1.5,
                            duration_multiplier=1.1,
                            seed=42,
                            watermark=False,
                        )
                        audio, sr = sf.read(output_file)
                        duration = len(audio) / sr
                        response_queue.put({"id": req_id, "success": True, "output_file": output_file, "duration": duration})
                    except Exception:
                        response_queue.put({"id": req_id, "success": False})

                elif method == "generate_line":
                    text = kwargs["text"]
                    voice_path = kwargs["voice_path"]
                    output_path = kwargs["output_path"]
                    cfg_scale = kwargs.get("cfg_scale", 2.5)
                    stg_scale = kwargs.get("stg_scale", 1.5)

                    try:
                        server.generate_to_file(
                            prompt=text,
                            output=output_path,
                            voice_ref=voice_path if voice_path else None,
                            cfg_scale=cfg_scale,
                            stg_scale=stg_scale,
                            duration_multiplier=1.1,
                            seed=42,
                            watermark=False,
                        )
                        response_queue.put({"id": req_id, "success": True})
                    except Exception:
                        response_queue.put({"id": req_id, "success": False})

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
        validation_model: Optional[Any] = None,
        cfg_scale: float = 1.3,
        max_new_tokens: int = 19200,
        verbose: bool = False,
    ) -> bool:
        resp = self._worker_request(
            "generate_line",
            text=text,
            voice_path=voice_path,
            output_path=output_path,
            cfg_scale=2.5,
            stg_scale=1.5,
        )
        self._clear_cuda_cache()
        return resp.get("success", False)


def _convert_description_to_prompt(description: str) -> str:
    """Convert a character description into a Dramabox-style prompt."""
    prompt = description.strip()
    if not prompt.endswith("."):
        prompt = prompt + "."
    return prompt