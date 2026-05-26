"""MOSS-TTS engine implementation."""

from __future__ import annotations

import types
from multiprocessing import Queue
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ..config import DEFAULTS, TTS_MODEL_PATHS
from ..utils import _get_attn_implementation
from .base import TTSEngine


class MossEngine(TTSEngine):
    """MOSS-TTS voice cloning engine."""

    ENV_NAME = "moss"

    def __init__(self, device: str = "cuda", turbo: bool = False):
        self._device = device
        self._turbo = turbo

    @classmethod
    def _run_worker(cls, request_queue: Queue, response_queue: Queue) -> None:
        from transformers import AutoModel, AutoProcessor
        import torch
        import soundfile as sf

        model = None
        processor = None

        def load_model(device: str) -> None:
            nonlocal model, processor
            if model is not None:
                return
            from ..config import TTS_MODEL_PATHS
            from ..utils import _get_attn_implementation

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

            if not hasattr(model.config, "num_hidden_layers"):
                model.config.num_hidden_layers = getattr(
                    model.config, "num_layers",
                    getattr(model.config, "n_layers", 32)
                )

            # Patch: MOSS model calls _get_initial_cache_position with custom signature
            # (cur_len, device, model_kwargs) but GenerationMixin expects (input_ids, model_kwargs).
            # The method may also be missing in some transformers versions.
            if not hasattr(model, "_get_initial_cache_position"):
                def _get_initial_cache_position(*args, **kwargs):
                    # Handle both old signature (cur_len, device, model_kwargs)
                    # and new signature (input_ids, model_kwargs)
                    if len(args) == 3:
                        cur_len, device, model_kwargs = args
                    elif len(args) == 2:
                        input_ids, model_kwargs = args
                        cur_len = input_ids.shape[-1]
                        device = input_ids.device
                    else:
                        model_kwargs = kwargs.get("model_kwargs", {})
                        cur_len = 0
                        device = "cpu"
                    model_kwargs["cache_position"] = torch.arange(
                        cur_len, device=device
                    )
                    return model_kwargs
                model._get_initial_cache_position = types.MethodType(
                    _get_initial_cache_position, model
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
                load_model(device)
                assert model is not None and processor is not None

                if method == "generate_voice_sample":
                    character_name = kwargs["character_name"]
                    description = kwargs["description"]
                    output_dir = kwargs["output_dir"]

                    if not description or not description.strip():
                        response_queue.put({"id": req_id, "success": False})
                        continue

                    sample_text = DEFAULTS["static_voice_text"]
                    voice_instruction = (
                        f"Generate speech with this voice: {description}. "
                        f"Speak the following text in this voice style."
                    )
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
                        response_queue.put({"id": req_id, "success": False})
                        continue

                    out_dir = Path(output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    output_file = str(out_dir / f"{character_name}.wav")
                    if hasattr(audio, 'cpu'):
                        audio_np = audio.cpu().numpy()
                    else:
                        audio_np = audio
                    if audio_np.ndim > 1:
                        audio_np = audio_np.squeeze()
                    sf.write(output_file, audio_np, sr)
                    duration = len(audio) / sr
                    response_queue.put({"id": req_id, "success": True, "output_file": output_file, "duration": duration})

                elif method == "generate_line":
                    text = kwargs["text"]
                    voice_path = kwargs["voice_path"]
                    output_path = kwargs["output_path"]

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
                        response_queue.put({"id": req_id, "success": False})
                        continue

                    if hasattr(audio, 'cpu'):
                        audio_np = audio.cpu().numpy()
                    else:
                        audio_np = audio
                    if audio_np.ndim > 1:
                        audio_np = audio_np.squeeze()
                    sf.write(output_path, audio_np, sr)
                    response_queue.put({"id": req_id, "success": True})

                else:
                    response_queue.put({"id": req_id, "error": f"Unknown method: {method}"})

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                response_queue.put({"id": req_id, "error": str(e), "traceback": tb})

    def setup(self, device: str, turbo: bool = False) -> Tuple[Any, Any]:
        return None, None
