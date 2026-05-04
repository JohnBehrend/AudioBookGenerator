"""VibeVoice engine implementation."""

from __future__ import annotations

from multiprocessing import Queue
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ..config import DEFAULTS, TTS_MODEL_PATHS
from ..utils import _get_attn_implementation
from .base import TTSEngine


class VibeVoiceEngine(TTSEngine):
    """VibeVoice voice cloning engine."""

    ENV_NAME = "vibevoice"

    def __init__(self, device: str = "cuda", turbo: bool = False):
        self._device = device
        self._turbo = turbo

    @classmethod
    def _run_worker(cls, request_queue: Queue, response_queue: Queue) -> None:
        import site
        import re
        
        vibevoice_pkg = None
        for site_pkg in site.getsitepackages():
            vibevoice_path = Path(site_pkg) / "vibevoice"
            if vibevoice_path.exists():
                vibevoice_pkg = vibevoice_path
                break
        
        if vibevoice_pkg:
            tokenizer_file = vibevoice_pkg / "modular" / "modular_vibevoice_tokenizer.py"
            if tokenizer_file.exists():
                content = tokenizer_file.read_text()
                if "exist_ok=True" not in content:
                    content = content.replace(
                        "AutoModel.register(VibeVoiceAcousticTokenizerConfig, VibeVoiceAcousticTokenizerModel)",
                        "AutoModel.register(VibeVoiceAcousticTokenizerConfig, VibeVoiceAcousticTokenizerModel, exist_ok=True)"
                    )
                    tokenizer_file.write_text(content)
        
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
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
                    inputs = processor(text=[sample_text], padding=True, return_tensors="pt", return_attention_mask=True)

                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device),
                            max_new_tokens=DEFAULTS["max_new_tokens"],
                        )

                    wavs = outputs.cpu().numpy()
                    sr = model.config.sampling_rate

                    if not wavs or len(wavs) == 0:
                        response_queue.put({"id": req_id, "success": False})
                        continue

                    out_dir = Path(output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    output_file = str(out_dir / f"{character_name}.wav")
                    sf.write(output_file, wavs[0], sr)
                    duration = len(wavs[0]) / sr
                    response_queue.put({"id": req_id, "success": True, "output_file": output_file, "duration": duration})

                elif method == "generate_line":
                    text = kwargs["text"]
                    voice_path = kwargs["voice_path"]
                    output_path = kwargs["output_path"]
                    cfg_scale = kwargs.get("cfg_scale", 1.3)

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
                    processor.save_audio(outputs.speech_outputs[0], output_path=output_path)
                    response_queue.put({"id": req_id, "success": True})

                else:
                    response_queue.put({"id": req_id, "error": f"Unknown method: {method}"})

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                response_queue.put({"id": req_id, "error": str(e), "traceback": tb})

    def setup(self, device: str, turbo: bool = False) -> Tuple[Any, Any]:
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
        resp = self._worker_request(
            "generate_line",
            text=text,
            voice_path=voice_path,
            output_path=output_path,
            cfg_scale=cfg_scale,
        )
        self._clear_cuda_cache()
        return resp.get("success", False)
