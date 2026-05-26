"""Echo TTS engine adapter."""

from __future__ import annotations

from functools import partial
from multiprocessing import Queue
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

from .base import TTSEngine
from .utils import split_text_for_echo_tts

if TYPE_CHECKING:
    from ..config import DEFAULTS, TTS_MODEL_PATHS


class EchoTTSAdapter(TTSEngine):
    """Adapter for Echo TTS voice cloning engine."""

    ENV_NAME = "echo-tts"

    def __init__(self, device: str = "cuda", turbo: bool = False):
        self._device = device
        self._turbo = turbo

    @classmethod
    def _run_worker(cls, request_queue: Queue, response_queue: Queue) -> None:
        import torch
        from functools import partial
        from .echo_tts import inference as echo_inf

        model = None
        fish_ae = None
        pca_state = None
        sampler = None

        def load_models(device: str) -> None:
            nonlocal model, fish_ae, pca_state, sampler
            if model is not None:
                return
            from ..config import TTS_MODEL_PATHS
            model_path = TTS_MODEL_PATHS["echo-tts"]
            print("Loading Echo TTS models...")
            model = echo_inf.load_model_from_hf(
                repo_id=model_path,
                device=device,
                dtype=torch.bfloat16,
                delete_blockwise_modules=True,
            )
            fish_ae = echo_inf.load_fish_ae_from_hf(
                device=device,
                dtype=torch.float32,
            )
            pca_state = echo_inf.load_pca_state_from_hf(
                repo_id=model_path,
                device=device,
            )
            sampler = partial(
                echo_inf.sample_euler_cfg_independent_guidances,
                num_steps=50,
                cfg_scale_text=3.0,
                cfg_scale_speaker=8.0,
                cfg_min_t=0.5,
                cfg_max_t=1.0,
                truncation_factor=None,
                rescale_k=None,
                rescale_sigma=None,
                speaker_kv_scale=None,
                speaker_kv_max_layers=None,
                speaker_kv_min_t=None,
                sequence_length=640,
            )
            print("Echo TTS models loaded successfully!")

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
                load_models(device)

                if method == "generate_line":
                    assert model is not None and fish_ae is not None
                    assert pca_state is not None and sampler is not None

                    voice_path = kwargs["voice_path"]
                    text = kwargs["text"]
                    output_path = kwargs["output_path"]

                    speaker_audio = echo_inf.load_audio(voice_path).to(model.device)
                    text_chunks = split_text_for_echo_tts(text)

                    audio_chunks = []
                    for i, chunk in enumerate(text_chunks):
                        text_prompt = f"[S1] {chunk}"
                        audio_out, _ = echo_inf.sample_pipeline(
                            model=model,
                            fish_ae=fish_ae,
                            pca_state=pca_state,
                            sample_fn=sampler,
                            text_prompt=text_prompt,
                            speaker_audio=speaker_audio,
                            rng_seed=42 + i,
                        )
                        if audio_out is None or audio_out.numel() == 0:
                            break

                        audio_chunks.append(audio_out[0])

                    if not audio_chunks:
                        response_queue.put({"id": req_id, "success": False})
                        continue

                    if len(audio_chunks) == 1:
                        audio_final = audio_chunks[0]
                    else:
                        silence = torch.zeros(1, 2205, device=audio_chunks[0].device)
                        audio_final = audio_chunks[0]
                        for chunk in audio_chunks[1:]:
                            audio_final = torch.cat([audio_final, silence, chunk], dim=1)

                    import soundfile as sf
                    sf.write(output_path, audio_final.cpu().numpy(), 44100)
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
        """Echo TTS is only used for voice cloning, not voice sample generation."""
        if verbose:
            print(f"  WARNING: Echo TTS does not support voice sample generation")
        return False, None, 0

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
        return super().generate_line(text, voice_path, output_path, device, validation_model, cfg_scale, max_new_tokens, verbose)
