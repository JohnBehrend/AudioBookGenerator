"""Echo TTS engine adapter."""

from typing import Tuple, Any, Optional
from pathlib import Path

import torch

from config import DEFAULTS, TTS_MODEL_PATHS
from ..base import TTSEngine
from ..utils import split_text_for_echo_tts


class EchoTTSAdapter(TTSEngine):
    """Adapter for Echo TTS voice cloning engine."""

    def setup(self, device: str, turbo: bool = False) -> Tuple[Any, None]:
        from echo_tts import EchoTTSLoader

        model_path = TTS_MODEL_PATHS["echo-tts"]
        loader = EchoTTSLoader(device, model_path)
        return loader, None

    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: Path,
        device: str,
        verbose: bool = False,
    ) -> Tuple[bool, Optional[str], float]:
        """Echo TTS is only used for voice cloning, not voice sample generation.

        Returns failure since echo-tts requires a reference voice.
        """
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
        from functools import partial
        import sys
        import importlib

        loader, _ = self._get_model(device)

        if not hasattr(loader, "_loaded") or loader._loaded is False:
            loader.load()

        # Load speaker audio for voice cloning
        inference_module = sys.modules.get("inference")
        if inference_module is None:
            inference_module = importlib.import_module("inference")
        speaker_audio_tensor = inference_module.load_audio(voice_path).to(loader.echo_model.device)

        text_chunks = split_text_for_echo_tts(text)

        audio_chunks = []
        for i, chunk in enumerate(text_chunks):
            text_prompt = f"[S1] {chunk}"

            audio_out, _ = loader.sample_pipeline(
                model=loader.echo_model,
                fish_ae=loader.fish_ae,
                pca_state=loader.pca_state,
                sample_fn=loader.sampler,
                text_prompt=text_prompt,
                speaker_audio=speaker_audio_tensor,
                rng_seed=42 + i,
            )

            if audio_out is None or audio_out.numel() == 0:
                return False

            audio_chunks.append(audio_out[0])

        if len(audio_chunks) == 1:
            audio_final = audio_chunks[0]
        else:
            silence = torch.zeros(1, 2205, device=audio_chunks[0].device)
            audio_final = audio_chunks[0]
            for chunk in audio_chunks[1:]:
                audio_final = torch.cat([audio_final, silence, chunk], dim=1)

        import torchaudio
        sr = 44100
        torchaudio.save(output_path, audio_final.cpu(), sr)
        return True

    def _get_model(self, device: str):
        if not hasattr(self, "_cached_model"):
            self._cached_model, self._cached_processor = self.setup(device)
        return self._cached_model, self._cached_processor
