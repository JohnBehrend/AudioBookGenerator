#!/usr/bin/env python3
"""Dramabox engine - standalone subprocess entry point.

This script is run by EngineWorker via subprocess. It communicates with the
main process via JSON lines over stdin/stdout.

Usage:
    python main.py --device cuda:0
    python main.py --probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Constants
SAMPLE_RATE = 24000
STATIC_VOICE_TEXT = "Hello, this is my voice."


def probe() -> None:
    """Print engine capabilities as JSON."""
    caps = {
        "name": "dramabox",
        "methods": ["generate_line", "generate_voice_sample"],
        "sample_rate": SAMPLE_RATE,
    }
    print(json.dumps(caps))


def convert_description_to_prompt(description: str, text: str) -> str:
    """Convert a voice description and text to Dramabox prompt format.

    Dramabox expects: <speaker description>, "<dialogue>"
    """
    speaker = description.strip().rstrip(".")
    return f'A {speaker} speaks, "{text}"'


def run_worker(device: str) -> None:
    """Run the Dramabox worker loop.

    Reads JSON requests from stdin, writes JSON responses to stdout.
    """
    import os

    APP_DIR = Path(__file__).parent
    sys.path.insert(0, str(APP_DIR / "ltx2"))
    sys.path.insert(0, str(APP_DIR / "src"))

    import torch
    import soundfile as sf
    from inference_server import TTSServer
    from model_downloader import get_all_paths

    model = None
    _voice_clone_prompts: dict[str, Any] = {}

    def load_model(device: str) -> None:
        nonlocal model
        if model is not None:
            return
        paths = get_all_paths()
        model = TTSServer(
            checkpoint=paths["transformer"],
            full_checkpoint=paths["audio_components"],
            gemma_root=paths["gemma_root"],
            device=device,
            dtype="bf16",
            compile_model=False,
            bnb_4bit=True,
        )

    # Signal ready
    print(json.dumps({"type": "ready"}), flush=True)

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        if req.get("type") == "shutdown":
            break
        if req.get("type") != "request":
            continue

        req_id = req["id"]
        method = req["method"]
        kwargs = req["kwargs"]
        device = kwargs.get("device", device)

        try:
            load_model(device)
            assert model is not None

            if method == "generate_voice_sample":
                character_name = kwargs["character_name"]
                description = kwargs["description"]
                output_dir = kwargs["output_dir"]
                static_voice_text = kwargs.get("static_voice_text", STATIC_VOICE_TEXT)

                if not description or not description.strip():
                    print(json.dumps({"id": req_id, "success": False}), flush=True)
                    continue

                prompt = convert_description_to_prompt(description, static_voice_text)

                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                output_file = str(out_dir / f"{character_name}.wav")

                waveform, sr = model.generate(
                    prompt=prompt,
                )

                if waveform is None or (hasattr(waveform, 'numel') and waveform.numel() == 0):
                    print(json.dumps({"id": req_id, "success": False}), flush=True)
                    continue

                wav_cpu = waveform.cpu().float() if hasattr(waveform, 'cpu') else waveform
                # Resample to 24kHz to match pipeline expectations
                if sr != SAMPLE_RATE:
                    import torchaudio.transforms as T
                    resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                    wav_cpu = resampler(wav_cpu)
                    sr = SAMPLE_RATE
                sf.write(output_file, wav_cpu.numpy().flatten(), sr)
                duration = wav_cpu.shape[-1] / sr
                print(json.dumps({
                    "id": req_id,
                    "success": True,
                    "output_file": output_file,
                    "duration": duration,
                }), flush=True)

            elif method == "generate_line":
                text = kwargs["text"]
                voice_path = kwargs["voice_path"]
                output_path = kwargs["output_path"]
                description = kwargs.get("description", "")

                if description:
                    prompt = convert_description_to_prompt(description, text)
                else:
                    prompt = f'A speaker says, "{text}"'

                waveform, sr = model.generate(
                    prompt=prompt,
                    voice_ref=voice_path,
                )

                if waveform is None or (hasattr(waveform, 'numel') and waveform.numel() == 0):
                    print(json.dumps({"id": req_id, "success": False}), flush=True)
                    continue

                wav_cpu = waveform.cpu().float() if hasattr(waveform, 'cpu') else waveform
                # Resample to 24kHz to match pipeline expectations
                if sr != SAMPLE_RATE:
                    import torchaudio.transforms as T
                    resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                    wav_cpu = resampler(wav_cpu)
                    sr = SAMPLE_RATE
                sf.write(output_path, wav_cpu.numpy().flatten(), sr)
                print(json.dumps({"id": req_id, "success": True}), flush=True)

            else:
                print(json.dumps({"id": req_id, "error": f"Unknown method: {method}"}), flush=True)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(json.dumps({"id": req_id, "error": str(e), "traceback": tb}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dramabox TTS engine")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use")
    parser.add_argument("--probe", action="store_true", help="Print engine capabilities and exit")
    args = parser.parse_args()

    if args.probe:
        probe()
        return

    run_worker(args.device)


if __name__ == "__main__":
    main()
