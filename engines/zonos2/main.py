#!/usr/bin/env python3
"""ZONOS2 engine - standalone subprocess entry point.

This script is run by EngineWorker via subprocess. It starts the ZONOS2 TTS
server, communicates with it via HTTP, and forwards requests/responses to the
main process via JSON lines over stdin/stdout.

Usage:
    python main.py --device cuda:0
    python main.py --probe
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests

SAMPLE_RATE = 44100
SERVER_URL = "http://localhost:1919"
MODEL_PATH = os.environ.get("ZONOS2_MODEL_PATH", "Zyphra/ZONOS2")


def probe() -> None:
    """Print engine capabilities as JSON."""
    caps = {
        "name": "zonos2",
        "methods": ["generate_line", "generate_voice_sample"],
        "sample_rate": SAMPLE_RATE,
    }
    print(json.dumps(caps))


def wait_for_server(url: str, timeout: int = 600, stderr_path: str | None = None) -> None:
    """Wait for the ZONOS2 server to become ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = requests.get(f"{url}/v1", timeout=5)
            if resp.status_code == 200:
                return
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(5)

    err_info = ""
    if stderr_path and Path(stderr_path).exists():
        err_info = f" Server stderr:\n{Path(stderr_path).read_text()[-500:]}"
    raise RuntimeError(f"ZONOS2 server did not become ready within {timeout}s.{err_info}")


def resample_audio(audio_bytes: bytes, orig_sr: int, target_sr: int) -> bytes:
    """Resample float32 PCM audio bytes from orig_sr to target_sr using numpy."""
    import scipy.signal as sig

    audio = np.frombuffer(audio_bytes, dtype=np.float32)
    if orig_sr == target_sr:
        return audio.tobytes()

    num_samples = int(len(audio) * target_sr / orig_sr)
    indices = np.linspace(0, len(audio) - 1, num_samples)
    resampled = np.interp(indices, np.arange(len(audio)), audio)
    return resampled.astype(np.float32).tobytes()


def pcm_f32_to_wav(audio_f32_bytes: bytes, sr: int, output_path: str) -> None:
    """Write float32 PCM audio to a WAV file (int16)."""
    import wave

    audio = np.frombuffer(audio_f32_bytes, dtype=np.float32)
    clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (clipped * 32767).astype(np.int16)

    with wave.open(output_path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(audio_int16.tobytes())


def run_worker(device: str) -> None:
    """Run the ZONOS2 worker loop.

    Starts the ZONOS2 server subprocess, then reads JSON requests from stdin
    and writes JSON responses to stdout.
    """
    server_proc: Optional[subprocess.Popen] = None
    server_stderr: Optional[str] = None
    _voice_cache: dict[str, str] = {}

    def start_server() -> None:
        nonlocal server_proc, server_stderr

        env = os.environ.copy()
        cuda_idx = device.replace("cuda:", "") if device.startswith("cuda:") else "0"
        env["CUDA_VISIBLE_DEVICES"] = cuda_idx

        # tvm_ffi picks the CUDA arch from `nvidia-smi`'s first *physical* GPU,
        # which need not match the device this server actually runs on (e.g. the
        # RTX 4090 is CUDA device 0 but physical GPU 2). Pin the arch so the
        # JIT-compiled kernels target the real device (RTX 4090 -> sm_89).
        try:
            import torch

            _prev = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx
            cap = torch.cuda.get_device_capability(0)
            env["TVM_FFI_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"
            if _prev is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = _prev
        except Exception:
            pass

        stderr_path = tempfile.mktemp(suffix=".log", prefix="zonos2_stderr_")

        server_proc = subprocess.Popen(
            [
                sys.executable, "-m", "zonos2",
                "--model-path", MODEL_PATH,
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=open(stderr_path, "w"),
        )
        print(f"  ZONOS2 server starting (pid={server_proc.pid})...", flush=True)
        wait_for_server(SERVER_URL, timeout=600, stderr_path=stderr_path)
        server_stderr = Path(stderr_path).read_text()
        print("  ZONOS2 server ready.", flush=True)

    def stop_server() -> None:
        nonlocal server_proc
        if server_proc is not None:
            try:
                server_proc.send_signal(signal.SIGTERM)
                server_proc.wait(timeout=15)
            except Exception:
                try:
                    server_proc.terminate()
                    server_proc.wait(timeout=5)
                except Exception:
                    server_proc.kill()
            server_proc = None

    def get_speaker_base64(voice_path: str) -> str:
        """Load voice audio file and return as base64 for speaker cloning."""
        if voice_path not in _voice_cache:
            with open(voice_path, "rb") as f:
                audio_data = f.read()
            _voice_cache[voice_path] = base64.b64encode(audio_data).decode("utf-8")
        return _voice_cache[voice_path]

    # Signal ready to parent
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

        try:
            if server_proc is None or server_proc.poll() is not None:
                start_server()

            if method == "generate_voice_sample":
                character_name = kwargs["character_name"]
                description = kwargs["description"]
                output_dir = kwargs["output_dir"]
                static_voice_text = kwargs.get(
                    "static_voice_text", "Hello, this is my voice."
                )

                if not description or not description.strip():
                    print(
                        json.dumps({"id": req_id, "success": False}), flush=True
                    )
                    continue

                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                output_file = str(out_dir / f"{character_name}.wav")

                payload = {
                    "text": static_voice_text,
                    "stream": False,
                    "fade_out_ms": 200.0,
                }

                resp = requests.post(
                    f"{SERVER_URL}/tts/generate",
                    json=payload,
                    timeout=120,
                )
                if resp.status_code != 200:
                    print(
                        json.dumps({
                            "id": req_id,
                            "success": False,
                            "error": resp.text,
                        }),
                        flush=True,
                    )
                    continue

                audio_f32 = resp.content
                orig_sr = int(resp.headers.get("X-Audio-Sample-Rate", SAMPLE_RATE))

                if orig_sr != 24000:
                    audio_f32 = resample_audio(audio_f32, orig_sr, 24000)

                pcm_f32_to_wav(audio_f32, 24000, output_file)
                duration = len(audio_f32) / 4 / 24000
                print(
                    json.dumps({
                        "id": req_id,
                        "success": True,
                        "output_file": output_file,
                        "duration": duration,
                    }),
                    flush=True,
                )

            elif method == "generate_line":
                text = kwargs["text"]
                voice_path = kwargs["voice_path"]
                output_path = kwargs["output_path"]

                payload = {
                    "text": text,
                    "stream": False,
                    "fade_out_ms": 200.0,
                    "accurate_mode": True,
                }

                if voice_path:
                    payload["speaker_audio_base64"] = get_speaker_base64(voice_path)

                resp = requests.post(
                    f"{SERVER_URL}/tts/generate",
                    json=payload,
                    timeout=120,
                )
                if resp.status_code != 200:
                    print(
                        json.dumps({
                            "id": req_id,
                            "success": False,
                            "error": resp.text,
                        }),
                        flush=True,
                    )
                    continue

                audio_f32 = resp.content
                orig_sr = int(resp.headers.get("X-Audio-Sample-Rate", SAMPLE_RATE))

                if orig_sr != 24000:
                    audio_f32 = resample_audio(audio_f32, orig_sr, 24000)

                pcm_f32_to_wav(audio_f32, 24000, output_path)
                print(json.dumps({"id": req_id, "success": True}), flush=True)

            else:
                print(
                    json.dumps({
                        "id": req_id,
                        "error": f"Unknown method: {method}",
                    }),
                    flush=True,
                )

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            print(
                json.dumps({"id": req_id, "error": str(e), "traceback": tb}),
                flush=True,
            )

    stop_server()


def main() -> None:
    parser = argparse.ArgumentParser(description="ZONOS2 TTS engine")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use")
    parser.add_argument(
        "--probe", action="store_true", help="Print engine capabilities and exit"
    )
    args = parser.parse_args()

    if args.probe:
        probe()
        return

    run_worker(args.device)


if __name__ == "__main__":
    main()
