#!/usr/bin/env python3
"""Breeze-TTS-2 (blue) engine - standalone subprocess entry point.

This script is run by EngineWorker via subprocess. It starts the Breeze TTS 2
streaming API server locally, communicates with it via HTTP, and forwards
requests/responses to the main process via JSON lines over stdin/stdout.

Behavior follows the Breeze-TTS-2 model card:
  - Voice Design (``generate_voice_sample``): a natural-language ``instruction``
    describing the desired voice, with no reference audio. The model card
    recommends ``--cfg-scale 4`` to strengthen instruction-following, so we use
    cfg_scale=4.
  - Voice Clone / Direction (``generate_line``): a reference audio sample paired
    with its *exact* transcript (``ref_audio`` + ``ref_text``). The pipeline
    produces every voice sample by speaking ``static_voice_text``, so that is
    used as the reference transcript. cfg_scale=4 matches the voice-direction
    example.

Usage:
    python main.py --device cuda:0
    python main.py --probe
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests

SAMPLE_RATE = 24000
MODEL_REPO = "BreezeBlue/Breeze-TTS-2"
# Local checkpoint directory (overridable). Downloaded from MODEL_REPO on first
# use if it does not already exist.
CHECKPOINT_DIR = Path(
    os.environ.get("BLUE_MODEL_PATH", str(Path(__file__).resolve().parent / "checkpoint"))
)
# Base HTTP port for the Breeze API server; offset by device index so a server on
# each GPU gets its own port (the worker is shared per engine_dir+device anyway).
BASE_PORT = int(os.environ.get("BLUE_BASE_PORT", "7931"))
# Default reference transcript for cloned voices. The pipeline guarantees every
# voice sample it produces speaks this exact static text, so it is the correct
# ``ref_text`` for Breeze's voice-clone template.
DEFAULT_REF_TEXT = os.environ.get(
    "BLUE_REF_TEXT",
    "Hello there. Good morning everyone. After all these years, it's finally here "
    "for us. The journey has been long and difficult, but we've learned to trust "
    "each other through every challenge. Now we stand together, ready to face "
    "whatever comes next. There's no turning back from here! We're going to make "
    "this work, no matter what!",
)
CFG_SCALE_VOICE_DESIGN = 4.0
CFG_SCALE_VOICE_CLONE = 4.0


def probe() -> None:
    """Print engine capabilities as JSON."""
    caps = {
        "name": "breeze",
        "methods": ["generate_line", "generate_voice_sample"],
        "sample_rate": SAMPLE_RATE,
    }
    print(json.dumps(caps))


def ensure_checkpoint() -> Path:
    """Return the Breeze checkpoint dir, downloading it if necessary."""
    if not (CHECKPOINT_DIR / "config.json").exists():
        print(f"  [blue] Downloading {MODEL_REPO} -> {CHECKPOINT_DIR} ...", flush=True)
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL_REPO, local_dir=str(CHECKPOINT_DIR))
    _patch_eager_attention(CHECKPOINT_DIR)
    return CHECKPOINT_DIR


def _patch_eager_attention(ckpt: Path) -> None:
    """Force the text encoder to eager attention when flash_attn is unavailable.

    Breeze's checkpoint ships with ``preferred_attn_implementation =
    flash_attention_2`` on the text encoder config, which makes transformers try
    to import ``flash_attn`` (not installed here). Eager attention is the model
    card's default inference path, so this only avoids a hard crash; it does not
    change quality. The checkpoint is local/ignored, so the one-line config edit
    is safe and idempotent.
    """
    cfg_path = ckpt / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return
    te = cfg.get("text_encoder_config")
    if not isinstance(te, dict):
        return
    if te.get("preferred_attn_implementation", "flash_attention_2") == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            te["preferred_attn_implementation"] = "eager"
            te["_attn_implementation"] = "eager"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            print("  [blue] Forced text-encoder attention to eager (flash_attn not installed).", flush=True)


def device_port(device: str) -> int:
    """Pick a per-device port for the Breeze API server."""
    try:
        idx = int(device.replace("cuda:", "")) if device.startswith("cuda:") else 0
    except ValueError:
        idx = 0
    return BASE_PORT + idx


def wait_for_server(url: str, timeout: int = 1200, stderr_path: str | None = None) -> None:
    """Wait for the Breeze API server to finish loading the model."""
    deadline = time.monotonic() + timeout
    last_body = ""
    while time.monotonic() < deadline:
        try:
            resp = requests.get(f"{url}/health", timeout=10)
            if resp.status_code == 200:
                return
            last_body = resp.text[:200]
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(5)
        # Loading a 3B model takes a while; poll frequently but don't spin.
        time.sleep(3)

    err_info = ""
    if stderr_path and Path(stderr_path).exists():
        err_info = f" Server stderr:\n{Path(stderr_path).read_text()[-500:]}"
    raise RuntimeError(
        f"Breeze server did not become ready within {timeout}s. "
        f"(last health: {last_body}){err_info}"
    )


def _json_obj(description: str) -> Optional[dict]:
    """Parse the universal-JSON character description, or None."""
    text = (description or "").strip()
    candidate = text
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else candidate
        if candidate.endswith("```"):
            candidate = candidate[:-3]
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, AttributeError):
        return None
    return obj if isinstance(obj, dict) else None


def _description_to_instruction(description: str) -> str:
    """Convert a character voice description to a natural-language instruction.

    Breeze voice design takes a free-form instruction such as "A warm,
    thoughtful young woman with a clear voice and a calm, reflective delivery."
    The universal JSON form carries gender, age, pitch, accent, style, and a
    free-text ``description``; this flattens them into one rich voice prompt so
    distinct characters do not collapse onto the same voice.
    """
    text = (description or "").strip()
    if not text:
        return "Speak clearly and naturally."

    obj = _json_obj(text)

    if isinstance(obj, dict):
        base = []
        gender = obj.get("gender")
        age = obj.get("age")
        if gender:
            base.append(str(gender))
        if age:
            base.append(str(age))

        voice_adj = []
        pitch = obj.get("pitch")
        if pitch:
            voice_adj.append(f"{pitch} pitch")
        style = obj.get("style")
        if isinstance(style, list):
            styles = [str(s).strip() for s in style if str(s).strip()]
            whispers = [s for s in styles if s.lower() == "whisper"]
            styles = [s for s in styles if s.lower() != "whisper"]
            voice_adj.extend(styles)
            if whispers:
                voice_adj.append("whispery")
        elif style and str(style).strip():
            voice_adj.append(str(style).strip())
        for key in ("tone", "register", "quality", "energy"):
            v = obj.get(key)
            if v and str(v).strip():
                voice_adj.append(str(v).strip())
        if voice_adj:
            base.append("with a " + ", ".join(dict.fromkeys(voice_adj)) + " voice")

        accent = obj.get("accent")
        if accent:
            base.append(f"with a {accent} accent")

        if base:
            phrase = "A " + " ".join(p for p in base if p) + "."
            desc = obj.get("description")
            if desc and str(desc).strip():
                return f"{phrase} {str(desc).strip()}"
            return phrase

    # Legacy comma-separated form: "male, middle-aged, moderate pitch"
    words = [w.strip() for w in text.replace(".", ",").split(",") if w.strip()]
    if words:
        phrase = ", ".join(words)
        if phrase[0:2].lower() == "a ":
            return phrase + "."
        return "A " + phrase + "."
    return "Speak clearly and naturally."


def _pcm16_to_wav(pcm_bytes: bytes, sr: int, output_path: str) -> None:
    """Write signed-16-bit little-endian PCM to a mono WAV file."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    with wave.open(output_path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(audio.tobytes())


def _post_speech(url: str, data: dict, files: dict, timeout: float = 900.0) -> bytes:
    """POST to /v1/audio/speech and return the full streaming PCM body."""
    resp = requests.post(f"{url}/v1/audio/speech", data=data, files=files, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Breeze API error {resp.status_code}: {resp.text[:400]}")
    return resp.content


def run_worker(device: str) -> None:
    """Run the Breeze worker loop.

    Starts the Breeze streaming API server subprocess, then reads JSON requests
    from stdin and writes JSON responses to stdout.
    """
    server_proc: Optional[subprocess.Popen] = None
    server_stderr: Optional[str] = None
    # Track the reference transcript for each voice sample we generate, so
    # generate_line can pass the exact transcript Breeze's clone template needs.
    _voice_ref_text: dict[str, str] = {}

    checkpoint = ensure_checkpoint()
    url = f"http://127.0.0.1:{device_port(device)}"

    def start_server() -> None:
        nonlocal server_proc, server_stderr

        env = os.environ.copy()
        cuda_idx = device.replace("cuda:", "") if device.startswith("cuda:") else "0"
        env["CUDA_VISIBLE_DEVICES"] = cuda_idx

        stderr_path = tempfile.mktemp(suffix=".log", prefix="blue_stderr_")
        # Run from the breeze-tts repo root so `models.*` and `breeze_infer.*`
        # imports resolve (the API module imports models.fast_streaming etc).
        repo_root = Path(__file__).resolve().parent / "breeze-tts"

        server_proc = subprocess.Popen(
            [
                sys.executable, "-m", "breeze_infer.api", str(checkpoint),
                "--host", "127.0.0.1", "--port", str(device_port(device)),
                # Fast path: CUDA-graph backbone decode + codec + depth decoder
                # (~2.5x faster than eager). The text-encoder / backbone-prefill
                # fast graphs are deliberately excluded: --fast-all hangs warmup
                # on this setup (measured), these three give the full speedup.
                "--fast-backbone-decode", "--fast-codec", "--fast-depth-decoder",
            ],
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=open(stderr_path, "w"),
            # Own process group so stop_server() can kill the server AND any of
            # its multiprocessing children, which otherwise leak and pin VRAM.
            start_new_session=True,
        )
        print(f"  [blue] Breeze server starting (pid={server_proc.pid})...", flush=True)
        wait_for_server(url, timeout=1200, stderr_path=stderr_path)
        server_stderr = Path(stderr_path).read_text()
        print("  [blue] Breeze server ready.", flush=True)

    def stop_server() -> None:
        nonlocal server_proc
        if server_proc is not None:
            pgid = os.getpgid(server_proc.pid)
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                server_proc.wait(timeout=5)
            server_proc = None

    def ref_text_for(voice_path: str) -> str:
        """Return the exact transcript for a voice file, else the default."""
        return _voice_ref_text.get(voice_path) or DEFAULT_REF_TEXT

    def post_with_restart(data: dict, files: dict, timeout: float = 120.0) -> bytes:
        """POST speech, restarting the server if the request hangs/fails.

        The fast-path (CUDA-graph) server intermittently hard-deadlocks the GPU
        on certain inputs, blocking the single-concurrency request forever. When
        a request does not complete within ``timeout``, kill and relaunch the
        server (releasing the stuck CUDA context) and retry once. Without this a
        single deadlocked line stalls the entire audiobook run.
        """
        try:
            return _post_speech(url, data, files, timeout=timeout)
        except Exception as e:
            print(f"  [blue] server request failed ({e}); restarting server...", flush=True)
            stop_server()
            start_server()
            try:
                return _post_speech(url, data, files, timeout=timeout)
            except Exception as e2:
                raise RuntimeError(f"Breeze server failed after restart: {e2}")

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
                    "static_voice_text", DEFAULT_REF_TEXT
                )

                if not description or not description.strip():
                    print(json.dumps({"id": req_id, "success": False}), flush=True)
                    continue

                instruction = _description_to_instruction(description)

                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                output_file = str(out_dir / f"{character_name}.wav")

                data = {
                    "text": static_voice_text,
                    "instruction": instruction,
                    "cfg_scale": str(CFG_SCALE_VOICE_DESIGN),
                    "seed": "42",
                }
                pcm = post_with_restart(data, files={})
                if not pcm:
                    print(json.dumps({"id": req_id, "success": False}), flush=True)
                    continue

                _pcm16_to_wav(pcm, SAMPLE_RATE, output_file)
                duration = len(pcm) / 2 / SAMPLE_RATE
                # The voice sample speaks `static_voice_text`; record it as the
                # exact transcript for later cloning.
                _voice_ref_text[output_file] = static_voice_text
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

                ref_text = ref_text_for(voice_path)
                data = {
                    "text": text,
                    "ref_text": ref_text,
                    "instruction": "Speak clearly and naturally.",
                    "cfg_scale": str(CFG_SCALE_VOICE_CLONE),
                    "seed": "42",
                }
                files = {"ref_audio": open(voice_path, "rb")}
                try:
                    pcm = post_with_restart(data, files)
                finally:
                    files["ref_audio"].close()

                if not pcm:
                    print(json.dumps({"id": req_id, "success": False}), flush=True)
                    continue

                _pcm16_to_wav(pcm, SAMPLE_RATE, output_path)
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
    parser = argparse.ArgumentParser(description="Breeze-TTS-2 (blue) engine")
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
