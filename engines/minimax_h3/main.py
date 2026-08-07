#!/usr/bin/env python3
"""MiniMax H3 engine - standalone subprocess entry point.

This script is run by EngineWorker via subprocess. It communicates with the
main process via JSON lines over stdin/stdout.

H3 is a joint video+audio generation model (MiniMax Hailuo-03 / fl2va). It is
*not* a classic TTS voice-cloner, so this engine only implements
``generate_voice_sample`` (Stage 4 voice design): it drives a locally-running
ComfyUI instance to generate a low-resolution video of a character speaking the
reference (static voice) text, then extracts the synchronized audio track and
writes it out as the character's voice reference WAV.

Line generation (Stage 5, ``generate_line``) is intentionally *not* supported:
the pipeline uses other engines (e.g. omni) to clone voices from these samples.

Usage:
    python main.py --device cuda:0
    python main.py --probe
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests

SAMPLE_RATE = 24000

# ComfyUI HTTP + model configuration (overridable via env). Defaults target an
# RTX 4090 (24GB): the pruned int8 diffusion model + NVFP4 text encoder are the
# pair that fit in VRAM. Swap to the full int8 files via env if you have more VRAM.
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
H3_UNET = os.environ.get("H3_UNET", "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
H3_CLIP = os.environ.get("H3_CLIP", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
H3_VIDEO_VAE = os.environ.get("H3_VIDEO_VAE", "minimax_h3_video_vae_fp16.safetensors")
H3_AUDIO_VAE = os.environ.get("H3_AUDIO_VAE", "minimax_h3_audio_vae_fp32.safetensors")
H3_WIDTH = int(os.environ.get("H3_WIDTH", "64"))
H3_HEIGHT = int(os.environ.get("H3_HEIGHT", "64"))
# Optional single-value resolution override (square). When >0 it sets both
# H3_WIDTH and H3_HEIGHT, e.g. for higher-res video to inspect the character
# appearance. Note: VRAM scales with resolution; the 4090 fits only ~64-128.
H3_RESOLUTION = int(os.environ.get("H3_RESOLUTION", "0"))
# Also save the generated video (mp4) alongside the audio, so the speaker's
# appearance can be visually verified against the character description.
H3_SAVE_VIDEO = os.environ.get("H3_SAVE_VIDEO", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
# Annotate each sentence of the reference text with an inline emotion cue so
# the spoken passage carries the character's emotional flavor.
H3_EMOTION_TAGS = os.environ.get("H3_EMOTION_TAGS", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
# Clip length in frames (~24 fps). "auto" (default) sizes the clip to fit the
# full voice-sample reference text so nothing gets truncated; a positive integer
# forces a fixed length (with text truncated to fit). H3_MAX_LENGTH caps auto at
# 15s (~360 frames), H3's supported clip ceiling.
H3_LENGTH = os.environ.get("H3_LENGTH", "auto").strip().lower()
H3_MAX_LENGTH = int(os.environ.get("H3_MAX_LENGTH", "360"))
H3_STEPS = int(os.environ.get("H3_STEPS", "40"))
H3_CFG = float(os.environ.get("H3_CFG", "1.0"))
H3_SAMPLER = os.environ.get("H3_SAMPLER", "euler")
H3_SCHEDULER = os.environ.get("H3_SCHEDULER", "simple")
H3_SEED = int(os.environ.get("H3_SEED", "0"))
# Free ComfyUI VRAM after every generation. ComfyUI caches the loaded H3 model
# in VRAM across prompts, which would otherwise (a) pin ~20GB on the 4090 and
# (b) collide with a downstream engine (e.g. omni) that needs the same GPU.
# Default on so sequential H3 voices and the switch to omni both stay safe.
# Set to 0 to keep the model resident for faster back-to-back H3 generations
# (you must then free VRAM manually before running another engine).
H3_FREE_VRAM = os.environ.get("H3_FREE_VRAM", "1").lower() in ("1", "true", "yes")


def probe() -> None:
    """Print engine capabilities as JSON."""
    caps = {
        "name": "minimax_h3",
        "methods": ["generate_voice_sample"],
        "sample_rate": SAMPLE_RATE,
    }
    print(json.dumps(caps))


def _align_frame_count(n: int) -> int:
    while n % 17 != 5:
        n += 1
    return n


def _audio_seconds(frames: int) -> float:
    """Duration of a generated clip given a frame count at 24 fps."""
    return _align_frame_count(max(5, frames)) / 24.0


def _frames_for_text(text: str) -> int:
    """Frames (~24 fps) needed to speak the full ``text``.

    Inverts the ~2.6 words/sec / ~6 chars-per-word estimate used by
    ``_truncate_text_for_length`` so auto-length clips cover every word.
    """
    frames = max(5, int(len(text) * 24 / (0.9 * 2.6 * 6)))
    return min(_align_frame_count(frames), _align_frame_count(max(5, H3_MAX_LENGTH)))


def _effective_length(static_voice_text: str) -> int:
    """Resolve the clip length in frames (auto -> fit full reference text)."""
    if H3_LENGTH == "auto":
        return _frames_for_text(static_voice_text)
    return _align_frame_count(max(5, int(H3_LENGTH)))


def _truncate_text_for_length(text: str, frames: int) -> str:
    """Roughly fit text to the clip duration (~2.6 words/sec of speech)."""
    seconds = _audio_seconds(frames) * 0.9
    max_chars = int(seconds * 2.6 * 6)  # ~6 chars per word average
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars * 0.6:
        cut = cut[:last_space]
    return cut.strip()


def _description_to_text(description: str) -> str:
    """Convert a character voice description to a natural-language phrase.

    The universal JSON form carries far more voice detail than H3 was being
    told about: gender, age, pitch, accent, a full ``style`` list, extra tone/
    register/quality attributes, and a free-text ``description``. This builds a
    rich, distinct voice instruction so characters with different attributes
    (e.g. Rand vs Perrin) do not collapse onto the same voice.
    """
    text = (description or "").strip()
    if not text:
        return "a neutral voice"

    candidate = text
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else candidate
        if candidate.endswith("```"):
            candidate = candidate[:-3]
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, AttributeError):
        obj = None

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
            base.append("a " + ", ".join(dict.fromkeys(voice_adj)) + " voice")

        accent = obj.get("accent")
        if accent:
            base.append(f"with a {accent} accent")

        if base:
            phrase = " ".join(p for p in base if p)
            desc = obj.get("description")
            if desc and str(desc).strip():
                return f"{phrase}. {str(desc).strip()}"
            return phrase

    # Legacy comma-separated form: "male, middle-aged, moderate pitch"
    words = [w.strip() for w in text.replace(".", ",").split(",") if w.strip()]
    if words:
        return ", ".join(words)
    return "a neutral voice"


def _h3_speaker_phrase(description: str) -> str:
    """Build H3's speaker-identity phrase from a character description.

    H3 encodes the speaker's voice in the identifying phrase placed before the
    ``<d>`` dialogue tag, e.g. "The young woman with a quiet, breathy voice (S1)".
    Without the voice attributes here (plus the <d> dialogue structure), H3
    falls back to a default voice, so all characters end up sounding alike.
    """
    text = (description or "").strip()
    candidate = text
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else candidate
        if candidate.endswith("```"):
            candidate = candidate[:-3]
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, AttributeError):
        obj = None

    if isinstance(obj, dict):
        attrs = []
        gender = obj.get("gender")
        age = obj.get("age")
        if age and gender:
            attrs.append(f"{age} {gender}")
        elif age:
            attrs.append(f"{age} person")
        elif gender:
            attrs.append(str(gender))

        voice_words = []
        pitch = obj.get("pitch")
        if pitch:
            voice_words.append(str(pitch))
        style = obj.get("style")
        if isinstance(style, list):
            voice_words.extend(
                str(s).strip()
                for s in style
                if str(s).strip() and str(s).strip().lower() != "whisper"
            )
            if any(str(s).strip().lower() == "whisper" for s in style):
                voice_words.append("whispery")
        elif style and str(style).strip():
            voice_words.append(str(style).strip())
        for key in ("tone", "register", "quality", "energy"):
            v = obj.get(key)
            if v and str(v).strip():
                voice_words.append(str(v).strip())
        seen: set[str] = set()
        voice_words = [w for w in voice_words if not (w in seen or seen.add(w))]
        if voice_words:
            attrs.append("with a " + ", ".join(voice_words) + " voice")

        accent = obj.get("accent")
        if accent:
            attrs.append("with a " + str(accent) + " accent")

        # Anchor the voice to a well-known real voice when the pipeline provided
        # one (e.g. "Tom Hanks"). This is a *textual* cue for H3's delivery —
        # the reference-audio path is intentionally not used.
        celeb = obj.get("celebrity_voice") or obj.get("celebrity")
        if celeb and str(celeb).strip():
            attrs.append("in the style of " + str(celeb).strip())

        if attrs:
            return "The " + " ".join(attrs) + " (S1)"
        return "The speaker (S1)"

    # Legacy comma-separated fallback -> a neutral but phrased speaker.
    words = [w.strip() for w in text.replace(".", ",").split(",") if w.strip()]
    if words:
        return "The speaker with a " + ", ".join(words) + " voice (S1)"
    return "The speaker (S1)"


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


def _h3_delivery(description: str) -> str:
    """Build the delivery/emotion clause, placed OUTSIDE the <d> dialogue tag.

    H3's docs require emotion/delivery direction in the prose around the
    dialogue, never inside ``<d>``. This expresses the emotional progression
    (from ``emotion_arc``) or the character's style traits as a delivery phrase.
    """
    obj = _json_obj(description)
    pace = ""
    arc: list[str] = []
    if obj:
        pace = str(obj.get("pace") or obj.get("speaking_rate") or "").strip()
        a = obj.get("emotion_arc")
        if isinstance(a, list):
            arc = [str(x).strip() for x in a if str(x).strip()]
    pace_clause = f", at a {pace} pace" if pace else ""

    def _art(word: str) -> str:
        return "an" if word.lstrip().lower().startswith(("a", "e", "i", "o", "u")) else "a"

    if len(arc) == 1:
        return f"with {_art(arc[0])} {arc[0]} delivery{pace_clause}"
    if len(arc) == 2:
        return f"with a delivery that moves from {arc[0]} to {arc[1]}{pace_clause}"
    if len(arc) >= 3:
        middle = ", then ".join(arc[1:-1])
        return (
            f"with a delivery that begins {arc[0]}, then {middle}, "
            f"and finally {arc[-1]}{pace_clause}"
        )

    # Fall back to style traits.
    traits: list[str] = []
    if obj:
        style = obj.get("style")
        if isinstance(style, list):
            traits.extend(
                str(s).strip()
                for s in style
                if str(s).strip() and str(s).strip().lower() != "whisper"
            )
        elif style and str(style).strip():
            traits.append(str(style).strip())
        for key in ("tone", "energy", "intensity", "register"):
            v = obj.get(key)
            if v and str(v).strip():
                traits.append(str(v).strip())
    traits = [t for t in traits if t.lower() != "whisper"]
    if not traits:
        return f"with a neutral, even delivery{pace_clause}"
    joined = ", ".join(traits[:-1]) + " and " + traits[-1] if len(traits) > 1 else traits[0]
    return f"with {_art(joined)} {joined} delivery{pace_clause}"


def _clean_spoken(text: str) -> str:
    """Collapse whitespace while preserving every word and punctuation verbatim.

    H3's docs require ``<d>`` to carry the spoken content verbatim (every
    original word and punctuation mark). Only whitespace runs (including
    newlines) are normalized to single spaces so the line stays on one line.
    """
    return " ".join(str(text).split())


def _clean_subject(text: str) -> str:
    """Collapse whitespace and strip trailing sentence punctuation from a
    free-text subject so it flows into the rest of the shot sentence."""
    s = " ".join(str(text).split())
    return s.rstrip(".,;: ")


def build_prompt(description: str, spoken_text: str) -> str:
    """Build an H3 T2VA prompt that follows the model's official format.

    The model is trained on the three-field prompt structure
    (``integrated_multimodal_description`` / ``overall_soundscape`` /
    ``non_diegetic_music``) with dialogue wrapped in ``<d>[English] ...</d>``
    tags, the speaker's voice attributes named in the identifying phrase, the
    character's physical appearance described visually, and an explicit
    delivery/emotion clause so the line is performed with character, not as a
    flat read. Matching that structure is what lets H3 vary voice, appearance,
    and emotional performance.
    """
    speaker = _h3_speaker_phrase(description)
    appearance = _h3_appearance(description)
    delivery = _h3_delivery(description)
    subject = _clean_subject(appearance) or "one person"
    # Per H3 docs, <d> holds ONLY the language tag + verbatim spoken content;
    # emotion/delivery goes in the prose before it (the `delivery` clause).
    spoken = _clean_spoken(spoken_text)
    return (
        "integrated_multimodal_description: [Shot 1] Live-action, cinematic, "
        "a medium close-up shot frames "
        f"{subject} facing the camera directly, motionless, with soft even "
        "lighting and a neutral studio background. "
        f"{speaker} looks steadily at the camera and delivers the line "
        f"{delivery}, with clear lip movement: <d>[English] {spoken}</d>\n\n"
        "overall_soundscape: Faint room tone with no other sounds.\n\n"
        "non_diegetic_music: N/A"
    )


def _h3_appearance(description: str) -> str:
    """Extract the character's physical appearance from the description.

    The universal JSON's free-text ``description`` field carries the character's
    defining visual traits (hair color, eyes, build, clothing). Without passing
    these, H3 renders a generic face (e.g. brown hair instead of Rand's
    copper-red hair).
    """
    obj = _json_obj(description)
    if obj:
        v = obj.get("description")
        if v and str(v).strip():
            return str(v).strip()
    return ""


class ComfyClient:
    """Thin client for submitting and collecting a ComfyUI workflow."""

    def __init__(self, base_url: str, request_timeout: float = 1200.0):
        self.base_url = base_url
        self.request_timeout = request_timeout
        self._gpu_idx: Optional[int] = None
        self._idle_used_mib: Optional[int] = None

    def ping(self) -> None:
        # ComfyUI can be momentarily unresponsive right after finishing a long
        # generation, so retry a few times before giving up.
        last_err: Optional[Exception] = None
        for attempt in range(4):
            try:
                resp = requests.get(f"{self.base_url}/system_stats", timeout=10)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                last_err = e
                time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(
                f"ComfyUI not reachable at {self.base_url}. Start it, e.g.: "
                f"cd <ComfyUI> && python main.py --listen 127.0.0.1 --port 8188. "
                f"({last_err})"
            )
        # Record the idle (no H3 model loaded) VRAM baseline of the ComfyUI GPU
        # so free_vram can wait until the driver actually returns the memory.
        self._gpu_idx = self._find_comfy_gpu()
        if self._gpu_idx is not None:
            self._idle_used_mib = self._nvsmi_used(self._gpu_idx)

    def submit(self, graph: dict[str, Any]) -> str:
        """Submit a graph-format workflow and return the prompt_id."""
        client_id = str(uuid.uuid4())
        payload = {"prompt": graph, "client_id": client_id}
        try:
            resp = requests.post(f"{self.base_url}/prompt", json=payload, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to submit prompt to ComfyUI: {e}")
        data = resp.json()
        if "error" in data or data.get("node_errors"):
            raise RuntimeError(
                f"ComfyUI rejected workflow: {json.dumps(data, default=str)}"
            )
        return data["prompt_id"]

    def wait_until_done(self, prompt_id: str) -> dict[str, Any]:
        """Poll /history until the prompt finishes; return its history entry."""
        deadline = time.monotonic() + self.request_timeout
        while time.monotonic() < deadline:
            try:
                resp = requests.get(
                    f"{self.base_url}/history/{prompt_id}", timeout=15
                )
                resp.raise_for_status()
                history = resp.json()
            except requests.RequestException:
                time.sleep(2)
                continue
            entry = history.get(prompt_id)
            if entry is None:
                time.sleep(2)
                continue
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") in (
                "success",
                "completed",
            ):
                return entry
            if status.get("status_str") in ("error", "failed"):
                msgs = entry.get("status", {}).get("messages", [])
                raise RuntimeError(
                    f"ComfyUI workflow failed: {json.dumps(msgs, default=str)}"
                )
            time.sleep(2)
        raise TimeoutError(
            f"ComfyUI prompt {prompt_id} did not finish within "
            f"{self.request_timeout:.0f}s"
        )

    def get_saved_audio(self, entry: dict[str, Any], save_node: str) -> bytes:
        """Locate and download the audio saved by ``save_node`` from history."""
        return self._get_saved_file(entry, save_node, ("audio", "audios"))[0]

    def get_saved_video(self, entry: dict[str, Any], save_node: str) -> tuple[bytes, str]:
        """Locate and download the video saved by ``save_node`` from history.

        Returns ``(bytes, filename)``. SaveVideo records its output under the
        "images" history key (an mp4/webm file).
        """
        return self._get_saved_file(entry, save_node, ("images", "gifs", "video", "videos"))

    def _get_saved_file(
        self, entry: dict[str, Any], save_node: str, keys: tuple[str, ...]
    ) -> tuple[bytes, str]:
        """Download the first file saved by ``save_node`` in ``history``."""
        outputs = entry.get("outputs", {})
        node_out = outputs.get(save_node, {})
        files = []
        for k in keys:
            files = node_out.get(k) or []
            if files:
                break
        if not files:
            raise RuntimeError(
                f"No file output found for node {save_node!r} in history "
                f"(keys {keys})"
            )
        item = files[0]
        params = {
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        }
        try:
            resp = requests.get(
                f"{self.base_url}/view", params=params, timeout=120
            )
            resp.raise_for_status()
            return resp.content, item.get("filename", "output.bin")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download file result: {e}")

    def free_vram(self, timeout: float = 90.0) -> None:
        """Unload loaded models so the GPU is free for a downstream engine.

        ComfyUI's ``/free`` endpoint releases the H3 model without restarting
        the server, but the memory lands in ComfyUI's cudaMallocAsync caching
        pool and is only returned to the driver asynchronously (several seconds
        later). /system_stats vram_free reflects the pool, so it can report
        "free" while nvidia-smi still shows ~20GB resident.

        To guarantee the memory is actually returned (so a downstream engine
        like omni won't OOM), we poll nvidia-smi (driver-level) until the
        ComfyUI GPU's used memory drops back to its idle baseline.
        """
        try:
            requests.post(
                f"{self.base_url}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=30,
            )
        except requests.RequestException as e:
            print(
                f"  [minimax_h3] Warning: failed to free ComfyUI VRAM: {e}",
                file=sys.stderr,
                flush=True,
            )
            return

        gpu_idx = self._gpu_idx
        baseline = self._idle_used_mib
        if gpu_idx is None or baseline is None:
            print(
                "  [minimax_h3] Warning: could not identify ComfyUI GPU / VRAM "
                "baseline for freed check",
                file=sys.stderr,
                flush=True,
            )
            return

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            used = self._nvsmi_used(gpu_idx)
            if used is not None and used <= baseline + 700:
                return
            time.sleep(1.0)
        print(
            f"  [minimax_h3] Warning: timed out waiting for ComfyUI to free VRAM",
            file=sys.stderr,
            flush=True,
        )

    def _comfy_gpu_name(self) -> Optional[str]:
        """Name of the GPU ComfyUI runs on, from /system_stats.

        e.g. "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync" ->
             "NVIDIA GeForce RTX 4090"
        """
        try:
            stats = requests.get(f"{self.base_url}/system_stats", timeout=10).json()
            dev = (stats.get("devices") or [None])[0]
            if not dev:
                return None
            raw = dev.get("name", "")
            for tok in raw.split():
                if tok.startswith("cuda:"):
                    raw = raw.replace(tok, "", 1).strip()
                    break
            return raw.split(" : ")[0].strip() or None
        except (requests.RequestException, ValueError, KeyError, IndexError):
            return None

    def _find_comfy_gpu(self) -> Optional[int]:
        """nvidia-smi index of the GPU matching ComfyUI's reported name."""
        name = self._comfy_gpu_name()
        if not name:
            return None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            ).stdout
        except Exception:
            return None
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1] == name:
                return int(parts[0])
        return None

    def _nvsmi_used(self, idx: int) -> Optional[int]:
        """Used memory (MiB) of nvidia-smi GPU ``idx``."""
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu=memory.used",
                    f"--id={idx}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            ).stdout.strip()
            return int(out.split()[0])
        except (Exception, ValueError):
            return None


def build_workflow(
    prompt_text: str, filename_prefix: str, length: Optional[int] = None
) -> tuple[dict[str, Any], str, Optional[str]]:
    """Build the H3 local graph and return (graph, audio_node, video_node).

    ``video_node`` is the id of the SaveVideo node (when H3_SAVE_VIDEO is on),
    otherwise ``None``.
    """
    if H3_RESOLUTION > 0:
        width = height = max(32, round(H3_RESOLUTION / 32) * 32)
    else:
        width = max(32, round(H3_WIDTH / 32) * 32)
        height = max(32, round(H3_HEIGHT / 32) * 32)
    if length is None:
        length = 160 if H3_LENGTH == "auto" else int(H3_LENGTH)
    length = _align_frame_count(max(5, length))
    seed = H3_SEED if H3_SEED else random.randint(0, 2**31 - 1)

    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP, "type": "minimax"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": H3_AUDIO_VAE}},
        "5": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": prompt_text,
                "width": width,
                "height": height,
                "length": length,
            },
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "7": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": ["1", 0], "shift_video": 12.0, "shift_audio": 3.0},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["7", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["5", 1],
                "seed": seed,
                "steps": H3_STEPS,
                "cfg": H3_CFG,
                "sampler_name": H3_SAMPLER,
                "scheduler": H3_SCHEDULER,
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {"samples": ["8", 0], "audio_vae": ["4", 0]},
        },
        "10": {
            "class_type": "SaveAudio",
            "inputs": {"audio": ["9", 0], "filename_prefix": filename_prefix},
        },
    }

    video_node: Optional[str] = None
    if H3_SAVE_VIDEO:
        graph["11"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        }
        graph["12"] = {
            "class_type": "CreateVideo",
            "inputs": {"images": ["11", 0], "fps": 24.0},
        }
        graph["13"] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["12", 0],
                "filename_prefix": filename_prefix,
                "format": "mp4",
                "codec": "h264",
            },
        }
        video_node = "13"

    return graph, "10", video_node


def _to_mono_24k_wav(audio_bytes: bytes) -> bytes:
    """Resample (stereo) audio bytes to mono 24 kHz WAV bytes."""
    import io
    import wave

    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())
    return buf.getvalue()


def run_worker(device: str) -> None:
    """Run the MiniMax H3 worker loop."""
    client = ComfyClient(COMFYUI_URL)

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
            if H3_FREE_VRAM:
                client.free_vram()
            break
        if req.get("type") != "request":
            continue

        req_id = req["id"]
        method = req["method"]
        kwargs = req["kwargs"]

        try:
            if method == "generate_voice_sample":
                character_name = kwargs["character_name"]
                description = kwargs["description"]
                output_dir = kwargs["output_dir"]
                static_voice_text = kwargs.get(
                    "static_voice_text", "Hello, this is my voice."
                )

                if not description or not description.strip():
                    print(json.dumps({"id": req_id, "success": False}), flush=True)
                    continue

                client.ping()

                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                output_file = str(out_dir / f"{character_name}.wav")

                length = _effective_length(static_voice_text)
                spoken = _truncate_text_for_length(static_voice_text, length)
                prompt_text = build_prompt(description, spoken)
                filename_prefix = f"minimax_h3/{character_name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

                graph, save_node, video_node = build_workflow(
                    prompt_text, filename_prefix, length=length
                )
                prompt_id = client.submit(graph)
                entry = client.wait_until_done(prompt_id)
                audio_bytes = client.get_saved_audio(entry, save_node)

                with open(output_file, "wb") as f:
                    f.write(_to_mono_24k_wav(audio_bytes))
                duration = _audio_seconds(length)

                video_file: Optional[str] = None
                if video_node is not None:
                    vid_bytes, _ = client.get_saved_video(entry, video_node)
                    video_file = str(out_dir / f"{character_name}.mp4")
                    with open(video_file, "wb") as f:
                        f.write(vid_bytes)

                if H3_FREE_VRAM:
                    client.free_vram()

                result = {
                    "id": req_id,
                    "success": True,
                    "output_file": output_file,
                    "duration": duration,
                }
                if video_file is not None:
                    result["video_file"] = video_file
                print(json.dumps(result), flush=True)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax H3 engine")
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
