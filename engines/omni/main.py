#!/usr/bin/env python3
"""OmniVoice engine - standalone subprocess entry point.

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
MODEL_PATH = "drbaph/OmniVoice-bf16"
NUM_STEP = 32
CLASS_TEMPERATURE = 0.5
CLASS_TEMPERATURE_FALLBACK = 3.0


def probe() -> None:
    """Print engine capabilities as JSON."""
    caps = {
        "name": "omni",
        "methods": ["generate_line", "generate_voice_sample"],
        "sample_rate": SAMPLE_RATE,
    }
    print(json.dumps(caps))


def convert_description_to_instruct(description: str) -> str:
    """Convert a voice description to OmniVoice instruct format.

    Accepts either:
    - Universal JSON: {"gender": "male", "age": "middle-aged", ...}
    - Legacy comma-separated: "male, middle-aged, moderate pitch"
    """
    # Try parsing as JSON first
    try:
        text = description.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            if text.endswith("```"):
                text = text[:-3]
        obj = json.loads(text)
        # Convert from universal JSON to OmniVoice format
        parts = []
        gender = obj.get("gender", "").lower()
        if gender in ("male", "female"):
            parts.append(gender)

        age = obj.get("age", "").lower()
        age_map = {"child": "child", "teenager": "teenager", "young adult": "young adult", "young": "young adult", "middle-aged": "middle-aged", "middle aged": "middle-aged", "elderly": "elderly", "old": "elderly"}
        if age in age_map:
            parts.append(age_map[age])

        pitch = obj.get("pitch", "moderate").lower()
        pitch_map = {"very low": "very low pitch", "low": "low pitch", "moderate": "moderate pitch", "high": "high pitch", "very high": "very high pitch"}
        if pitch in pitch_map:
            parts.append(pitch_map[pitch])

        accent = obj.get("accent", "").lower()
        accent_map = {"american": "american accent", "british": "british accent", "australian": "australian accent", "canadian": "canadian accent", "indian": "indian accent", "chinese": "chinese accent", "korean": "korean accent", "japanese": "japanese accent", "portuguese": "portuguese accent", "russian": "russian accent"}
        if accent in accent_map:
            parts.append(accent_map[accent])

        style = obj.get("style", "")
        if style:
            if style.lower() == "whisper":
                parts.append("whisper")

        return ", ".join(parts)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Legacy comma-separated format
    instruct = description.replace(".", ",")
    parts = [p.strip().lower() for p in instruct.split(",") if p.strip()]

    gender_map = {"male": "male", "female": "female"}
    age_map = {
        "child": "child", "young": "young adult", "teen": "teenager",
        "teenager": "teenager", "young adult": "young adult",
        "middle aged": "middle-aged", "middle-aged": "middle-aged",
        "elderly": "elderly", "old": "elderly",
    }
    pitch_map = {
        "very low": "very low pitch", "very low pitch": "very low pitch",
        "low": "low pitch", "low pitch": "low pitch",
        "medium": "moderate pitch", "mid": "moderate pitch",
        "moderate": "moderate pitch", "moderate pitch": "moderate pitch",
        "high": "high pitch", "high pitch": "high pitch",
        "very high": "very high pitch", "very high pitch": "very high pitch",
    }
    accent_map = {
        "american": "american accent", "british": "british accent",
        "australian": "australian accent", "canadian": "canadian accent",
        "indian": "indian accent", "chinese": "chinese accent",
        "korean": "korean accent", "japanese": "japanese accent",
        "portuguese": "portuguese accent", "russian": "russian accent",
    }

    mapped_parts = []
    gender_val = None
    age_val = None
    for part in parts:
        if part in gender_map:
            mapped_parts.append(gender_map[part])
            gender_val = gender_map[part]
        elif part in age_map:
            mapped_parts.append(age_map[part])
            age_val = age_map[part]
        elif part in pitch_map:
            mapped_parts.append(pitch_map[part])
        elif part == "whisper":
            mapped_parts.append("whisper")
        elif part in accent_map:
            mapped_parts.append(accent_map[part])
        elif part.endswith(" accent"):
            mapped_parts.append(part)
        elif any(c in part for c in "河南陕西四川贵云南桂济石甘宁青岛东北话"):
            mapped_parts.append(part)

    return ", ".join(mapped_parts)


def get_fallback_instruct(description: str) -> Optional[str]:
    """Get fallback instruct from description."""
    parts = [p.strip().lower() for p in description.replace(".", ",").split(",") if p.strip()]
    for part in parts:
        if part in ("male", "female"):
            return part
    return None


def run_worker(device: str) -> None:
    """Run the OmniVoice worker loop.

    Reads JSON requests from stdin, writes JSON responses to stdout.
    """
    from omnivoice import OmniVoice
    import torch
    import soundfile as sf

    model = None
    # Voice clone prompts can be large (ASR + voice embeddings). Keep them
    # bounded so distinct voice refs (e.g. per-chapter narrator seeds) don't
    # accumulate unbounded GPU memory.
    _voice_clone_prompts: dict[str, Any] = {}
    _MAX_CLONE_PROMPTS = 32

    def load_model(device: str) -> None:
        nonlocal model
        if model is not None:
            return
        model = OmniVoice.from_pretrained(
            MODEL_PATH,
            device_map=device,
            dtype=torch.float16,
        )

    def _load_asr_lazy() -> None:
        """Load the ASR model only when a voice clone prompt is requested.

        The celebrity path uses voice *design* (generate_voice_sample), which
        never needs ASR. Loading it unconditionally wasted several GB on the
        shared 24GB GPU, causing OOM when Whisper validation also needs memory.
        """
        try:
            model.load_asr_model()
        except Exception as e:
            print(f"  Warning: Could not pre-load ASR model: {e}", file=sys.stderr)

    def _get_voice_clone_prompt(voice_path: str) -> Any:
        if voice_path not in _voice_clone_prompts:
            if len(_voice_clone_prompts) >= _MAX_CLONE_PROMPTS:
                _voice_clone_prompts.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            _load_asr_lazy()
            _voice_clone_prompts[voice_path] = model.create_voice_clone_prompt(
                ref_audio=voice_path,
                preprocess_prompt=True,
            )
        return _voice_clone_prompts[voice_path]

    def _free_activations() -> None:
        """Release cached-but-unused GPU memory after a generation."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
                static_voice_text = kwargs.get("static_voice_text", "")

                if not description or not description.strip():
                    print(json.dumps({"id": req_id, "success": False}), flush=True)
                    continue

                instruct = convert_description_to_instruct(description)

                try:
                    audio = model.generate(
                        text=static_voice_text,
                        num_step=NUM_STEP,
                        class_temperature=CLASS_TEMPERATURE,
                        instruct=instruct,
                    )
                    if audio is None or len(audio) == 0:
                        print(json.dumps({"id": req_id, "success": False}), flush=True)
                        continue
                    audio_arr = audio[0]
                    if hasattr(audio_arr, 'numel'):
                        audio_len = audio_arr.numel()
                    else:
                        audio_len = len(audio_arr)
                    if audio_len == 0:
                        print(json.dumps({"id": req_id, "success": False}), flush=True)
                        continue

                    out_dir = Path(output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    output_file = str(out_dir / f"{character_name}.wav")
                    if hasattr(audio[0], 'cpu'):
                        audio_np = audio[0].cpu().numpy()
                    else:
                        audio_np = audio[0]
                    sf.write(output_file, audio_np, SAMPLE_RATE)
                    duration = len(audio[0]) / SAMPLE_RATE
                    print(json.dumps({
                        "id": req_id,
                        "success": True,
                        "output_file": output_file,
                        "duration": duration,
                    }), flush=True)
                    _free_activations()

                except ValueError as e:
                    error_msg = str(e)
                    if "Conflicting instruct items" in error_msg or "Each category" in error_msg:
                        fallback = get_fallback_instruct(description)
                        if fallback:
                            try:
                                audio = model.generate(
                                    text=static_voice_text,
                                    num_step=NUM_STEP,
                                    class_temperature=CLASS_TEMPERATURE_FALLBACK,
                                    instruct=fallback,
                                )
                                if audio is None or len(audio) == 0:
                                    print(json.dumps({"id": req_id, "success": False}), flush=True)
                                    continue
                                audio_arr = audio[0]
                                audio_len = audio_arr.numel() if hasattr(audio_arr, 'numel') else len(audio_arr)
                                if audio_len == 0:
                                    print(json.dumps({"id": req_id, "success": False}), flush=True)
                                    continue
                                out_dir = Path(output_dir)
                                out_dir.mkdir(parents=True, exist_ok=True)
                                output_file = str(out_dir / f"{character_name}.wav")
                                if hasattr(audio[0], 'cpu'):
                                    audio_np = audio[0].cpu().numpy()
                                else:
                                    audio_np = audio[0]
                                sf.write(output_file, audio_np, SAMPLE_RATE)
                                duration = len(audio[0]) / SAMPLE_RATE
                                print(json.dumps({
                                    "id": req_id,
                                    "success": True,
                                    "output_file": output_file,
                                    "duration": duration,
                                }), flush=True)
                                _free_activations()
                            except Exception:
                                print(json.dumps({"id": req_id, "success": False}), flush=True)
                        else:
                            print(json.dumps({"id": req_id, "success": False}), flush=True)
                    else:
                        print(json.dumps({"id": req_id, "success": False}), flush=True)

            elif method == "generate_line":
                text = kwargs["text"]
                voice_path = kwargs["voice_path"]
                output_path = kwargs["output_path"]

                voice_clone_prompt = _get_voice_clone_prompt(voice_path)

                audio = model.generate(
                    text=text,
                    voice_clone_prompt=voice_clone_prompt,
                )
                if audio is None or len(audio) == 0:
                    print(json.dumps({"id": req_id, "success": False}), flush=True)
                    continue
                audio_arr = audio[0]
                audio_len = audio_arr.numel() if hasattr(audio_arr, 'numel') else len(audio_arr)
                if audio_len == 0:
                    print(json.dumps({"id": req_id, "success": False}), flush=True)
                    continue

                if hasattr(audio[0], 'cpu'):
                    sf.write(output_path, audio[0].cpu().numpy(), SAMPLE_RATE)
                else:
                    sf.write(output_path, audio[0], SAMPLE_RATE)
                print(json.dumps({"id": req_id, "success": True}), flush=True)
                _free_activations()

            else:
                print(json.dumps({"id": req_id, "error": f"Unknown method: {method}"}), flush=True)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(json.dumps({"id": req_id, "error": str(e), "traceback": tb}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniVoice TTS engine")
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use")
    parser.add_argument("--probe", action="store_true", help="Print engine capabilities and exit")
    args = parser.parse_args()

    if args.probe:
        probe()
        return

    run_worker(args.device)


if __name__ == "__main__":
    main()
