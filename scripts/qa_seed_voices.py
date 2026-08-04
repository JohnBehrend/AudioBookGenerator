#!/usr/bin/env python3
"""QA pass over the seeded character voices for the next WOT book.

For every voice in a seed map, run Whisper and extract the reference text it
actually speaks, then score how well that transcription matches the configured
static reference text (``static_voice_text``). This verifies two things that
matter for reliable TTS later:

1. The audio is real, transcribable speech (not broken/garbage/empty).
2. The voice already speaks the reference text well enough (>= 80% word match)
   that the pipeline can copy it directly instead of re-cloning — and that the
   extracted ref_text will align with the configured static text during the
   Whisper-based clipping/validation steps.

Classification per voice:
  PASS    >= 80% of reference words matched (speaks the reference text)
  PARTIAL 30-79% matched (real speech, but not the reference text)
  FAIL    < 30% matched or empty/error transcription

Usage:
    python scripts/qa_seed_voices.py \
        --seed-map voice_test/wot_book4_shadow_rising/seed_voices_map.json \
        --out voice_test/wot_book4_shadow_rising/seed_qa_report.json \
        --device cuda
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audiobook_generator.audiobook_generator import setup_validation_model  # noqa: E402
from audiobook_generator.config import DEFAULTS  # noqa: E402
from audiobook_generator.generate_voice_samples import _word_match_count  # noqa: E402
from audiobook_generator.utils import transcribe_audio_for_ref_text  # noqa: E402

REF_TEXT = DEFAULTS["static_voice_text"]
REF_WORDS = REF_TEXT.lower().split()

PASS_THRESHOLD = 0.8
PARTIAL_THRESHOLD = 0.3


def classify(ratio: float) -> str:
    if ratio is None or ratio < PARTIAL_THRESHOLD:
        return "FAIL"
    if ratio >= PASS_THRESHOLD:
        return "PASS"
    return "PARTIAL"


def qa_voice(model, wav_path: str):
    """Run Whisper on one voice and return (ref_text, ratio, verdict)."""
    try:
        ref_text = transcribe_audio_for_ref_text(model, wav_path, verbose=False)
    except Exception as e:
        return None, 0.0, f"ERROR: {e}"
    if not ref_text:
        return None, 0.0, "EMPTY"
    ratio = _word_match_count(REF_WORDS, ref_text.lower()) / len(REF_WORDS)
    return ref_text, ratio, classify(ratio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Only QA the first N voices (for a quick sanity check)")
    args = ap.parse_args()

    seed_map = json.loads(Path(args.seed_map).read_text())
    if args.limit:
        seed_map = dict(list(seed_map.items())[: args.limit])

    model = setup_validation_model(args.device, cpu=args.cpu)

    results = {}
    for i, (char, wav) in enumerate(seed_map.items(), 1):
        ref_text, ratio, verdict = qa_voice(model, wav)
        results[char] = {
            "wav": wav,
            "ref_text": ref_text,
            "match_ratio": round(ratio, 3),
            "verdict": verdict,
        }
        if i % 20 == 0 or i == len(seed_map):
            print(f"  [{i}/{len(seed_map)}] {char}: {verdict} ({ratio:.0%})", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    for r in results.values():
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(f"\nWrote {len(results)} voice QA results to {out}")
    print(f"  PASS: {counts.get('PASS', 0)}")
    print(f"  PARTIAL: {counts.get('PARTIAL', 0)}")
    print(f"  FAIL: {counts.get('FAIL', 0)}")
    if counts.get("FAIL", 0):
        print("  Failing voices:", [c for c, r in results.items() if r["verdict"] == "FAIL"])


if __name__ == "__main__":
    main()
