#!/usr/bin/env python3
"""Rebuild the listening-review clip folder from benchmark run outputs.

The clipping benchmark (scripts/benchmark_clipping.py) writes raw_*/clip_* wavs
and a summary.json for each sampled line. This script collects the FLAGGED lines
(premature clips, or endings missing after clipping) across one or more runs and
copies their raw + clipped audio into voice_test/clip_review/ so a human can
listen and judge whether the clip genuinely lost content.

Usage:
    python scripts/generate_review_clips.py \
        --voice narr=/tmp/abg_energy4 \
        --voice ra3=/tmp/abg_energy4_ra3 \
        [--out voice_test/clip_review]
"""
import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="append", required=True,
                    help="name=/path/to/run-dir (repeatable)")
    ap.add_argument("--out", default=str(REPO / "voice_test" / "clip_review"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for spec in args.voice:
        name, run_dir = spec.split("=", 1)
        run_dir = Path(run_dir)
        results = json.load(open(run_dir / "summary.json"))
        for d in results:
            if "error" in d:
                continue
            if not d["premature"] and d["clip_end_ok"]:
                continue
            tag = "PREM" if d["premature"] else "MISS"
            c, ln = d["chapter"], d["line"]
            stem = f"{name}_{tag}_c{c}_l{ln}"
            raw_src = run_dir / f"raw_c{c}_l{ln}.wav"
            clip_src = run_dir / f"clip_c{c}_l{ln}.wav"
            raw_dst = out / f"{stem}_raw.wav"
            clip_dst = out / f"{stem}_clip.wav"
            if raw_src.exists():
                shutil.copy(raw_src, raw_dst)
            if clip_src.exists():
                shutil.copy(clip_src, clip_dst)
            manifest.append({
                "voice": name, "tag": tag, "chapter": c, "line": ln,
                "text": d["text"], "raw_det": d["raw_det"], "clip_det": d["clip_det"],
                "ratio": d["ratio"], "premature": d["premature"],
                "clip_end_ok": d["clip_end_ok"],
                "raw": f"{stem}_raw.wav", "clip": f"{stem}_clip.wav",
            })

    manifest.sort(key=lambda m: (m["voice"], m["line"]))
    with open(out / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    lines = ["Listening review of flagged clips (premature / missing ending)\n"]
    lines.append(f"{'voice':6} {'tag':4} {'line':4}  ending-ok  line text")
    for m in manifest:
        lines.append(
            f"{m['voice']:6} {m['tag']:4} l{m['line']:<3}  "
            f"{str(m['clip_end_ok']):9}  {m['text'][:48]}"
        )
        lines.append(f"          raw : ...{m['raw_det'][-60:]}")
        lines.append(f"          clip: ...{m['clip_det'][-60:]}")
    with open(out / "MANIFEST.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(manifest)} flagged clips to {out}")
    for m in manifest:
        print(f"  {m['voice']}_{m['tag']}_c{m['chapter']}_l{m['line']}")


if __name__ == "__main__":
    main()
