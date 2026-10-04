#!/usr/bin/env python3
"""Convert existing audiobooks (mp4/m4b/mp3...) into JBAB containers.

Post-processes a book you already have — no audio re-encoding, files are
stored as-is in the .jbab. Adds the text track two ways:

  --epub given : chapter text comes from the EPUB (best reading text);
                 whisper supplies word timings, anchored onto the EPUB
                 words with gap interpolation (same matcher as
                 jbab_align.py).
  no --epub    : the whisper transcript itself becomes the reading text
                 (clean narration transcribes very well) and word
                 timings are 1:1.

Handles:
  - a directory of chapter files (per-file text/timings)
  - a single big mp4/m4b with embedded chapters (whole-file text +
    timings; the app slices per chapter)
  - a single file without chapters (one unit)

Usage:
  uv run python scripts/jbab_convert.py <book_dir_or_file> [--epub x.epub]
      [--model medium] [--device cuda] [--compute-type float16]
      [--language en] [--out-dir DIR] [--no-pack] [--force]

Then load <name>.jbab on the device (adb push / SAF import).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from jbab_align import AUDIO_EXTS, align, strip_tts_markers, tokenize  # noqa: E402


def probe_chapters(path: Path) -> list[tuple[int, int]]:
    """Embedded chapter (start_ms, end_ms) list, [] if none."""
    import subprocess as sp
    out = sp.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_chapters", str(path)],
        capture_output=True, text=True).stdout
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    chs = []
    for c in data.get("chapters", []):
        chs.append((int(float(c["start_time"]) * 1000),
                    int(float(c["end_time"]) * 1000)))
    return chs


def whisper_words(model, audio: Path, language, vad=True):
    segs, _ = model.transcribe(str(audio), word_timestamps=True,
                               language=language, vad_filter=vad)
    words = []
    text = ""
    for seg in segs:
        text += seg.text
        for w in seg.words or []:
            words.append((w.word, w.start, w.end))
    return words, text.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("book", type=Path,
                    help="book directory or single audio file")
    ap.add_argument("--epub", type=Path, default=None)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--language", default=None)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="stage converted layout here (default: in place)")
    ap.add_argument("--no-pack", action="store_true",
                    help="write sidecars only, skip .jbab packing")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src: Path = args.book
    if src.is_file():
        work = args.out_dir or src.parent / (src.stem + "_jbab")
        work.mkdir(parents=True, exist_ok=True)
        units = [(work / src.name, src)]
        import shutil
        shutil.copy2(src, work / src.name)
    elif src.is_dir():
        work = args.out_dir or src
        if args.out_dir and args.out_dir != src:
            work.mkdir(parents=True, exist_ok=True)
            import shutil
            for p in sorted(src.iterdir()):
                if p.is_file():
                    shutil.copy2(p, work / p.name)
        units = [(work / p.name, p) for p in sorted(work.iterdir())
                 if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    else:
        print(f"no such path: {src}", file=sys.stderr)
        return 1
    if not units:
        print("no audio found", file=sys.stderr)
        return 1

    # EPUB chapter texts, if provided
    epub_texts: list[str] = []
    if args.epub:
        from audiobook_generator.parse_chapter import (
            cleanup_text, parse_epub_to_chapters)
        chap_lists = parse_epub_to_chapters(str(args.epub))
        for blocks in chap_lists:
            t = cleanup_text("\n".join(b.text for b in blocks))
            if t.strip():
                epub_texts.append(t.strip())
        print(f"epub: {len(epub_texts)} chapters")

    from faster_whisper import WhisperModel
    print(f"loading whisper '{args.model}' on {args.device} ...")
    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type)

    ok = 0
    for i, (dst, real) in enumerate(units):
        stem = dst.stem
        txt_path = dst.parent / (stem + ".txt")
        wj_path = dst.parent / (stem + ".words.json")
        if txt_path.exists() and wj_path.exists() and not args.force:
            print(f"  {stem}: sidecars exist, skipping (--force to redo)")
            ok += 1
            continue

        wwords, wtext = whisper_words(model, real, args.language)
        if not wwords:
            print(f"  {stem}: whisper produced nothing, skipping")
            continue

        # pick the reading text: EPUB chapter text or the transcript
        epub_t = epub_texts[i] if i < len(epub_texts) else None
        if epub_t:
            epub_t = strip_tts_markers(epub_t)
            toks = tokenize(epub_t)
            times = align(wwords, [epub_t[o:l] for o, l in toks])
            if times:
                text = epub_t
                src_kind = "epub-aligned"
            else:
                text, times = build_from_transcript(wwords, wtext)
                src_kind = "transcript (epub align too weak)"
        else:
            text, times = build_from_transcript(wwords, wtext)
            src_kind = "transcript"
        if times is None:
            print(f"  {stem}: no usable timings, skipping")
            continue

        txt_path.write_text(text, encoding="utf-8")
        wj_path.write_text(json.dumps({"v": 1, "words": times}),
                           encoding="utf-8")
        ok += 1
        print(f"  {stem}: {len(times)} words [{src_kind}]")

    if ok == 0:
        print("nothing converted", file=sys.stderr)
        return 1

    if args.no_pack:
        print(f"done: {ok} units (no pack)")
        return 0
    print("packing...")
    r = subprocess.run([sys.executable, str(HERE / "jbab_pack.py"),
                        str(work), "--force"] +
                       (["--title", src.stem] if src.is_file() else []))
    return r.returncode


def build_from_transcript(wwords, wtext):
    """Transcript text + 1:1 word timings."""
    text = "".join(w for w, _, _ in wwords).strip()
    toks = tokenize(text)
    if len(toks) != len(wwords):
        # rare: internal spaces in whisper words; scale evenly instead
        if not wwords:
            return None, None
        span = wwords[-1][2] * 1000
        n = len(toks)
        times = [[int(span * i / n), int(span * (i + 1) / n)]
                 for i in range(n)]
        return wtext, times
    times = [[int(s * 1000), int(e * 1000)] for _, s, e in wwords]
    return text, times


if __name__ == "__main__":
    raise SystemExit(main())
