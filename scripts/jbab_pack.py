#!/usr/bin/env python3
"""Pack a JBAB book directory into a single .jbab container.

A .jbab file is a ZIP bundle: audio chapters are STORED (AAC-in-mp4
is already compressed; restoring is free) while text, word timings and
the manifest are deflate-compressed. The app extracts it once on first
scan and deletes the archive, so it exists purely as a transfer unit.

    books/WOT1/
      01 - Prologue.mp4
      01 - Prologue.txt
      01 - Prologue.words.json

    ->  WOT1.jbab

Usage:
    python scripts/jbab_pack.py <book_dir> [-o out.jbab] [--title T]
                                [--author A] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

AUDIO = {".mp3", ".m4a", ".m4b", ".mp4", ".m4v", ".aac", ".ogg", ".oga",
         ".opus", ".flac", ".wav", ".wma"}
TEXTY = {".txt", ".json", ".jpg", ".png", ".opf"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("book_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--author", default=None)
    ap.add_argument("--series", default=None)
    ap.add_argument("--series-num", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    book: Path = args.book_dir
    if not book.is_dir():
        print(f"not a directory: {book}", file=sys.stderr)
        return 1
    out: Path = args.out or book.with_suffix(".jbab")
    if out.exists() and not args.force:
        print(f"exists: {out} (--force to rebuild)", file=sys.stderr)
        return 1

    files = sorted(p for p in book.rglob("*") if p.is_file())
    audio = [p for p in files if p.suffix.lower() in AUDIO]
    if not audio:
        print(f"no chapter audio in {book}", file=sys.stderr)
        return 1

    # keep only what the app needs: audio + same-stem sidecars (+ cover
    # images). Pipeline work dirs are full of TTS junk; skip it all.
    keep = set()
    for a in audio:
        keep.add(a)
        for ext in (".txt", ".words.json"):
            s = a.with_suffix(ext)
            if s.is_file():
                keep.add(s)
    keep.update(p for p in files
                if p.suffix.lower() in TEXTY
                and p.name in ("manifest.json", "cover.jpg", "cover.png"))
    files = [p for p in files if p in keep]

    total = sum(p.stat().st_size for p in files)
    if total > 3.8e9:
        print(f"WARNING: {total/1e9:.2f} GB exceeds the non-zip64 "
              f"container limit (~3.8 GB); split the book further.",
              file=sys.stderr)
        return 1

    manifest = {
        "v": 1,
        "title": args.title or book.name,
        "author": args.author or "",
        "series": args.series or "",
        "series_num": args.series_num or 0,
        "chapters": [{"file": str(p.relative_to(book)), "title": p.stem}
                     for p in audio],
    }

    with zipfile.ZipFile(out, "w", allowZip64=False) as z:
        z.writestr("manifest.json",
                   json.dumps(manifest, indent=1, ensure_ascii=False),
                   compress_type=zipfile.ZIP_DEFLATED)
        for p in files:
            arc = str(p.relative_to(book))
            comp = (zipfile.ZIP_DEFLATED if p.suffix.lower() in TEXTY
                    else zipfile.ZIP_STORED)
            z.write(p, arc, compress_type=comp)

    size = out.stat().st_size
    print(f"{out.name}: {len(audio)} chapters, {len(files)} files, "
          f"{size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
