#!/usr/bin/env python3
"""Build a combined WOT character-voice seed map from prior processed books.

For each source book directory, map character names to their canonical voice wav
and merge sources in priority order so a character keeps the highest-priority
book's voice when it appears in more than one source.

Source resolution per book:
  * If the book has a ``voices_map.json`` that is *clean* (every entry resolves
    to an existing non-sample, non-chapter, non-mp3 audio file), use it — it is
    the authoritative char->voice mapping and avoids picking up leftover
    ``*.cropped.wav`` / ``*_ref.wav`` artifacts.
  * Otherwise fall back to globbing plain-name audio files (``plain_voices``),
    which handles older books whose stale ``voices_map.json`` points at deleted
    ``.sampleN`` leftovers.

Usage:
    python scripts/build_wot_seed_map.py \
        --out voice_test/wot_book4_shadow_rising/seed_voices_map.json \
        --priority voice_test/wot_book3_dragon_reborn voice_test/teotw \
            voice_test/eye_of_the_world voice_test/new_spring
"""
import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AUDIO_EXT = {".wav", ".mp3", ".flac"}


def plain_voices(book_dir: Path):
    """Map plain character name -> wav path for one book directory."""
    result = {}
    for f in book_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in AUDIO_EXT:
            continue
        stem = f.stem
        # Skip .sampleN variant files (these are alt takes, not the canonical
        # character voice) and non-audio artifacts.
        if re.search(r"\.sample\d+$", stem, flags=re.IGNORECASE):
            continue
        # Skip audiobook chapter files (e.g. "chapter_00.mp3"): these are full
        # chapter recordings from a prior book, NOT character voices. Including
        # them would (a) seed bogus "chapter_XX" characters and (b) cause the
        # pipeline to copy a prior book's chapter MP3s over the current output.
        if re.search(r"^chapter_\d+$", stem, flags=re.IGNORECASE):
            continue
        # Use an absolute path so the seed loader (which resolves relative
        # paths against the seed-map's directory) finds the file regardless of
        # where the seed map lives.
        result[stem] = str(f.resolve())
    return result


def _voices_map_is_usable(book_dir: Path, voices: dict) -> bool:
    """A voices_map.json is usable only if every entry is a real, canonical
    character voice (exists, non-sample, non-chapter, audio ext). Otherwise it
    is stale (e.g. pointing at deleted .sampleN leftovers) and we fall back to
    globbing."""
    for path in voices.values():
        p = Path(path)
        if not p.is_absolute():
            p = book_dir / p
        if not p.is_file():
            return False
        stem = p.stem
        if re.search(r"\.sample\d+$", stem, flags=re.IGNORECASE):
            return False
        if re.search(r"^chapter_\d+$", stem, flags=re.IGNORECASE):
            return False
        if p.suffix.lower() not in AUDIO_EXT:
            return False
    return True


def book_voices(book_dir: Path):
    """Return the authoritative char->voice map for a book directory.

    Uses ``voices_map.json`` when it is clean, else globs plain-name audio.
    """
    vm = book_dir / "voices_map.json"
    if vm.is_file():
        try:
            data = json.loads(vm.read_text())
        except Exception:
            data = {}
        if isinstance(data, dict) and data and _voices_map_is_usable(book_dir, data):
            result = {}
            for name, path in data.items():
                p = Path(path)
                if not p.is_absolute():
                    p = book_dir / p
                result[name] = str(p.resolve())
            return result
    return plain_voices(book_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Path to write the seed voices_map.json")
    ap.add_argument("--priority", nargs="+", required=True,
                    help="Book dirs in priority order (highest first) for name conflicts")
    args = ap.parse_args()

    merged: dict = {}
    sources = {}
    for book in args.priority:
        book_dir = Path(book)
        voices = book_voices(book_dir)
        sources[book_dir.name] = voices
        for name, path in voices.items():
            merged.setdefault(name, path)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Wrote {len(merged)} seeded characters to {out}")
    for book_name, voices in sources.items():
        print(f"  {book_name}: {len(voices)} voices")
    print(f"  merged total: {len(merged)}")


if __name__ == "__main__":
    main()
