#!/usr/bin/env python3
"""Build a character -> celebrity traceability map for a book.

For most characters the current book's ``characters_descriptions.json`` records
who voiced them (``celebrity_voice``). But seeded/recurring characters often
have an empty/None ``celebrity_voice`` here, because their voice was carried
over from an earlier book as a ``{char}.wav`` rather than being celebrity-matched
in this book. To keep those voices traceable, this tool digs through the prior
books' ``characters_descriptions.json`` (most-recent-first) and fills in the
celebrity from the first book that assigned one.

Usage:
    python scripts/build_celebrity_trace.py \
        --book voice_test/wot_book5_fires_of_heaven \
        --prior voice_test/wot_book4_shadow_rising voice_test/wot_book3_dragon_reborn \
        --out voice_test/wot_book5_fires_of_heaven/celebrity_trace.json
"""
import argparse
import json
import sys
from pathlib import Path


def _celebrity_for(obj) -> str:
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except (json.JSONDecodeError, TypeError):
            return ""
    if isinstance(obj, dict):
        return (obj.get("celebrity_voice") or "").strip()
    return ""


def character_celebrity_map(book_dirs):
    """Return {character: celebrity} by scanning current + prior book descriptions.

    ``book_dirs`` is ordered by priority (the current book first, then prior
    books most-recent-first). The first book that records a celebrity for a
    character wins, so a newly-assigned celebrity in the current book takes
    precedence over a stale assignment from an older book.
    """
    result = {}
    for d in book_dirs:
        d = Path(d)
        desc_file = d / "characters_descriptions.json"
        if not desc_file.is_file():
            continue
        try:
            descs = json.loads(desc_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for char, v in descs.items():
            if char in result:
                continue
            celeb = _celebrity_for(v)
            if celeb:
                result[char] = celeb
    return result


def _norm(name: str) -> str:
    """Normalize a name for comparison: lowercase, drop spaces/underscores."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def filename_celebrity_map(book_dir, known):
    """Fill celebrity from celebrity-named voice files.

    A character whose voice file is ``{char}.wav`` was seeded (its celebrity is
    recovered from prior books, if anywhere). A character whose voice file is
    NOT named after the character is a ``{celebrity}.wav`` file, so its stem is a
    normalized celebrity identifier we can use for traceability even when the
    current book's description omitted ``celebrity_voice``.
    """
    book_dir = Path(book_dir)
    vm_file = book_dir / "voices_map.json"
    if not vm_file.is_file():
        return {}
    try:
        vm = json.loads(vm_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    filled = {}
    for char, fname in vm.items():
        if char in known or not isinstance(fname, str):
            continue
        stem = Path(fname).stem
        if stem and _norm(stem) != _norm(char):
            filled[char] = stem
    return filled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True, help="Current book output dir")
    ap.add_argument("--prior", nargs="*", default=[],
                    help="Prior book output dirs, most-recent-first")
    ap.add_argument("--out", required=True, help="Path to write celebrity_trace.json")
    args = ap.parse_args()

    mapping = character_celebrity_map([args.book] + args.prior)
    mapping.update(filename_celebrity_map(args.book, mapping))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(mapping)} character->celebrity mappings to {out}")


if __name__ == "__main__":
    main()
