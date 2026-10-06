#!/usr/bin/env python3
"""Batch-convert pipeline book dirs into .jbab containers.

For each book dir: stage a clean layout (chapter audio + normalized
text sidecars), run the ASR alignment pass (waits for free GPU
memory first), pack into <out>/<BookName>.jbab. Resumable: books whose
.jbab already exists are skipped.

Book metadata (title/series/number per dir) comes from --books JSON
(default voice_test/books.json, gitignored — book data stays out of
the repo).

    uv run python scripts/jbab_batch.py [--only SUBSTR ...]
        [--model medium] [--device cuda] [--compute-type float16]
        [--out voice_test/jbab_out] [--skip-align]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
AUDIO = {".mp3", ".m4a", ".m4b", ".mp4", ".ogg", ".opus", ".flac", ".wav",
         ".aac"}

DEFAULT_BOOKS = HERE.parent / "voice_test" / "books.json"


def load_books(path: Path) -> dict[str, tuple[str, str, int]]:
    """Load dir -> (title, series, number) metadata from a JSON file.

    The books file is personal data (which books you generated, their local
    dir names) and lives OUTSIDE the repo, gitignored under voice_test/.
    Format: {"<dir-name>": {"title": str, "series": str, "num": int}, ...}
    Missing file or entries are fine: the script falls back to dir names.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"warning: could not parse {path}: {e}", file=sys.stderr)
        return {}
    return {
        k: (v.get("title", k), v.get("series", ""), int(v.get("num", 0)))
        for k, v in raw.items() if isinstance(v, dict)
    }


def wait_for_gpu(min_free_mb: int, poll_s: int = 60) -> None:
    """Block until the GPU has at least min_free_mb of free memory."""
    while True:
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout
            free = max(int(x.strip()) for x in out.splitlines() if x.strip())
        except Exception:
            return  # no nvidia-smi; just go
        if free >= min_free_mb:
            return
        print(f"  GPU busy ({free} MB free, need {min_free_mb}); "
              f"waiting {poll_s}s ...", flush=True)
        time.sleep(poll_s)


def chapter_num(p: Path) -> int | None:
    m = re.search(r"(\d+)$", p.stem)
    return int(m.group(1)) if m else None


CHAPTER_RE = re.compile(r"^(chapter|ch|part)[_\- ]?\d+\.(mp3|m4a|mp4|ogg|opus|flac)$", re.I)


def pick_audio(src: Path) -> list[Path]:
    """Prefer per-chapter files; fall back to the merged m4b/mp4."""
    chapter_files = [p for p in src.iterdir()
                     if p.is_file() and CHAPTER_RE.match(p.name)]
    if chapter_files:
        return sorted(chapter_files, key=lambda p: chapter_num(p) or 0)
    return [p for p in sorted(src.iterdir())
            if p.is_file() and p.suffix.lower() in {".m4b", ".m4a", ".mp4"}]


def stage(src: Path, dst: Path) -> int:
    """Clean layout: audio files + text sidecars under matching stems."""
    dst.mkdir(parents=True, exist_ok=True)
    txts = {chapter_num(p): p for p in src.glob("*.txt")
            if chapter_num(p) is not None and "prompt" not in p.stem
            and "think" not in p.stem and "result" not in p.stem}
    n = 0
    for a in pick_audio(src):
        shutil.copy2(a, dst / a.name)
        num = chapter_num(a)
        t = txts.get(num)
        if t:
            shutil.copy2(t, dst / (a.stem + ".txt"))
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).parent.parent / "voice_test")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent.parent / "voice_test"
                    / "jbab_out")
    ap.add_argument("--only", nargs="*", default=None,
                    help="only dirs containing this substring")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="skip dirs with exactly these names")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--gpu-free-mb", type=int, default=6000)
    ap.add_argument("--skip-align", action="store_true",
                    help="stage + pack only (no words.json; the app "
                         "falls back to even word timing)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-align + repack even if the .jbab exists")
    ap.add_argument("--force", action="store_true",
                    help="pass --force to the aligner (re-do chapters "
                         "that already have words.json)")
    ap.add_argument("--sidecar-text", action="store_true",
                    help="keep using the .txt sidecar as reading text "
                         "(default: ASR transcript, which always "
                         "matches what was spoken)")
    ap.add_argument("--one", type=Path, default=None,
                    help="process a single book dir (stage+align+pack)")
    ap.add_argument("--title", default=None,
                    help="book title for --one / manifest")
    ap.add_argument("--books", type=Path, default=DEFAULT_BOOKS,
                    help="JSON mapping book dir -> {title, series, num} "
                         "(personal data, kept out of the repo; "
                         "missing file = dir names used as titles)")
    args = ap.parse_args()
    args.books_map = load_books(args.books)

    if args.one:
        args.out.mkdir(parents=True, exist_ok=True)
        title = args.title or args.one.name
        return process_book(args.one, title, args)
    return run_batch(args)


def process_book(book: Path, title: str, args) -> int:
    """stage -> (ASR align, GPU-gated) -> pack one book dir."""
    title, series, snum = args.books_map.get(book.name, (title, "", 0))
    out_jbab = args.out / f"{title}.jbab"
    print(f"\n=== {book.name} -> {out_jbab.name} ===", flush=True)
    if out_jbab.exists() and not getattr(args, "refresh", False):
        print("  .jbab exists, skipping", flush=True)
        return 0

    staged = args.out / "_staging" / title
    n = stage(book, staged)
    if n == 0:
        print("  no audio, skipping", flush=True)
        return 1
    print(f"  staged {n} chapters", flush=True)

    if not args.skip_align:
        wait_for_gpu(args.gpu_free_mb)
        cmd = [sys.executable, str(HERE / "jbab_align.py"), str(staged),
               "--model", args.model, "--device", args.device,
               "--compute-type", args.compute_type]
        if getattr(args, "force", False):
            cmd.append("--force")
        if not getattr(args, "sidecar_text", False):
            cmd.append("--transcript")
        r = subprocess.run(cmd, cwd=str(HERE.parent))
        if r.returncode != 0:
            print(f"  align failed for {book.name} "
                  "(packing without timings)", flush=True)

    pack_cmd = [sys.executable, str(HERE / "jbab_pack.py"), str(staged),
                "-o", str(out_jbab), "--title", title, "--force"]
    if series:
        pack_cmd += ["--series", series, "--series-num", str(snum)]
    r = subprocess.run(pack_cmd, cwd=str(HERE.parent))
    if r.returncode == 0:
        print(f"  packed {out_jbab.name}", flush=True)
        return 0
    print(f"  PACK FAILED for {book.name}", flush=True)
    return 1


def run_batch(args) -> int:
    books = [d for d in sorted(args.root.iterdir()) if d.is_dir()]
    if args.only:
        books = [d for d in books if any(s in d.name for s in args.only)]
    for s in args.exclude:
        books = [d for d in books if d.name != s]  # exact-name skips

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(books)} books -> {args.out}", flush=True)

    for book in books:
        title, _, _ = args.books_map.get(book.name, (book.name, "", 0))
        process_book(book, title, args)
    print("\nbatch complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
