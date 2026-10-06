#!/usr/bin/env python3
"""JBAB word-alignment pass: emit per-word timings for the reader view.

For each chapter audio file in a book directory, runs faster-whisper
with word timestamps and merges them onto the chapter text (the .txt
sidecar is ground truth for WHAT is displayed; whisper provides WHEN).
Output: <chapter>.words.json next to the audio, as

    {"v": 1, "words": [[start_ms, end_ms], ...]}

one entry per whitespace token of the .txt file, times relative to the
start of the audio file. JBAB's reader highlights the token sounding at
the current playback position; tap a word to seek.

Usage (from the AudioBookGenerator repo):
    uv run python scripts/jbab_align.py <book_dir> \
        [--model medium] [--device cuda] [--compute-type float16] \
        [--txt-dir DIR] [--language en]

If a chapter has no .txt sidecar it is skipped (use --txt-dir to point
at the pipeline's chapter text output).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

AUDIO_EXTS = {".mp3", ".m4a", ".m4b", ".mp4", ".ogg", ".opus", ".flac", ".wav", ".aac"}
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w']+", re.UNICODE)


def norm(w: str) -> str:
    return _PUNCT.sub("", w).lower()


# TTS chunk markers ("Line 12: ") are written into the chapter text but
# never spoken; strip them before tokenizing so text words == spoken
# words and the karaoke alignment stays tight.
_LINE_MARKER = re.compile(r"(?m)^Line \d+:[ \t]?")


def strip_tts_markers(text: str) -> str:
    return _LINE_MARKER.sub("", text)


def tokenize(text: str) -> list[tuple[int, int]]:
    """Byte offsets of whitespace-delimited tokens (matches the app)."""
    toks = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        s = i
        while i < n and not text[i].isspace():
            i += 1
        toks.append((s, i))
    return toks


def align(whisper_words: list[tuple[str, float, float]],
          twords: list[str]) -> list[tuple[int, int]] | None:
    """Merge whisper timings onto text word slots via banded LCS.

    Full monotonic alignment (not a greedy window), so as many text
    words as possible anchor directly to whisper word timestamps; only
    true gaps get interpolated. Returns [start_ms, end_ms] per slot.
    """
    if not whisper_words or not twords:
        return None
    W = len(whisper_words)
    n = len(twords)
    wn = [norm(w[0]) for w in whisper_words]
    tn = [norm(t) for t in twords]
    # character-trigram sets for fuzzy scoring: garbled transcriptions
    # ("Mikal" for "Michael") become usable anchors instead of gaps
    def _tri(w):
        w = w.lower()
        if len(w) < 3:
            return {w}
        return {w[i:i + 3] for i in range(len(w) - 2)}
    wtri = [_tri(w) for w in wn]
    ttri = [_tri(t) for t in tn]

    # banded DP: dp[i][j] -> (score, prev (i', j')) LCS-style.
    # The band is centered on the expected AUDIO TIME of text word i,
    # not its index: the audio may read a slightly different text
    # (typos, normalized names), so index proportionality drifts while
    # audio time stays honest.
    total_t = whisper_words[-1][2] if whisper_words else 0.0
    t_band = max(20.0, total_t * 0.10)
    dp = [dict() for _ in range(n)]
    starts = [w[1] for w in whisper_words]
    lo = hi = 0
    for i in range(n):
        E = (i + 0.5) / n * total_t
        while lo < W and starts[lo] < E - t_band:
            lo += 1
        if hi < lo:
            hi = lo
        while hi + 1 < W and starts[hi + 1] <= E + t_band:
            hi += 1
        j0, j1 = lo, hi
        if j1 < j0:
            # band beyond the whisper text: carry the best score
            # forward so long unspoken text tails can be skipped
            if i > 0:
                pv = dp[i - 1].get(W - 1)
                if pv:
                    dp[i][W - 1] = (pv[0], (i - 1, W - 1))
            continue
        ti = tn[i]
        row_prev = dp[i - 1] if i > 0 else {}
        for j in range(j0, j1 + 1):
            # skips cost, so the path cannot slide past genuine matches
            # into later occurrences of repeated strings
            best = (-0.5, None)                      # skip whisper j
            if j - 1 >= j0 and (j - 1) in dp[i]:
                v = dp[i][j - 1]
                if v[0] - 0.5 > best[0]:
                    best = (v[0] - 0.5, (i, j - 1))
            if i > 0:
                v = row_prev.get(j)
                if v and v[0] - 0.5 > best[0]:       # skip text i
                    best = (v[0] - 0.5, (i - 1, j))
                if j - 1 >= 0:
                    v = row_prev.get(j - 1)
                    if v is not None:
                        wj = wn[j]
                        sc = v[0]
                        if ti and wj:
                            if ti == wj:
                                sc += 3
                            elif len(ti) >= 3 and len(wj) >= 3 and (
                                    ti in wj or wj in ti):
                                sc += 2
                            else:
                                jt = ttri[i] & wtri[j]
                                if jt:
                                    jac = len(jt) / (len(ttri[i]) +
                                                     len(wtri[j]) - len(jt))
                                    if jac >= 0.34:
                                        sc += 1 + jac  # fuzzy anchor
                        else:
                            sc -= 1.5              # forced mismatch
                        if sc > best[0]:
                            best = (sc, (i - 1, j - 1))
            dp[i][j] = best

    # backtrack from the best cell in the last NON-EMPTY row
    bs, bi, bj = -1, -1, -1
    for i in range(n - 1, -1, -1):
        if dp[i]:
            for j, v in dp[i].items():
                if v[0] > bs:
                    bs, bi, bj = v[0], i, j
            break
    if bi < 0 or bs <= 0:
        return None
    anchors: dict[int, tuple[float, float]] = {}
    i, j = bi, bj
    while i >= 0 and j >= 0:
        cell = dp[i].get(j)
        if cell is None:
            break
        back = cell[1]
        if back is None:
            break
        pi, pj = back
        if pi == i - 1 and pj == j - 1:
            _, s, e = whisper_words[j]
            anchors[i] = (s, e)
        i, j = pi, pj

    if len(anchors) < max(2, n // 3):
        return None  # too few anchors -> let the app distribute evenly

    out: list[tuple[int, int]] = [(0, 0)] * n
    for i, (s, e) in anchors.items():
        out[i] = (int(s * 1000), int(e * 1000))
    # interpolate gaps between anchors, weighted by word length:
    # long words take longer to speak than short ones
    idxs = sorted(anchors)
    for k in range(len(idxs) - 1):
        a, b = idxs[k], idxs[k + 1]
        if b - a < 2:
            continue
        gap_slots = list(range(a + 1, b))
        weights = [max(1, len(twords[i])) for i in gap_slots]
        total = sum(weights)
        t0 = anchors[a][1] * 1000
        t1 = anchors[b][0] * 1000
        if t1 <= t0:
            continue
        acc = 0
        prev_end = t0
        for off, wl in enumerate(weights):
            i = gap_slots[off]
            s = t0 + (t1 - t0) * acc / total
            acc += wl
            e = t0 + (t1 - t0) * acc / total
            if e <= s:
                e = s + 1
            out[i] = (int(s), int(e))
            prev_end = e
    # heads/tails: stretch from the nearest anchor to 0 / file end
    first, last = idxs[0], idxs[-1]
    for i in range(first):
        t = (i + 1) / (first + 1)
        out[i] = (0, int(anchors[first][0] * 1000 * t) or 1)
    tail_end = int(whisper_words[-1][2] * 1000)
    span = max(last + 1, n - last)
    for i in range(last + 1, n):
        t = (i - last) / span
        out[i] = (int(anchors[last][1] * 1000 * (1 - t) + tail_end * t),
                  tail_end)
    # enforce monotonic, non-zero spans
    prev_end = 0
    for i in range(n):
        s, e = out[i]
        s = max(s, prev_end)
        if e <= s:
            e = s + 40
        out[i] = (s, e)
        prev_end = e
    return out


def transcript_text(wwords):
    """Transcript text + 1:1 word timings (text == what was spoken).

    Whisper words do not reliably carry leading whitespace, so join
    with single spaces after stripping: the token stream then equals
    the whisper word stream exactly, keeping the 1:1 index mapping."""
    toks = [w.strip() for w, _, _ in wwords]
    if not toks or not any(toks):
        return None, None
    text = " ".join(toks)
    assert len(tokenize(text)) == len(toks)
    times = [[int(s * 1000), int(e * 1000)] for _, s, e in wwords]
    return text, times


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("book_dir", type=Path)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--asr-backend", default="parakeet", choices=["whisper", "parakeet"],
                    help="word-timing ASR backend (parakeet: ~13x faster, native timestamps)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--language", default=None)
    ap.add_argument("--txt-dir", type=Path, default=None,
                    help="where chapter .txt files live (default: book dir)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing .words.json")
    ap.add_argument("--transcript", action="store_true",
                    help="use the whisper transcript as the reading "
                         "text (guarantees sync when the sidecar text "
                         "differs from what the TTS actually spoke)")
    args = ap.parse_args()

    book: Path = args.book_dir
    txt_dir: Path = args.txt_dir or book
    chapters = sorted(p for p in book.iterdir()
                      if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not chapters:
        print(f"no chapter audio found in {book}", file=sys.stderr)
        return 1

    if args.asr_backend == "parakeet":
        from audiobook_generator.asr import ParakeetModel
        pk_name = args.model if (args.model.startswith("nvidia/") or args.model.endswith(".nemo")) else "nvidia/parakeet-tdt-0.6b-v3"
        print(f"loading parakeet '{pk_name}' on {args.device} ...")
        model = ParakeetModel(pk_name, device=args.device)
    else:
        from faster_whisper import WhisperModel
        print(f"loading whisper '{args.model}' on {args.device} ...")
        model = WhisperModel(args.model, device=args.device,
                             compute_type=args.compute_type)

    ok = skipped = 0
    for audio in chapters:
        txt_path = txt_dir / (audio.stem + ".txt")
        out_path = audio.parent / (audio.stem + ".words.json")
        if out_path.exists() and not args.force:
            print(f"  {audio.name}: words.json exists, skipping (--force to redo)")
            skipped += 1
            continue
        if not txt_path.exists() and not args.transcript:
            print(f"  {audio.name}: no {txt_path.name}, skipping")
            skipped += 1
            continue

        t0 = time.time()
        segs, _ = model.transcribe(
            str(audio), word_timestamps=True, language=args.language,
            vad_filter=True)
        wwords: list[tuple[str, float, float]] = []
        for seg in segs:
            for w in seg.words or []:
                wwords.append((w.word, w.start, w.end))
        dt = time.time() - t0

        if args.transcript:
            text, times = transcript_text(wwords)
            if text is None:
                print(f"  {audio.name}: transcript unusable, skipping")
                skipped += 1
                continue
            txt_path.write_text(text, encoding="utf-8")
            out_path.write_text(json.dumps({"v": 1, "words": times}),
                                encoding="utf-8")
            ok += 1
            print(f"  {audio.name}: {len(times)} words [transcript] "
                  f"in {dt:.1f}s")
            continue

        text = strip_tts_markers(
            txt_path.read_text(encoding="utf-8", errors="replace"))
        toks = tokenize(text)
        if not toks:
            print(f"  {audio.name}: empty text, skipping")
            skipped += 1
            continue

        times = align(wwords, [text[o:l] for o, l in toks])
        if times is None:
            # sidecar text is untrustworthy (TTS engines paraphrase):
            # fall back to the transcript as the reading text
            text2, times2 = transcript_text(wwords)
            if text2 is not None:
                txt_path.write_text(text2, encoding="utf-8")
                out_path.write_text(json.dumps({"v": 1, "words": times2}),
                                    encoding="utf-8")
                ok += 1
                print(f"  {audio.name}: {len(times2)} words "
                      f"[transcript fallback] in {dt:.1f}s")
                continue
            print(f"  {audio.name}: weak alignment ({len(wwords)} whisper "
                  f"words vs {len(toks)} text words) - falling back to even")
            continue
        payload = {"v": 1, "words": [list(t) for t in times]}
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        ok += 1
        print(f"  {audio.name}: {len(toks)} words aligned "
              f"({len(wwords)} whisper words) in {dt:.1f}s")

    print(f"done: {ok} aligned, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
