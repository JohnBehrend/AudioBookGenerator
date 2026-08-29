"""Tests for scripts/build_wot_seed_map.py.

Guards the regression where audiobook chapter MP3 files from prior books
(e.g. ``chapter_00.mp3`` in teotw/) were mistakenly ingested as "character
voices". Those bogus ``chapter_XX`` entries pointed the pipeline's seed loader
at a prior book's *full chapter* recordings, which it then copied into the
current output dir — overwriting the real chapter MP3s and making stage 5 skip
them. The result was an audiobook silently filled with the wrong book's audio.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from audiobook_generator.testing import write_silence_wav

_SEED_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_wot_seed_map.py"


@pytest.fixture(scope="module")
def seed_module():
    """Load build_wot_seed_map.py via importlib (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("build_wot_seed_map", _SEED_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_book_dir(tmp_path, name, wavs=(), mp3s=()):
    """Build a synthetic prior-book dir with given character wav names and
    chapter mp3 names (chapter mp3s need only minimal bytes to exist)."""
    d = tmp_path / name
    d.mkdir()
    for w in wavs:
        write_silence_wav(d / f"{w}.wav", duration=0.1)
    for m in mp3s:
        (d / f"{m}.mp3").write_bytes(b"\xff\xfb\x90\x64")
    return d


def _write_voices_map(book_dir, mapping):
    """Write a voices_map.json (paths relative to the book dir)."""
    (book_dir / "voices_map.json").write_text(json.dumps(mapping, indent=2))


def _run_main(seed_module, tmp_path, books):
    """Invoke seed_module.main() with argv pointing at the given book dirs."""
    out = tmp_path / "seed_voices_map.json"
    old = sys.argv
    sys.argv = ["build_wot_seed_map", "--out", str(out), "--priority", *[str(b) for b in books]]
    try:
        seed_module.main()
    finally:
        sys.argv = old
    return out


class TestPlainVoices:
    def test_includes_character_and_narrator_wavs(self, seed_module, tmp_path):
        d = _make_book_dir(tmp_path, "book", wavs=("rand", "narrator"))
        assert set(seed_module.plain_voices(d)) == {"rand", "narrator"}

    def test_excludes_sample_variant_files(self, seed_module, tmp_path):
        d = _make_book_dir(tmp_path, "book", wavs=("rand", "rand.sample1"))
        voices = seed_module.plain_voices(d)
        assert "rand" in voices
        assert "rand.sample1" not in voices

    def test_excludes_chapter_mp3s_regression(self, seed_module, tmp_path):
        """Regression: a prior book's chapter_XX.mp3 must NOT become a voice."""
        d = _make_book_dir(tmp_path, "book", wavs=("rand",), mp3s=("chapter_00", "chapter_53"))
        voices = seed_module.plain_voices(d)
        assert "rand" in voices
        assert not any(k.startswith("chapter_") for k in voices)

    def test_excludes_non_audio_files(self, seed_module, tmp_path):
        d = _make_book_dir(tmp_path, "book", wavs=("rand",))
        (d / "chapter_00.txt").write_text("not audio")
        (d / "notes.json").write_text("{}")
        voices = seed_module.plain_voices(d)
        assert "rand" in voices
        assert not any(k.startswith("chapter_") for k in voices)


class TestBookVoices:
    """Tests for book_voices: prefers a clean voices_map.json, else globs."""

    def test_uses_clean_voices_map_over_glob(self, seed_module, tmp_path):
        """A clean voices_map.json (all entries exist) is authoritative, even when
        the dir also contains leftover .cropped.wav artifacts that globbing
        would wrongly pick up."""
        d = _make_book_dir(tmp_path, "book", wavs=("rand", "narrator"))
        # Leftover artifacts globbing would mis-seed as bogus characters.
        (d / "rand.wav.cropped.wav").write_bytes(b"\x00")
        (d / "narrator.wav.cropped.wav").write_bytes(b"\x00")
        _write_voices_map(d, {"rand": "rand.wav", "narrator": "narrator.wav"})

        voices = seed_module.book_voices(d)
        assert set(voices) == {"rand", "narrator"}
        assert not any("cropped" in v for v in voices.values())

    def test_falls_back_to_glob_when_voices_map_stale(self, seed_module, tmp_path):
        """A stale voices_map.json pointing at deleted .sampleN files is ignored,
        and book_voices globs the real character wavs instead."""
        d = _make_book_dir(tmp_path, "book", wavs=("rand", "moiraine"))
        _write_voices_map(d, {"rand": "rand.sample1.wav", "moiraine": "moiraine.sample1.wav"})

        voices = seed_module.book_voices(d)
        # Stale map ignored -> globbing yields the real character voices.
        assert set(voices) == {"rand", "moiraine"}
        assert all("sample" not in v for v in voices.values())

    def test_falls_back_to_glob_when_voices_map_missing(self, seed_module, tmp_path):
        d = _make_book_dir(tmp_path, "book", wavs=("rand",))
        voices = seed_module.book_voices(d)
        assert set(voices) == {"rand"}


class TestBuildSeedMap:
    def test_no_chapter_keys_and_no_mp3_values(self, seed_module, tmp_path):
        prior = _make_book_dir(
            tmp_path, "teotw",
            wavs=("rand", "narrator"),
            mp3s=("chapter_00", "chapter_01", "chapter_53"),
        )
        out = _run_main(seed_module, tmp_path, [prior])
        m = json.loads(out.read_text())
        assert not any(k.startswith("chapter_") for k in m)
        assert not any(str(v).endswith(".mp3") for v in m.values())
        assert "narrator" in m
        assert "rand" in m

    def test_priority_merge_first_book_wins(self, seed_module, tmp_path):
        hi = _make_book_dir(tmp_path, "hi", wavs=("rand",))
        lo = _make_book_dir(tmp_path, "lo", wavs=("rand",))
        out = _run_main(seed_module, tmp_path, [hi, lo])
        m = json.loads(out.read_text())
        assert str(m["rand"]).startswith(str(hi.resolve()))


class TestLoaderInvariant:
    def test_built_map_cannot_collide_with_chapter_outputs(self, seed_module, tmp_path):
        """Defense-in-depth: the seed loader copies basename(voice_path) into the
        output dir. If any seed entry were a chapter mp3, the loader would plant
        chapter_XX.mp3 and falsely satisfy stage 5's skip check. Assert the built
        map can never do that."""
        prior = _make_book_dir(
            tmp_path, "book",
            wavs=("narrator", "rand"),
            mp3s=("chapter_00",),
        )
        out = _run_main(seed_module, tmp_path, [prior])
        m = json.loads(out.read_text())
        for v in m.values():
            base = Path(v).name
            assert not base.startswith("chapter_")
            assert not base.endswith(".mp3")



