"""Tests for find_missing_voices() in audiobook_generator.py.

A character counts as voiced if it has a ``{char}.wav`` in the output dir OR its
voice_map entry resolves to an existing audio file. The latter handles celebrity
voices stored under the celebrity name (``{celebrity}_ref.wav``) via the
traceability feature rather than ``{char}.wav``.
"""

from audiobook_generator.audiobook_generator import find_missing_voices


def test_char_with_wav_not_missing(temp_dir):
    (temp_dir / "rand.wav").write_bytes(b"x")
    missing = find_missing_voices(
        ["rand", "mat"], {"rand": "rand.wav"}, str(temp_dir)
    )
    assert missing == {"mat"}


def test_char_with_celebrity_ref_not_missing(temp_dir):
    (temp_dir / "sally_field_ref.wav").write_bytes(b"x")
    # 'amellia arene' has no {char}.wav, but its voice_map points to the ref file.
    missing = find_missing_voices(
        ["amellia arene", "thad"],
        {"amellia arene": "sally_field_ref.wav"},
        str(temp_dir),
    )
    assert missing == {"thad"}


def test_char_with_missing_mapped_file_is_missing(temp_dir):
    # voice_map points to a file that does not exist on disk -> still missing.
    missing = find_missing_voices(
        ["nicola"], {"nicola": "does_not_exist.wav"}, str(temp_dir)
    )
    assert missing == {"nicola"}


def test_char_with_no_voice_map_entry_is_missing(temp_dir):
    (temp_dir / "rand.wav").write_bytes(b"x")
    missing = find_missing_voices(
        ["rand", "unknown"], {"rand": "rand.wav"}, str(temp_dir)
    )
    assert missing == {"unknown"}


def test_all_voiced_returns_empty(temp_dir):
    (temp_dir / "rand.wav").write_bytes(b"x")
    (temp_dir / "ciarn_hinds_ref.wav").write_bytes(b"x")
    missing = find_missing_voices(
        ["rand", "unknown"],
        {"rand": "rand.wav", "unknown": "ciarn_hinds_ref.wav"},
        str(temp_dir),
    )
    assert missing == set()
