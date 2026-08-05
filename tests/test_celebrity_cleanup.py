"""Tests for celebrity voice intermediate cleanup.

``build_celebrity_voice`` tries several YouTube videos per character, each
leaving working files prefixed ``{character}_v<idx>_`` (``*_segment<N>.wav``,
``*_s<N>_ref.wav``, ``*_celebrity_voice.wav``). Only the single winning
reference is kept — saved as ``{celebrity}.wav`` (traceability design, see
``save_celebrity_voice_as``) or ``{character}.wav`` (non-celebrity path); the
rest are orphaned and were accumulating unbounded (a few GB) in the output dir.
These tests pin the cleanup helper and verify it is wired into the real
voice-sample pipeline.
"""
from pathlib import Path

import pytest
from unittest.mock import patch

from audiobook_generator.celebrity_voices import cleanup_celebrity_intermediates
from audiobook_generator.config import DEFAULTS

STATIC_VOICE_TEXT = DEFAULTS["static_voice_text"]


class TestCleanupCelebrityIntermediates:
    def _make_char_files(self, d, char, versions=(1, 2)):
        """Create a realistic mix of a char's intermediate + final files."""
        d.mkdir(parents=True, exist_ok=True)
        for v in versions:
            for suffix in (f"segment0.wav", f"s0_ref.wav", f"celebrity_voice.wav"):
                (d / f"{char}_v{v}_{suffix}").write_bytes(b"\x00")
        (d / f"{char}.wav").write_bytes(b"\x00")
        return d

    def test_removes_all_per_video_intermediates_keeps_final(self, tmp_path):
        d = self._make_char_files(tmp_path, "jane")
        removed = cleanup_celebrity_intermediates("jane", str(tmp_path))
        leftovers = [p.name for p in tmp_path.iterdir()]
        assert removed == 6
        assert leftovers == ["jane.wav"]

    def test_respects_keep_path(self, tmp_path):
        d = self._make_char_files(tmp_path, "jane")
        keep = str(d / "jane_v1_s0_ref.wav")
        removed = cleanup_celebrity_intermediates("jane", str(tmp_path), keep_path=keep)
        leftovers = sorted(p.name for p in tmp_path.iterdir())
        assert removed == 5
        assert "jane_v1_s0_ref.wav" in leftovers
        assert "jane.wav" in leftovers

    def test_does_not_touch_other_characters(self, tmp_path):
        self._make_char_files(tmp_path, "jane")
        (tmp_path / "adan_v1_segment0.wav").write_bytes(b"\x00")
        (tmp_path / "adan.wav").write_bytes(b"\x00")
        cleanup_celebrity_intermediates("jane", str(tmp_path))
        assert (tmp_path / "adan_v1_segment0.wav").exists()
        assert (tmp_path / "adan.wav").exists()

    def test_prefix_does_not_clip_similar_names(self, tmp_path):
        # "jane_v*" must not match "jane_doe_v0_segment0.wav" or "jane2_v0".
        (tmp_path / "jane_v0_segment0.wav").write_bytes(b"\x00")
        (tmp_path / "jane_doe_v0_segment0.wav").write_bytes(b"\x00")
        (tmp_path / "jane2_v0_segment0.wav").write_bytes(b"\x00")
        cleanup_celebrity_intermediates("jane", str(tmp_path))
        assert not (tmp_path / "jane_v0_segment0.wav").exists()
        assert (tmp_path / "jane_doe_v0_segment0.wav").exists()
        assert (tmp_path / "jane2_v0_segment0.wav").exists()

    def test_returns_zero_when_nothing_to_clean(self, tmp_path):
        assert cleanup_celebrity_intermediates("nobody", str(tmp_path)) == 0

    def test_ignores_missing_files(self, tmp_path):
        assert cleanup_celebrity_intermediates("ghost", str(tmp_path)) == 0


class TestCleanupWiredIntoPipeline:
    def _run(self, engine, output_dir, descriptions, **kwargs):
        from audiobook_generator.generate_voice_samples import generate_voice_samples
        with patch("audiobook_generator.audiobook_generator.setup_validation_model", return_value=object()), \
             patch("audiobook_generator.utils.transcribe_audio_with_whisper",
                   return_value=(STATIC_VOICE_TEXT, [], [])), \
             patch("audiobook_generator.utils.crop_to_ref_text", return_value=False), \
             patch("audiobook_generator.voice_mapper.get_engine", return_value=engine):
            return generate_voice_samples(
                descriptions=descriptions,
                output_dir=str(output_dir),
                device="cpu",
                verbose=False,
                voice_engine="mock",
                engine=engine,
                **kwargs,
            )

    def test_generated_voice_cleans_up_pre_existing_intermediates(
        self, temp_dir, mock_tts_engine, sample_character_descriptions
    ):
        """Real pipeline: a character's *_v*_* junk is removed once {char}.wav
        is finalized, while other characters' files are left alone."""
        # Pre-seed the orphaned per-video files the celebrity path would leave.
        for v in (0, 1):
            (temp_dir / f"jane_v{v}_segment0.wav").write_bytes(b"\x00")
            (temp_dir / f"jane_v{v}_s0_ref.wav").write_bytes(b"\x00")
            (temp_dir / f"jane_v{v}_celebrity_voice.wav").write_bytes(b"\x00")
        (temp_dir / "other_v0_segment0.wav").write_bytes(b"\x00")

        status, voices = self._run(
            mock_tts_engine, temp_dir, sample_character_descriptions
        )

        # jane.wav finalized; its *_v*_* intermediates removed.
        assert "jane" in voices
        assert (temp_dir / "jane.wav").exists()
        assert list(temp_dir.glob("jane_v*")) == []
        # Unrelated character's junk untouched.
        assert (temp_dir / "other_v0_segment0.wav").exists()

    def test_cleanup_invoked_for_each_generated_character(self, temp_dir, mock_tts_engine, sample_character_descriptions):
        """The pipeline calls cleanup_celebrity_intermediates for generated chars."""
        with patch("audiobook_generator.celebrity_voices.cleanup_celebrity_intermediates") as mock_cleanup:
            self._run(mock_tts_engine, temp_dir, sample_character_descriptions)
        # narrator/jane/elizabeth should each trigger a cleanup call.
        assert mock_cleanup.call_count == 3
        args = [c.args[0] for c in mock_cleanup.call_args_list]
        assert set(args) == {"narrator", "jane", "elizabeth"}

    def test_celebrity_path_cleans_intermediates_keeps_celebrity_ref(
        self, temp_dir, mock_llm_client, mock_tts_engine, sample_character_descriptions
    ):
        """Celebrity path: *_v*_* junk removed while the celebrity-named final
        ref (``{celebrity}.wav``) is preserved — the file voices_map points to."""
        from audiobook_generator.testing import write_silence_wav

        # Pre-seed orphaned per-video files for jane, plus junk for another char.
        for v in (0, 1):
            (temp_dir / f"jane_v{v}_segment0.wav").write_bytes(b"\x00")
            (temp_dir / f"jane_v{v}_s0_ref.wav").write_bytes(b"\x00")
            (temp_dir / f"jane_v{v}_celebrity_voice.wav").write_bytes(b"\x00")
        (temp_dir / "elizabeth_v0_segment0.wav").write_bytes(b"\x00")

        celebrities = {
            "narrator": "morgan_freeman",
            "jane": "emma_watson",
            "elizabeth": "natalie_portman",
        }

        def fake_build(client, model, character, description, output_dir,
                       pre_matched_celebrity=None, whisper_model=None,
                       tts_engine=None, verbose=False, **kwargs):
            # build_celebrity_voice receives "<char>.sampleN"; base is the char.
            base = character.split(".")[0]
            celeb = celebrities[base]
            ref = str(Path(output_dir) / f"{celeb}_ref.wav")
            if not Path(ref).exists():
                write_silence_wav(ref, 22050, 1)
            return ref, {
                "character": base,
                "celebrity": celeb,
                "reason": "test",
                "search_query": f"{celeb} interview",
                "segment": ref,
                "audio_source": None,
            }

        with patch("audiobook_generator.celebrity_voices.build_celebrity_voice",
                   side_effect=fake_build):
            self._run(
                mock_tts_engine, temp_dir, sample_character_descriptions,
                use_celebrity_voices=True,
                llm_client=mock_llm_client,
            )

        # jane's *_v*_* intermediates gone; her celebrity-named ref preserved.
        assert list(temp_dir.glob("jane_v*")) == []
        assert (temp_dir / "emma_watson_ref.wav").exists()
        # Unrelated character's junk is untouched (elizabeth's own was cleaned).
        assert list(temp_dir.glob("elizabeth_v*")) == []
