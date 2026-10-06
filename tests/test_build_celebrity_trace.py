"""Tests for scripts/build_celebrity_trace.py traceability helper."""
import json

from scripts.build_celebrity_trace import (
    _celebrity_for,
    character_celebrity_map,
    filename_celebrity_map,
    _norm,
)


class TestCelebrityFor:
    def test_parses_string_json(self):
        assert _celebrity_for('{"celebrity_voice": "Chris Evans"}') == "Chris Evans"

    def test_parses_dict(self):
        assert _celebrity_for({"celebrity_voice": "Chris Evans"}) == "Chris Evans"

    def test_handles_empty_and_none(self):
        assert _celebrity_for({"celebrity_voice": ""}) == ""
        assert _celebrity_for({"celebrity_voice": None}) == ""
        assert _celebrity_for({"foo": "bar"}) == ""
        assert _celebrity_for("") == ""

    def test_strips_whitespace(self):
        assert _celebrity_for({"celebrity_voice": "  Chris Evans  "}) == "Chris Evans"


class TestNorm:
    def test_drops_spaces_underscores_case(self):
        assert _norm("Amellia Arene") == "amelliaarene"
        assert _norm("tal_nethin") == "talnethin"
        assert _norm("HERO") == "hero"


class TestCharacterCelebrityMap:
    def test_current_book_wins_over_prior(self, tmp_path):
        cur = tmp_path / "book5"
        cur.mkdir()
        (cur / "characters_descriptions.json").write_text(json.dumps({
            "hero": {"celebrity_voice": "Keanu Reeves"},
            "mentor": {"celebrity_voice": "Claire Foy"},
            "scout": {"celebrity_voice": ""},
        }))
        prior = tmp_path / "book4"
        prior.mkdir()
        (prior / "characters_descriptions.json").write_text(json.dumps({
            "hero": {"celebrity_voice": "Chris Evans"},
            "scout": {"celebrity_voice": "Jeffrey Dean Morgan"},
            "lan": {"celebrity_voice": "Michael Caine"},
        }))

        mapping = character_celebrity_map([str(cur), str(prior)])
        assert mapping["hero"] == "Keanu Reeves"  # current wins
        assert mapping["mentor"] == "Claire Foy"
        assert mapping["scout"] == "Jeffrey Dean Morgan"  # dug from prior
        assert mapping["lan"] == "Michael Caine"
        assert "scout" in mapping

    def test_prior_most_recent_first(self, tmp_path):
        b4 = tmp_path / "b4"; b4.mkdir()
        b3 = tmp_path / "b3"; b3.mkdir()
        (b4 / "characters_descriptions.json").write_text(json.dumps(
            {"hero": {"celebrity_voice": "Chris Evans"}}))
        (b3 / "characters_descriptions.json").write_text(json.dumps(
            {"hero": {"celebrity_voice": "Richard Armitage"}}))
        # Caller passes most-recent-first; first book that records a celebrity wins
        mapping = character_celebrity_map([str(b4), str(b3)])
        assert mapping["hero"] == "Chris Evans"

    def test_missing_prior_dir_is_skipped(self, tmp_path):
        cur = tmp_path / "book"; cur.mkdir()
        (cur / "characters_descriptions.json").write_text(json.dumps(
            {"hero": {"celebrity_voice": "Chris Evans"}}))
        mapping = character_celebrity_map([str(cur), str(tmp_path / "nope")])
        assert mapping == {"hero": "Chris Evans"}

    def test_no_celebrity_returns_empty(self, tmp_path):
        cur = tmp_path / "book"; cur.mkdir()
        (cur / "characters_descriptions.json").write_text(json.dumps(
            {"lan": {"celebrity_voice": ""}}))
        assert character_celebrity_map([str(cur)]) == {}


class TestFilenameCelebrityMap:
    def test_celebrity_named_file_identified(self, tmp_path):
        cur = tmp_path / "book"; cur.mkdir()
        (cur / "voices_map.json").write_text(json.dumps({
            "amellia arene": "sally_field.wav",
            "hero": "hero.wav",
            "nicola": "hailee_steinfeld.wav",
        }))
        mapping = filename_celebrity_map(str(cur), known=set())
        assert mapping["amellia arene"] == "sally_field"
        assert mapping["nicola"] == "hailee_steinfeld"
        # {char}.wav (seeded) is not treated as celebrity-named
        assert "hero" not in mapping

    def test_known_chars_not_overwritten(self, tmp_path):
        cur = tmp_path / "book"; cur.mkdir()
        (cur / "voices_map.json").write_text(json.dumps({
            "hero": "chris_evans.wav",
        }))
        mapping = filename_celebrity_map(str(cur), known={"hero": "Chris Evans"})
        assert mapping == {}  # hero already known

    def test_missing_voices_map_returns_empty(self, tmp_path):
        assert filename_celebrity_map(str(tmp_path), known=set()) == {}
