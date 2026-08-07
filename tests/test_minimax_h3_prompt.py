"""Unit tests for the MiniMax H3 prompt structure (no GPU / ComfyUI needed).

These verify ``build_prompt`` produces a prompt that matches H3's official
T2VA structure:

  * the three field headers in order
      (``integrated_multimodal_description`` / ``overall_soundscape`` /
       ``non_diegetic_music``)
  * a ``[Shot N]`` marker
  * dialogue wrapped in ``<d>[English] ...</d>``
  * ``<d>`` holds ONLY the language tag + verbatim spoken content (every word
    and punctuation mark preserved, quotes intact)
  * emotion/delivery lives in the prose OUTSIDE ``<d>``
  * a speaker-identity phrase ``(S1)`` precedes the dialogue
"""

import importlib.util
from pathlib import Path

import pytest

_MAIN = Path(__file__).resolve().parents[1] / "engines" / "minimax_h3" / "main.py"


def _load():
    spec = importlib.util.spec_from_file_location("mh3_prompt", _MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def h3():
    return _load()


RAND_DESC = (
    '{"gender": "male", "age": "young adult", "pitch": "low", '
    '"accent": "two rivers rural", "style": ["earnest", "determined", "weary"], '
    '"pace": "measured, deliberate", '
    '"emotion_arc": ["warm and earnest", "rising, defiant"], '
    '"description": "a tall, lean young man with distinctive copper-red hair and grey eyes, '
    'wearing simple grey Two Rivers farm clothing."}'
)

SPOKEN = (
    "Hello there. Good morning everyone. After all these years, it's finally "
    'here for us! "We\'re going to make this work, no matter what."'
)


def test_three_field_headers_in_order(h3):
    prompt = h3.build_prompt(RAND_DESC, SPOKEN)
    md = prompt.index("integrated_multimodal_description:")
    ss = prompt.index("overall_soundscape:")
    nm = prompt.index("non_diegetic_music:")
    assert md < ss < nm, "field headers out of order"


def test_shot_marker_and_speaker_identity(h3):
    prompt = h3.build_prompt(RAND_DESC, SPOKEN)
    assert "[Shot 1]" in prompt
    assert "(S1)" in prompt
    # Speaker identity must come before the dialogue tag.
    assert prompt.index("(S1)") < prompt.index("<d>")


def test_dialogue_tag_holds_language_and_verbatim_speech(h3):
    prompt = h3.build_prompt(RAND_DESC, SPOKEN)
    inner = prompt.split("<d>", 1)[1].split("</d>", 1)[0]
    # Language tag + verbatim spoken content.
    assert inner.startswith("[English] ")
    verbatim = inner[len("[English] ") :]
    assert verbatim == SPOKEN, "spoken text not preserved verbatim (words/punctuation/quotes)"
    # Quotes must NOT be stripped.
    assert '"' in verbatim


def test_delivery_is_outside_dialogue_tag(h3):
    prompt = h3.build_prompt(RAND_DESC, SPOKEN)
    inner = prompt.split("<d>", 1)[1].split("</d>", 1)[0]
    # Delivery/emotion words from the emotion_arc must appear in the prose
    # BEFORE <d>, never inside the <d> block.
    for word in ("warm and earnest", "rising, defiant"):
        assert word in prompt
        assert word not in inner, f"emotion/delivery leaked inside <d>: {word!r}"


def test_celebrity_voice_anchored_in_speaker_phrase(h3):
    desc = RAND_DESC.replace(
        '"style": ["earnest", "determined", "weary"]',
        '"style": ["earnest", "determined", "weary"], "celebrity_voice": "Tom Hanks"',
    )
    prompt = h3.build_prompt(desc, SPOKEN)
    # Celebrity is a textual anchor in the speaker identity, OUTSIDE <d>.
    assert "in the style of Tom Hanks" in prompt
    inner = prompt.split("<d>", 1)[1].split("</d>", 1)[0]
    assert "Tom Hanks" not in inner


def test_subject_flows_into_shot_sentence(h3):
    prompt = h3.build_prompt(RAND_DESC, SPOKEN)
    # The free-text appearance must flow directly into "facing the camera"
    # with no double-period mid-sentence.
    assert "clothing facing the camera" in prompt
    assert "clothing. facing" not in prompt
