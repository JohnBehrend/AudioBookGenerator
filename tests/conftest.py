"""pytest fixtures for the audiobook pipeline tests."""
import pytest
import tempfile
import os
import json
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_chapter_text():
    """Sample chapter text for testing."""
    return """This is the beginning of the chapter. The narrator speaks here.
"Hello there," said John.
"I am fine," replied Mary.
The story continues with more narrative text.
"Are you coming to the party?" John asked.
"Yes, I will be there," Mary said.
End of chapter."""


@pytest.fixture
def sample_chapter_with_quoted_lines():
    """Sample chapter with clearly quoted lines for speaker attribution."""
    return """Chapter One

The morning sun shone brightly.

"Good morning, everyone!" shouted the captain.
"Is everyone ready?" he asked.

"Yes, sir!" came the reply.
"We are all set," added the first mate.

The ship set sail immediately."""


@pytest.fixture
def sample_character_map():
    """Sample character map for testing."""
    return {
        "1": "narrator",
        "2": "john",
        "3": "mary"
    }


@pytest.fixture
def sample_line_map():
    """Sample line map for testing."""
    return {
        "2": 2,
        "4": 3,
        "6": 2,
        "8": 3
    }


@pytest.fixture
def sample_map_json(sample_character_map, sample_line_map):
    """Sample map JSON content."""
    return json.dumps([sample_character_map, sample_line_map])


@pytest.fixture
def sample_characters():
    """Sample characters list."""
    return ["narrator", "john", "mary", "susan"]


@pytest.fixture
def sample_characters_json(sample_characters):
    """Sample characters JSON content."""
    return json.dumps({"characters": sample_characters})


@pytest.fixture
def sample_character_descriptions():
    """Sample character descriptions."""
    return {
        "narrator": "Male, middle-aged, neutral accent, calm speaking style.",
        "john": "Male, young adult, British accent, enthusiastic.",
        "mary": "Female, mid-30s, Southern accent, slow speaking."
    }


@pytest.fixture
def sample_character_descriptions_json(sample_character_descriptions):
    """Sample character descriptions JSON content."""
    return json.dumps(sample_character_descriptions, indent=2)


@pytest.fixture
def chapter_files(temp_dir, sample_map_json):
    """Create sample chapter map files."""
    chapters = []
    for i in range(3):
        chapter_file = temp_dir / f"chapter_{i}.map.json"
        chapter_file.write_text(sample_map_json)
        chapters.append(chapter_file)
    return chapters


@pytest.fixture
def chapter_texts_dir(temp_dir, sample_chapter_text):
    """Create sample chapter text files."""
    chapters_dir = temp_dir / "chapters"
    chapters_dir.mkdir()

    for i in range(3):
        chapter_file = chapters_dir / f"chapter_{i}.txt"
        chapter_file.write_text(sample_chapter_text)

    return chapters_dir