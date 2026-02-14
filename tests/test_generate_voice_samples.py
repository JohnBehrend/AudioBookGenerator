"""Tests for generate_voice_samples.py (Stage 5: Voice Sample Generation)."""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from generate_voice_samples import (
    load_character_descriptions,
    create_sample_text_from_description
)


class TestLoadCharacterDescriptions:
    """Tests for load_character_descriptions function."""

    def test_load_valid_descriptions(self, temp_dir):
        """Test loading valid character descriptions."""
        descriptions_file = temp_dir / 'characters_descriptions.json'
        descriptions = {
            'narrator': 'Male, middle-aged, calm voice.',
            'john': 'Male, young, enthusiastic voice.'
        }
        descriptions_file.write_text(json.dumps(descriptions))

        result = load_character_descriptions(str(descriptions_file))

        assert result == descriptions

    def test_load_empty_descriptions(self, temp_dir):
        """Test loading empty descriptions."""
        descriptions_file = temp_dir / 'characters_descriptions.json'
        descriptions_file.write_text(json.dumps({}))

        result = load_character_descriptions(str(descriptions_file))

        assert result == {}


class TestCreateSampleTextFromDescription:
    """Tests for create_sample_text_from_description function."""

    def test_basic_description(self):
        """Test basic description processing."""
        description = 'Male, young adult, friendly voice.'
        result = create_sample_text_from_description(description)

        assert result == 'Male, young adult, friendly voice'

    def test_description_with_pipes(self):
        """Test description with pipe separators."""
        description = 'Male | Young | Friendly'
        result = create_sample_text_from_description(description)

        assert '|' not in result
        assert ',' in result

    def test_description_truncation(self):
        """Test that long descriptions are truncated."""
        long_desc = 'A' * 200
        result = create_sample_text_from_description(long_desc)

        assert len(result) <= 150

    def test_empty_description(self):
        """Test empty description."""
        result = create_sample_text_from_description('')

        assert result == ''


class TestIntegration:
    """Integration tests for voice sample generation pipeline."""

    def test_full_generation_workflow(self, temp_dir):
        """Test the full generation workflow."""
        descriptions = {
            'narrator': 'Male, middle-aged, calm narrator voice.',
            'john': 'Male, young adult, energetic hero voice.',
            'mary': 'Female, mid-30s, authoritative villain voice.'
        }

        descriptions_file = temp_dir / 'characters_descriptions.json'
        descriptions_file.write_text(json.dumps(descriptions))

        loaded = load_character_descriptions(str(descriptions_file))
        assert len(loaded) == 3


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
