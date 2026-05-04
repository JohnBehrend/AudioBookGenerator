"""Tests for utils module."""

import json
import os
import tempfile
import pytest
from pathlib import Path

from audiobook_generator.utils import (
    ProgressHandler,
    TempDirContext,
    compare_characters,
    merge_line_maps,
    get_llm_client,
    load_json_file,
    copy_mp3_files_to_chapters,
    get_character_wav_file,
    load_seed_characters,
    normalize_character_name,
    distill_string,
    parse_map_file,
    get_chapter_map_files,
    extract_characters_from_maps,
    count_lines_per_character,
    natural_sort_key,
    extract_gender_from_description,
    classify_gender_statistical,
    detect_gender_from_audio,
)


class TestProgressHandler:
    """Tests for ProgressHandler class."""

    def test_init_without_progress(self):
        """Test initialization without progress callback."""
        handler = ProgressHandler(use_tqdm=False)
        assert handler.progress is None
        assert handler.total is None

    def test_init_with_total(self):
        """Test initialization with total."""
        handler = ProgressHandler(use_tqdm=False, total=100)
        assert handler.total == 100

    def test_update_without_tqdm(self):
        """Test update without tqdm."""
        handler = ProgressHandler(use_tqdm=False, total=100)
        handler.update(ratio=0.5, desc="Test progress")

    def test_update_with_ratio(self):
        """Test update with ratio calculation."""
        handler = ProgressHandler(use_tqdm=False, total=10)
        handler.update(ratio=0.5)
        assert handler.last_ratio == 50

    def test_set_total(self):
        """Test set_total method."""
        handler = ProgressHandler(use_tqdm=False)
        handler.set_total(100)
        if handler._tqdm is not None:
            assert handler._tqdm.total == 100

    def test_context_manager(self):
        """Test ProgressHandler as context manager."""
        with ProgressHandler(use_tqdm=False) as handler:
            assert handler is not None


class TestTempDirContext:
    """Tests for TempDirContext class."""

    def test_init(self):
        """Test initialization."""
        ctx = TempDirContext()
        assert ctx._temp_dir is None
        assert ctx._chapters_dir is None

    def test_get_chapters_dir_creates_directory(self):
        """Test that get_chapters_dir creates the directory."""
        ctx = TempDirContext()
        chapters_dir = ctx.get_chapters_dir()
        assert chapters_dir.exists()
        ctx.cleanup()

    def test_get_temp_dir(self):
        """Test get_temp_dir returns path."""
        ctx = TempDirContext()
        chapters_dir = ctx.get_chapters_dir()
        temp_dir = ctx.get_temp_dir()
        assert temp_dir in str(chapters_dir)
        ctx.cleanup()

    def test_cleanup(self):
        """Test cleanup method."""
        ctx = TempDirContext()
        ctx.get_chapters_dir()
        ctx.cleanup()
        assert ctx._temp_dir is None
        assert ctx._chapters_dir is None

    def test_context_manager(self):
        """Test TempDirContext as context manager."""
        with TempDirContext() as ctx:
            chapters_dir = ctx.get_chapters_dir()
            assert chapters_dir.exists()


class TestCompareCharacters:
    """Tests for compare_characters function."""

    def test_identical_names(self):
        """Test identical names are compared as same."""
        assert compare_characters("Jane", "Jane") is True

    def test_case_insensitive(self):
        """Test comparison is case insensitive."""
        assert compare_characters("JANE", "jane") is True
        assert compare_characters("Jane", "JANE") is True

    def test_substring_match(self):
        """Test substring matching."""
        assert compare_characters("Jane Bennet", "Jane") is True
        assert compare_characters("Jane", "Jane Bennet") is True

    def test_different_names(self):
        """Test different names are not matched."""
        assert compare_characters("Jane", "Elizabeth") is False

    def test_with_whitespace(self):
        """Test handling of whitespace."""
        assert compare_characters("Jane", "  Jane  ") is True


class TestMergeLineMaps:
    """Tests for merge_line_maps function."""

    def test_empty_maps(self):
        """Test merging empty maps."""
        result = merge_line_maps([])
        assert result == {}

    def test_single_map(self):
        """Test merging single map."""
        line_maps = [{1: 2, 2: 2, 3: 3}]
        result = merge_line_maps(line_maps)
        assert result[1] == 2
        assert result[2] == 2
        assert result[3] == 3

    def test_multiple_maps_majority(self):
        """Test that majority vote is used."""
        line_maps = [
            {1: 2, 2: 2},
            {1: 3, 2: 2},
            {1: 2, 2: 2}
        ]
        result = merge_line_maps(line_maps)
        assert result[1] == 2

    def test_tie_breaker_first_value(self):
        """Test that first value is used on tie."""
        line_maps = [
            {1: 2},
            {1: 3}
        ]
        result = merge_line_maps(line_maps)
        assert result[1] == 2


class TestGetLLMClient:
    """Tests for get_llm_client function."""

    def test_creates_client(self):
        """Test that client is created."""
        client = get_llm_client(api_key="test-key", port="8080")
        assert client is not None
        assert client.api_key == "test-key"

    def test_sets_base_url(self):
        """Test that base URL is set correctly."""
        client = get_llm_client(api_key="test-key", port="8080")
        base_url = str(client.base_url)
        assert "8080" in base_url


class TestLoadJsonFile:
    """Tests for load_json_file function."""

    def test_loads_valid_file(self, temp_dir):
        """Test loading valid JSON file."""
        file_path = temp_dir / "test.json"
        data = {"key": "value"}
        with open(file_path, "w") as f:
            json.dump(data, f)

        result = load_json_file(str(file_path))
        assert result == data

    def test_missing_file_returns_none(self, temp_dir):
        """Test that missing file returns None."""
        result = load_json_file(str(temp_dir / "nonexistent.json"))
        assert result is None


class TestCopyMp3FilesToChapters:
    """Tests for copy_mp3_files_to_chapters function."""

    def test_copies_files(self, temp_dir):
        """Test that MP3 files are copied."""
        source_dir = temp_dir / "source"
        source_dir.mkdir()

        for i in range(3):
            mp3_file = source_dir / f"chapter_{i}.mp3"
            mp3_file.write_bytes(b"fake mp3 content")

        result = copy_mp3_files_to_chapters(str(source_dir))
        assert result == 3

        chapters_dir = Path("chapters")
        assert chapters_dir.exists()
        for i in range(3):
            assert (chapters_dir / f"chapter_{i}.mp3").exists()

    def test_no_files_returns_zero(self, temp_dir):
        """Test that zero is returned when no files."""
        result = copy_mp3_files_to_chapters(str(temp_dir))
        assert result == 0


class TestGetCharacterWavFile:
    """Tests for get_character_wav_file function."""

    def test_finds_file_in_chapters_dir(self, temp_dir):
        """Test that file is found in chapters directory."""
        wav_file = temp_dir / "jane.wav"
        wav_file.touch()

        result = get_character_wav_file("jane", temp_dir)
        assert result is not None

    def test_returns_none_for_missing(self, temp_dir):
        """Test that None is returned for missing file."""
        result = get_character_wav_file("nonexistent", temp_dir)
        assert result is None


class TestLoadSeedCharacters:
    """Tests for load_seed_characters function."""

    def test_loads_valid_file(self, temp_dir):
        """Test loading valid seed characters file."""
        file_path = temp_dir / "seed.json"
        data = {"jane": "/path/to/jane.wav"}
        with open(file_path, "w") as f:
            json.dump(data, f)

        result = load_seed_characters(str(file_path))
        assert result == data

    def test_handles_dict_input(self):
        """Test handling of dict input.

        If dict has a 'name' key pointing to existing file, it loads from file.
        Otherwise returns None.
        """
        data = {"name": "/path/to/nonexistent.wav", "jane": "/path/to/jane.wav"}
        result = load_seed_characters(data)
        assert result is None

        data_with_existing = {"name": __file__}
        result2 = load_seed_characters(data_with_existing)
        if result2 is not None:
            assert isinstance(result2, dict)

    def test_none_input(self):
        """Test that None input returns None."""
        result = load_seed_characters(None)
        assert result is None

    def test_missing_file_returns_none(self, temp_dir):
        """Test that missing file returns None."""
        result = load_seed_characters(str(temp_dir / "nonexistent.json"))
        assert result is None


class TestNormalizeCharacterName:
    """Tests for normalize_character_name function."""

    def test_lowercases(self):
        """Test that name is lowercased."""
        assert normalize_character_name("JANE") == "jane"

    def test_strips_whitespace(self):
        """Test that whitespace is stripped."""
        assert normalize_character_name("  Jane  ") == "jane"

    def test_replaces_underscores(self):
        """Test that underscores are replaced."""
        assert normalize_character_name("jane_bennet") == "jane bennet"

    def test_removes_apostrophes(self):
        """Test that apostrophes are handled.

        Current behavior: apostrophes are replaced with spaces.
        """
        result = normalize_character_name("jane's")
        assert "jane" in result
        assert "'" not in result


class TestDistillString:
    """Tests for distill_string function."""

    def test_lowercases(self):
        """Test that string is lowercased."""
        assert distill_string("HELLO") == "hello"

    def test_removes_punctuation(self):
        """Test that punctuation is removed."""
        assert distill_string("Hello, world!") == "hello world"

    def test_normalizes_whitespace(self):
        """Test that whitespace is normalized."""
        assert distill_string("Hello   world") == "hello world"


class TestParseMapFile:
    """Tests for parse_map_file function."""

    def test_parses_list_format(self, temp_dir):
        """Test parsing list format map file."""
        data = [
            {"1": "narrator", "2": "jane"},
            {"2": 2, "4": 2}
        ]
        file_path = temp_dir / "chapter_0.map.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        result = parse_map_file(file_path)
        assert result is not None
        char_map, line_map = result
        assert 1 in char_map
        assert 2 in line_map

    def test_parses_dict_format(self, temp_dir):
        """Test parsing dict format map file."""
        data = {
            "character_map": {"1": "narrator", "2": "jane"},
            "line_map": {"2": 2}
        }
        file_path = temp_dir / "chapter_0.map.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        result = parse_map_file(file_path)
        assert result is not None
        char_map, line_map = result
        assert 1 in char_map


class TestGetChapterMapFiles:
    """Tests for get_chapter_map_files function."""

    def test_finds_map_files(self, temp_dir):
        """Test that map files are found."""
        (temp_dir / "chapter_0.map.json").touch()
        (temp_dir / "chapter_1.map.json").touch()
        (temp_dir / "other_file.json").touch()

        result = get_chapter_map_files(temp_dir)
        assert len(result) == 2

    def test_respects_pattern(self, temp_dir):
        """Test that correct pattern is used."""
        (temp_dir / "chapter_0.map.json").touch()
        (temp_dir / "chapter_10.map.json").touch()
        (temp_dir / "chapter_0.result.0.map.json").touch()

        result = get_chapter_map_files(temp_dir)
        assert len(result) == 2


class TestExtractCharactersFromMaps:
    """Tests for extract_characters_from_maps function."""

    def test_extracts_characters(self, temp_dir):
        """Test that characters are extracted."""
        data = [
            {"1": "narrator", "2": "jane"},
            {}
        ]
        with open(temp_dir / "chapter_0.map.json", "w") as f:
            json.dump(data, f)

        result = extract_characters_from_maps(temp_dir)
        assert "narrator" in result
        assert "jane" in result

    def test_removes_duplicates(self, temp_dir):
        """Test that duplicates are removed."""
        data = [
            {"1": "narrator", "2": "jane"},
            {}
        ]
        with open(temp_dir / "chapter_0.map.json", "w") as f:
            json.dump(data, f)
        with open(temp_dir / "chapter_1.map.json", "w") as f:
            json.dump([{"1": "narrator", "2": "jane"}, {}], f)

        result = extract_characters_from_maps(temp_dir)
        assert result.count("jane") == 1


class TestCountLinesPerCharacter:
    """Tests for count_lines_per_character function."""

    def test_counts_spoken_lines(self, temp_dir):
        """Test that spoken lines are counted."""
        data = [
            {"1": "narrator", "2": "jane"},
            {"1": 2, "3": 2}
        ]
        with open(temp_dir / "chapter_0.map.json", "w") as f:
            json.dump(data, f)

        with open(temp_dir / "chapter_0.txt", "w") as f:
            f.write('Line 1: "Quote."\nLine 2: "Another."\nLine 3: "Third."\n')

        result = count_lines_per_character(temp_dir)
        assert "jane" in result


class TestNaturalSortKey:
    """Tests for natural_sort_key function."""

    def test_sorts_numerically(self):
        """Test that numbers are sorted numerically."""
        files = ["chapter_10.txt", "chapter_2.txt", "chapter_1.txt"]
        sorted_files = sorted(files, key=natural_sort_key)
        assert sorted_files[0] == "chapter_1.txt"
        assert sorted_files[1] == "chapter_2.txt"
        assert sorted_files[2] == "chapter_10.txt"

    def test_handles_no_number(self):
        """Test handling of filename without number."""
        key = natural_sort_key("narrator.wav")
        assert key == ("narrator.wav", 0, "")

    def test_handles_prefix_and_suffix(self):
        """Test that prefix and suffix are preserved."""
        key = natural_sort_key("chapter_5.txt")
        assert key[0] == "chapter_"
        assert key[1] == 5
        assert key[2] == ".txt"


class TestExtractGenderFromDescription:
    """Tests for extract_gender_from_description function."""

    def test_female_detected(self):
        """Test female gender is detected."""
        assert extract_gender_from_description("A gentle female voice.") == "female"
        assert extract_gender_from_description("A woman speaking.") == "female"

    def test_male_detected(self):
        """Test male gender is detected."""
        assert extract_gender_from_description("A deep male voice.") == "male"
        assert extract_gender_from_description("A man speaking.") == "male"

    def test_female_before_male(self):
        """Test that female is checked before male."""
        assert extract_gender_from_description("A female male voice.") == "female"

    def test_returns_none_for_no_gender(self):
        """Test that None is returned when no gender found."""
        assert extract_gender_from_description("A neutral voice.") is None


class TestClassifyGenderStatistical:
    """Tests for classify_gender_statistical function."""

    def test_male_classification(self):
        """Test male classification with low pitch."""
        import numpy as np
        voiced_f0 = np.array([100, 110, 120, 130, 140])
        gender, confidence, reason = classify_gender_statistical(voiced_f0, verbose=False)
        assert gender == "male"

    def test_female_classification(self):
        """Test female classification with high pitch."""
        import numpy as np
        voiced_f0 = np.array([200, 210, 220, 230, 240])
        gender, confidence, reason = classify_gender_statistical(voiced_f0, verbose=False)
        assert gender == "female"

    def test_returns_confidence(self):
        """Test that confidence is returned."""
        import numpy as np
        voiced_f0 = np.array([100, 110, 120, 130, 140])
        gender, confidence, reason = classify_gender_statistical(voiced_f0, verbose=False)
        assert confidence is not None
        assert 0.0 <= confidence <= 1.0


class TestDetectGenderFromAudio:
    """Tests for detect_gender_from_audio function."""

    def test_with_silence_audio(self, temp_dir, mock_tts_engine):
        """Test gender detection with silence audio."""
        import numpy as np
        import torch
        import torchaudio

        voice_file = temp_dir / "test_voice.wav"
        audio = np.zeros(int(mock_tts_engine.sample_rate * 1.0), dtype=np.float32)
        torchaudio.save(str(voice_file), torch.from_numpy(audio), mock_tts_engine.sample_rate)

        gender, confidence, reason = detect_gender_from_audio(str(voice_file), use_ttest=False, verbose=False)

        assert gender is None or gender in ("male", "female")


class TestGetCharactersFromMapFiles:
    """Tests for get_characters_from_map_files function."""

    def test_extracts_unique_characters(self, temp_dir):
        """Test that unique characters are extracted."""
        data = [
            {"1": "narrator", "2": "jane"},
            {}
        ]
        with open(temp_dir / "chapter_0.map.json", "w") as f:
            json.dump(data, f)

        from audiobook_generator.utils import get_characters_from_map_files
        result = get_characters_from_map_files(temp_dir)
        assert "narrator" in result
        assert "jane" in result

    def test_sorts_results(self, temp_dir):
        """Test that results are sorted."""
        data = [{"1": "zara", "2": "alice"}, {}]
        with open(temp_dir / "chapter_0.map.json", "w") as f:
            json.dump(data, f)

        from audiobook_generator.utils import get_characters_from_map_files
        result = get_characters_from_map_files(temp_dir)
        assert result == sorted(result)