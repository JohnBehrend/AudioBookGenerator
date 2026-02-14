# Test Suite for Audiobook Pipeline

This directory contains pytest test files for validating each stage of the audiobook pipeline.

## Pipeline Stages Tested

| Stage | Test File | Description |
|-------|-----------|-------------|
| 1. EPUB Parsing | `test_parse_chapter.py` | Tests for `parse_chapter.py` module |
| 2. LLM Speaker Labeling | `test_llm_label_speakers.py` | Tests for `llm_label_speakers.py` module |
| 3. Chapter Analysis | `test_analyze_chapters.py` | Tests for `analyze_chapters.py` module |
| 4. Character Descriptions | `test_llm_describe_character.py` | Tests for `llm_describe_character.py` module |
| 5. Voice Sample Generation | `test_generate_voice_samples.py` | Tests for `generate_voice_samples.py` module |
| 6. Full Audiobook Generation | `test_parse_epub.py` | Tests for `parse_epub.py` module |

## Running Tests

### Install Test Dependencies
```bash
pip install -r requirements-tests.txt
```

### Run All Tests
```bash
pytest
```

### Run Tests with Verbose Output
```bash
pytest -v
```

### Run Specific Test File
```bash
pytest tests/test_parse_chapter.py
pytest tests/test_llm_label_speakers.py
pytest tests/test_analyze_chapters.py
pytest tests/test_llm_describe_character.py
pytest tests/test_generate_voice_samples.py
pytest tests/test_parse_epub.py
```

### Run Specific Test Class
```bash
pytest tests/test_parse_chapter.py::TestChapterObj
pytest tests/test_llm_label_speakers.py::TestAddQuotesAroundKeys
```

### Run Tests Matching Pattern
```bash
pytest -k "narrator"
pytest -k "load"
```

## Test Fixtures

The `conftest.py` file provides these fixtures:

- `temp_dir` - Temporary directory for test outputs
- `sample_chapter_text` - Sample chapter text for testing
- `sample_chapter_with_quoted_lines` - Sample chapter with quoted lines
- `sample_character_map` - Sample character map
- `sample_line_map` - Sample line map
- `sample_map_json` - Sample map JSON content
- `sample_characters` - Sample characters list
- `sample_characters_json` - Sample characters JSON
- `sample_character_descriptions` - Sample character descriptions
- `sample_character_descriptions_json` - Sample descriptions JSON
- `chapter_files` - Sample chapter map files
- `chapter_texts_dir` - Sample chapter text directory

## Code Coverage

Run tests with coverage:
```bash
pytest --cov=. --cov-report=term-missing --cov-report=html
```

## CI/CD Integration

The test suite is configured to run on CI/CD pipelines. The `pytest.ini` file defines:
- Test discovery paths
- Test naming conventions
- Timeout settings
- Logging configuration

## Writing New Tests

1. Create a new test file following the naming pattern `test_*.py`
2. Import fixtures from `conftest.py` as needed
3. Write test functions starting with `test_`
4. Use `assert` statements to validate behavior
5. Group related tests in classes starting with `Test`

Example:
```python
def test_something(temp_dir):
    # Use temp_dir fixture for test files
    assert True
```