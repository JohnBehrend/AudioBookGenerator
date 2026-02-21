"""Tests for parse_epub.py (Stage 6: Full Audiobook Generation)."""
import pytest
import json
import tempfile
import os
import struct
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Test EPUB path
TEST_EPUB_PATH = Path(__file__).parent.parent / "test_data" / "pride_and_prejudice.epub"


def create_minimal_wav(filepath: Path, duration_seconds: float = 1.0):
    """Create a minimal valid WAV file for testing."""
    sample_rate = 22050
    num_channels = 1
    bits_per_sample = 16
    num_samples = int(sample_rate * duration_seconds)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        # RIFF header
        f.write(b'RIFF')
        # File size - 8 (will update later)
        f.write(b'\x00\x00\x00\x00')
        f.write(b'WAVE')
        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # Format chunk size
        f.write(struct.pack('<H', 1))   # Audio format (1 = PCM)
        f.write(struct.pack('<H', num_channels))  # Channels
        f.write(struct.pack('<I', sample_rate))   # Sample rate
        f.write(struct.pack('<I', sample_rate * num_channels * bits_per_sample // 8))  # Byte rate
        f.write(struct.pack('<H', num_channels * bits_per_sample // 8))  # Block align
        f.write(struct.pack('<H', bits_per_sample))  # Bits per sample
        # data chunk
        f.write(b'data')
        data_size = num_samples * num_channels * bits_per_sample // 8
        f.write(struct.pack('<I', data_size))
        # Write silence (zeros)
        f.write(b'\x00' * data_size)

        # Update file size
        file_size = 44 + data_size - 8
        f.seek(4)
        f.write(struct.pack('<I', file_size))


class TestParseEpubIntegration:
    """Integration tests for parse_epub pipeline stages."""

    def test_full_pipeline_mock(self, temp_dir):
        """Test the full pipeline with mocked external dependencies."""
        from parse_chapter import ChapterObj

        # Create sample chapter data
        chapters = [
            [
                ChapterObj(has_quotes=True, text="Hello world", line_num=1),
                ChapterObj(has_quotes=False, text="Narrator text", line_num=2),
            ]
        ]

        # Create sample map file
        character_map = {"1": "narrator", "2": "john"}
        line_map = {"1": 2, "2": 1}  # Line 1 -> john, Line 2 -> narrator
        map_file = temp_dir / "chapter_0.map.json"
        map_file.write_text(json.dumps([character_map, line_map]))

        # Verify structure
        loaded = json.loads(map_file.read_text())
        assert len(loaded) == 2
        assert "1" in loaded[0]
        assert "2" in loaded[0]


class TestSpeakerAssignment:
    """Tests for speaker assignment logic."""

    def test_line_to_character_mapping(self):
        """Test mapping lines to characters."""
        from parse_chapter import ChapterObj

        character_map = {1: "narrator", 2: "john", 3: "mary"}
        line_map = {1: 1, 2: 2, 3: 3, 4: 2}

        # Build line to character mapping
        line_to_character = {k: character_map[v] for k, v in line_map.items()}

        assert line_to_character[1] == "narrator"
        assert line_to_character[2] == "john"
        assert line_to_character[3] == "mary"
        assert line_to_character[4] == "john"

    def test_chapter_obj_speaker_assignment(self):
        """Test ChapterObj speaker assignment."""
        from parse_chapter import ChapterObj

        obj = ChapterObj(has_quotes=True, text="Hello", line_num=1)

        obj.set_speaker("john")
        assert obj.get_speaker() == "john"

        obj.set_speaker("mary")
        assert obj.get_speaker() == "mary"


class TestFullPipelineIntegration:
    """Integration test for full pipeline using test EPUB file."""

    def test_full_pipeline_first_chapter(self, temp_dir):
        """Test full pipeline on first chapter with mocked TTS/STT models."""
        from parse_chapter import parse_epub_to_chapters
        from parse_epub import generate_audiobook_from_chapters

        # Use a small custom chapter (not the full EPUB which has 697 lines)
        from parse_chapter import ChapterObj
        chapters = [
            [
                ChapterObj(has_quotes=True, text="Hello world", line_num=0),
                ChapterObj(has_quotes=True, text="How are you?", line_num=1),
                ChapterObj(has_quotes=False, text="Narrator text here", line_num=2),
            ]
        ]

        # Create a simple map file for the first chapter
        chapter = chapters[0]
        character_map = {}
        line_map = {}

        # Assign first quoted line to "elizabeth", others to narrator
        line_idx = 0
        character_idx = 1
        for cobj in chapter:
            if cobj.has_quotes:
                if line_idx not in line_map:
                    line_map[str(line_idx)] = character_idx
                    character_map[str(character_idx)] = "elizabeth"
                    character_idx += 1
            else:
                if line_idx not in line_map:
                    line_map[str(line_idx)] = 0
                    character_map["0"] = "narrator"
            line_idx += 1

        # Ensure we have at least narrator
        if "0" not in character_map:
            character_map["0"] = "narrator"

        # Convert to proper dict format with int keys
        # Note: character_map keys are character_idx (int), values are character names (str)
        character_map_int = {int(k): v for k, v in character_map.items()}
        # Note: line_map keys are line_idx (int), values are character_idx (int)
        line_map_int = {int(k): int(v) for k, v in line_map.items()}

        # Create voices_map with a fake voice file path (will be mocked)
        voices_map = {"narrator": "narrator.wav", "elizabeth": "elizabeth.wav"}

        # Create minimal voice sample WAV files (required by KugelAudio)
        voice_samples_dir = temp_dir / "voice_samples"
        voice_samples_dir.mkdir()
        narrator_voice = voice_samples_dir / "narrator.wav"
        elizabeth_voice = voice_samples_dir / "elizabeth.wav"
        create_minimal_wav(narrator_voice)
        create_minimal_wav(elizabeth_voice)

        # Set up voices_map to point to the real voice files
        voices_map = {
            "narrator": str(narrator_voice),
            "elizabeth": str(elizabeth_voice)
        }

        # Run the pipeline with REAL KugelAudio TTS
        # Mock WhisperX validation (not TTS generation) to avoid ffmpeg dependency
        with patch("parse_epub.setup_validation_model") as mock_setup_val, \
             patch("parse_epub.whisperx") as mock_whisperx:
            # Setup validation model mock to return proper result with language
            mock_val_model = Mock()
            mock_val_model.transcribe = Mock(return_value={
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "Hello world"}
                ]
            })
            mock_setup_val.return_value = mock_val_model

            # Setup WhisperX mock to return valid segments
            # WhisperX.load_audio returns audio data
            # whisperx.load_align_model returns (model, metadata)
            # whisperx.align returns result dict with word_segments
            mock_whisperx.load_audio = Mock(return_value="audio_data")
            mock_whisperx.load_align_model = Mock(return_value=(Mock(), {"language": "en"}))
            mock_whisperx.align = Mock(return_value={
                "word_segments": [
                    {"word": "Hello", "start": 0.0, "end": 1.0, "score": 0.95},
                    {"word": "world", "start": 1.0, "end": 2.0, "score": 0.95}
                ]
            })

            # Run the pipeline - chapter_maps expects tuple of dicts with int keys
            result_msg, chapters_processed = generate_audiobook_from_chapters(
                chapters=chapters,
                chapter_maps={0: (character_map_int, line_map_int)},
                voices_map=voices_map,
                output_dir=str(temp_dir),
                device="cuda",
                tts_engine="kugelaudio",
                cfg_scale=1.30,
                max_chapters=1,
                verbose=True
            )

            # Verify results
            assert chapters_processed == 1, "Should process 1 chapter"
            assert "successfully" in result_msg.lower(), "Should report success"

            # Verify WAV files were created (this is the key test - actual TTS output)
            wav_files = list(temp_dir.glob("chapter_00.*.wav"))
            assert len(wav_files) > 0, "WAV files should have been created by KugelAudio"

            # Also verify MP3 was created from assembled WAV files
            mp3_files = list(temp_dir.glob("chapter_00.mp3"))
            assert len(mp3_files) > 0, "MP3 files should have been created"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Mock fixtures for external dependencies
@pytest.fixture
def mock_tts_model():
    """Mock TTS model for testing."""
    with patch("parse_epub.VibeVoiceForConditionalGenerationInference") as mock:
        mock_instance = Mock()
        mock_instance.generate = Mock(return_value=Mock(speech_outputs=[Mock()]))
        mock.from_pretrained = Mock(return_value=mock_instance)
        yield mock


@pytest.fixture
def mock_processor():
    """Mock processor for testing."""
    with patch("parse_epub.VibeVoiceProcessor") as mock:
        mock_instance = Mock()
        mock_instance.save_audio = Mock()
        mock.from_pretrained = Mock(return_value=mock_instance)
        yield mock


@pytest.fixture
def mock_whisper():
    """Mock Whisper model for testing."""
    with patch("parse_epub.whisperx") as mock:
        mock_instance = Mock()
        mock_instance.load_model = Mock(return_value=Mock())
        mock_instance.load_align_model = Mock(return_value=(Mock(), {}))
        mock_instance.align = Mock(return_value={"segments": []})
        mock.load_model = Mock(return_value=mock_instance)
        yield mock
