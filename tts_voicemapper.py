"""VoiceMapper wrapper for TTS engines to map voice names to audio file paths."""

import os


class VoiceMapper:
    """Maps voice character names to their corresponding audio sample files.

    Looks for voice samples in the character_voice_samples directory with
    support for various audio extensions (.wav, .mp3, .flac).
    """

    def __init__(self, voices_dir=None):
        """Initialize the VoiceMapper.

        Args:
            voices_dir: Path to directory containing voice sample files.
                       Defaults to 'character_voice_samples' in script directory.
        """
        if voices_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            voices_dir = os.path.join(script_dir, 'character_voice_samples')
        self.voices_dir = voices_dir

    def get_voice_path(self, voice_name):
        """Get the path to a voice sample file for the given voice name.

        Args:
            voice_name: The character/voice name to find a sample for.

        Returns:
            Path to the voice sample file.

        Raises:
            FileNotFoundError: If no matching voice sample is found.
        """
        # Try to find the voice file with various extensions
        for ext in ['.wav', '.mp3', '.flac']:
            path = os.path.join(self.voices_dir, f"{voice_name}{ext}")
            if os.path.exists(path):
                return path

        # If exact match not found, try to find partial match (case-insensitive)
        voice_files = os.listdir(self.voices_dir)
        voice_name_lower = voice_name.lower()
        for vf in voice_files:
            if voice_name_lower in vf.lower():
                return os.path.join(self.voices_dir, vf)

        raise FileNotFoundError(f"Voice file not found for: {voice_name}")