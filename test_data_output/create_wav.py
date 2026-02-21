import struct
import wave

def create_empty_wav(filename, channels=1, sample_width=2, sample_rate=24000):
    """Create a minimal valid WAV file."""
    with wave.open(filename, 'w') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        # Write a small amount of silence
        wav.writeframes(b'\x00' * 100)

create_empty_wav('narrator.wav')
create_empty_wav('gentleman.wav')
create_empty_wav('lady.wav')
print("Created voice sample WAV files")
