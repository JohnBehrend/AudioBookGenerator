"""Shared utilities for TTS engines."""


def split_text_for_echo_tts(text: str, max_chunk_size: int = 500) -> list:
    """Split text into chunks that fit within Echo TTS token limits.

    Echo TTS has a sequence_length limit of 640 tokens (~30 seconds of audio).
    Splits at sentence boundaries first, then commas, then word boundaries.
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chunk_size:
            chunks.append(remaining)
            break

        split_point = max_chunk_size

        for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
            idx = remaining[:max_chunk_size].rfind(sep)
            if idx > max_chunk_size // 2:
                split_point = idx + len(sep)
                break

        if split_point == max_chunk_size:
            for sep in [", ", "; "]:
                idx = remaining[:max_chunk_size].rfind(sep)
                if idx > max_chunk_size // 2:
                    split_point = idx + len(sep)
                    break

        if split_point == max_chunk_size:
            last_space = remaining[:max_chunk_size].rfind(" ")
            if last_space > max_chunk_size // 2:
                split_point = last_space

        chunks.append(remaining[:split_point].strip())
        remaining = remaining[split_point:].strip()

    return chunks
