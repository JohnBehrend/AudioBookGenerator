#!/usr/bin/env python3
"""Testing utilities for audiobook_generator."""

import os
import json
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Tuple
from unittest.mock import MagicMock

import torch
import torchaudio


def write_silence_wav(path, sample_rate: int = 22050, duration: float = 1.0) -> Path:
    """Write a WAV file containing silence, for tests.

    Replaces the repeated ``np.zeros`` + ``torchaudio.save`` boilerplate scattered
    across the test suite.

    Args:
        path: Output path (str or Path)
        sample_rate: Sample rate in Hz
        duration: Duration in seconds

    Returns:
        The output path as a Path object
    """
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    torchaudio.save(str(path), torch.from_numpy(audio), sample_rate)
    return Path(path)


VOICE_FILE_NAMES = ("narrator.wav", "jane.wav", "elizabeth.wav")


def create_voice_files(voice_dir, sample_rate: int = 22050, duration: float = 1.0) -> Path:
    """Create dummy WAV voice files (narrator/jane/elizabeth) in ``voice_dir``.

    Args:
        voice_dir: Directory to write voice files into (created if needed)
        sample_rate: Sample rate in Hz
        duration: Duration in seconds

    Returns:
        The voice_dir as a Path object
    """
    voice_dir = Path(voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    for name in VOICE_FILE_NAMES:
        write_silence_wav(voice_dir / name, sample_rate, duration)
    return voice_dir


@contextmanager
def patch_audiobook_pipeline(
    *,
    output_dir: Optional[str] = None,
    exists: Any = False,
    glob_wavs: Optional[list] = None,
    voice_path: Optional[str] = "/tmp/test_voice.wav",
    create_voices: bool = False,
    patch_join: bool = True,
    patch_rename: bool = False,
):
    """Patch all dependencies of ``generate_audiobook_from_chapters``.

    Centralizes the deep mock harness previously duplicated across several test
    files, so refactors of the pipeline break one fixture instead of N copies.

    Args:
        output_dir: Directory passed as ``output_dir`` to the pipeline (needed
            only when ``create_voices`` is True).
        exists: Return value or side_effect for ``os.path.exists``.
        glob_wavs: List returned by ``glob.glob`` (defaults to one wav path).
        voice_path: Value returned by ``VoiceMapper.get_voice_path``.
        create_voices: If True, write dummy voice files into ``output_dir``.
        patch_join: Whether to patch ``os.path.join`` with the real one.
        patch_rename: Whether to also patch ``os.rename``.

    Yields:
        The mocked ``generate_tts_for_line``.
    """
    from contextlib import ExitStack
    from unittest.mock import MagicMock, patch

    from audiobook_generator import audiobook_generator as abg

    if create_voices:
        if not output_dir:
            raise ValueError("output_dir is required when create_voices=True")
        create_voice_files(output_dir)

    glob_default = ["/tmp/chapter_00.0002.wav"] if glob_wavs is None else glob_wavs

    with patch.object(abg, "setup_validation_model") as mock_validation:
        mock_validation.return_value = MagicMock()
        with patch.object(abg, "get_validation_client"):
            with patch.object(abg, "VoiceMapper") as mock_mapper_cls:
                mock_mapper_cls.return_value = MagicMock()
                mock_mapper_cls.return_value.add_voice_path.return_value = None
                mock_mapper_cls.return_value.get_voice_path.return_value = voice_path
                with patch.object(abg, "generate_tts_for_line") as mock_tts:
                    mock_tts.return_value = (True, 0.95)
                    with patch.object(abg, "_validate_and_clip_audio", return_value=(0.95, None)):
                        with patch.object(abg, "get_non_silent_audio_from_wavs") as mock_wavs:
                            mock_wavs.return_value = MagicMock()
                            with patch.object(abg.glob, "glob", return_value=glob_default):
                                with patch.object(abg, "ProgressHandler"):
                                    with patch.object(abg.os, "makedirs"):
                                        with patch.object(abg.os, "unlink"):
                                            with ExitStack() as stack:
                                                if callable(exists):
                                                    stack.enter_context(
                                                        patch.object(abg.os.path, "exists", side_effect=exists)
                                                    )
                                                else:
                                                    stack.enter_context(
                                                        patch.object(abg.os.path, "exists", return_value=exists)
                                                    )
                                                stack.enter_context(patch.object(abg.gc, "collect"))
                                                if patch_join:
                                                    stack.enter_context(
                                                        patch.object(abg.os.path, "join", side_effect=os.path.join)
                                                    )
                                                if patch_rename:
                                                    stack.enter_context(patch.object(abg.os, "rename"))
                                                yield mock_tts


class MockLLMClient:
    """Mock LLM client for testing without a running LLM server.

    This mock captures chat.completions.create() calls and returns
    configurable responses.

    Usage:
        from audiobook_generator.testing import MockLLMClient

        mock = MockLLMClient()
        mock.set_response({"role": "assistant", "content": '{"speaker_map": {"1": "narrator"}}'})

        # Use in label_speakers
        label_speakers(txt_file, api_key, port, client=mock)

        # Check what was sent
        print(mock.last_request)
    """

    def __init__(self):
        """Initialize mock client."""
        self.chat = _ChatCompletionsWrapper(self)
        self.base_url = "http://localhost:1234/v1"
        self.api_key = "mock-key"
        self.last_request: Optional[dict] = None

    def set_response(self, response: dict) -> None:
        """Set the response for the next chat.completions.create() call.

        Args:
            response: Response dict with 'content' field for assistant message
        """
        self._next_response = response

    def set_responses(self, responses: list) -> None:
        """Set multiple responses for sequential calls.

        Args:
            responses: List of response dicts
        """
        self._responses = responses
        self._response_index = 0

    def set_exception(self, exception: Exception) -> None:
        """Set an exception to raise on the next chat.completions.create() call.

        Args:
            exception: Exception instance to raise
        """
        self._next_exception = exception

    def get_next_response(self) -> dict:
        """Get the next response from the queue."""
        if hasattr(self, "_responses") and self._responses:
            if self._response_index < len(self._responses):
                response = self._responses[self._response_index]
                self._response_index += 1
                return response
        if hasattr(self, "_next_response"):
            return self._next_response
        return {"role": "assistant", "content": "{}"}


class _ChatCompletionsMock:
    """Mock for openai.ChatCompletions."""

    def __init__(self, client: MockLLMClient):
        self._client = client

    def create(self, model: str, messages: list, **kwargs):
        """Mock chat.completions.create().

        Captures the request and returns a mock response, or raises
        a configured exception if set via set_exception().
        """
        self._client.last_request = {
            "model": model,
            "messages": messages,
            "kwargs": kwargs,
        }

        if hasattr(self._client, "_next_exception"):
            exc = self._client._next_exception
            del self._client._next_exception
            raise exc

        response_content = self._client.get_next_response()

        return ChatCompletion(response_content)


class _ChatCompletionsWrapper:
    """Wrapper that provides both .completions and .chat.completions access patterns."""

    def __init__(self, client: MockLLMClient):
        self._client = client
        self._completions = _ChatCompletionsMock(client)

    @property
    def completions(self):
        """Provide .completions access (for newer OpenAI API)."""
        return self._completions

    @property
    def chat(self):
        """Provide .chat access for chained .chat.completions pattern."""
        return self


class ChatCompletionChoice:
    """Mock for openai.ChatCompletionChoice."""

    def __init__(self, message: dict):
        self.message = MockMessage(message)
        self.choices = [self]


class ChatCompletion:
    """Mock for openai.ChatCompletion response object."""

    def __init__(self, message: dict):
        self.choices = [ChatCompletionChoice(message)]


class MockMessage:
    """Mock for openai.types.chat.ChatCompletionMessage."""

    def __init__(self, message: dict):
        self.content = message.get("content", "{}")
        self.role = message.get("role", "assistant")
        self.reasoning = None


class MockTTSEngine:
    """Mock TTS engine for testing without GPU/heavy dependencies.

    This engine simulates TTS generation by writing silence audio files.
    It can be configured to return success or failure for testing.

    Usage:
        from audiobook_generator.testing import MockTTSEngine

        engine = MockTTSEngine()
        voice_mapper = VoiceMapper(output_dir="/tmp/test", engine=engine)
    """

    def __init__(
        self,
        generate_success: bool = True,
        generate_voice_success: bool = True,
        duration: float = 1.0,
        sample_rate: int = 22050,
    ):
        """Initialize mock engine.

        Args:
            generate_success: Whether generate_line should return success
            generate_voice_success: Whether generate_voice_sample should return success
            duration: Duration of generated audio in seconds
            sample_rate: Sample rate for generated audio
        """
        self.generate_success = generate_success
        self.generate_voice_success = generate_voice_success
        self.duration = duration
        self.sample_rate = sample_rate
        self._device = "cpu"
        self._worker = None

        self.last_generate_line_args: dict = {}
        self.last_generate_voice_args: dict = {}

    def setup(self, device: str, turbo: bool = False) -> Tuple[Any, Optional[Any]]:
        """Mock setup - returns dummy values."""
        self._device = device
        return None, None

    def generate_line(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        device: str = "cpu",
        verbose: bool = False,
        **kwargs,
    ) -> bool:
        """Mock line generation - writes silence audio.

        Args:
            text: Text that would be synthesized
            voice_path: Path to voice reference (ignored)
            output_path: Where to write the output audio
            device: Device string (ignored)
            verbose: Print verbose output
            **kwargs: Additional arguments (ref_text, validation_model, etc.)

        Returns:
            generate_success value
        """
        self.last_generate_line_args = {
            "text": text,
            "voice_path": voice_path,
            "output_path": output_path,
            "device": device,
            "kwargs": kwargs,
        }

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        num_samples = int(self.sample_rate * self.duration)
        audio = np.zeros(num_samples, dtype=np.float32)

        torchaudio.save(output_path, torch.from_numpy(audio), self.sample_rate)

        if verbose:
            print(f"[MockTTS] Generated {output_path} ({self.duration}s silence)")

        return self.generate_success

    def generate_voice_sample(
        self,
        character_name: str,
        description: str,
        output_dir: Path,
        device: str,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[bool, Optional[str], float]:
        """Mock voice sample generation.

        Args:
            character_name: Character name
            description: Voice description (ignored)
            output_dir: Directory to save sample
            device: Device string (ignored)
            verbose: Print verbose output

        Returns:
            Tuple of (generate_voice_success, output_path, duration)
        """
        self.last_generate_voice_args = {
            "character_name": character_name,
            "description": description,
            "output_dir": str(output_dir),
            "device": device,
        }

        output_path = output_dir / f"{character_name}.wav"
        os.makedirs(output_dir, exist_ok=True)

        num_samples = int(self.sample_rate * self.duration)
        audio = np.zeros(num_samples, dtype=np.float32)

        torchaudio.save(str(output_path), torch.from_numpy(audio), self.sample_rate)

        if verbose:
            print(f"[MockTTS] Generated voice sample {output_path}")

        return self.generate_voice_success, str(output_path), self.duration

    def shutdown(self) -> None:
        """Mock shutdown - no-op."""
        pass

    def shutdown_worker(self) -> None:
        """Mock shutdown - no-op."""
        pass

    def reset(self) -> None:
        """Reset captured arguments."""
        self.last_generate_line_args = {}
        self.last_generate_voice_args = {}
