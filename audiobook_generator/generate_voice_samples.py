#!/usr/bin/env python3
"""
Module to generate voice samples for audiobook characters.
"""
import json
import os
import re
import sys
import shutil
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple
from openai import OpenAI

# Import config for default values
from .config import DEFAULTS, AUDIO_SETTINGS, VOICE_SAMPLES_DIR
from .utils import get_validation_client

# Import VoiceMapper for centralized TTS management
from .voice_mapper import VoiceMapper


def load_character_descriptions(descriptions_file: str) -> Dict[str, str]:
    """Load character descriptions from JSON file."""
    with open(descriptions_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def _word_match_count(ref_words, transcribed_lower):
    """Count whole-word matches of ref_words in transcribed text."""
    return sum(1 for w in ref_words if re.search(r'\b' + re.escape(w) + r'\b', transcribed_lower))


def _validate_with_nemotron(voice_path: str, description: str, sample_text: str, nemotron_client, nemotron_model, verbose: bool = False) -> Tuple[bool, str]:
    """Validate voice matches description using Nemotron Omni."""
    import soundfile as sf
    import tempfile

    # Nemotron was trained on 16kHz audio; resample if needed
    data, sr = sf.read(voice_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000
        except ImportError:
            if verbose:
                print("    Warning: librosa not available, skipping resampling")
    tmp_path = tempfile.mktemp(suffix=".wav")
    sf.write(tmp_path, data, 16000, subtype="PCM_16")

    try:
        from pathlib import Path
        file_url = Path(tmp_path).resolve().as_uri()
        prompt = (
            'You will hear a voice sample. The spoken text is: "' + sample_text + '"\n\n'
            'The intended voice description is: "' + description + '"\n\n'
            'Analyze the voice and output JSON with these fields:\n'
            '{\n'
            '  "gender_match": true/false,\n'
            '  "age_match": true/false,\n'
            '  "tone_match": true/false,\n'
            '  "overall_match": true/false,\n'
            '  "reasons": "brief explanation of any mismatches"\n'
            '}'
        )
        response = nemotron_client.chat.completions.create(
            model=nemotron_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": file_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=512,
            temperature=0.2,
            extra_body={"top_k": 1, "chat_template_kwargs": {"enable_thinking": False}},
        )
        result = response.choices[0].message.content.strip()
        try:
            import json as json_module
            data = json_module.loads(result)
            is_valid = data.get("overall_match", False)
            if verbose:
                print(f"      Gender: {data.get('gender_match', 'N/A')}, Age: {data.get('age_match', 'N/A')}, Tone: {data.get('tone_match', 'N/A')}, Overall: {'PASS' if is_valid else 'FAIL'}")
                if data.get('reasons'):
                    print(f"      Reasons: {data['reasons']}")
            return is_valid, result
        except json_module.JSONDecodeError:
            is_valid = "YES" in result.upper() or "true" in result.lower()
            if verbose:
                print(f"    Nemotron result: {result[:100]}...")
            return is_valid, result
    except Exception as e:
        if verbose:
            print(f"    Nemotron validation error: {e}")
        return True, str(e)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def generate_voice_sample(character_name: str, description: str, voice_mapper: VoiceMapper,
                          output_dir: str, max_new_tokens: Optional[int] = None, verbose: bool = False,
                          validate: bool = False, validation_client: Optional[OpenAI] = None) -> Tuple[bool, Optional[str], float, bool, str]:
    """Generate a short voice sample for a character using VoiceDesign model via VoiceMapper.

    Uses voice design with an instruct prompt to generate speech
    in the designed voice style.

    Args:
        character_name: Name of the character
        description: Voice description from LLM
        voice_mapper: Shared VoiceMapper instance (prevents repeated model loading)
        output_dir: Directory to save voice samples
        max_new_tokens: Max tokens for generation
        verbose: Print verbose output
        validate: If True, validate the generated voice with LLM
        validation_client: OpenAI client for validation (created if None)

    Returns:
        Tuple of (success, output_file_path, duration_seconds, is_valid, validation_msg)
        When validate=True, is_valid indicates if the voice passed validation.
    """
    if max_new_tokens is None:
        max_new_tokens = DEFAULTS["max_new_tokens"]

    try:
        success, output_file, duration = voice_mapper.generate_voice_sample(
            character_name=character_name,
            description=description,
            output_dir=output_dir,
            max_new_tokens=max_new_tokens,
            verbose=verbose
        )

        is_valid = True  # Default: no validation = accepted
        validation_msg = ""

        if success and output_file and validate:
            if verbose:
                print(f"    Validating voice sample...")

            sample_text = DEFAULTS.get("static_voice_text", "")

            if validation_client is None:
                validation_client = get_validation_client()

            is_valid, validation_msg = VoiceMapper.validate_voice_with_llm(
                voice_path=output_file,
                description=description,
                sample_text=sample_text,
                client=validation_client,
                verbose=verbose
            )

            if not is_valid:
                print(f"    Warning: Voice validation failed: {validation_msg}")

        return success, output_file, duration, is_valid, validation_msg

    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)
        print(f"    Exception type: {type(e).__name__}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False, None, 0, False, "Exception during generation"


def generate_voice_samples(
    descriptions: Dict[str, str],
    output_dir: str,
    device: str = "cuda:0",
    max_tokens: int = DEFAULTS["max_new_tokens"],
    single_character: Optional[str] = None,
    verbose: bool = False,
    progress=None,
    seed_characters: Dict[str, str] = None,
    voice_engine: str = "moss",
    force_regenerate: bool = False,
    validate: bool = False,
    engine=None,
    tts_engine: str = None,
    nemotron_endpoint: str = None,
) -> Tuple[str, Dict[str, str]]:
    """Generate voice samples for characters via VoiceMapper.

    Args:
        descriptions: Dict mapping character names to their descriptions
        output_dir: Directory to save voice samples
        device: CUDA device (e.g., cuda:0, cpu)
        max_tokens: Max tokens for generation
        single_character: Generate only one character
        verbose: Print verbose output
        progress: Gradio progress bar to update during generation
        seed_characters: Dict mapping character names to existing voice paths from seed voices_map
        voice_engine: TTS engine for voice generation ('moss', 'omni', 'vox')
        force_regenerate: If True, regenerate voices even if they already exist
        validate: Deprecated - ignored

    Returns:
        Tuple of (status_message, character_voice_paths)
    """
    try:
        if single_character:
            if single_character not in descriptions:
                return f"Character '{single_character}' not found in descriptions.", {}
            descriptions = {single_character: descriptions[single_character]}
            if verbose:
                print(f"Generating voice for single character: {single_character}")
        else:
            if verbose:
                print(f"Found {len(descriptions)} characters")

        # Ensure narrator voice is included (it's always needed as fallback)
        if "narrator" not in descriptions:
            descriptions["narrator"] = "A neutral, clear narrator voice suitable for audiobook narration"
            if verbose:
                print("Added narrator voice to generation list (fallback voice)")

        # Pre-compute reference words for scoring
        ref_words = [w.strip("!\"',.").lower() for w in DEFAULTS["static_voice_text"].split() if w.strip("!\"',.").isalpha()]

        # Pre-load whisper validation model once for all samples
        from .utils import transcribe_audio_with_whisper, crop_to_ref_text, get_nemotron_client
        from .audiobook_generator import setup_validation_model
        vm = setup_validation_model('cpu', cpu=True, fast=True)

        # Set up Nemotron client if endpoint provided
        nemotron_client = None
        nemotron_model = None
        if nemotron_endpoint:
            try:
                from .config import NEMOTRON_VALIDATION
                nemotron_client = get_nemotron_client(nemotron_endpoint)
                nemotron_model = NEMOTRON_VALIDATION["model"]
                if verbose:
                    print(f"Nemotron validation enabled: {nemotron_endpoint}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to connect to Nemotron: {e}")

        # Filter out seed characters
        if seed_characters:
            initial_count = len(descriptions)
            descriptions = {k: v for k, v in descriptions.items() if k not in seed_characters}
            if verbose:
                print(f"Filtered out {initial_count - len(descriptions)} seeded characters, {len(descriptions)} remaining to generate")

            # Clone seed voices through TTS engine so they speak the configured
            # static_voice_text, ensuring ref_text always matches during TTS.
            clone_engine = tts_engine or voice_engine
            if verbose:
                print(f"Cloning seed voices through {clone_engine} engine...")
            voice_mapper_seed = VoiceMapper(output_dir=output_dir, device=device, tts_engine=clone_engine)
            tts_engine_obj = voice_mapper_seed.get_engine()
            for char_name, voice_path in seed_characters.items():
                if not os.path.exists(voice_path):
                    if verbose:
                        print(f"  Warning: Seed voice file not found: {voice_path}")
                    continue
                dest_path = os.path.join(output_dir, f"{char_name}.wav")
                if os.path.exists(dest_path):
                    if verbose:
                        print(f"  Skipped {char_name} - already exists")
                    continue
                # Check if seed voice already speaks the correct text by transcribing
                _seed_verified = False
                try:
                    _transcribed, _, _ = transcribe_audio_with_whisper(vm, voice_path)
                    _matches = _word_match_count(ref_words, _transcribed.lower())
                    if _matches >= len(ref_words) - 1:
                        shutil.copy2(voice_path, dest_path)
                        if verbose:
                            print(f"  Copied {voice_path} -> {dest_path} (already correct text)")
                        _seed_verified = True
                except Exception:
                    pass
                if not _seed_verified:
                    # Generate up to 10 samples, pick the best match
                    import random as _random
                    _candidates = []
                    _all_attempts = []
                    _att = 0
                    _max_attempts = 10
                    while len(_candidates) == 0 and _att < _max_attempts:
                        _att += 1
                        _random.seed(42 + _att)
                        _tmp_path = dest_path + f".seed{_att}.tmp.wav"
                        try:
                            tts_engine_obj.generate_line(
                                text=DEFAULTS["static_voice_text"],
                                voice_path=voice_path,
                                output_path=_tmp_path,
                                device=device,
                                validation_model=None,
                                verbose=False,
                                ref_text="",
                            )
                        except Exception as e:
                            if verbose:
                                print(f"    Attempt {_att} failed: {e}")
                            continue
                        if not os.path.exists(_tmp_path):
                            continue
                        try:
                            _transcribed, _starts, _ends = transcribe_audio_with_whisper(vm, _tmp_path)
                            _transcribed_words = _transcribed.split()
                            _matches = _word_match_count(ref_words, _transcribed.lower())
                            _cropped_path = _tmp_path + ".cropped.wav"
                            if crop_to_ref_text(_tmp_path, _cropped_path, ref_words, _transcribed_words, _starts, _ends, verbose=False):
                                _use_path = _cropped_path
                            else:
                                _use_path = _tmp_path
                            _all_attempts.append((_matches, _use_path, _att))
                            if _matches >= len(ref_words) * 0.8:
                                _candidates.append((_matches, _use_path, _att))
                                if verbose:
                                    print(f"    Sample {_att}: {_matches}/{len(ref_words)} words: {_transcribed[:80]}...")
                            else:
                                if verbose:
                                    print(f"    Sample {_att}: {_matches}/{len(ref_words)} words (too few): {_transcribed[:80]}...")
                        except Exception as e:
                            if verbose:
                                print(f"    Sample {_att}: processing failed ({e})")
                    if _candidates:
                        _candidates.sort(key=lambda x: x[0], reverse=True)
                        _best_score, _best_path, _best_att = _candidates[0]
                        shutil.copy2(_best_path, dest_path)
                        if verbose:
                            print(f"  Cloned {voice_path} -> {dest_path} (best: sample {_best_att}, {_best_score}/{len(ref_words)} words)")
                    else:
                        # All attempts failed, remove dest and add back to descriptions for Dramabox
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        descriptions[char_name] = f"Seed voice clone failed after {_max_attempts} attempts. Generate from description."
                        if verbose:
                            print(f"  Failed to clone {char_name}, will generate from description instead")
                    # Clean up all temp files after copying
                    for _fa in range(1, _att + 1):
                        _fp = dest_path + f".seed{_fa}.tmp.wav"
                        try:
                            os.remove(_fp)
                        except OSError:
                            pass
                        try:
                            os.remove(_fp + ".cropped.wav")
                        except OSError:
                            pass

        os.makedirs(output_dir, exist_ok=True)

        generated = {}
        failed = []
        total_chars = len(descriptions)

        if verbose:
            print("\n" + "=" * 60)
            print("NOTE: Character descriptions are no longer used for voice generation.")
            print("      Voices use a static string from config instead.")
            print("=" * 60 + "\n")

        # Create VoiceMapper once to cache the TTS model across all characters
        # Use tts_engine for voice generation (voice cloning), not voice_engine
        gen_engine = tts_engine or voice_engine
        voice_mapper = VoiceMapper(output_dir=output_dir, device=device, tts_engine=gen_engine, engine=engine)

        try:
            for i, (char_name, char_desc) in enumerate(descriptions.items()):
                if verbose:
                    print(f"[{i+1}/{total_chars}] {char_name}")

                # Update progress bar for each character
                if progress is not None:
                    progress((i + 1) / total_chars, desc=f"Generating voice for '{char_name}'...")

                # Check if voice sample already exists (resume mode)
                if char_name in generated:
                    if verbose:
                        print(f"    Skipped - already generated")
                    continue

                # Check for existing voice file in output_dir (skip if exists, unless force_regenerate)
                if not force_regenerate:
                    voice_found = False
                    for ext in [".wav", ".mp3", ".flac"]:
                        test_path = os.path.join(output_dir, f"{char_name}{ext}")
                        if os.path.exists(test_path):
                            generated[char_name] = test_path
                            if verbose:
                                print(f"    Found existing voice: {test_path}")
                            voice_found = True
                            break

                    if voice_found:
                        continue

                # Generate up to 10 samples, pick the best match
                import random
                candidates = []
                all_attempts = []
                attempt = 0
                max_attempts = 10
                while len(candidates) == 0 and attempt < max_attempts:
                    attempt += 1
                    random.seed(42 + attempt)
                    _tmp_name = f"{char_name}.sample{attempt}"
                    success, output_file, duration, is_valid, validation_msg = generate_voice_sample(
                        character_name=_tmp_name,
                        description=char_desc,
                        voice_mapper=voice_mapper,
                        output_dir=output_dir,
                        max_new_tokens=max_tokens,
                        verbose=False,
                        validate=False,
                        validation_client=None
                    )
                    if not success or not output_file:
                        continue
                    try:
                        transcribed, starts, ends = transcribe_audio_with_whisper(vm, output_file)
                        transcribed_words = transcribed.split()
                        matches = _word_match_count(ref_words, transcribed.lower())
                        if matches < len(ref_words) * 0.8:
                            if verbose:
                                print(f"    Sample {attempt}: {matches}/{len(ref_words)} words (too few): {transcribed[:80]}...")
                            continue
                        cropped_path = output_file + ".cropped.wav"
                        if crop_to_ref_text(output_file, cropped_path, ref_words, transcribed_words, starts, ends, verbose=False):
                            use_path = cropped_path
                        else:
                            use_path = output_file
                        all_attempts.append((matches, use_path, attempt, duration))
                        # Validate against description with Nemotron if available
                        if nemotron_client:
                            if verbose:
                                print(f"    Validating against description with Nemotron...")
                            nemotron_ok, nemotron_msg = _validate_with_nemotron(
                                use_path, char_desc, DEFAULTS["static_voice_text"],
                                nemotron_client, nemotron_model, verbose=verbose
                            )
                            if not nemotron_ok:
                                if verbose:
                                    print(f"    Sample {attempt}: Nemotron FAIL")
                                continue
                            else:
                                if verbose:
                                    print(f"    Sample {attempt}: Nemotron PASS")
                        if verbose:
                            print(f"    Sample {attempt}: {matches}/{len(ref_words)} words ({duration:.1f}s): {transcribed[:80]}...")
                        candidates.append((matches, use_path, attempt, duration))
                    except Exception as e:
                        if verbose:
                            print(f"    Sample {attempt}: processing failed ({e})")
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    best_score, best_file, best_att, best_dur = candidates[0]
                    final_path = os.path.join(output_dir, f"{char_name}.wav")
                    shutil.copy2(best_file, final_path)
                    generated[char_name] = final_path
                    if verbose:
                        print(f"    Best: sample {best_att}, {best_score}/{len(ref_words)} words ({best_dur:.1f}s): {final_path}")
                else:
                    failed.append(char_name)
                    if verbose:
                        print(f"    All {attempt} samples failed for '{char_name}'")
                # Always clean up all temp files
                for _fa in range(1, attempt + 1):
                    _fp = os.path.join(output_dir, f"{char_name}.sample{_fa}.wav")
                    try:
                        os.remove(_fp)
                    except OSError:
                        pass
                    try:
                        os.remove(_fp + ".cropped.wav")
                    except OSError:
                        pass

            # Clean up TTS models after all characters are processed
            voice_mapper.cleanup_tts_models()
        except Exception as e:
            return f"Error generating voices: {str(e)}\n{traceback.format_exc()}", {}

        if verbose:
            print("\n" + "=" * 60)
            print(f"Summary: {len(generated)} generated, {len(failed)} failed")
            print("=" * 60)

        # VoiceMapper automatically saves voices_map.json when voice paths are added
        if verbose:
            print(f"\nGenerated voices_map.json automatically by VoiceMapper")

        return f"Successfully generated {len(generated)} voice sample(s).", generated

    except Exception as e:
        error_msg = f"Error generating voice samples: {str(e)}\n{traceback.format_exc()}"
        if verbose:
            print(error_msg)
        return error_msg, {}
