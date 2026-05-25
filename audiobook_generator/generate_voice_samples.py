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
from typing import Dict, List, Optional, Tuple
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


def _validate_with_chunkformer(voice_path: str, description: str, chunkformer_model, verbose: bool = False) -> Tuple[bool, str]:
    """Validate voice matches description using ChunkFormer model.

    Uses ChunkFormer to classify the voice (gender, emotion, dialect, age),
    then compares the classification to the description to check gender match.
    """
    try:
        result = chunkformer_model.classify_audio(audio_path=voice_path)

        predicted_gender = result["gender"]["label"]
        predicted_emotion = result["emotion"]["label"]
        predicted_age = result["age"]["label"]
        predicted_dialect = result["dialect"]["label"]

        # Extract expected gender from description
        desc_lower = description.lower().strip()
        expected_gender = "female" if any(w in desc_lower for w in ["female", "woman", "women", "girl"]) else ("male" if any(w in desc_lower for w in ["male", "man", "men", "boy"]) else None)

        # Compare gender only
        gender_ok = expected_gender is None or predicted_gender == expected_gender

        is_valid = gender_ok
        reasons = []
        if not gender_ok:
            reasons.append(f"gender mismatch: expected {expected_gender}, got {predicted_gender}")

        if verbose:
            print(f"      Description: {description[:80]}")
            print(f"      Classified: {predicted_gender} / {predicted_age} / {predicted_emotion} / {predicted_dialect}")
            print(f"      Expected: {expected_gender}")
            print(f"      Overall: {'PASS' if is_valid else 'FAIL'}")
            if reasons:
                print(f"      Reasons: {'; '.join(reasons)}")

        # Log result to file for debugging
        log_entry = {
            "voice": os.path.basename(voice_path),
            "description": description,
            "classification": {
                "gender": {"label": predicted_gender, "prob": result["gender"]["prob"]},
                "emotion": {"label": predicted_emotion, "prob": result["emotion"]["prob"]},
                "age": {"label": predicted_age, "prob": result["age"]["prob"]},
                "dialect": {"label": predicted_dialect, "prob": result["dialect"]["prob"]},
            },
            "expected_gender": expected_gender,
            "gender_ok": gender_ok,
            "is_valid": is_valid,
            "reasons": reasons,
        }
        log_path = os.path.join(os.path.dirname(voice_path), ".chunkformer_validation.json")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return is_valid, json.dumps({"classification": {
            "gender": predicted_gender, "emotion": predicted_emotion,
            "age": predicted_age, "dialect": predicted_dialect,
        }, "gender_ok": gender_ok, "reasons": reasons})

    except Exception as e:
        if verbose:
            print(f"    ChunkFormer validation error: {e}")
        return True, str(e)


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
    use_chunkformer: bool = False,
    seed_clone_fallback_engines: List[str] = None,
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
        use_chunkformer: If True, validate voices with ChunkFormer model
        seed_clone_fallback_engines: List of engine names to try if primary engine fails

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

        # Pre-compute reference words for scoring, skipping first 2 sentences (throwaway prefix for TTS clipping)
        import re as _re
        _sentences = _re.split(r'[.!?]+', DEFAULTS["static_voice_text"])
        _validation_text = ' '.join(s.strip() for s in _sentences[2:] if s.strip())
        ref_words = [w.strip("!\"',.").lower() for w in _validation_text.split() if w.strip("!\"',.").isalpha()]

        # Pre-load whisper validation model once for all samples
        from .utils import transcribe_audio_with_whisper, crop_to_ref_text, get_chunkformer_model
        from .audiobook_generator import setup_validation_model
        vm = setup_validation_model('cpu', cpu=True, fast=True)

        # Set up ChunkFormer model if enabled
        chunkformer_model = None
        if use_chunkformer:
            try:
                chunkformer_model = get_chunkformer_model()
                if verbose:
                    print(f"ChunkFormer validation enabled")
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to load ChunkFormer: {e}")

        # Filter out seed characters, keeping their descriptions for validation
        seed_descriptions = {}
        if seed_characters:
            initial_count = len(descriptions)
            for k in seed_characters:
                if k in descriptions:
                    seed_descriptions[k] = descriptions[k]
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
                    if _matches >= len(ref_words) * 0.8:
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
                        # All attempts failed with primary engine, try fallback engines
                        _clone_success = False
                        _all_best_attempts = list(_all_attempts)  # Keep primary engine attempts
                        _fallback_temp_files = []
                        if seed_clone_fallback_engines:
                            for _fallback_engine_name in seed_clone_fallback_engines:
                                if _clone_success:
                                    break
                                if verbose:
                                    print(f"  Trying fallback engine: {_fallback_engine_name}")
                                try:
                                    _fallback_mapper = VoiceMapper(output_dir=output_dir, device=device, tts_engine=_fallback_engine_name)
                                    _fallback_obj = _fallback_mapper.get_engine()
                                except Exception as e:
                                    print(f"    Could not load {_fallback_engine_name}: {e}")
                                    continue
                                _fallback_candidates = []
                                _fallback_att = 0
                                while len(_fallback_candidates) == 0 and _fallback_att < _max_attempts:
                                    _fallback_att += 1
                                    _random.seed(42 + _fallback_att + 100)
                                    _tmp_path = dest_path + f".seed{_att + _fallback_att}.tmp.wav"
                                    try:
                                        _fallback_obj.generate_line(
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
                                            print(f"    Sample {_fallback_att}: generation failed ({e})")
                                        continue
                                    if not os.path.exists(_tmp_path):
                                        if verbose:
                                            print(f"    Sample {_fallback_att}: no output file")
                                        continue
                                    _fallback_temp_files.append(_tmp_path)
                                    try:
                                        _transcribed, _starts, _ends = transcribe_audio_with_whisper(vm, _tmp_path)
                                        _transcribed_words = _transcribed.split()
                                        _matches = _word_match_count(ref_words, _transcribed.lower())
                                        _cropped_path = _tmp_path + ".cropped.wav"
                                        if crop_to_ref_text(_tmp_path, _cropped_path, ref_words, _transcribed_words, _starts, _ends, verbose=False):
                                            _use_path = _cropped_path
                                            _fallback_temp_files.append(_cropped_path)
                                        else:
                                            _use_path = _tmp_path
                                        _fallback_candidates.append((_matches, _use_path, _fallback_att))
                                        _all_best_attempts.append((_matches, _use_path, _fallback_att))
                                        if verbose:
                                            print(f"    Sample {_fallback_att}: {_matches}/{len(ref_words)} words: {_transcribed[:80]}...")
                                    except Exception as e:
                                        if verbose:
                                            print(f"    Sample {_fallback_att}: processing failed ({e})")
                                        pass
                                # Unload fallback engine to free GPU memory
                                _fallback_mapper.cleanup_engines()
                                if _fallback_candidates:
                                    _fallback_candidates.sort(key=lambda x: x[0], reverse=True)
                                    _best_score, _best_path, _best_att = _fallback_candidates[0]
                                    shutil.copy2(_best_path, dest_path)
                                    _clone_success = True
                                    if verbose:
                                        print(f"  Cloned via {_fallback_engine_name} (best: sample {_best_att}, {_best_score}/{len(ref_words)} words)")
                                    # Clean up fallback temp files
                                    for _ft in _fallback_temp_files:
                                        try:
                                            os.remove(_ft)
                                        except OSError:
                                            pass
                                    _fallback_temp_files = []
                        if not _clone_success:
                            # All engines failed, accept the best sample from any attempt
                            # The cloned voice will speak whatever text the engine produced,
                            # which is fine — ref_text will be set via Whisper during TTS
                            if _all_best_attempts:
                                _all_best_attempts.sort(key=lambda x: x[0], reverse=True)
                                _best_score, _best_path, _best_att = _all_best_attempts[0]
                                shutil.copy2(_best_path, dest_path)
                                if verbose:
                                    print(f"  Cloned {voice_path} -> {dest_path} (best: sample {_best_att}, {_best_score}/{len(ref_words)} words — below threshold but accepted)")
                                # Clean up all fallback temp files except winner
                                for _ft in _fallback_temp_files:
                                    if _ft != _best_path:
                                        try:
                                            os.remove(_ft)
                                        except OSError:
                                            pass
                            else:
                                # Truly no samples at all, fall back to Dramabox
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
                        # Validate against description with ChunkFormer if available
                        if chunkformer_model:
                            if verbose:
                                print(f"    Validating against description with ChunkFormer...")
                            cf_ok, cf_msg = _validate_with_chunkformer(
                                use_path, char_desc,
                                chunkformer_model, verbose=verbose
                            )
                            if not cf_ok:
                                if verbose:
                                    print(f"    Sample {attempt}: ChunkFormer FAIL")
                                continue
                            else:
                                if verbose:
                                    print(f"    Sample {attempt}: ChunkFormer PASS")
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
