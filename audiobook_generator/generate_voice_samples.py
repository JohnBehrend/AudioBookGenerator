#!/usr/bin/env python3
"""
Voice Sample Generator for Audiobook Characters

Generates short voice sample .wav files for each character based on their
description. Uses TTS engine from AUDIO_SETTINGS (default: kugelaudio).
"""

import argparse
import json
import os
import sys
import shutil
import traceback
from pathlib import Path
from typing import Optional
from openai import OpenAI

# Import config for default values
from config import DEFAULTS, AUDIO_SETTINGS, VOICE_SAMPLES_DIR, VOICE_GENDER_CORRECTION
from utils import get_validation_client, correct_voice_gender, extract_gender_from_description

# Import VoiceMapper for centralized TTS management
from voice_mapper import VoiceMapper


def load_character_descriptions(descriptions_file):
    """Load character descriptions from JSON file."""
    with open(descriptions_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_voice_sample(character_name: str, description: str, output_dir: str,
                          device: str = None, max_new_tokens: int = None, verbose: bool = False,
                          validate: bool = False, validation_client: OpenAI = None) -> tuple:
    """Generate a short voice sample for a character using VoiceDesign model via VoiceMapper.

    Uses voice design with an instruct prompt to generate speech
    in the designed voice style.

    Args:
        character_name: Name of the character
        description: Voice description from LLM
        output_dir: Directory to save voice samples
        device: CUDA device (uses config default if not specified)
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
    if device is None:
        device = AUDIO_SETTINGS["default_device"]

    # Stage 4 always uses MOSS TTS engine
    engine = "moss"
    voice_mapper = VoiceMapper(output_dir=output_dir, device=device, tts_engine=engine)

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

        voice_mapper.cleanup_tts_models()

        return success, output_file, duration, is_valid, validation_msg

    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)
        print(f"    Exception type: {type(e).__name__}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False, None, 0, False, "Exception during generation"


def main():
    parser = argparse.ArgumentParser(
        description="Generate voice samples for audiobook characters"
    )
    parser.add_argument(
        "--descriptions",
        default="characters_descriptions.json",
        help="Path to characters_descriptions.json"
    )
    parser.add_argument(
        "--output-dir",
        default=VOICE_SAMPLES_DIR,
        help="Directory to save voice samples"
    )
    parser.add_argument(
        "--device",
        default=AUDIO_SETTINGS["default_device"],
        help="CUDA device (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULTS["max_new_tokens"],
        help="Max tokens for generation"
    )
    parser.add_argument(
        "--single-character",
        help="Generate only one character"
    )
    parser.add_argument(
        "--seed-voice-map",
        help="Path to existing voices_map.json to seed voices"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose printing for debug."
    )
    parser.add_argument(
        "--tts-engine",
        default=AUDIO_SETTINGS.get("default_tts_engine", "moss"),
        help="TTS engine to use ('kugelaudio', 'vibevoice', 'moss', 'echo-tts')"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable LLM validation to verify generated voices match character descriptions"
    )

    args = parser.parse_args()

    if not os.path.exists(args.descriptions):
        print(f"Error: Descriptions file not found: {args.descriptions}", file=sys.stderr)
        sys.exit(1)

    descriptions = load_character_descriptions(args.descriptions)

    if args.single_character:
        if args.single_character not in descriptions:
            print(f"Error: Character '{args.single_character}' not found.", file=sys.stderr)
            sys.exit(1)
        descriptions = {args.single_character: descriptions[args.single_character]}

    # Load seed voices if provided
    seed_characters = None
    if args.seed_voice_map:
        if os.path.exists(args.seed_voice_map):
            with open(args.seed_voice_map, 'r', encoding='utf-8') as f:
                seed_characters = json.load(f)

    status, generated = generate_voice_samples(
        descriptions=descriptions,
        output_dir=args.output_dir,
        device=args.device,
        max_tokens=args.max_tokens,
        single_character=args.single_character,
        verbose=args.verbose,
        seed_characters=seed_characters,
        tts_engine=args.tts_engine,
        validate=args.validate
    )

    print(status)


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================
# generate_voice_samples is the public function used by both CLI and Gradio UI.
# The CLI main() function above calls this function to do the actual work.
# This provides a single, consistent interface for all callers.

from typing import Dict, Tuple


def generate_voice_samples(
    descriptions: Dict[str, str],
    output_dir: str,
    device: str = "cuda:0",
    max_tokens: int = DEFAULTS["max_new_tokens"],
    single_character: Optional[str] = None,
    verbose: bool = False,
    progress=None,
    seed_characters: Dict[str, str] = None,
    tts_engine: str = None,
    force_regenerate: bool = False,
    validate: bool = False
) -> Tuple[str, Dict[str, str]]:
    """Generate voice samples for characters via VoiceMapper.

    This is a simplified interface for calling voice sample generation directly
    from the Gradio UI without subprocess.

    Args:
        descriptions: Dict mapping character names to their descriptions
        output_dir: Directory to save voice samples
        device: CUDA device (e.g., cuda:0, cpu)
        max_tokens: Max tokens for generation
        single_character: Generate only one character
        verbose: Print verbose output
        progress: Gradio progress bar to update during generation
        seed_characters: Dict mapping character names to existing voice paths from seed voices_map
        tts_engine: TTS engine to use (ignored - always uses 'moss' for voice generation)
        force_regenerate: If True, regenerate voices even if they already exist
        validate: If True, validate generated voices with LLM

    Returns:
        Tuple of (status_message, character_voice_paths)
    """
    try:
        # Create validation client once if needed (efficiency - avoid creating per-voice)
        validation_client = get_validation_client() if validate else None
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
        # Add narrator with a default description if not already present
        if "narrator" not in descriptions:
            descriptions["narrator"] = "A neutral, clear narrator voice suitable for audiobook narration"
            if verbose:
                print("Added narrator voice to generation list (fallback voice)")

        # Filter out seed characters
        if seed_characters:
            initial_count = len(descriptions)
            descriptions = {k: v for k, v in descriptions.items() if k not in seed_characters}
            if verbose:
                print(f"Filtered out {initial_count - len(descriptions)} seeded characters, {len(descriptions)} remaining to generate")

            # Copy seed voice files to output directory
            if verbose:
                print(f"Copying seed voice files to {output_dir}...")
            for char_name, voice_path in seed_characters.items():
                if os.path.exists(voice_path):
                    dest_path = os.path.join(output_dir, os.path.basename(voice_path))
                    shutil.copy2(voice_path, dest_path)
                    if verbose:
                        print(f"  Copied {voice_path} -> {dest_path}")
                else:
                    if verbose:
                        print(f"  Warning: Seed voice file not found: {voice_path}")

        os.makedirs(output_dir, exist_ok=True)

        generated = {}
        failed = []
        total_chars = len(descriptions)

        # Note: Descriptions are no longer used for voice generation
        # Voices use static text from config instead
        if verbose:
            print("\n" + "=" * 60)
            print("NOTE: Character descriptions are no longer used for voice generation.")
            print("      Voices use a static string from config instead.")
            print("=" * 60 + "\n")

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
                        # Validate existing voice if --validate is enabled
                        if validate and validation_client is None:
                            validation_client = get_validation_client()
                        if validate and validation_client is not None:
                            if verbose:
                                print(f"    Validating existing voice...")
                            sample_text = DEFAULTS.get("static_voice_text", "")
                            is_valid, validation_msg = VoiceMapper.validate_voice_with_llm(
                                voice_path=generated[char_name],
                                description=char_desc,
                                sample_text=sample_text,
                                client=validation_client,
                                verbose=verbose
                            )
                            if not is_valid:
                                if verbose:
                                    print(f"    Validation failed: {validation_msg}, regenerating...")
                                # Remove failed voice and continue to regeneration
                                del generated[char_name]
                                voice_found = False
                            else:
                                if verbose:
                                    print(f"    Validation passed for existing voice")
                        if voice_found:
                            continue

                # Generate and validate voice, retrying on validation failure
                max_retries = 50
                voice_accepted = False
                voice_temporarily_saved = False  # Track if we saved a "good enough" voice
                retry_count = 0

                # Cache config values to avoid repeated dict lookups in the loop
                gender_correction_enabled = VOICE_GENDER_CORRECTION.get("enable", True)
                pitch_threshold = VOICE_GENDER_CORRECTION.get("pitch_threshold_hz", 160.0)
                male_target_pitch = VOICE_GENDER_CORRECTION.get("male_target_pitch_hz", 130.0)
                female_target_pitch = VOICE_GENDER_CORRECTION.get("female_target_pitch_hz", 220.0)
                use_ttest = VOICE_GENDER_CORRECTION.get("use_ttest", True)
                ttest_alpha = VOICE_GENDER_CORRECTION.get("ttest_alpha", 0.05)
                male_ref_mean = VOICE_GENDER_CORRECTION.get("male_ref_mean_hz", 122.5)
                female_ref_mean = VOICE_GENDER_CORRECTION.get("female_ref_mean_hz", 210.0)
                plot_histogram = VOICE_GENDER_CORRECTION.get("plot_histogram", True)

                while not voice_accepted and retry_count < max_retries:
                    if retry_count > 0:
                        if verbose:
                            print(f"    Retrying generation (attempt {retry_count + 1}/{max_retries})...")

                    success, output_file, duration, is_valid, validation_msg = generate_voice_sample(
                        character_name=char_name,
                        description=char_desc,
                        output_dir=output_dir,
                        device=device,
                        max_new_tokens=max_tokens,
                        verbose=verbose,
                        validate=False,  # Disable LLM validation here - we do our own checks below
                        validation_client=validation_client
                    )

                    if not success:
                        retry_count += 1
                        continue

                    # STEP 1: Fix gender algorithmically first - cheaper than regenerating after LLM rejection
                    if gender_correction_enabled:
                        if verbose:
                            print(f"    Detecting gender from audio...")
                        correction_success, correction_msg, final_gender, final_pitch, confidence = correct_voice_gender(
                            audio_path=output_file,
                            description=char_desc,
                            threshold_hz=pitch_threshold,
                            male_target_pitch_hz=male_target_pitch,
                            female_target_pitch_hz=female_target_pitch,
                            verbose=verbose,
                            use_ttest=use_ttest,
                            alpha=ttest_alpha,
                            male_ref_mean=male_ref_mean,
                            female_ref_mean=female_ref_mean,
                            plot_histogram=plot_histogram,
                            histogram_dir=output_dir
                        )

                        if not correction_success:
                            if "Could not detect pitch" in correction_msg or "beyond correction range" in correction_msg.lower():
                                if verbose:
                                    print(f"    Gender correction failed: {correction_msg}, regenerating...")
                                os.remove(output_file)
                                retry_count += 1
                                continue
                            elif "Could not extract target gender" in correction_msg:
                                if verbose:
                                    print(f"    Warning: {correction_msg}, skipping gender correction")
                                gender_match = True  # No target gender means we can't fail on gender
                            else:
                                if verbose:
                                    print(f"    Gender correction failed: {correction_msg}")
                                os.remove(output_file)
                                retry_count += 1
                                continue

                        # Use the final_gender returned from correct_voice_gender (no need to re-detect)
                        target_gender = extract_gender_from_description(char_desc)
                        if target_gender is None:
                            gender_match = final_gender is not None
                            if verbose:
                                print(f"    No target gender in description, detected={final_gender}, assuming match")
                        else:
                            gender_match = (final_gender == target_gender)
                            if verbose:
                                print(f"    Gender check: detected={final_gender}, target={target_gender}, match={gender_match}")

                        if not gender_match:
                            if verbose:
                                print(f"    Gender still doesn't match after correction, regenerating...")
                            os.remove(output_file)
                            retry_count += 1
                            continue

                    # STEP 2: Run LLM validation (now that gender is corrected)
                    if validate and validation_client is not None:
                        if verbose:
                            print(f"    Running LLM validation...")
                        is_valid, validation_msg = VoiceMapper.validate_voice_with_llm(
                            voice_path=output_file,
                            description=char_desc,
                            sample_text=DEFAULTS.get("static_voice_text", ""),
                            client=validation_client,
                            verbose=verbose
                        )

                        if not is_valid:
                            should_save_temporarily = False
                            has_tone_match = False
                            try:
                                validation_data = json.loads(validation_msg)
                                age_match = validation_data.get("age_match", False)
                                clarity_match = validation_data.get("clarity_match", False)
                                has_tone_match = validation_data.get("tone_match", False)

                                # Core attributes: gender (from our check), age, clarity
                                if gender_match and age_match and clarity_match:
                                    should_save_temporarily = True
                                    if verbose:
                                        print(f"    Core attributes match (gender, age, clarity) - saving temporarily")
                                        print(f"    Tone match: {has_tone_match}, Emotion match: {validation_data.get('emotion_match')}")
                                        print(f"    Continuing search for better tone/emotion match...")
                            except (json.JSONDecodeError, KeyError):
                                if verbose:
                                    print(f"    Validation failed: {validation_msg}")

                            if should_save_temporarily:
                                temp_output_file = str(Path(output_file).with_suffix(".temp.wav"))
                                should_replace_temp = not voice_temporarily_saved or has_tone_match
                                if should_replace_temp:
                                    shutil.move(output_file, temp_output_file)
                                    generated[char_name] = temp_output_file
                                    voice_temporarily_saved = True
                                    if verbose:
                                        if has_tone_match:
                                            print(f"    Saved improved voice (4/5 match - has tone): {temp_output_file}")
                                        else:
                                            print(f"    Saved temporary voice: {temp_output_file}")
                                retry_count += 1
                                continue

                            # Full failure - delete and retry
                            os.remove(output_file)
                            retry_count += 1
                            continue

                    # Voice passed all checks (or validation disabled)
                    voice_accepted = True
                    generated[char_name] = output_file
                    if verbose:
                        print(f"    Accepted after {retry_count + 1} attempt(s)")

                if not voice_accepted:
                    if voice_temporarily_saved:
                        # Rename temp file to final .wav
                        temp_file = generated[char_name]
                        final_file = temp_file.replace(".temp.wav", ".wav")
                        if os.path.exists(temp_file):
                            os.rename(temp_file, final_file)
                            generated[char_name] = final_file
                            if verbose:
                                print(f"    Using best available voice: {final_file}")
                    else:
                        # Clean up any leftover .temp.wav file for this character
                        temp_file = os.path.join(output_dir, f"{char_name}.temp.wav")
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            if verbose:
                                print(f"    Cleaned up orphaned temp file: {temp_file}")

                        # Fallback: use existing narrator voice if available
                        narrator_voice = voice_mapper.get_voice_path("narrator")
                        if narrator_voice and os.path.exists(narrator_voice):
                            generated[char_name] = narrator_voice
                            if verbose:
                                print(f"    Using narrator voice as fallback: {narrator_voice}")
                        else:
                            failed.append(char_name)
                            if verbose:
                                print(f"    Failed: no valid voice found and no narrator voice available")
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
if __name__ == "__main__":
    main()