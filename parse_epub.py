#!/usr/bin/env python3
"""
Simple Python script to parse an EPUB file into an array of chapters.
"""

import argparse
import sys
import os
import time
import json
import re

from parse_chapter import parse_epub_to_chapters

# Text to speach generation
import torch

# TTS Engine configuration - set this to use 'vibevoice' or 'kugelaudio'
TTS_ENGINE = os.environ.get('TTS_ENGINE', 'kugelaudio')

if TTS_ENGINE == 'kugelaudio':
    # Add kugelaudio-open to path (located at ../kugelaudio-open/src)
    kugelaudio_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kugelaudio-open', 'src')
    if kugelaudio_path not in sys.path:
        sys.path.insert(0, kugelaudio_path)

    from kugelaudio_open.processors.kugelaudio_processor import KugelAudioProcessor
    from kugelaudio_open.models.kugelaudio_inference import KugelAudioForConditionalGenerationInference
else:
    # VibeVoice imports
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference


# Voice to Text for validation
from difflib import SequenceMatcher
from faster_whisper import WhisperModel
import whisperx

# combine audio files
import glob
import pydub

# Filter audio files - try to import from sidon_demo_app, provide fallback
try:
    from sidon_demo_app import denoise_speech
except ImportError:
    # Fallback: simple noise reduction using scipy
    from scipy.io import wavfile
    import numpy as np
    from scipy.signal import wiener

    def denoise_speech(audio_data):
        """Simple fallback denoising using Wiener filter."""
        sample_rate, waveform = audio_data
        # Apply Wiener filter for noise reduction
        denoised = wiener(waveform.astype(np.float32))
        return sample_rate, denoised.astype(np.int16)
else:
    from scipy.io import wavfile
import pandas as pd

# garbage collection
import gc

# consistent seeding
from transformers import set_seed


def get_non_silent_audio_from_wavs(wav_filepath_list, min_silence_len=1250, silence_thresh=-60):
    """Remove silent audio from list of wave filepaths of wavs together. Return AudioSegment."""
    all_audio_segments = None
    for wav in wav_filepath_list:
        raw_audio_segment = pydub.AudioSegment.from_wav(wav)
        # remove silence
        this_audio_segment = pydub.AudioSegment.empty()
        for (start_time, end_time) in pydub.silence.detect_nonsilent(raw_audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh):
            this_audio_segment += raw_audio_segment[start_time:end_time]
        if all_audio_segments is None:
            all_audio_segments = this_audio_segment
        else:
            all_audio_segments = all_audio_segments + this_audio_segment
    return all_audio_segments


def load_json(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return None


def color_word(word, score):
    """
    Annotates terminal color strings around word.
    The gradient progresses from red (score=0) to green (score=1).
    """
    reset_code = "\033[0m"
    red = int(255 * (1 - score))
    green = int(255 * score)
    blue = 0

    color_code = f"\033[38;2;{red};{green};{blue}m"
    return f"{color_code}{word}{reset_code}"


def distill_string(input_str):
    return input_str.lower().replace("?", "").replace(".", "").replace("-", "").replace(";", "").replace(",", "").replace("!", "")


def score_strings_pop(i_str, d_str, lookahead=5, postfix="and also with you"):
    # Ensure lookahead is non-negative
    lookahead = max(0, lookahead)
    prev_undetected = False
    results = []
    input_tokens = i_str.split(" ")
    detected_tokens = d_str.split(" ")
    diff_list = []
    for i, i_tok in enumerate(input_tokens):
        if i_tok in diff_list:
            detected = True
            this_idx = diff_list.index(i_tok)
            detected_tokens = diff_list[this_idx+1:] + detected_tokens
            diff_list = diff_list[:this_idx]
        else:
            detected = False
            if prev_undetected and len(diff_list) > 0:
                diff_list.pop(0)
            else:
                diff_list = []

            if detected_tokens:
                n = max(min(lookahead, len(detected_tokens) - len(diff_list)), 0)
                for j in range(n):
                    d_tok = detected_tokens.pop(0)
                    diff_list.append(d_tok)
                    if i_tok in diff_list:
                        detected = True
                        break
                if not detected:
                    prev_undetected = True

        diff_str = " ".join(diff_list)
        results.append((i, i_tok, diff_str, detected, " ".join(detected_tokens[:lookahead])))
    df_temp = pd.DataFrame(results, columns=["i", "i_tok", "diff", "found", "next_tokens"])
    last_valid_token_index = df_temp[df_temp["found"] == True]["i"].max()
    last_valid_token = df_temp[df_temp["i"] == last_valid_token_index]["i_tok"]
    if len(last_valid_token.values) == 0:
        return 0, None
    else:
        return float(df_temp["found"].mean()) - 0.5 * (postfix not in d_str[-len(postfix):]), last_valid_token.values[0]


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


def parse_epub():
    parser = argparse.ArgumentParser(description="Parse an EPUB file into an array of chapters")
    parser.add_argument("epub_file", help="Path to the EPUB file")
    parser.add_argument("-voices_map", metavar="voices_map.json", help="Map voice numbers to corresponding wav audio files in demos/voices/*.wav")
    parser.add_argument("--speaker_histogram", action="store_true", help="Print out a histogram of speakers.")
    parser.add_argument("--by_chapter", action="store_true", help="Save a file per chapter in a new folder labled chapters")
    parser.add_argument("--resume", action="store_true", help="Try to resume crunching in the directory based on files present.")
    parser.add_argument("--alt_gpu", action="store_true", help="Use other gpu for processing.")
    parser.add_argument("--alt_order", action="store_true", help="Use same gpu but high chapters to low chapters for processing.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logging information")
    parser.add_argument("--tts-engine", default="kugelaudio", choices=["kugelaudio", "vibevoice"], help="TTS engine to use")
    parser.add_argument("--output-dir", metavar="DIR", help="Output directory for chapter files (default: script_dir/chapters)")
    parser.add_argument("--max-chapters", type=int, metavar="N", help="Maximum number of chapters to parse (default: all)")
    args = parser.parse_args()

    # Override TTS_ENGINE with command line argument
    global TTS_ENGINE
    TTS_ENGINE = args.tts_engine

    end_characters = ["?", ".", "-", ";", ",", "!"]
    # Parse the EPUB file
    chapters = parse_epub_to_chapters(args.epub_file, max_chapters=args.max_chapters)

    if not chapters:
        print("No chapters found or error occurred")
        sys.exit(1)

    # Print each chapter (you can modify this to output in different formats)
    speaker_counts = {}
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "chapters")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[OUTPUT_DIR] Using output directory: {output_dir}", flush=True)
    voice_mapper = VoiceMapper()

    if args.alt_gpu:
        target_device = "cuda:1"
        torch.cuda.set_device(1)
    else:
        target_device = "cuda:0"
        torch.cuda.set_device(0)

    cfg_scale = 1.30

    if TTS_ENGINE == 'kugelaudio':
        model_path = "kugelaudio/kugelaudio-0-open"
    else:
        model_path = "Jmica/VibeVoice7B"

    voices_map = None
    if args.voices_map is not None:
        voices_map = load_json(args.voices_map)
    else:
        # Try to load voices_map.json from output directory (auto-generated from voice samples)
        voices_map_path = os.path.join(output_dir, "voices_map.json")
        voices_map = load_json(voices_map_path)

    if args.verbose:
        print("chapter_voice_map:", voices_map)

    if args.alt_gpu or args.alt_order:
        chapter_iterator = reversed(list(enumerate(chapters)))
    else:
        chapter_iterator = enumerate(chapters)

    for i, chapter in chapter_iterator:
        chapter_map = load_json(os.path.join(output_dir, f"chapter_{i}.map.json"))
        if args.verbose:
            print(f"Chapter {i}")
        if chapter_map:
            character_map, line_map = chapter_map
            character_map = {int(k): v for k, v in character_map.items()}
            line_map = {int(k): v for k, v in line_map.items()}
            line_to_character_map = {k: character_map[v] for k, v in line_map.items()}
            if voices_map is None:
                print(f"Error: voices_map is None. Please provide a voices map file using --voices_map.")
                exit()
            if all(x in voices_map.keys() for x in line_to_character_map.values()):
                line_to_voice_map = {k: voices_map[v] for k, v in line_to_character_map.items()}
            else:
                print(f"Please fill in the following characters in the voices map from chapter {i}:")
                print(json.dumps({v: "" for k, v in line_to_character_map.items() if v not in voices_map.keys()}, indent=4))
                exit()
            for cobj in chapter:
                if cobj.has_quotes:
                    if cobj.line_num in line_map.keys():
                        if args.verbose:
                            print(f"Line {cobj.line_num} -> {line_to_character_map[cobj.line_num]} -> {line_to_voice_map[cobj.line_num]}")
                        cobj.set_speaker(line_to_voice_map[cobj.line_num])
                    else:
                        print(f"Chapter {i} line {cobj.line_num} -> narrator even though this is a quote.", file=sys.stderr)
                        cobj.set_speaker(voices_map["narrator"])
                else:
                    cobj.set_speaker(voices_map["narrator"])

    # TODO: Give unique characters individual seed values to distinguish!
    short_text_postfix = "and also with you?".lower()
    postfix_detect_token = distill_string(short_text_postfix.strip().split(" ")[0])
    validation_model = whisperx.load_model("distil-medium.en", "cuda", compute_type="float16")

    if TTS_ENGINE == 'kugelaudio':
        # Initialize KugelAudio model and processor
        tts_model = KugelAudioForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=target_device,
            attn_implementation="flash_attention_2",
        )
        tts_model.set_ddpm_inference_steps(num_steps=13)
        tts_model.eval()

        processor = KugelAudioProcessor.from_pretrained(model_path)
    else:
        # Initialize VibeVoice model and processor
        tts_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=target_device,
            attn_implementation="flash_attention_2",
        )
        tts_model.set_ddpm_inference_steps(num_steps=13)
        tts_model.eval()

        processor = VibeVoiceProcessor.from_pretrained(model_path)

    still_skip = True
    if args.alt_gpu or args.alt_order:
        chapter_iterator = reversed(list(enumerate(chapters)))
    else:
        chapter_iterator = enumerate(chapters)

    for i, chapter in chapter_iterator:
        if args.resume:
            if os.path.exists(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.mp3")):
                print(f"Skipping chapter {str(i).zfill(2)}.", end="\r")
                continue

        print(f"[CHAPTER_START] Chapter {i}/{len(chapters)}", flush=True)

        voices_used = list(dict.fromkeys([chapter_obj.get_speaker() for chapter_obj in chapter]).keys())

        for voice in voices_used:
            if args.resume:
                already_generated = [int(x.split(".")[-2]) for x in glob.glob(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.*.wav")) if not x.endswith(".tmp.wav")]
            else:
                already_generated = []

            for j, chapter_obj in enumerate(chapter):
                if voice != chapter_obj.get_speaker():
                    continue

                if still_skip:
                    if j not in already_generated:
                        still_skip = False
                        print(f"\nResuming with chapter {i}.{j}.")
                    else:
                        print(f"Skipping chapter {str(i).zfill(2)}.{str(j).zfill(4)}", end="\r")
                        continue

                full_script = str(chapter_obj.text[0].upper() + chapter_obj.text[1:])
                full_script = re.sub(r"(\s\.)+", r".", full_script)

                short_text_flag = True
                if short_text_flag:
                    full_script = full_script + (" " if full_script[0] in end_characters else ". ") + short_text_postfix

                ratio = 0.0
                max_ratio = 0.0
                retries = 0
                input_string = distill_string(full_script)
                print("INPUT:", input_string)
                set_seed(42)

                while ratio < 0.85 and retries < 5:
                    set_seed(42 + retries)

                    if TTS_ENGINE == 'kugelaudio':
                        # Use KugelAudio processor with voice prompt
                        voice_path = voice_mapper.get_voice_path(voice)
                        inputs = processor(
                            text=full_script,
                            voice_prompt=voice_path,
                            padding=True,
                            return_tensors="pt",
                        )
                    else:
                        # Use VibeVoice processor with voice_samples
                        inputs = processor(
                            text=["Speaker 1: ? " + full_script],
                            voice_samples=[voice_mapper.get_voice_path(voice)],
                            padding=True,
                            return_tensors="pt",
                            return_attention_mask=True,
                        )

                    for k, v in inputs.items():
                        if torch.is_tensor(v):
                            inputs[k] = v.to(target_device)

                    outputs = tts_model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=processor.tokenizer,
                        do_sample=False,
                        verbose=False
                    )

                    output_path = os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav")

                    if TTS_ENGINE == 'kugelaudio':
                        # KugelAudio returns audio directly in speech_outputs
                        processor.save_audio(
                            outputs.speech_outputs[0],
                            output_path=output_path,
                        )
                    else:
                        # VibeVoice also returns audio in speech_outputs
                        processor.save_audio(
                            outputs.speech_outputs[0],
                            output_path=output_path,
                        )

                    del inputs
                    del outputs

                    if args.alt_gpu:
                        gc.collect()
                        torch.cuda.empty_cache()

                    # send through a cleaning ML algo
                    sample_rate, waveform = wavfile.read(output_path)
                    sample_rate, waveform = denoise_speech((sample_rate, waveform))
                    wavfile.write(output_path, sample_rate, waveform)

                    audio = whisperx.load_audio(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav"))
                    result = validation_model.transcribe(audio, batch_size=1)
                    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
                    result = whisperx.align(result["segments"], model_a, metadata, audio, "cuda", return_char_alignments=False)

                    prev_end = None
                    pauses = []
                    for segment in result["word_segments"]:
                        if prev_end is not None:
                            pauses.append(segment["start"] - prev_end)
                        prev_end = segment["end"]
                    pauses.append(0)

                    segments = [distill_string(s["word"]) for s in result["word_segments"]]
                    scores = [s["score"] for s in result["word_segments"]]
                    start_times = [s["start"] for s in result["word_segments"]]
                    end_times = [s["end"] for s in result["word_segments"]]
                    print(" ".join([color_word(word, score) + "#" * int(pause) for word, score, pause in zip(segments, scores, pauses)]))

                    detected_string = " ".join(segments)

                    ratio, last_valid_token = score_strings_pop(input_string, detected_string, lookahead=5, postfix=distill_string(short_text_postfix))

                    print(f"[LINE_PROGRESS] Chapter {i}, Line {j+1}/{len(chapter)}, Attempt {retries + 1}, Ratio: {int(ratio * 100)}, Voice: {voice}", flush=True)

                    if short_text_flag:
                        if (distill_string(short_text_postfix) in detected_string) and (postfix_detect_token in segments):
                            if detected_string.startswith(distill_string(short_text_postfix)):
                                print("POSTFIX DETECTED BUT ONLY POSTFIX! -> Ratio 0")
                                ratio = 0
                            else:
                                postfix_start_index = segments[::-1].index(postfix_detect_token)
                                clip_end1 = start_times[::-1][postfix_start_index]
                                if len(end_times) > postfix_start_index + 1:
                                    clip_end2 = end_times[::-1][postfix_start_index + 1]
                                else:
                                    clip_end2 = end_times[-1]
                                print(f"POSTFIX DETECTED CLIPPING to {clip_end1} - {clip_end2}")
                                audio = pydub.AudioSegment.from_wav(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav"))
                                trimmed_audio = audio[0:((clip_end1 + clip_end2) * 500)]
                                trimmed_audio.export(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav"), format="wav")
                        else:
                            if ((last_valid_token is None) or (last_valid_token == "")):
                                print("POSTFIX UN-DETECTED and INVALID VALUES. SKIP.")
                            else:
                                lastvalid_index = segments[::-1].index(last_valid_token)
                                clip_end1 = end_times[::-1][lastvalid_index]
                                print(f"POSTFIX UN-DETECTED LAST VALID CLIPPING TO {last_valid_token} {clip_end1} ")
                                audio = pydub.AudioSegment.from_wav(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav"))
                                trimmed_audio = audio[0:(clip_end1 * 1000)]
                                trimmed_audio.export(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav"), format="wav")

                    if ratio > max_ratio:
                        max_ratio = ratio
                        if os.path.exists(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.wav")):
                            os.unlink(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.wav"))
                        time.sleep(2)
                        os.rename(
                            os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav"),
                            os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.wav")
                        )

                    retries += 1

                if os.path.exists(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav")):
                    os.unlink(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav"))

        wavs = glob.glob(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.*.wav"))
        audio = get_non_silent_audio_from_wavs(wavs)
        audio.export(os.path.join(output_dir, f"chapter_{str(i).zfill(2)}.mp3"), format="mp3")
        [os.unlink(x) for x in wavs]

        print(f"[CHAPTER_COMPLETE] Chapter {i}/{len(chapters)}", flush=True)

        if args.speaker_histogram:
            this_speaker = str(chapter_obj.get_speaker())
            if this_speaker in speaker_counts.keys():
                speaker_counts[this_speaker] += 1
            else:
                speaker_counts[this_speaker] = 1

    print("\n".join([str(x) for x in sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)]))


if __name__ == "__main__":
    parse_epub()