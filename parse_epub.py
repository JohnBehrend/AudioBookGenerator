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
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from demo.inference_from_file import VoiceMapper

# Voice to Text for validation
from difflib import SequenceMatcher
from faster_whisper import WhisperModel
import whisperx

# combine auido files
import glob
import pydub

# Filter audio files
from sidon_demo_app import denoise_speech
from scipy.io import wavfile
import pandas as pd

# garbage collection
import gc

# consistent seeding
from transformers import set_seed

def get_non_silent_audio_from_wavs(wav_filepath_list, min_silence_len=1250, silence_thresh=-60):
    """Remove silent audio from list of wave filepaths of wavs together. Return AudioSegement."""
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
            all_audio_segments = all_audio_segments+this_audio_segment
    return all_audio_segments

def load_json(filename):
    if os.path.exists(filename):
        with open(filename, "r",encoding="utf-8") as f:
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
def distill_string(input):
    return input.lower().replace("?","").replace(".", "").replace("-","").replace(";","").replace(",","").replace("!","")

def score_strings_pop(i_str, d_str, lookahead=5, postfix="and also with you"):
    # Ensure lookahead is non-negative
    lookahead = max(0, lookahead)
    prev_undetected=False
    results = []
    input_tokens = i_str.split(" ")
    detected_tokens = d_str.split(" ")
    diff_list = []
    for i, i_tok in enumerate(input_tokens):
        if i_tok in diff_list:
            detected=True
            this_idx = diff_list.index(i_tok)
            detected_tokens = diff_list[this_idx+1:] + detected_tokens 
            diff_list = diff_list[:this_idx]            
        else:
            detected = False
            if prev_undetected and len(diff_list)>0: # Just remove one token
                diff_list.pop(0)
            else:
                diff_list = []  # Reset for each input token
            
            if detected_tokens:  # Only process if tokens remain
                n = max(min(lookahead, len(detected_tokens)-len(diff_list)),0)  # Safe number of pops
                for j in range(n):
                    d_tok = detected_tokens.pop(0)
                    diff_list.append(d_tok)
                    # Check if current input token is in the popped tokens so far
                    if i_tok in diff_list:
                        detected = True
                        break
                if not detected:
                    prev_undetected = True
        
        diff_str = " ".join(diff_list)  # Join current token's popped tokens
        results.append((i, i_tok, diff_str, detected, " ".join(detected_tokens[:lookahead])))
    df_temp = pd.DataFrame(results, columns=["i", "i_tok", "diff", "found", "next_tokens"])
    last_valid_token_index = df_temp[df_temp["found"]==True]["i"].max()
    last_valid_token = df_temp[df_temp["i"]==last_valid_token_index]["i_tok"]
    if len(last_valid_token.values)==0:
        return 0, None
    else:
        return float(df_temp["found"].mean()) - 0.5 * (postfix not in d_str[-len(postfix):]), last_valid_token.values[0]

def parse_epub():
    parser = argparse.ArgumentParser(description="Parse an EPUB file into an array of chapters")
    parser.add_argument("epub_file", help="Path to the EPUB file")
    parser.add_argument("-voices_map", metavar="voices_map.json", help="Map voice numbers to corresponding wav audio files in demos/voices/*.wav")
    parser.add_argument("--speaker_histogram", action="store_true", help="Print out a histogram of speakers.")
    parser.add_argument("--by_chapter", action="store_true", help="Save a file per chapter in a new folder labled chapters")
    parser.add_argument("--resume",action="store_true", help="Try to resume crunching in the directory based on files present.")
    parser.add_argument("--alt_gpu",action="store_true", help="Use other gpu for processing.")
    parser.add_argument("--alt_order",action="store_true", help="Use same gpu but high chapters to low chapters for processing.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logging information")
    args = parser.parse_args()
    end_characters = ["?", ".", "-", ";", ",","!"]
    # Parse the EPUB file
    chapters = parse_epub_to_chapters(args.epub_file)
    
    if not chapters:
        print("No chapters found or error occurred")
        sys.exit(1)

    # Print each chapter (you can modify this to output in different formats)
    speaker_counts={}
    os.makedirs("./chapters", exist_ok=True)
    voice_mapper = VoiceMapper()

    if args.alt_gpu:
        target_device="cuda:1"
        torch.cuda.set_device(1)
    else:
        target_device="cuda:0"
        torch.cuda.set_device(0)
    cfg_scale=1.30
    model_path="Jmica/VibeVoice7B"#"tensorbanana/vibevoice-7b-no-llm-bf16"#"FabioSarracino/VibeVoice-Large-Q8""microsoft/VibeVoice-1.5B" 
    voices_map = None
    if args.voices_map is not None:
        voices_map = load_json(args.voices_map)
    if args.verbose:
        print ("chapter_voice_map:",voices_map)
    if args.alt_gpu or args.alt_order:
        chapter_iterator = reversed(list(enumerate(chapters)))
    else:    
        chapter_iterator =enumerate(chapters)

    for i, chapter in chapter_iterator:
        chapter_map = load_json(f"./chapters/chapter_{i}.map.json")
        if args.verbose:
            print(f"Chapter {i}")
        if chapter_map:
            character_map, line_map = chapter_map
            character_map = {int(k):v for k,v in character_map.items()}
            line_map = {int(k): v for k, v in line_map.items()}
            line_to_character_map = {k: character_map[v] for k, v in line_map.items()}
            if all(x in voices_map.keys() for x in line_to_character_map.values()):
                line_to_voice_map = {k: voices_map[v] for k,v in line_to_character_map.items()}
            else:
                print(f"Please fill in the following characters in the voices map from chapter {i}:")
                print(json.dumps({v: "" for k,v in line_to_character_map.items() if v not in voices_map.keys()}, indent=4))
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
    # TODO: Give unique characters individual seed values to distringuish!
    #"large-v2"
    short_text_postfix = "and also with you?".lower()
    postfix_detect_token = distill_string(short_text_postfix.strip().split(" ")[0])
    validation_model = whisperx.load_model("distil-medium.en", "cuda", compute_type="float16") # WhisperModel("tiny.en")
    # Re-initialize the processor for a new voice
    tts_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path, # model_path 
        torch_dtype=torch.bfloat16,
        device_map=target_device,
        attn_implementation="flash_attention_2",
    )
    tts_model.set_ddpm_inference_steps(num_steps=13)
    tts_model.eval()

    processor = VibeVoiceProcessor.from_pretrained(model_path)
    still_skip=True
    if args.alt_gpu or args.alt_order:
        chapter_iterator = reversed(list(enumerate(chapters)))
    else:    
        chapter_iterator =enumerate(chapters)

    for i, chapter in chapter_iterator:
        if args.resume:
            if os.path.exists(f"./chapters/chapter_{str(i).zfill(2)}.mp3"):
                print(f"Skipping chapter {str(i).zfill(2)}.",end="\r")
                continue
        voices_used =list(dict.fromkeys([chapter_obj.get_speaker() for chapter_obj in chapter]).keys())
        for voice in voices_used:
            if args.resume:
                already_generated = [int(x.split(".")[-2]) for x in glob.glob(f"./chapters/chapter_{str(i).zfill(2)}.*.wav") if not x.endswith(".tmp.wav")]
            else:
                already_generated = []
            for j, chapter_obj in enumerate(chapter):
                if voice != chapter_obj.get_speaker():
                    continue # skip if its a different voice
                if still_skip:
                    if j not in already_generated:
                        still_skip=False # Found point to resume from
                        print(f"\nResuming with chapter {i}.{j}.")
                    else:
                        print(f"Skipping chapter {str(i).zfill(2)}.{str(j).zfill(4)}", end="\r")
                        continue
                full_script=str(chapter_obj.text[0].upper()+chapter_obj.text[1:])
                # Fix halucination when " . . ." occurs
                full_script = re.sub(r"(\s\.)+", r".", full_script)
                short_text_flag = True#len(chapter_obj.text) < 30
                if short_text_flag: # always enable as a test
                    full_script = full_script +(" " if full_script[0] in end_characters else ". ")+ short_text_postfix
                ratio = 0.0
                max_ratio = 0.0
                retries = 0
                input_string = distill_string(full_script)
                print("INPUT:", input_string)
                set_seed(42)
                while ratio < 0.85 and retries < 5:
                    set_seed(42+retries)
                    # Prepare inputs for the model
                    inputs = processor(
                        text=["Speaker 1: ? "+full_script], # Wrap in list for batch processing
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
                        #use_exllama=True
                        # use_external_llm=False
                    )

                    # Save output (processor handles device internally)
                    output_path = f"./chapters/chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav"            
                    processor.save_audio(
                        outputs.speech_outputs[0], # First (and only) batch item
                        output_path=output_path,
                    )
                    del inputs
                    del outputs
                    if args.alt_gpu:
                        # Explicitly collect garbage (optional, but can help) on alternate gpu only
                        gc.collect()
                        # Clear the CUDA memory cache
                        torch.cuda.empty_cache()
                        #torch.cuda.synchronize()

                    # send through a cleaning ML algo
                    sample_rate, waveform = wavfile.read(output_path)
                    sample_rate, waveform = denoise_speech((sample_rate,waveform))
                    wavfile.write(output_path, sample_rate, waveform)
                    #print(f"Saved output to {output_path}")

                    audio = whisperx.load_audio(f"./chapters/chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav")
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
                    print(" ".join([color_word(word, score)+"#"*int(pause) for word, score, pause in zip(segments, scores, pauses)]))
                    detected_string = " ".join(segments)
                    # if short_text_flag:
                    #     input_string = input_string + short_text_postfix
                    # ratio = SequenceMatcher(None, input_string, detected_string).ratio() # quick ratio doesn't care about oder just set match
                    ratio, last_valid_token = score_strings_pop(input_string, detected_string, lookahead=5, postfix=distill_string(short_text_postfix))
                    # ratio, last_valid_token = score_and_clean_segements(input_string, segements, start_times, end_times, lookahead=5, postfix=short_text_postfix)
                    
                    print(str(j).zfill(4),", Attempt: ", retries+1, "Ratio: ", int(ratio*100),"Voice: ", voice)
                    if short_text_flag:
                        if (distill_string(short_text_postfix) in detected_string) and (postfix_detect_token in segments):
                            if detected_string.startswith(distill_string(short_text_postfix)):
                                print("POSTFIX DETECTED BUT ONLY POSTFIX! -> Ratio 0")
                                ratio = 0
                            else:
                                postfix_start_index = segments[::-1].index(postfix_detect_token)
                                clip_end1 = start_times[::-1][postfix_start_index]
                                if len(end_times) > postfix_start_index+1:
                                    clip_end2 = end_times[::-1][postfix_start_index+1]
                                else:
                                    clip_end2 = end_times[-1]
                                print(f"POSTFIX DETECTED CLIPPING to {clip_end1} - {clip_end2}")
                                #Trim the clip to no longer include the postfix string.
                                audio = pydub.AudioSegment.from_wav(f"./chapters/chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav")
                                trimmed_audio = audio[0:((clip_end1+clip_end2)*500)]
                                trimmed_audio.export(f"./chapters/chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav", format="wav")
                        else:
                            if ((last_valid_token is None) or (last_valid_token == "")):
                                print("POSTFIX UN-DETECTED and INVALID VALUES. SKIP.")
                            else:
                                lastvalid_index = segments[::-1].index(last_valid_token)
                                clip_end1 = end_times[::-1][lastvalid_index]
                                print(f"POSTFIX UN-DETECTED LAST VALID CLIPPING TO {last_valid_token} {clip_end1} ")
                                #Trim the clip to no longer include the postfix string.
                                audio = pydub.AudioSegment.from_wav(f"./chapters/chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav")
                                trimmed_audio = audio[0:(clip_end1*1000)]
                                trimmed_audio.export(f"./chapters/chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav", format="wav")
                    # break
                    if ratio > max_ratio:
                        max_ratio = ratio
                        if os.path.exists( f"./chapters/chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.wav"):
                            os.unlink(f"./chapters/chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.wav")
                        time.sleep(2) # make sure the file is closed by the time we rename it
                        os.rename(f"./chapters/chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav",
                                f"./chapters/chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.wav")
                    # input_ids', 'attention_mask', 'speech_input_mask', 'speech_tensors', 'speech_masks', 'parsed_scripts', 'all_speakers_list'
                    retries+=1
                if os.path.exists(f"./chapters/chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav"):
                    os.unlink(f"./chapters/chapter_{str(i).zfill(2)}.{str(j).zfill(4)}.tmp.wav")
                # break# chapter_obj
        wavs = glob.glob(f"./chapters/chapter_{str(i).zfill(2)}.*.wav")
        audio = get_non_silent_audio_from_wavs(wavs)
        audio.export(f"./chapters/chapter_{str(i).zfill(2)}.mp3", format="mp3")
        # remove the wav files
        [os.unlink(x) for x in wavs]
        # break # chapters
        if args.speaker_histogram:
            this_speaker = str(chapter_obj.get_speaker())
            if this_speaker in speaker_counts.keys():
                speaker_counts[this_speaker]+=1
            else:
                speaker_counts[this_speaker]=1
        # break
    print("\n".join([str(x) for x in sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)]))

if __name__ == "__main__":
    parse_epub()
