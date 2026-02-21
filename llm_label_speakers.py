#!/usr/bin/env python3
"""
Module to take a chapter and automate sending through LLM for labeling characters.
"""
import argparse
import os
import sys
import json
from collections import Counter
from openai import OpenAI

from config import LLM_SETTINGS, DEFAULTS

def add_quotes_around_keys(json_body):
    """For some json text, we don't have quotes around keys. Example:
    char_map : {1: "narrator", 2: "char1", 3: "char2", 4: "char3"}
    ->
    char_map : {"1": "narrator", "2": "char1", "3": "char2", "4": "char3"}    
    """
    entries = []
    for entry in json_body.replace("{","").replace("}","").split(","):
        k,v = entry.split(":")
        k = k.strip()
        v = v.strip()
        if '"' not in k:
            k = '"'+k+'"'
        if '"' not in v:
            v = '"'+v+'"'
        entries.append(k+": "+v)
    revised_json_body = "{"+", ".join(entries)+"}"
    return revised_json_body

def interpret_new_result(result, attempt_num):
    """Process result of new LLM query of the following format:
    {
  "speaker_map": {
    "1": "narrator",
    "2": "First character",
    "3": "Second Character",
  },
  "attributions": {
    "7": 2,
    "9": 2,
    "11": 2,
    "13": 3,
    "15": 3
  }
  -----
      return character_map, line_map
    """
    line_map = {}
    char_map = {}
    # Filter out markdown code blocks and combine lines
    filtered_result = [x for x in result if not x.startswith("```")]
    result_text = "\n".join(filtered_result)

    # Count how many valid JSON objects are in the response
    # If there are multiple, this is an error (LLM outputting duplicates)
    brace_positions = []
    brace_count = 0
    first_brace = result_text.find("{")
    if first_brace != -1:
        for i, char in enumerate(result_text[first_brace:], start=first_brace):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    brace_positions.append(i)

    # If we found more than one complete JSON object, raise an error
    if len(brace_positions) > 1:
        # Find where the second JSON starts
        first_json_end = brace_positions[0]
        second_json_start = result_text.find("{", first_json_end)
        raise ValueError(
            f"LLM returned duplicate JSON objects in attempt {attempt_num}. "
            f"First JSON ends at position {first_json_end}, "
            f"second JSON starts at position {second_json_start}. "
            f"Response preview: {result_text[:200]}..."
        )

    # Extract only the first valid JSON object
    if first_brace != -1 and brace_positions:
        end_pos = brace_positions[0] + 1
        result_text = result_text[first_brace:end_pos]

    json_result = json.loads(result_text)                
    # convert keys to int
    char_map = {int(k): v.lower().strip().replace("_"," ").replace("'","").split("/")[0].split(" (")[0] for k,v in json_result["speaker_map"].items()}
    # remove line_map entries that are invalid.
    for line_num_str, char_num in json_result["attributions"].items():
        if char_num in char_map.keys():
            if "-" in line_num_str:
                start, end = line_num_str.split("-")
                for line in range(int(start), int(end), 1):
                    line_map[line] = char_num
            else:
                line_map[int(line_num_str)] = char_num
    # merge same name indexes
    valid_characters = {}
    for idx, char in char_map.items():
        if char not in valid_characters.keys():
            valid_characters[char] = idx
        else:
            # replace duplicates in line_map
            print(f"Removing duplicate {idx}: {char}")
            line_map = {k: valid_characters[char] if (v==idx) else v for k,v in line_map.items() }
    # remove unused characters
    used_char_indexes = [1]+list(set(line_map.values())) # 1 added for narrator
    char_map = {k:v for k,v in char_map.items() if k in used_char_indexes}
    return char_map, line_map 

def interpret_result(result, attempt_num):
    """Process result of a LLM query of the following format:
-----
char_map : {"1": "narrator", "2": "First Character", "3": "Second Character"}
7:2
9:2
11:2
13:3
15:3
-----
      return character_map, line_map"""
    line_map = {}
    char_map = {}
    IN_CHARMAP = False
    MISSING_OPEN = False
    for line in result:
        if len(char_map) > 0:
            if ":" in line and not (line.startswith("#") or line.startswith("`") or line.startswith("*")):
                try:
                    this_line, speaker_num = line.split(":")
                    this_line = this_line.replace("Line ","").replace("Lines ","").replace("- ", "")
                    if "-" in this_line:
                        line_start, line_stop = this_line.split("-")
                        for x in range(int(line_start),int(line_stop)+1):
                            line_map[int(x)] = int(speaker_num)
                    else:
                        line_map[int(this_line)] = int(speaker_num)
                except:
                    print(f"INVALID SPEAKER FORMAT FROM LLM RUN {attempt_num}: {line}", file=sys.stderr)
        elif IN_CHARMAP:
            if MISSING_OPEN:
                if "{" in line:
                    MISSING_OPEN=False
                    json_body = line.strip()
            else:
                json_body = json_body+line.strip()
            if "}" in line:
                IN_CHARMAP=False
                print("TRYING MULTLINE CHARMAP:", json_body)
                try:
                    char_map = json.loads(json_body)
                except:
                    char_map = json.loads(add_quotes_around_keys(json_body))
        else:
            if ("char_map" in line) and ("{" in line) and ("}" in line):
                json_body = "{" + line.split("{")[1]
                try:
                    char_map = json.loads(json_body)
                except:
                    char_map = json.loads(add_quotes_around_keys(json_body))
            elif ("char_map" in line) and ("{" in line):
                IN_CHARMAP=True
                json_body = "{"
            elif "char_map" in line:
                IN_CHARMAP=True
                MISSING_OPEN=True
            # could eventually add a check for """json""" with unquoted keys.
    for k in char_map.keys():
        char_map[k] = (char_map[k].split("/")[0]).split(" (")[0].lower().strip().replace("‑","")
    char_map = {k.lower().strip() : v for k,v in char_map.items()}
    # convert keys to int
    char_map = {int(k): v for k,v in char_map.items()}
    # remove line_map entries that are invalid.
    line_map = {line_num: char_num for line_num, char_num in line_map.items() if char_num in char_map.keys()}
    return char_map, line_map

def merge_line_maps(line_maps, verbose=False):
    """Take multiple line maps and determine the most common mapping for each line.
    If there is only one value for a line, we will pick that value.
    If there are two values for a line, we'll pick the first.
    If there more than two values for a line pick the majority. If all different pick first.
    """
    merged_line_map = {}# line_maps[0].copy()
    if len(line_maps)>0:
        for line_map in line_maps:
            for line, speaker_num in line_map.items():
                if not (line in merged_line_map.keys()):
                    merged_line_map[line] = [speaker_num]
                else:
                    merged_line_map[line].append(speaker_num)
    if verbose:
        print("Merged Line Map:")
        print(merged_line_map)
    return { k: Counter(v).most_common(1)[0][0] for k,v in merged_line_map.items()}

def is_same_character_by_line_mapping(character_key, character, line_map, merged_character_map, merged_line_map):
    """
    Determine if two characters are actually the same based on their line mappings.
    Returns True if at least half of the lines in the current character's map
    go to the same speaker as the existing character.
    """
    # Get all lines for both characters
    current_char_lines = [line for line, char_key in line_map.items() 
                         if char_key == character_key]
    
    # Check how many of these lines map to the other_character_key in merged_line_map
    matching_lines = 0
    total_lines_to_check = len(current_char_lines)
    
    if total_lines_to_check == 0:
        return False, None
    unique_speaker_keys = merged_character_map.keys()
    for unique_speaker_key in unique_speaker_keys:
        to_match_lines = [line for line, key in merged_line_map.items() if key == unique_speaker_key ]
        matching_lines = set(to_match_lines).intersection(set(current_char_lines))
        # check if we have match
        if  len(matching_lines) > total_lines_to_check/2:
            print(f"Found '{character}' as alternative name for '{merged_character_map[unique_speaker_key]}' matching lines [{len(matching_lines)} / {total_lines_to_check}]")
            return True, unique_speaker_key
    return False, None

def compare_characters(character_name, other_character):
    """Check if two characters are likely the same based on name similarity."""
    if (character_name == other_character) or \
       (character_name in other_character) or \
       (other_character in character_name):
        return True
    else:
        return False
    
OLD_PROMPT_TXT = """
Prompt: Audiobook Dialogue Annotation Expert

You are an expert in audiobook dialogue annotation. Your task is to identify all speakers in a given chapter and provide detailed attribution for each quoted line.

Step-by-Step Instructions:

1. Character Identification: Scan the entire text and identify ALL characters (including narrator)
- Create a char_map with numbers starting from 1
- Include narrator as character 1
- Format: char_map : {1: "narrator", 2: "Character Name", ...}
- Use simple names when possible.

2. Quoted lines: Find each full line surrounded by double quotation marks.
- Find ALL lines that start AND end with double quotation marks ("). These will likely have multiple sentences each.

3. Speaker Attribution for Each Line
- For EACH line:
-- if the line is a quote:
--- print the line number : followed by n where n is the key for the appropriate character that speaks the quote.

Important Rules:
- Quote lines are lines that START and END with double quotes.
- Focus on narrative context to determine who is speaking
- Use surrounding text, character mentions, and narrative flow for attribution
- Focus on the dialog itself. Speakers will not refer to themselves.
- Make sure the conversations make sense for sequential text.

5. Example Output Format:
- char_map : {1: "narrator", 2:"Name", 3:"OtherName"}
- 2:2
- 4:3
- 10:2
- 11:2

Process:
- First identify ALL characters and create char_map
- Print the char_map in JSON format
- Scan for quoted lines
- For each quoted line, determine speaker based on context
- Output line number : speaker number

IMPORTANT:
- TAKE YOUR TIME AND PROCESS ALL QUOTED LINES INDIVIDUALLY.
- Report every line with a quote. There will be many times where thinking will have a range of lines. We need to process each quoted line and print the speaker for each line.
"""
PROMPT_TXT = """
# Role: Provide detailed character attribution annotations for dialogues within audiobook chapters.
## Goals
- Identify all speakers in a chapter, including the narrator, and provide detailed character attribution labels for each citation line.
## Constraints
- All speakers must be identified and numbered, with the narrator numbered 1.
- Quoted lines must begin and end with double quotes on the SAME physical line.
- You MUST capture and attribute EVERY line that starts with " and ends with " — even if the quote is very short (e.g. "No." or "Yes, Master.") OR very long.
- Skipping even one correctly-quoted line is unacceptable. Double-check at the end that nothing was missed.
- NEVER use ranges (82-83, 100-102, etc.). Every quoted line gets its own explicit line number.
- Determine the speaker based on the context of the narrative.
- Ensure that the dialogue flows logically within the continuous text.
## Skills
- Accurately identify all speakers in the text.
- Accurately identify the speaker of each quoted line.
- Navigate contextual ambiguity with deep analysis.
## Output format (exactly this, no extra text, NO DUPLICATES)
{
  "speaker_map": {"1": "narrator", "2": "Name", "3": "OtherName"},
  "attributions": {
    "7": 2,
    "9": 2,
    ...
  }
}
## IMPORTANT - DO NOT REPEAT OR DUPLICATE
- Output ONLY the JSON object above - nothing else
- Do not output the JSON twice
- Do not add any text before or after the JSON
- Do not include thinking/analysis after the JSON
- If you output duplicate JSON or any extra text, the pipeline will fail

## Workflow
1. Scan the text line by line and identify every line that both begins and ends with ".
2. The speaker for each such line is determined based on context.
3. After finishing, verify again that no qualifying line was missed.
4. Output the speaker map in proper JSON format. Simplify the names of the speakers.
5. Output the attributions exactly as shown - one explicit key per quoted line, no ranges, no comments.

Begin processing the chapter now.
"""
#TODO: Update prompt to ensre we give non numbered, first name driven naming for characters...And narrator for occasions where it is too difficult to determine.

#- Do NOT base attribution solely on the quote content itself
#- Print with final format in mind.
#- Do not stop until the full text is processed!
#- Do not summarize, go thought the entire text!
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label a chapter file by character. Speaker (Narrator) for non quoted lines. Speaker (char_name) for spoken lines.")
    parser.add_argument("-txt_file", help="Path to the EPUB file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose printing for debug.")
    parser.add_argument("--skip_llm", action="store_true", help="Skip call to LLM and just try to process files into character maps.")
    parser.add_argument("-num_llm_attempts", type=int, default=DEFAULTS["num_llm_attempts"], help="Number of llm attempts submitted.")
    parser.add_argument("--old_format", action="store_true", help="Use older format for LLM query and parsing.")
    parser.add_argument("-api_key", metavar="lm-studio", default=LLM_SETTINGS["api_key"], help="Provide custom api key.")
    parser.add_argument("-port", metavar="1234", default=LLM_SETTINGS["port"], help="Provide custom port for invference.")
    args = parser.parse_args()
    client = OpenAI(base_url="http://localhost:"+args.port+"/v1", api_key=args.api_key) # api_key can be any string as it's not used by LM Studio
    
    if not os.path.exists(args.txt_file):
        print("Invalid txt_file. Please specify a valid text file and retry.",args.txt_file, file=sys.stderr)
        exit()

    chapter_file_base, _ = os.path.splitext(args.txt_file)

    if not args.skip_llm:
        with open(args.txt_file,"r",  encoding='utf-8') as f:
            lines = f.readlines()
        # Define your chat messages
        if args.old_format:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."+OLD_PROMPT_TXT}]
        else:
            messages = [
                {"role": "system", "content": PROMPT_TXT}]
        [messages.append({"role": "user", "content": x}) for x in lines]

        for a, attempt in enumerate(range(args.num_llm_attempts)):
            # Send the chat completion request
            print(f"Processing attempt {a}")
            response = client.chat.completions.create(
                model="local-model",  # Use a placeholder model name or the specific ID from LM Studio
                messages=messages,
                temperature=0.7,
                #stream=True # Set to True for streaming responses
            ).choices[0].message
            try:
                if "</think>" in response.content:
                    thought_process, result = response.content.split("</think>")
                else:
                    result = response.content
                    thought_process = response.reasoning
            except:
                result = response.content
                thought_process = None
            # Save think files
            if thought_process is not None:
                with open(chapter_file_base+f".think.{a}.txt", "w", encoding='utf-8') as f:
                    f.write(thought_process)
            # Save result files
            with open(chapter_file_base+f".result.{a}.txt", "w", encoding='utf-8') as f:
                f.write(result)
    character_maps = []
    line_maps = []
    merged_character_map = {}
    alternate_names = {}
    for a, attempt in enumerate(range(args.num_llm_attempts)):
        try:
            with open(chapter_file_base+f".result.{a}.txt", "r", encoding='utf-8') as f:
                result = f.readlines()
            if args.old_format:
                character_map, line_map = interpret_result(result, a)
            else:
                character_map, line_map = interpret_new_result(result, a)
                print(character_map)
        except Exception as e:
            # Read and show the actual content for debugging
            with open(chapter_file_base + f".result.{a}.txt", "r", encoding='utf-8') as f:
                debug_content = f.read()
            print(f"Error parsing {chapter_file_base}.result.{a}.txt: {e}", file=sys.stderr)
            print("\n--- LLM Output (for debugging) ---", file=sys.stderr)
            print(debug_content, file=sys.stderr)
            print("--- End of LLM Output ---\n", file=sys.stderr)
            raise e
        # print(a)
        # print(character_map)
        used_characters = set(line_map.values())
        # print("used_characters", character_map)
        if len(merged_character_map)==0:
            merged_character_map = character_map.copy()
            # Audit for character that is actually used.
            for key, character in character_map.items():
                if key not in used_characters and character != "narrator":
                    print(f"Removing un-used character in first map:")
                    print(key)
                    print(f"[{key}]{character}")
                    del merged_character_map[key]
        else:
            key_remap = {}
            for key, character in character_map.items():
                existing_character=False
                for m_key, m_character in merged_character_map.items():
                    if compare_characters(character, m_character):
                        existing_character=True
                        key_remap[key] = m_key
                        if key == m_key:
                            print(f"Matched character with same key: [{key}] {character}->{m_character}") 
                        else:
                            print(f"Matched character with different key: [{key}->{m_key}] {character}->{m_character}")
                    elif m_character in alternate_names.keys():
                        # alternative name for same character
                        for i, (alt_character, alt_count) in enumerate(alternate_names[m_character]):
                            if character == alt_character:
                                existing_character=True
                                alternate_names[m_character][i] = (alt_character, alt_count+1)
                                key_remap[key] = m_key
                                if key == m_key:
                                    print(f"Matched alt_char with same key: [{key}] {character}->{m_character}") 
                                else:
                                    print(f"Matched alt_char with different key: [{key}->{m_key}] {character}->{m_character}")
                if not existing_character:
                    character_is_used = key in line_map.values()
                    if character_is_used:
                        # check to see if we align to an existing character in prior run based on the lines spoken.
                        found_match, m_key = is_same_character_by_line_mapping(key, character, line_map, merged_character_map, merge_line_maps(line_maps, args.verbose))
                        if found_match:
                            m_character = merged_character_map[m_key]
                            key_remap[key] = m_key
                            if key==m_key:
                                print(f"Alternate character with same key: [{key}->{m_key}] {character}->{m_character}.")
                            else:
                                print(f"Alternate character with different key: [{key}->{m_key}] {character}->{m_character}.")
                            if m_character in alternate_names.keys():
                                alternate_names[m_character].append((character,1))
                            else:
                                alternate_names[m_character] = [(character,1)]
                        else:
                            new_m_key = max(merged_character_map.keys())+1
                            merged_character_map[new_m_key] = character
                            key_remap[new_m_key]=new_m_key
                            print(f"New character with new key: [{new_m_key}] {character}")                            
                    else:
                        print(f"Unused character '{character}', will not be added.")
            # use key_remap on line_map
            print(key_remap)
            line_map = {k:key_remap[v] for k,v in line_map.items() if v in key_remap.keys()}
        line_maps.append(line_map)
    print("Alternate names", alternate_names)
    for alt_name, alt_map in alternate_names.items():
        for other_name, other_count in alt_map:
            if other_count >= (args.num_llm_attempts / 2):
                orig_index = next((k for k, v in merged_character_map.items() if v == alt_name), None)
                if orig_index is not None:
                    print(f"Remapping index {orig_index} : '{alt_name}'->'{other_name}' [{other_count}/{args.num_llm_attempts}]")
                    merged_character_map[orig_index] = other_name                
    if args.verbose:
        print(merged_character_map)
        print("line_maps:", len(line_maps))
    merged_line_map = merge_line_maps(line_maps, args.verbose)
    merged_line_map = dict(sorted(merged_line_map.items(), key=lambda x: int(x[0])))
    if args.verbose:
        print("Overall line map:")
        print(merged_line_map)
    with open(chapter_file_base+f".map.json", "w", encoding='utf-8') as f:
        f.write(json.dumps([merged_character_map, merged_line_map], indent=4))


# ============================================================================
# MODULE FUNCTIONS FOR GRADIO INTERFACE
# ============================================================================

from typing import Dict, Tuple, List


def label_speakers_in_file(
    txt_file: str,
    api_key: str,
    port: str,
    num_attempts: int = 10,
    old_format: bool = False,
    skip_llm: bool = False,
    verbose: bool = False
) -> Tuple[str, Dict, Dict]:
    """Label speakers in a chapter file using LLM.

    This is a simplified interface for calling speaker labeling directly
    from the Gradio UI without subprocess.

    Args:
        txt_file: Path to the chapter text file
        api_key: API key for the LLM (can be any string for LM Studio)
        port: Port for the LLM inference
        num_attempts: Number of LLM attempts to make
        old_format: Use older format for LLM query and parsing
        skip_llm: Skip call to LLM and just try to process existing files
        verbose: Print verbose output

    Returns:
        Tuple of (status_message, character_map, line_map)
    """
    try:
        from openai import OpenAI

        chapter_file_base, _ = os.path.splitext(txt_file)

        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key=api_key)

        if not skip_llm:
            with open(txt_file, "r", encoding='utf-8') as f:
                lines = f.readlines()

            # Define chat messages
            if old_format:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant." + OLD_PROMPT_TXT}]
            else:
                messages = [
                    {"role": "system", "content": PROMPT_TXT}]
            messages.extend([{"role": "user", "content": x} for x in lines])

            for a, attempt in enumerate(range(num_attempts)):
                # Send the chat completion request
                if verbose:
                    print(f"Processing attempt {a}")
                response = client.chat.completions.create(
                    model="local-model",
                    messages=messages,
                    temperature=0.7,
                ).choices[0].message
                try:
                    if "</think>" in response.content:
                        thought_process, result = response.content.split("</think>")
                    else:
                        result = response.content
                        thought_process = response.reasoning
                except:
                    result = response.content
                    thought_process = None
                # Save think files
                if thought_process is not None:
                    with open(chapter_file_base + f".think.{a}.txt", "w", encoding='utf-8') as f:
                        f.write(thought_process)
                # Save result files
                with open(chapter_file_base + f".result.{a}.txt", "w", encoding='utf-8') as f:
                    f.write(result)

        character_maps = []
        line_maps = []
        merged_character_map = {}
        alternate_names = {}

        for a in range(num_attempts):
            try:
                with open(chapter_file_base + f".result.{a}.txt", "r", encoding='utf-8') as f:
                    result = f.readlines()
                if old_format:
                    character_map, line_map = interpret_result(result, a)
                else:
                    character_map, line_map = interpret_new_result(result, a)
                    if verbose:
                        print(character_map)
            except Exception as e:
                # Read and show the actual content for debugging
                with open(chapter_file_base + f".result.{a}.txt", "r", encoding='utf-8') as f:
                    debug_content = f.read()
                print(f"Error parsing {chapter_file_base}.result.{a}.txt: {e}", file=sys.stderr)
                print("\n--- LLM Output (for debugging) ---", file=sys.stderr)
                print(debug_content, file=sys.stderr)
                print("--- End of LLM Output ---\n", file=sys.stderr)
                raise e

            used_characters = set(line_map.values())
            if len(merged_character_map) == 0:
                merged_character_map = character_map.copy()
                # Audit for character that is actually used.
                for key, character in character_map.items():
                    if key not in used_characters and character != "narrator":
                        if verbose:
                            print(f"Removing un-used character in first map:")
                            print(key)
                            print(f"[{key}]{character}")
                            del merged_character_map[key]
            else:
                key_remap = {}
                for key, character in character_map.items():
                    existing_character = False
                    for m_key, m_character in merged_character_map.items():
                        if compare_characters(character, m_character):
                            existing_character = True
                            key_remap[key] = m_key
                            if key == m_key:
                                if verbose:
                                    print(f"Matched character with same key: [{key}] {character}->{m_character}")
                            else:
                                if verbose:
                                    print(f"Matched character with different key: [{key}->{m_key}] {character}->{m_character}")
                        elif m_character in alternate_names.keys():
                            # alternative name for same character
                            for i, (alt_character, alt_count) in enumerate(alternate_names[m_character]):
                                if character == alt_character:
                                    existing_character = True
                                    alternate_names[m_character][i] = (alt_character, alt_count + 1)
                                    key_remap[key] = m_key
                                    if key == m_key:
                                        if verbose:
                                            print(f"Matched alt_char with same key: [{key}] {character}->{m_character}")
                                    else:
                                        if verbose:
                                            print(f"Matched alt_char with different key: [{key}->{m_key}] {character}->{m_character}")
                    if not existing_character:
                        character_is_used = key in line_map.values()
                        if character_is_used:
                            # check to see if we align to an existing character in prior run based on the lines spoken.
                            found_match, m_key = is_same_character_by_line_mapping(key, character, line_map, merged_character_map, merge_line_maps(line_maps, verbose))
                            if found_match:
                                m_character = merged_character_map[m_key]
                                key_remap[key] = m_key
                                if key == m_key:
                                    if verbose:
                                        print(f"Alternate character with same key: [{key}->{m_key}] {character}->{m_character}.")
                                else:
                                    if verbose:
                                        print(f"Alternate character with different key: [{key}->{m_key}] {character}->{m_character}.")
                                if m_character in alternate_names.keys():
                                    alternate_names[m_character].append((character, 1))
                            else:
                                new_m_key = max(merged_character_map.keys()) + 1
                                merged_character_map[new_m_key] = character
                                key_remap[new_m_key] = new_m_key
                                if verbose:
                                    print(f"New character with new key: [{new_m_key}] {character}")
                        else:
                            if verbose:
                                print(f"Unused character '{character}', will not be added.")
                # use key_remap on line_map
                if verbose:
                    print(key_remap)
                line_map = {k: key_remap[v] for k, v in line_map.items() if v in key_remap.keys()}
            line_maps.append(line_map)

        if verbose:
            print("Alternate names", alternate_names)
        for alt_name, alt_map in alternate_names.items():
            for other_name, other_count in alt_map:
                if other_count >= (num_attempts / 2):
                    orig_index = next((k for k, v in merged_character_map.items() if v == alt_name), None)
                    if orig_index is not None:
                        if verbose:
                            print(f"Remapping index {orig_index} : '{alt_name}'->'{other_name}' [{other_count}/{num_attempts}]")
                        merged_character_map[orig_index] = other_name

        merged_line_map = merge_line_maps(line_maps, verbose)
        merged_line_map = dict(sorted(merged_line_map.items(), key=lambda x: int(x[0])))

        if verbose:
            print("Overall line map:")
            print(merged_line_map)

        # Save the merged map
        with open(chapter_file_base + f".map.json", "w", encoding='utf-8') as f:
            f.write(json.dumps([merged_character_map, merged_line_map], indent=4))

        return f"Successfully labeled speakers in {txt_file}", merged_character_map, merged_line_map

    except Exception as e:
        error_msg = f"Error labeling speakers: {str(e)}"
        if verbose:
            print(error_msg)
        return error_msg, {}, {}