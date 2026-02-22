#!/usr/bin/env python3
"""
Utility functions shared across multiple modules.
"""

import json
import os
from pathlib import Path
from collections import Counter
from openai import OpenAI

from .config import LLM_SETTINGS


def get_characters_from_map_files(chapters_dir: Path) -> list:
    """Extract unique character names from map.json files.

    Args:
        chapters_dir: Path to the directory containing chapter map files

    Returns:
        Sorted list of unique character names
    """
    characters = set()

    map_files = sorted(chapters_dir.glob("*.map.json"))
    for map_file in map_files:
        try:
            with open(map_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Handle both list format [char_map, line_map] and dict format
            if isinstance(data, list) and len(data) >= 1:
                character_map = data[0]
            elif isinstance(data, dict):
                character_map = data.get("character_map", {})
            else:
                continue

            if isinstance(character_map, dict):
                for char_name in character_map.values():
                    if isinstance(char_name, str):
                        characters.add(char_name)
        except Exception:
            pass

    return sorted(list(characters))


def compare_characters(character_name: str, other_character: str) -> bool:
    """Check if two characters are likely the same based on name similarity.

    Uses simple substring matching to detect if one name contains the other
    or they are identical.

    Args:
        character_name: First character name to compare
        other_character: Second character name to compare

    Returns:
        True if the characters are likely the same, False otherwise
    """
    if character_name == other_character:
        return True

    lower1, lower2 = character_name.lower(), other_character.lower()

    # Check if one contains the other as a substring
    if lower1 in lower2 or lower2 in lower1:
        return True

    return False


def merge_line_maps(line_maps: list, verbose: bool = False) -> dict:
    """Take multiple line maps and determine the most common mapping for each line.

    If there is only one value for a line, we will pick that value.
    If there are two values for a line, we'll pick the first.
    If there are more than two values for a line, pick the majority. If all different pick first.

    Args:
        line_maps: List of line maps (each map is dict mapping line number to speaker key)
        verbose: Print verbose output

    Returns:
        Merged line map with most common speaker for each line
    """
    merged_line_map = {}
    if len(line_maps) > 0:
        for line_map in line_maps:
            for line, speaker_num in line_map.items():
                if not (line in merged_line_map.keys()):
                    merged_line_map[line] = [speaker_num]
                else:
                    merged_line_map[line].append(speaker_num)

    if verbose:
        print("Merged Line Map:")
        print(merged_line_map)

    return {k: Counter(v).most_common(1)[0][0] for k, v in merged_line_map.items()}


def get_llm_client(api_key: str, port: str) -> OpenAI:
    """Create and return an OpenAI client for LM Studio.

    Args:
        api_key: API key for the LLM (can be any string for LM Studio)
        port: Port for the LLM inference

    Returns:
        OpenAI client configured for LM Studio
    """
    return OpenAI(base_url=f"http://localhost:{port}/v1", api_key=api_key)
