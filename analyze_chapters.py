#!/usr/bin/env python3
"""
Script to analyze chapter map files in a directory.
Processes all chapter_*.map.json files and generates statistics.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Import the chapter file finder from list_chapters.py
from list_chapters import find_chapter_files


def load_chapter_map(file_path: str) -> Tuple[Dict, Dict]:
    """Load a chapter map JSON file and return character_map and line_map."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[0], data[1]  # (character_map, line_map)
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return {}, {}


def extract_speaker_info(chapter_num: int, character_map: Dict, line_map: Dict, key_to_names: Dict[int, List[str]]) -> Dict:
    """Extract speaker information for a specific chapter."""
    speakers = {}
    for line_num, speaker_key in line_map.items():
        speaker_name = get_preferred_name_for_key(speaker_key, key_to_names)
        if speaker_name not in speakers:
            speakers[speaker_name] = []
        speakers[speaker_name].append(line_num)
    return speakers


def get_character_key_mapping(chapter_files: List[str]) -> Dict[int, List[str]]:
    """
    Load all character maps from chapter files and create a mapping of
    which character names can belong to each key number.
    Returns a dict: {key: [name1, name2, name3, ...]}
    """
    key_to_names = {}
    for chapter_file in chapter_files:
        chapter_num = int(chapter_file.stem.split('_')[1].split('.')[0])
        character_map, _ = load_chapter_map(str(chapter_file))
        if character_map:
            for k, v in character_map.items():
                key_num = int(k)
                if key_num not in key_to_names:
                    key_to_names[key_num] = []
                # Add name if not already in the list (case-insensitive)
                name_lower = v.lower().strip()
                if not any(n.lower().strip() == name_lower for n in key_to_names[key_num]):
                    key_to_names[key_num].append(v)
    return key_to_names


def get_preferred_name_for_key(key: int, key_to_names: Dict[int, List[str]]) -> str:
    """
    Get the preferred name for a key. Uses the narrator (key 1) and
    then the most frequently appearing name, or returns all names if ambiguous.
    """
    if key in key_to_names and len(key_to_names[key]) > 0:
        # Priority 1: Narrator (key 1) should always be "narrator"
        if key == 1 and "narrator" in key_to_names[key]:
            return "narrator"
        # Priority 2: If only one name, use it
        if len(key_to_names[key]) == 1:
            return key_to_names[key][0]
        # Priority 3: Use the most common name (if we had counts)
        # For now, pick the first name or return all names
        # We'll return all names if there's ambiguity
        return ", ".join(key_to_names[key])
    return f"Unknown_{key}"


def analyze_directory(directory: str, args) -> None:
    """Analyze all chapter map files in the given directory."""
    directory = Path(directory)

    # Find all chapter_*.map.json files using the function from list_chapters.py
    chapter_files = find_chapter_files(str(directory), verbose=args.verbose)

    if not chapter_files:
        print(f"No chapter map files found in {directory}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(chapter_files)} chapter files in {directory}\n")

    # Load all character maps upfront to get complete speaker names
    key_to_names = get_character_key_mapping(chapter_files)

    # Data storage
    all_speakers = []
    chapter_stats = []
    combined_line_map = defaultdict(list)

    # Analyze each chapter
    for chapter_file in chapter_files:
        # Extract chapter number from filename (e.g., chapter_0.map.json -> 0)
        chapter_num = int(chapter_file.stem.split('_')[1].split('.')[0])
        character_map, line_map = load_chapter_map(str(chapter_file))

        if not character_map:
            continue

        # Extract speaker info for this chapter using key_to_names for names
        chapter_speakers = extract_speaker_info(chapter_num, character_map, line_map, key_to_names)
        chapter_speakers_list = list(chapter_speakers.keys())
        all_speakers.extend(chapter_speakers_list)

        # Statistics for this chapter
        total_lines = len(line_map)
        unique_speakers = len(chapter_speakers)
        narrator_lines = sum(1 for spk in line_map.values() if spk == 1)
        dialogue_lines = total_lines - narrator_lines

        chapter_stats.append({
            'chapter': chapter_num,
            'total_lines': total_lines,
            'narrator_lines': narrator_lines,
            'dialogue_lines': dialogue_lines,
            'unique_speakers': unique_speakers,
            'speakers': chapter_speakers_list,
            'line_map': line_map,
            'character_map': character_map
        })

        # Combine with overall data
        for line_num, speaker_key in line_map.items():
            combined_line_map[line_num].append(speaker_key)

    # Generate overall statistics
    print("=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"Total Chapters Analyzed: {len(chapter_files)}")
    print(f"Total Lines: {len(combined_line_map)}")
    print(f"Unique Speakers Across All Chapters: {len(all_speakers)}")
    print(f"Narrator Lines: {sum(1 for spk in combined_line_map.values() if spk == 1)}")
    print(f"Dialogue Lines: {len(combined_line_map) - sum(1 for spk in combined_line_map.values() if spk == 1)}")

    print("\n" + "=" * 60)
    print("TOP SPEAKERS BY LINE COUNT")
    print("=" * 60)

    # Count lines per speaker across all chapters
    speaker_line_counts = Counter()

    # Iterate through each chapter's data and count speaker lines
    for chapter in chapter_stats:
        # Get the line_map from this chapter
        line_map = chapter['line_map']

        # Count each speaker's lines using key_to_names for names
        for speaker_key in line_map.values():
            speaker_name = get_preferred_name_for_key(speaker_key, key_to_names)
            speaker_line_counts[speaker_name] += 1

    top_speakers = speaker_line_counts.most_common(10)
    for rank, (speaker, count) in enumerate(top_speakers, 1):
        percentage = (count / len(combined_line_map)) * 100
        print(f"{rank}. {speaker}: {count} lines ({percentage:.1f}%)")

    print("\n" + "=" * 60)
    print("PER CHAPTER STATISTICS")
    print("=" * 60)

    for stat in chapter_stats:
        print(f"\nChapter {stat['chapter']}:")
        print(f"  Total Lines: {stat['total_lines']}")
        print(f"  Narrator Lines: {stat['narrator_lines']}")
        print(f"  Dialogue Lines: {stat['dialogue_lines']}")
        print(f"  Unique Speakers: {stat['unique_speakers']}")
        print(f"  Top Speakers: {', '.join(stat['speakers'][:5])}")

    # Optional output
    if args.json_output:
        output_data = {
            'total_chapters': len(chapter_files),
            'total_lines': len(combined_line_map),
            'unique_speakers': len(all_speakers),
            'speaker_counts': {s: int(c) for s, c in speaker_line_counts.most_common()},
            'chapter_stats': chapter_stats
        }

        output_path = Path(directory) / "chapter_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nFull analysis saved to: {output_path}")

    if args.csv_output:
        import csv
        csv_path = Path(directory) / "chapter_stats.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['chapter', 'total_lines', 'narrator_lines',
                                                  'dialogue_lines', 'unique_speakers', 'speakers'])
            writer.writeheader()
            writer.writerows(chapter_stats)
        print(f"CSV statistics saved to: {csv_path}")

    if args.verbose:
        print("\n" + "=" * 60)
        print("ALL SPEAKERS DETECTED")
        print("=" * 60)
        for speaker in sorted(all_speakers):
            print(f"  - {speaker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze chapter map JSON files and generate statistics"
    )
    parser.add_argument(
        "directory",
        help="Directory containing chapter_*.map.json files"
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Generate detailed JSON output file"
    )
    parser.add_argument(
        "--csv-output",
        action="store_true",
        help="Generate CSV file with chapter statistics"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed speaker list"
    )
    parser.add_argument(
        "--chapter-filter",
        type=int,
        help="Filter to specific chapter number (e.g., --chapter-filter 1)"
    )
    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
        sys.exit(1)

    analyze_directory(args.directory, args)
