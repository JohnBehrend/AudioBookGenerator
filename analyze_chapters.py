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


def load_chapter_map(file_path: str) -> Tuple[Dict, Dict]:
    """Load a chapter map JSON file and return character_map and line_map."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[0], data[1]  # (character_map, line_map)
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return {}, {}


def extract_speaker_info(chapter_num: int, character_map: Dict, line_map: Dict) -> Dict:
    """Extract speaker information for a specific chapter."""
    speakers = {}
    for line_num, speaker_key in line_map.items():
        speaker_name = character_map.get(speaker_key, f"Unknown_{speaker_key}")
        if speaker_name not in speakers:
            speakers[speaker_name] = []
        speakers[speaker_name].append(line_num)
    return speakers


def analyze_directory(directory: str, args) -> None:
    """Analyze all chapter map files in the given directory."""
    directory = Path(directory)

    # Find all chapter_*.map.json files
    chapter_files = sorted(directory.glob("chapter_*.map.json"))

    if not chapter_files:
        print(f"No chapter map files found in {directory}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(chapter_files)} chapter files in {directory}\n")

    # Data storage
    all_speakers = []
    chapter_stats = []
    combined_character_map = {}
    combined_line_map = defaultdict(list)

    # Analyze each chapter
    for chapter_file in chapter_files:
        chapter_num = int(chapter_file.stem.split('_')[1])
        character_map, line_map = load_chapter_map(str(chapter_file))

        if not character_map:
            continue

        # Extract speaker info for this chapter
        chapter_speakers = extract_speaker_info(chapter_num, character_map, line_map)
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
            'speakers': chapter_speakers_list
        })

        # Combine with overall data
        combined_character_map.update(character_map)
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
    for line_map in [s['line_map'] for s in [extract_speaker_info(i, c_map, l_map) for i, (c_map, l_map) in
                                                 [(ch['chapter'], (chapter_stats[i]['speakers'], {l: s for l, s in zip(
                                                     [k for k, v in extract_speaker_info(ch['chapter'], chapter_stats[i]['character_map'],
                                                                 chapter_stats[i]['line_map'].keys(), chapter_stats[i]['line_map'].values()) for
                                                     k, v in chapter_stats[i]['line_map'].items()}))} for i, ch in enumerate(chapter_stats)]
                                                 for i, ch in enumerate(chapter_stats)]:
        for speaker_key in line_map.values():
            speaker_name = combined_character_map.get(speaker_key, f"Unknown_{speaker_key}")
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
