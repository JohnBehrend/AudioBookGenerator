#!/usr/bin/env python3
"""
Script to analyze chapter map files in a directory.
Processes all chapter_*.map.json files and generates statistics.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from list_chapters import find_chapter_files

# Global variable for key-to-names mapping (used by Chapter class)
_global_key_to_names: Dict[int, List[str]] = {}


class Chapter:
    """Represents a single chapter with its speaker annotations."""

    def __init__(self, chapter_num: int, character_map: Dict, line_map: Dict) -> None:
        self.chapter_num = chapter_num
        self.character_map = character_map
        self.line_map = line_map

    @property
    def total_lines(self) -> int:
        """Total number of lines in the chapter."""
        return len(self.line_map)

    @property
    def narrator_lines(self) -> int:
        """Number of lines spoken by the narrator (key 1)."""
        return sum(1 for speaker in self.line_map.values() if speaker == 1)

    @property
    def dialogue_lines(self) -> int:
        """Number of dialogue lines (non-narrator)."""
        return self.total_lines - self.narrator_lines

    @property
    def unique_speakers(self) -> List[str]:
        """List of unique speaker names in this chapter."""
        speaker_keys = set(self.line_map.values())
        return [get_preferred_name_for_key(k, _global_key_to_names) for k in speaker_keys]

    @property
    def speaker_counts(self) -> Dict[str, int]:
        """Count of lines per speaker."""
        counts: Dict[str, int] = {}
        for speaker_key in self.line_map.values():
            name = get_preferred_name_for_key(speaker_key, _global_key_to_names)
            counts[name] = counts.get(name, 0) + 1
        return counts

    def get_speakers(self) -> Dict[str, List[int]]:
        """Return a mapping of speaker names to their line numbers."""
        speakers: Dict[str, List[int]] = {}
        for line_num, speaker_key in self.line_map.items():
            name = get_preferred_name_for_key(speaker_key, _global_key_to_names)
            if name not in speakers:
                speakers[name] = []
            speakers[name].append(line_num)
        return speakers


class ChapterAnalyzer:
    """Analyzes multiple chapter files and generates statistics."""

    def __init__(self, directory: str, verbose: bool = False) -> None:
        self.directory = Path(directory)
        self.verbose = verbose
        self.chapters: List[Chapter] = []
        self._load_chapters()

    def _load_chapters(self) -> None:
        """Load all chapter map files from the directory."""
        chapter_files = find_chapter_files(str(self.directory), verbose=self.verbose)

        if not chapter_files:
            print(f"No chapter map files found in {self.directory}", file=sys.stderr)
            sys.exit(1)

        if self.verbose:
            print(f"Found {len(chapter_files)} chapter files")

        # Build global key-to-names mapping from all chapters
        self._build_key_to_names(chapter_files)

        # Load each chapter
        for chapter_file in chapter_files:
            chapter_num = self._extract_chapter_number(chapter_file)
            character_map, line_map = load_chapter_map(str(chapter_file))

            if character_map:
                self.chapters.append(Chapter(chapter_num, character_map, line_map))

        if self.verbose:
            print(f"Successfully loaded {len(self.chapters)} chapters")

    def _extract_chapter_number(self, file_path: Path) -> int:
        """Extract chapter number from filename (e.g., chapter_0.map.json -> 0)."""
        stem = file_path.stem  # e.g., "chapter_0.map"
        # Remove .map suffix if present
        if stem.endswith('.map'):
            stem = stem[:-4]
        return int(stem.split('_')[1])

    def _build_key_to_names(self, chapter_files: List[Path]) -> None:
        """Build global mapping of speaker keys to preferred names."""
        global _global_key_to_names
        _global_key_to_names = {}

        for chapter_file in chapter_files:
            character_map, _ = load_chapter_map(str(chapter_file))
            if not character_map:
                continue

            for k, v in character_map.items():
                key_num = int(k)
                if key_num not in _global_key_to_names:
                    _global_key_to_names[key_num] = []

                name_lower = v.lower().strip()
                # Avoid duplicates (case-insensitive)
                if not any(n.lower().strip() == name_lower for n in _global_key_to_names[key_num]):
                    _global_key_to_names[key_num].append(v)

    def get_overall_statistics(self) -> Dict:
        """Get overall statistics across all chapters."""
        total_lines = sum(c.total_lines for c in self.chapters)
        narrator_lines = sum(c.narrator_lines for c in self.chapters)

        # Collect all unique speakers
        all_speakers = set()
        for chapter in self.chapters:
            all_speakers.update(chapter.unique_speakers)

        # Count lines per speaker across all chapters
        speaker_counts: Dict[str, int] = {}
        for chapter in self.chapters:
            for name, count in chapter.speaker_counts.items():
                speaker_counts[name] = speaker_counts.get(name, 0) + count

        return {
            'total_chapters': len(self.chapters),
            'total_lines': total_lines,
            'narrator_lines': narrator_lines,
            'dialogue_lines': total_lines - narrator_lines,
            'unique_speakers': len(all_speakers),
            'speaker_counts': speaker_counts
        }

    def get_top_speakers(self, n: int = 10) -> List[tuple]:
        """Get top N speakers by line count."""
        stats = self.get_overall_statistics()
        counts = stats['speaker_counts']
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_chapter_stats(self) -> List[Dict]:
        """Get statistics for each chapter."""
        return [{
            'chapter': c.chapter_num,
            'total_lines': c.total_lines,
            'narrator_lines': c.narrator_lines,
            'dialogue_lines': c.dialogue_lines,
            'unique_speakers': len(c.unique_speakers),
            'speakers': c.unique_speakers
        } for c in self.chapters]

    def generate_histogram_data(self) -> pd.DataFrame:
        """Generate histogram data showing speaker counts per chapter."""
        records = []
        for chapter in self.chapters:
            for speaker, count in chapter.speaker_counts.items():
                records.append({
                    'chapter': chapter.chapter_num,
                    'speaker': speaker,
                    'line_count': count
                })
        return pd.DataFrame(records)

    def print_overall_statistics(self, stats: Dict) -> None:
        """Print overall statistics to console."""
        print("=" * 60)
        print("OVERALL STATISTICS")
        print("=" * 60)
        print(f"Total Chapters Analyzed: {stats['total_chapters']}")
        print(f"Total Lines: {stats['total_lines']}")
        print(f"Unique Speakers Across All Chapters: {stats['unique_speakers']}")
        print(f"Narrator Lines: {stats['narrator_lines']}")
        print(f"Dialogue Lines: {stats['dialogue_lines']}")

    def print_top_speakers(self, top_speakers: List[tuple], total_lines: int) -> None:
        """Print top speakers by line count."""
        print("\n" + "=" * 60)
        print("TOP SPEAKERS BY LINE COUNT")
        print("=" * 60)

        for rank, (speaker, count) in enumerate(top_speakers, 1):
            percentage = (count / total_lines) * 100 if total_lines > 0 else 0
            print(f"{rank}. {speaker}: {count} lines ({percentage:.1f}%)")

    def print_chapter_statistics(self, chapter_stats: List[Dict]) -> None:
        """Print per-chapter statistics."""
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

    def print_histogram(self, df: pd.DataFrame) -> None:
        """Print ASCII histogram of speaker counts."""
        if df.empty:
            print("\nNo data to display.")
            return

        # Create pivot table
        pivot = df.pivot_table(
            index='speaker',
            columns='chapter',
            values='line_count',
            fill_value=0,
            aggfunc='sum'
        )
        pivot['Total'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('Total', ascending=False)

        print("\n" + "=" * 60)
        print("HISTOGRAM: SPEAKER DIALOGUE LINE COUNTS")
        print("=" * 60)
        print("\nPer-Chapter Dialogue Line Counts (by speaker):")
        print("-" * 60)
        print(pivot.to_string())

        # ASCII bar chart
        print("\n" + "-" * 60)
        print("ASCII BAR CHART (Total Lines per Speaker)")
        print("-" * 60)

        speakers = pivot.index.tolist()
        totals = pivot['Total'].tolist()
        max_count = max(totals) if totals else 1
        max_bar_width = 40

        for speaker, count in zip(speakers, totals):
            bar_length = int((count / max_count) * max_bar_width) if max_count > 0 else 0
            bar = "#" * bar_length
            print(f"{speaker:30} | {bar} ({count})")

    def print_all_speakers(self, stats: Dict) -> None:
        """Print all detected speakers."""
        print("\n" + "=" * 60)
        print("ALL SPEAKERS DETECTED")
        print("=" * 60)
        for speaker in sorted(stats['speaker_counts'].keys()):
            print(f"  - {speaker}")

    def save_json(self, output_dir: Path, stats: Dict, chapter_stats: List[Dict]) -> None:
        """Save full analysis to JSON file."""
        output_data = {
            'total_chapters': stats['total_chapters'],
            'total_lines': stats['total_lines'],
            'unique_speakers': stats['unique_speakers'],
            'speaker_counts': {s: int(c) for s, c in stats['speaker_counts'].items()},
            'chapter_stats': chapter_stats
        }

        output_path = output_dir / "chapter_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nFull analysis saved to: {output_path}")

    def save_csv(self, output_dir: Path, chapter_stats: List[Dict]) -> None:
        """Save chapter statistics to CSV file."""
        csv_path = output_dir / "chapter_stats.csv"
        fieldnames = ['chapter', 'total_lines', 'narrator_lines', 'dialogue_lines', 'unique_speakers', 'speakers']

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(chapter_stats)
        print(f"CSV statistics saved to: {csv_path}")

    def analyze(self, args) -> None:
        """Run the full analysis and output results."""
        # Get all statistics
        overall_stats = self.get_overall_statistics()
        chapter_stats = self.get_chapter_stats()
        top_speakers = self.get_top_speakers(10)
        histogram_df = self.generate_histogram_data()

        # Print reports
        self.print_overall_statistics(overall_stats)
        self.print_top_speakers(top_speakers, overall_stats['total_lines'])
        self.print_chapter_statistics(chapter_stats)

        # Optional outputs
        if args.histogram:
            self.print_histogram(histogram_df)

        if args.json_output:
            self.save_json(self.directory, overall_stats, chapter_stats)

        if args.csv_output:
            self.save_csv(self.directory, chapter_stats)

        if args.verbose:
            self.print_all_speakers(overall_stats)


def load_chapter_map(file_path: str) -> tuple:
    """Load a chapter map JSON file and return (character_map, line_map)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[0], data[1]  # (character_map, line_map)
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return {}, {}


def get_preferred_name_for_key(key: int, key_to_names: Dict[int, List[str]]) -> str:
    """
    Get the preferred name for a speaker key.

    Priority:
    1. Narrator (key 1) -> "narrator"
    2. Single name -> that name
    3. Multiple names -> concatenated with ", "
    4. Unknown key -> "Unknown_{key}"
    """
    if key in key_to_names and key_to_names[key]:
        if key == 1 and "narrator" in key_to_names[key]:
            return "narrator"
        if len(key_to_names[key]) == 1:
            return key_to_names[key][0]
        return ", ".join(key_to_names[key])
    return f"Unknown_{key}"


def main() -> None:
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
        "--histogram",
        action="store_true",
        help="Generate ASCII histogram of speaker dialogue counts"
    )
    args = parser.parse_args()

    # Validate directory
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    analyzer = ChapterAnalyzer(str(directory), verbose=args.verbose)
    analyzer.analyze(args)


if __name__ == "__main__":
    main()