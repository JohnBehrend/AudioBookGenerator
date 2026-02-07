#!/usr/bin/env python3
"""
Simple script to list all chapter_*.map.json files in a directory.
"""

import argparse
import sys
from pathlib import Path
from typing import List


def find_chapter_files(directory: str, chapter_filter: int = None, verbose: bool = False) -> List[str]:
    """
    Find all chapter_*.map.json files in the specified directory.

    Args:
        directory: Directory to search
        chapter_filter: Optional chapter number to filter (e.g., 1 for chapter_1.map.json)
        verbose: Enable debug output

    Returns:
        List of chapter file paths
    """
    path = Path(directory).resolve()

    # Debug: Print what path we're actually searching
    if verbose:
        print(f"DEBUG: Searching in: {path}")
        print(f"DEBUG: Absolute path: {path.absolute()}")
        print(f"DEBUG: Exists: {path.exists()}")
        if path.exists():
            print(f"DEBUG: Contents: {list(path.iterdir())[:5]}")

    if not path.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    # Find all chapter_*.map.json files
    if chapter_filter is not None:
        # Find specific chapter number
        pattern = f"chapter_{chapter_filter:02d}*.map.json"
    else:
        # Find all chapters
        pattern = "chapter_*.map.json"

    # Use Path.rglob() which works with the pattern relative to the directory
    # It searches recursively from the path object
    all_files = sorted(path.rglob(pattern))

    # Debug: Show what we found
    if verbose:
        print(f"DEBUG: rglob found {len(all_files)} files total")

    # Filter out non-map.json files
    # Check if .map.json is in the filename (since rglob matches the full pattern)
    chapter_files = [f for f in all_files if ".map.json" in f.name]

    return sorted(chapter_files)


def main():
    parser = argparse.ArgumentParser(
        description="List all chapter_*.map.json files in a directory"
    )
    parser.add_argument(
        "directory",
        help="Directory to search for chapter map files"
    )
    parser.add_argument(
        "--chapter",
        type=int,
        help="Filter to specific chapter number (e.g., 1 for chapter_1.map.json)"
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Just show the count of files found"
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Show relative paths instead of absolute paths"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed debug information"
    )

    args = parser.parse_args()

    try:
        files = find_chapter_files(args.directory, args.chapter, args.verbose)

        if args.count:
            print(len(files))
        else:
            print(f"Found {len(files)} chapter map file(s) in {args.directory}")
            print("-" * 60)

            for file_path in files:
                if args.relative:
                    print(f"  {file_path.name}")
                else:
                    print(f"  {file_path}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
