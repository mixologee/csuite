"""
scripts/run_indexer.py

Manual entry point for the knowledge indexer.

Usage:
    python scripts/run_indexer.py --company janky_games
    python scripts/run_indexer.py --company janky_games --force

First run indexes everything from scratch. Subsequent runs only
re-index if enough new decisions have accumulated (configurable
via index_threshold in company config). Use --force to bypass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import argparse
from core.memory.indexer import run_indexer


def main():
    parser = argparse.ArgumentParser(
        description="Run the knowledge indexer for a company."
    )
    parser.add_argument("--company", required=True, help="Company ID")
    parser.add_argument("--force", action="store_true",
                        help="Re-index regardless of threshold")
    args = parser.parse_args()

    success = run_indexer(args.company, force=args.force)
    if success:
        print("\nIndexing complete.")
    else:
        print("\nNo indexing performed.")


if __name__ == "__main__":
    main()
