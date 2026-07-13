"""Load and display saved OPM event-structure review files."""

from __future__ import annotations

import argparse
from pathlib import Path

from opm_v2.engine.event_review_v2 import (
    load_event_structure_review,
    print_event_structure_review,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Display an event structure review JSON created by setup_events_v2."
    )
    parser.add_argument("review_json", type=Path, help="Path to review JSON file.")
    return parser.parse_args()


def main() -> None:
    """Load a saved event review and print a compact console summary."""
    args = parse_args()
    review = load_event_structure_review(args.review_json)
    print_event_structure_review(review)


if __name__ == "__main__":
    main()
