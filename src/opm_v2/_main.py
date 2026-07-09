"""Command-line entry point for the OPM Micro-Manager GUI."""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

from opm_v2._app_v2 import main as run_app


def configure_logging() -> None:
    """Reduce noisy third-party logging before launching the GUI."""
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    logging.raiseExceptions = False
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    # Keep warnings/errors visible; avoid burying hardware or MMCore failures.
    logging.getLogger("pymmcore-plus").setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parse launcher arguments."""
    parser = argparse.ArgumentParser(description="Launch the OPM Micro-Manager GUI.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to the OPM JSON config file.",
    )
    return parser.parse_args()


def main() -> None:
    """Configure process-level behavior and launch the app."""
    configure_logging()
    args = parse_args()
    run_app(config_path=args.config)


if __name__ == "__main__":
    main()