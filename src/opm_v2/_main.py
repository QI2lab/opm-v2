"""Command-line entry point for the OPM Micro-Manager GUI."""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

from opm_v2._app import DEFAULT_CONFIG_PATH, launch_opm_app


def configure_logging() -> None:
    """Reduce noisy third-party logging before launching the GUI."""
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    logging.raiseExceptions = False
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    # Keep warnings/errors visible; avoid burying hardware or MMCore failures.
    logging.getLogger("pymmcore-plus").setLevel(logging.INFO)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured launcher parser.
    """
    parser = argparse.ArgumentParser(description="Launch the OPM Micro-Manager GUI.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to the OPM JSON config file.",
    )
    parser.add_argument(
        "--mm-config",
        default=None,
        help="Optional Micro-Manager configuration to load.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use Micro-Manager demo devices and simulated external OPM hardware.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Configure process-level behavior and launch the app.

    Parameters
    ----------
    argv : list[str] or None
        Arguments to parse, or ``None`` to use ``sys.argv``.

    Returns
    -------
    int
        Process exit status.
    """
    configure_logging()
    args = build_arg_parser().parse_args(argv)
    mm_config = "MMConfig_demo.cfg" if args.demo else args.mm_config
    launch_opm_app(
        config_path=args.config or DEFAULT_CONFIG_PATH,
        mm_config=mm_config,
        simulate_hardware=True if args.demo else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
