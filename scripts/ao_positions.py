"""Select acquisition positions for adaptive-optics sampling."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from opm_v2.utils.position_tools import select_ao_positions
from opm_v2.utils.script_io import load_json, save_json


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positions", type=Path, help="Input stage-position JSON")
    parser.add_argument("output", type=Path, help="Output AO-position JSON")
    parser.add_argument("--scan-samples", type=int, default=3)
    parser.add_argument("--tile-samples", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Select and save AO sampling positions.

    Parameters
    ----------
    argv : Sequence[str] or None
        Command-line arguments, excluding the executable name.

    Returns
    -------
    int
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    selected = select_ao_positions(
        load_json(args.positions), args.scan_samples, args.tile_samples
    )
    save_json(selected, args.output)
    print(f"Saved {len(selected)} AO positions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
