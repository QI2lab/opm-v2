"""Map acquisition positions to measured adaptive-optics grid positions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from opm_v2.utils.position_tools import map_stage_positions_to_ao
from opm_v2.utils.script_io import load_json, save_json


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positions", type=Path, help="Acquisition-position JSON")
    parser.add_argument("ao_positions", type=Path, help="Measured AO-position JSON")
    parser.add_argument("output", type=Path, help="Output position-to-AO map JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Calculate and save the nearest AO index for every stage position.

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
    mapping = map_stage_positions_to_ao(
        load_json(args.positions), load_json(args.ao_positions)
    )
    save_json(mapping, args.output)
    print(f"Saved {len(mapping)} AO assignments to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
