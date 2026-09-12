"""Discover and plot every current-format AO result store in a grid run."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

if __package__:
    from .parse_ao_results import plot_results
else:
    from parse_ao_results import plot_results


def discover_results(root: Path) -> list[Path]:
    """Find AO result stores below a directory.

    Parameters
    ----------
    root : Path
        Grid acquisition directory.

    Returns
    -------
    list[Path]
        Sorted ``ao_results.zarr`` paths.

    Raises
    ------
    FileNotFoundError
        If the root does not exist or contains no result stores.
    """
    if not root.exists():
        raise FileNotFoundError(root)
    paths = sorted(root.rglob("ao_results.zarr"))
    if not paths:
        raise FileNotFoundError(f"No ao_results.zarr stores found below {root}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="AO grid result directory")
    parser.add_argument("--output", type=Path, help="Shared plot output directory")
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Plot all current AO result stores below a grid directory.

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
    paths = discover_results(args.root)
    for index, path in enumerate(paths):
        output = args.output / f"grid_{index}" if args.output else path.parent
        plot_results(path, output, args.show)
    print(f"Processed {len(paths)} AO result stores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
