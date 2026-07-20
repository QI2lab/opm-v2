"""Inspect deprecated AO result stores using the current zarr-python API."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import zarr

DEPRECATED_ARRAY_KEYS = (
    "images_per_iteration",
    "images_per_mode",
    "metrics_per_iteration",
    "metrics_per_mode",
    "coefficients_per_iteration",
    "modes_to_optimize",
)


def load_deprecated_results(results_path: Path) -> dict[str, Any]:
    """Load a legacy AO result store without ``DirectoryStore``.

    Parameters
    ----------
    results_path : Path
        Deprecated ``ao_results.zarr`` path.

    Returns
    -------
    dict[str, Any]
        Legacy arrays, mode names, and attributes.

    Raises
    ------
    KeyError
        If a required legacy array is absent.
    """
    root = zarr.open_group(str(results_path), mode="r")
    required_keys = (*DEPRECATED_ARRAY_KEYS, "zernike_mode_names")
    missing_keys = [key for key in required_keys if key not in root]
    if missing_keys:
        raise KeyError(f"Missing legacy AO arrays: {', '.join(missing_keys)}")
    results = {key: np.asarray(root[key][:]) for key in DEPRECATED_ARRAY_KEYS}
    raw_names = np.asarray(root["zernike_mode_names"][:])
    results["mode_names"] = [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in raw_names
    ]
    results["metadata"] = dict(root.attrs)
    return results


def discover_deprecated_results(root: Path) -> list[Path]:
    """Find deprecated AO result stores below a directory.

    Parameters
    ----------
    root : Path
        Directory to search.

    Returns
    -------
    list[Path]
        Sorted result-store paths.
    """
    return sorted(root.rglob("ao_results.zarr"))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Legacy AO result directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load and summarize deprecated AO grid result stores.

    Parameters
    ----------
    argv : Sequence[str] or None
        Command-line arguments, excluding the executable name.

    Returns
    -------
    int
        Process exit status.

    Raises
    ------
    FileNotFoundError
        If no legacy result stores exist below the supplied directory.
    """
    args = build_parser().parse_args(argv)
    paths = discover_deprecated_results(args.root)
    if not paths:
        raise FileNotFoundError(f"No ao_results.zarr stores found below {args.root}")
    for path in paths:
        results = load_deprecated_results(path)
        print(path, {key: np.shape(value) for key, value in results.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
