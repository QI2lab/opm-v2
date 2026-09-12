"""Inspect an array written by the OPM TensorStore data handler."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from opm_v2.utils.script_io import load_json, load_tensorstore


def summarize_array(data: np.ndarray) -> dict[str, object]:
    """Summarize a materialized acquisition array.

    Parameters
    ----------
    data : numpy.ndarray
        Acquisition pixels.

    Returns
    -------
    dict[str, object]
        Shape, data type, and intensity range.
    """
    return {
        "shape": tuple(data.shape),
        "dtype": str(data.dtype),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("array", type=Path, help="Zarr v3 array node to inspect")
    parser.add_argument("--positions", type=Path, help="Optional position JSON")
    parser.add_argument("--show", action="store_true", help="Display 2D planes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load and inspect a TensorStore acquisition array.

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
    data = load_tensorstore(args.array)
    print(summarize_array(data))
    if args.positions:
        positions = load_json(args.positions)
        print(f"positions: {len(positions)}")
    if args.show:
        import matplotlib.pyplot as plt

        planes = data.reshape((-1, *data.shape[-2:]))
        for plane in planes:
            plt.figure()
            plt.imshow(plane)
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
