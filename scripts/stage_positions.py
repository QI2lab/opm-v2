"""Generate stage positions for a configured OPM scan volume."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from opm_v2.utils.position_tools import generate_stage_positions
from opm_v2.utils.script_io import save_json


def build_parser() -> argparse.ArgumentParser:
    """Build the stage-position command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output stage-position JSON")
    parser.add_argument("--x", nargs=2, type=float, default=(0.0, 1000.0))
    parser.add_argument("--y", nargs=2, type=float, default=(0.0, 1000.0))
    parser.add_argument("--z", nargs=2, type=float, default=(0.0, 20.0))
    parser.add_argument("--camera-shape", nargs=2, type=int, default=(1900, 386))
    parser.add_argument("--pixel-size-um", type=float, default=0.115)
    parser.add_argument("--angle-deg", type=float, default=30.0)
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--max-stage-range-um", type=float, default=1000.0)
    parser.add_argument("--scan-overlap-um", type=float, default=40.0)
    parser.add_argument("--coverslip-slope-x", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate and save stage positions.

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
    positions, geometry = generate_stage_positions(
        x_bounds_um=tuple(args.x),
        y_bounds_um=tuple(args.y),
        z_bounds_um=tuple(args.z),
        camera_shape=tuple(args.camera_shape),
        pixel_size_um=args.pixel_size_um,
        angle_deg=args.angle_deg,
        tile_overlap=args.tile_overlap,
        max_stage_scan_range_um=args.max_stage_range_um,
        scan_axis_overlap_um=args.scan_overlap_um,
        coverslip_slope_x=args.coverslip_slope_x,
    )
    save_json(positions, args.output)
    print(f"Saved {geometry['num_stage_positions']} positions to {args.output}")
    print(geometry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
