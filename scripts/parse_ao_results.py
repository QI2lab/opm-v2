"""Plot current sensorless-AO optimization metrics and coefficients."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from opm_v2.utils import sensorless_ao as ao


def plot_results(
    results_path: Path, output_directory: Path | None = None, show: bool = False
) -> dict:
    """Load one AO result store and create its diagnostic plots.

    Parameters
    ----------
    results_path : Path
        Current-format ``ao_results.zarr`` store.
    output_directory : Path or None
        Plot destination. Defaults to the result store's parent.
    show : bool
        Whether to display plots interactively.

    Returns
    -------
    dict
        Loaded AO result mapping.
    """
    results = ao.load_optimization_results(results_path)
    output = output_directory or results_path.parent
    output.mkdir(parents=True, exist_ok=True)
    mode_names = results["mode_names"]
    ao.plot_metric_progress(
        ao_results=results,
        zernike_mode_names=mode_names,
        save_dir_path=output,
        show_fig=show,
    )
    ao.plot_zernike_coeffs(
        ao_results=results,
        zernike_mode_names=mode_names,
        save_dir_path=output,
        show_fig=show,
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="AO results.zarr path")
    parser.add_argument("--output", type=Path, help="Plot output directory")
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Create plots for a current AO result store.

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
    plot_results(args.results, args.output, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
