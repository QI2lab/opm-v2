"""Load and inspect current sensorless-AO optimization results."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from opm_v2.utils.sensorless_ao import load_optimization_results


def summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    """Summarize current AO result arrays and metadata.

    Parameters
    ----------
    results : dict[str, Any]
        Result mapping returned by ``load_optimization_results``.

    Returns
    -------
    dict[str, Any]
        Compact array and optimization summary.
    """
    metadata = results["metadata"]
    return {
        "image_count": int(np.asarray(results["all_images"]).shape[0]),
        "metric_count": int(np.asarray(results["all_metrics"]).size),
        "iterations": int(metadata["num_iterations"]),
        "mode_samples": int(metadata["num_mode_samples"]),
        "modes_to_optimize": list(metadata["modes_to_optimize"]),
    }


def show_image_stack(images: np.ndarray) -> None:
    """Display an AO image stack with an interactive frame slider.

    Parameters
    ----------
    images : numpy.ndarray
        Image stack with Y and X as its final dimensions.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    frames = np.asarray(images).reshape((-1, *np.asarray(images).shape[-2:]))
    figure, axis = plt.subplots()
    figure.subplots_adjust(bottom=0.2)
    display = axis.imshow(frames[0], cmap="gray", aspect="equal")
    slider_axis = figure.add_axes((0.2, 0.05, 0.6, 0.03))
    slider = Slider(slider_axis, "Image", 0, len(frames) - 1, valinit=0, valstep=1)

    def update(_value: float) -> None:
        """Update the displayed image after a slider change.

        Parameters
        ----------
        _value : float
            New slider value.
        """
        display.set_data(frames[int(slider.val)])
        figure.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="AO results.zarr path")
    parser.add_argument("--show-images", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load and summarize an AO result store.

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
    results = load_optimization_results(args.results)
    print(summarize_results(results))
    if args.show_images:
        show_image_stack(results["all_images"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
