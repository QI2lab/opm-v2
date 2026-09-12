"""Generate an animated GIF from an OPM TensorStore array."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from opm_v2.utils.script_io import load_tensorstore


def scale_to_uint8(
    array: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    gamma: float = 1.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Scale image data to unsigned 8-bit intensities.

    Parameters
    ----------
    array : numpy.ndarray
        Input image data.
    vmin, vmax : float or None
        Intensity window. Data extrema are used when omitted.
    gamma : float
        Positive gamma correction exponent.
    contrast : float
        Non-negative contrast factor around mid-gray.

    Returns
    -------
    numpy.ndarray
        Unsigned 8-bit data with the input shape.

    Raises
    ------
    ValueError
        If gamma or contrast is outside its valid range.
    """
    if gamma <= 0 or contrast < 0:
        raise ValueError("Gamma must be positive and contrast must be non-negative")
    scaled = np.asarray(array, dtype=np.float32)
    lower = float(np.nanmin(scaled)) if vmin is None else float(vmin)
    upper = float(np.nanmax(scaled)) if vmax is None else float(vmax)
    scaled = np.clip((scaled - lower) / max(upper - lower, 1e-12), 0, 1)
    scaled = np.power(scaled, gamma)
    scaled = np.clip(0.5 + contrast * (scaled - 0.5), 0, 1)
    return (scaled * 255).astype(np.uint8)


def frames_from_array(array: np.ndarray) -> np.ndarray:
    """Flatten acquisition indices into a sequence of grayscale frames.

    Parameters
    ----------
    array : numpy.ndarray
        Array whose final two dimensions are Y and X.

    Returns
    -------
    numpy.ndarray
        Three-dimensional ``(frame, y, x)`` array.

    Raises
    ------
    ValueError
        If the input has fewer than two dimensions.
    """
    data = np.asarray(array)
    if data.ndim < 2:
        raise ValueError("GIF input must contain Y and X dimensions")
    return data.reshape((-1, *data.shape[-2:]))


def save_gif(
    array: np.ndarray,
    path: Path | str,
    *,
    fps: float = 10.0,
    loop: int = 0,
    gamma: float = 0.8,
    contrast: float = 1.05,
) -> Path:
    """Save image data as an animated GIF.

    Parameters
    ----------
    array : numpy.ndarray
        Acquisition data with Y and X as its final dimensions.
    path : Path or str
        Output GIF path.
    fps : float
        Playback frames per second.
    loop : int
        Number of animation loops, where zero repeats forever.
    gamma : float
        Gamma correction exponent.
    contrast : float
        Contrast factor around mid-gray.

    Returns
    -------
    Path
        Written GIF path.

    Raises
    ------
    ValueError
        If ``fps`` is not positive.
    """
    if fps <= 0:
        raise ValueError("FPS must be positive")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = scale_to_uint8(frames_from_array(array), gamma=gamma, contrast=contrast)
    imageio.mimsave(output, list(frames), format="GIF", duration=1.0 / fps, loop=loop)
    return output


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("array", type=Path, help="Input Zarr v3 array node")
    parser.add_argument("output", type=Path, help="Output GIF path")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--loop", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--contrast", type=float, default=1.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load TensorStore pixels and save an animated GIF.

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
    output = save_gif(
        load_tensorstore(args.array),
        args.output,
        fps=args.fps,
        loop=args.loop,
        gamma=args.gamma,
        contrast=args.contrast,
    )
    print(f"Saved GIF to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
