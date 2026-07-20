"""Pure geometry helpers shared by OPM position scripts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _ceil_ratio(distance: float, stride: float) -> int:
    """Return the number of strides needed to cover a distance.

    Parameters
    ----------
    distance : float
        Distance to cover.
    stride : float
        Positive stride length.

    Returns
    -------
    int
        Covering stride count without floating-point boundary inflation.
    """
    ratio = distance / stride
    nearest_integer = round(ratio)
    if np.isclose(ratio, nearest_integer, rtol=1e-12, atol=1e-12):
        return int(nearest_integer)
    return int(np.ceil(ratio))


def positions_array(stage_positions: Sequence[dict[str, float]]) -> np.ndarray:
    """Convert position dictionaries to a Z/Y/X coordinate array.

    Parameters
    ----------
    stage_positions : Sequence[dict[str, float]]
        Stage positions containing ``x``, ``y``, and ``z`` values.

    Returns
    -------
    numpy.ndarray
        Coordinate array with shape ``(positions, 3)`` in Z/Y/X order.

    Raises
    ------
    ValueError
        If no stage positions are supplied.
    """
    if not stage_positions:
        raise ValueError("At least one stage position is required")
    return np.asarray(
        [(position["z"], position["y"], position["x"]) for position in stage_positions],
        dtype=float,
    )


def select_ao_positions(
    stage_positions: Sequence[dict[str, float]],
    num_scan_positions: int,
    num_tile_positions: int,
) -> list[dict[str, float]]:
    """Select real stage positions nearest a regular AO sampling grid.

    Parameters
    ----------
    stage_positions : Sequence[dict[str, float]]
        Acquisition stage positions.
    num_scan_positions : int
        Requested AO samples along X.
    num_tile_positions : int
        Requested AO samples along Y.

    Returns
    -------
    list[dict[str, float]]
        Unique real stage positions selected for AO optimization.

    Raises
    ------
    ValueError
        If either requested sample count is less than one.
    """
    if num_scan_positions < 1 or num_tile_positions < 1:
        raise ValueError("AO grid dimensions must be positive")
    coordinates = positions_array(stage_positions)
    targets_x = np.linspace(
        coordinates[:, 2].min(), coordinates[:, 2].max(), num_scan_positions
    )
    targets_y = np.linspace(
        coordinates[:, 1].min(), coordinates[:, 1].max(), num_tile_positions
    )
    targets_z = np.unique(coordinates[:, 0])
    selected_indices: list[int] = []
    for z_value in targets_z:
        for y_value in targets_y:
            for x_value in targets_x:
                distances = np.linalg.norm(
                    coordinates - np.asarray([z_value, y_value, x_value]), axis=1
                )
                index = int(np.argmin(distances))
                if index not in selected_indices:
                    selected_indices.append(index)
    return [
        {axis: float(stage_positions[index][axis]) for axis in ("x", "y", "z")}
        for index in selected_indices
    ]


def map_stage_positions_to_ao(
    stage_positions: Sequence[dict[str, float]],
    ao_positions: Sequence[dict[str, float]],
) -> list[int]:
    """Map every acquisition position to its nearest AO position.

    Parameters
    ----------
    stage_positions : Sequence[dict[str, float]]
        Acquisition stage positions.
    ao_positions : Sequence[dict[str, float]]
        Positions at which AO corrections were measured.

    Returns
    -------
    list[int]
        AO-position index assigned to each acquisition position.
    """
    stage_coordinates = positions_array(stage_positions)
    ao_coordinates = positions_array(ao_positions)
    return [
        int(np.argmin(np.linalg.norm(ao_coordinates - coordinate, axis=1)))
        for coordinate in stage_coordinates
    ]


def compose_stage_positions(
    scan_positions_um: Sequence[float],
    tile_positions_um: Sequence[float],
    z_positions_um: Sequence[float],
    z_offset_per_scan_um: float = 0.0,
) -> list[dict[str, float]]:
    """Compose OPM stage positions in Z, scan, then tile order.

    Parameters
    ----------
    scan_positions_um : Sequence[float]
        Scan-axis starting positions in micrometers.
    tile_positions_um : Sequence[float]
        Tile-axis positions in micrometers.
    z_positions_um : Sequence[float]
        Nominal focus positions in micrometers.
    z_offset_per_scan_um : float
        Coverslip-slope correction added for each scan tile.

    Returns
    -------
    list[dict[str, float]]
        Stage positions using the ordering required by OPM event generation.
    """
    return [
        {
            "x": float(np.round(scan_position, 2)),
            "y": float(np.round(tile_position, 2)),
            "z": float(np.round(z_position + z_offset_per_scan_um * scan_index, 2)),
        }
        for z_position in z_positions_um
        for scan_index, scan_position in enumerate(scan_positions_um)
        for tile_position in tile_positions_um
    ]


def generate_stage_positions(
    *,
    x_bounds_um: tuple[float, float],
    y_bounds_um: tuple[float, float],
    z_bounds_um: tuple[float, float],
    camera_shape: tuple[int, int],
    pixel_size_um: float,
    angle_deg: float,
    tile_overlap: float,
    max_stage_scan_range_um: float,
    scan_axis_overlap_um: float,
    coverslip_slope_x: float,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Generate stage-scan tile origins using the current OPM geometry.

    Parameters
    ----------
    x_bounds_um, y_bounds_um, z_bounds_um : tuple[float, float]
        Inclusive grid bounds in micrometers.
    camera_shape : tuple[int, int]
        Camera crop width and height.
    pixel_size_um : float
        Sample-plane pixel size.
    angle_deg : float
        OPM sheet angle.
    tile_overlap : float
        Fractional overlap along Y and Z.
    max_stage_scan_range_um : float
        Maximum X distance for one stage scan.
    scan_axis_overlap_um : float
        X overlap between adjacent stage scans.
    coverslip_slope_x : float
        Z rise per X run.

    Returns
    -------
    tuple[list[dict[str, float]], dict[str, Any]]
        Stage positions and calculated grid dimensions.

    Raises
    ------
    ValueError
        If geometry parameters are outside their valid ranges.
    """
    if pixel_size_um <= 0 or max_stage_scan_range_um <= 0:
        raise ValueError("Pixel size and maximum stage range must be positive")
    if not 0 <= tile_overlap < 1:
        raise ValueError("Tile overlap must be in the range [0, 1)")
    crop_x, crop_y = camera_shape
    x_min, x_max = sorted(x_bounds_um)
    y_min, y_max = sorted(y_bounds_um)
    z_start, z_stop = z_bounds_um
    range_x = x_max - x_min
    range_y = y_max - y_min
    range_z = abs(z_stop - z_start)
    angle_scale = abs(np.sin(np.deg2rad(angle_deg)))
    y_step_max = crop_x * pixel_size_um * (1 - tile_overlap)
    z_step_max = crop_y * pixel_size_um * angle_scale * (1 - tile_overlap)
    scan_stride = max(max_stage_scan_range_um - scan_axis_overlap_um, 1e-9)
    n_scan = max(1, _ceil_ratio(range_x, scan_stride))
    n_tile = max(1, _ceil_ratio(range_y, y_step_max) + 1)
    n_z = max(1, _ceil_ratio(range_z, z_step_max) + 1)
    x_positions = np.linspace(x_min, x_max, n_scan, endpoint=False)
    y_positions = np.linspace(y_min, y_max, n_tile)
    z_positions = np.linspace(z_start, z_stop, n_z)
    positions = compose_stage_positions(
        x_positions,
        y_positions,
        z_positions,
        z_offset_per_scan_um=(
            float(x_positions[1] - x_positions[0]) * coverslip_slope_x
            if len(x_positions) > 1
            else 0.0
        ),
    )
    geometry = {
        "num_scan_positions": n_scan,
        "num_tile_positions": n_tile,
        "num_z_positions": n_z,
        "num_stage_positions": len(positions),
    }
    return positions, geometry
