"""Pure geometry helpers shared by OPM position scripts."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

FloatArray = NDArray[np.float64]


def _angle_components(angle_deg: float) -> tuple[float, float]:
    """Return the sine and cosine of a valid OPM angle.

    Parameters
    ----------
    angle_deg : float
        OPM sheet angle in degrees.

    Returns
    -------
    tuple[float, float]
        Sine and cosine of the angle.

    Raises
    ------
    ValueError
        If the angle cannot define an oblique camera-to-laboratory transform.
    """
    theta = np.deg2rad(float(angle_deg))
    sin_theta = float(np.sin(theta))
    cos_theta = float(np.cos(theta))
    if np.isclose(sin_theta, 0.0):
        raise ValueError("OPM angle must have a nonzero sine")
    return sin_theta, cos_theta


def lab2cam(
    lab_x: ArrayLike,
    lab_scan: ArrayLike,
    lab_z: ArrayLike,
    angle_deg: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Transform laboratory coordinates into OPM camera coordinates.

    This follows the axis convention in ``opm-processing-v2``.  Its laboratory
    scan axis is mapped onto this microscope's physical stage X by callers.

    Parameters
    ----------
    lab_x : ArrayLike
        Laboratory coordinate parallel to camera X.
    lab_scan : ArrayLike
        Laboratory coordinate along the acquisition scan direction.
    lab_z : ArrayLike
        Laboratory axial coordinate.
    angle_deg : float
        OPM sheet angle in degrees.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Camera X, camera Y, and raw scan-stage coordinates.
    """
    sin_theta, cos_theta = _angle_components(angle_deg)
    x = np.asarray(lab_x, dtype=float)
    scan = np.asarray(lab_scan, dtype=float)
    z = np.asarray(lab_z, dtype=float)
    camera_y = z / sin_theta
    raw_scan = scan - z * cos_theta / sin_theta
    return x, camera_y, raw_scan


def cam2lab(
    camera_x: ArrayLike,
    camera_y: ArrayLike,
    raw_scan: ArrayLike,
    angle_deg: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Transform OPM camera coordinates into laboratory coordinates.

    Parameters
    ----------
    camera_x : ArrayLike
        Coordinate along camera X.
    camera_y : ArrayLike
        Coordinate along camera Y in sample-space units.
    raw_scan : ArrayLike
        Hardware scan-stage coordinate.
    angle_deg : float
        OPM sheet angle in degrees.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Laboratory X, scan-axis, and Z coordinates.
    """
    sin_theta, cos_theta = _angle_components(angle_deg)
    x = np.asarray(camera_x, dtype=float)
    y = np.asarray(camera_y, dtype=float)
    scan = np.asarray(raw_scan, dtype=float)
    lab_scan = scan + y * cos_theta
    lab_z = y * sin_theta
    return x, lab_scan, lab_z


def oblique_camera_extents_um(
    camera_crop_y: int,
    pixel_size_um: float,
    angle_deg: float,
) -> tuple[float, float]:
    """Return camera-height contributions along lab scan and lab Z.

    Parameters
    ----------
    camera_crop_y : int
        Camera ROI height in pixels.
    pixel_size_um : float
        Sample-space pixel size in micrometers.
    angle_deg : float
        OPM sheet angle in degrees.

    Returns
    -------
    tuple[float, float]
        Absolute scan-axis and Z extents in micrometers.

    Raises
    ------
    ValueError
        If the camera shape or pixel size is not positive.
    """
    if camera_crop_y <= 0 or pixel_size_um <= 0:
        raise ValueError("Camera crop and pixel size must be positive")
    camera_height_um = float(camera_crop_y) * float(pixel_size_um)
    _, lab_scan, lab_z = cam2lab(0.0, camera_height_um, 0.0, angle_deg)
    return abs(float(lab_scan)), abs(float(lab_z))


def sample_depth_levels_um(
    start_um: float,
    end_um: float,
    axial_footprint_um: float,
    overlap_fraction: float,
) -> FloatArray:
    """Return slab origins that cover the requested sample-depth interval.

    ``start_um`` and ``end_um`` are coverage bounds, not a list of slab origins.
    Each OPM slab extends one axial camera footprint from its origin into the
    sample.  Origins are distributed so the first slab starts at ``start_um``,
    the final slab ends at ``end_um``, and adjacent slabs overlap by at least
    ``overlap_fraction``.  An interval no thicker than one footprint requires
    only one slab.

    Parameters
    ----------
    start_um, end_um : float
        Requested shallow and deep coverage bounds relative to the fitted
        coverslip plane.  ``end_um`` must not be less than ``start_um``.
    axial_footprint_um : float
        Reconstructed laboratory-Z extent of one OPM camera frame.
    overlap_fraction : float
        Fractional overlap between neighboring axial slabs.

    Returns
    -------
    numpy.ndarray
        Slab-origin depths in increasing biological-depth order.

    Raises
    ------
    ValueError
        If the footprint, overlap, or numeric inputs are invalid.
    """
    values = np.asarray((start_um, end_um, axial_footprint_um, overlap_fraction))
    if not np.all(np.isfinite(values)):
        raise ValueError("Sample-depth settings must be finite")
    if axial_footprint_um <= 0:
        raise ValueError("Axial OPM footprint must be positive")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("Z slab overlap must be in the range [0, 1)")

    start_um = float(start_um)
    end_um = float(end_um)
    if end_um < start_um and not np.isclose(end_um, start_um):
        raise ValueError("Sample-depth end must be greater than or equal to start")

    span_um = end_um - start_um
    if span_um <= axial_footprint_um or np.isclose(span_um, axial_footprint_um):
        return np.asarray([float(start_um)], dtype=float)

    maximum_step_um = float(axial_footprint_um) * (1.0 - overlap_fraction)
    uncovered_after_first_slab_um = span_um - float(axial_footprint_um)
    interval_count = max(
        1, int(np.ceil(uncovered_after_first_slab_um / maximum_step_um))
    )
    final_origin_um = end_um - float(axial_footprint_um)
    return np.linspace(start_um, final_origin_um, interval_count + 1)


def sample_depth_stage_sign(camera_zstage_orientation: str) -> float:
    """Return the provisional stage sign for positive sample depth.

    The mapping is intentionally isolated here because it must be confirmed on
    the microscope.  User-facing depth remains positive into the sample while
    this sign converts it to the configured physical Z-stage direction.

    Returns
    -------
    float
        ``1`` for positive orientation or ``-1`` for negative orientation.

    Raises
    ------
    ValueError
        If the configured orientation is not positive or negative.
    """
    orientation = str(camera_zstage_orientation).strip().casefold()
    if orientation == "positive":
        return 1.0
    if orientation == "negative":
        return -1.0
    raise ValueError(
        "camera_Zstage_orientation must be either 'positive' or 'negative'"
    )


def apply_oblique_scan_correction(
    stage_positions: Sequence[dict[str, float]],
    angle_deg: float,
    camera_zstage_orientation: str,
    reference_z_um: float | None = None,
) -> list[dict[str, float]]:
    """Convert intended lab scan origins into raw hardware X coordinates.

    Physical stage X is this microscope's fast scan axis. In an oblique
    acquisition, changing lab Z while retaining a fixed lab scan origin
    requires a compensating raw stage-X shift. The intended lab coordinate is
    retained for acquisition metadata and downstream fusion.

    Parameters
    ----------
    stage_positions : Sequence[dict[str, float]]
        Positions whose ``x`` values currently express intended lab scan
        origins.
    angle_deg : float
        OPM sheet angle in degrees.
    camera_zstage_orientation : str
        Sign relating physical stage Z to camera/lab Z.
    reference_z_um : float or None
        Physical Z at which raw stage X and lab scan coordinates coincide.
        Defaults to the first position's Z. A per-position
        ``scan_reference_z_um`` value takes precedence.

    Returns
    -------
    list[dict[str, float]]
        Positions with corrected hardware ``x`` and retained ``lab_scan_um``.
    """
    if not stage_positions:
        return []
    default_reference_z = (
        float(stage_positions[0]["z"])
        if reference_z_um is None
        else float(reference_z_um)
    )
    stage_sign = sample_depth_stage_sign(camera_zstage_orientation)
    corrected: list[dict[str, float]] = []
    for position in stage_positions:
        updated = dict(position)
        lab_scan_um = float(position.get("lab_scan_um", position["x"]))
        scan_reference_z_um = float(
            position.get("scan_reference_z_um", default_reference_z)
        )
        # Moving the specimen stage in +Z moves the sampled plane in the
        # opposite lab-frame direction.  The configured sign describes the
        # physical stage direction for increasing biological depth; lab2cam
        # consumes the corresponding sample-coordinate displacement.
        lab_z_offset_um = -stage_sign * (float(position["z"]) - scan_reference_z_um)
        _, _, raw_scan = lab2cam(
            0.0,
            lab_scan_um,
            lab_z_offset_um,
            angle_deg,
        )
        updated["x"] = float(raw_scan)
        updated["lab_scan_um"] = lab_scan_um
        updated["scan_reference_z_um"] = scan_reference_z_um
        corrected.append(updated)
    return corrected


def expand_stage_positions_for_depth(
    stage_positions: Sequence[dict[str, float]],
    sample_depths_um: Sequence[float],
    camera_zstage_orientation: str,
    angle_deg: float | None = None,
) -> list[dict[str, float]]:
    """Repeat XYZ positions at coverslip-relative sample depths.

    Positions are ordered by depth slab and then by their existing lateral
    order.  Each output carries both the biological sample depth and the signed
    hardware offset so acquisition metadata can describe the conversion.

    Returns
    -------
    list[dict[str, float]]
        Expanded depth-major physical stage positions.

    Raises
    ------
    ValueError
        If no valid depths are supplied or stage orientation is invalid.
    """
    depths = np.asarray(sample_depths_um, dtype=float)
    if depths.ndim != 1 or len(depths) == 0 or not np.all(np.isfinite(depths)):
        raise ValueError("At least one finite sample-depth level is required")
    if len(depths) == 1 and np.isclose(depths[0], 0.0):
        # Preserve the exact pre-depth position payload and event metadata when
        # the new controls are disabled.
        return [dict(position) for position in stage_positions]
    stage_sign = sample_depth_stage_sign(camera_zstage_orientation)

    expanded: list[dict[str, float]] = []
    for depth_index, sample_depth_um in enumerate(depths):
        stage_offset_um = stage_sign * float(sample_depth_um)
        for position in stage_positions:
            expanded_position = dict(position)
            expanded_position["z"] = float(position["z"]) + stage_offset_um
            if "lab_scan_um" in expanded_position:
                if angle_deg is None:
                    raise ValueError(
                        "OPM angle is required for depth-expanded lab scan positions"
                    )
                expanded_position = apply_oblique_scan_correction(
                    [expanded_position],
                    angle_deg,
                    camera_zstage_orientation,
                )[0]
            expanded_position["depth_index"] = int(depth_index)
            expanded_position["sample_depth_um"] = float(sample_depth_um)
            expanded_position["stage_depth_offset_um"] = stage_offset_um
            expanded.append(expanded_position)
    return expanded


def covering_tile_origins(
    start_um: float,
    stop_um: float,
    footprint_um: float,
    overlap_fraction: float,
) -> FloatArray:
    """Return regular tile origins that cover an interval.

    The footprint is expressed in reconstructed laboratory coordinates, not as
    the raw hardware scan length.

    Returns
    -------
    numpy.ndarray
        Tile origins in ascending order, expressed in micrometers.

    Raises
    ------
    ValueError
        If the footprint or overlap fraction is invalid.
    """
    if footprint_um <= 0:
        raise ValueError("Tile footprint must be positive")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("Tile overlap must be in the range [0, 1)")
    lower, upper = sorted((float(start_um), float(stop_um)))
    distance = upper - lower
    if distance <= footprint_um or np.isclose(distance, footprint_um):
        return np.asarray([lower], dtype=float)
    stride_um = footprint_um * (1.0 - overlap_fraction)
    tile_count = _ceil_ratio(distance - footprint_um, stride_um) + 1
    return lower + np.arange(tile_count, dtype=float) * stride_um


def split_stage_scan_bounds(
    start_um: float,
    stop_um: float,
    max_raw_scan_length_um: float,
    configured_overlap_um: float,
    camera_crop_y: int,
    pixel_size_um: float,
    angle_deg: float,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Split a lab-space interval into overlapping stage-scan trajectories.

    Deskewing spreads each oblique camera plane along the laboratory scan axis.
    Adjacent raw trajectories must therefore overlap by that scan-axis camera
    extent *plus* the requested fully sampled overlap.  Using the camera axial
    extent here leaves only ``configured_overlap - (scan_extent - z_extent)``
    of valid overlap and may create seams between reconstructed scan tiles.
    The split trajectories are equal length and their union begins and ends at
    the requested bounds.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float, float]
        Raw scan starts, raw scan ends, effective trajectory overlap, and the
        camera scan-axis footprint, all in micrometers.

    Raises
    ------
    ValueError
        If the interval or scan constraints cannot form a valid plan.
    """
    lower, upper = sorted((float(start_um), float(stop_um)))
    distance_um = upper - lower
    if distance_um <= 0:
        raise ValueError("Stage scan bounds must define a nonzero interval")
    if max_raw_scan_length_um <= 0:
        raise ValueError("Maximum raw stage-scan length must be positive")
    if configured_overlap_um < 0:
        raise ValueError("Configured stage-scan overlap cannot be negative")

    camera_scan_extent_um, _ = oblique_camera_extents_um(
        camera_crop_y, pixel_size_um, angle_deg
    )
    effective_overlap_um = camera_scan_extent_um + float(configured_overlap_um)

    if distance_um <= max_raw_scan_length_um or np.isclose(
        distance_um, max_raw_scan_length_um
    ):
        starts = np.asarray([lower], dtype=float)
        ends = np.asarray([upper], dtype=float)
        return starts, ends, effective_overlap_um, camera_scan_extent_um

    # This is algebraically identical to main:
    #
    #   length = range / count + (count - 1) * overlap / count
    #   stride = length - overlap
    #
    # Retain floating-point micrometre precision here; main's subsequent
    # 0.01 mm rounding was separately fixed because it erased small scans.
    scan_count = int(np.ceil(distance_um / float(max_raw_scan_length_um)))
    raw_scan_length_um = (
        distance_um / scan_count + (scan_count - 1) * effective_overlap_um / scan_count
    )
    raw_stride_um = raw_scan_length_um - effective_overlap_um
    if raw_stride_um <= 0:
        raise ValueError(
            "Camera footprint plus configured overlap is too large for the "
            "calculated stage-scan segment"
        )
    starts = lower + np.arange(scan_count, dtype=float) * raw_stride_um
    ends = starts + raw_scan_length_um
    ends[-1] = upper
    return starts, ends, effective_overlap_um, camera_scan_extent_um


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


def ao_grid_positions(
    stage_positions: Sequence[dict[str, float]],
    num_scan_positions: int,
    num_tile_positions: int,
    *,
    bounds_um: dict[str, Sequence[float]],
    angle_deg: float | None = None,
    camera_zstage_orientation: str | None = None,
) -> list[dict[str, float]]:
    """Sample equal acquisition cells on the existing focus surface.

    Bounds, cells and tile footprints use depth-normalized laboratory scan X
    and stage Y, in micrometers. Physical Z already includes the depth offset.
    Fit that Z surface in the declared lateral frame, then convert each center
    to physical stage X once, using its fitted Z and scan reference. Requested
    sampling density is independent of the number of acquisition tile origins.

    Parameters
    ----------
    stage_positions : Sequence[dict[str, float]]
        Acquisition positions for one logical depth level.
    num_scan_positions : int
        Requested number of AO samples along physical stage X.
    num_tile_positions : int
        Requested number of AO samples along physical stage Y.
    bounds_um : dict[str, Sequence[float]]
        Explicit overall grid bounds with ``x`` and ``y`` pairs. These must
        come from the acquisition plan or declared acquired support.
    angle_deg : float or None
        Calibrated OPM angle; required for compensated source positions.
    camera_zstage_orientation : str or None
        Physical Z direction for depth; required for compensated positions.

    Returns
    -------
    list[dict[str, float]]
        AO target positions in physical stage coordinates.

    Raises
    ------
    ValueError
        If the position list is empty or either requested count is invalid.
    """
    coordinates = positions_array(stage_positions)
    z_values, y_values, _ = coordinates.T
    x_values = np.asarray(
        [position.get("lab_scan_um", position["x"]) for position in stage_positions],
        dtype=float,
    )
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(x_values)):
        raise ValueError("AO surface positions must be finite")
    cells = ao_grid_cells(bounds_um, num_scan_positions, num_tile_positions)
    compensated = any("lab_scan_um" in position for position in stage_positions)
    if compensated and (
        angle_deg is None
        or camera_zstage_orientation is None
        or not all(
            "lab_scan_um" in position and "scan_reference_z_um" in position
            for position in stage_positions
        )
    ):
        raise ValueError(
            "Compensated AO positions require angle, orientation and scan reference"
        )

    # Center coordinates before fitting to keep the intercept well-conditioned
    # for the large absolute positions used by the microscope stages. Include
    # only axes that vary; this also handles one-dimensional and single-position
    # acquisitions without inventing an unconstrained slope.
    x_center = float(np.mean(x_values))
    y_center = float(np.mean(y_values))
    fit_columns = [np.ones(len(coordinates), dtype=float)]
    varying_x = not np.isclose(np.ptp(x_values), 0.0)
    varying_y = not np.isclose(np.ptp(y_values), 0.0)
    if varying_x:
        fit_columns.append(x_values - x_center)
    if varying_y:
        fit_columns.append(y_values - y_center)
    design = np.column_stack(fit_columns)
    coefficients, *_ = np.linalg.lstsq(design, z_values, rcond=None)
    reference_coefficients = None
    if compensated:
        reference_coefficients, *_ = np.linalg.lstsq(
            design,
            [position["scan_reference_z_um"] for position in stage_positions],
            rcond=None,
        )

    targets: list[dict[str, float]] = []
    for x_min, x_max, y_min, y_max in cells:
        x_value, y_value = (x_min + x_max) / 2, (y_min + y_max) / 2
        prediction = [1.0]
        if varying_x:
            prediction.append(float(x_value) - x_center)
        if varying_y:
            prediction.append(float(y_value) - y_center)
        z_value = float(np.dot(coefficients, prediction))
        # A long trajectory can have only one origin along an axis. Retain a
        # known planner surface slope there instead of inventing a flat plane.
        if not varying_x:
            z_value += float(stage_positions[0].get("focus_surface_slope_x", 0.0)) * (
                x_value - x_center
            )
        if not varying_y:
            z_value += float(stage_positions[0].get("focus_surface_slope_y", 0.0)) * (
                y_value - y_center
            )
        target = {"z": z_value, "y": float(y_value), "x": float(x_value)}
        if reference_coefficients is not None:
            target["lab_scan_um"] = float(x_value)
            target["scan_reference_z_um"] = float(
                np.dot(reference_coefficients, prediction)
            )
            target = apply_oblique_scan_correction(
                [target], angle_deg, camera_zstage_orientation
            )[0]
        for key in ("depth_index", "sample_depth_um", "stage_depth_offset_um"):
            if key in stage_positions[0]:
                target[key] = stage_positions[0][key]
        targets.append(
            {
                key: float(np.round(value, 3)) if key != "depth_index" else int(value)
                for key, value in target.items()
            }
        )
    return targets


def ao_grid_cells(
    bounds_um: dict[str, Sequence[float]],
    num_scan_positions: int,
    num_tile_positions: int,
) -> FloatArray:
    """Return equal cells as X-min/X-max/Y-min/Y-max, Y-major then X-minor.

    Parameters
    ----------
    bounds_um : dict[str, Sequence[float]]
        Explicit laboratory scan X and stage Y bounds in micrometers.
    num_scan_positions, num_tile_positions : int
        Positive sampling counts independent of the acquisition tile layout.

    Returns
    -------
    numpy.ndarray
        Cell rectangles in the same frame as tile footprints, without rounding.
    """
    edges = []
    for axis, count in (("x", num_scan_positions), ("y", num_tile_positions)):
        if not np.isfinite(count) or int(count) != count or count < 1:
            raise ValueError("AO grid dimensions must be positive integers")
        pair = np.asarray(bounds_um[axis], dtype=float)
        if pair.shape != (2,) or not np.all(np.isfinite(pair)) or pair[1] <= pair[0]:
            raise ValueError(
                "AO grid bounds must be finite increasing nonzero intervals"
            )
        edges.append(np.linspace(pair[0], pair[1], int(count) + 1))
    x_edges, y_edges = edges
    return np.asarray(
        [
            (x0, x1, y0, y1)
            for y0, y1 in pairwise(y_edges)
            for x0, x1 in pairwise(x_edges)
        ],
        dtype=float,
    )


def tile_footprints_um(
    stage_positions: Sequence[dict[str, float]],
    *,
    scan_displacements_um: ArrayLike,
    camera_scan_extent_um: float,
    tile_width_um: float,
) -> FloatArray:
    """Return camera-extended trajectory rectangles in laboratory scan X/Y.

    Parameters
    ----------
    stage_positions : Sequence[dict[str, float]]
        Commanded scan starts; ``lab_scan_um`` takes precedence over raw X.
    scan_displacements_um : ArrayLike
        Signed commanded endpoint minus start, scalar or one per position.
    camera_scan_extent_um, tile_width_um : float
        Calibrated centered camera extents along scan X and tile Y, in um.

    Returns
    -------
    numpy.ndarray
        X-min/X-max/Y-min/Y-max rectangles, including reverse trajectories.
    """
    positions_array(stage_positions)
    if not np.isfinite(camera_scan_extent_um) or camera_scan_extent_um < 0:
        raise ValueError("Camera scan extent must be finite and nonnegative")
    if not np.isfinite(tile_width_um) or tile_width_um <= 0:
        raise ValueError("Tile width must be finite and positive")
    displacements = np.broadcast_to(
        np.asarray(scan_displacements_um, dtype=float), (len(stage_positions),)
    )
    if not np.all(np.isfinite(displacements)):
        raise ValueError("Scan displacements must be finite")
    starts = np.asarray([p.get("lab_scan_um", p["x"]) for p in stage_positions])
    ends = starts + displacements
    ys = np.asarray([p["y"] for p in stage_positions])
    return np.column_stack(
        (
            np.minimum(starts, ends) - camera_scan_extent_um / 2,
            np.maximum(starts, ends) + camera_scan_extent_um / 2,
            ys - tile_width_um / 2,
            ys + tile_width_um / 2,
        )
    )


def stage_coordinate_metadata(
    position: dict[str, float],
    *,
    angle_deg: float,
    camera_zstage_orientation: str,
    scan_offset_um: float = 0.0,
) -> dict[str, Any]:
    """Describe commanded stage and depth-normalized laboratory coordinates.

    Parameters
    ----------
    position : dict[str, float]
        Commanded tile position with optional retained lab scan/reference Z.
    angle_deg : float
        OPM angle used for the oblique transform.
    camera_zstage_orientation : str
        Stage sign for increasing sample depth.
    scan_offset_um : float
        Signed displacement from the tile origin for the current scan plane.

    Returns
    -------
    dict[str, Any]
        Additional explicit metadata; legacy coordinate fields stay available.
        Planned commands are never represented as measured stage readbacks.
    """
    sign = sample_depth_stage_sign(camera_zstage_orientation)
    reference = float(position.get("scan_reference_z_um", position["z"]))
    normalized_z = -sign * (float(position["z"]) - reference)
    physical = {
        axis: float(
            np.round(position[axis] + (scan_offset_um if axis == "x" else 0), 3)
        )
        for axis in ("x", "y", "z")
    }
    lab = {
        "x": float(
            np.round(position.get("lab_scan_um", position["x"]) + scan_offset_um, 3)
        ),
        "y": physical["y"],
        "z": float(np.round(normalized_z, 3)),
    }
    return {
        "physical_stage_command_um": physical,
        "laboratory_scan_um": lab,
        "stage_readback_um": None,
        "coordinate_transform": {
            "units": "um",
            "physical_frame": "physical_stage_xyz",
            "laboratory_frame": "depth_normalized_laboratory_scan_x_stage_y",
            "angle_deg": float(angle_deg),
            "sample_depth_stage_sign": sign,
            "scan_reference_z_um": float(np.round(reference, 3)),
            "oblique_compensation_applied": "lab_scan_um" in position,
            "equation": "lab_z = -stage_sign * (stage_z - reference_z); stage_x = lab_scan_x - lab_z * cot(angle)",
        },
    }


def rounded_geometry_metadata(value: Any) -> Any:
    """Copy a spatial geometry payload with micrometer coordinates rounded to 3.

    Parameters
    ----------
    value : Any
        Nested spatial bounds, footprints or positions already converted to um.
        Calibration and surface coefficients must be kept outside this payload.

    Returns
    -------
    Any
        Rounded storage copy; the calculation inputs are left unchanged.
    """
    if isinstance(value, dict):
        return {key: rounded_geometry_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded_geometry_metadata(item) for item in value]
    return float(np.round(value, 3)) if isinstance(value, float) else value


def nearest_ao_grid_indices(
    stage_positions: Sequence[dict[str, float]],
    ao_stage_positions: Sequence[dict[str, float]],
    *,
    cell_bounds_um: ArrayLike,
    tile_bounds_um: ArrayLike,
) -> np.ndarray:
    """Assign each entire tile to the AO cell of largest physical overlap.

    The legacy helper name is retained for import compatibility. Both rectangle
    collections must use the same declared laboratory scan X/stage Y frame,
    in micrometers. Numerically tied areas choose the first cell in stable
    Y-major/X-minor grid order. No constraint is imposed on tile counts per cell.

    Parameters
    ----------
    stage_positions : Sequence[dict[str, float]]
        Acquisition positions on one logical depth surface.
    ao_stage_positions : Sequence[dict[str, float]]
        AO optimization positions on the same surface.
    cell_bounds_um, tile_bounds_um : ArrayLike
        AO-cell and full tile rectangles as X-min/X-max/Y-min/Y-max, in um.

    Returns
    -------
    numpy.ndarray
        Integer AO-grid index for each acquisition position.

    Raises
    ------
    ValueError
        If either position collection is empty.
    """
    if not stage_positions:
        raise ValueError("At least one acquisition position is required")
    if not ao_stage_positions:
        raise ValueError("At least one AO-grid position is required")
    cells = np.asarray(cell_bounds_um, dtype=float)
    tiles = np.asarray(tile_bounds_um, dtype=float)
    for rectangles, count in (
        (cells, len(ao_stage_positions)),
        (tiles, len(stage_positions)),
    ):
        if (
            rectangles.shape != (count, 4)
            or not np.all(np.isfinite(rectangles))
            or np.any(rectangles[:, 1] <= rectangles[:, 0])
            or np.any(rectangles[:, 3] <= rectangles[:, 2])
        ):
            raise ValueError(
                "Cell and tile bounds must be finite nonzero rectangles matching their positions"
            )
    x_overlap = np.maximum(
        0,
        np.minimum(tiles[:, None, 1], cells[None, :, 1])
        - np.maximum(tiles[:, None, 0], cells[None, :, 0]),
    )
    y_overlap = np.maximum(
        0,
        np.minimum(tiles[:, None, 3], cells[None, :, 3])
        - np.maximum(tiles[:, None, 2], cells[None, :, 2]),
    )
    areas = x_overlap * y_overlap
    maximum = np.max(areas, axis=1, keepdims=True)
    if np.any(maximum <= 0):
        raise ValueError("Every tile must have positive overlap with the AO grid")
    ties = (areas > 0) & np.isclose(areas, maximum, rtol=1e-12, atol=1e-9)
    return np.argmax(ties, axis=1).astype(int, copy=False)


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
            # ASI scan origins are calculated in millimetres to six decimal
            # places. Preserve that sub-micrometre precision after converting
            # back to micrometres.
            "x": float(np.round(scan_position, 3)),
            "y": float(np.round(tile_position, 3)),
            "z": float(np.round(z_position + z_offset_per_scan_um * scan_index, 3)),
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
    range_y = y_max - y_min
    range_z = abs(z_stop - z_start)
    camera_scan_extent_um, camera_z_extent_um = oblique_camera_extents_um(
        crop_y, pixel_size_um, angle_deg
    )
    y_step_max = crop_x * pixel_size_um * (1 - tile_overlap)
    z_step_max = camera_z_extent_um * (1 - tile_overlap)
    scan_starts, scan_ends, raw_scan_overlap_um, _ = split_stage_scan_bounds(
        x_min,
        x_max,
        max_stage_scan_range_um,
        scan_axis_overlap_um,
        crop_y,
        pixel_size_um,
        angle_deg,
    )
    n_scan = len(scan_starts)
    n_tile = max(1, _ceil_ratio(range_y, y_step_max) + 1)
    n_z = max(1, _ceil_ratio(range_z, z_step_max) + 1)
    x_positions = scan_starts
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
        "camera_scan_extent_um": camera_scan_extent_um,
        "raw_scan_overlap_um": raw_scan_overlap_um,
        "scan_end_positions_um": scan_ends.tolist(),
    }
    return positions, geometry
