"""Pure geometry helpers shared by OPM position scripts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
    """Return sample-depth slab centers relative to the coverslip.

    The endpoints are always retained and intermediate levels are inserted so
    adjacent OPM slabs overlap by at least ``overlap_fraction``.  A zero-width
    range intentionally returns one level, preserving single-layer acquisition.

    Parameters
    ----------
    start_um, end_um : float
        Requested sample depths relative to the fitted coverslip plane.
    axial_footprint_um : float
        Reconstructed laboratory-Z extent of one OPM camera frame.
    overlap_fraction : float
        Fractional overlap between neighboring axial slabs.

    Returns
    -------
    numpy.ndarray
        Depth levels in the user-requested direction, in micrometers.

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

    span_um = abs(float(end_um) - float(start_um))
    if np.isclose(span_um, 0.0):
        return np.asarray([float(start_um)], dtype=float)

    maximum_step_um = float(axial_footprint_um) * (1.0 - overlap_fraction)
    interval_count = max(1, int(np.ceil(span_um / maximum_step_um)))
    return np.linspace(float(start_um), float(end_um), interval_count + 1)


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
        lab_z_offset_um = -stage_sign * (
            float(position["z"]) - scan_reference_z_um
        )
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
    """Split a lab-space interval using the working main-branch placement rule.

    Adjacent stage trajectories overlap by the projected camera axial footprint
    plus the configured extra overlap.  This is the established hardware-tested
    calculation from ``main``.  The split trajectories are equal length and
    their union begins and ends at the requested bounds.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float, float]
        Raw scan starts, raw scan ends, effective trajectory overlap, and the
        camera axial footprint, all in micrometers.

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

    _, camera_z_extent_um = oblique_camera_extents_um(
        camera_crop_y, pixel_size_um, angle_deg
    )
    effective_overlap_um = camera_z_extent_um + float(configured_overlap_um)

    if distance_um <= max_raw_scan_length_um or np.isclose(
        distance_um, max_raw_scan_length_um
    ):
        starts = np.asarray([lower], dtype=float)
        ends = np.asarray([upper], dtype=float)
        return starts, ends, effective_overlap_um, camera_z_extent_um

    # This is algebraically identical to main:
    #
    #   length = range / count + (count - 1) * overlap / count
    #   stride = length - overlap
    #
    # Retain floating-point micrometre precision here; main's subsequent
    # 0.01 mm rounding was separately fixed because it erased small scans.
    scan_count = int(np.ceil(distance_um / float(max_raw_scan_length_um)))
    raw_scan_length_um = (
        distance_um / scan_count
        + (scan_count - 1) * effective_overlap_um / scan_count
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
    return starts, ends, effective_overlap_um, camera_z_extent_um


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
) -> list[dict[str, float]]:
    """Generate AO sample positions on the acquisition's physical XYZ surface.

    The supplied positions represent one logical sample-depth level. Their
    physical Z values already include coverslip tilt, sample-depth offset, and
    oblique-scan correction. Fitting in physical stage coordinates therefore
    preserves the complete X/Y-dependent focus surface at intermediate AO
    sample locations.

    Parameters
    ----------
    stage_positions : Sequence[dict[str, float]]
        Acquisition positions for one logical depth level.
    num_scan_positions : int
        Requested number of AO samples along physical stage X.
    num_tile_positions : int
        Requested number of AO samples along physical stage Y.

    Returns
    -------
    list[dict[str, float]]
        AO target positions in physical stage coordinates.

    Raises
    ------
    ValueError
        If the position list is empty or either requested count is invalid.
    """
    if num_scan_positions < 1 or num_tile_positions < 1:
        raise ValueError("AO grid dimensions must be positive")

    coordinates = positions_array(stage_positions)
    z_values, y_values, x_values = coordinates.T
    unique_x = np.unique(x_values)
    unique_y = np.unique(y_values)

    def _axis_targets(unique_values: FloatArray, count: int) -> FloatArray:
        if len(unique_values) == 1:
            return unique_values.copy()
        count = min(int(count), len(unique_values) + 1)
        if count == 1:
            return np.asarray([np.mean(unique_values)], dtype=float)
        return np.linspace(
            float(unique_values[0]),
            float(unique_values[-1]),
            count + 2,
            endpoint=True,
        )[1:-1]

    target_x = _axis_targets(unique_x, num_scan_positions)
    target_y = _axis_targets(unique_y, num_tile_positions)

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

    targets: list[dict[str, float]] = []
    for y_value in target_y:
        for x_value in target_x:
            prediction = [1.0]
            if varying_x:
                prediction.append(float(x_value) - x_center)
            if varying_y:
                prediction.append(float(y_value) - y_center)
            z_value = float(np.dot(coefficients, prediction))
            targets.append(
                {
                    "z": float(np.round(z_value, 2)),
                    "y": float(np.round(y_value, 2)),
                    "x": float(np.round(x_value, 2)),
                }
            )
    return targets


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
            "x": float(np.round(scan_position, 6)),
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
