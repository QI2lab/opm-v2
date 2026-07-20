"""
Create OPM acquisition event structures.

- Optimize now (o2o3 autofocus and adaptive optics)
- Timelapse (Fast image time series, 2d or 3d with multiple image-mirror positions)
- Projection scan (2d 'sum' projection)
- Mirror scan (3d image-mirror scanning)
- Stage scan (3d stage scanning)

2025/09/05 SJS: Update to use opm_custom_events
"""

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import MappingProxyType as mappingproxy

import numpy as np
from numpy.typing import NDArray
from pymmcore_plus import CMMCorePlus
from tqdm import trange
from useq import (
    AbsolutePosition,
    Channel,
    MDAEvent,
    MDASequence,
    TIntervalLoops,
    ZRelativePositions,
)

from opm_v2.engine.debug_printing import (
    debug as _debug,
)
from opm_v2.engine.debug_printing import (
    info,
    warning,
)
from opm_v2.engine.opm_custom_events import (
    create_ao_grid_event,
    create_ao_mirror_update_event,
    create_ao_optimize_event,
    create_asi_scan_setup_event,
    create_daq_event,
    create_daq_move_event,
    create_fluidics_event,
    create_o2o3_autofocus_event,
    create_stage_event,
)
from opm_v2.handlers.opm_data_handler import OpmDataHandler
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.utils.position_tools import compose_stage_positions

DEBUGGING = True
MAX_IMAGE_MIRROR_RANGE_UM = 250


def debug(header: str, *lines: object) -> None:
    """Log a debug message when module-level debugging is enabled.

    Parameters
    ----------
    header : str
        Message heading.
    *lines : object
        Detail values included in the message.
    """
    _debug(header, *lines, enabled=DEBUGGING)


# ---------------------------------------------------------#
# Helper methods for consistency
# ---------------------------------------------------------#


def stage_positions_from_grid(
    mda_grid_plan,
    mda_z_plan,
    opm_mode: str,
    camera_crop_x: int,
    camera_crop_y: int,
    scan_range_um: float,
    scan_axis_overlap: float | None = 0.2,
    tile_axis_overlap: float | None = 0.2,
    z_axis_overlap: float | None = 0.2,
    coverslip_max_dz: float | None = None,
    coverslip_slope_x: float | None = 0,
    coverslip_slope_y: float | None = 0,
) -> list[dict]:
    """Generate stage positions from the active grid and Z plans.

    TODO: Break up scans based on coverslip slope in each direction.

    Parameters
    ----------
    mda_grid_plan : dict
        Grid bounds selected in the MDA widget.
    mda_z_plan : dict or None
        Optional Z-range plan selected in the MDA widget.
    opm_mode : str
        OPM acquisition mode.
    camera_crop_x : int
        Camera ROI width in pixels.
    camera_crop_y : int
        Camera ROI height in pixels.
    scan_range_um : float
        Scan-axis range in micrometers.
    scan_axis_overlap : Optional[float], optional
        Fractional scan-axis overlap.
    tile_axis_overlap : Optional[float], optional
        Fractional tile-axis overlap.
    z_axis_overlap : Optional[float], optional
        Fractional Z-axis overlap.
    coverslip_max_dz : Optional[float], optional
        Maximum allowed coverslip Z change in micrometers.
    coverslip_slope_x : Optional[float], optional
        Coverslip slope along X.
    coverslip_slope_y : Optional[float], optional
        Coverslip slope along Y.

    Returns
    -------
    List[Dict]
        Stage positions stored as X, Y, and Z dictionaries.
    """
    stage_positions = []
    _mmc = CMMCorePlus.instance()

    if mda_z_plan is not None:
        max_z_pos = float(mda_z_plan["top"])
        min_z_pos = float(mda_z_plan["bottom"])
    else:
        min_z_pos = _mmc.getZPosition()
        max_z_pos = _mmc.getZPosition()

    # Force projection mode to a single z-position
    if "projection" in opm_mode:
        n_z_positions = 1
        z_step_um = 0
    elif ("mirror" in opm_mode) or ("stage" in opm_mode):
        if min_z_pos == max_z_pos:
            n_z_positions = 1
            z_step_um = 0
        else:
            range_z_um = np.round(np.abs(max_z_pos - min_z_pos), 2)
            z_step_max = (
                camera_crop_y
                * _mmc.getPixelSizeUm()
                * (1.0 - z_axis_overlap)
                * np.sin((np.pi / 180.0) * float(30))
            )
            n_z_positions = int(np.ceil(range_z_um / z_step_max))
            z_step_um = np.round(range_z_um / n_z_positions, 2)

    # grab grid plan extents
    min_y_pos = mda_grid_plan["bottom"]
    max_y_pos = mda_grid_plan["top"]
    min_x_pos = mda_grid_plan["left"]
    max_x_pos = mda_grid_plan["right"]
    cs_min_pos = min_z_pos

    # Correct directions for stage moves
    if min_z_pos > max_z_pos:
        z_step_um *= -1
    if min_x_pos > max_x_pos:
        min_x_pos, max_x_pos = max_x_pos, min_x_pos

    # Set grid axes ranges
    range_x_um = np.round(np.abs(max_x_pos - min_x_pos), 2)
    range_y_um = np.round(np.abs(max_y_pos - min_y_pos), 2)

    if range_y_um == 0:
        y_step_um = 0
        n_y_positions = 1
    else:
        y_step_max = camera_crop_x * _mmc.getPixelSizeUm() * (1 - tile_axis_overlap)
        n_y_positions = int(np.ceil(range_y_um / y_step_max))
        y_step_um = np.round(range_y_um / n_y_positions, 2)

    if scan_range_um == 0:
        warning("SCAN RANGE", "Scan range == 0.")
        return []
    else:
        x_step_max = scan_range_um * (1 - scan_axis_overlap)
        n_x_positions = int(np.ceil(range_x_um / x_step_max))
        x_step_um = np.round(range_x_um / n_x_positions, 2)

    if "projection" in opm_mode or "mirror" in opm_mode:
        if x_step_max > MAX_IMAGE_MIRROR_RANGE_UM:
            x_step_max = MAX_IMAGE_MIRROR_RANGE_UM

    # account for coverslip slopes
    if coverslip_slope_x != 0:
        cs_x_max_pos = cs_min_pos + range_x_um * coverslip_slope_x
        cs_x_range_um = np.round(np.abs(cs_x_max_pos - cs_min_pos), 2)
        dz_per_x_tile = np.round(cs_x_range_um / n_x_positions, 2)
    else:
        dz_per_x_tile = 0

    if coverslip_slope_y != 0:
        cs_y_max_pos = cs_min_pos + range_y_um * coverslip_slope_y
        cs_y_range_um = np.round(np.abs(cs_y_max_pos - cs_min_pos), 2)
        dz_per_y_tile = np.round(cs_y_range_um / n_y_positions, 2)
    else:
        dz_per_y_tile = 0

    # populate positions list
    for kk in range(n_z_positions):
        for jj in range(n_y_positions):
            # For mirror and projection modes, generate a snake-pattern grid
            if "mirror" in opm_mode or "projection" in opm_mode:
                # move stage left to right
                if jj % 2 == 0:
                    x_iterator = range(n_x_positions)
                else:
                    x_iterator = reversed(range(n_x_positions))
            else:
                x_iterator = range(n_x_positions)

            for ii in x_iterator:
                stage_positions.append({
                    "x": min_x_pos + ii * x_step_um,
                    "y": min_y_pos + jj * y_step_um,
                    "z": min_z_pos
                    + kk * z_step_um
                    + ii * dz_per_x_tile
                    + jj * dz_per_y_tile,
                })
    debug(
        "XYZ STAGE POSITIONS",
        f"x start: {min_x_pos}",
        f"x end: {max_x_pos}",
        f"y start: {min_y_pos}",
        f"y end: {max_y_pos}",
        f"z position min: {min_z_pos}",
        f"z position max: {max_z_pos}",
        f"scan range (um): {scan_range_um}",
        f"coverslip slope (x/y): {coverslip_slope_x}/{coverslip_slope_y}",
        f"number x tiles: {n_x_positions}",
        f"number y tiles: {n_y_positions}",
        f"number z tiles: {n_z_positions}",
        f"x tile length um: {x_step_um}",
        f"y tile length um: {y_step_um}",
    )

    return stage_positions


def populate_opm_metadata(
    daq_mode: str,
    channel_states: list,
    channel_exposures_ms: list,
    laser_powers: list,
    interleaved: bool,
    blanking: bool,
    current_channel: str,
    exposure_ms: float,
    camera_center_x: int,
    camera_center_y: int,
    camera_crop_x: int,
    camera_crop_y: int,
    offset: float,
    e_to_ADU: float,
    angle_deg: float,
    camera_Zstage_orientation: str,
    camera_XYstage_orientation: str,
    camera_mirror_orientation: str,
    stage_position: dict,
    mirror_voltage: float = None,
    image_mirror_range_um: float = None,
    mirror_step: float = None,
    ao_mirror_coeffs: NDArray = None,
    ao_mirror_volts: NDArray = None,
):
    """Build metadata shared by OPM image events.

    Parameters
    ----------
    daq_mode : str
        DAQ acquisition mode.
    channel_states : list
        Enabled laser-channel flags.
    channel_exposures_ms : list
        Exposure for each channel in milliseconds.
    laser_powers : list
        Power for each laser channel.
    interleaved : bool
        Whether channel frames are interleaved.
    blanking : bool
        Whether camera exposure gates the lasers.
    current_channel : str
        Channel associated with the image event.
    exposure_ms : float
        Camera exposure in milliseconds.
    camera_center_x, camera_center_y : int
        Camera ROI center coordinates in pixels.
    camera_crop_x, camera_crop_y : int
        Camera ROI width and height in pixels.
    offset : float
        Camera conversion offset.
    e_to_ADU : float
        Camera electrons-to-ADU conversion factor.
    angle_deg : float
        OPM imaging angle in degrees.
    camera_Zstage_orientation : str
        Camera-to-Z-stage orientation.
    camera_XYstage_orientation : str
        Camera-to-XY-stage orientation.
    camera_mirror_orientation : str
        Camera-to-image-mirror orientation.
    stage_position : dict
        Current X, Y, and Z stage coordinates.
    mirror_voltage : float or None
        Static image-mirror voltage.
    image_mirror_range_um : float or None
        Image-mirror scan range in micrometers.
    mirror_step : float or None
        Image-mirror scan step in micrometers.
    ao_mirror_coeffs : numpy.ndarray or None
        AO modal coefficients.
    ao_mirror_volts : numpy.ndarray or None
        AO actuator voltages.

    Returns
    -------
    dict
        Structured DAQ, camera, OPM, stage, and AO metadata.
    """
    # Assign the DAQ-image mode specific variables
    image_mirror_position = float(mirror_voltage) if mirror_voltage else None
    image_mirror_range_um = (
        float(image_mirror_range_um) if image_mirror_range_um else None
    )
    image_mirror_step_um = float(mirror_step) if mirror_step else None
    ao_mirror_coeffs = (
        ao_mirror_coeffs.tolist() if ao_mirror_coeffs is not None else None
    )
    ao_mirror_volts = ao_mirror_volts.tolist() if ao_mirror_volts is not None else None
    metadata = {
        "DAQ": {
            "mode": str(daq_mode),
            "image_mirror_position": image_mirror_position,
            "image_mirror_range_um": image_mirror_range_um,
            "image_mirror_step_um": image_mirror_step_um,
            "channel_states": channel_states,
            "exposure_channels_ms": channel_exposures_ms,
            "laser_powers": laser_powers,
            "interleaved": interleaved,
            "blanking": blanking,
            "current_channel": current_channel,
        },
        "Camera": {
            "exposure_ms": float(exposure_ms),
            "camera_center_x": int(camera_center_x),
            "camera_center_y": int(camera_center_y),
            "camera_crop_x": int(camera_crop_x),
            "camera_crop_y": int(camera_crop_y),
            "offset": float(offset),
            "e_to_ADU": float(e_to_ADU),
        },
        "OPM": {
            "angle_deg": float(angle_deg),
            "camera_Zstage_orientation": str(camera_Zstage_orientation),
            "camera_XYstage_orientation": str(camera_XYstage_orientation),
            "camera_mirror_orientation": str(camera_mirror_orientation),
        },
        "Stage": {
            "x_pos": float(stage_position["x"]),
            "y_pos": float(stage_position["y"]),
            "z_pos": float(stage_position["z"]),
        },
        "AO_mirror": {"modal_coeffs": ao_mirror_coeffs, "voltages": ao_mirror_volts},
    }
    return metadata


def get_sequence_plans(sequence: MDASequence) -> dict:
    """Extract active MDA sequence plans.

    Parameters
    ----------
    sequence : MDASequence
        Sequence from the MDA widget.

    Returns
    -------
    dict
        Grid, time, position, and Z plans.
    """
    sequence_dict = json.loads(sequence.model_dump_json())
    return {
        "grid": sequence_dict["grid_plan"],
        "time": sequence_dict["time_plan"],
        "positions": sequence_dict["stage_positions"],
        "z": sequence_dict["z_plan"],
    }


def normalize_ao_mode(mode: str) -> str:
    """Normalize GUI AO labels to the event scheduler vocabulary.

    Parameters
    ----------
    mode : str
        AO frequency label stored by the custom widget.

    Returns
    -------
    str
        Canonical event-builder label.
    """
    return {"at xyz position": "at xyz positions"}.get(mode, mode)


def normalize_autofocus_mode(mode: str) -> str:
    """Normalize GUI autofocus labels to event-builder vocabulary.

    Parameters
    ----------
    mode : str
        O2/O3 frequency label stored by the custom widget.

    Returns
    -------
    str
        Canonical event-builder label.
    """
    return {
        "per timepoint": "at timepoints",
        "per xyz position": "at xyz positions",
    }.get(mode, mode)


def get_indexed_acquisition_order(
    sequence: MDASequence, index_sizes: dict[str, int]
) -> tuple[str, ...]:
    """Determine indexed-axis order selected by the MDA widget.

    Parameters
    ----------
    sequence : MDASequence
        Sequence from the MDA widget.
    index_sizes : dict[str, int]
        Indexed acquisition shape.

    Returns
    -------
    tuple[str, ...]
        Indexed axes in camera frame-arrival order.
    """
    ordered_axes: list[str] = []
    for axis in sequence.axis_order:
        axis_name = str(getattr(axis, "value", axis))
        if axis_name in index_sizes and axis_name not in ordered_axes:
            ordered_axes.append(axis_name)
    ordered_axes.extend(axis for axis in index_sizes if axis not in ordered_axes)
    return tuple(ordered_axes)


def get_channel_plan(
    config: dict,
    mode_key: str,
    round_active_exposures: bool = False,
) -> dict:
    """Read channel settings without mutating configuration lists.

    Parameters
    ----------
    config : dict
        OPM configuration.
    mode_key : str
        Acquisition subsection containing channel settings.
    round_active_exposures : bool
        Whether to round enabled-channel exposures to two decimals.

    Returns
    -------
    dict
        Channel states, powers, exposures, names, and interleaving state.

    Raises
    ------
    Exception
        If all configured laser powers are zero.
    """
    mode_config = config["acq_config"][mode_key]
    channel_states = list(mode_config["channel_states"])
    channel_powers = list(mode_config["channel_powers"])
    channel_exposures_ms = list(mode_config["channel_exposures_ms"])
    channel_names = config["OPM"]["channel_ids"]

    active_channel_exps = []
    for index, channel_is_active in enumerate(channel_states):
        if channel_is_active:
            exposure = channel_exposures_ms[index]
            if round_active_exposures:
                exposure = np.round(exposure, 2)
            active_channel_exps.append(exposure)
        else:
            channel_powers[index] = 0

    if sum(channel_powers) == 0:
        raise Exception("All lasers set to 0!")

    return {
        "states": channel_states,
        "powers": channel_powers,
        "exposures_ms": channel_exposures_ms,
        "names": channel_names,
        "active_exposures_ms": active_channel_exps,
        "active_count": sum(channel_states),
        "active_names": [
            name for is_active, name in zip(channel_states, channel_names) if is_active
        ],
        "active_indices": tuple(
            index for index, is_active in enumerate(channel_states) if is_active
        ),
        "interleaved": len(set(active_channel_exps)) == 1,
    }


def get_camera_conversion(mmc: CMMCorePlus, config: dict) -> tuple[float, float]:
    """Read camera offset and electrons-to-ADU conversion.

    Parameters
    ----------
    mmc : CMMCorePlus
        Active Micro-Manager core.
    config : dict
        OPM configuration.

    Returns
    -------
    tuple[float, float]
        Offset and electrons-to-ADU factor, using neutral values when the
        camera does not expose those properties.
    """
    try:
        offset = mmc.getProperty(
            config["Camera"]["camera_id"], "CONVERSION FACTOR OFFSET"
        )
        e_to_ADU = mmc.getProperty(
            config["Camera"]["camera_id"], "CONVERSION FACTOR COEFF"
        )
    except Exception as exc:
        debug(
            "CAMERA CONVERSION PROPERTIES",
            "Failed to get offset or e_to_adu properties.",
            f"Exception: {exc}",
        )
        offset = 0.0
        e_to_ADU = 1.0

    return float(offset), float(e_to_ADU)


def create_zarr_handler(
    output: Path,
    indice_sizes: dict,
    acquisition_order: tuple[str, ...] | None = None,
    events: list[MDAEvent] | None = None,
    config: dict | None = None,
) -> OpmDataHandler:
    """Create the OPM Zarr handler after validating its output path.

    Parameters
    ----------
    output : Path
        Requested ``.zarr`` or ``.ome.zarr`` destination.
    indice_sizes : dict
        Indexed acquisition shape.
    acquisition_order : tuple[str, ...] or None
        Camera frame-arrival axis order.
    events : list[MDAEvent] or None
        Prepared events supplying OPM semantic metadata.
    config : dict or None
        Complete GUI acquisition configuration stored as global metadata.

    Returns
    -------
    OpmDataHandler
        TensorStore-backed ome-writers output handler.

    Raises
    ------
    Exception
        If the output does not use a supported Zarr suffix.
    """
    output = Path(output)
    if output.name.endswith((".zarr", ".ome.zarr")):
        handler = OpmDataHandler(
            path=output,
            index_sizes=indice_sizes,
            acquisition_order=acquisition_order,
            delete_existing=True,
            events=events,
            acquisition_metadata=config,
        )
        info("QI2LAB HANDLER", f"indices: {indice_sizes}")
        return handler
    raise Exception("Default handler selected, modify save path!")


def single_channel_settings(
    channel_states: list,
    channel_powers: list,
    channel_exposures_ms: list,
    chan_idx: int,
) -> tuple[list, list, list]:
    """Create DAQ lists that enable one channel.

    Parameters
    ----------
    channel_states : list
        Original enabled-channel flags.
    channel_powers : list
        Original channel powers.
    channel_exposures_ms : list
        Original channel exposures.
    chan_idx : int
        Channel to enable.

    Returns
    -------
    tuple[list, list, list]
        Single-channel states, powers, and exposures.
    """
    temp_channels = [False] * len(channel_states)
    temp_exposures = [0] * len(channel_exposures_ms)
    temp_powers = [0] * len(channel_powers)
    temp_channels[chan_idx] = True
    temp_exposures[chan_idx] = channel_exposures_ms[chan_idx]
    temp_powers[chan_idx] = channel_powers[chan_idx]
    return temp_channels, temp_powers, temp_exposures


def clone_event(event: MDAEvent) -> MDAEvent:
    """Return an independent deep copy of a useq event.

    Parameters
    ----------
    event : MDAEvent
        Event to clone.

    Returns
    -------
    MDAEvent
        Independent event copy.
    """
    return event.model_copy(deep=True)


def apply_timepoint_timing(
    events: list[MDAEvent],
    start_index: int,
    time_index: int,
    interval_s: float,
    *,
    reset_event_timer: bool = False,
) -> None:
    """Apply upstream runner timing to the first event in a timepoint.

    Parameters
    ----------
    events : list[MDAEvent]
        Event list containing the completed timepoint.
    start_index : int
        Index at which the timepoint's events begin.
    time_index : int
        Zero-based timepoint index.
    interval_s : float
        Requested interval between timepoints in seconds.
    reset_event_timer : bool
        Whether this event starts a new independent timing series.
    """
    if start_index >= len(events):
        return
    update: dict[str, object] = {}
    if interval_s > 0:
        update["min_start_time"] = float(time_index) * float(interval_s)
    if reset_event_timer:
        update["reset_event_timer"] = True
    if update:
        events[start_index] = events[start_index].model_copy(update=update)


def iter_planned_image_events(
    index_sizes: dict[str, int],
    acquisition_order: tuple[str, ...],
    *,
    time_interval_s: float = 0.0,
) -> Iterator[MDAEvent]:
    """Yield image-plan events using useq's native sequence iterator.

    The returned events are planning records: channel configurations and physical
    positions are removed so OPM can attach its actual hardware state while retaining
    useq's indexing, axis ordering, and timepoint scheduling.

    Parameters
    ----------
    index_sizes : dict[str, int]
        Positive sizes for the ``t``, ``p``, ``c``, and ``z`` axes.
    acquisition_order : tuple[str, ...]
        Axis order in which camera frames are expected.
    time_interval_s : float
        Time interval passed to useq's time plan.

    Yields
    ------
    MDAEvent
        Standard useq event carrying normalized string indices and timing.

    Raises
    ------
    ValueError
        If axis sizes and acquisition order do not describe the same axes.
    """
    sizes = {str(axis): int(size) for axis, size in index_sizes.items() if size > 0}
    if set(sizes) != set(acquisition_order):
        raise ValueError("acquisition_order must contain every planned axis once")

    kwargs: dict[str, object] = {"axis_order": acquisition_order}
    if "p" in sizes:
        kwargs["stage_positions"] = tuple(
            AbsolutePosition(name=str(index)) for index in range(sizes["p"])
        )
    if "c" in sizes:
        kwargs["channels"] = tuple(
            Channel(config=f"opm-{index}") for index in range(sizes["c"])
        )
    if "t" in sizes:
        kwargs["time_plan"] = TIntervalLoops(
            interval=float(time_interval_s), loops=sizes["t"]
        )
    if "z" in sizes:
        kwargs["z_plan"] = ZRelativePositions(
            relative=[float(index) for index in range(sizes["z"])]
        )

    for event in MDASequence(**kwargs).iter_events():
        index = {
            str(getattr(axis, "value", axis)): int(value)
            for axis, value in event.index.items()
        }
        yield event.model_copy(
            update={
                "index": mappingproxy(index),
                "channel": None,
                "x_pos": None,
                "y_pos": None,
                "z_pos": None,
                "exposure": None,
            },
            deep=True,
        )


class AOEventScheduler:
    """Schedule repeated AO custom-event mutations.

    Parameters
    ----------
    ao_mode : str
        AO scheduling mode.
    optimize_event : MDAEvent or None
        Template AO optimization event.
    grid_event : MDAEvent or None
        Template AO grid event.
    """

    def __init__(
        self,
        ao_mode: str,
        optimize_event: MDAEvent | None = None,
        grid_event: MDAEvent | None = None,
    ) -> None:
        """Initialize the AO event scheduler.

        Parameters
        ----------
        ao_mode : str
            AO scheduling mode.
        optimize_event : MDAEvent or None
            Template AO optimization event.
        grid_event : MDAEvent or None
            Template AO grid event.
        """
        self.ao_mode = ao_mode
        self.optimize_event = optimize_event
        self.grid_event = grid_event

    def _event_for_mode(self) -> MDAEvent:
        """Clone the AO template for the current mode.

        Returns
        -------
        MDAEvent
            AO event appropriate for the configured mode.

        Raises
        ------
        ValueError
            If the required event template was not supplied.
        """
        if "grid" in self.ao_mode:
            if self.grid_event is None:
                raise ValueError("AO grid mode requested without a grid event.")
            return clone_event(self.grid_event)
        if self.optimize_event is None:
            raise ValueError("AO optimize mode requested without an optimize event.")
        return clone_event(self.optimize_event)

    def append_timepoint_event(
        self,
        opm_events: list[MDAEvent],
        time_idx: int,
        output_dir: Path,
        optimize_dir_template: str = "time_{time_idx}_ao_optimize",
        grid_dir_template: str = "time_{time_idx}_ao_grid",
        interval: int = 1,
    ) -> None:
        """Append an AO event for a scheduled timepoint.

        Parameters
        ----------
        opm_events : list[MDAEvent]
            Event list to extend.
        time_idx : int
            Current timepoint index.
        output_dir : Path
            Root AO result directory.
        optimize_dir_template : str
            Optimization result-directory template.
        grid_dir_template : str
            Grid result-directory template.
        interval : int
            Number of timepoints between AO events.
        """
        if interval <= 0 or time_idx % interval != 0:
            return

        if self.ao_mode == "at timepoints":
            current_ao_dir = output_dir / Path(
                optimize_dir_template.format(time_idx=time_idx)
            )
            current_ao_dir.mkdir(exist_ok=True)
            ao_event = self._event_for_mode()
            ao_event.action.data["AO"]["output_path"] = current_ao_dir
            ao_event.action.data["AO"]["time_idx"] = time_idx
            opm_events.append(ao_event)
        elif self.ao_mode == "grid at timepoints":
            current_ao_dir = output_dir / Path(
                grid_dir_template.format(time_idx=time_idx)
            )
            current_ao_dir.mkdir(exist_ok=True)
            ao_event = self._event_for_mode()
            ao_event.action.data["AO"]["output_path"] = current_ao_dir
            ao_event.action.data["AO"]["time_idx"] = time_idx
            opm_events.append(ao_event)

    def append_position_update(
        self,
        opm_events: list[MDAEvent],
        pos_idx: int,
        use_position_index_for_optimize: bool = False,
    ) -> None:
        """Append an AO update for a stage position.

        Parameters
        ----------
        opm_events : list[MDAEvent]
            Event list to extend.
        pos_idx : int
            Current stage-position index.
        use_position_index_for_optimize : bool
            Whether optimization events retain the current position index.
        """
        if "grid" in self.ao_mode:
            ao_event = self._event_for_mode()
            ao_event.action.data["AO"]["apply_ao_map"] = True
            ao_event.action.data["AO"]["pos_idx"] = pos_idx
        else:
            ao_event = self._event_for_mode()
            ao_event.action.data["AO"]["apply_existing"] = True
            ao_event.action.data["AO"]["pos_idx"] = (
                int(pos_idx) if use_position_index_for_optimize else 0
            )
        opm_events.append(ao_event)


def create_opm_image_event(
    *,
    index: dict,
    config: dict,
    stage_position: dict,
    daq_mode: str,
    channel_states: list,
    channel_exposures_ms: list,
    laser_powers: list,
    interleaved: bool,
    blanking: bool,
    current_channel: str,
    exposure_ms: float,
    camera_center_x: int,
    camera_center_y: int,
    camera_crop_x: int,
    camera_crop_y: int,
    offset: float,
    e_to_ADU: float,
    mirror_voltage: float = None,
    image_mirror_range_um: float = None,
    mirror_step: float = None,
    ao_mirror_coeffs: NDArray = None,
    ao_mirror_volts: NDArray = None,
) -> MDAEvent:
    """Create an image event with standard OPM metadata.

    Parameters
    ----------
    index : dict
        Indexed acquisition position.
    config : dict
        OPM configuration.
    stage_position : dict
        Current X, Y, and Z stage position.
    daq_mode : str
        DAQ acquisition mode.
    channel_states : list
        Enabled laser-channel flags.
    channel_exposures_ms : list
        Per-channel exposures in milliseconds.
    laser_powers : list
        Per-channel laser powers.
    interleaved : bool
        Whether channel frames are interleaved.
    blanking : bool
        Whether camera exposure gates the lasers.
    current_channel : str
        Channel associated with the event.
    exposure_ms : float
        Camera exposure in milliseconds.
    camera_center_x, camera_center_y : int
        Camera ROI center coordinates.
    camera_crop_x, camera_crop_y : int
        Camera ROI dimensions.
    offset : float
        Camera conversion offset.
    e_to_ADU : float
        Camera electrons-to-ADU factor.
    mirror_voltage : float or None
        Static image-mirror voltage.
    image_mirror_range_um : float or None
        Image-mirror scan range.
    mirror_step : float or None
        Image-mirror scan step.
    ao_mirror_coeffs : numpy.ndarray or None
        AO modal coefficients.
    ao_mirror_volts : numpy.ndarray or None
        AO actuator voltages.

    Returns
    -------
    MDAEvent
        Indexed camera acquisition event.
    """
    return MDAEvent(
        index=mappingproxy(index),
        x_pos=float(stage_position["x"]),
        y_pos=float(stage_position["y"]),
        z_pos=float(stage_position["z"]),
        exposure=float(exposure_ms),
        metadata=populate_opm_metadata(
            daq_mode=daq_mode,
            mirror_voltage=mirror_voltage,
            image_mirror_range_um=image_mirror_range_um,
            mirror_step=mirror_step,
            channel_states=channel_states,
            channel_exposures_ms=channel_exposures_ms,
            laser_powers=laser_powers,
            interleaved=interleaved,
            blanking=blanking,
            current_channel=current_channel,
            exposure_ms=exposure_ms,
            camera_center_x=camera_center_x,
            camera_center_y=camera_center_y,
            camera_crop_x=camera_crop_x,
            camera_crop_y=camera_crop_y,
            offset=offset,
            e_to_ADU=e_to_ADU,
            angle_deg=config["OPM"]["angle_deg"],
            camera_Zstage_orientation=config["OPM"]["camera_Zstage_orientation"],
            camera_XYstage_orientation=config["OPM"]["camera_XYstage_orientation"],
            camera_mirror_orientation=config["OPM"]["camera_mirror_orientation"],
            stage_position=stage_position,
            ao_mirror_coeffs=ao_mirror_coeffs,
            ao_mirror_volts=ao_mirror_volts,
        ),
    )


def create_stage_scan_image_event(
    *,
    index: dict,
    config: dict,
    stage_position: dict,
    scan_axis_idx: int,
    scan_axis_step_um: float,
    channel_states: list,
    channel_exposures_ms: list,
    laser_powers: list,
    laser_blanking: bool,
    current_channel: str,
    exposure_ms: float,
    camera_center_x: int,
    camera_center_y: int,
    camera_crop_x: int,
    camera_crop_y: int,
    offset: float,
    e_to_ADU: float,
    excess_start_images: int,
    excess_end_images: int,
    is_excess_image: bool,
) -> MDAEvent:
    """Create a stage-scan camera event with stage-specific metadata.

    Parameters
    ----------
    index : dict
        Indexed acquisition position.
    config : dict
        OPM configuration.
    stage_position : dict
        Current X, Y, and Z stage position.
    scan_axis_idx : int
        Current scan-axis frame index.
    scan_axis_step_um : float
        Scan-axis sampling step in micrometers.
    channel_states : list
        Enabled laser-channel flags.
    channel_exposures_ms : list
        Per-channel exposures in milliseconds.
    laser_powers : list
        Per-channel laser powers.
    laser_blanking : bool
        Whether camera exposure gates the lasers.
    current_channel : str
        Channel associated with the event.
    exposure_ms : float
        Camera exposure in milliseconds.
    camera_center_x, camera_center_y : int
        Camera ROI center coordinates.
    camera_crop_x, camera_crop_y : int
        Camera ROI dimensions.
    offset : float
        Camera conversion offset.
    e_to_ADU : float
        Camera electrons-to-ADU factor.
    excess_start_images, excess_end_images : int
        Extra camera frames at the scan boundaries.
    is_excess_image : bool
        Whether this frame lies outside the retained scan range.

    Returns
    -------
    MDAEvent
        Indexed camera acquisition event.
    """
    return MDAEvent(
        index=mappingproxy(index),
        x_pos=float(stage_position["x"]),
        y_pos=float(stage_position["y"]),
        z_pos=float(stage_position["z"]),
        exposure=float(exposure_ms),
        metadata={
            "DAQ": {
                "mode": "stage",
                "scan_axis_step_um": float(scan_axis_step_um),
                "channel_states": channel_states,
                "exposure_channels_ms": channel_exposures_ms,
                "interleaved": True,
                "laser_powers": laser_powers,
                "blanking": laser_blanking,
                "current_channel": current_channel,
            },
            "Camera": {
                "exposure_ms": float(exposure_ms),
                "camera_center_x": camera_center_x - int(camera_crop_x // 2),
                "camera_center_y": camera_center_y - int(camera_crop_y // 2),
                "camera_crop_x": int(camera_crop_x),
                "camera_crop_y": int(camera_crop_y),
                "offset": float(offset),
                "e_to_ADU": float(e_to_ADU),
            },
            "OPM": {
                "angle_deg": float(config["OPM"]["angle_deg"]),
                "camera_Zstage_orientation": str(
                    config["OPM"]["camera_Zstage_orientation"]
                ),
                "camera_XYstage_orientation": str(
                    config["OPM"]["camera_XYstage_orientation"]
                ),
                "camera_mirror_orientation": str(
                    config["OPM"]["camera_mirror_orientation"]
                ),
                "excess_scan_positions": int(excess_start_images),
                "excess_scan_end_positions": int(excess_end_images),
                "excess_scan_start_positions": int(excess_start_images),
            },
            "Stage": {
                "x_pos": stage_position["x"] + (scan_axis_idx * scan_axis_step_um),
                "y_pos": stage_position["y"],
                "z_pos": stage_position["z"],
                "excess_image": is_excess_image,
            },
        },
    )


# ---------------------------------------------------------#
# Methods for generating OPM custom acquisitions
# ---------------------------------------------------------#


def setup_optimizenow(
    mmc: CMMCorePlus,
    config: dict,
) -> tuple[list[MDAEvent], None]:
    """Build immediate AO optimization or O2/O3 autofocus events.

    Parameters
    ----------
    mmc : CMMCorePlus
        Active Micro-Manager core.
    config : dict
        OPM configuration loaded from disk.

    Returns
    -------
    tuple[list[MDAEvent], None]
        Custom optimization events and no output handler.
    """
    acq_config = config["acq_config"]
    ao_mode = normalize_ao_mode(acq_config["AO"]["ao_mode"])
    o2o3_mode = normalize_autofocus_mode(acq_config["o2o3_mode"])

    # Sequentially run auto focus then AO optmization
    opm_events: list[MDAEvent] = []

    if "now" in o2o3_mode:
        roi_center_x = acq_config["camera_roi"]["center_x"]
        roi_center_y = acq_config["camera_roi"]["center_y"]
        camera_crop_x = acq_config["camera_roi"]["crop_x"]
        camera_crop_y = config["O2O3-autofocus"]["crop_y"]
        o2o3_event = create_o2o3_autofocus_event(
            exposure_ms=config["O2O3-autofocus"]["exposure_ms"],
            camera_center=[roi_center_x, roi_center_y],
            camera_crop=[camera_crop_x, camera_crop_y],
        )
        opm_events.append(o2o3_event)

    if "now" in ao_mode:
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        ao_output_path = Path(str(config["acq_config"]["AO"]["save_dir_path"])) / Path(
            f"{time}_ao_optimizeNOW"
        )
        ao_optimize_event = create_ao_optimize_event(config, ao_output_path)
        opm_events.append(ao_optimize_event)

    return opm_events, None


def setup_timelapse(
    mmc: CMMCorePlus,
    config: dict,
    sequence: MDASequence,
    output: Path,
) -> tuple[list[MDAEvent], OpmDataHandler]:
    """Timelapse acquisition with optional multiple scan mirror positions.

    This imaging mode holds the mirror static and acquires the N timepoints,
    then moves to a new scan or stage position. Intended for acquiring
    fast static 2D images in sequence.

    For a static 2D timelapse, set image mirror range to 0.
    Produces an event structure where for each scan mirror position,
    N timepoints are acquired for each active channel.

    NOTE: Not tested with multiple channels

    t / c / z / p


    Parameters
    ----------
    mmc : CMMCorePlus
        MMC core in use
    config : dict
        OPM config loaded from disk
    sequence : MDASequence
        MDA sequence to run
    output : Path
        Output path from MDA widget

    Returns
    -------
    tuple[list[MDAEvent], OpmDataHandler]
        Camera acquisition events and TensorStore-backed output handler.

    Raises
    ------
    Exception
        If required MDA position or time plans are missing.
    """
    debug("SETUP TIMELAPSE", "Setting up a timelapse acquisition.")

    OPMdaq_setup = OPMNIDAQ.instance()
    opm_events = []

    # --------------------------------------------------------------------#
    # Compile acquisition settings from configuration
    # --------------------------------------------------------------------#

    # Get the acquisition modes
    acq_config = config["acq_config"]
    daq_config = acq_config["DAQ"]
    ao_mode = normalize_ao_mode(acq_config["AO"]["ao_mode"])
    o2o3_mode = normalize_autofocus_mode(acq_config["o2o3_mode"])

    # Get the camera crop values
    camera_crop_y = int(acq_config["camera_roi"]["crop_y"])
    camera_crop_x = int(acq_config["camera_roi"]["crop_x"])
    camera_center_y = int(acq_config["camera_roi"]["center_y"])
    camera_center_x = int(acq_config["camera_roi"]["center_x"])

    # ----------------------------------------------------------------#
    # Get channel and camera metadata settings
    channel_plan = get_channel_plan(config, "DAQ")
    channel_states = channel_plan["states"]
    channel_powers = channel_plan["powers"]
    channel_exposures_ms = channel_plan["exposures_ms"]
    channel_names = channel_plan["names"]
    active_channel_indices = channel_plan["active_indices"]
    n_active_channels = channel_plan["active_count"]
    active_channel_exps = channel_plan["active_exposures_ms"]
    interleaved_acq = channel_plan["interleaved"]
    offset, e_to_ADU = get_camera_conversion(mmc, config)

    # ----------------------------------------------------------------#
    # get the scan mirror positions
    scan_range_um = daq_config["scan_range_um"]
    scan_step_um = daq_config["scan_axis_step_um"]
    laser_blanking = daq_config["laser_blanking"]

    # get the number of scan steps as expected by the daq
    if scan_range_um == 0.0:
        # setup daq for 2d scan
        scan_mode = "2d"
        OPMdaq_setup.set_acquisition_params(
            scan_type=scan_mode,
            channel_states=channel_states,
        )
        n_scan_steps = 1
        mirror_voltages = np.array([config["NIDAQ"]["image_mirror_neutral_v"]])
    else:
        scan_mode = "mirror"
        OPMdaq_setup.set_acquisition_params(
            scan_type=scan_mode,
            channel_states=channel_states,
            image_mirror_range_um=scan_range_um,
            image_mirror_step_um=scan_step_um,
        )
        OPMdaq_setup.generate_waveforms()
        mirror_voltages = np.unique(OPMdaq_setup.analog_waveform[:, 0])
        n_scan_steps = mirror_voltages.shape[0]

    # --------------------------------------------------------------------#
    # Compile mda acquisition settings from active tabs
    # --------------------------------------------------------------------#

    # Split apart sequence dictionary
    sequence_plans = get_sequence_plans(sequence)
    mda_positions_plan = sequence_plans["positions"]
    mda_time_plan = sequence_plans["time"]

    if (mda_positions_plan is None) or (mda_time_plan is None):
        raise Exception("Must select MDA Positions AND Time plan")

    # ----------------------------------------------------------------#
    # Create custom events
    # ----------------------------------------------------------------#

    # ----------------------------------------------------------------#
    # Create DAQ event
    daq_event = create_daq_event(
        "2d",
        channel_states=channel_states,
        channel_powers=channel_powers,
        channel_exposures_ms=channel_exposures_ms,
        camera_center=[camera_center_x, camera_center_y],
        camera_crop=[camera_crop_x, camera_crop_y],
        interleaved=interleaved_acq,
        laser_blanking=laser_blanking,
    )

    # ----------------------------------------------------------------#
    # Create the o2o3 AF event data
    if o2o3_mode != "none":
        af_camera_crop_y = config["O2O3-autofocus"]["crop_y"]
        o2o3_event = create_o2o3_autofocus_event(
            exposure_ms=config["O2O3-autofocus"]["exposure_ms"],
            camera_center=[camera_center_x, camera_center_y],
            camera_crop=[camera_crop_x, af_camera_crop_y],
        )

    # ----------------------------------------------------------------#
    # Create the AO custom events
    AOmirror_setup = AOMirror.instance()

    # Create an OA optimization event
    if ao_mode != "none":
        if "grid" in ao_mode:
            ao_mode = "per xyz position"
            info(
                "AO MODE UPDATED",
                "AO grid selected.",
                "Running optimization at each XYZ position.",
            )
        ao_root_dir = output.parent / Path(f"{output.stem}_ao_results")
        ao_root_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------#
    # Compile tine points and positions from active MDA tabs and config
    # ----------------------------------------------------------------#

    # ----------------------------------------------------------------#
    # Get time points
    n_time_steps = mda_time_plan["loops"]
    time_interval = float(mda_time_plan["interval"])

    # estimate the timeloop duration
    estimated_loop_duration_s = (
        (sum(active_channel_exps) / 1000.0) * n_time_steps + 1.0  # daq start/stop
    )

    # If the timelapse loop is longer tan 6 hours,
    # include a AO mirror update in the middle of the loop
    if estimated_loop_duration_s > 6 * 60 * 60:
        update_ao_mirror_mid_loop = True
    else:
        update_ao_mirror_mid_loop = False

    # ----------------------------------------------------------------#
    # Get xyz stage position
    stage_positions = []
    for stage_pos in mda_positions_plan:
        stage_positions.append({
            "x": float(stage_pos["x"]),
            "y": float(stage_pos["y"]),
            "z": float(stage_pos["z"]),
        })
    n_stage_positions = len(stage_positions)

    # ----------------------------------------------------------------#
    # Create MDA event structure
    # ----------------------------------------------------------------#

    opm_events: list[MDAEvent] = []

    # move stage to starting position
    opm_events.append(create_stage_event(stage_positions[0]))

    # ----------------------------------------------------------------#
    # Check for optimization at start

    if "start" in o2o3_mode:
        opm_events.append(o2o3_event)
    if "start" in ao_mode:
        current_ao_dir = ao_root_dir / Path("start_ao_optimize")
        current_ao_event = create_ao_optimize_event(config, current_ao_dir)
        opm_events.append(current_ao_event)

    # ----------------------------------------------------------------#
    # setup np x 1C x nT x nZ mirror-based AO-OPM acquisition event structure

    debug(
        "TIMELAPSE ACQUISITION SHAPE",
        f"timepoints: {n_time_steps}",
        f"num scan positions: {n_scan_steps}",
        f"stage positions: {n_stage_positions}",
        f"active channels: {n_active_channels}",
        f"estimated loop duration (s): {estimated_loop_duration_s:.2f}",
        f"update AO mid loop: {update_ao_mirror_mid_loop}",
    )

    for pos_idx in trange(n_stage_positions, desc="Stage positions:", leave=True):
        if pos_idx > 0:
            opm_events.append(create_stage_event(stage_positions[pos_idx]))

        # Loop over mirror scan positions
        for scan_idx in trange(
            n_scan_steps, desc="Mirror scan positions:", leave=False
        ):
            # ------------------------------------------------------------#
            # Move the image mirror to position
            daq_move_event = create_daq_move_event(mirror_voltages[scan_idx])
            opm_events.append(daq_move_event)

            # ------------------------------------------------------------#
            # Check for optimization per position (Stage or Mirror),
            # or update the AO mirror state
            # NOTE: Currently, the AO is running every Stage and Mirror Position
            #       Because the time for a single Mirror scan can be hours long.
            if "xyz" in o2o3_mode:
                opm_events.append(o2o3_event)
            if "xyz" in ao_mode:
                current_ao_dir = ao_root_dir / Path(
                    f"stage_pos_{pos_idx}_scan_pos_{scan_idx}_ao_optimize"
                )
                current_ao_event = create_ao_optimize_event(config, current_ao_dir)
                current_ao_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                current_ao_event.action.data["AO"]["scan_idx"] = int(scan_idx)
                opm_events.append(current_ao_event)
            else:
                current_coeffs = AOmirror_setup.current_coeffs.copy()
                ao_mirror_update = create_ao_mirror_update_event(
                    mirror_coeffs=current_coeffs
                )
                opm_events.append(ao_mirror_update)

            # ------------------------------------------------------------#
            # acquire sequenced timelapse images
            # Update daq to perform a 2d scan
            opm_events.append(daq_event)
            for time_idx in trange(n_time_steps, desc="Timepoints:", leave=False):
                timepoint_start_index = len(opm_events)
                # Check for AO update mid loop
                if update_ao_mirror_mid_loop and time_idx == int(n_time_steps // 2):
                    current_coeffs = AOmirror_setup.current_coeffs.copy()
                    ao_mirror_update = create_ao_mirror_update_event(
                        mirror_coeffs=current_coeffs
                    )
                    opm_events.append(ao_mirror_update)

                # Create the timelapse image events
                for planned_event in iter_planned_image_events(
                    {"c": n_active_channels}, ("c",)
                ):
                    current_chan_idx = planned_event.index["c"]
                    chan_idx = active_channel_indices[current_chan_idx]
                    image_event = create_opm_image_event(
                        index={
                            "t": time_idx,
                            "p": pos_idx,
                            "c": current_chan_idx,
                            "z": scan_idx,
                        },
                        config=config,
                        stage_position=stage_positions[pos_idx],
                        daq_mode="2d",
                        mirror_voltage=mirror_voltages[scan_idx],
                        mirror_step=scan_step_um,
                        channel_states=channel_states,
                        channel_exposures_ms=channel_exposures_ms,
                        laser_powers=channel_powers,
                        interleaved=interleaved_acq,
                        blanking=laser_blanking,
                        current_channel=channel_names[chan_idx],
                        exposure_ms=channel_exposures_ms[chan_idx],
                        camera_center_x=camera_center_x,
                        camera_center_y=camera_center_y,
                        camera_crop_x=camera_crop_x,
                        camera_crop_y=camera_crop_y,
                        offset=offset,
                        e_to_ADU=e_to_ADU,
                    )
                    opm_events.append(image_event)
                apply_timepoint_timing(
                    opm_events,
                    timepoint_start_index,
                    time_idx,
                    time_interval,
                    reset_event_timer=time_idx == 0,
                )

    # ----------------------------------------------------------------#
    # Setup OPM custom handler
    # NOTE: output path needs to only have a single '.', or multiple suffixes are found!
    indice_sizes = {
        "t": int(np.maximum(1, n_time_steps)),
        "p": int(n_stage_positions),
        "c": int(np.maximum(1, n_active_channels)),
        "z": int(np.maximum(1, n_scan_steps)),
    }
    return opm_events, create_zarr_handler(
        output,
        indice_sizes,
        acquisition_order=("p", "z", "t", "c"),
        events=opm_events,
        config=config,
    )


def setup_projection(
    mmc: CMMCorePlus,
    config: dict,
    sequence: MDASequence,
    output: Path,
) -> tuple[list[MDAEvent], OpmDataHandler]:
    """OPM projection scan acquisition.

    Creates an event structure:
    t / p / c

    Parameters
    ----------
    mmc : CMMCorePlus
        MMC core in use
    config : dict
        OPM config loaded from disk
    sequence : MDASequence
        MDA sequence to run
    output : Path
        Output path from MDA widget

    Returns
    -------
    tuple[list[MDAEvent], OpmDataHandler]
        Camera acquisition events and TensorStore-backed output handler.

    Raises
    ------
    Exception
        If neither grid nor position plans are configured.
    """
    AOmirror_setup = AOMirror.instance()
    # TODO: add as an option to the OPM setup config
    timepoint_interval = 6

    # --------------------------------------------------------------------#
    # Compile acquisition settings from configuration
    # --------------------------------------------------------------------#
    # Get the acquisition modes
    acq_config = config["acq_config"]
    daq_config = acq_config["DAQ"]
    positions_config = acq_config["Positions"]
    opm_mode = acq_config["opm_mode"]
    ao_mode = normalize_ao_mode(acq_config["AO"]["ao_mode"])
    o2o3_mode = normalize_autofocus_mode(acq_config["o2o3_mode"])
    fluidics_mode = acq_config["fluidics"]

    # Get pixel size
    pixel_size_um = np.round(float(mmc.getPixelSizeUm()), 3)  # unit: um

    # Get the scan range, coverslip slope and overlaps
    coverslip_slope_x = positions_config["coverslip_slope_x"]
    coverslip_slope_y = positions_config["coverslip_slope_y"]
    scan_range_um = float(daq_config["scan_range_um"])
    tile_axis_overlap = float(positions_config["tile_axis_overlap"])

    # Get the camera crop values
    camera_crop_y = int(scan_range_um / pixel_size_um)
    camera_crop_x = int(acq_config["camera_roi"]["crop_x"])
    camera_center_y = int(acq_config["camera_roi"]["center_y"])
    camera_center_x = int(acq_config["camera_roi"]["center_x"])

    # Get channel and camera metadata settings
    laser_blanking = daq_config["laser_blanking"]
    channel_plan = get_channel_plan(config, "DAQ", round_active_exposures=True)
    channel_states = channel_plan["states"]
    channel_powers = channel_plan["powers"]
    channel_exposures_ms = channel_plan["exposures_ms"]
    channel_names = channel_plan["names"]
    active_channel_indices = channel_plan["active_indices"]
    n_active_channels = channel_plan["active_count"]
    interleaved_acq = channel_plan["interleaved"]
    offset, e_to_ADU = get_camera_conversion(mmc, config)

    # --------------------------------------------------------------------#
    # Compile mda acquisition settings from active tabs
    # --------------------------------------------------------------------#

    # Split apart sequence dictionary
    sequence_plans = get_sequence_plans(sequence)
    mda_grid_plan = sequence_plans["grid"]
    mda_time_plan = sequence_plans["time"]
    mda_positions_plan = sequence_plans["positions"]
    mda_z_plan = sequence_plans["z"]

    # ----------------------------------------------------------------#
    # Generate xyz stage positions
    # ----------------------------------------------------------------#
    if (mda_grid_plan is None) and (mda_positions_plan is None):
        raise Exception("Must select MDA grid or Positions plan")

    if mda_grid_plan is not None:
        stage_positions = stage_positions_from_grid(
            opm_mode=opm_mode,
            mda_grid_plan=mda_grid_plan,
            mda_z_plan=mda_z_plan,
            camera_crop_x=camera_crop_x,
            camera_crop_y=camera_crop_y,
            scan_range_um=scan_range_um,
            tile_axis_overlap=tile_axis_overlap,
            scan_axis_overlap=tile_axis_overlap,
            coverslip_slope_x=coverslip_slope_x,
            coverslip_slope_y=coverslip_slope_y,
        )
    elif mda_positions_plan is not None:
        stage_positions = []
        for stage_pos in mda_positions_plan:
            stage_positions.append({
                "x": float(stage_pos["x"]),
                "y": float(stage_pos["y"]),
                "z": float(stage_pos["z"]),
            })
    n_stage_positions = len(stage_positions)

    # ----------------------------------------------------------------#
    # Define the time indexing, check for fluidics
    # ----------------------------------------------------------------#
    if fluidics_mode != "none":
        fluidics_rounds = int(fluidics_mode)
        n_time_steps = fluidics_rounds
        time_interval = 0
    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan["loops"]
        time_interval = mda_time_plan["interval"]
    else:
        n_time_steps = 1
        time_interval = 0

    # --------------------------------------------------------------------#
    # Create custom action data
    # --------------------------------------------------------------------#

    # ----------------------------------------------------------------#
    # Create DAQ event
    # ----------------------------------------------------------------#
    daq_event = create_daq_event(
        mode="projection",
        channel_states=channel_states,
        channel_powers=channel_powers,
        channel_exposures_ms=channel_exposures_ms,
        camera_center=[camera_center_x, camera_center_y],
        camera_crop=[camera_crop_x, camera_crop_y],
        interleaved=interleaved_acq,
        laser_blanking=laser_blanking,
        image_mirror_range_um=scan_range_um,
    )

    # ----------------------------------------------------------------#
    # Create the initial AO event data
    # ----------------------------------------------------------------#
    ao_starting_coeffs = AOmirror_setup.current_coeffs.copy()
    ao_starting_volts = AOmirror_setup.current_voltage.copy()

    ao_optimize_event = None
    ao_grid_event = None
    if ao_mode != "none":
        # Create root directory to store AO opt.results
        if "grid" in ao_mode:
            ao_output_dir = output.parent / Path("ao_grid_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_grid_event = create_ao_grid_event(config, ao_output_dir)
            ao_grid_event.action.data["AO"]["stage_positions"] = stage_positions

        else:
            ao_output_dir = output.parent / Path("ao_optimized_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_optimize_event = create_ao_optimize_event(config, ao_output_dir)
        # Update the mirror's position array
        AOmirror_setup.n_positions = n_stage_positions
    ao_scheduler = AOEventScheduler(ao_mode, ao_optimize_event, ao_grid_event)

    # ----------------------------------------------------------------#
    # Create the o2o3 AF event data
    # ----------------------------------------------------------------#
    if o2o3_mode != "none":
        af_camera_crop_y = config["O2O3-autofocus"]["crop_y"]
        o2o3_event = create_o2o3_autofocus_event(
            exposure_ms=config["O2O3-autofocus"]["exposure_ms"],
            camera_center=[camera_center_x, camera_center_y],
            camera_crop=[camera_crop_x, af_camera_crop_y],
        )

    # --------------------------------------------------------------------#
    # Create MDA event structure, Nt/ Np / Nc
    # --------------------------------------------------------------------#
    opm_events: list[MDAEvent] = []

    debug(
        "PROJECTION ACQUISITION PARAMETERS",
        f"timepoints / interval: {n_time_steps} / {time_interval}",
        f"stage positions: {n_stage_positions}",
        f"active channels: {n_active_channels}",
        f"o2o3 focus frequency: {o2o3_mode}",
        f"AO frequency: {ao_mode}",
        "AO starting state:",
        f"coeffs: {ao_starting_coeffs}",
        f"volts: {ao_starting_volts}",
    )

    for time_idx in trange(n_time_steps, desc="Timepoints:", leave=True):
        timepoint_start_index = len(opm_events)
        # --------------------------------------------------------------------#
        # Create events to run before acquisition
        # --------------------------------------------------------------------#
        if time_idx == 0:
            if o2o3_mode == "once at start":
                opm_events.append(o2o3_event)

            if ao_mode == "once at start":
                # Move stage to middle position and run AO once, move stage back
                opm_events.append(
                    create_stage_event(stage_positions[n_stage_positions // 2])
                )
                opm_events.append(clone_event(ao_optimize_event))

            elif ao_mode == "grid at start":
                opm_events.append(clone_event(ao_grid_event))

            # Move stage to starting position
            opm_events.append(create_stage_event(stage_positions[0]))

        # --------------------------------------------------------------------#
        # Create events to run each time-point
        # --------------------------------------------------------------------#
        # Run fluidics if requested
        if fluidics_mode != "none" and time_idx != 0:
            current_fluidics_event = create_fluidics_event(fluidics_rounds, time_idx)
            opm_events.append(current_fluidics_event)

        # Create autofocus event
        if o2o3_mode == "at timepoints":
            opm_events.append(o2o3_event)

        # Create AO optimization events
        ao_scheduler.append_timepoint_event(
            opm_events,
            time_idx,
            ao_output_dir if ao_mode != "none" else output.parent,
            optimize_dir_template="time_{time_idx}_ao_results",
            grid_dir_template="time_{time_idx}_ao_grid_results",
            interval=timepoint_interval,
        )

        # Update the mirror state if not running optimization.
        if ao_mode == "none":
            opm_events.append(
                create_ao_mirror_update_event(mirror_coeffs=ao_starting_coeffs.copy())
            )

        # --------------------------------------------------------------------#
        # iterate over stage positions
        # --------------------------------------------------------------------#
        for pos_idx in range(n_stage_positions):
            # Move stage to position
            opm_events.append(create_stage_event(stage_positions[pos_idx]))

            # Auto-focus event at xyz positions
            if o2o3_mode == "at xyz positions":
                opm_events.append(o2o3_event)

            # ----------------------------------------------------------------#
            # AO mirror events
            # ----------------------------------------------------------------#

            # Run at xyz position optimizations
            if (ao_mode == "at xyz positions") and (time_idx == 0):
                current_ao_dir = ao_output_dir / Path(
                    f"time_{time_idx}_pos_{pos_idx}_ao_results"
                )
                current_ao_dir.mkdir(exist_ok=True)
                current_ao_event = clone_event(ao_optimize_event)
                current_ao_event.action.data["AO"]["output_path"] = current_ao_dir
                current_ao_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                current_ao_event.action.data["AO"]["time_idx"] = int(time_idx)
                current_ao_event.action.data["AO"]["apply_existing"] = False
                opm_events.append(current_ao_event)

            # update mirror state using grid optimization
            if (ao_mode == "grid at start") or (ao_mode == "grid at timepoints"):
                ao_scheduler.append_position_update(opm_events, pos_idx)

            # update mirror state from optimized per xyz position
            if ao_mode == "at xyz positions":
                ao_scheduler.append_position_update(
                    opm_events,
                    pos_idx,
                    use_position_index_for_optimize=True,
                )

            if "xyz" in ao_mode and ao_mode != "at xyz positions":
                # Run the ao optmization at the first time-point for each position
                if time_idx == 0:
                    current_ao_dir = ao_output_dir / Path(f"pos_{pos_idx}_ao_results")
                    current_ao_dir.mkdir(exist_ok=True)
                    current_ao_event = clone_event(ao_optimize_event)
                    current_ao_event.action.data["AO"]["output_path"] = current_ao_dir
                    current_ao_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                    current_ao_event.action.data["AO"]["time_idx"] = int(time_idx)
                    current_ao_event.action.data["AO"]["apply_existing"] = False
                    opm_events.append(current_ao_event)

                # Update the mirror state for at each position for all time-points
                current_ao_event = clone_event(ao_optimize_event)
                current_ao_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                current_ao_event.action.data["AO"]["apply_existing"] = True
                opm_events.append(current_ao_event)

            # ----------------------------------------------------------------#
            # Handle acquiring images
            # ----------------------------------------------------------------#

            if interleaved_acq:
                # Update daq state to sequence all channels
                opm_events.append(clone_event(daq_event))

            # Create image events
            for planned_event in iter_planned_image_events(
                {"c": n_active_channels}, ("c",)
            ):
                current_chan_idx = planned_event.index["c"]
                chan_idx = active_channel_indices[current_chan_idx]
                if not interleaved_acq:
                    # Update daq for each channel separately
                    temp_channels, temp_powers, temp_exposures = (
                        single_channel_settings(
                            channel_states,
                            channel_powers,
                            channel_exposures_ms,
                            chan_idx,
                        )
                    )

                    # Create daq event for a single channel
                    current_daq_event = clone_event(daq_event)
                    current_daq_event.action.data["DAQ"]["channel_states"] = (
                        temp_channels
                    )
                    current_daq_event.action.data["DAQ"]["channel_powers"] = temp_powers
                    current_daq_event.action.data["Camera"]["exposure_channels"] = (
                        temp_exposures
                    )
                    opm_events.append(current_daq_event)

                # Create image event for current t / p / c
                image_event = create_opm_image_event(
                    index={"t": time_idx, "p": pos_idx, "c": current_chan_idx},
                    config=config,
                    stage_position=stage_positions[pos_idx],
                    daq_mode="projection",
                    image_mirror_range_um=scan_range_um,
                    channel_states=channel_states,
                    channel_exposures_ms=channel_exposures_ms,
                    laser_powers=channel_powers,
                    interleaved=interleaved_acq,
                    blanking=laser_blanking,
                    current_channel=channel_names[chan_idx],
                    exposure_ms=channel_exposures_ms[chan_idx],
                    camera_center_x=camera_center_x,
                    camera_center_y=camera_center_y,
                    camera_crop_x=camera_crop_x,
                    camera_crop_y=camera_crop_y,
                    offset=offset,
                    e_to_ADU=e_to_ADU,
                )
                opm_events.append(image_event)

        apply_timepoint_timing(
            opm_events,
            timepoint_start_index,
            time_idx,
            float(time_interval),
        )

    # Save events for debugging
    # --------------------------------------------------------------------#
    # Setup our OPM Engine
    # --------------------------------------------------------------------#
    # NOTE: output path needs to only have a single '.', or multiple suffixes are found!
    indice_sizes = {
        "t": int(np.maximum(1, n_time_steps)),
        "p": int(np.maximum(1, n_stage_positions)),
        "c": int(np.maximum(1, n_active_channels)),
    }
    return opm_events, create_zarr_handler(
        output, indice_sizes, events=opm_events, config=config
    )


def setup_mirrorscan(
    mmc: CMMCorePlus,
    config: dict,
    sequence: MDASequence,
    output: Path,
) -> tuple[list[MDAEvent], OpmDataHandler]:
    """Create an OPM mirror-scan acquisition.

    For a static mirror acquisition set image range to 0.
    When mirror scan range == 0, produces an image sequence similar to Timelapse

    t / p / c / z

    Parameters
    ----------
    mmc : CMMCorePlus
        MMC core in use
    config : dict
        OPM config loaded from disk
    sequence : MDASequence
        MDA sequence to run
    output : Path
        Output path from MDA widget

    Returns
    -------
    tuple[list[MDAEvent], OpmDataHandler]
        Camera acquisition events and TensorStore-backed output handler.

    Raises
    ------
    Exception
        If neither grid nor position plans are configured.
    """
    AOmirror_setup = AOMirror.instance()
    OPMdaq_setup = OPMNIDAQ.instance()

    # --------------------------------------------------------------------#
    # Compile acquisition settings from configuration
    # --------------------------------------------------------------------#
    # Get the acquisition modes
    acq_config = config["acq_config"]
    daq_config = acq_config["DAQ"]
    positions_config = acq_config["Positions"]
    opm_mode = acq_config["opm_mode"]
    ao_mode = normalize_ao_mode(acq_config["AO"]["ao_mode"])
    o2o3_mode = normalize_autofocus_mode(acq_config["o2o3_mode"])
    fluidics_mode = acq_config["fluidics"]

    # Get the scan range, coverslip slope and overlaps
    coverslip_slope_x = positions_config["coverslip_slope_x"]
    coverslip_slope_y = positions_config["coverslip_slope_y"]
    scan_range_um = float(daq_config["scan_range_um"])
    scan_step_um = float(daq_config["scan_axis_step_um"])
    tile_axis_overlap = float(positions_config["tile_axis_overlap"])
    z_axis_overlap = float(positions_config["z_axis_overlap"])

    # Flag for setting up a static mirror acquisition
    if scan_range_um == 0.0:
        scan_mode = "2d"
        info("SCAN MODE", "Setting up a 2d scan mode.")
        OPMdaq_setup.set_acquisition_params(scan_type="2d")
        n_scan_steps = 1
    else:
        scan_mode = "mirror"
        OPMdaq_setup.set_acquisition_params(
            scan_type="mirror",
            image_mirror_range_um=scan_range_um,
            image_mirror_step_um=scan_step_um,
        )
        n_scan_steps = OPMdaq_setup.n_scan_steps

    # Get the camera crop values
    camera_crop_y = int(acq_config["camera_roi"]["crop_y"])
    camera_crop_x = int(acq_config["camera_roi"]["crop_x"])
    camera_center_y = int(acq_config["camera_roi"]["center_y"])
    camera_center_x = int(acq_config["camera_roi"]["center_x"])

    # Get channel and camera metadata settings
    laser_blanking = daq_config["laser_blanking"]
    channel_plan = get_channel_plan(config, "DAQ", round_active_exposures=True)
    channel_states = channel_plan["states"]
    channel_powers = channel_plan["powers"]
    channel_exposures_ms = channel_plan["exposures_ms"]
    channel_names = channel_plan["names"]
    active_channel_indices = channel_plan["active_indices"]
    n_active_channels = channel_plan["active_count"]
    interleaved_acq = channel_plan["interleaved"]
    offset, e_to_ADU = get_camera_conversion(mmc, config)

    # --------------------------------------------------------------------#
    # Compile mda acquisition settings from active tabs
    # --------------------------------------------------------------------#

    # Split apart sequence dictionary
    sequence_plans = get_sequence_plans(sequence)
    mda_grid_plan = sequence_plans["grid"]
    mda_time_plan = sequence_plans["time"]
    mda_positions_plan = sequence_plans["positions"]
    mda_z_plan = sequence_plans["z"]

    if (mda_grid_plan is None) and (mda_positions_plan is None):
        raise Exception("Must select MDA grid or positions plan for mirror scanning")

    # ----------------------------------------------------------------#
    # Create custom action data
    # ----------------------------------------------------------------#

    # ----------------------------------------------------------------#
    # Create DAQ event
    daq_event = create_daq_event(
        mode=scan_mode,
        channel_states=channel_states,
        channel_powers=channel_powers,
        channel_exposures_ms=channel_exposures_ms,
        image_mirror_range_um=scan_range_um,
        image_mirror_step_um=scan_step_um,
        camera_center=[camera_center_x, camera_center_y],
        camera_crop=[camera_crop_x, camera_crop_y],
        interleaved=interleaved_acq,
        laser_blanking=laser_blanking,
    )

    # ----------------------------------------------------------------#
    # Create the AO event data
    ao_optimize_event = None
    ao_grid_event = None
    if ao_mode != "none":
        if "grid" in ao_mode:
            ao_output_dir = output.parent / Path(f"{output.stem}_ao_grid_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_grid_event = create_ao_grid_event(config, ao_output_dir)
        else:
            ao_output_dir = output.parent / Path(f"{output.stem}_ao_optimize_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_optimize_event = create_ao_optimize_event(config, ao_output_dir)
    # ----------------------------------------------------------------#
    # Create the o2o3 AF event data
    if o2o3_mode != "none":
        af_camera_crop_y = config["O2O3-autofocus"]["crop_y"]
        o2o3_event = create_o2o3_autofocus_event(
            exposure_ms=config["O2O3-autofocus"]["exposure_ms"],
            camera_center=[camera_center_x, camera_center_y],
            camera_crop=[camera_crop_x, af_camera_crop_y],
        )

    # ----------------------------------------------------------------#
    # Create the fluidics event data
    if fluidics_mode != "none":
        fluidics_rounds = int(fluidics_mode)

    # ----------------------------------------------------------------#
    # Compile mda positions from active tabs and config
    # ----------------------------------------------------------------#

    # ----------------------------------------------------------------#
    # Define the time indexing
    if fluidics_mode != "none":
        n_time_steps = fluidics_rounds
        time_interval = 0
    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan["loops"]
        time_interval = mda_time_plan["interval"]
    else:
        n_time_steps = 1
        time_interval = 1

    if time_interval == 0:
        need_to_setup_daq = False
        if "timepoint" in ao_mode:
            ao_mode = "grid at start" if "grid" in ao_mode else "once at start"
            warning(
                "AO MODE UPDATED",
                "AO mode is set to timepoint, but 0 interval was selected.",
                "Running AO at start.",
            )
        if "timepoint" in o2o3_mode:
            o2o3_mode = "once at start"
            warning(
                "AF MODE UPDATED",
                "AF mode is set to timepoint, but 0 interval was selected.",
                "Running autofocus at start.",
            )
    else:
        need_to_setup_daq = True
    # ----------------------------------------------------------------#
    # Generate xyz stage positions
    if mda_grid_plan is not None:
        stage_positions = stage_positions_from_grid(
            opm_mode=opm_mode,
            mda_grid_plan=mda_grid_plan,
            mda_z_plan=mda_z_plan,
            camera_crop_x=camera_crop_x,
            camera_crop_y=camera_crop_y,
            scan_range_um=scan_range_um,
            tile_axis_overlap=tile_axis_overlap,
            scan_axis_overlap=tile_axis_overlap,
            z_axis_overlap=z_axis_overlap,
            coverslip_slope_x=coverslip_slope_x,
            coverslip_slope_y=coverslip_slope_y,
        )
    elif mda_positions_plan is not None:
        stage_positions = []
        for stage_pos in mda_positions_plan:
            stage_positions.append({
                "x": float(stage_pos["x"]),
                "y": float(stage_pos["y"]),
                "z": float(stage_pos["z"]),
            })
    n_stage_positions = len(stage_positions)

    # update AO grid event with stage positions
    if "grid" in ao_mode:
        ao_grid_event.action.data["AO"]["stage_positions"] = stage_positions
    if ao_mode != "none":
        AOmirror_setup.n_positions = n_stage_positions
    ao_scheduler = AOEventScheduler(ao_mode, ao_optimize_event, ao_grid_event)

    # ----------------------------------------------------------------#
    # Create MDA event structure
    # ----------------------------------------------------------------#
    opm_events: list[MDAEvent] = []

    # ----------------------------------------------------------------#
    # Setup Nt / Np / Nc / Nz mirror scan acquisition
    debug(
        "MIRROR SCAN ACQUISITION SETTINGS",
        f"timepoints / interval: {n_time_steps} / {time_interval}",
        f"stage positions: {n_stage_positions}",
        f"active channels: {n_active_channels}",
        f"AO frequency: {ao_mode}",
        f"o2o3 focus frequency: {o2o3_mode}",
        f"num scan steps: {n_scan_steps}",
        f"scan range (um): {scan_range_um}",
        f"scan step (um): {scan_step_um}",
        f"DAQ scan mode: {scan_mode}",
    )

    for time_idx in trange(n_time_steps, desc="Timepoints:", leave=True):
        timepoint_start_index = len(opm_events)
        # --------------------------------------------------------------------#
        # Run fluidics starting at the second timepoint if present
        if fluidics_mode != "none" and time_idx != 0:
            current_fluidics_event = create_fluidics_event(fluidics_rounds, time_idx)
            opm_events.append(current_fluidics_event)

        # --------------------------------------------------------------------#
        # Create events to run before acquisition
        if time_idx == 0:
            # move stage to starting position
            initial_stage_event = create_stage_event(stage_positions[0])
            opm_events.append(initial_stage_event)

            if n_stage_positions == 1:
                need_to_setup_stage = False
            else:
                need_to_setup_stage = True

            # Create 'start' optimization events

            if "start" in o2o3_mode:
                opm_events.append(o2o3_event)
            if "start" in ao_mode:
                if "grid" in ao_mode:
                    curr_ao_grid_event = clone_event(ao_grid_event)
                    opm_events.append(curr_ao_grid_event)
                else:
                    curr_ao_opt_event = clone_event(ao_optimize_event)
                    opm_events.append(curr_ao_opt_event)

        # --------------------------------------------------------------------#
        # Create events to run each timepoint
        if o2o3_mode == "at timepoints":
            opm_events.append(o2o3_event)

        ao_scheduler.append_timepoint_event(
            opm_events,
            time_idx,
            ao_output_dir if ao_mode != "none" else output.parent,
        )

        if "none" in ao_mode and (time_interval > 0):
            ao_mirror_update_event = create_ao_mirror_update_event(
                mirror_coeffs=AOmirror_setup.current_coeffs.copy()
            )
            opm_events.append(ao_mirror_update_event)

        # --------------------------------------------------------------------#
        # iterate over stage positions
        for pos_idx in range(n_stage_positions):
            # Move stage to position
            if need_to_setup_stage and pos_idx != 0:
                stage_event = create_stage_event(stage_positions[pos_idx])
                opm_events.append(stage_event)

            # ----------------------------------------------------------------#
            # Create mirror state update events for 'start' and 'time-point' ao modes
            if ("start" in ao_mode) or (
                (ao_mode in ("at timepoints", "grid at timepoints"))
                and (time_interval > 0)
            ):
                ao_scheduler.append_position_update(opm_events, pos_idx)

            # ----------------------------------------------------------------#
            # Create 'xyz' optimization events
            if "xyz" in o2o3_mode:
                opm_events.append(o2o3_event)

            if "xyz" in ao_mode:
                # Run the ao optimization at the first time-point for each position
                if time_idx == 0:
                    current_ao_dir = ao_output_dir / Path(f"pos_{pos_idx}_ao_results")
                    current_ao_dir.mkdir(exist_ok=True)
                    current_ao_event = clone_event(ao_optimize_event)
                    current_ao_event.action.data["AO"]["output_path"] = current_ao_dir
                    current_ao_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                    current_ao_event.action.data["AO"]["apply_existing"] = False
                    opm_events.append(current_ao_event)

                # Update the mirror state for at each position for all time-points
                current_ao_event = clone_event(ao_optimize_event)
                current_ao_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                current_ao_event.action.data["AO"]["apply_existing"] = True
                opm_events.append(current_ao_event)

            # ----------------------------------------------------------------#
            # Handle acquiring images
            # ----------------------------------------------------------------#

            if interleaved_acq:
                if time_idx == 0:
                    # Update daq state to sequence all channels
                    opm_events.append(clone_event(daq_event))
                if need_to_setup_daq:
                    opm_events.append(clone_event(daq_event))

            local_axis_order = ("z", "c") if interleaved_acq else ("c", "z")
            active_daq_channel: int | None = None
            active_daq_settings = (
                channel_states,
                channel_powers,
                channel_exposures_ms,
            )
            for planned_event in iter_planned_image_events(
                {"c": n_active_channels, "z": n_scan_steps}, local_axis_order
            ):
                current_chan_idx = planned_event.index["c"]
                scan_idx = planned_event.index["z"]
                chan_idx = active_channel_indices[current_chan_idx]
                image_channel_states = channel_states
                image_channel_powers = channel_powers
                image_channel_exposures = channel_exposures_ms

                if not interleaved_acq and current_chan_idx != active_daq_channel:
                    active_daq_settings = single_channel_settings(
                        channel_states,
                        channel_powers,
                        channel_exposures_ms,
                        chan_idx,
                    )
                    (
                        image_channel_states,
                        image_channel_powers,
                        image_channel_exposures,
                    ) = active_daq_settings
                    current_daq_event = clone_event(daq_event)
                    current_daq_event.action.data["DAQ"]["channel_states"] = (
                        image_channel_states
                    )
                    current_daq_event.action.data["DAQ"]["channel_powers"] = (
                        image_channel_powers
                    )
                    current_daq_event.action.data["Camera"]["exposure_channels"] = (
                        image_channel_exposures
                    )
                    opm_events.append(current_daq_event)
                    active_daq_channel = current_chan_idx
                elif not interleaved_acq:
                    (
                        image_channel_states,
                        image_channel_powers,
                        image_channel_exposures,
                    ) = active_daq_settings

                image_event = create_opm_image_event(
                    index={
                        "t": time_idx,
                        "p": pos_idx,
                        "c": current_chan_idx,
                        "z": scan_idx,
                    },
                    config=config,
                    stage_position=stage_positions[pos_idx],
                    daq_mode=scan_mode,
                    image_mirror_range_um=scan_range_um,
                    mirror_step=scan_step_um,
                    channel_states=image_channel_states,
                    channel_exposures_ms=image_channel_exposures,
                    laser_powers=image_channel_powers,
                    interleaved=interleaved_acq,
                    blanking=laser_blanking,
                    current_channel=channel_names[chan_idx],
                    exposure_ms=channel_exposures_ms[chan_idx],
                    camera_center_x=camera_center_x,
                    camera_center_y=camera_center_y,
                    camera_crop_x=camera_crop_x,
                    camera_crop_y=camera_crop_y,
                    offset=offset,
                    e_to_ADU=e_to_ADU,
                )
                opm_events.append(image_event)

        apply_timepoint_timing(
            opm_events,
            timepoint_start_index,
            time_idx,
            float(time_interval),
        )

    # Check if path ends if .zarr. If so, use our OutputHandler
    indice_sizes = {
        "t": int(np.maximum(1, n_time_steps)),
        "p": int(np.maximum(1, n_stage_positions)),
        "c": int(np.maximum(1, n_active_channels)),
        "z": int(np.maximum(1, n_scan_steps)),
    }
    camera_acquisition_order = (
        ("t", "p", "z", "c") if interleaved_acq else ("t", "p", "c", "z")
    )
    return opm_events, create_zarr_handler(
        output,
        indice_sizes,
        acquisition_order=camera_acquisition_order,
        events=opm_events,
        config=config,
    )


def setup_stagescan(
    mmc: CMMCorePlus,
    config: dict,
    sequence: MDASequence,
    output: Path,
) -> tuple[list[MDAEvent], OpmDataHandler]:
    """Set up an OPM stage-scan acquisition.

    TODO: add logic to allow for non-interleaved acquisitions

    t / p / c / z

    Parameters
    ----------
    mmc : CMMCorePlus
        MMC core in use
    config : dict
        OPM config loaded from disk
    sequence : MDASequence
        MDA sequence to run
    output : Path
        Output path from MDA widget

    Returns
    -------
    tuple[list[MDAEvent], OpmDataHandler]
        Camera acquisition events and TensorStore-backed output handler.

    Raises
    ------
    Exception
        If neither grid nor position plans are configured.
    """
    AOmirror_setup = AOMirror.instance()

    # --------------------------------------------------------------------#
    # Compile acquisition settings from configuration
    # --------------------------------------------------------------------#
    # Get the acquisition modes
    acq_config = config["acq_config"]
    daq_config = acq_config["DAQ"]
    positions_config = acq_config["Positions"]
    stage_config = acq_config["stage_scan"]
    ao_mode = normalize_ao_mode(acq_config["AO"]["ao_mode"])
    o2o3_mode = normalize_autofocus_mode(acq_config["o2o3_mode"])
    fluidics_mode = acq_config["fluidics"]

    # Get the camera crop values
    camera_crop_y = int(acq_config["camera_roi"]["crop_y"])
    camera_crop_x = int(acq_config["camera_roi"]["crop_x"])
    camera_center_y = int(acq_config["camera_roi"]["center_y"])
    camera_center_x = int(acq_config["camera_roi"]["center_x"])

    # Get pixel size and deskew Y-scale factor
    pixel_size_um = np.round(float(mmc.getPixelSizeUm()), 3)  # unit: um
    opm_angle_scale = np.sin((np.pi / 180.0) * float(config["OPM"]["angle_deg"]))

    # Get the stage scan range, coverslip slope, and maximum CS dz change
    coverslip_slope = float(positions_config["coverslip_slope_x"])
    scan_axis_max_range = float(stage_config["max_stage_scan_range_um"])
    coverslip_max_dz = float(positions_config["coverslip_max_dz"])

    # Get the tile overlap settings
    tile_axis_overlap = float(positions_config["tile_axis_overlap"])
    scan_axis_step_um = float(daq_config["scan_axis_step_um"])
    scan_tile_overlap_um = camera_crop_y * opm_angle_scale * pixel_size_um + float(
        positions_config["scan_axis_overlap_um"]
    )
    scan_tile_overlap_mm = scan_tile_overlap_um / 1000.0

    # Get the excess start / end
    excess_start_images = int(stage_config["excess_start_frames"])
    excess_end_images = int(stage_config["excess_end_frames"])

    # ----------------------------------------------------------------#
    # Get channel and camera metadata settings
    laser_blanking = daq_config["laser_blanking"]
    channel_plan = get_channel_plan(config, "DAQ", round_active_exposures=True)
    channel_states = channel_plan["states"]
    channel_powers = channel_plan["powers"]
    channel_exposures_ms = channel_plan["exposures_ms"]
    active_channel_indices = channel_plan["active_indices"]
    n_active_channels = channel_plan["active_count"]
    active_channel_names = channel_plan["active_names"]
    active_channel_exps = channel_plan["active_exposures_ms"]
    interleaved_acq = channel_plan["interleaved"]
    if not interleaved_acq:
        warning(
            "STAGE SCAN CHANNELS",
            "Stage scan currently assumes interleaved acquisition.",
            "Active channels do not all have the same exposure.",
        )

    # Get the exposure, assumes equal exposures
    exposure_ms = np.round(active_channel_exps[0], 2)
    exposure_s = np.round(exposure_ms / 1000.0, 2)

    offset, e_to_ADU = get_camera_conversion(mmc, config)

    # --------------------------------------------------------------------#
    # Compile mda acquisition settings from active tabs
    # --------------------------------------------------------------------#

    # Split apart sequence dictionary
    sequence_plans = get_sequence_plans(sequence)
    mda_grid_plan = sequence_plans["grid"]
    mda_positions_plan = sequence_plans["positions"]
    mda_time_plan = sequence_plans["time"]
    mda_z_plan = sequence_plans["z"]

    if (mda_grid_plan is None) and (mda_positions_plan is None):
        raise Exception("Must select MDA grid or positions plan for stage scanning")

    # ----------------------------------------------------------------#
    # Create custom action data
    # ----------------------------------------------------------------#

    # ----------------------------------------------------------------#
    # Create DAQ event
    daq_event = create_daq_event(
        mode="stage",
        channel_states=channel_states,
        channel_powers=channel_powers,
        channel_exposures_ms=channel_exposures_ms,
        camera_center=[camera_center_x, camera_center_y],
        camera_crop=[camera_crop_x, camera_crop_y],
        interleaved=True,
        laser_blanking=laser_blanking,
    )

    # ----------------------------------------------------------------#
    # Create the AO event data
    ao_optimize_event = None
    ao_grid_event = None
    if ao_mode != "none":
        if "grid" in ao_mode:
            ao_output_dir = output.parent / Path(f"{output.stem}_ao_grid_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_grid_event = create_ao_grid_event(config, ao_output_dir)
        else:
            ao_output_dir = output.parent / Path(f"{output.stem}_ao_optmize_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_optimize_event = create_ao_optimize_event(config, ao_output_dir)

    # ----------------------------------------------------------------#
    # Create the o2o3 AF event data
    if o2o3_mode != "none":
        af_camera_crop_y = config["O2O3-autofocus"]["crop_y"]
        o2o3_event = create_o2o3_autofocus_event(
            exposure_ms=config["O2O3-autofocus"]["exposure_ms"],
            camera_center=[camera_center_x, camera_center_y],
            camera_crop=[camera_crop_x, af_camera_crop_y],
        )

    # ----------------------------------------------------------------#
    # Create the fluidics event data
    if fluidics_mode != "none":
        fluidics_rounds = int(fluidics_mode)

    # ----------------------------------------------------------------#
    # Compile mda positions from active tabs, and config
    # ----------------------------------------------------------------#

    # ----------------------------------------------------------------#
    # Define the time indexing
    if fluidics_mode != "none":
        n_time_steps = fluidics_rounds
        time_interval = 0
    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan["loops"]
        time_interval = mda_time_plan["interval"]
    else:
        n_time_steps = 1
        time_interval = 0

    # ----------------------------------------------------------------#
    # Generate xyz stage positions
    stage_positions = []

    if mda_grid_plan is None:
        warning("GRID PLAN", "No grid plan selected.")
        return None, None

    # grab grid plan extents
    min_y_pos = mda_grid_plan["bottom"]
    max_y_pos = mda_grid_plan["top"]
    min_x_pos = mda_grid_plan["left"]
    max_x_pos = mda_grid_plan["right"]

    if mda_z_plan is not None:
        max_z_pos = float(mda_z_plan["top"])
        min_z_pos = float(mda_z_plan["bottom"])
    else:
        min_z_pos = mmc.getZPosition()
        max_z_pos = mmc.getZPosition()

    # Set grid axes ranges
    range_x_um = np.round(np.abs(max_x_pos - min_x_pos), 2)
    range_y_um = np.round(np.abs(max_y_pos - min_y_pos), 2)
    range_z_um = np.round(np.abs(max_z_pos - min_z_pos), 2)

    # Define coverslip bounds, to offset Z positions
    cs_min_pos = min_z_pos
    cs_max_pos = cs_min_pos + range_x_um * coverslip_slope
    cs_range_um = np.round(np.abs(cs_max_pos - cs_min_pos), 2)

    # --------------------------------------------------------------------#
    # Calculate tile steps / range
    z_axis_step_max = (
        camera_crop_y * pixel_size_um * opm_angle_scale * (1 - tile_axis_overlap)
    )
    tile_axis_step_max = camera_crop_x * pixel_size_um * (1 - tile_axis_overlap)

    # Check if the coverslip slope determines the max scan range
    if coverslip_slope != 0:
        coverslip_max_scan_range = np.abs(coverslip_max_dz / coverslip_slope)
        if coverslip_max_scan_range < scan_axis_max_range:
            scan_axis_max_range = coverslip_max_scan_range

    # Correct directions for stage moves
    if min_z_pos > max_z_pos:
        z_axis_step_max *= -1
    if min_x_pos > max_x_pos:
        min_x_pos, max_x_pos = max_x_pos, min_x_pos

    debug(
        "XYZ STAGE SCAN POSITIONS",
        f"scan start: {min_x_pos}",
        f"scan end: {max_x_pos}",
        f"tile start: {min_y_pos}",
        f"tile end: {max_y_pos}",
        f"z position min: {min_z_pos}",
        f"z position max: {max_z_pos}",
        f"coverslip slope: {coverslip_slope}",
        f"coverslip low: {cs_min_pos}",
        f"coverslip high: {cs_max_pos}",
        f"max scan range (CS used? {coverslip_slope != 0}): {scan_axis_max_range}",
    )

    # --------------------------------------------------------------------#
    # Calculate scan axis tile locations, units: mm and s

    # Break scan range up using max scan range
    if scan_axis_max_range >= range_x_um:
        n_scan_positions = 1
        scan_tile_length_um = range_x_um
    else:
        # Round up so that the scan length is never longer than the max scan range
        n_scan_positions = int(np.ceil(range_x_um / (scan_axis_max_range)))
        scan_tile_length_um = np.round(
            (range_x_um / n_scan_positions)
            + (n_scan_positions - 1) * (scan_tile_overlap_um / n_scan_positions),
            2,
        )
    scan_axis_step_mm = scan_axis_step_um / 1000.0
    scan_axis_start_mm = min_x_pos / 1000.0
    scan_axis_end_mm = max_x_pos / 1000.0
    scan_tile_length_mm = np.round(scan_tile_length_um / 1000.0, 2)

    # Initialize scan position start/end arrays with the scan start / end values
    scan_axis_start_pos_mm = np.full(n_scan_positions, scan_axis_start_mm)
    scan_axis_end_pos_mm = np.full(n_scan_positions, scan_axis_end_mm)
    for ii in range(n_scan_positions):
        scan_axis_start_pos_mm[ii] = scan_axis_start_mm + ii * (
            scan_tile_length_mm - scan_tile_overlap_mm
        )
        scan_axis_end_pos_mm[ii] = scan_axis_start_pos_mm[ii] + scan_tile_length_mm

    scan_axis_start_pos_mm = np.round(scan_axis_start_pos_mm, 2)
    scan_axis_end_pos_mm = np.round(scan_axis_end_pos_mm, 2)
    scan_tile_length_w_overlap_mm = np.round(
        np.abs(scan_axis_end_pos_mm[0] - scan_axis_start_pos_mm[0]), 2
    )
    scan_axis_positions = np.rint(
        scan_tile_length_w_overlap_mm / scan_axis_step_mm
    ).astype(int)
    scan_axis_speed = np.round(scan_axis_step_mm / exposure_s / n_active_channels, 5)
    scan_tile_sizes = [
        np.round(np.abs(scan_axis_end_pos_mm[ii] - scan_axis_start_pos_mm[ii]), 2)
        for ii in range(len(scan_axis_end_pos_mm))
    ]
    n_scan_axis_indices = (
        scan_axis_positions + int(excess_start_images) + int(excess_end_images)
    )
    # Check for scan speed actual settings
    xy_stage = mmc.getXYStageDevice()
    if mmc.hasProperty(xy_stage, "MotorSpeedX-S(mm/s)"):
        mmc.setProperty(xy_stage, "MotorSpeedX-S(mm/s)", scan_axis_speed)
        mmc.waitForDevice(xy_stage)
        actual_speed_x = float(mmc.getProperty(xy_stage, "MotorSpeedX-S(mm/s)"))
    else:
        actual_speed_x = float(scan_axis_speed)
    actual_exposure = np.round(
        scan_axis_step_mm / actual_speed_x / n_active_channels, 5
    )
    channel_exposures_ms = [actual_exposure * 1000] * len(channel_exposures_ms)

    # update acq settings with the actual exposure and stage scan speed
    daq_event.action.data["Camera"]["exposure_channels"] = channel_exposures_ms
    exposure_ms = actual_exposure * 1000
    exposure_s = actual_exposure
    scan_axis_speed = actual_speed_x

    test_scan_length = scan_tile_length_mm == scan_tile_length_w_overlap_mm
    test_tile_sizes = np.allclose(scan_tile_sizes, scan_tile_sizes[0])
    debug(
        "SCAN AXIS CALCULATED PARAMETERS",
        f"number scan tiles: {n_scan_positions}",
        f"tile length um: {scan_tile_length_um}",
        f"tile overlap um: {scan_tile_overlap_um}",
        f"tile length mm: {scan_tile_length_mm}",
        f"tile length with overlap (mm): {scan_tile_length_w_overlap_mm}",
        f"scan tile with overlap equals scan tile length: {test_scan_length}",
        f"step size (mm): {scan_axis_step_mm}",
        f"exposure: {exposure_s}",
        f"number of active channels: {n_active_channels}",
        f"scan axis speed (mm/s): {scan_axis_speed}",
        "stage scan positions, units: mm",
        f"scan axis start positions: {scan_axis_start_pos_mm}",
        f"scan axis end positions: {scan_axis_end_pos_mm}",
        f"number of scan positions: {scan_axis_positions}",
        f"all scan tiles the same size: {test_tile_sizes}",
    )

    # --------------------------------------------------------------------#
    # Generate tile axis positions
    n_tile_positions = int(np.ceil(range_y_um / tile_axis_step_max)) + 1
    tile_axis_positions = np.round(
        np.linspace(min_y_pos, max_y_pos, n_tile_positions), 2
    )
    if n_tile_positions == 1:
        tile_axis_step = 0
    else:
        tile_axis_step = tile_axis_positions[1] - tile_axis_positions[0]

    debug(
        "TILE AXIS POSITIONS",
        "units: um",
        f"tile axis positions: {tile_axis_positions}",
        f"num tile axis positions: {n_tile_positions}",
        f"tile axis step: {tile_axis_step}",
    )

    # --------------------------------------------------------------------#
    # Generate z axis positions, ignoring coverslip slope
    n_z_positions = int(np.ceil(np.abs(range_z_um / z_axis_step_max))) + 1
    z_positions = np.round(np.linspace(min_z_pos, max_z_pos, n_z_positions), 2)
    if n_z_positions == 1:
        z_axis_step_um = 0.0
    else:
        z_axis_step_um = np.round(z_positions[1] - z_positions[0], 2)

    # Calculate the stage z change along the scan axis
    dz_per_scan_tile = (cs_range_um / n_scan_positions) * np.sign(coverslip_slope)

    debug(
        "Z AXIS POSITIONS",
        "units: um",
        f"z axis positions: {z_positions}",
        f"z axis range: {range_z_um} um",
        f"z axis step: {z_axis_step_um} um",
        f"num z axis positions: {n_z_positions}",
        f"z offset per x-scan-tile: {dz_per_scan_tile} um",
        f"z axis step max: {z_axis_step_max}",
    )

    # --------------------------------------------------------------------#
    # Generate stage positions
    n_stage_positions = n_scan_positions * n_tile_positions * n_z_positions
    stage_positions = compose_stage_positions(
        scan_axis_start_pos_mm * 1000,
        tile_axis_positions,
        z_positions,
        z_offset_per_scan_um=dz_per_scan_tile,
    )

    # update AO grid event with stage positions
    if "grid" in ao_mode:
        ao_grid_event.action.data["AO"]["stage_positions"] = stage_positions
    if ao_mode != "none":
        AOmirror_setup.n_positions = n_stage_positions
    ao_scheduler = AOEventScheduler(ao_mode, ao_optimize_event, ao_grid_event)

    # ----------------------------------------------------------------#
    # Create MDA event structure
    # ----------------------------------------------------------------#
    opm_events: list[MDAEvent] = []

    # ----------------------------------------------------------------#
    # Setup Nt / Np / Nc / Nz mirror scan acquisition
    debug(
        "STAGE SCAN ACQUISITION SHAPE",
        f"timepoints / interval: {n_time_steps} / {time_interval}",
        f"stage positions: {n_stage_positions}",
        f"scan positions: {n_scan_axis_indices}",
        f"active channels: {n_active_channels}",
        f"excess frames (S/E): {excess_start_images}/{excess_end_images}",
        f"AO frequency: {ao_mode}",
        f"o2o3 focus frequency: {o2o3_mode}",
    )

    for time_idx in trange(n_time_steps, desc="Timepoints:", leave=True):
        timepoint_start_index = len(opm_events)
        # --------------------------------------------------------------------#
        # Run fluidics starting at the second timepoint if present
        if fluidics_mode != "none" and time_idx != 0:
            current_fluidics_event = create_fluidics_event(fluidics_rounds, time_idx)
            opm_events.append(current_fluidics_event)

        # --------------------------------------------------------------------#
        # Create events to run before acquisition
        if time_idx == 0:
            # move stage to starting position
            initial_stage_event = create_stage_event(stage_positions[0])
            opm_events.append(initial_stage_event)

            # Create 'start' optimization events
            if "start" in o2o3_mode:
                opm_events.append(o2o3_event)
            if "start" in ao_mode:
                if "grid" in ao_mode:
                    curr_ao_grid_event = clone_event(ao_grid_event)
                    opm_events.append(curr_ao_grid_event)
                else:
                    curr_ao_opt_event = clone_event(ao_optimize_event)
                    opm_events.append(curr_ao_opt_event)

        # --------------------------------------------------------------------#
        # Create events to run each timepoint
        if o2o3_mode == "at timepoints":
            opm_events.append(o2o3_event)

        ao_scheduler.append_timepoint_event(
            opm_events,
            time_idx,
            ao_output_dir if ao_mode != "none" else output.parent,
        )

        if "none" in ao_mode and (time_interval > 0):
            ao_mirror_update_event = create_ao_mirror_update_event(
                mirror_coeffs=AOmirror_setup.current_coeffs.copy()
            )
            opm_events.append(ao_mirror_update_event)

        # --------------------------------------------------------------------#
        # iterate over stage positions
        pos_idx = 0
        for z_idx in trange(n_z_positions, desc="Z-axis-tiles:", leave=False):
            for scan_idx in trange(
                n_scan_positions, desc="Scan-axis-tiles:", leave=False
            ):
                for tile_idx in trange(
                    n_tile_positions, desc="Tile-axis-tiles:", leave=False
                ):
                    # ----------------------------------------------------------------#
                    # Move stage to position
                    current_stage_event = create_stage_event(stage_positions[pos_idx])
                    opm_events.append(current_stage_event)

                    # ----------------------------------------------------------------#
                    # Create mirror state update events for 'start' and 'time-point' AO
                    # NOTE: Update mirror every time-point and stage-position
                    # NOTE: for single position optimization, only refer to pos_idx==0,
                    #       Currently not filling the entire position array!
                    if ("start" in ao_mode) or (
                        ao_mode in ("at timepoints", "grid at timepoints")
                    ):
                        ao_scheduler.append_position_update(opm_events, pos_idx)

                    # ----------------------------------------------------------------#
                    # Create 'xyz' optimization events
                    if "xyz" in o2o3_mode:
                        opm_events.append(o2o3_event)

                    if "xyz" in ao_mode:
                        # AO optimization at the first timepoint at all positions
                        if time_idx == 0:
                            current_ao_dir = ao_output_dir / Path(
                                f"pos_{pos_idx}_ao_results"
                            )
                            current_ao_dir.mkdir(exist_ok=True)
                            current_ao_event = clone_event(ao_optimize_event)
                            current_ao_event.action.data["AO"]["output_path"] = (
                                current_ao_dir
                            )
                            current_ao_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                            current_ao_event.action.data["AO"]["apply_existing"] = False
                            opm_events.append(current_ao_event)

                        # Update the mirror state for at each position and timepoint
                        current_ao_event = clone_event(ao_optimize_event)
                        current_ao_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                        current_ao_event.action.data["AO"]["apply_existing"] = True
                        opm_events.append(current_ao_event)

                    # ----------------------------------------------------------------#
                    # Handle acquiring images
                    # ----------------------------------------------------------------#
                    # TODO: edit logic to include non-interleaved acq
                    opm_events.append(daq_event)

                    # Set ASI controller for stage scanning and Camera for external Trig
                    current_asi_setup_event = create_asi_scan_setup_event(
                        start_mm=float(scan_axis_start_pos_mm[scan_idx]),
                        end_mm=float(scan_axis_end_pos_mm[scan_idx]),
                        speed_mm_s=float(scan_axis_speed),
                    )
                    opm_events.append(current_asi_setup_event)

                    # The camera produces frames in the c/z order selected in the
                    # MDA widget.  The ASI action configures only stage/PLC hardware.
                    widget_axis_order = get_indexed_acquisition_order(
                        sequence, {"c": n_active_channels, "z": n_scan_axis_indices}
                    )

                    # Create image events
                    for planned_event in iter_planned_image_events(
                        {"c": n_active_channels, "z": n_scan_axis_indices},
                        widget_axis_order,
                    ):
                        scan_axis_idx = planned_event.index["z"]
                        chan_idx = planned_event.index["c"]
                        source_chan_idx = active_channel_indices[chan_idx]
                        end_excess_idx = scan_axis_positions + excess_start_images
                        if scan_axis_idx < excess_start_images:
                            is_excess_image = True
                        elif scan_axis_idx > end_excess_idx:
                            is_excess_image = True
                        else:
                            is_excess_image = False
                        image_event = create_stage_scan_image_event(
                            index={
                                "t": time_idx,
                                "p": pos_idx,
                                "c": chan_idx,
                                "z": scan_axis_idx,
                            },
                            config=config,
                            stage_position=stage_positions[pos_idx],
                            scan_axis_idx=scan_axis_idx,
                            scan_axis_step_um=scan_axis_step_um,
                            channel_states=channel_states,
                            channel_exposures_ms=channel_exposures_ms,
                            laser_powers=channel_powers,
                            laser_blanking=laser_blanking,
                            current_channel=active_channel_names[chan_idx],
                            exposure_ms=channel_exposures_ms[source_chan_idx],
                            camera_center_x=camera_center_x,
                            camera_center_y=camera_center_y,
                            camera_crop_x=camera_crop_x,
                            camera_crop_y=camera_crop_y,
                            offset=offset,
                            e_to_ADU=e_to_ADU,
                            excess_start_images=excess_start_images,
                            excess_end_images=excess_end_images,
                            is_excess_image=is_excess_image,
                        )
                        opm_events.append(image_event)
                    pos_idx = pos_idx + 1

        apply_timepoint_timing(
            opm_events,
            timepoint_start_index,
            time_idx,
            float(time_interval),
        )

    # Check if path ends if .zarr. If so, use our OutputHandler
    indice_sizes = {
        "t": int(np.maximum(1, n_time_steps)),
        "p": int(np.maximum(1, n_stage_positions)),
        "c": int(np.maximum(1, n_active_channels)),
        "z": int(np.maximum(1, n_scan_axis_indices)),
    }
    return opm_events, create_zarr_handler(
        output,
        indice_sizes,
        acquisition_order=get_indexed_acquisition_order(sequence, indice_sizes),
        events=opm_events,
        config=config,
    )


class OPMEventBuilder:
    """Build events with the mode-specific setup functions.

    Parameters
    ----------
    mmc : CMMCorePlus
        Active Micro-Manager core.
    config : dict
        OPM configuration.
    sequence : MDASequence
        Sequence from the MDA widget.
    """

    def __init__(self, mmc: CMMCorePlus, config: dict, sequence: MDASequence) -> None:
        """Initialize an OPM event builder.

        Parameters
        ----------
        mmc : CMMCorePlus
            Active Micro-Manager core.
        config : dict
            OPM configuration.
        sequence : MDASequence
            Sequence from the MDA widget.
        """
        self.mmc = mmc
        self.config = config
        self.sequence = sequence

    def build(self, output: Path, mode: str | None = None):
        """Build events for a requested or configured mode.

        Parameters
        ----------
        output : Path
            Acquisition output path.
        mode : str or None
            Acquisition mode, or ``None`` to use configuration.

        Returns
        -------
        tuple[list[MDAEvent], OpmDataHandler]
            Acquisition events and output handler.

        Raises
        ------
        ValueError
            If the acquisition mode is unknown.
        """
        mode = mode or self.config["acq_config"]["opm_mode"]
        if "timelapse" in mode:
            return setup_timelapse(self.mmc, self.config, self.sequence, output)
        if "projection" in mode:
            return setup_projection(self.mmc, self.config, self.sequence, output)
        if "mirror" in mode:
            return setup_mirrorscan(self.mmc, self.config, self.sequence, output)
        if "stage" in mode:
            return setup_stagescan(self.mmc, self.config, self.sequence, output)
        raise ValueError(f"Unknown OPM event builder mode: {mode}")
