"""Provide factories and constants for canonical OPM custom MDA events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

from useq import CustomAction, MDAEvent

ACTION_FLUIDICS = "Fluidics"
ACTION_AO_OPTIMIZE = "AO-optimize"
ACTION_AO_GRID = "AO-grid"
ACTION_AO_MIRROR_UPDATE = "AO-mirrorUpdate"
ACTION_O2O3_AUTOFOCUS = "O2O3-autofocus"
ACTION_MIRROR_MOVE = "Mirror-Move"
ACTION_DAQ = "DAQ"
ACTION_STAGE_MOVE = "Stage-Move"
ACTION_ASI_SETUP_SCAN = "ASI-setupscan"

# MDASequence metadata used only by Stage Explorer preview scans.  The OPM
# engine consumes this before ordinary useq image events are delegated to the
# upstream engine, then restores the captured speeds during teardown.
STAGE_MOVE_SPEED_METADATA_KEY = "opm_stage_move_speeds_mm_s"

VALID_DAQ_MODES = {"2d", "projection", "mirror", "stage"}


def _positive_integer(value: object, name: str) -> int:
    """Normalize an integral numeric configuration value.

    Parameters
    ----------
    value : object
        Configuration value to normalize.
    name : str
        Setting name included in validation errors.

    Returns
    -------
    int
        Positive integer value.

    Raises
    ------
    ValueError
        If the value is not finite, integral, and positive.
    """
    numeric_value = float(value)
    if (
        not isfinite(numeric_value)
        or not numeric_value.is_integer()
        or numeric_value < 1
    ):
        raise ValueError(f"{name} must be a positive integer; received {value!r}")
    return int(numeric_value)


def _as_list(value: Any) -> list | None:
    """Convert an array-like value to a JSON-friendly list.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    list or None
        Converted list, or ``None`` when the input is ``None``.
    """
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _custom_event(name: str, data: Mapping[str, Any]) -> MDAEvent:
    """Create a custom MDA event with a consistent action payload.

    Parameters
    ----------
    name : str
        Custom action name.
    data : Mapping[str, Any]
        Action payload.

    Returns
    -------
    MDAEvent
        Custom-action event.
    """
    return MDAEvent(action=CustomAction(name=name, data=dict(data)))


def _camera_crop_from_center(
    camera_center: Sequence[float],
    camera_crop: Sequence[float],
) -> list[int]:
    """Convert center and crop values to Micro-Manager ROI coordinates.

    Parameters
    ----------
    camera_center : Sequence[float]
        ROI center in pixels as X and Y.
    camera_crop : Sequence[float]
        ROI width and height in pixels.

    Returns
    -------
    list[int]
        ROI as X, Y, width, and height.
    """
    return [
        int(camera_center[0] - camera_crop[0] // 2),
        int(camera_center[1] - camera_crop[1] // 2),
        int(camera_crop[0]),
        int(camera_crop[1]),
    ]


def _ao_camera_crop(config: dict) -> tuple[int, int, int, int]:
    """Read AO camera center and crop values from configuration.

    Parameters
    ----------
    config : dict
        OPM configuration.

    Returns
    -------
    tuple[int, int, int, int]
        Center X, center Y, crop width, and crop height.

    Raises
    ------
    ValueError
        If the configured AO acquisition mode is unsupported.
    """
    acq_config = config["acq_config"]
    ao_config = acq_config["AO"]
    camera_crop_x = int(acq_config["camera_roi"]["crop_x"])
    camera_center_y = int(acq_config["camera_roi"]["center_y"])
    camera_center_x = int(acq_config["camera_roi"]["center_x"])

    daq_mode = str(ao_config["daq_mode"])
    if "2d" in daq_mode:
        camera_crop_y = int(acq_config["camera_roi"]["crop_y"])
    elif "projection" in daq_mode:
        camera_crop_y = int(
            acq_config["AO"]["image_mirror_range_um"] / config["OPM"]["pixel_size_um"]
        )
    else:
        raise ValueError(f"Unknown AO daq mode: {daq_mode}")

    return camera_center_x, camera_center_y, camera_crop_x, camera_crop_y


def _ao_channel_settings(config: dict) -> tuple[list[bool], list[float]]:
    """Read AO channel states and powers from configuration.

    Parameters
    ----------
    config : dict
        OPM configuration.

    Returns
    -------
    tuple[list[bool], list[float]]
        Channel-enabled flags and laser powers.

    Raises
    ------
    ValueError
        If every configured AO laser power is zero.
    """
    ao_config = config["acq_config"]["AO"]
    channel_ids = config["OPM"]["channel_ids"]
    active_channel_id = ao_config["active_channel_id"]

    channel_states = [channel_id == active_channel_id for channel_id in channel_ids]
    channel_powers = [0.0] * len(channel_ids)
    for index, is_active in enumerate(channel_states):
        if is_active:
            channel_powers[index] = float(ao_config["active_channel_power"])

    if sum(channel_powers) == 0:
        raise ValueError("All AO laser powers are set to 0.")

    return channel_states, channel_powers


def _ao_payload(config: dict, output_dir_path: Path | None = None) -> dict:
    """Build the payload shared by AO optimization and grid events.

    Parameters
    ----------
    config : dict
        OPM configuration.
    output_dir_path : Path or None
        AO result directory.

    Returns
    -------
    dict
        AO custom-action payload.
    """
    ao_config = config["acq_config"]["AO"]
    channel_states, channel_powers = _ao_channel_settings(config)
    return {
        "channel_states": channel_states,
        "channel_powers": channel_powers,
        "mirror_state": str(ao_config["mirror_state"]),
        "metric_acceptance": str(ao_config["metric_acceptance"]),
        "daq_mode": str(ao_config["daq_mode"]),
        "exposure_ms": float(ao_config["exposure_ms"]),
        "modal_delta": float(ao_config["mode_delta"]),
        "metric_precision": int(ao_config["metric_precision"]),
        "num_averaged_frames": int(ao_config["num_averaged_frames"]),
        "modal_alpha": float(ao_config["mode_alpha"]),
        "iterations": int(ao_config["num_iterations"]),
        "num_mode_samples": int(ao_config["num_mode_samples"]),
        "metric": str(ao_config["metric"]),
        "modes_to_optimize": str(ao_config["modes_to_optimize"]),
        "image_mirror_range_um": ao_config["image_mirror_range_um"],
        "lightsheet_mode": str(ao_config["lightsheet_mode"]),
        "readout_ms": float(ao_config["readout_ms"]),
        "apply_existing": False,
        "pos_idx": 0,
        "scan_idx": 0,
        "time_idx": 0,
        "output_path": output_dir_path,
    }


def create_fluidics_event(total_rounds: int, current_round: int) -> MDAEvent:
    """Create an event that runs a fluidics round.

    Parameters
    ----------
    total_rounds : int
        Total fluidics rounds.
    current_round : int
        Current fluidics round index.

    Returns
    -------
    MDAEvent
        Fluidics custom-action event.
    """
    return _custom_event(
        ACTION_FLUIDICS,
        {
            "Fluidics": {
                "total_rounds": int(total_rounds),
                "current_round": int(current_round),
            }
        },
    )


def create_ao_optimize_event(
    config: dict,
    output_dir_path: Path | None = None,
) -> MDAEvent:
    """Create a sensorless AO optimization event.

    Parameters
    ----------
    config : dict
        OPM configuration.
    output_dir_path : Path or None
        AO result directory.

    Returns
    -------
    MDAEvent
        AO optimization event.
    """
    ao_config = config["acq_config"]["AO"]
    camera_center_x, camera_center_y, camera_crop_x, camera_crop_y = _ao_camera_crop(
        config
    )
    return _custom_event(
        ACTION_AO_OPTIMIZE,
        {
            "AO": _ao_payload(config, output_dir_path),
            "Camera": {
                "exposure_ms": int(ao_config["exposure_ms"]),
                "camera_crop": _camera_crop_from_center(
                    [camera_center_x, camera_center_y],
                    [camera_crop_x, camera_crop_y],
                ),
            },
        },
    )


def create_ao_grid_event(
    config: dict,
    output_dir_path: Path | None = None,
) -> MDAEvent:
    """Create an AO grid-optimization event.

    Parameters
    ----------
    config : dict
        OPM configuration.
    output_dir_path : Path or None
        AO result directory.

    Returns
    -------
    MDAEvent
        AO grid event.
    """
    ao_config = config["acq_config"]["AO"]
    camera_center_x, camera_center_y, camera_crop_x, camera_crop_y = _ao_camera_crop(
        config
    )
    ao_payload = _ao_payload(config, output_dir_path)
    return _custom_event(
        ACTION_AO_GRID,
        {
            "AO": {
                "stage_positions": None,
                # Older GUI snapshots serialized QSpinBox values as integral
                # JSON floats.  Normalize the event payload at its boundary.
                "num_scan_positions": _positive_integer(
                    ao_config["num_scan_positions"], "num_scan_positions"
                ),
                "num_tile_positions": _positive_integer(
                    ao_config["num_tile_positions"], "num_tile_positions"
                ),
                "output_path": output_dir_path,
                "apply_ao_map": False,
                "pos_idx": 0,
                "scan_idx": 0,
                "time_idx": 0,
                "ao_dict": ao_payload,
            },
            "Camera": {
                "exposure_ms": int(ao_config["exposure_ms"]),
                "camera_crop": _camera_crop_from_center(
                    [camera_center_x, camera_center_y],
                    [camera_crop_x, camera_crop_y],
                ),
            },
        },
    )


def create_ao_mirror_update_event(
    mirror_coeffs: Sequence[float] | None = None,
    mirror_positions: Sequence[float] | None = None,
) -> MDAEvent | None:
    """Create an event to apply a known AO mirror state.

    ``modal_coeffs`` is preserved for compatibility with the current engine.
    ``positions`` is included when supplied so future engine logic can apply
    actuator positions directly without changing the event schema again.

    Parameters
    ----------
    mirror_coeffs : Sequence[float] or None
        Modal coefficients to apply.
    mirror_positions : Sequence[float] or None
        Direct actuator positions to apply.

    Returns
    -------
    MDAEvent or None
        Mirror-update event, or ``None`` when no state was supplied.
    """
    if mirror_coeffs is None and mirror_positions is None:
        return None
    return _custom_event(
        ACTION_AO_MIRROR_UPDATE,
        {
            "AOmirror": {
                "modal_coeffs": _as_list(mirror_coeffs),
                "positions": _as_list(mirror_positions),
            }
        },
    )


def create_o2o3_autofocus_event(
    exposure_ms: int,
    camera_center: Sequence[int],
    camera_crop: Sequence[int],
) -> MDAEvent:
    """Create an O2/O3 autofocus event.

    Parameters
    ----------
    exposure_ms : int
        Autofocus exposure in milliseconds.
    camera_center : Sequence[int]
        Camera ROI center.
    camera_crop : Sequence[int]
        Camera ROI width and height.

    Returns
    -------
    MDAEvent
        Autofocus event.
    """
    return _custom_event(
        ACTION_O2O3_AUTOFOCUS,
        {
            "Camera": {
                "exposure_ms": int(exposure_ms),
                "camera_crop": _camera_crop_from_center(camera_center, camera_crop),
            }
        },
    )


def create_daq_move_event(image_mirror_v: float) -> MDAEvent:
    """Create an event that changes the image-mirror neutral voltage.

    Parameters
    ----------
    image_mirror_v : float
        Requested image-mirror voltage.

    Returns
    -------
    MDAEvent
        Mirror-move event.
    """
    return _custom_event(
        ACTION_MIRROR_MOVE,
        {"DAQ": {"image_mirror_v": float(image_mirror_v)}},
    )


def create_daq_event(
    mode: str = "2d",
    channel_states: Sequence[bool] | None = None,
    channel_powers: Sequence[float] | None = None,
    channel_exposures_ms: Sequence[float] | None = None,
    camera_center: Sequence[int] | None = None,
    camera_crop: Sequence[int] | None = None,
    interleaved: bool = False,
    laser_blanking: bool = True,
    image_mirror_range_um: float | None = 0,
    image_mirror_step_um: float | None = 0.4,
) -> MDAEvent:
    """Create an event that programs DAQ and camera acquisition state.

    Parameters
    ----------
    mode : str
        DAQ acquisition mode.
    channel_states : Sequence[bool] or None
        Enabled laser channels.
    channel_powers : Sequence[float] or None
        Laser powers for each channel.
    channel_exposures_ms : Sequence[float] or None
        Camera exposures for each channel.
    camera_center : Sequence[int] or None
        Camera ROI center.
    camera_crop : Sequence[int] or None
        Camera ROI width and height.
    interleaved : bool
        Whether channels share an interleaved scan.
    laser_blanking : bool
        Whether camera exposure gates laser output.
    image_mirror_range_um : float or None
        Image-mirror scan range in micrometers.
    image_mirror_step_um : float or None
        Image-mirror step in micrometers.

    Returns
    -------
    MDAEvent
        DAQ configuration event.

    Raises
    ------
    ValueError
        If ``mode`` is not a supported DAQ mode.
    """
    if mode not in VALID_DAQ_MODES:
        raise ValueError(f"Unknown DAQ mode: {mode}")

    channel_states = list(channel_states or [False] * 5)
    channel_powers = list(channel_powers or [0.0] * len(channel_states))
    channel_exposures_ms = list(channel_exposures_ms or [0.0] * len(channel_states))
    camera_center = list(camera_center or [0, 0])
    camera_crop = list(camera_crop or [0, 0])

    return _custom_event(
        ACTION_DAQ,
        {
            "DAQ": {
                "mode": mode,
                "channel_states": channel_states,
                "channel_powers": channel_powers,
                "interleaved": bool(interleaved),
                "blanking": bool(laser_blanking),
                "image_mirror_range_um": image_mirror_range_um,
                "image_mirror_step_um": image_mirror_step_um,
            },
            "Camera": {
                "exposure_channels": channel_exposures_ms,
                "camera_crop": _camera_crop_from_center(camera_center, camera_crop),
            },
        },
    )


def create_stage_event(stage_position: Mapping[str, float]) -> MDAEvent:
    """Create an event that moves the XYZ stage.

    Parameters
    ----------
    stage_position : Mapping[str, float]
        Target X, Y, and Z coordinates.

    Returns
    -------
    MDAEvent
        Stage-move event.
    """
    x_pos = round(float(stage_position["x"]), 2)
    y_pos = round(float(stage_position["y"]), 2)
    z_pos = round(float(stage_position["z"]), 2)
    return MDAEvent(
        x_pos=x_pos,
        y_pos=y_pos,
        z_pos=z_pos,
        action=CustomAction(
            name=ACTION_STAGE_MOVE,
            data={
                "Stage": {
                    "x_pos": x_pos,
                    "y_pos": y_pos,
                    "z_pos": z_pos,
                }
            },
        ),
    )


def create_asi_scan_setup_event(
    start_mm: float,
    end_mm: float,
    speed_mm_s: float,
) -> MDAEvent:
    """Create an event that configures ASI stage-scan hardware.

    Parameters
    ----------
    start_mm : float
        Stage-scan start coordinate in millimeters.
    end_mm : float
        Stage-scan end coordinate in millimeters.
    speed_mm_s : float
        Stage speed in millimeters per second.

    Returns
    -------
    MDAEvent
        ASI hardware-configuration event.
    """
    return _custom_event(
        ACTION_ASI_SETUP_SCAN,
        {
            "ASI": {
                "mode": "scan",
                "scan_axis_start_mm": float(start_mm),
                "scan_axis_end_mm": float(end_mm),
                "scan_axis_speed_mm_s": float(speed_mm_s),
            }
        },
    )
