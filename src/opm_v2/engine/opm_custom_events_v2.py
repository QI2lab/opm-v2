"""Factories and constants for OPM custom MDA events.

The event payload keys intentionally match ``opm_custom_events.py`` so existing
metadata and ``OPMEngine`` behavior remain compatible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from useq import CustomAction, MDAEvent

ACTION_TIMELAPSE = "Timelapse"
ACTION_FLUIDICS = "Fluidics"
ACTION_AO_OPTIMIZE = "AO-optimize"
ACTION_AO_GRID = "AO-grid"
ACTION_AO_MIRROR_UPDATE = "AO-mirrorUpdate"
ACTION_O2O3_AUTOFOCUS = "O2O3-autofocus"
ACTION_MIRROR_MOVE = "Mirror-Move"
ACTION_DAQ = "DAQ"
ACTION_STAGE_MOVE = "Stage-Move"
ACTION_ASI_SETUP_SCAN = "ASI-setupscan"

VALID_DAQ_MODES = {"2d", "projection", "mirror", "stage"}


def _as_list(value: Any) -> list | None:
    """Return a JSON-friendly list for numpy arrays, tuples, or lists."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _custom_event(name: str, data: Mapping[str, Any]) -> MDAEvent:
    """Create a custom MDA event with a consistent action payload."""
    return MDAEvent(action=CustomAction(name=name, data=dict(data)))


def _camera_crop_from_center(
    camera_center: Sequence[float],
    camera_crop: Sequence[float],
) -> list[int]:
    """Convert center/crop values into Micro-Manager ROI coordinates."""
    return [
        int(camera_center[0] - camera_crop[0] // 2),
        int(camera_center[1] - camera_crop[1] // 2),
        int(camera_crop[0]),
        int(camera_crop[1]),
    ]


def _ao_camera_crop(config: dict) -> tuple[int, int, int, int]:
    """Return AO camera center/crop values from config."""
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
    """Return AO channel-state and power lists from config."""
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
    """Build the AO payload shared by optimize and grid events."""
    ao_config = config["acq_config"]["AO"]
    channel_states, channel_powers = _ao_channel_settings(config)
    readout_us = (
        ao_config["readout_us"] if "readout_us" in ao_config else ao_config["readout_ms"]
    )
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
        "z_project_metric": str(ao_config.get("z_project_metric", "DCT")),
        "z_project_range_um": float(ao_config.get("z_project_range_um", 4.0)),
        "z_project_step_um": float(ao_config.get("z_project_step_um", 0.115)),
        "modal_alpha": float(ao_config["mode_alpha"]),
        "iterations": int(ao_config["num_iterations"]),
        "num_mode_samples": int(ao_config["num_mode_samples"]),
        "metric": str(ao_config["metric"]),
        "modes_to_optimize": str(ao_config["modes_to_optimize"]),
        "image_mirror_range_um": ao_config["image_mirror_range_um"],
        "lightsheet_mode": str(ao_config["lightsheet_mode"]),
        "readout_ms": float(readout_us),
        "apply_existing": False,
        "pos_idx": 0,
        "scan_idx": 0,
        "time_idx": 0,
        "output_path": output_dir_path,
    }


def create_timelapse_event(interval: int, time_steps: int, timepoint: int) -> MDAEvent:
    """Create an event that pauses acquisition until the next timepoint."""
    return _custom_event(
        ACTION_TIMELAPSE,
        {
            "plan": {
                "interval": int(interval),
                "timepoint": int(timepoint),
                "time_steps": int(time_steps),
            }
        },
    )


def create_fluidics_event(total_rounds: int, current_round: int) -> MDAEvent:
    """Create an event that runs the requested fluidics round."""
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
    """Create an event that runs sensorless AO optimization."""
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
    """Create an event that runs AO grid optimization."""
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
                "num_scan_positions": ao_config["num_scan_positions"],
                "num_tile_positions": ao_config["num_tile_positions"],
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
    """Create an event that runs O2/O3 autofocus."""
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
    """Create an event that changes the image-mirror neutral voltage."""
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
    """Create an event that programs the DAQ/camera acquisition state."""
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
    """Create an event that moves the XYZ stage."""
    return _custom_event(
        ACTION_STAGE_MOVE,
        {
            "Stage": {
                "x_pos": stage_position["x"],
                "y_pos": stage_position["y"],
                "z_pos": stage_position["z"],
            }
        },
    )


def create_asi_scan_setup_event(
    start_mm: float,
    end_mm: float,
    speed_mm_s: float,
) -> MDAEvent:
    """Create an event that prepares the ASI controller for stage scanning."""
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
