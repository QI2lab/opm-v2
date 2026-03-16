"""
Methods to create OPM acquisition custom events

2025/09/05 SJS: initialization
"""

from pathlib import Path
from typing import Dict, List, Optional

from useq import CustomAction, MDAEvent

ROI_KEYS = ("center_x", "center_y", "crop_x", "crop_y")


def create_timelapse_event(interval: int, time_steps: int, timepoint: int) -> MDAEvent:
    """Create an event that pauses the acquisition

    Parameters
    ----------
    interval: int
        time interval for pause in seconds
    time_steps: int
        number of time points in acquisition
    timepoint: int
        the current time point

    Returns
    -------
    MDAEvent
        Custom event which pauses the running acquisition
    """
    timelapse_event = MDAEvent(
        action=CustomAction(
            name="Timelapse",
            data={
                "plan": {
                    "interval": interval,
                    "timepoint": timepoint,
                    "time_steps": time_steps,
                }
            },
        )
    )
    return timelapse_event


def create_fluidics_event(total_rounds: int, current_round: int) -> MDAEvent:
    """Create an event that runs the fluidics program

    Parameters
    ----------
    total_rounds: int
        number of fluidics rounds
    current_round: int
        current fluidics round to run

    Returns
    -------
    MDAEvent
        Custom event that runs the current fluidics round
    """
    FP_event = MDAEvent(
        # exposure = exposure_ms,
        action=CustomAction(
            name="Fluidics",
            data={
                "Fluidics": {
                    "total_rounds": int(total_rounds),
                    "current_round": int(current_round),
                }
            },
        )
    )
    return FP_event


def create_ao_optimize_event(
    ao_data: Dict, output_dir_path: Optional[Path] = None
) -> MDAEvent:
    """_summary_

    Parameters
    ----------
    config: Dict
        OPM configuration from disk
    output_dir_path: Optional[Path], optional
        Path to save optimization results, by default None

    Returns
    -------
    MDAEvent
        Custom event that runs the sensorless A.O.
    """
    # Event for running AO optimization
    ao_optimize_event = MDAEvent(
        action=CustomAction(
            name="AO-optimize",
            data={
                "AO": {
                    "channel_states": ao_data["channel_states"],
                    "channel_powers": ao_data["channel_powers"],
                    "mirror_state": str(ao_data["mirror_state"]),
                    "metric_acceptance": str(ao_data["metric_acceptance"]),
                    "daq_mode": str(ao_data["daq_mode"]),
                    "exposure_ms": float(ao_data["exposure_ms"]),
                    "modal_delta": float(ao_data["mode_delta"]),
                    "metric_precision": int(ao_data["metric_precision"]),
                    "num_averaged_frames": int(ao_data["num_averaged_frames"]),
                    "modal_alpha": float(ao_data["mode_alpha"]),
                    "iterations": int(ao_data["num_iterations"]),
                    "num_mode_samples": int(ao_data["num_mode_samples"]),
                    "metric": str(ao_data["metric"]),
                    "modes_to_optimize": str(ao_data["modes_to_optimize"]),
                    "image_mirror_range_um": ao_data["image_mirror_range_um"],
                    "lightsheet_mode": str(ao_data["lightsheet_mode"]),
                    "readout_ms": float(ao_data["readout_ms"]),
                    "apply_existing": bool(False),
                    "pos_idx": int(0),
                    "scan_idx": int(0),
                    "time_idx": int(0),
                    "output_path": output_dir_path,
                },
                "Camera": {
                    "exposure_ms": float(ao_data["exposure_ms"]),
                    "camera_roi": ao_data["camera_roi"],
                },
            },
        )
    )

    return ao_optimize_event


def create_ao_grid_event(
    ao_data: Dict, output_dir_path: Optional[Path] = None
) -> MDAEvent:
    """Custom MDA event to run AO grid function

    Parameters
    ----------
    config: Dict
        OPM configuration from disk
    output_dir_path: Optional[Path], optional
        Path to save optimization results, by default None

    Returns
    -------
    MDAEvent
        custom event to run ao grid
    """
    ao_grid_event = MDAEvent(
        action=CustomAction(
            name="AO-grid",
            data={
                "AO": {
                    "stage_positions": None,
                    "num_scan_positions": ao_data["num_scan_positions"],
                    "num_tile_positions": ao_data["num_tile_positions"],
                    "output_path": output_dir_path,
                    "apply_ao_map": bool(False),
                    "pos_idx": int(0),
                    "scan_idx": int(0),
                    "time_idx": int(0),
                    "ao_dict": {
                        "channel_states": ao_data["channel_states"],
                        "channel_powers": ao_data["channel_powers"],
                        "mirror_state": str(ao_data["mirror_state"]),
                        "metric_acceptance": str(ao_data["metric_acceptance"]),
                        "daq_mode": str(ao_data["daq_mode"]),
                        "exposure_ms": float(ao_data["exposure_ms"]),
                        "modal_delta": float(ao_data["mode_delta"]),
                        "metric_precision": int(ao_data["metric_precision"]),
                        "num_averaged_frames": int(ao_data["num_averaged_frames"]),
                        "modal_alpha": float(ao_data["mode_alpha"]),
                        "iterations": int(ao_data["num_iterations"]),
                        "num_mode_samples": int(ao_data["num_mode_samples"]),
                        "metric": str(ao_data["metric"]),
                        "modes_to_optimize": str(ao_data["modes_to_optimize"]),
                        "image_mirror_range_um": ao_data["image_mirror_range_um"],
                        "lightsheet_mode": str(ao_data["lightsheet_mode"]),
                        "readout_ms": float(ao_data["readout_ms"]),
                        "apply_existing": bool(False),
                    },
                },
                "Camera": {
                    "exposure_ms": float(ao_data["exposure_ms"]),
                    "camera_roi": ao_data["camera_roi"],
                },
            },
        )
    )
    return ao_grid_event


def create_ao_mirror_update_event(
    mirror_coeffs: Optional[List]
) -> MDAEvent:
    """Create an event to set the AO mirror state

    Parameters
    ----------
    mirror_coeffs: Optional[List], optional
        Mirror modal coefficients, by default None
    mirror_positions: Optional[List], optional
        Mirror voltage values, by default None

    Returns
    -------
    MDAEvent
        custom event to set mirror state
    """
    ao_mirror_update = MDAEvent(
        action=CustomAction(
            name="AO-mirrorUpdate",
            data={
                "AOmirror": {
                    "modal_coeffs": mirror_coeffs.tolist()
                }
            },
        )
    )
    return ao_mirror_update


def create_o2o3_autofocus_event(config: dict) -> MDAEvent:
    """Create a custom MDA event to run the o2-o3 autofocus

    Parameters
    ----------
    exposure_ms: int
        camera exposure in ms
    camera_roi[0]: List[int]
        camera center, [x, y]
    camera_roi: List[int]
        camera crop, [x, y]

    Returns
    -------
    MDAEvent
        Custom event that runs the o2-o3 autofocus routine
    """
    camera_roi = [
        config["Camera"]["center_x"],
        config["Camera"]["center_y"],
        config["Camera"]["crop_x"],
        config["O2O3-autofocus"]["crop_y"],
    ]
    af_event = MDAEvent(
        action=CustomAction(
            name="O2O3-autofocus",
            data={
                "Camera": {
                    "exposure_ms": config["O2O3-autofocus"]["exposure_ms"],
                    "camera_roi": [
                        int(camera_roi[0] - camera_roi[1] // 2),
                        int(camera_roi[1] - camera_roi[2] // 2),
                        int(camera_roi[0]),
                        int(camera_roi[1]),
                    ],
                }
            },
        )
    )
    return af_event


def create_daq_move_event(image_mirror_v: float) -> MDAEvent:
    """Create a custom event to modify the image mirror nuetral position

    Parameters
    ----------
    image_mirror_v: float
        mirror voltage to apply as nuetral position

    Returns
    -------
    MDAEvent
        Custom event that modifies the image mirror neutral position
    """
    daq_move_event = MDAEvent(
        action=CustomAction(
            name="Mirror-Move", data={"DAQ": {"image_mirror_v": image_mirror_v}}
        )
    )
    return daq_move_event


def create_daq_event(daq_data: dict, camera_data: dict) -> MDAEvent:
    """Creates a daq event that updates the daq state to run in a given mode

    Parameters
    ----------
    mode: str, optional
        daq mode to use, [2d, projection, mirror, stage], by default 2d
    channel_states: List[bool], optional
        channel states for all sources, by default [False, False, False, False, False]
    channel_powers: List[bool], optional
        laser powers for each channel, by default [0, 0, 0, 0, 0]
    channel_exposures_ms: List[int], optional
        camera exposures for each channel, by default [0, 0, 0, 0, 0]
    camera_roi[0]: List[int], optional
        camera center [x,y], by default [0, 0]
    camera_roi: List[int], optional
        camera crop [x,y], by default [0, 0]
    interleaved: bool, optional
        by default False
    laser_blanking: bool, optional
        by default True
    image_mirror_range_um: Optional[float], optional
        by default 0
    image_mirror_step_um: Optional[float], optional
        by default 0.4

    Returns
    -------
    MDAEvent
        Custom event that programs the daq
    """
    # create DAQ hardware setup event
    DAQ_event = MDAEvent(
        action=CustomAction(
            name="DAQ",
            data={
                "DAQ": {
                    "mode": daq_data["mode"],
                    "channel_states": daq_data["channel_states"],
                    "channel_powers": daq_data["channel_powers"],
                    "interleaved": daq_data["interleaved"],
                    "blanking": daq_data["laser_blanking"],
                    "image_mirror_range_um": daq_data["image_mirror_range_um"],
                    "image_mirror_step_um": daq_data["image_mirror_step_um"],
                },
                "Camera": {
                    "channel_exposures_ms": camera_data["channel_exposures_ms"],
                    "camera_roi": camera_data["camera_roi"],
                },
            },
        )
    )
    return DAQ_event


def create_stage_event(stage_position: Dict) -> MDAEvent:
    """Create an event that moves the stage to given position

    Parameters
    ----------
    stage_position: Dict
        Dict containing the "x", "y" and "z" stage position

    Returns
    -------
    MDAEvent
        Custom event that moves the xyz stage
    """
    stage_event = MDAEvent(
        action=CustomAction(
            name="Stage-Move",
            data={
                "Stage": {
                    "x_pos": stage_position["x"],
                    "y_pos": stage_position["y"],
                    "z_pos": stage_position["z"],
                }
            },
        )
    )
    return stage_event


def create_asi_scan_setup_event(
    start_mm: float, end_mm: float, speed_mm_s: float
) -> MDAEvent:
    """Create a custom event that sets the ASI controller up for stage scan
    NOTE: positions are in mm and rounded to 2

    Parameters
    ----------
    start_mm: float
        scan start position in mm
    end_mm: float
        scan end position in mm
    speed_mm_s: _type_
        stage scan speed in mm/s

    Returns
    -------
    MDAEvent
        Custom event that sets the ASI controller up for a stage scan along X-axis
    """
    asi_setup_event = MDAEvent(
        action=CustomAction(
            name="ASI-setupscan",
            data={
                "ASI": {
                    "mode": "scan",
                    "scan_axis_start_mm": start_mm,
                    "scan_axis_end_mm": end_mm,
                    "scan_axis_speed_mm_s": speed_mm_s,
                }
            },
        )
    )

    return asi_setup_event
