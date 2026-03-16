"""
Methods for creating OPM acquisition event structures

- Optimize now (o2o3 autofocus and adaptive optics)
- Timelapse (Fast image time series, 2d or 3d with multiple image-mirror positions)
- Projection scan (2d 'sum' projection)
- Mirror scan (3d image-mirror scanning)
- Stage scan (3d stage scanning)

2025/09/05 SJS: Update to use opm_custom_events
2026/03/16: Refactoring changes made to config and setup_events
"""

import json
from datetime import datetime
from pathlib import Path
from types import MappingProxyType as mappingproxy
from typing import Dict, List

import numpy as np
from pymmcore_plus import CMMCorePlus
from tqdm import trange
from useq import MDAEvent, MDASequence

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
    create_timelapse_event,
)
from opm_v2.handlers.opm_mirror_handler import OPMMirrorHandler
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ

DEBUGGING = True
MAX_IMAGE_MIRROR_RANGE_UM = 250
ROI_KEYS = ("center_x", "center_y", "crop_x", "crop_y")

# TODO starting mirror state may not be correct, need to set miror state first


# ---------------------------------------------------------#
# Helper methods for consistency
# ---------------------------------------------------------#
def clone_event(ev: MDAEvent) -> MDAEvent:
    """Deep-clone an MDAEvent template."""
    return MDAEvent(**ev.model_dump())


def stage_positions_from_grid(
    opm_mode: str,
    position_data: dict,
    daq_data: dict,
    mda_grid_plan: dict,
    camera_roi: list,
    mda_z_plan: dict = None,
    verbose: bool = False,
) -> List[Dict]:
    """Generates stage positions list

    TODO: Break up scans based on coverslip slope in each direction.

    Parameters
    ----------
    mda_grid_plan : _type_
        _description_
    mda_z_plan : _type_
        _description_
    opm_mode : str
        _description_
    camera_roi: list[int,int,int,int]
        Camera center x, center y, crop x, crop y
    scan_range_um : float
        _description_
    scan_axis_overlap : Optional[float], optional
        _description_, by default 0.2
    tile_axis_overlap : Optional[float], optional
        _description_, by default 0.2
    z_axis_overlap : Optional[float], optional
        _description_, by default 0.2
    coverslip_max_dz : Optional[float], optional
        _description_, by default None
    coverslip_slope_x : Optional[float], optional
        _description_, by default 0
    coverslip_slope_y : Optional[float], optional
        _description_, by default 0

    Returns
    -------
    List[Dict]
        List of stage positions stored as dictionaries
    """
    _mmc = CMMCorePlus.instance()
    pixel_size_um = _mmc.getPixelSizeUm()

    # --------------------------------------------------------------------------#
    # Compile the Z-axis settings
    # --------------------------------------------------------------------------#
    # Get the min and max z-positions
    if mda_z_plan is not None:
        max_z_pos = float(mda_z_plan["top"])
        min_z_pos = float(mda_z_plan["bottom"])
    else:
        min_z_pos = _mmc.getZPosition()
        max_z_pos = _mmc.getZPosition()

    # Calculate the z-step
    if min_z_pos == max_z_pos:
        n_z_positions = 1
        z_step_um = 0
    else:
        range_z_um = np.round(np.abs(max_z_pos - min_z_pos), 2)
        z_step_max = (
            camera_roi[3]
            * pixel_size_um
            * (1.0 - position_data["z_axis_overlap"])
            * np.sin((np.pi / 180.0) * float(30))
        )
        n_z_positions = int(np.ceil(range_z_um / z_step_max))
        z_step_um = np.round(range_z_um / n_z_positions, 2)

    # Correct stage step direction
    if min_z_pos > max_z_pos:
        z_step_um *= -1

    # --------------------------------------------------------------------------#
    # Compile the XY-axis settings
    # --------------------------------------------------------------------------#
    # grab grid plan extents
    min_y_pos = mda_grid_plan["bottom"]
    max_y_pos = mda_grid_plan["top"]
    min_x_pos = mda_grid_plan["left"]
    max_x_pos = mda_grid_plan["right"]

    # Correct stage step directions moves
    if min_x_pos > max_x_pos:
        min_x_pos, max_x_pos = max_x_pos, min_x_pos
    if min_y_pos > max_y_pos:
        min_y_pos, max_y_pos = max_y_pos, min_y_pos

    # Set grid axes ranges
    range_x_um = np.round(np.abs(max_x_pos - min_x_pos), 2)
    range_y_um = np.round(np.abs(max_y_pos - min_y_pos), 2)

    # Calculate Y-axis step and number
    if range_y_um == 0:
        y_step_um = 0
        n_y_positions = 1
    else:
        y_step_max = (
            camera_roi[2] * pixel_size_um * (1 - position_data["tile_axis_overlap"])
        )
        n_y_positions = int(np.ceil(range_y_um / y_step_max))
        y_step_um = np.round(range_y_um / n_y_positions, 2)

    # Calculate X-axis step and number
    if daq_data["image_mirror_range_um"] == 0:
        raise Exception("Image scan range set to 0")
    else:
        x_step_max = daq_data["image_mirror_range_um"] * (
            1 - position_data["scan_axis_overlap"]
        )
        n_x_positions = int(np.ceil(range_x_um / x_step_max))
        x_step_um = np.round(range_x_um / n_x_positions, 2)

    if "projection" in opm_mode or "mirror" in opm_mode:
        if x_step_max > MAX_IMAGE_MIRROR_RANGE_UM:
            x_step_max = MAX_IMAGE_MIRROR_RANGE_UM

    # Modify x, y, & z ranges for coverslip slope
    if position_data["coverslip_slope_x"] != 0:
        cs_x_max_pos = min_z_pos + range_x_um * position_data["coverslip_slope_x"]
        cs_x_range_um = np.round(np.abs(cs_x_max_pos - min_z_pos), 2)
        dz_per_x_tile = np.round(cs_x_range_um / n_x_positions, 2)
    else:
        dz_per_x_tile = 0

    if position_data["coverslip_slope_y"] != 0:
        cs_y_max_pos = min_z_pos + range_y_um * position_data["coverslip_slope_y"]
        cs_y_range_um = np.round(np.abs(cs_y_max_pos - min_z_pos), 2)
        dz_per_y_tile = np.round(cs_y_range_um / n_y_positions, 2)
    else:
        dz_per_y_tile = 0

    # --------------------------------------------------------------------------#
    # Generate XYZ position list
    # --------------------------------------------------------------------------#
    stage_positions = []
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
                stage_positions.append(
                    {
                        "x": min_x_pos + ii * x_step_um,
                        "y": min_y_pos + jj * y_step_um,
                        "z": min_z_pos
                        + kk * z_step_um
                        + ii * dz_per_x_tile
                        + jj * dz_per_y_tile,
                    }
                )

    if verbose:
        print(
            "\nXYZ Stage position settings:",
            f"x start: {min_x_pos}",
            f"x end: {max_x_pos}",
            f"y start: {min_y_pos}",
            f"y end: {max_y_pos}",
            f"z position min:{min_z_pos}",
            f"z position max:{max_z_pos}",
            f"scan range (um): {daq_data['image_mirror_range_um']}",
            "Coverslip slope (x/y): "
            f"{position_data['coverslip_slope_x']}/{position_data['coverslip_slope_y']}",
            f"Number x tiles: {n_x_positions}",
            f"Number y tiles: {n_y_positions}",
            f"Number z tiles: {n_z_positions}",
            f"x tile length um: {x_step_um}",
            f"y tile length um: {y_step_um}",
            sep="\n",
        )

    return stage_positions


def cameraROI(config: dict):
    """Return the camera ROI

    Parameters
    ----------
    config : dict
        OPM configuration

    Returns
    -------
    list
        [center_x, center_y, crop_x, crop_y]
    """
    camera_roi = [int(config["acq_config"]["camera_roi"][k]) for k in ROI_KEYS]

    # Calculate the Y-crop for projection mode
    if config["acq_config"]["opm_mode"] == "projection":
        camera_roi[3] = int(
            config["acq_config"]["DAQ"]["scan_range_um"]
            / config["OPM"]["pixel_size_um"]
        )
    return camera_roi


def acq_modes(config: dict):
    """Returns the settings for each available mode"""
    opm_mode = config["acq_config"]["opm_mode"]
    ao_mode = config["acq_config"]["AO"]["ao_mode"]
    o2o3_mode = config["acq_config"]["O2O3-autofocus"]["o2o3_mode"]
    fluidics_mode = config["acq_config"]["fluidics"]
    return [opm_mode, ao_mode, o2o3_mode, fluidics_mode]


def camera_metadata(config: dict):
    """Return the camera metadata

    Parameters
    ----------
    config : dict
        OPM configuration

    Returns
    -------
    dict
        camera_roi:[center_x, center_y, crop_x, crop_y],
        camera_offset, e_to_ADU
    """
    _mmc = CMMCorePlus.instance()

    camera_roi = cameraROI(config)
    try:
        offset = _mmc.getProperty(
            config["Camera"]["camera_id"], "CONVERSION FACTOR OFFSET"
        )
        e_to_ADU = _mmc.getProperty(
            config["Camera"]["camera_id"], "CONVERSION FACTOR COEFF"
        )
    except Exception:
        offset = 0.0
        e_to_ADU = 1.0
    camera_data = {
        "camera_roi": camera_roi,
        "offset": offset,
        "e_to_ADU": e_to_ADU,
        "channel_exposures_ms": config["acq_config"]["DAQ"]["channel_exposures_ms"],
    }
    return camera_data


def position_metadata(
    config: dict,
):
    """Generate the stage position overlaps and coverslip slope metadata

    Parameters
    ----------
    config : dict
        OPM configuration

    Returns
    -------
    dict
        Metadta for generating stage positions
    """
    position_config = config["acq_config"]["Positions"]
    position_data = {
        "coverslip_slope_x": position_config["coverslip_slope_x"],
        "coverslip_slope_y": position_config["coverslip_slope_y"],
        "coverslip_max_dz": position_config["coverslip_max_dz"],
        "tile_axis_overlap": float(position_config["tile_axis_overlap"]),
        "scan_axis_overlap": float(position_config["scan_axis_overlap"]),
        "scan_axis_overlap_um": float(position_config["scan_axis_overlap_um"]),
        "z_axis_overlap": float(position_config["z_axis_overlap"]),
    }
    return position_data


def daq_metadata(config: dict, current_channel: str = None):
    """Generate the DAQ metadata

    Parameters
    ----------
    config : dict
        OPM configuration
    current_channel : str, optional
        The current channel ID, by default None

    Returns
    -------
    dict
        DAQ data dict for creating DAQ events
    """
    daq_config = config["acq_config"]["DAQ"]

    # Compile the active channel exposures
    channel_states = daq_config["channel_states"]
    channel_exposures_ms = daq_config["channel_exposures_ms"]
    active_channel_exps = []
    for ii, ch_state in enumerate(channel_states):
        if ch_state:
            active_channel_exps.append(np.round(channel_exposures_ms[ii], 2))

    # Determine if interleaved possible
    interleaved_acq = len(set(active_channel_exps)) == 1

    # Compile DAQ data dictionary
    daq_data = {
        "mode": str(config["acq_config"]["opm_mode"]),
        "image_mirror_range_um": float(daq_config["scan_range_um"]),
        "image_mirror_step_um": float(daq_config["scan_axis_step_um"]),
        "scan_axis_step_um": float(daq_config["scan_axis_step_um"]),
        "channel_states": channel_states,
        "channel_exposures_ms": channel_exposures_ms,
        "active_channel_exposures": active_channel_exps,
        "laser_powers": daq_config["channel_powers"],
        "blanking": daq_config["laser_blanking"],
        "interleaved": interleaved_acq,
        "current_channel": current_channel if current_channel is not None else None,
        "n_active_channels": sum(daq_config["channel_states"]),
    }
    return daq_data


def ao_metadata(config: dict):
    """Generate the AO metadata

    Parameters
    ----------
    config : dict
        OPM configuration

    Returns
    -------
    dict
        AO data dict for creating AO events
    """
    ao_config = config["acq_config"]["AO"]

    # Create channel selection lists
    channel_states = [False] * len(config["OPM"]["channel_ids"])
    channel_powers = [0.0] * len(config["OPM"]["channel_ids"])
    for chan_idx, chan_str in enumerate(config["OPM"]["channel_ids"]):
        if ao_config["channel_id"] == chan_str:
            channel_states[chan_idx] = True
            channel_powers[chan_idx] = ao_config["channel_power"]

    if sum(channel_powers) == 0:
        raise Exception("AO channel power is 0!")

    # Get the camera ROI
    camera_roi = cameraROI(config)
    if ao_config["daq_mode"] == "projection":
        camera_roi[3] = int(
            ao_config["image_mirror_range_um"] / config["OPM"]["pixel_size_um"]
        )
    # Compile AD data dictionary
    ao_data = {
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
        "apply_existing": bool(False),
        "pos_idx": int(0),
        "scan_idx": int(0),
        "time_idx": int(0),
        "output_path": None,
        "camera_roi": camera_roi,
        "num_scan_posiitons": ao_config["num_scan_positions"],
        "num_tile_positions": ao_config["num_tile_positions"],
        "time_interval": ao_config["time_interval"],
        "starting_modal_coeffs": None,
        "current_modal_coeffs": None,
    }

    return ao_data


def populate_opm_metadata(
    config: dict,
    daq_data: dict,
    ao_data: dict,
    camera_data: dict,
    stage_position: dict,
    current_channel: str,
    current_exposure_ms: float,
):
    try:
        image_mirror_voltage = daq_data["image_mirror_voltage"]
    except Exception:
        image_mirror_voltage = config["NIDAQ"]["image_mirror_neutral_v"]

    if daq_data["mode"] == "stage":
        excess_start_images = int(
            config["acq_config"]["stage_scan"]["excess_start_frames"]
        )
        excess_end_images = int(config["acq_config"]["stage_scan"]["excess_end_frames"])
        excess_image = bool(stage_position["excess_image"])
    else:
        excess_start_images = None
        excess_end_images = None
        excess_image = False

    # Get the stage orientations
    camera_Zstage_orientation = config["OPM"]["camera_Zstage_orientation"]
    camera_XYstage_orientation = config["OPM"]["camera_XYstage_orientation"]
    camera_mirror_orientation = config["OPM"]["camera_mirror_orientation"]

    # Compile metadata
    metadata = {
        "DAQ": {
            "mode": str(daq_data["mode"]),
            "image_mirror_voltage": float(image_mirror_voltage),
            "image_mirror_range_um": float(daq_data["image_mirror_range_um"]),
            "image_mirror_step_um": float(daq_data["image_mirror_step_um"]),
            "scan_axis_step_um": float(daq_data["scan_axis_step_um"]),
            "channel_states": daq_data["channel_states"],
            "exposure_channels_ms": daq_data["channel_exposures_ms"],
            "laser_powers": daq_data["laser_powers"],
            "interleaved": daq_data["interleaved"],
            "blanking": daq_data["blanking"],
            "current_channel": current_channel,
        },
        "Camera": {
            "exposure_ms": float(current_exposure_ms),
            "camera_center_x": int(camera_data["camera_roi"][0]),
            "camera_center_y": int(camera_data["camera_roi"][1]),
            "camera_crop_x": int(camera_data["camera_roi"][2]),
            "camera_crop_y": int(camera_data["camera_roi"][3]),
            "offset": float(camera_data["offset"]),
            "e_to_ADU": float(camera_data["e_to_ADU"]),
        },
        "OPM": {
            "angle_deg": float(config["OPM"]["angle_deg"]),
            "camera_Zstage_orientation": str(camera_Zstage_orientation),
            "camera_XYstage_orientation": str(camera_XYstage_orientation),
            "camera_mirror_orientation": str(camera_mirror_orientation),
            "excess_scan_start_positions": int(excess_start_images),
            "excess_scan_end_positions": int(excess_end_images),
        },
        "Stage": {
            "x_pos": float(stage_position["x"]),
            "y_pos": float(stage_position["y"]),
            "z_pos": float(stage_position["z"]),
            "excess_image": excess_image,
        },
        "AO_mirror": {
            "starting_modal_coeffs": ao_data["starting_modal_coeffs"],
            "current_modal_coeffs": ao_data["current_modal_coeffs"],
        },
    }
    return metadata


def save_events_json(opm_events, filepath: Path):
    """
    Save a list of MDAEvent objects to a JSON file for inspection.

    Parameters
    ----------
    opm_events : list[MDAEvent]
        Event list returned by setup_events
    filepath : Path
        Destination json file
    """

    events_dict = []

    for ii, event in enumerate(opm_events):
        # Convert to plain dictionary
        ev_dict = event.model_dump()

        # add index for easier debugging
        ev_dict["event_index"] = ii

        events_dict.append(ev_dict)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(events_dict, f, indent=2)


# ---------------------------------------------------------#
# Methods for generating OPM custom acquisitions
# ---------------------------------------------------------#


def setup_optimizenow(mmc: CMMCorePlus, config: dict, output: Path) -> list[MDAEvent]:
    """Runs either A.O. optimization or O2O3 auto focus

    Parameters
    ----------
    mmc : CMMCorePlus
        MMCor instance
    config : dict
        OPM config from disk

    Returns
    -------
    list[MDAEvent]
        OPM events
    """
    ao_mode = config["acq_config"]["AO"]["ao_mode"]
    o2o3_mode = config["acq_config"]["O2O3-autofocus"]["o2o3_mode"]

    # Sequentially run auto focus then AO optmization
    opm_events: list[MDAEvent] = []

    if "now" in o2o3_mode:
        o2o3_event = create_o2o3_autofocus_event(config)
        opm_events.append(o2o3_event)

    if "now" in ao_mode:
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output is None:
            ao_output = Path(str(config["acq_config"]["AO"]["save_dir_path"])) / Path(
                f"{time}_ao_optimize_now"
            )
        else:
            ao_output = output / f"{time}_ao_optimize_now"
        ao_data = ao_metadata(config)
        ao_optimize_event = create_ao_optimize_event(ao_data, ao_output)
        opm_events.append(ao_optimize_event)

    return opm_events, None


def setup_timelapse(
    mmc: CMMCorePlus,
    config: dict,
    sequence: MDASequence,
    output: Path,
) -> list[MDAEvent]:
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
    list[MDAEvent]
        _description_
    Handler
        OPM zarr file saving handler
    """
    OPMdaq_setup = OPMNIDAQ.instance()
    AOmirror_setup = AOMirror.instance()

    # --------------------------------------------------------------------------#
    # Compile acquisition settings from OPM configuration
    # --------------------------------------------------------------------------#
    # Get the acquisition modes
    opm_mode, ao_mode, o2o3_mode, fluidics_mode = acq_modes(config)
    # Get the stage position settings
    position_data = position_metadata(config)
    # Get the DAQ setttings
    daq_data = daq_metadata(config)
    # Get the camera metadata
    camera_data = camera_metadata(config)

    # Catch no laser power cases
    if sum(daq_data["channel_powers"]) == 0:
        raise Exception("All lasers set to 0!")
    elif sum(daq_data["channel_states"]) == 0:
        raise Exception("No channels selected!")

    # --------------------------------------------------------------------------#
    # Get MDA settings
    # --------------------------------------------------------------------------#
    sequence_dict = json.loads(sequence.model_dump_json())
    mda_grid_plan = sequence_dict["grid_plan"]
    mda_time_plan = sequence_dict["time_plan"]
    mda_positions_plan = sequence_dict["stage_positions"]
    mda_z_plan = sequence_dict["z_plan"]

    if (mda_grid_plan is None) and (mda_positions_plan is None):
        raise Exception("Must select MDA grid or positions plan for mirror scanning")

    # --------------------------------------------------------------------------#
    # Generate xyz stage positions
    # --------------------------------------------------------------------------#
    if mda_grid_plan is not None:
        stage_positions = stage_positions_from_grid(
            opm_mode=opm_mode,
            daq_data=daq_data,
            position_data=position_data,
            mda_grid_plan=mda_grid_plan,
            mda_z_plan=mda_z_plan,
            camera_roi=camera_data["camera_roi"],
        )
    elif mda_positions_plan is not None:
        stage_positions = []
        for stage_pos in mda_positions_plan:
            stage_positions.append(
                {
                    "x": float(stage_pos["x"]),
                    "y": float(stage_pos["y"]),
                    "z": float(stage_pos["z"]),
                }
            )
    n_stage_positions = len(stage_positions)

    # --------------------------------------------------------------------------#
    # Determine time axis settings
    # --------------------------------------------------------------------------#
    n_time_steps = mda_time_plan["loops"]

    # estimate the timeloop duration
    estimated_loop_duration_s = (
        sum(daq_data["active_channel_exposures"]) / 1000.0
    ) * n_time_steps + 1.0

    # If the timelapse loop is longer tan 6 hours,
    # include a AO mirror update in the middle of the loop
    if estimated_loop_duration_s > 6 * 60 * 60:
        update_ao_mirror_mid_loop = True
    else:
        update_ao_mirror_mid_loop = False

    # --------------------------------------------------------------------------#
    # Determine the number of scan positions from the DAQ
    # --------------------------------------------------------------------------#
    if daq_data["image_mirror_range_um"] == 0.0:
        # setup daq for 2d scan
        scan_mode = "2d"
        OPMdaq_setup.set_acquisition_params(
            scan_type=scan_mode,
            channel_states=daq_data["channel_states"],
        )
        n_scan_steps = 1
        mirror_voltages = np.array([config["NIDAQ"]["image_mirror_neutral_v"]])
    else:
        scan_mode = "mirror"
        OPMdaq_setup.set_acquisition_params(
            scan_type=scan_mode,
            channel_states=daq_data["channel_states"],
            image_mirror_range_um=daq_data["image_mirror_range_um"],
            image_mirror_step_um=daq_data["image_mirror_step_um"],
        )
        OPMdaq_setup.generate_waveforms()
        mirror_voltages = np.unique(OPMdaq_setup._ao_waveform[:, 0])
        n_scan_steps = mirror_voltages.shape[0]

    # --------------------------------------------------------------------------#
    # Compile mda acquisition settings from active tabs
    # --------------------------------------------------------------------------#

    # Split apart sequence dictionary
    sequence_dict = json.loads(sequence.model_dump_json())
    mda_positions_plan = sequence_dict["stage_positions"]
    mda_time_plan = sequence_dict["time_plan"]

    if (mda_positions_plan is None) or (mda_time_plan is None):
        raise Exception("Must select MDA Positions AND Time plan")

        # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#
    # Create custom action data
    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#

    # Create DAQ event
    daq_event = create_daq_event(daq_data, camera_data)

    # Update the AO mirror positions array with starting values
    AOmirror_setup.n_positions = n_stage_positions
    starting_modal_coeffs = AOmirror_setup.current_coeffs.copy()
    AOmirror_setup.positions_modal_array[:] = starting_modal_coeffs.copy()

    # Initialize ao metadata
    ao_data = ao_metadata(config)
    ao_data["starting_modal_coeffs"] = starting_modal_coeffs.copy()

    # Create the initial AO event data
    if ao_mode != "none":
        ao_data["starting_modal_coeffs"] = starting_modal_coeffs.copy()

        # Create root directory to store AO optimize results
        if "grid" in ao_mode:
            ao_output_dir = output.parent / Path("ao_grid_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_grid_event = create_ao_grid_event(config, ao_output_dir)
            ao_grid_event.action.data["AO"]["stage_positions"] = stage_positions

        else:
            ao_output_dir = output.parent / Path("ao_optimized_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_optimize_event = create_ao_optimize_event(config, ao_output_dir)

    # Create the o2o3 AF event data
    if o2o3_mode != "none":
        o2o3_event = create_o2o3_autofocus_event(config)

    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#
    # Create MDA event structure, Nt / Np / Nc / Nz
    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#
    opm_events: list[MDAEvent] = []

    if DEBUGGING:
        print(
            "Acquisition shape values:"
            f"timepoints: {n_time_steps}"
            f"Num scan positions: {n_scan_steps}",
            f"Stage positions: {n_stage_positions}"
            f"Active channels: {daq_data['n_active_channels']}"
            f"Estimated loop duration (s): {estimated_loop_duration_s:.2f}",
            f"Update AO mid loop: {update_ao_mirror_mid_loop}",
        )

    # --------------------------------------------------------------------------#
    # Create events to run before acquisition
    # --------------------------------------------------------------------------#
    if o2o3_mode == "once at start":
        opm_events.append(o2o3_event)

    if ao_mode == "once at start":
        # Move stage to middle position and run AO once, move stage back
        opm_events.append(create_stage_event(stage_positions[n_stage_positions // 2]))
        opm_events.append(clone_event(ao_optimize_event))

    elif ao_mode == "grid at start":
        opm_events.append(clone_event(ao_grid_event))

    # --------------------------------------------------------------------------#
    # Start acquisition events
    # --------------------------------------------------------------------------#
    for pos_idx in trange(n_stage_positions, desc="Stage positions:", leave=True):
        opm_events.append(create_stage_event(stage_positions[pos_idx]))

        for scan_idx in trange(
            n_scan_steps, desc="Mirror scan positions:", leave=False
        ):
            # ------------------------------------------------------------#
            # Move the image mirror to position
            daq_move_event = create_daq_move_event(mirror_voltages[scan_idx])
            opm_events.append(daq_move_event)

            # Auto-focus event at xyz positions
            if o2o3_mode == "at xyz positions":
                opm_events.append(o2o3_event)

            # ---------------------------------------------------------#
            # AO mirror events
            # ---------------------------------------------------------#
            # Run at xyz position optimizations
            if ao_mode == "at xyz positions":
                ao_dir = ao_output_dir / Path(f"pos_{pos_idx}_ao_results")
                ao_dir.mkdir(exist_ok=True)
                current_ao_event = clone_event(ao_optimize_event)
                current_ao_event.action.data["AO"]["output_path"] = ao_dir
                current_ao_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                current_ao_event.action.data["AO"]["apply_existing"] = False
                opm_events.append(current_ao_event)

            # Update mirror state from optimized per xyz position
            elif ao_mode == "at xyz positions":
                mirror_update_event = clone_event(ao_optimize_event)
                mirror_update_event.action.data["AO"]["apply_existing"] = True
                mirror_update_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                ao_data["current_modal_coeffs"] = AOmirror_setup.positions_modal_array[
                    pos_idx
                ]
                opm_events.append(mirror_update_event)

            # Update mirror state using starting coeffs
            elif ao_mode == "none":
                opm_events.append(
                    create_ao_mirror_update_event(
                        mirror_coeffs=starting_modal_coeffs.copy()
                    )
                )
                ao_data["current_modal_coeffs"] = starting_modal_coeffs.copy()

            # ------------------------------------------------------------#
            # acquire sequenced timelapse images
            # Update daq to perform a 2d scan
            opm_events.append(daq_event)
            for time_idx in trange(n_time_steps, desc="Timepoints:", leave=False):
                # Check for AO update mid loop
                if update_ao_mirror_mid_loop and time_idx == int(n_time_steps // 2):
                    ao_mirror_update = create_ao_mirror_update_event(
                        mirror_coeffs=ao_data["current_modal_coeffs"]
                    )
                    opm_events.append(ao_mirror_update)

                # Create the timelapse image events
                current_chan_idx = 0
                for chan_idx, chan_bool in enumerate(daq_data["channel_states"]):
                    if chan_bool:
                        # Create image event for current t / p / c / z
                        image_event = MDAEvent(
                            index=mappingproxy(
                                {
                                    "t": time_idx,
                                    "p": pos_idx,
                                    "c": current_chan_idx,
                                    "z": scan_idx,
                                }
                            ),
                            metadata=populate_opm_metadata(
                                config,
                                daq_data,
                                ao_data,
                                camera_data,
                                stage_positions[pos_idx],
                                config["OPM"]["channel_ids"][chan_idx],
                                daq_data["channel_exposures_ms"][chan_idx],
                            ),
                        )
                        opm_events.append(image_event)
                        current_chan_idx += 1

    # --------------------------------------------------------------------------#
    # Setup OPM custom handler
    # NOTE: output path needs to only have a single '.', or multiple suffixes are found!
    if len(Path(output).suffixes) == 1 and Path(output).suffix == ".zarr":
        indice_sizes = {
            "t": int(np.maximum(1, n_time_steps)),
            "p": int(n_stage_positions),
            "c": int(np.maximum(1, daq_data["n_active_channels"])),
            "z": int(np.maximum(1, n_scan_steps)),
        }
        handler = OPMMirrorHandler(
            path=Path(output), indice_sizes=indice_sizes, delete_existing=True
        )
        print(f"Using Qi2lab handler\nindices: {indice_sizes}")
        return opm_events, handler
    else:
        raise Exception("Defualt handler selected, modify save path!")


def setup_projection(
    mmc: CMMCorePlus,
    config: dict,
    sequence: MDASequence,
    output: Path,
) -> list[MDAEvent]:
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
    list[MDAEvent]
        _description_
    Handler
        OPM zarr file saving handler
    """

    AOmirror_setup = AOMirror.instance()

    # --------------------------------------------------------------------------#
    # Compile acquisition settings from OPM configuration
    # --------------------------------------------------------------------------#
    # Get the acquisition modes
    opm_mode, ao_mode, o2o3_mode, fluidics_mode = acq_modes(config)
    # Get the stage position settings
    position_data = position_metadata(config)
    # Get the DAQ setttings
    daq_data = daq_metadata(config)
    # Get the camera metadata
    camera_data = camera_metadata(config)

    # Catch no laser power cases
    if sum(daq_data["channel_powers"]) == 0:
        raise Exception("All lasers set to 0!")
    elif sum(daq_data["channel_states"]) == 0:
        raise Exception("No channels selected!")

    # --------------------------------------------------------------------------#
    # Get MDA settings
    # --------------------------------------------------------------------------#
    sequence_dict = json.loads(sequence.model_dump_json())
    mda_grid_plan = sequence_dict["grid_plan"]
    mda_time_plan = sequence_dict["time_plan"]
    mda_positions_plan = sequence_dict["stage_positions"]
    mda_z_plan = sequence_dict["z_plan"]

    if (mda_grid_plan is None) and (mda_positions_plan is None):
        raise Exception("Must select MDA grid or positions plan for Projection")

    # --------------------------------------------------------------------------#
    # Generate xyz stage positions
    # --------------------------------------------------------------------------#
    if mda_grid_plan is not None:
        stage_positions = stage_positions_from_grid(
            opm_mode=opm_mode,
            daq_data=daq_data,
            position_data=position_data,
            mda_grid_plan=mda_grid_plan,
            mda_z_plan=mda_z_plan,
            camera_roi=camera_data["camera_roi"],
        )
    elif mda_positions_plan is not None:
        stage_positions = []
        for stage_pos in mda_positions_plan:
            stage_positions.append(
                {
                    "x": float(stage_pos["x"]),
                    "y": float(stage_pos["y"]),
                    "z": float(stage_pos["z"]),
                }
            )
    n_stage_positions = len(stage_positions)

    # --------------------------------------------------------------------------#
    # Determine time axis settings
    # --------------------------------------------------------------------------#
    if fluidics_mode != "none":
        n_time_steps = int(fluidics_mode)
        time_interval = 0
    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan["loops"]
        time_interval = mda_time_plan["interval"]
    else:
        n_time_steps = 1
        time_interval = 1

    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#
    # Create custom action data
    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#

    # Create DAQ event
    daq_event = create_daq_event(daq_data, camera_data)

    # Update the AO mirror positions array with starting values
    AOmirror_setup.n_positions = n_stage_positions
    starting_modal_coeffs = AOmirror_setup.current_coeffs.copy()
    AOmirror_setup.positions_modal_array[:] = starting_modal_coeffs.copy()

    # Initialize ao metadata
    ao_data = ao_metadata(config)
    ao_data["starting_modal_coeffs"] = starting_modal_coeffs.copy()

    # Create the initial AO event data
    if ao_mode != "none":
        ao_data["starting_modal_coeffs"] = starting_modal_coeffs.copy()
        ao_time_interval = ao_data["time_interval"]

        # Create root directory to store AO optimize results
        if "grid" in ao_mode:
            ao_output_dir = output.parent / Path("ao_grid_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_grid_event = create_ao_grid_event(config, ao_output_dir)
            ao_grid_event.action.data["AO"]["stage_positions"] = stage_positions

        else:
            ao_output_dir = output.parent / Path("ao_optimized_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_optimize_event = create_ao_optimize_event(config, ao_output_dir)
    else:
        ao_time_interval = 1

    # Create the o2o3 AF event data
    if o2o3_mode != "none":
        o2o3_event = create_o2o3_autofocus_event(config)

    # Set flag to avoid programming DAQ, and sequence timepoints
    if time_interval == 0 and ao_mode == "none" and o2o3_mode == "none":
        need_to_setup_daq = False
    else:
        need_to_setup_daq = True

    # Set flag to avoid creating unnecessary stage events
    if n_stage_positions > 1 and "grid" not in ao_mode:
        need_to_setup_stage = False
    else:
        need_to_setup_stage = True

    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#
    # Create MDA event structure, Nt/ Np / Nc
    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#
    opm_events: list[MDAEvent] = []

    if DEBUGGING:
        print(
            "Projection Acquisition Parameters:"
            f"  timepoints / interval: {n_time_steps} / {time_interval}",
            f"  Stage positions: {n_stage_positions}",
            f"  Active channels: {daq_data['n_active_channels']}",
            f"  o2o3 focus frequency: {o2o3_mode}",
            f"  AO frequency: {ao_mode}",
            f"  AO mirror starting coeffs: {starting_modal_coeffs}",
            sep="\n",
        )

    # --------------------------------------------------------------------------#
    # Create events to run before acquisition
    # --------------------------------------------------------------------------#
    if o2o3_mode == "once at start":
        opm_events.append(o2o3_event)

    if ao_mode == "once at start":
        # Move stage to middle position and run AO once, move stage back
        opm_events.append(create_stage_event(stage_positions[n_stage_positions // 2]))
        opm_events.append(clone_event(ao_optimize_event))

    elif ao_mode == "grid at start":
        opm_events.append(clone_event(ao_grid_event))

    # --------------------------------------------------------------------------#
    # Start acquisition events
    # --------------------------------------------------------------------------#
    for time_idx in trange(n_time_steps, desc="Timepoints:", leave=True):
        # ----------------------------------------------------------------------#
        # Create events to run each time-point
        # ----------------------------------------------------------------------#
        # Move stage to starting position
        opm_events.append(create_stage_event(stage_positions[0]))

        # Run fluidics if requested
        if fluidics_mode != "none" and time_idx != 0:
            current_fluidics_event = create_fluidics_event(int(fluidics_mode), time_idx)
            opm_events.append(current_fluidics_event)

        # Create timelapse pause events
        if (mda_time_plan is not None) and (time_idx > 0) and (int(time_interval) > 0):
            opm_events.append(
                create_timelapse_event(time_interval, n_time_steps, time_idx)
            )

        # Create autofocus event
        if o2o3_mode == "at timepoints":
            opm_events.append(o2o3_event)

        # Create AO optimization events
        if time_idx % ao_time_interval == 0:
            if ao_mode == "at timepoints":
                ao_dir = ao_output_dir / Path(f"time_{time_idx}_ao_results")
                ao_dir.mkdir(exist_ok=True)
                curr_ao_opt_event = clone_event(ao_optimize_event)
                curr_ao_opt_event.action.data["AO"]["output_path"] = ao_dir
                curr_ao_opt_event.action.data["AO"]["time_idx"] = time_idx
                curr_ao_opt_event.action.data["AO"]["pos_idx"] = 0
                opm_events.append(curr_ao_opt_event)

            elif ao_mode == "grid at timepoints":
                ao_dir = ao_output_dir / Path(f"time_{time_idx}_ao_grid_results")
                ao_dir.mkdir(exist_ok=True)
                curr_ao_grid_event = clone_event(ao_grid_event)
                curr_ao_grid_event.action.data["AO"]["output_path"] = ao_dir
                curr_ao_grid_event.action.data["AO"]["time_idx"] = time_idx
                opm_events.append(curr_ao_grid_event)

        # ----------------------------------------------------------------------#
        # iterate over stage positions
        # ----------------------------------------------------------------------#
        for pos_idx in range(n_stage_positions):
            # Move stage to position
            if need_to_setup_stage:
                opm_events.append(create_stage_event(stage_positions[pos_idx]))

            # Auto-focus event at xyz positions
            if o2o3_mode == "at xyz positions":
                opm_events.append(o2o3_event)

            # ---------------------------------------------------------#
            # AO mirror events
            # ---------------------------------------------------------#
            # Run at xyz position optimizations
            if (ao_mode == "at xyz positions") and (time_idx == 0):
                ao_dir = ao_output_dir / Path(
                    f"time_{time_idx}_pos_{pos_idx}_ao_results"
                )
                ao_dir.mkdir(exist_ok=True)
                current_ao_event = clone_event(ao_optimize_event)
                current_ao_event.action.data["AO"]["output_path"] = ao_dir
                current_ao_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                current_ao_event.action.data["AO"]["time_idx"] = int(time_idx)
                current_ao_event.action.data["AO"]["apply_existing"] = False
                opm_events.append(current_ao_event)

            # update mirror state using grid optimization
            elif ao_mode == "grid at start" or ao_mode == "grid at timepoints":
                mirror_update_event = clone_event(ao_grid_event)
                mirror_update_event.action.data["AO"]["apply_ao_map"] = True
                mirror_update_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                ao_data["current_modal_coeffs"] = AOmirror_setup.positions_modal_array[
                    pos_idx
                ]
                opm_events.append(mirror_update_event)

            # Update mirror state from optimized at start or timepoints
            elif ao_mode == "at timepoints" or ao_mode == "once at start":
                mirror_update_event = clone_event(ao_optimize_event)
                mirror_update_event.action.data["AO"]["apply_existing"] = True
                # Note at timepoints only populates the 0 pos-idx of the array
                mirror_update_event.action.data["AO"]["pos_idx"] = 0
                ao_data["current_modal_coeffs"] = AOmirror_setup.positions_modal_array[
                    0
                ]
                opm_events.append(mirror_update_event)

            # Update mirror state from optimized per xyz position
            elif ao_mode == "at xyz positions":
                mirror_update_event = clone_event(ao_optimize_event)
                mirror_update_event.action.data["AO"]["apply_existing"] = True
                mirror_update_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                ao_data["current_modal_coeffs"] = AOmirror_setup.positions_modal_array[
                    pos_idx
                ]
                opm_events.append(mirror_update_event)

            # Update mirror state using starting coeffs
            elif ao_mode == "none":
                opm_events.append(
                    create_ao_mirror_update_event(
                        mirror_coeffs=starting_modal_coeffs.copy()
                    )
                )

            # ------------------------------------------------------------------#
            # Handle acquiring images
            # ------------------------------------------------------------------#
            if daq_data["interleaved_acq"]:
                if time_idx == 0:
                    opm_events.append(clone_event(daq_event))
                if need_to_setup_daq:
                    opm_events.append(clone_event(daq_event))

            # Create image events
            current_chan_idx = 0
            for chan_idx, chan_bool in enumerate(daq_data["channel_states"]):
                if chan_bool:
                    if not (daq_data["interleaved_acq"]):
                        # Update daq for each channel separately
                        temp_daq_data = daq_data.copy()
                        temp_channel_states = [False] * len(daq_data["channel_states"])
                        temp_channel_states[chan_idx] = True
                        temp_daq_data["channel_states"] = temp_channel_states
                        temp_daq_data["Camera"]
                        opm_events.append(create_daq_event(temp_daq_data))

                    # Create image event for current t / p / c
                    image_event = MDAEvent(
                        index=mappingproxy(
                            {"t": time_idx, "p": pos_idx, "c": current_chan_idx}
                        ),
                        metadata=populate_opm_metadata(
                            config,
                            daq_data,
                            ao_data,
                            camera_data,
                            stage_positions[pos_idx],
                            config["OPM"]["channel_ids"][chan_idx],
                            daq_data["channel_exposures_ms"][chan_idx],
                        ),
                    )
                    opm_events.append(image_event)
                    current_chan_idx += 1

    # Save events for debugging
    save_events_json(opm_events, Path(output).parent / "opm_events.json")

    # --------------------------------------------------------------------------#
    # Setup our OPM Engine
    # --------------------------------------------------------------------------#
    # NOTE: output path needs to only have a single '.', or multiple suffixes are found!
    if len(Path(output).suffixes) == 1 and Path(output).suffix == ".zarr":
        indice_sizes = {
            "t": int(np.maximum(1, n_time_steps)),
            "p": int(np.maximum(1, n_stage_positions)),
            "c": int(np.maximum(1, daq_data["n_active_channels"])),
        }
        handler = OPMMirrorHandler(
            path=Path(output), indice_sizes=indice_sizes, delete_existing=True
        )
        return opm_events, handler
    else:
        raise Exception("Defualt handler selected, modify save path!")


def setup_mirrorscan(
    mmc: CMMCorePlus,
    config: dict,
    sequence: MDASequence,
    output: Path,
) -> list[MDAEvent]:
    """Creates an OPM mirror scan acquisition

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
    list[MDAEvent]
        _description_
    Handler
        OPM zarr file saving handler
    """
    AOmirror_setup = AOMirror.instance()
    OPMdaq_setup = OPMNIDAQ.instance()

    # --------------------------------------------------------------------------#
    # Compile acquisition settings from OPM configuration
    # --------------------------------------------------------------------------#
    # Get the acquisition modes
    opm_mode, ao_mode, o2o3_mode, fluidics_mode = acq_modes(config)
    # Get the stage position settings
    position_data = position_metadata(config)
    # Get the DAQ setttings
    daq_data = daq_metadata(config)
    # Get the camera metadata
    camera_data = camera_metadata(config)

    # Catch no laser power cases
    if sum(daq_data["channel_powers"]) == 0:
        raise Exception("All lasers set to 0!")
    elif sum(daq_data["channel_states"]) == 0:
        raise Exception("No channels selected!")

    # --------------------------------------------------------------------------#
    # Get MDA settings
    # --------------------------------------------------------------------------#
    sequence_dict = json.loads(sequence.model_dump_json())
    mda_grid_plan = sequence_dict["grid_plan"]
    mda_time_plan = sequence_dict["time_plan"]
    mda_positions_plan = sequence_dict["stage_positions"]
    mda_z_plan = sequence_dict["z_plan"]

    if (mda_grid_plan is None) and (mda_positions_plan is None):
        raise Exception("Must select MDA grid or positions plan for mirror scanning")

    # --------------------------------------------------------------------------#
    # Generate xyz stage positions
    # --------------------------------------------------------------------------#
    if mda_grid_plan is not None:
        stage_positions = stage_positions_from_grid(
            opm_mode=opm_mode,
            daq_data=daq_data,
            position_data=position_data,
            mda_grid_plan=mda_grid_plan,
            mda_z_plan=mda_z_plan,
            camera_roi=camera_data["camera_roi"],
        )
    elif mda_positions_plan is not None:
        stage_positions = []
        for stage_pos in mda_positions_plan:
            stage_positions.append(
                {
                    "x": float(stage_pos["x"]),
                    "y": float(stage_pos["y"]),
                    "z": float(stage_pos["z"]),
                }
            )
    n_stage_positions = len(stage_positions)

    # --------------------------------------------------------------------------#
    # Determine time axis settings
    # --------------------------------------------------------------------------#
    if fluidics_mode != "none":
        n_time_steps = int(fluidics_mode)
        time_interval = 0
    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan["loops"]
        time_interval = mda_time_plan["interval"]
    else:
        n_time_steps = 1
        time_interval = 1

    # --------------------------------------------------------------------------#
    # Determine the number of scan positions from the DAQ
    # --------------------------------------------------------------------------#
    if daq_data["image_mirror_range_um"] == 0.0:
        daq_data["mode"] = "2d"
        OPMdaq_setup.set_acquisition_params(scan_type="2d")
        n_scan_steps = 1
    else:
        daq_data["mode"] = "mirror"
        OPMdaq_setup.set_acquisition_params(
            scan_type="mirror",
            image_mirror_range_um=daq_data["image_mirror_range_um"],
            image_mirror_step_um=daq_data["image_mirror_step_um"],
        )
        n_scan_steps = OPMdaq_setup.n_scan_steps

    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#
    # Create custom action data
    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#

    # Create DAQ event
    daq_event = create_daq_event(daq_data, camera_data)

    # Update the AO mirror positions array with starting values
    AOmirror_setup.n_positions = n_stage_positions
    starting_modal_coeffs = AOmirror_setup.current_coeffs.copy()
    AOmirror_setup.positions_modal_array[:] = starting_modal_coeffs.copy()

    # Initialize ao metadata
    ao_data = ao_metadata(config)
    ao_data["starting_modal_coeffs"] = starting_modal_coeffs.copy()

    # Create the initial AO event data
    if ao_mode != "none":
        ao_data["starting_modal_coeffs"] = starting_modal_coeffs.copy()
        ao_time_interval = ao_data["time_interval"]

        # Create root directory to store AO optimize results
        if "grid" in ao_mode:
            ao_output_dir = output.parent / Path("ao_grid_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_grid_event = create_ao_grid_event(config, ao_output_dir)
            ao_grid_event.action.data["AO"]["stage_positions"] = stage_positions

        else:
            ao_output_dir = output.parent / Path("ao_optimized_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_optimize_event = create_ao_optimize_event(config, ao_output_dir)
    else:
        ao_time_interval = 1

    # Create the o2o3 AF event data
    if o2o3_mode != "none":
        o2o3_event = create_o2o3_autofocus_event(config)

    # Set flag to avoid programming DAQ, and sequence timepoints
    if time_interval == 0 and ao_mode == "none" and o2o3_mode == "none":
        need_to_setup_daq = False
    else:
        need_to_setup_daq = True

    # Set flag to avoid creating unnecessary stage events
    if n_stage_positions > 1 and "grid" not in ao_mode:
        need_to_setup_stage = False
    else:
        need_to_setup_stage = True

    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#
    # Create MDA event structure, Nt / Np / Nc / Nz
    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#
    opm_events: list[MDAEvent] = []

    if DEBUGGING:
        print(
            "Mirror-Scan Acquisition settings:",
            f"timepoints / interval: {n_time_steps} / {time_interval}",
            f"Stage positions: {n_stage_positions}",
            f"Active channels: {daq_data['n_active_channels']}",
            f"AO frequency: {ao_mode}",
            f"o2o3 focus frequency: {o2o3_mode}",
            f"num scan steps: {n_scan_steps}",
            f"scan range (um): {daq_data['image_mirror_range_um']}",
            f"scan step (um): {daq_data['image_mirror_step_um']}",
            f"  AO mirror starting coeffs: {starting_modal_coeffs}",
            f"DAQ scan mode: {daq_data['mode']}",
            sep="\n",
        )

    # --------------------------------------------------------------------------#
    # Create events to run before acquisition
    # --------------------------------------------------------------------------#
    if o2o3_mode == "once at start":
        opm_events.append(o2o3_event)

    if ao_mode == "once at start":
        # Move stage to middle position and run AO once, move stage back
        opm_events.append(create_stage_event(stage_positions[n_stage_positions // 2]))
        opm_events.append(clone_event(ao_optimize_event))

    elif ao_mode == "grid at start":
        opm_events.append(clone_event(ao_grid_event))

    # --------------------------------------------------------------------------#
    # Start acquisition events
    # --------------------------------------------------------------------------#
    for time_idx in trange(n_time_steps, desc="Timepoints:", leave=True):
        # ----------------------------------------------------------------------#
        # Create events to run each time-point
        # ----------------------------------------------------------------------#
        # Move stage to starting position
        opm_events.append(create_stage_event(stage_positions[0]))

        # Run fluidics if requested
        if fluidics_mode != "none" and time_idx != 0:
            current_fluidics_event = create_fluidics_event(int(fluidics_mode), time_idx)
            opm_events.append(current_fluidics_event)

        # Create timelapse pause events
        if (mda_time_plan is not None) and (time_idx > 0) and (int(time_interval) > 0):
            opm_events.append(
                create_timelapse_event(time_interval, n_time_steps, time_idx)
            )

        # Create autofocus event
        if o2o3_mode == "at timepoints":
            opm_events.append(o2o3_event)

        # Create AO optimization events
        if time_idx % ao_time_interval == 0:
            if ao_mode == "at timepoints":
                ao_dir = ao_output_dir / Path(f"time_{time_idx}_ao_results")
                ao_dir.mkdir(exist_ok=True)
                curr_ao_opt_event = clone_event(ao_optimize_event)
                curr_ao_opt_event.action.data["AO"]["output_path"] = ao_dir
                curr_ao_opt_event.action.data["AO"]["time_idx"] = time_idx
                curr_ao_opt_event.action.data["AO"]["pos_idx"] = 0
                opm_events.append(curr_ao_opt_event)

            elif ao_mode == "grid at timepoints":
                ao_dir = ao_output_dir / Path(f"time_{time_idx}_ao_grid_results")
                ao_dir.mkdir(exist_ok=True)
                curr_ao_grid_event = clone_event(ao_grid_event)
                curr_ao_grid_event.action.data["AO"]["output_path"] = ao_dir
                curr_ao_grid_event.action.data["AO"]["time_idx"] = time_idx
                opm_events.append(curr_ao_grid_event)

        # ----------------------------------------------------------------------#
        # iterate over stage positions
        # ----------------------------------------------------------------------#
        for pos_idx in range(n_stage_positions):
            # Move stage to position
            if need_to_setup_stage:
                opm_events.append(create_stage_event(stage_positions[pos_idx]))

            # Auto-focus event at xyz positions
            if o2o3_mode == "at xyz positions":
                opm_events.append(o2o3_event)

            # ---------------------------------------------------------#
            # AO mirror events
            # ---------------------------------------------------------#
            # Run at xyz position optimizations
            if (ao_mode == "at xyz positions") and (time_idx == 0):
                ao_dir = ao_output_dir / Path(
                    f"time_{time_idx}_pos_{pos_idx}_ao_results"
                )
                ao_dir.mkdir(exist_ok=True)
                current_ao_event = clone_event(ao_optimize_event)
                current_ao_event.action.data["AO"]["output_path"] = ao_dir
                current_ao_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                current_ao_event.action.data["AO"]["time_idx"] = int(time_idx)
                current_ao_event.action.data["AO"]["apply_existing"] = False
                opm_events.append(current_ao_event)

            # update mirror state using grid optimization
            elif ao_mode == "grid at start" or ao_mode == "grid at timepoints":
                mirror_update_event = clone_event(ao_grid_event)
                mirror_update_event.action.data["AO"]["apply_ao_map"] = True
                mirror_update_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                ao_data["current_modal_coeffs"] = AOmirror_setup.positions_modal_array[
                    pos_idx
                ]
                opm_events.append(mirror_update_event)

            # Update mirror state from optimized at start or timepoints
            elif ao_mode == "at timepoints" or ao_mode == "once at start":
                mirror_update_event = clone_event(ao_optimize_event)
                mirror_update_event.action.data["AO"]["apply_existing"] = True
                # Note at timepoints only populates the 0 pos-idx of the array
                mirror_update_event.action.data["AO"]["pos_idx"] = 0
                ao_data["current_modal_coeffs"] = AOmirror_setup.positions_modal_array[
                    0
                ]
                opm_events.append(mirror_update_event)

            # Update mirror state from optimized per xyz position
            elif ao_mode == "at xyz positions":
                mirror_update_event = clone_event(ao_optimize_event)
                mirror_update_event.action.data["AO"]["apply_existing"] = True
                mirror_update_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                ao_data["current_modal_coeffs"] = AOmirror_setup.positions_modal_array[
                    pos_idx
                ]
                opm_events.append(mirror_update_event)

            # Update mirror state using starting coeffs
            elif ao_mode == "none":
                opm_events.append(
                    create_ao_mirror_update_event(
                        mirror_coeffs=starting_modal_coeffs.copy()
                    )
                )

            # ------------------------------------------------------------------#
            # Handle acquiring images: t / p / c / scan idx
            # ------------------------------------------------------------------#
            if daq_data["interleaved_acq"]:
                if time_idx == 0:
                    opm_events.append(clone_event(daq_event))
                if need_to_setup_daq:
                    opm_events.append(clone_event(daq_event))

                for scan_idx in range(n_scan_steps):
                    current_chan_idx = 0
                    for chan_idx, chan_bool in enumerate(daq_data["channel_states"]):
                        if chan_bool:
                            image_event = MDAEvent(
                                index=mappingproxy(
                                    {
                                        "t": time_idx,
                                        "p": pos_idx,
                                        "c": current_chan_idx,
                                        "z": scan_idx,
                                    }
                                ),
                                metadata=populate_opm_metadata(
                                    config,
                                    daq_data,
                                    ao_data,
                                    camera_data,
                                    stage_positions[pos_idx],
                                    config["OPM"]["channel_ids"][chan_idx],
                                    daq_data["channel_exposures_ms"][chan_idx],
                                ),
                            )
                            opm_events.append(image_event)
                            current_chan_idx += 1
            else:
                # iterate of active channels then acquire scan position
                current_chan_idx = 0
                for chan_idx, chan_bool in enumerate(daq_data["channel_states"]):
                    if chan_bool:
                        # Need a custom daq event for each channel
                        num_channels = len(daq_data["channel_states"])
                        temp_states = [False] * num_channels
                        temp_states[chan_idx] = True

                        # Create daq event for a single channel
                        _daq_event = clone_event(daq_event)
                        _daq_event.action.data["DAQ"]["channel_states"] = temp_states
                        opm_events.append(_daq_event)

                        # Create image event for current t / p / c / scan idx
                        for scan_idx in range(n_scan_steps):
                            image_event = MDAEvent(
                                index=mappingproxy(
                                    {
                                        "t": time_idx,
                                        "p": pos_idx,
                                        "c": current_chan_idx,
                                        "z": scan_idx,
                                    }
                                ),
                                metadata=populate_opm_metadata(
                                    config,
                                    daq_data,
                                    ao_data,
                                    camera_data,
                                    stage_positions[pos_idx],
                                    config["OPM"]["channel_ids"][chan_idx],
                                    daq_data["channel_exposures_ms"][chan_idx],
                                ),
                            )
                            opm_events.append(image_event)
                            current_chan_idx += 1

    # Check if path ends if .zarr. If so, use our OutputHandler
    if len(Path(output).suffixes) == 1 and Path(output).suffix == ".zarr":
        indice_sizes = {
            "t": int(np.maximum(1, n_time_steps)),
            "p": int(np.maximum(1, n_stage_positions)),
            "c": int(np.maximum(1, daq_data["n_active_channels"])),
            "z": int(np.maximum(1, n_scan_steps)),
        }
        handler = OPMMirrorHandler(
            path=Path(output), indice_sizes=indice_sizes, delete_existing=True
        )
        print(f"\nUsing Qi2lab handler,\nindices: {indice_sizes}\n")
        return opm_events, handler
    else:
        raise Exception("Modify save path with .zarr!")


def setup_stagescan(
    mmc: CMMCorePlus,
    config: dict,
    sequence: MDASequence,
    output: Path,
) -> list[MDAEvent]:
    """Setup an OPM stage scan acquisition

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
    list[MDAEvent]
        _description_
    Handler
        OPM zarr file saving handler
    """

    AOmirror_setup = AOMirror.instance()

    # --------------------------------------------------------------------------#
    # Compile acquisition settings from OPM configuration
    # --------------------------------------------------------------------------#
    # Get the acquisition modes
    opm_mode, ao_mode, o2o3_mode, fluidics_mode = acq_modes(config)
    # Get the stage position settings
    position_data = position_metadata(config)
    # Get the DAQ setttings
    daq_data = daq_metadata(config)
    # Get the camera metadata
    camera_data = camera_metadata(config)

    # Catch no laser power cases
    if sum(daq_data["channel_powers"]) == 0:
        raise Exception("All lasers set to 0!")
    elif sum(daq_data["channel_states"]) == 0:
        raise Exception("No channels selected!")

    # --------------------------------------------------------------------------#
    # Get MDA settings
    # --------------------------------------------------------------------------#
    sequence_dict = json.loads(sequence.model_dump_json())
    mda_grid_plan = sequence_dict["grid_plan"]
    mda_time_plan = sequence_dict["time_plan"]
    mda_z_plan = sequence_dict["z_plan"]

    if mda_grid_plan is None:
        raise Exception("Must select MDA grid stage scanning")

    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#
    # Generate xyz stage positions
    # NOTE: the 'scan' axis is along the x-axis
    # NOTE: the 'tile' axis is along the y-axis
    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#

    # Get camera expsoure in ms and s
    exposure_ms = np.round(np.max(daq_data["channel_exposures_ms"]), 2)
    exposure_s = np.round(exposure_ms / 1000.0, 2)

    # Get pixel size and deskew Y-scale factor
    pixel_size_um = config["OPM"]["pixel_size_um"]
    opm_angle_scale = np.sin((np.pi / 180.0) * float(config["OPM"]["angle_deg"]))

    # Get coverslip slope
    coverslip_slope_x = float(position_data["coverslip_slope_x"])
    coverslip_slope_y = float(position_data["coverslip_slope_y"])

    # Get the maximum coverslip change along z-axis
    coverslip_max_dz = float(config["acq_config"]["stage_scan"]["coverslip_max_dz"])

    # Get the maximum scan-axis distance, for a single scan.
    x_axis_max_scan_range = float(
        config["acq_config"]["stage_scan"]["max_stage_scan_range_um"]
    )

    # Get the excess start / end
    excess_start_images = int(config["acq_config"]["stage_scan"]["excess_start_frames"])
    excess_end_images = int(config["acq_config"]["stage_scan"]["excess_end_frames"])

    # Get the XY extents
    min_y_pos = mda_grid_plan["bottom"]
    max_y_pos = mda_grid_plan["top"]
    min_x_pos = mda_grid_plan["left"]
    max_x_pos = mda_grid_plan["right"]

    # Get the Z-axis extents
    if mda_z_plan is not None:
        max_z_pos = float(mda_z_plan["top"])
        min_z_pos = float(mda_z_plan["bottom"])
    else:
        min_z_pos = mmc.getZPosition()
        max_z_pos = mmc.getZPosition()

    # Calculate the XYZ ranges
    range_x_um = np.round(np.abs(max_x_pos - min_x_pos), 2)
    range_y_um = np.round(np.abs(max_y_pos - min_y_pos), 2)
    range_z_um = np.round(np.abs(max_z_pos - min_z_pos), 2)

    # Calculate the coverslip change, used to offset Z positions
    cs_min_pos = min_z_pos
    cs_max_pos_x = cs_min_pos + range_x_um * coverslip_slope_x
    cs_max_pos_y = cs_min_pos + range_y_um * coverslip_slope_y
    cs_range_x_um = np.round(np.abs(cs_max_pos_x - cs_min_pos), 2)
    cs_range_y_um = np.round(np.abs(cs_max_pos_y - cs_min_pos), 2)

    # Calculate XYZ tile steps
    z_axis_step_max = (
        camera_data["camera_roi"][3]
        * pixel_size_um
        * opm_angle_scale
        * (1 - position_data["tile_axis_overlap"])
    )
    y_axis_step_max = (
        camera_data["camera_roi"][2]
        * pixel_size_um
        * (1 - position_data["tile_axis_overlap"])
    )

    # Calculate the x-axis overlap
    x_tile_overlap_um = (
        camera_data["camera_roi"][3] * opm_angle_scale * pixel_size_um
        + position_data["scan_axis_overlap_um"]
    )

    # Check if the coverslip slope determines the maximum scan range
    if coverslip_slope_x != 0:
        coverslip_max_scan_range = np.abs(coverslip_max_dz / coverslip_slope_x)
        if coverslip_max_scan_range < x_axis_max_scan_range:
            x_axis_max_scan_range = coverslip_max_scan_range

    # Correct directions for stage moves, x must go low to high
    if min_z_pos > max_z_pos:
        z_axis_step_max *= -1
    if min_x_pos > max_x_pos:
        min_x_pos, max_x_pos = max_x_pos, min_x_pos

    if DEBUGGING:
        print(
            "\nXYZ Stage scan position settings:",
            f"Scan start / end: {min_x_pos} / {max_x_pos}",
            f"Tile start / end: {min_y_pos} / {max_y_pos}",
            f"Z position min / max:{min_z_pos} / {max_z_pos}",
            f"Coverslip slope x / y: {coverslip_slope_x} / {coverslip_slope_y}",
            f"Coverslip X-axis low / high: {cs_min_pos} / {cs_max_pos_x}",
            f"Coverslip Y-axis low / high: {cs_min_pos} / {cs_max_pos_y}",
            f"Max scan range: {x_axis_max_scan_range}",
            sep="\n",
        )

    # --------------------------------------------------------------------------#
    # Generate scan-axis tile locations, units: mm and s
    # --------------------------------------------------------------------------#
    # Convert scan tile overlap to mm
    scan_tile_overlap_mm = x_tile_overlap_um / 1000.0

    # Calculate the number of scan positions and tile length
    if x_axis_max_scan_range >= range_x_um:
        n_scan_positions = 1
        scan_tile_length_um = range_x_um
    else:
        # Round up so that the scan length is never longer than the max scan range
        n_scan_positions = int(np.ceil(range_x_um / (x_axis_max_scan_range)))
        scan_tile_length_um = np.round(
            (range_x_um / n_scan_positions)
            + (n_scan_positions - 1) * (x_tile_overlap_um / n_scan_positions),
            2,
        )

    # Convert scan start, stop, step, and lengths to mm
    x_axis_step_mm = daq_data["scan_axis_step_um"] / 1000.0
    x_axis_start_mm = min_x_pos / 1000.0
    x_axis_end_mm = max_x_pos / 1000.0
    x_tile_length_mm = np.round(scan_tile_length_um / 1000.0, 2)

    # Populae the scan-axis start and stop arrays
    scan_start_pos_mm = np.full(n_scan_positions, x_axis_start_mm)
    scan_end_pos_mm = np.full(n_scan_positions, x_axis_end_mm)
    for ii in range(n_scan_positions):
        scan_start_pos_mm[ii] = x_axis_start_mm + ii * (
            x_tile_length_mm - scan_tile_overlap_mm
        )
        scan_end_pos_mm[ii] = scan_start_pos_mm[ii] + x_tile_length_mm

    # Round start and stop positions
    scan_start_pos_mm = np.round(scan_start_pos_mm, 2)
    scan_end_pos_mm = np.round(scan_end_pos_mm, 2)

    # Calculate the actual scan tile length
    x_tile_length_mm = np.round(np.abs(scan_end_pos_mm[0] - scan_start_pos_mm[0]), 2)

    # Calculate the number of scan-axis positions
    num_scan_axis_positions = np.rint(x_tile_length_mm / x_axis_step_mm).astype(int)

    # Set the number of scan-axis indices, includes excess images
    n_scan_axis_indices = (
        num_scan_axis_positions + int(excess_start_images) + int(excess_end_images)
    )

    # Calculate the scan speed, mm/s
    scan_speed_mm_s = np.round(
        x_axis_step_mm / exposure_s / daq_data["n_active_channels"], 5
    )

    # Ge the actual scan speed returned by ASI stage
    mmc.setProperty(mmc.getXYStageDevice(), "MotorSpeedX-S(mm/s)", scan_speed_mm_s)
    mmc.waitForDevice(mmc.getXYStageDevice())
    actual_speed_x = float(
        mmc.getProperty(mmc.getXYStageDevice(), "MotorSpeedX-S(mm/s)")
    )
    # Calculate the exposure using the actual scan speed of the stage
    actual_exposure = np.round(
        x_axis_step_mm / actual_speed_x / daq_data["n_active_channels"], 5
    )
    channel_exposures_ms = [actual_exposure * 1000] * len(daq_data["channel_states"])

    # update acq settings with the actual exposure and stage scan speed
    daq_data["channel_exposures_ms"] = channel_exposures_ms
    exposure_ms = actual_exposure * 1000
    exposure_s = actual_exposure
    scan_speed_mm_s = actual_speed_x

    if DEBUGGING:
        test_scan_length = x_tile_length_mm == x_tile_length_mm
        # Verify all the scan tiles are the same size
        scan_tile_sizes = [
            np.round(np.abs(scan_end_pos_mm[ii] - scan_start_pos_mm[ii]), 2)
            for ii in range(len(scan_end_pos_mm))
        ]
        test_tile_sizes = np.allclose(scan_tile_sizes, scan_tile_sizes[0])
        print(
            "\nScan-axis calculated parameters:",
            f"Number scan tiles: {n_scan_positions}",
            f"tile length um / mm: {scan_tile_length_um} / {x_tile_length_mm}",
            f"tile overlap um: {x_tile_overlap_um}",
            f"tile length with overlap (mm): {x_tile_length_mm}",
            f"Does scan tile w/ overlap equal scan tile length?: {test_scan_length}",
            f"step size (mm): {x_axis_step_mm}",
            f"exposure (s): {exposure_s}",
            f"number of active channels: {daq_data['n_active_channels']}",
            f"Scan axis speed (mm/s): {scan_speed_mm_s}",
            f"Scan axis start positions (mm): {scan_start_pos_mm}",
            f"Scan axis end positions (mm): {scan_end_pos_mm}.",
            f"Number of scan positions (mm): {num_scan_axis_positions}",
            f"Are all scan tiles the same size?: {test_tile_sizes}",
            sep="\n",
        )

    # --------------------------------------------------------------------------#
    # Generate tile axis positions
    # --------------------------------------------------------------------------#
    # Caclulate the number of tiles
    n_y_positions = int(np.ceil(range_y_um / y_axis_step_max)) + 1

    # Generate the tile-axis positions
    y_axis_positions = np.round(np.linspace(min_y_pos, max_y_pos, n_y_positions), 2)
    if n_y_positions == 1:
        tile_axis_step = 0
    else:
        tile_axis_step = y_axis_positions[1] - y_axis_positions[0]

    if DEBUGGING:
        print(
            f"Tile axis positions (um): {y_axis_positions}",
            f"Num tile axis positions: {n_y_positions}",
            f"Tile axis step: {tile_axis_step}",
            sep="\n",
        )

    # --------------------------------------------------------------------------#
    # Generate z axis positions
    # --------------------------------------------------------------------------#
    # Calculate the number of positions
    n_z_positions = int(np.ceil(np.abs(range_z_um / z_axis_step_max))) + 1

    # Generate z-axis positions, ignoring the coverslip slope
    z_positions = np.round(np.linspace(min_z_pos, max_z_pos, n_z_positions), 2)

    # Ge the z-axis step
    if n_z_positions == 1:
        z_axis_step_um = 0.0
    else:
        z_axis_step_um = np.round(z_positions[1] - z_positions[0], 2)

    # Calculate the stage z change along the scan axis
    dz_per_x_tile = (cs_range_x_um / n_scan_positions) * np.sign(coverslip_slope_x)
    dz_per_y_tile = (cs_range_y_um / n_scan_positions) * np.sign(coverslip_slope_y)

    if DEBUGGING:
        print(
            "Z axis positions, units: um",
            f"Z axis positions: {z_positions}"
            f"Z axis range: {range_z_um} um"
            f"Z axis step: {z_axis_step_um} um"
            f"Num z axis positions: {n_z_positions}"
            f"Z offset per x-axis-tile: {dz_per_x_tile} um"
            f"Z offset per y-axis-tile: {dz_per_y_tile} um"
            f"Z axis step max: {z_axis_step_max}",
        )

    # --------------------------------------------------------------------------#
    # Generate stage positions
    # --------------------------------------------------------------------------#
    n_stage_positions = n_scan_positions * n_y_positions * n_z_positions
    stage_positions = []
    for z_idx in range(n_z_positions):
        for scan_idx in range(n_scan_positions):
            for tile_idx in range(n_y_positions):
                stage_positions.append(
                    {
                        "x": float(np.round(scan_start_pos_mm[scan_idx] * 1000, 2)),
                        "y": float(np.round(y_axis_positions[tile_idx], 2)),
                        "z": float(
                            np.round(
                                z_positions[z_idx]
                                + dz_per_x_tile * scan_idx
                                + dz_per_y_tile * tile_idx,
                                2,
                            )
                        ),
                    }
                )

    # --------------------------------------------------------------------------#
    # Determine time axis settings
    # --------------------------------------------------------------------------#
    if fluidics_mode != "none":
        n_time_steps = int(fluidics_mode)
        time_interval = 0
    elif mda_time_plan is not None:
        n_time_steps = mda_time_plan["loops"]
        time_interval = mda_time_plan["interval"]
    else:
        n_time_steps = 1
        time_interval = 1

    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#
    # Create custom action data
    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#

    # Create DAQ event
    daq_event = create_daq_event(daq_data, camera_data)

    # Update the AO mirror positions array with starting values
    AOmirror_setup.n_positions = n_stage_positions
    starting_modal_coeffs = AOmirror_setup.current_coeffs.copy()
    AOmirror_setup.positions_modal_array[:] = starting_modal_coeffs.copy()

    # Initialize ao metadata
    ao_data = ao_metadata(config)
    ao_data["starting_modal_coeffs"] = starting_modal_coeffs.copy()

    # Create the initial AO event data
    if ao_mode != "none":
        ao_data["starting_modal_coeffs"] = starting_modal_coeffs.copy()
        ao_time_interval = ao_data["time_interval"]

        # Create root directory to store AO optimize results
        if "grid" in ao_mode:
            ao_output_dir = output.parent / Path("ao_grid_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_grid_event = create_ao_grid_event(config, ao_output_dir)
            ao_grid_event.action.data["AO"]["stage_positions"] = stage_positions

        else:
            ao_output_dir = output.parent / Path("ao_optimized_results")
            ao_output_dir.mkdir(exist_ok=True)
            ao_optimize_event = create_ao_optimize_event(config, ao_output_dir)
    else:
        ao_time_interval = 1

    # Create the o2o3 AF event data
    if o2o3_mode != "none":
        o2o3_event = create_o2o3_autofocus_event(config)

    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#
    # Create MDA event structure, Nt / Np / Nscan / Nc
    # --------------------------------------------------------------------------#
    # --------------------------------------------------------------------------#
    opm_events: list[MDAEvent] = []

    if DEBUGGING:
        print(
            "Acquisition shape values:",
            f"timepoints / interval: {n_time_steps} / {time_interval}",
            f"Stage positions: {n_stage_positions}",
            f"Scan positions: {n_scan_axis_indices}",
            f"Active channels: {daq_data['n_active_channels']}",
            f"Excess frames (S/E):{excess_start_images}/{excess_end_images}",
            f"AO frequency: {ao_mode}",
            f"o2o3 focus frequency: {o2o3_mode}",
            sep="\n",
        )

    # --------------------------------------------------------------------------#
    # Create events to run before acquisition
    # --------------------------------------------------------------------------#
    if o2o3_mode == "once at start":
        opm_events.append(o2o3_event)

    if ao_mode == "once at start":
        # Move stage to middle position and run AO once, move stage back
        opm_events.append(create_stage_event(stage_positions[n_stage_positions // 2]))
        opm_events.append(clone_event(ao_optimize_event))

    elif ao_mode == "grid at start":
        opm_events.append(clone_event(ao_grid_event))

    # --------------------------------------------------------------------------#
    # Start acquisition events
    # --------------------------------------------------------------------------#
    for time_idx in trange(n_time_steps, desc="Timepoints:", leave=True):
        # ----------------------------------------------------------------------#
        # Create events to run each time-point
        # ----------------------------------------------------------------------#
        # Move stage to starting position
        opm_events.append(create_stage_event(stage_positions[0]))

        # Run fluidics if requested
        if fluidics_mode != "none" and time_idx != 0:
            current_fluidics_event = create_fluidics_event(int(fluidics_mode), time_idx)
            opm_events.append(current_fluidics_event)

        # Create timelapse pause events
        if (mda_time_plan is not None) and (time_idx > 0) and (int(time_interval) > 0):
            opm_events.append(
                create_timelapse_event(time_interval, n_time_steps, time_idx)
            )

        # Create autofocus event
        if o2o3_mode == "at timepoints":
            opm_events.append(o2o3_event)

        # Create AO optimization events
        if time_idx % ao_time_interval == 0:
            if ao_mode == "at timepoints":
                ao_dir = ao_output_dir / Path(f"time_{time_idx}_ao_results")
                ao_dir.mkdir(exist_ok=True)
                curr_ao_opt_event = clone_event(ao_optimize_event)
                curr_ao_opt_event.action.data["AO"]["output_path"] = ao_dir
                curr_ao_opt_event.action.data["AO"]["time_idx"] = time_idx
                curr_ao_opt_event.action.data["AO"]["pos_idx"] = 0
                opm_events.append(curr_ao_opt_event)

            elif ao_mode == "grid at timepoints":
                ao_dir = ao_output_dir / Path(f"time_{time_idx}_ao_grid_results")
                ao_dir.mkdir(exist_ok=True)
                curr_ao_grid_event = clone_event(ao_grid_event)
                curr_ao_grid_event.action.data["AO"]["output_path"] = ao_dir
                curr_ao_grid_event.action.data["AO"]["time_idx"] = time_idx
                opm_events.append(curr_ao_grid_event)

        # ----------------------------------------------------------------------#
        # iterate over stage positions
        # ----------------------------------------------------------------------#
        pos_idx = 0
        for z_idx in trange(n_z_positions, desc="Z-axis-tiles:", leave=False):
            for scan_idx in trange(
                n_scan_positions, desc="Scan-axis-tiles:", leave=False
            ):
                for tile_idx in trange(
                    n_y_positions, desc="Tile-axis-tiles:", leave=False
                ):
                    # Move stage to position
                    opm_events.append(create_stage_event(stage_positions[pos_idx]))

                    # Auto-focus event at xyz positions
                    if o2o3_mode == "at xyz positions":
                        opm_events.append(o2o3_event)

                    # ---------------------------------------------------------#
                    # AO mirror events
                    # ---------------------------------------------------------#
                    # Run at xyz position optimizations
                    if (ao_mode == "at xyz positions") and (time_idx == 0):
                        ao_dir = ao_output_dir / Path(
                            f"time_{time_idx}_pos_{pos_idx}_ao_results"
                        )
                        ao_dir.mkdir(exist_ok=True)
                        current_ao_event = clone_event(ao_optimize_event)
                        current_ao_event.action.data["AO"]["output_path"] = ao_dir
                        current_ao_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                        current_ao_event.action.data["AO"]["time_idx"] = int(time_idx)
                        current_ao_event.action.data["AO"]["apply_existing"] = False
                        opm_events.append(current_ao_event)

                    # update mirror state using grid optimization
                    elif ao_mode == "grid at start" or ao_mode == "grid at timepoints":
                        mirror_update_event = clone_event(ao_grid_event)
                        mirror_update_event.action.data["AO"]["apply_ao_map"] = True
                        mirror_update_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                        ao_data["current_modal_coeffs"] = (
                            AOmirror_setup.positions_modal_array[pos_idx]
                        )
                        opm_events.append(mirror_update_event)

                    # Update mirror state from optimized at start or timepoints
                    elif ao_mode == "at timepoints" or ao_mode == "once at start":
                        mirror_update_event = clone_event(ao_optimize_event)
                        mirror_update_event.action.data["AO"]["apply_existing"] = True
                        # Note at timepoints only populates the 0 pos-idx of the array
                        mirror_update_event.action.data["AO"]["pos_idx"] = 0
                        ao_data["current_modal_coeffs"] = (
                            AOmirror_setup.positions_modal_array[0]
                        )
                        opm_events.append(mirror_update_event)

                    # Update mirror state from optimized per xyz position
                    elif ao_mode == "at xyz positions":
                        mirror_update_event = clone_event(ao_optimize_event)
                        mirror_update_event.action.data["AO"]["apply_existing"] = True
                        mirror_update_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                        ao_data["current_modal_coeffs"] = (
                            AOmirror_setup.positions_modal_array[pos_idx]
                        )
                        opm_events.append(mirror_update_event)

                    # Update mirror state using starting coeffs
                    elif ao_mode == "none":
                        opm_events.append(
                            create_ao_mirror_update_event(
                                mirror_coeffs=starting_modal_coeffs.copy()
                            )
                        )

                    # ---------------------------------------------------------#
                    # Handle acquiring images: t / p / c / scan idx
                    # ---------------------------------------------------------#
                    opm_events.append(daq_event)

                    # Set ASI controller for stage scanning and Camera for external Trig
                    current_asi_setup_event = create_asi_scan_setup_event(
                        start_mm=float(scan_start_pos_mm[scan_idx]),
                        end_mm=float(scan_end_pos_mm[scan_idx]),
                        speed_mm_s=float(scan_speed_mm_s),
                    )
                    opm_events.append(current_asi_setup_event)

                    # Create image events
                    for scan_axis_idx in range(n_scan_axis_indices):
                        # Set as excess image
                        end_excess_idx = num_scan_axis_positions + excess_start_images
                        if scan_axis_idx < excess_start_images:
                            is_excess_image = True
                        elif scan_axis_idx > end_excess_idx:
                            is_excess_image = True
                        else:
                            is_excess_image = False

                        for chan_idx, chan_bool in enumerate(
                            daq_data["channel_states"]
                        ):
                            if chan_bool:
                                image_event = MDAEvent(
                                    index=mappingproxy(
                                        {
                                            "t": time_idx,
                                            "p": pos_idx,
                                            "c": chan_idx,
                                            "z": scan_axis_idx,
                                        }
                                    ),
                                    metadata=populate_opm_metadata(
                                        config=config,
                                        daq_data=daq_data,
                                        ao_data=ao_data,
                                        camera_data=camera_data,
                                        stage_position={
                                            "x_pos": stage_positions[pos_idx]["x"]
                                            + (
                                                scan_axis_idx
                                                * daq_data["scan_axis_step_um"]
                                            ),
                                            "y_pos": stage_positions[pos_idx]["y"],
                                            "z_pos": stage_positions[pos_idx]["z"],
                                            "excess_image": is_excess_image,
                                        },
                                        current_channel=daq_data["channel_names"][
                                            chan_idx
                                        ],
                                        current_exposure_ms=daq_data[
                                            "channel_exposures_ms"
                                        ][chan_idx],
                                    ),
                                )
                                opm_events.append(image_event)
                    pos_idx = pos_idx + 1

    # Save events for debugging
    save_events_json(opm_events, Path(output).parent / "opm_events.json")

    # Check if path ends if .zarr. If so, use our OutputHandler
    if len(Path(output).suffixes) == 1 and Path(output).suffix == ".zarr":
        indice_sizes = {
            "t": int(np.maximum(1, n_time_steps)),
            "p": int(np.maximum(1, n_stage_positions)),
            "c": int(np.maximum(1, daq_data["n_active_channels"])),
            "z": int(np.maximum(1, n_scan_axis_indices)),
        }
        handler = OPMMirrorHandler(
            path=Path(output), indice_sizes=indice_sizes, delete_existing=True
        )
        print(f"\nUsing Qi2lab handler,\nindices: {indice_sizes}\n")
        return opm_events, handler
    else:
        raise Exception("Defualt handler selected, modify save path!")
