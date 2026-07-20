"""Implement the OPM pymmcore-plus MDA engine.

This version keeps the original hardware behavior but shares custom-action
constants with ``opm_custom_events`` so event factories and engine dispatch
stay synchronized.

Change Log:
2025-02-07: New version that includes all possible modes
2025/09/05: Synchronized opm_config options and A.O.
"""

import json
import logging
from pathlib import Path
from time import perf_counter, sleep
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from pymmcore_plus.mda import MDAEngine
from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1
from PyQt6.QtCore import QThread
from useq import CustomAction, MDAEvent, MDASequence

from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.engine.opm_custom_events import (
    ACTION_AO_GRID,
    ACTION_AO_MIRROR_UPDATE,
    ACTION_AO_OPTIMIZE,
    ACTION_ASI_SETUP_SCAN,
    ACTION_DAQ,
    ACTION_FLUIDICS,
    ACTION_MIRROR_MOVE,
    ACTION_O2O3_AUTOFOCUS,
    ACTION_STAGE_MOVE,
    ACTION_TIMELAPSE,
)
from opm_v2.engine.debug_printing import (
    debug as _debug,
    info,
    warning,
)
from opm_v2.utils.autofocus_remote_unit import manage_O3_focus
from opm_v2.utils.elveflow_control import run_fluidic_program
from opm_v2.utils.sensorless_ao import run_ao_grid_mapping, run_ao_optimization

logging.getLogger("pymmcore-plus")

DEBUGGING = True
POWER_STR = " - PowerSetpoint (%)"


def debug(header: str, *lines: object) -> None:
    """Print a debug block when module-level debugging is enabled.

    Parameters
    ----------
    header : str
        Block heading.
    *lines : object
        Values to print beneath the heading.
    """
    _debug(header, *lines, enabled=DEBUGGING)


class OPMEngineV2(MDAEngine):
    """Execute OPM custom actions and camera acquisition events.

    Parameters
    ----------
    mmc : CMMCorePlus
        Existing Micro-Manager core instance.
    config_path : Path
        OPM JSON configuration path.
    use_hardware_sequencing : bool
        Whether pymmcore-plus may sequence compatible camera events.
    simulate_hardware : bool or None
        Override for external OPM hardware simulation.
    """

    def __init__(
        self,
        mmc,
        config_path: Path,
        use_hardware_sequencing: bool = True,
        simulate_hardware: bool | None = None,
    ) -> None:
        """Initialize the OPM acquisition engine.

        Parameters
        ----------
        mmc : CMMCorePlus
            Existing Micro-Manager core instance.
        config_path : Path
            OPM JSON configuration path.
        use_hardware_sequencing : bool
            Whether pymmcore-plus may sequence compatible camera events.
        simulate_hardware : bool or None
            Override for external OPM hardware simulation.
        """
        super().__init__(
            mmc,
            use_hardware_sequencing=use_hardware_sequencing,
            restore_initial_state=False,
        )
        self._mmc = mmc
        self.opmDAQ = OPMNIDAQ.instance()
        self.AOMirror = AOMirror.instance()
        self.start_asi_scan_after_camera_sequence = False
        self.start_time = None
        self.elapsed_time = None
        self._config_path = config_path
        self._config = None
        self.update_config()
        configured_simulation = bool(
            self._config.get("OPM", {}).get("simulate_hardware", False)
        )
        self.simulate_hardware = (
            configured_simulation
            if simulate_hardware is None
            else bool(simulate_hardware)
        )
        self.simulated_laser_powers: dict[str, float] = {}
        self.simulated_asi_state: dict[str, float | str] = {}
        self.simulated_asi_transitions: list[str] = []

    def update_config(self):
        """Update the class config dict. from file."""
        with open(self._config_path, "r") as config_file:
            self._config = json.load(config_file)

    def configure_camera(self, data_dict: dict, setting: str = None):
        """Set the camera ROI and exposure.

        Parameters
        ----------
        data_dict : dict
            Custom action data dict
        setting : str or None
            Camera configuration context.
        """
        if not (int(data_dict["Camera"]["camera_crop"][3]) == self._mmc.getROI()[-1]):
            current_roi = self._mmc.getROI()
            self._mmc.clearROI()
            self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
            self._mmc.setROI(
                data_dict["Camera"]["camera_crop"][0],
                data_dict["Camera"]["camera_crop"][1],
                data_dict["Camera"]["camera_crop"][2],
                data_dict["Camera"]["camera_crop"][3],
            )

        if setting == "DAQ":
            exposure_ms = np.max(data_dict["Camera"]["exposure_channels"])
        else:
            exposure_ms = data_dict["Camera"]["exposure_ms"]
        self._mmc.setProperty(
            str(self._config["Camera"]["camera_id"]),
            "Exposure",
            np.round(float(exposure_ms), 2),
        )
        self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
        self._mmc.getROI()

    def configure_stage_camera_trigger(self) -> None:
        """Configure the camera hardware to accept the ASI stage-sync trigger."""
        camera = str(self._config["Camera"]["camera_id"])
        trigger_properties = (
            ("Trigger", "START"),
            ("TriggerPolarity", "POSITIVE"),
            ("TRIGGER SOURCE", "EXTERNAL"),
        )
        for property_name, value in trigger_properties:
            if not self._mmc.hasProperty(camera, property_name):
                continue
            self._mmc.setProperty(camera, property_name, value)
            self._mmc.waitForDevice(camera)
            while self._mmc.getProperty(camera, property_name) != value:
                sleep(0.1)
                self._mmc.setProperty(camera, property_name, value)
                self._mmc.waitForDevice(camera)

    def configure_lasers(self, data_dict: dict, setting: str):
        """Set laser powers.

        Parameters
        ----------
        data_dict : dict
            Custom action data dict
        setting : str
            Payload section used to configure the lasers.

        Returns
        -------
        float or None
            Active camera exposure for DAQ setup, otherwise ``None``.

        Raises
        ------
        Exception
            If ``setting`` is not a supported laser configuration context.
        """
        if (
            not (setting == "AO")
            and not (setting == "DAQ")
            and not (setting == "AO_grid")
        ):
            raise Exception("Engine laser configuration missing setting")
        if setting == "AO_grid":
            enumerator = enumerate(data_dict["AO"]["ao_dict"]["channel_states"])
        else:
            enumerator = enumerate(data_dict[setting]["channel_states"])
        exposure_ms = 0.0
        for chan_idx, chan_bool in enumerator:
            laser_name = str(self._config["Lasers"]["laser_names"][chan_idx])
            if chan_bool:
                if setting == "DAQ":
                    exposure_ms = float(
                        data_dict["Camera"]["exposure_channels"][chan_idx]
                    )
                if setting == "AO_grid":
                    laser_power = float(
                        data_dict["AO"]["ao_dict"]["channel_powers"][chan_idx]
                    )
                else:
                    laser_power = float(data_dict[setting]["channel_powers"][chan_idx])
                if self.simulate_hardware:
                    self.simulated_laser_powers[laser_name] = laser_power
                else:
                    self._mmc.setProperty(
                        self._config["Lasers"]["name"],
                        laser_name + " - PowerSetpoint (%)",
                        laser_power,
                    )
            else:
                if self.simulate_hardware:
                    self.simulated_laser_powers[laser_name] = 0.0
                else:
                    self._mmc.setProperty(
                        self._config["Lasers"]["name"],
                        laser_name + " - PowerSetpoint (%)",
                        0.0,
                    )
        if setting == "DAQ":
            return exposure_ms

    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Set up system state before an MDA sequence.

        This method is called once at the beginning of a sequence.
        (The sequence object needn't be used here if not necessary)

        Parameters
        ----------
        sequence : MDASequence
            Sequence about to run.

        Returns
        -------
        SummaryMetaV1 or None
            Summary metadata returned by the base engine.
        """
        self.start_time = perf_counter()
        self.elapsed_time = 0
        buffer_mb = (
            64
            if self.simulate_hardware
            else int(self._config["OPM"].get("circular_buffer_mb", 32000))
        )
        self._mmc.setCircularBufferMemoryFootprint(buffer_mb)
        return super().setup_sequence(sequence)

    def setup_event(self, event: MDAEvent) -> None:
        """Prepare state of system (hardware, etc.) for `event`.

        This method is called before each event in the sequence. It is
        responsible for preparing the state of the system for the event.
        The engine should be in a state where it can call `exec_event`
        without any additional preparation.

        Parameters
        ----------
        event : MDAEvent
            Event whose hardware state should be prepared.
        """
        if isinstance(event.action, CustomAction):
            action_name = event.action.name
            data_dict = event.action.data
            if action_name == ACTION_O2O3_AUTOFOCUS:
                # Stop DAQ playback
                if self.opmDAQ.running():
                    self.opmDAQ.stop_waveform_playback()
                self.opmDAQ.reset_ao_channels()

                # Setup camera properties
                self.configure_camera(data_dict)

            elif action_name == ACTION_STAGE_MOVE:
                # update config from file for up-to-date stage move speed
                self.update_config()

                # --------------------------------------------------------#
                # Move stage to position, with normal speed
                stage_move_speed = self._config["OPM"]["stage_move_speed"]
                xy_stage = self._mmc.getXYStageDevice()
                if self._mmc.hasProperty(xy_stage, "MotorSpeedX-S(mm/s)"):
                    self._mmc.setProperty(
                        xy_stage, "MotorSpeedX-S(mm/s)", stage_move_speed
                    )
                if self._mmc.hasProperty(xy_stage, "MotorSpeedY-S(mm/s)"):
                    self._mmc.setProperty(
                        xy_stage, "MotorSpeedY-S(mm/s)", stage_move_speed
                    )
                self._mmc.setPosition(np.round(float(data_dict["Stage"]["z_pos"]), 2))
                self._mmc.waitForDevice(self._mmc.getFocusDevice())
                target_x = np.round(float(data_dict["Stage"]["x_pos"]), 2)
                target_y = np.round(float(data_dict["Stage"]["y_pos"]), 2)
                current_x, current_y = self._mmc.getXYPosition()
                old_x = current_x
                old_y = current_y
                self._mmc.setXYPosition(target_x, target_y)
                counter = 0
                # Move stage and wait until we are within 1um of the target position.
                while not (np.isclose(current_x, target_x, rtol=0.0, atol=1.0)) or not (
                    np.isclose(current_y, target_y, rtol=0.0, atol=1.0)
                ):
                    sleep(0.5)
                    current_x, current_y = self._mmc.getXYPosition()
                    if old_x == current_x and old_y == current_y:
                        counter = counter + 1
                        debug(
                            "STAGE MOVE STATIONARY",
                            f"current_x: {current_x}",
                            f"current_y: {current_y}",
                            f"target_x: {target_x}",
                            f"target_y: {target_y}",
                        )
                    else:
                        old_x = current_x
                        old_y = current_y
                    if counter >= 5:
                        break

            elif action_name == ACTION_ASI_SETUP_SCAN:
                if self.simulate_hardware:
                    self.simulated_asi_state = {
                        "scan_axis_start_mm": float(
                            data_dict["ASI"]["scan_axis_start_mm"]
                        ),
                        "scan_axis_end_mm": float(data_dict["ASI"]["scan_axis_end_mm"]),
                        "scan_axis_speed_mm_s": float(
                            data_dict["ASI"]["scan_axis_speed_mm_s"]
                        ),
                        "scan_state": "Idle",
                    }
                    self.simulated_asi_transitions.append("Idle")
                    self.start_asi_scan_after_camera_sequence = True
                    return
                # --------------------------------------------------------#
                # Setup PLC controller for TTL output to stage sync signal
                plcName = self._config["PLC"]["name"]  # 'PLogic:E:36'
                propPosition = self._config["PLC"]["position"]  # 'PointerPosition'
                propCellConfig = self._config["PLC"]["cellconfig"]  # 'EditCellConfig'
                addrOutputBNC1 = int(
                    self._config["PLC"]["pin"]
                )  # 33 BNC1 on the PLC front panel
                addrStageSync = int(
                    self._config["PLC"]["signalid"]
                )  # 46 TTL5 on Tiger backplane = stage sync signal
                self._mmc.setProperty(plcName, propPosition, addrOutputBNC1)
                self._mmc.setProperty(plcName, propCellConfig, addrStageSync)

                # --------------------------------------------------------#
                # Set stage speed, scan axis (x) and tile axis (y)
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "MotorSpeedX-S(mm/s)",
                    np.round(data_dict["ASI"]["scan_axis_speed_mm_s"], 4),
                )
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "MotorSpeedY-S(mm/s)",
                    self._config["OPM"]["stage_move_speed"],
                )

                # --------------------------------------------------------#
                # Set scan axis to true 1D scan with no backlash
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(), "ScanPattern", "Raster"
                )
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(), "ScanSlowAxis", "Null (1D scan)"
                )
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(), "ScanFastAxis", "1st axis"
                )
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(), "ScanSettlingTime(ms)", 3000
                )

                # --------------------------------------------------------#
                # Set scan axis start/end positions
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "ScanFastAxisStartPosition(mm)",
                    np.round(data_dict["ASI"]["scan_axis_start_mm"], 2),
                )
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "ScanFastAxisStopPosition(mm)",
                    np.round(data_dict["ASI"]["scan_axis_end_mm"], 2),
                )

                # --------------------------------------------------------#
                # Set the scan state
                self._mmc.setProperty(self._mmc.getXYStageDevice(), "ScanState", "Idle")

                if DEBUGGING:
                    actual_speed_x = float(
                        self._mmc.getProperty(
                            self._mmc.getXYStageDevice(), "MotorSpeedX-S(mm/s)"
                        )
                    )
                    scanaxis_start = self._mmc.getProperty(
                        self._mmc.getXYStageDevice(), "ScanFastAxisStartPosition(mm)"
                    )
                    scanaxis_stop = self._mmc.getProperty(
                        self._mmc.getXYStageDevice(), "ScanFastAxisStopPosition(mm)"
                    )
                    scan_settling_ms = self._mmc.getProperty(
                        self._mmc.getXYStageDevice(), "ScanSettlingTime(ms)"
                    )
                    scanaxis_speed = np.round(
                        data_dict["ASI"]["scan_axis_speed_mm_s"], 4
                    )
                    validate_speed = actual_speed_x == scanaxis_speed
                    debug(
                        "SCAN POSITIONS",
                        f"start: {scanaxis_start}",
                        f"end: {scanaxis_stop}",
                        f"scan settling time: {scan_settling_ms}",
                        f"actual speed: {actual_speed_x}",
                        f"requested speed: {scanaxis_speed}",
                        f"stage speeds match: {validate_speed}",
                    )

                # The ASI action owns only PLC/stage hardware.  Camera trigger
                # configuration belongs to the stage-mode DAQ/camera setup.
                self.start_asi_scan_after_camera_sequence = True

            elif action_name == ACTION_AO_OPTIMIZE:
                # --------------------------------------------------------#
                # apply optimized mirror position
                if data_dict["AO"]["apply_existing"]:
                    pass

                # --------------------------------------------------------#
                # Set hardware state to run adaptive optics
                else:
                    # Clear DAQ tasks to re-program
                    self.opmDAQ.clear_tasks()
                    # Setup camera properties
                    self.configure_camera(data_dict)
                    # Set laser powers
                    self.configure_lasers(data_dict, setting="AO")

            elif action_name == ACTION_AO_GRID:
                # --------------------------------------------------------#
                # apply optimized position
                if data_dict["AO"]["apply_ao_map"]:
                    debug("AO GRID", "Applying existing mirror position.")
                    pass

                # --------------------------------------------------------#
                # run adaptive optics over a grid of positions.
                else:
                    # Clear DAQ tasks to re-program
                    self.opmDAQ.clear_tasks()

                    # Setup camera properties
                    self.configure_camera(data_dict)

                    # Set laser powers
                    self.configure_lasers(data_dict, setting="AO_grid")

                    # Set ASI stage speed for moves
                    stage_move_speed = self._config["OPM"]["stage_move_speed"]
                    self._mmc.setProperty(
                        self._mmc.getXYStageDevice(),
                        "MotorSpeedX-S(mm/s)",
                        stage_move_speed,
                    )
                    self._mmc.setProperty(
                        self._mmc.getXYStageDevice(),
                        "MotorSpeedY-S(mm/s)",
                        stage_move_speed,
                    )

            elif action_name == ACTION_DAQ:
                # --------------------------------------------------------#
                # Update daq waveform values and setup daq for playback
                self.opmDAQ.stop_waveform_playback()
                self.opmDAQ.clear_tasks()

                exposure_ms = np.round(
                    float(self.configure_lasers(data_dict, setting="DAQ")), 2
                )

                if str(data_dict["DAQ"]["mode"]) == "stage":
                    self.opmDAQ.set_acquisition_params(
                        scan_type="stage",
                        channel_states=data_dict["DAQ"]["channel_states"],
                        laser_blanking=bool(data_dict["DAQ"]["blanking"]),
                        exposure_ms=exposure_ms,
                    )
                    if not self.simulate_hardware:
                        self.configure_stage_camera_trigger()
                elif str(data_dict["DAQ"]["mode"]) == "projection":
                    self.opmDAQ.set_acquisition_params(
                        scan_type="projection",
                        channel_states=data_dict["DAQ"]["channel_states"],
                        image_mirror_range_um=float(
                            data_dict["DAQ"]["image_mirror_range_um"]
                        ),
                        laser_blanking=bool(data_dict["DAQ"]["blanking"]),
                        exposure_ms=exposure_ms,
                    )
                    # # Setup camera in progressive scan mode
                    # self._mmc.setProperty(
                    #     str(self._config["Camera"]["camera_id"]),
                    #     "SENSOR MODE",
                    #     "PROGRESSIVE",
                    # )
                elif str(data_dict["DAQ"]["mode"]) == "mirror":
                    self.opmDAQ.set_acquisition_params(
                        scan_type="mirror",
                        channel_states=data_dict["DAQ"]["channel_states"],
                        image_mirror_step_um=float(
                            data_dict["DAQ"]["image_mirror_step_um"]
                        ),
                        image_mirror_range_um=float(
                            data_dict["DAQ"]["image_mirror_range_um"]
                        ),
                        laser_blanking=bool(data_dict["DAQ"]["blanking"]),
                        exposure_ms=exposure_ms,
                    )
                elif str(data_dict["DAQ"]["mode"]) == "2d":
                    self.opmDAQ.set_acquisition_params(
                        scan_type="2d",
                        channel_states=data_dict["DAQ"]["channel_states"],
                        laser_blanking=bool(data_dict["DAQ"]["blanking"]),
                        exposure_ms=exposure_ms,
                    )
                self.opmDAQ.generate_waveforms()
                self.opmDAQ.program_daq_waveforms()

                # --------------------------------------------------------#
                # Setup camera properties
                self.configure_camera(data_dict, setting="DAQ")

                self._mmc.setProperty(
                    str(self._config["Camera"]["camera_id"]), "Exposure", exposure_ms
                )
                self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))

                # Wait for MM core
                self._mmc.waitForSystem()

                debug(
                    "CAMERA EXPOSURES",
                    f"actual: {np.round(self._mmc.getExposure(), 2)}",
                    f"requested: {exposure_ms}",
                )

            elif action_name == ACTION_MIRROR_MOVE:
                # --------------------------------------------------------#
                # Update daq waveform values and setup daq for playback
                self.opmDAQ.stop_waveform_playback()
                self.opmDAQ.clear_tasks()

                # Modify the image neutral position
                self.opmDAQ._ao_neutral_positions[0] = data_dict["DAQ"][
                    "image_mirror_v"
                ]

                self.opmDAQ.set_acquisition_params(scan_type="2d")
                debug(
                    "IMAGE MIRROR MOVE",
                    f"image mirror voltage: {data_dict['DAQ']['image_mirror_v']}",
                )
                self.opmDAQ.generate_waveforms()
                self.opmDAQ.program_daq_waveforms()
        else:
            super().setup_event(event)

    def post_sequence_started(self, event):
        """Start configured ASI hardware after the camera sequence is ready.

        Parameters
        ----------
        event : MDAEvent
            First event in the started camera sequence.
        """
        if self.start_asi_scan_after_camera_sequence:
            if self.simulate_hardware:
                self.simulated_asi_state["scan_state"] = "Running"
                self.simulated_asi_transitions.append("Running")
            else:
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(), "ScanState", "Running"
                )
            self.start_asi_scan_after_camera_sequence = False

    def exec_event(
        self, event: MDAEvent
    ) -> Iterable[tuple[NDArray, MDAEvent, FrameMetaV1]]:
        """Execute `event`.

        This method is called after `setup_event` and is responsible for
        executing the event. The default assumption is to acquire an image,
        but more elaborate events will be possible.

        Parameters
        ----------
        event : MDAEvent
            Event to execute.

        Returns
        -------
        Iterable[tuple[numpy.ndarray, MDAEvent, FrameMetaV1]] or None
            Camera frames from the base engine for image events; custom actions
            do not produce frames.
        """
        if isinstance(event.action, CustomAction):
            action_name = event.action.name
            data_dict = event.action.data

            if action_name == ACTION_O2O3_AUTOFOCUS:
                manage_O3_focus(
                    self._config["O2O3-autofocus"]["O3_stage_name"], verbose=DEBUGGING
                )

            elif action_name == ACTION_AO_OPTIMIZE:
                pos_idx = data_dict["AO"]["pos_idx"]
                if data_dict["AO"]["apply_existing"]:
                    self.AOMirror.apply_positions_array(int(pos_idx))
                    debug(
                        "AO MIRROR EXISTING POSITIONS",
                        f"pos: {int(pos_idx)}",
                        f"modal coefficients: {self.AOMirror.current_coeffs.copy()}",
                    )
                else:
                    run_ao_optimization(
                        exposure_ms=float(data_dict["Camera"]["exposure_ms"]),
                        channel_states=data_dict["AO"]["channel_states"],
                        metric_to_use=data_dict["AO"]["metric"],
                        daq_mode=data_dict["AO"]["daq_mode"],
                        image_mirror_range_um=float(
                            data_dict["AO"]["image_mirror_range_um"]
                        ),
                        num_iterations=int(data_dict["AO"]["iterations"]),
                        num_mode_samples=int(data_dict["AO"]["num_mode_samples"]),
                        starting_coef_delta=float(data_dict["AO"]["modal_delta"]),
                        coef_delta_scale=float(data_dict["AO"]["modal_alpha"]),
                        metric_precision=int(data_dict["AO"]["metric_precision"]),
                        modes_to_optimize=data_dict["AO"]["modes_to_optimize"],
                        starting_mirror_state=str(data_dict["AO"]["mirror_state"]),
                        mode_acceptance=data_dict["AO"]["metric_acceptance"],
                        num_averaged_frames=int(data_dict["AO"]["num_averaged_frames"]),
                        pos_idx=pos_idx,
                        save_dir_path=data_dict["AO"]["output_path"],
                        verbose=DEBUGGING,
                    )
                    if pos_idx is not None:
                        try:
                            self.AOMirror.update_positions_array(int(pos_idx))
                            debug(
                                "AO POSITIONS ARRAY SAVED",
                                f"pos: {int(pos_idx)}",
                                f"modal coefficients: {self.AOMirror.current_coeffs.copy()}",
                            )
                        except Exception as e:
                            warning(
                                "AO POSITIONS ARRAY",
                                "Not setting AO positions array.",
                                f"Exception: {e}",
                            )

            elif action_name == ACTION_AO_GRID:
                pos_idx = data_dict["AO"]["pos_idx"]
                if data_dict["AO"]["apply_ao_map"]:
                    self.AOMirror.apply_positions_array(int(pos_idx))
                    debug(
                        "AO GRID EXISTING POSITIONS",
                        f"pos: {int(pos_idx)}",
                        f"positions: {self.AOMirror.current_coeffs.copy()}",
                    )
                else:
                    run_ao_grid_mapping(
                        stage_positions=data_dict["AO"]["stage_positions"],
                        ao_dict=data_dict["AO"]["ao_dict"],
                        num_tile_positions=data_dict["AO"]["num_tile_positions"],
                        num_scan_positions=data_dict["AO"]["num_scan_positions"],
                        save_dir_path=data_dict["AO"]["output_path"],
                        verbose=DEBUGGING,
                    )

            elif action_name == ACTION_DAQ:
                self.opmDAQ.start_waveform_playback()

            elif action_name == ACTION_FLUIDICS:
                info(
                    "FLUIDICS",
                    "Sending TTL pulse to OB1 to CLEAVE and apply READOUTS.",
                )
                run_fluidic_program(True)

            elif action_name == ACTION_TIMELAPSE:
                interval = data_dict["plan"]["interval"]
                self.elapsed_time = perf_counter() - self.start_time
                sleep_time = interval - self.elapsed_time
                if sleep_time < 0:
                    sleep_time = 1
                    debug(
                        "TIMELAPSE INTERVAL",
                        "Imaging did not finish before interval time.",
                        "Running now.",
                    )

                QThread.sleep(int(sleep_time))
                self.start_time = perf_counter()

                debug(
                    "TIMELAPSE",
                    f"elapsed: {self.elapsed_time}",
                    f"start time: {self.start_time}",
                    f"requested interval: {interval}",
                    f"sleep time: {sleep_time}",
                )

            elif action_name == ACTION_AO_MIRROR_UPDATE:
                coeffs = data_dict["AOmirror"]["modal_coeffs"]
                if coeffs is not None:
                    self.AOMirror.set_modal_coefficients(np.array(coeffs))
                    debug(
                        "AO MIRROR MODAL COEFFICIENTS",
                        f"modal coefficients: {self.AOMirror.current_coeffs.copy()}",
                    )
                elif data_dict["AOmirror"].get("positions") is not None:
                    self.AOMirror.set_mirror_voltage(
                        np.array(data_dict["AOmirror"]["positions"])
                    )
                else:
                    warning(
                        "AO MIRROR UPDATE",
                        "No coefficients or positions sent.",
                    )
        else:
            result = super().exec_event(event)
            return result

    def teardown_event(self, event):
        """Release per-event state after execution.

        Parameters
        ----------
        event : MDAEvent
            Event that has completed.
        """
        if isinstance(event.action, CustomAction):
            self._mmc.clearCircularBuffer()
        super().teardown_event(event)

    def teardown_sequence(self, sequence: MDASequence) -> None:
        """Restore hardware state after an acquisition sequence.

        Parameters
        ----------
        sequence : MDASequence
            Sequence that has completed or been canceled.
        """
        debug("TEARDOWN", "Acquisition finished, tearing down.")

        # Shut down DAQ
        self.opmDAQ.clear_tasks()
        self.opmDAQ.reset()
        self.opmDAQ._ao_neutral_positions[0] = self._config["NIDAQ"][
            "image_mirror_neutral_v"
        ]
        if self.simulate_hardware and self.simulated_asi_state:
            self.simulated_asi_state["scan_state"] = "Idle"
            self.simulated_asi_transitions.append("Idle")

        # Put cameras that expose the Hamamatsu trigger properties back in internal mode.
        camera = str(self._config["Camera"]["camera_id"])
        if self._mmc.hasProperty(camera, "TriggerPolarity"):
            self._mmc.setProperty(camera, "TriggerPolarity", "POSITIVE")
            self._mmc.waitForDevice(camera)
        if self._mmc.hasProperty(camera, "TRIGGER SOURCE"):
            self._mmc.setProperty(camera, "TRIGGER SOURCE", "INTERNAL")
            self._mmc.waitForDevice(camera)

        stage_move_speed = self._config["OPM"]["stage_move_speed"]
        xy_stage = self._mmc.getXYStageDevice()
        if self._mmc.hasProperty(xy_stage, "MotorSpeedX-S(mm/s)"):
            self._mmc.setProperty(xy_stage, "MotorSpeedX-S(mm/s)", stage_move_speed)
        if self._mmc.hasProperty(xy_stage, "MotorSpeedY-S(mm/s)"):
            self._mmc.setProperty(xy_stage, "MotorSpeedY-S(mm/s)", stage_move_speed)

        # Set all lasers to zero emission
        for laser in self._config["Lasers"]["laser_names"]:
            if self.simulate_hardware:
                self.simulated_laser_powers[laser] = 0.0
            else:
                self._mmc.setProperty(
                    self._config["Lasers"]["name"], laser + " - PowerSetpoint (%)", 0.0
                )

        # save mirror positions array
        if self.AOMirror.output_path:
            self.AOMirror.save_positions_array()
        self._mmc.clearCircularBuffer()
        # TODO
        # self._mmc.setCircularBufferMemoryFootprint(16000)

        super().teardown_sequence(sequence)
