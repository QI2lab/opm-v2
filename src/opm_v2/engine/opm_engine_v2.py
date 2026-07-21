"""OPM pymmcore-plus MDA Engine v2.

This version keeps the original hardware behavior but shares custom-action
constants with ``opm_custom_events_v2`` so event factories and engine dispatch
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
from opm_v2.engine.opm_custom_events_v2 import (
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
from opm_v2.engine.debug_printing_v2 import (
    debug as _debug,
    info,
    warning,
)
from opm_v2.utils.autofocus_remote_unit import manage_O3_focus
from opm_v2.utils.elveflow_control import run_fluidic_program
from opm_v2.utils.sensorless_ao_v2 import run_ao_grid_mapping, run_ao_optimization

logging.getLogger("pymmcore-plus")

DEBUGGING = True
POWER_STR = " - PowerSetpoint (%)"


def debug(header: str, *lines: object) -> None:
    """Print a debug block when module-level debugging is enabled."""
    _debug(header, *lines, enabled=DEBUGGING)


class OPMEngineV2(MDAEngine):
    def __init__(
        self,
        mmc,
        config_path: Path,
        use_hardware_sequencing: bool = True
    ) -> None:

        super().__init__(mmc, use_hardware_sequencing=use_hardware_sequencing, restore_initial_state=False)
        self._mmc = mmc
        self.opmDAQ = OPMNIDAQ.instance()
        self.AOMirror = AOMirror.instance()
        self.execute_stage_scan = False
        self.start_time = None
        self.elapsed_time = None
        self._config_path = config_path
        self._config = None
        self.update_config()
        self._setup_handlers = {
            ACTION_O2O3_AUTOFOCUS: self._setup_o2o3_autofocus,
            ACTION_STAGE_MOVE: self._setup_stage_move,
            ACTION_ASI_SETUP_SCAN: self._setup_asi_scan,
            ACTION_AO_OPTIMIZE: self._setup_ao_optimize,
            ACTION_AO_GRID: self._setup_ao_grid,
            ACTION_DAQ: self._setup_daq,
            ACTION_MIRROR_MOVE: self._setup_mirror_move,
        }
        self._exec_handlers = {
            ACTION_O2O3_AUTOFOCUS: self._exec_o2o3_autofocus,
            ACTION_AO_OPTIMIZE: self._exec_ao_optimize,
            ACTION_AO_GRID: self._exec_ao_grid,
            ACTION_DAQ: self._exec_daq,
            ACTION_FLUIDICS: self._exec_fluidics,
            ACTION_TIMELAPSE: self._exec_timelapse,
            ACTION_AO_MIRROR_UPDATE: self._exec_ao_mirror_update,
        }
        
    def update_config(self):
        """Update the class config dict. from file
        """
        with open(self._config_path, "r") as config_file:
            self._config = json.load(config_file)

    def _camera_id(self) -> str:
        return str(self._config["Camera"]["camera_id"])

    def _xy_stage_device(self) -> str:
        return self._mmc.getXYStageDevice()

    def _requested_roi(self, data_dict: dict) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in data_dict["Camera"]["camera_crop"])

    def _active_channel_indices(self, channel_states: list) -> list[int]:
        return [idx for idx, state in enumerate(channel_states) if state]

    def _validate_channel_values(
        self,
        setting: str,
        channel_states: list,
        channel_powers: list,
        exposure_channels: list | None = None,
    ) -> None:
        laser_count = len(self._config["Lasers"]["laser_names"])
        if len(channel_states) != laser_count:
            raise ValueError(
                f"{setting} channel_states has {len(channel_states)} values; "
                f"expected {laser_count}."
            )
        if len(channel_powers) != laser_count:
            raise ValueError(
                f"{setting} channel_powers has {len(channel_powers)} values; "
                f"expected {laser_count}."
            )
        if exposure_channels is not None and len(exposure_channels) != laser_count:
            raise ValueError(
                f"Camera exposure_channels has {len(exposure_channels)} values; "
                f"expected {laser_count}."
            )

    def _set_property(self, device: str, prop: str, value) -> None:
        self._mmc.setProperty(device, prop, value)
        self._mmc.waitForDevice(str(device))

    def _set_property_until(
        self,
        device: str,
        prop: str,
        value,
        timeout_s: float = 5.0,
        poll_s: float = 0.1,
    ) -> None:
        """Set a hardware property and wait until Micro-Manager reports it."""
        deadline = perf_counter() + timeout_s
        last_value = None
        while perf_counter() < deadline:
            self._set_property(device, prop, value)
            last_value = self._mmc.getProperty(device, prop)
            if str(last_value) == str(value):
                return
            sleep(poll_s)
        raise TimeoutError(
            f"Timed out setting {device}.{prop} to {value!r}; last value was {last_value!r}."
        )

    def _set_stage_move_speed(self, speed: float) -> None:
        stage = self._xy_stage_device()
        self._set_property(stage, "MotorSpeedX-S(mm/s)", speed)
        self._set_property(stage, "MotorSpeedY-S(mm/s)", speed)

    def _log_custom_action(self, phase: str, action_name: str, *lines: object) -> None:
        debug(f"{phase}: {action_name}", *lines)

    def _warn_unknown_custom_action(self, phase: str, action_name: str) -> None:
        warning(
            "UNKNOWN CUSTOM ACTION",
            f"phase: {phase}",
            f"action: {action_name}",
        )
    
    def configure_camera(self, data_dict: dict, setting: str = None):
        """Set the camera ROI and exposure

        Parameters
        ----------
        data_dict : dict
            Custom action data dict
        """
        requested_roi = self._requested_roi(data_dict)
        current_roi = tuple(int(value) for value in self._mmc.getROI())
        if requested_roi != current_roi:
            self._mmc.clearROI()
            self._mmc.waitForDevice(self._camera_id())
            self._mmc.setROI(*requested_roi)
            self._mmc.waitForDevice(self._camera_id())
            debug(
                "CAMERA ROI UPDATED",
                f"old: {current_roi}",
                f"new: {requested_roi}",
            )

        if setting=="DAQ":
            exposure_ms = np.max(data_dict["Camera"]["exposure_channels"])
        else:
            exposure_ms = data_dict["Camera"]["exposure_ms"]
        self._set_property(
            self._camera_id(),
            "Exposure",
            np.round(float(exposure_ms), 2),
        )
        debug(
            "CAMERA CONFIGURED",
            f"roi: {tuple(int(value) for value in self._mmc.getROI())}",
            f"exposure_ms: {np.round(float(exposure_ms), 2)}",
        )
        
    def configure_lasers(self, data_dict: dict, setting: str):
        """Set laser powers

        Parameters
        ----------
        data_dict : dict
            Custom action data dict
        """
        if setting not in {"AO", "DAQ", "AO_grid"}:
            raise ValueError(f"Engine laser configuration missing setting: {setting}")

        if setting == "AO_grid":
            channel_states = data_dict["AO"]["ao_dict"]["channel_states"]
            channel_powers = data_dict["AO"]["ao_dict"]["channel_powers"]
        else:
            channel_states = data_dict[setting]["channel_states"]
            channel_powers = data_dict[setting]["channel_powers"]
        exposure_channels = (
            data_dict["Camera"]["exposure_channels"]
            if setting == "DAQ"
            else None
        )
        self._validate_channel_values(
            setting,
            channel_states,
            channel_powers,
            exposure_channels=exposure_channels,
        )

        active_indices = self._active_channel_indices(channel_states)
        if not active_indices:
            raise ValueError(f"No active laser channels for {setting} configuration.")

        exposure_ms = None
        if setting == "DAQ":
            exposure_ms = max(
                float(exposure_channels[idx])
                for idx in active_indices
            )

        for chan_idx, chan_bool in enumerate(channel_states):
            laser_name = str(self._config["Lasers"]["laser_names"][chan_idx])   
            if chan_bool:
                laser_power = float(channel_powers[chan_idx])
                self._mmc.setProperty(
                    self._config["Lasers"]["name"],
                    laser_name + POWER_STR,
                    laser_power
                )
            else:
                self._mmc.setProperty(
                    self._config["Lasers"]["name"],
                    laser_name + POWER_STR,
                    0.0
                )

        debug(
            "LASERS CONFIGURED",
            f"setting: {setting}",
            f"active channel indices: {active_indices}",
            f"powers: {[channel_powers[idx] for idx in active_indices]}",
            f"exposures_ms: {[exposure_channels[idx] for idx in active_indices] if exposure_channels else None}",
            f"selected_exposure_ms: {exposure_ms}",
        )

        if setting == "DAQ":
            return exposure_ms
    
    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Setup state of system (hardware, etc.) before an MDA is run.

        This method is called once at the beginning of a sequence.
        (The sequence object needn't be used here if not necessary)
        """
        self.start_time = perf_counter()
        self.elapsed_time = 0
        # TODO
        self._mmc.setCircularBufferMemoryFootprint(32000)
        # self._mmc.setCircularBufferMemoryFootprint(16000)
        super().setup_sequence(sequence)

    def setup_event(self, event: MDAEvent) -> None:
        """Prepare state of system (hardware, etc.) for `event`.

        This method is called before each event in the sequence. It is
        responsible for preparing the state of the system for the event.
        The engine should be in a state where it can call `exec_event`
        without any additional preparation.
        """
        if isinstance(event.action, CustomAction):
            action_name = event.action.name
            handler = self._setup_handlers.get(action_name)
            if handler is None:
                self._warn_unknown_custom_action("setup", action_name)
                return
            self._log_custom_action("SETUP", action_name)
            handler(event.action.data)
        else:
            super().setup_event(event)

    def _setup_o2o3_autofocus(self, data_dict: dict) -> None:
        if self.opmDAQ.running():
            self.opmDAQ.stop_waveform_playback()
        self.opmDAQ.reset_ao_channels()
        self.configure_camera(data_dict)

    def _setup_stage_move(self, data_dict: dict) -> None:
        self.update_config()
        stage_move_speed = self._config["OPM"]["stage_move_speed"]
        self._set_stage_move_speed(stage_move_speed)

        target_z = np.round(float(data_dict["Stage"]["z_pos"]), 2)
        target_x = np.round(float(data_dict["Stage"]["x_pos"]), 2)
        target_y = np.round(float(data_dict["Stage"]["y_pos"]), 2)
        self._mmc.setPosition(target_z)
        self._mmc.waitForDevice(self._mmc.getFocusDevice())

        current_x, current_y = self._mmc.getXYPosition()
        old_x, old_y = current_x, current_y
        self._mmc.setXYPosition(target_x, target_y)
        stationary_count = 0
        while (
            not np.isclose(current_x, target_x, rtol=0.0, atol=1.0)
            or not np.isclose(current_y, target_y, rtol=0.0, atol=1.0)
        ):
            sleep(0.5)
            current_x, current_y = self._mmc.getXYPosition()
            if old_x == current_x and old_y == current_y:
                stationary_count += 1
                debug(
                    "STAGE MOVE STATIONARY",
                    f"current_x: {current_x}",
                    f"current_y: {current_y}",
                    f"target_x: {target_x}",
                    f"target_y: {target_y}",
                    f"stationary count: {stationary_count}",
                )
            else:
                old_x, old_y = current_x, current_y
            if stationary_count >= 5:
                warning(
                    "STAGE MOVE",
                    "Stage stopped changing before reaching target tolerance.",
                    f"current_x: {current_x}",
                    f"current_y: {current_y}",
                    f"target_x: {target_x}",
                    f"target_y: {target_y}",
                )
                break

    def _setup_asi_scan(self, data_dict: dict) -> None:
        plc_name = self._config["PLC"]["name"]
        stage = self._xy_stage_device()
        camera = self._camera_id()
        self._set_property(plc_name, self._config["PLC"]["position"], int(self._config["PLC"]["pin"]))
        self._set_property(
            plc_name,
            self._config["PLC"]["cellconfig"],
            int(self._config["PLC"]["signalid"]),
        )

        scan_speed = np.round(data_dict["ASI"]["scan_axis_speed_mm_s"], 4)
        self._set_property(stage, "MotorSpeedX-S(mm/s)", scan_speed)
        self._set_property(stage, "MotorSpeedY-S(mm/s)", self._config["OPM"]["stage_move_speed"])
        self._set_property(stage, "ScanPattern", "Raster")
        self._set_property(stage, "ScanSlowAxis", "Null (1D scan)")
        self._set_property(stage, "ScanFastAxis", "1st axis")
        self._set_property(stage, "ScanSettlingTime(ms)", 3000)
        self._set_property(
            stage,
            "ScanFastAxisStartPosition(mm)",
            np.round(data_dict["ASI"]["scan_axis_start_mm"], 2),
        )
        self._set_property(
            stage,
            "ScanFastAxisStopPosition(mm)",
            np.round(data_dict["ASI"]["scan_axis_end_mm"], 2),
        )
        self._set_property(stage, "ScanState", "Idle")

        if DEBUGGING:
            actual_speed_x = float(self._mmc.getProperty(stage, "MotorSpeedX-S(mm/s)"))
            scanaxis_start = self._mmc.getProperty(stage, "ScanFastAxisStartPosition(mm)")
            scanaxis_stop = self._mmc.getProperty(stage, "ScanFastAxisStopPosition(mm)")
            scan_settling_ms = self._mmc.getProperty(stage, "ScanSettlingTime(ms)")
            debug(
                "SCAN POSITIONS",
                f"start: {scanaxis_start}",
                f"end: {scanaxis_stop}",
                f"scan settling time: {scan_settling_ms}",
                f"actual speed: {actual_speed_x}",
                f"requested speed: {scan_speed}",
                f"stage speeds match: {actual_speed_x == scan_speed}",
            )

        self._set_property_until(camera, "Trigger", "START")
        self._set_property_until(camera, "TriggerPolarity", "POSITIVE")
        self._set_property_until(camera, "TRIGGER SOURCE", "EXTERNAL")
        self.execute_stage_scan = True

    def _setup_ao_optimize(self, data_dict: dict) -> None:
        if data_dict["AO"]["apply_existing"]:
            debug("AO OPTIMIZE", "Using existing mirror position.")
            return
        self.opmDAQ.clear_tasks()
        self.configure_camera(data_dict)
        self.configure_lasers(data_dict, setting="AO")

    def _setup_ao_grid(self, data_dict: dict) -> None:
        if data_dict["AO"]["apply_ao_map"]:
            debug("AO GRID", "Applying existing mirror position.")
            return
        self.opmDAQ.clear_tasks()
        self.configure_camera(data_dict)
        self.configure_lasers(data_dict, setting="AO_grid")
        self._set_stage_move_speed(self._config["OPM"]["stage_move_speed"])

    def _setup_daq(self, data_dict: dict) -> None:
        self.opmDAQ.stop_waveform_playback()
        self.opmDAQ.clear_tasks()
        exposure_ms = np.round(float(self.configure_lasers(data_dict, setting="DAQ")), 2)

        daq_mode = str(data_dict["DAQ"]["mode"])
        params = {
            "scan_type": daq_mode,
            "channel_states": data_dict["DAQ"]["channel_states"],
            "laser_blanking": bool(data_dict["DAQ"]["blanking"]),
            "exposure_ms": exposure_ms,
        }
        if daq_mode in {"projection", "mirror"}:
            params["image_mirror_range_um"] = float(data_dict["DAQ"]["image_mirror_range_um"])
        if daq_mode == "mirror":
            params["image_mirror_step_um"] = float(data_dict["DAQ"]["image_mirror_step_um"])
        if daq_mode not in {"stage", "projection", "mirror", "2d"}:
            raise ValueError(f"Unsupported DAQ mode: {daq_mode}")

        debug(
            "DAQ CONFIG",
            f"mode: {daq_mode}",
            f"exposure_ms: {exposure_ms}",
            f"active channels: {self._active_channel_indices(data_dict['DAQ']['channel_states'])}",
            f"laser blanking: {bool(data_dict['DAQ']['blanking'])}",
        )
        self.opmDAQ.set_acquisition_params(**params)
        self.opmDAQ.generate_waveforms()
        self.opmDAQ.program_daq_waveforms()
        self.configure_camera(data_dict, setting="DAQ")
        self._set_property(self._camera_id(), "Exposure", exposure_ms)
        self._mmc.waitForSystem()
        debug(
            "CAMERA EXPOSURES",
            f"actual: {np.round(self._mmc.getExposure(), 2)}",
            f"requested: {exposure_ms}",
        )

    def _setup_mirror_move(self, data_dict: dict) -> None:
        self.opmDAQ.stop_waveform_playback()
        self.opmDAQ.clear_tasks()
        image_mirror_v = data_dict["DAQ"]["image_mirror_v"]
        self.opmDAQ._ao_neutral_positions[0] = image_mirror_v
        self.opmDAQ.set_acquisition_params(scan_type="2d")
        debug("IMAGE MIRROR MOVE", f"image mirror voltage: {image_mirror_v}")
        self.opmDAQ.generate_waveforms()
        self.opmDAQ.program_daq_waveforms()
            
    def post_sequence_started(self, event):
        # TODO: catch sequence timpoints
        # execute stage scan if requested
        if self.execute_stage_scan:
            self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "ScanState",
                    "Running"
                )
            self.execute_stage_scan = False
            
    def exec_event(self, event: MDAEvent) -> Iterable[tuple[NDArray, MDAEvent, FrameMetaV1]]:
        """Execute `event`.

        This method is called after `setup_event` and is responsible for
        executing the event. The default assumption is to acquire an image,
        but more elaborate events will be possible.
        """
        if isinstance(event.action, CustomAction):
            action_name = event.action.name
            handler = self._exec_handlers.get(action_name)
            if handler is None:
                self._warn_unknown_custom_action("exec", action_name)
                return None
            self._log_custom_action("EXEC", action_name)
            handler(event.action.data)
            return None
        else:
            result = super().exec_event(event)
            return result

    def _exec_o2o3_autofocus(self, data_dict: dict) -> None:
        manage_O3_focus(
            self._config["O2O3-autofocus"]["O3_stage_name"],
            verbose=DEBUGGING,
        )

    def _exec_ao_optimize(self, data_dict: dict) -> None:
        pos_idx = data_dict["AO"]["pos_idx"]
        if data_dict["AO"]["apply_existing"]:
            self.AOMirror.apply_positions_array(int(pos_idx))
            debug(
                "AO MIRROR EXISTING POSITIONS",
                f"pos: {int(pos_idx)}",
                f"modal coefficients: {self.AOMirror.current_coeffs.copy()}",
            )
            return

        run_ao_optimization(
            exposure_ms=float(data_dict["Camera"]["exposure_ms"]),
            channel_states=data_dict["AO"]["channel_states"],
            metric_to_use=data_dict["AO"]["metric"],
            daq_mode=data_dict["AO"]["daq_mode"],
            image_mirror_range_um=float(data_dict["AO"]["image_mirror_range_um"]),
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
            camera_id=self._camera_id(),
            focus_device_name=self._config["O2O3-autofocus"]["O3_stage_name"],
            z_project_metric=data_dict["AO"].get("z_project_metric", "DCT"),
            z_project_range_um=float(data_dict["AO"].get("z_project_range_um", 4.0)),
            z_project_step_um=float(data_dict["AO"].get("z_project_step_um", 0.115)),
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

    def _exec_ao_grid(self, data_dict: dict) -> None:
        pos_idx = data_dict["AO"]["pos_idx"]
        if data_dict["AO"]["apply_ao_map"]:
            self.AOMirror.apply_positions_array(int(pos_idx))
            debug(
                "AO GRID EXISTING POSITIONS",
                f"pos: {int(pos_idx)}",
                f"positions: {self.AOMirror.current_coeffs.copy()}",
            )
            return

        run_ao_grid_mapping(
            stage_positions=data_dict["AO"]["stage_positions"],
            ao_dict=data_dict["AO"]["ao_dict"],
            num_tile_positions=data_dict["AO"]["num_tile_positions"],
            num_scan_positions=data_dict["AO"]["num_scan_positions"],
            save_dir_path=data_dict["AO"]["output_path"],
            verbose=DEBUGGING,
            camera_id=self._camera_id(),
            focus_device_name=self._config["O2O3-autofocus"]["O3_stage_name"],
        )

    def _exec_daq(self, data_dict: dict) -> None:
        self.opmDAQ.start_waveform_playback()

    def _exec_fluidics(self, data_dict: dict) -> None:
        info(
            "FLUIDICS",
            "Sending TTL pulse to OB1 to CLEAVE and apply READOUTS.",
        )
        run_fluidic_program(True)

    def _exec_timelapse(self, data_dict: dict) -> None:
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

    def _exec_ao_mirror_update(self, data_dict: dict) -> None:
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
        
    def teardown_event(self, event):
        if isinstance(event.action, CustomAction):
            self._mmc.clearCircularBuffer()
        super().teardown_event(event)
        
    def teardown_sequence(self, sequence: MDASequence) -> None:
        debug("TEARDOWN", "Acquisition finished, tearing down.")
        
        # Shut down DAQ
        self.opmDAQ.clear_tasks()
        self.opmDAQ.reset()
        self.opmDAQ._ao_neutral_positions[0] = self._config["NIDAQ"]["image_mirror_neutral_v"]

        # Put camera back into internal mode
        self._mmc.setProperty(self._config["Camera"]["camera_id"],"TriggerPolarity","POSITIVE")
        self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
        self._mmc.setProperty(self._config["Camera"]["camera_id"],"TRIGGER SOURCE","INTERNAL")
        self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))

        stage_move_speed = self._config['OPM']['stage_move_speed']
        self._mmc.setProperty(self._mmc.getXYStageDevice(),"MotorSpeedX-S(mm/s)",stage_move_speed)
        self._mmc.setProperty(self._mmc.getXYStageDevice(),"MotorSpeedY-S(mm/s)",stage_move_speed)
                
        # Set all lasers to zero emission
        for laser in self._config["Lasers"]["laser_names"]:
            self._mmc.setProperty(
                self._config["Lasers"]["name"],
                laser + " - PowerSetpoint (%)",
                0.0
            )
        
        # save mirror positions array
        if self.AOMirror.output_path:
            self.AOMirror.save_positions_array()
        self._mmc.clearCircularBuffer()
        # TODO
        # self._mmc.setCircularBufferMemoryFootprint(16000)
        
        super().teardown_sequence(sequence)


# Backwards-friendly alias for callers that import OPMEngine from this v2 module.
OPMEngine = OPMEngineV2
