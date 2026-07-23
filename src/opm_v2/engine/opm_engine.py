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
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from threading import Event as ThreadEvent
from time import monotonic, sleep

import numpy as np
from numpy.typing import NDArray
from pymmcore_plus.mda import MDAEngine, SkipEvent
from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1
from useq import CustomAction, MDAEvent, MDASequence

from opm_v2.engine.debug_printing import (
    debug as _debug,
)
from opm_v2.engine.debug_printing import (
    info,
    warning,
)
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
    STAGE_MOVE_SPEED_METADATA_KEY,
)
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.utils.autofocus_remote_unit import manage_O3_focus
from opm_v2.utils.elveflow_control import run_fluidic_program
from opm_v2.utils.sensorless_ao import run_ao_grid_mapping, run_ao_optimization

logging.getLogger("pymmcore-plus")

DEBUGGING = True
POWER_STR = " - PowerSetpoint (%)"
LARGE_STAGE_MOVE_TIMEOUT_MS = 120_000
STAGE_MOVE_TIMEOUT_MARGIN_S = 2.0


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
    post_teardown : callable or None
        Optional callback used to prepare idle hardware after base teardown.
    """

    def __init__(
        self,
        mmc,
        config_path: Path,
        use_hardware_sequencing: bool = True,
        simulate_hardware: bool | None = None,
        config: dict | None = None,
        post_teardown: Callable[[], None] | None = None,
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
        config : dict or None
            In-memory OPM configuration snapshot. The JSON file is read only
            when no snapshot is supplied.
        post_teardown : callable or None
            Optional callback invoked on the MDA worker after the camera and
            Micro-Manager state have been restored.
        """
        super().__init__(
            mmc,
            use_hardware_sequencing=use_hardware_sequencing,
        )
        self.opmDAQ = OPMNIDAQ.instance()
        self.AOMirror = AOMirror.instance()
        self.start_asi_scan_after_camera_sequence = False
        self._config_path = config_path
        self._config = {}
        self._post_teardown = post_teardown
        if config is None:
            self.update_config()
        else:
            self.set_config(config)
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
        self.simulated_custom_actions: list[str] = []
        self.simulated_stage_move_speeds: list[dict[str, float]] = []
        self._stage_speeds_before_sequence: dict[str, str] = {}
        self._stage_explorer_timeout_before_sequence: int | None = None
        self._stage_move_count = 0
        self._safe_stop_requested = ThreadEvent()

    def update_config(self):
        """Load configuration from disk for standalone engine construction."""
        with open(self._config_path) as config_file:
            self._config = json.load(config_file)

    def set_config(self, config: dict) -> None:
        """Set the isolated configuration used by the next acquisition.

        Parameters
        ----------
        config : dict
            Complete OPM configuration snapshot from the GUI controller.
        """
        self._config = deepcopy(config)

    def request_safe_stop(self) -> bool:
        """Request cancellation at the next software-controlled event boundary.

        Returns
        -------
        bool
            ``True`` only for the first outstanding stop request.
        """
        was_pending = self._safe_stop_requested.is_set()
        self._safe_stop_requested.set()
        return not was_pending

    def clear_safe_stop(self) -> None:
        """Clear any outstanding cooperative stop request."""
        self._safe_stop_requested.clear()

    def configure_camera(self, data_dict: dict, setting: str = None):
        """Set the camera ROI and exposure.

        Parameters
        ----------
        data_dict : dict
            Custom action data dict
        setting : str or None
            Camera configuration context.

        Raises
        ------
        ValueError
            If a DAQ event has no enabled channel with a positive exposure or
            a non-DAQ camera exposure is not positive.
        """
        if not (int(data_dict["Camera"]["camera_crop"][3]) == self.mmcore.getROI()[-1]):
            self.mmcore.clearROI()
            self.mmcore.waitForDevice(str(self._config["Camera"]["camera_id"]))
            self.mmcore.setROI(
                data_dict["Camera"]["camera_crop"][0],
                data_dict["Camera"]["camera_crop"][1],
                data_dict["Camera"]["camera_crop"][2],
                data_dict["Camera"]["camera_crop"][3],
            )

        if setting == "DAQ":
            exposure_ms = max(self._active_daq_exposures(data_dict))
        else:
            exposure_ms = float(data_dict["Camera"]["exposure_ms"])
            if exposure_ms <= 0:
                raise ValueError(
                    f"Camera exposure must be greater than 0 ms; received {exposure_ms}"
                )
        self.mmcore.setProperty(
            str(self._config["Camera"]["camera_id"]),
            "Exposure",
            np.round(float(exposure_ms), 2),
        )
        self.mmcore.waitForDevice(str(self._config["Camera"]["camera_id"]))
        self.mmcore.getROI()

    @staticmethod
    def _active_daq_exposures(data_dict: dict) -> list[float]:
        """Return validated exposures for enabled DAQ channels.

        Parameters
        ----------
        data_dict : dict
            DAQ custom-action payload containing channel states and exposures.

        Returns
        -------
        list[float]
            Positive exposure times for enabled channels, in channel order.

        Raises
        ------
        ValueError
            If no DAQ channel is enabled, the state and exposure arrays differ
            in length, or an enabled channel has a non-positive exposure.
        """
        channel_states = data_dict["DAQ"]["channel_states"]
        channel_exposures = data_dict["Camera"]["exposure_channels"]
        try:
            active_exposures = [
                float(exposure)
                for enabled, exposure in zip(
                    channel_states,
                    channel_exposures,
                    strict=True,
                )
                if enabled
            ]
        except ValueError as exc:
            raise ValueError(
                "DAQ channel states and camera exposures must have equal lengths"
            ) from exc
        if not active_exposures:
            raise ValueError("DAQ event has no active acquisition channels")
        if any(exposure <= 0 for exposure in active_exposures):
            raise ValueError(
                "Every active DAQ channel must have an exposure greater than "
                f"0 ms; received {active_exposures}"
            )
        return active_exposures

    def configure_stage_camera_trigger(self) -> None:
        """Configure the camera hardware to accept the ASI stage-sync trigger."""
        camera = str(self._config["Camera"]["camera_id"])
        trigger_properties = (
            ("Trigger", "START"),
            ("TriggerPolarity", "POSITIVE"),
            ("TRIGGER SOURCE", "EXTERNAL"),
        )
        for property_name, value in trigger_properties:
            if not self.mmcore.hasProperty(camera, property_name):
                continue
            self.mmcore.setProperty(camera, property_name, value)
            self.mmcore.waitForDevice(camera)
            while self.mmcore.getProperty(camera, property_name) != value:
                sleep(0.1)
                self.mmcore.setProperty(camera, property_name, value)
                self.mmcore.waitForDevice(camera)

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
        if setting == "DAQ":
            self._active_daq_exposures(data_dict)
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
                    self.mmcore.setProperty(
                        self._config["Lasers"]["name"],
                        laser_name + " - PowerSetpoint (%)",
                        laser_power,
                    )
            else:
                if self.simulate_hardware:
                    self.simulated_laser_powers[laser_name] = 0.0
                else:
                    self.mmcore.setProperty(
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
        self._stage_move_count = 0
        self._capture_stage_speeds()
        metadata = getattr(sequence, "metadata", {})
        is_stage_explorer_preview = STAGE_MOVE_SPEED_METADATA_KEY in metadata
        if is_stage_explorer_preview:
            self._extend_stage_explorer_timeout()
        speed_override = metadata.get(STAGE_MOVE_SPEED_METADATA_KEY, {})
        if isinstance(speed_override, Mapping):
            self._apply_stage_move_speeds(speed_override)
        buffer_mb = (
            64
            if self.simulate_hardware
            else int(self._config["OPM"].get("circular_buffer_mb", 32000))
        )
        self.mmcore.setCircularBufferMemoryFootprint(buffer_mb)
        try:
            return super().setup_sequence(sequence)
        except Exception:
            if is_stage_explorer_preview:
                self._restore_stage_after_scan()
                self._restore_stage_explorer_timeout()
            raise

    def _extend_stage_explorer_timeout(self) -> None:
        """Keep a long core timeout active for a complete Explorer tile run."""
        if getattr(self, "_stage_explorer_timeout_before_sequence", None) is not None:
            return
        original_timeout_ms = int(self.mmcore.getTimeoutMs())
        self._stage_explorer_timeout_before_sequence = original_timeout_ms
        if original_timeout_ms < LARGE_STAGE_MOVE_TIMEOUT_MS:
            self.mmcore.setTimeoutMs(LARGE_STAGE_MOVE_TIMEOUT_MS)

    def _restore_stage_explorer_timeout(self) -> None:
        """Restore the core timeout saved before an Explorer tile run."""
        original_timeout_ms = getattr(
            self, "_stage_explorer_timeout_before_sequence", None
        )
        if original_timeout_ms is None:
            return
        self._stage_explorer_timeout_before_sequence = None
        self.mmcore.setTimeoutMs(original_timeout_ms)

    def _capture_stage_speeds(self) -> None:
        """Remember the physical XY move speeds in effect before an MDA run."""
        self._stage_speeds_before_sequence = {}
        xy_stage = self.mmcore.getXYStageDevice()
        if not xy_stage:
            return
        for prop in ("MotorSpeedX-S(mm/s)", "MotorSpeedY-S(mm/s)"):
            if self.mmcore.hasProperty(xy_stage, prop):
                self._stage_speeds_before_sequence[prop] = self.mmcore.getProperty(
                    xy_stage, prop
                )

    def _apply_stage_move_speeds(self, speeds: Mapping[str, float]) -> None:
        """Apply optional per-axis speeds after normal speeds have been captured."""
        speed_properties = {
            "move_speed_x_mm_s": ("x", "MotorSpeedX-S(mm/s)"),
            "move_speed_y_mm_s": ("y", "MotorSpeedY-S(mm/s)"),
        }
        xy_stage = self.mmcore.getXYStageDevice()
        applied: dict[str, float] = {}
        for speed_key, (axis, property_name) in speed_properties.items():
            if speed_key not in speeds:
                continue
            speed = float(speeds[speed_key])
            applied[axis] = speed
            if xy_stage and self.mmcore.hasProperty(xy_stage, property_name):
                self.mmcore.setProperty(xy_stage, property_name, speed)
        if self.simulate_hardware and applied:
            self.simulated_stage_move_speeds.append(applied)

    @staticmethod
    def _xy_move_duration_s(
        current_x_um: float,
        current_y_um: float,
        target_x_um: float,
        target_y_um: float,
        speed_x_mm_s: float,
        speed_y_mm_s: float,
    ) -> float:
        """Estimate an XY move duration from per-axis distance and speed.

        Returns
        -------
        float
            Estimated duration in seconds for the slower-moving axis.
        """
        axis_durations = []
        for distance_um, speed_mm_s in (
            (abs(target_x_um - current_x_um), speed_x_mm_s),
            (abs(target_y_um - current_y_um), speed_y_mm_s),
        ):
            if speed_mm_s <= 0:
                return float("inf")
            axis_durations.append(distance_um / (speed_mm_s * 1000.0))
        return max(axis_durations, default=0.0)

    def _stage_move_timeout_ms(
        self,
        current_x_um: float,
        current_y_um: float,
        target_x_um: float,
        target_y_um: float,
        speed_x_mm_s: float,
        speed_y_mm_s: float,
    ) -> tuple[int, float]:
        """Select a temporary timeout for initial or non-adjacent XY moves.

        Returns
        -------
        tuple[int, float]
            Selected timeout in milliseconds and estimated move duration in seconds.
        """
        original_timeout_ms = int(self.mmcore.getTimeoutMs())
        estimated_duration_s = self._xy_move_duration_s(
            current_x_um,
            current_y_um,
            target_x_um,
            target_y_um,
            speed_x_mm_s,
            speed_y_mm_s,
        )
        is_initial_move = getattr(self, "_stage_move_count", 0) == 0
        exceeds_normal_timeout = (
            estimated_duration_s + STAGE_MOVE_TIMEOUT_MARGIN_S
            >= original_timeout_ms / 1000.0
        )
        if is_initial_move or exceeds_normal_timeout:
            return max(original_timeout_ms, LARGE_STAGE_MOVE_TIMEOUT_MS), (
                estimated_duration_s
            )
        return original_timeout_ms, estimated_duration_s

    def _stop_active_stage_scan(self, timeout_s: float = 30.0) -> None:
        """Stop an ASI scan and wait for its controller cleanup to finish.

        Parameters
        ----------
        timeout_s : float
            Maximum time to wait for the scan state and XY device to become idle.

        Raises
        ------
        RuntimeError
            If the ASI controller does not become idle before the timeout.
        """
        self.start_asi_scan_after_camera_sequence = False
        if self.simulate_hardware:
            if self.simulated_asi_state.get("scan_state") == "Running":
                self.simulated_asi_state["scan_state"] = "Idle"
            return

        xy_stage = self.mmcore.getXYStageDevice()
        if not xy_stage or not self.mmcore.hasProperty(xy_stage, "ScanState"):
            return

        scan_state = self.mmcore.getProperty(xy_stage, "ScanState")
        if scan_state != "Idle":
            debug(
                "ASI SCAN CLEANUP",
                f"Stopping previous scan before the next stage move; state: {scan_state}",
            )
            self.mmcore.setProperty(xy_stage, "ScanState", "Idle")

        deadline = monotonic() + float(timeout_s)
        while True:
            scan_state = self.mmcore.getProperty(xy_stage, "ScanState")
            stage_busy = bool(self.mmcore.deviceBusy(xy_stage))
            if scan_state == "Idle" and not stage_busy:
                return
            if monotonic() >= deadline:
                raise RuntimeError(
                    f'ASI stage did not become idle within {timeout_s:g}s '
                    f'(ScanState={scan_state!r}, deviceBusy={stage_busy})'
                )
            sleep(0.05)

    def _restore_stage_after_scan(self) -> None:
        """Stop an ASI scan and restore the pre-acquisition XY move speeds."""
        self.start_asi_scan_after_camera_sequence = False
        if self.simulate_hardware:
            if self.simulated_asi_state:
                self.simulated_asi_state["scan_state"] = "Idle"
                self.simulated_asi_transitions.append("Idle")

        xy_stage = self.mmcore.getXYStageDevice()
        if not xy_stage:
            self._stage_speeds_before_sequence = {}
            return

        if not self.simulate_hardware:
            try:
                self._stop_active_stage_scan()
            except Exception:
                logging.getLogger(__name__).exception(
                    "Could not return the ASI stage scan to Idle"
                )

        fallback_speed = self._config["OPM"]["stage_move_speed"]
        for prop in ("MotorSpeedX-S(mm/s)", "MotorSpeedY-S(mm/s)"):
            if not self.mmcore.hasProperty(xy_stage, prop):
                continue
            speed = self._stage_speeds_before_sequence.get(prop, fallback_speed)
            try:
                self.mmcore.setProperty(xy_stage, prop, speed)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Could not restore %s to %s", prop, speed
                )
        try:
            self.mmcore.waitForDevice(xy_stage)
        except Exception:
            logging.getLogger(__name__).exception(
                "Stage did not become ready after restoring move speeds"
            )
        self._stage_speeds_before_sequence = {}

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

        Raises
        ------
        SkipEvent
            When a cooperative STOP is honored before a custom software command.
        RuntimeError
            If a completed stage move remains outside the target tolerance.
        """
        if isinstance(event.action, CustomAction):
            action_name = event.action.name
            if self._safe_stop_requested.is_set():
                self._safe_stop_requested.clear()
                info(
                    "OPM STOPPING AT SAFE POINT",
                    f"Skipping software command: {action_name}",
                    "Starting normal hardware teardown",
                )
                self.mmcore.mda.cancel()
                raise SkipEvent(
                    num_frames=0,
                    reason=f"OPM STOP before software command {action_name!r}",
                )
            data_dict = event.action.data
            if action_name == ACTION_O2O3_AUTOFOCUS:
                # Stop DAQ playback
                if self.opmDAQ.running():
                    self.opmDAQ.stop_waveform_playback()
                self.opmDAQ.reset_ao_channels()

                # Setup camera properties
                self.configure_camera(data_dict)

            elif action_name == ACTION_STAGE_MOVE:
                # Move the stage at the normal configured speed unless this event
                # explicitly requests an accelerated Stage Explorer preview move.
                stage_data = data_dict["Stage"]
                stage_move_speed = float(self._config["OPM"]["stage_move_speed"])
                stage_move_speed_x = float(
                    stage_data.get("move_speed_x_mm_s", stage_move_speed)
                )
                stage_move_speed_y = float(
                    stage_data.get("move_speed_y_mm_s", stage_move_speed)
                )
                target_x = np.round(float(stage_data["x_pos"]), 2)
                target_y = np.round(float(stage_data["y_pos"]), 2)
                current_x, current_y = self.mmcore.getXYPosition()
                original_timeout_ms = int(self.mmcore.getTimeoutMs())
                move_timeout_ms, estimated_duration_s = self._stage_move_timeout_ms(
                    current_x,
                    current_y,
                    target_x,
                    target_y,
                    stage_move_speed_x,
                    stage_move_speed_y,
                )
                if move_timeout_ms != original_timeout_ms:
                    debug(
                        "EXTENDED STAGE MOVE TIMEOUT",
                        f"move: ({current_x:.2f}, {current_y:.2f}) -> "
                        f"({target_x:.2f}, {target_y:.2f}) um",
                        f"estimated duration: {estimated_duration_s:.2f} s",
                        f"timeout: {original_timeout_ms} -> {move_timeout_ms} ms",
                    )
                    self.mmcore.setTimeoutMs(move_timeout_ms)

                try:
                    # A camera sequence can finish while the ASI controller is
                    # still completing its scan cleanup.  Stop it explicitly before
                    # changing speeds or commanding the next XY position.
                    self._stop_active_stage_scan()
                    self._apply_stage_move_speeds(
                        {
                            "move_speed_x_mm_s": stage_move_speed_x,
                            "move_speed_y_mm_s": stage_move_speed_y,
                        }
                    )
                    self.mmcore.setPosition(
                        np.round(float(stage_data["z_pos"]), 2)
                    )
                    self.mmcore.waitForDevice(self.mmcore.getFocusDevice())
                    self.mmcore.setXYPosition(target_x, target_y)
                    self.mmcore.waitForDevice(self.mmcore.getXYStageDevice())
                    current_x, current_y = self.mmcore.getXYPosition()
                    if not (
                        np.isclose(current_x, target_x, rtol=0.0, atol=1.0)
                        and np.isclose(current_y, target_y, rtol=0.0, atol=1.0)
                    ):
                        raise RuntimeError(
                            "Stage stopped outside the target tolerance: "
                            f"current=({current_x:.2f}, {current_y:.2f}) um, "
                            f"target=({target_x:.2f}, {target_y:.2f}) um"
                        )
                    self._stage_move_count = getattr(self, "_stage_move_count", 0) + 1
                finally:
                    if move_timeout_ms != original_timeout_ms:
                        self.mmcore.setTimeoutMs(original_timeout_ms)

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
                self.mmcore.setProperty(plcName, propPosition, addrOutputBNC1)
                self.mmcore.setProperty(plcName, propCellConfig, addrStageSync)

                # --------------------------------------------------------#
                # Set stage speed, scan axis (x) and tile axis (y)
                self.mmcore.setProperty(
                    self.mmcore.getXYStageDevice(),
                    "MotorSpeedX-S(mm/s)",
                    np.round(data_dict["ASI"]["scan_axis_speed_mm_s"], 4),
                )
                self.mmcore.setProperty(
                    self.mmcore.getXYStageDevice(),
                    "MotorSpeedY-S(mm/s)",
                    self._config["OPM"]["stage_move_speed"],
                )

                # --------------------------------------------------------#
                # Set scan axis to true 1D scan with no backlash
                self.mmcore.setProperty(
                    self.mmcore.getXYStageDevice(), "ScanPattern", "Raster"
                )
                self.mmcore.setProperty(
                    self.mmcore.getXYStageDevice(), "ScanSlowAxis", "Null (1D scan)"
                )
                self.mmcore.setProperty(
                    self.mmcore.getXYStageDevice(), "ScanFastAxis", "1st axis"
                )
                self.mmcore.setProperty(
                    self.mmcore.getXYStageDevice(), "ScanSettlingTime(ms)", 3000
                )

                # --------------------------------------------------------#
                # Set scan axis start/end positions
                self.mmcore.setProperty(
                    self.mmcore.getXYStageDevice(),
                    "ScanFastAxisStartPosition(mm)",
                    np.round(data_dict["ASI"]["scan_axis_start_mm"], 6),
                )
                self.mmcore.setProperty(
                    self.mmcore.getXYStageDevice(),
                    "ScanFastAxisStopPosition(mm)",
                    np.round(data_dict["ASI"]["scan_axis_end_mm"], 6),
                )

                # --------------------------------------------------------#
                # Set the scan state
                self.mmcore.setProperty(
                    self.mmcore.getXYStageDevice(), "ScanState", "Idle"
                )

                if DEBUGGING:
                    actual_speed_x = float(
                        self.mmcore.getProperty(
                            self.mmcore.getXYStageDevice(), "MotorSpeedX-S(mm/s)"
                        )
                    )
                    scanaxis_start = self.mmcore.getProperty(
                        self.mmcore.getXYStageDevice(), "ScanFastAxisStartPosition(mm)"
                    )
                    scanaxis_stop = self.mmcore.getProperty(
                        self.mmcore.getXYStageDevice(), "ScanFastAxisStopPosition(mm)"
                    )
                    scan_settling_ms = self.mmcore.getProperty(
                        self.mmcore.getXYStageDevice(), "ScanSettlingTime(ms)"
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

                # Match the working main-branch order: program DAQ, ROI, and
                # exposure first, then arm the camera for the ASI start pulse.
                self.configure_stage_camera_trigger()
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
                    xy_stage = self.mmcore.getXYStageDevice()
                    if self.mmcore.hasProperty(xy_stage, "MotorSpeedX-S(mm/s)"):
                        self.mmcore.setProperty(
                            xy_stage, "MotorSpeedX-S(mm/s)", stage_move_speed
                        )
                    if self.mmcore.hasProperty(xy_stage, "MotorSpeedY-S(mm/s)"):
                        self.mmcore.setProperty(
                            xy_stage, "MotorSpeedY-S(mm/s)", stage_move_speed
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
                    # self.mmcore.setProperty(
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

                self.mmcore.setProperty(
                    str(self._config["Camera"]["camera_id"]), "Exposure", exposure_ms
                )
                self.mmcore.waitForDevice(str(self._config["Camera"]["camera_id"]))

                # Wait for MM core
                self.mmcore.waitForSystem()

                debug(
                    "CAMERA EXPOSURES",
                    f"actual: {np.round(self.mmcore.getExposure(), 2)}",
                    f"requested: {exposure_ms}",
                )

            elif action_name == ACTION_MIRROR_MOVE:
                # --------------------------------------------------------#
                # Update daq waveform values and setup daq for playback
                self.opmDAQ.stop_waveform_playback()
                self.opmDAQ.clear_tasks()

                # Modify the image neutral position
                self.opmDAQ.set_mirror_neutral_position(
                    image_mirror_v=float(data_dict["DAQ"]["image_mirror_v"])
                )

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
                self.mmcore.setProperty(
                    self.mmcore.getXYStageDevice(), "ScanState", "Running"
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
            if self.simulate_hardware:
                self.simulated_custom_actions.append(action_name)
                if action_name in {
                    ACTION_O2O3_AUTOFOCUS,
                    ACTION_AO_OPTIMIZE,
                    ACTION_AO_GRID,
                    ACTION_FLUIDICS,
                }:
                    return ()

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
                        position_indices=data_dict["AO"].get("position_indices"),
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
            return ()
        return super().exec_event(event) or ()

    def teardown_event(self, event):
        """Release per-event state after execution.

        Parameters
        ----------
        event : MDAEvent
            Event that has completed.
        """
        if isinstance(event.action, CustomAction):
            self.mmcore.clearCircularBuffer()
            return
        super().teardown_event(event)

    def teardown_sequence(self, sequence: MDASequence) -> None:
        """Restore hardware state after an acquisition sequence.

        Parameters
        ----------
        sequence : MDASequence
            Sequence that has completed or been canceled.
        """
        debug("TEARDOWN", "Acquisition finished, tearing down.")
        sequence_metadata = getattr(sequence, "metadata", {})
        is_stage_explorer_preview = (
            STAGE_MOVE_SPEED_METADATA_KEY in sequence_metadata
        )
        preview_stage_restored = False

        try:
            # Shut down the trigger source before stopping the hardware scan.
            # Saved OPM acquisitions restore move speeds immediately.  Explorer
            # previews keep their temporary 4x speed through the upstream return
            # to the pre-scan XY position.
            try:
                self.opmDAQ.clear_tasks()
                self.opmDAQ.reset()
                self.opmDAQ.set_mirror_neutral_position(
                    image_mirror_v=float(
                        self._config["NIDAQ"]["image_mirror_neutral_v"]
                    )
                )
            finally:
                if not is_stage_explorer_preview:
                    self._restore_stage_after_scan()

            # Put cameras that expose the Hamamatsu trigger properties back in
            # internal mode.
            camera = str(self._config["Camera"]["camera_id"])
            if self.mmcore.hasProperty(camera, "TriggerPolarity"):
                self.mmcore.setProperty(camera, "TriggerPolarity", "POSITIVE")
                self.mmcore.waitForDevice(camera)
            if self.mmcore.hasProperty(camera, "TRIGGER SOURCE"):
                self.mmcore.setProperty(camera, "TRIGGER SOURCE", "INTERNAL")
                self.mmcore.waitForDevice(camera)

            # Explorer previews use the live Config Groups controls, so preserve
            # their laser-power properties.  Saved OPM acquisitions retain their
            # existing zero-emission cleanup behavior.
            if not is_stage_explorer_preview:
                for laser in self._config["Lasers"]["laser_names"]:
                    if self.simulate_hardware:
                        self.simulated_laser_powers[laser] = 0.0
                    else:
                        self.mmcore.setProperty(
                            self._config["Lasers"]["name"],
                            laser + " - PowerSetpoint (%)",
                            0.0,
                        )

            if self.AOMirror.output_path:
                self.AOMirror.save_positions_array()
            self.mmcore.clearCircularBuffer()

            # Upstream restores the pre-acquisition XYZ position here.  That
            # return is commonly much longer than a neighboring tile move, so
            # keep the extended timeout active until the return command and all
            # subsequent state restoration have completed.
            original_timeout_ms = int(self.mmcore.getTimeoutMs())
            teardown_timeout_ms = max(
                original_timeout_ms, LARGE_STAGE_MOVE_TIMEOUT_MS
            )
            if teardown_timeout_ms != original_timeout_ms:
                self.mmcore.setTimeoutMs(teardown_timeout_ms)
            try:
                super().teardown_sequence(sequence)
            finally:
                if teardown_timeout_ms != original_timeout_ms:
                    self.mmcore.setTimeoutMs(original_timeout_ms)

            if is_stage_explorer_preview:
                self._restore_stage_after_scan()
                preview_stage_restored = True

            if self._post_teardown is not None:
                try:
                    self._post_teardown()
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Could not prepare live preview after OPM teardown"
                    )
                    warning(
                        "LIVE PREVIEW PREPARATION FAILED",
                        "Live preview will rebuild the DAQ waveform when started.",
                    )
        finally:
            if is_stage_explorer_preview and not preview_stage_restored:
                self._restore_stage_after_scan()
            if is_stage_explorer_preview:
                self._restore_stage_explorer_timeout()
            self.clear_safe_stop()
