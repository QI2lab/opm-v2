"""Launch the OPM Micro-Manager GUI using an application controller."""

from __future__ import annotations

import json
import os
import sys
import traceback
from contextlib import suppress
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from pymmcore_gui import MicroManagerGUI, WidgetAction, create_mmgui
from pymmcore_gui._qt.QtCore import QTimer
from pymmcore_plus.mda import MDAEngine
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDockWidget

from opm_v2._update_config_widget import OPMSettingsV2
from opm_v2.engine.opm_engine import OPMEngineV2
from opm_v2.engine.setup_events import (
    setup_mirrorscan,
    setup_optimizenow,
    setup_projection,
    setup_stagescan,
    setup_timelapse,
)
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.ElveFlow import OB1Controller
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.PicardShutter import PicardShutter

DEBUGGING = True
DEBUG_SEPARATOR = "-" * 72
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "opm_config.json"
IS_FROZEN = getattr(sys, "frozen", False)
MIN_PROJECTION_EXPOSURE = 50  # ms
DEFUALT_PROJECTION_EXPOSURE = 150

if TYPE_CHECKING:
    from types import TracebackType

    ExcTuple = tuple[type[BaseException], BaseException, TracebackType | None]


class ConfigStore:
    """Provide a reloadable wrapper around an OPM JSON configuration.

    Parameters
    ----------
    path : Path
        Path to the JSON configuration file.
    """

    def __init__(self, path: Path) -> None:
        """Create a config store and immediately load the JSON file.

        Parameters
        ----------
        path : Path
            Path to the JSON configuration file.
        """
        self.path = Path(path)
        self.data: dict = {}
        self.reload()

    def reload(self) -> dict:
        """Reload the JSON configuration in place.

        Returns
        -------
        dict
            Shared configuration dictionary updated from disk.
        """
        with open(self.path, "r") as config_file:
            new_config = json.load(config_file)
        self.data.clear()
        self.data.update(new_config)
        return self.data


class OPMAppController:
    """Manage MMGUI, OPM hardware, live-state updates, and MDA execution.

    Parameters
    ----------
    config_path : Path
        Path to the OPM JSON configuration file.
    """

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        *,
        mmcore=None,
        mm_config: Path | str | None | Literal[False] = None,
        exec_app: bool = True,
        simulate_hardware: bool | None = None,
    ) -> None:
        """Load configuration and prepare placeholders for app-owned objects.

        Heavy GUI and hardware objects are initialized later by ``run()`` so the
        controller can be constructed without immediately touching hardware.

        Parameters
        ----------
        config_path : Path
            Path to the OPM JSON configuration file.
        mmcore : CMMCorePlus or None
            Existing core instance to reuse, or ``None`` to use MMGUI's instance.
        mm_config : Path, str, False, or None
            Micro-Manager configuration passed to ``create_mmgui``. ``False``
            preserves the state of a supplied core.
        exec_app : bool
            Whether to enter the Qt application event loop.
        simulate_hardware : bool or None
            Override for external OPM hardware simulation.
        """
        self.config_store = ConfigStore(config_path)
        self.config = self.config_store.data
        self.config_path = self.config_store.path
        self._provided_mmcore = mmcore
        self._mm_config = mm_config
        self._exec_app = exec_app
        configured_simulation = bool(
            self.config.get("OPM", {}).get("simulate_hardware", False)
        )
        self.simulate_hardware = (
            configured_simulation
            if simulate_hardware is None
            else bool(simulate_hardware)
        )

        self.win = None
        self.mmc = None
        self.mda_widget = None
        self.opm_settings_widget = None
        self.opm_ao_mirror = None
        self.opm_nidaq = None
        self.opm_picard_shutter = None
        self.ob1_controller = None
        self.bootstrap_complete = False

    def _print_block(self, header: str, *lines: object) -> None:
        """Print a visually separated console message block.

        Parameters
        ----------
        header : str
            Block heading.
        *lines : object
            Values to print beneath the heading.
        """
        print(f"\n{DEBUG_SEPARATOR}")
        print(f"----- {header} -----")
        for line in lines:
            print(line)
        print(DEBUG_SEPARATOR)

    def debug(self, header: str, *lines: object) -> None:
        """Print a debug block when module-level debugging is enabled.

        Parameters
        ----------
        header : str
            Block heading.
        *lines : object
            Values to print beneath the heading.
        """
        if DEBUGGING:
            self._print_block(f"DEBUGGING: {header}", *lines)

    def info(self, header: str, *lines: object) -> None:
        """Print a status block.

        Parameters
        ----------
        header : str
            Block heading.
        *lines : object
            Values to print beneath the heading.
        """
        self._print_block(header, *lines)

    def warning(self, header: str, *lines: object) -> None:
        """Print a warning block.

        Parameters
        ----------
        header : str
            Block heading.
        *lines : object
            Values to print beneath the heading.
        """
        self._print_block(f"WARNING: {header}", *lines)

    def run(self) -> MicroManagerGUI:
        """Create the GUI, schedule OPM bootstrap, and optionally start Qt.

        Returns
        -------
        MicroManagerGUI
            Configured Micro-Manager main window.
        """
        self.create_gui()
        QTimer.singleShot(0, partial(self._schedule_extension_step_two))
        if self._exec_app and (app := QApplication.instance()) is not None:
            app.exec()
        return self.win

    def _schedule_extension_step_two(self) -> None:
        """Wait one more event-loop turn for pymmcore-gui layout restoration."""
        QTimer.singleShot(0, self.finish_bootstrap)

    def finish_bootstrap(self) -> None:
        """Attach OPM UI and hardware after pymmcore-gui restores its layout."""
        self.add_settings_widget()
        self.initialize_hardware()
        self.connect_signals()
        self.bootstrap_complete = True

    def reload_config(self) -> dict:
        """Refresh the configuration while preserving its dictionary identity.

        Returns
        -------
        dict
            Reloaded shared configuration dictionary.
        """
        return self.config_store.reload()

    def create_gui(self) -> None:
        """Create the Micro-Manager GUI and apply OPM-specific widget defaults."""
        mm_config = self._mm_config
        if mm_config is None:
            mm_config = (
                False
                if self._provided_mmcore is not None
                else Path(self.config["OPM"]["mm_config_path"])
            )
        self.win = create_mmgui(
            mm_config=mm_config,
            mmcore=self._provided_mmcore,
            exec_app=False,
        )
        self.mmc = self.win.mmcore
        self.win.opm_controller = self

        self.mda_widget = self.win.get_widget(WidgetAction.MDA_WIDGET)
        self.mda_widget.save_info.save_dir.setText(r"E:/")
        self.mda_widget.tab_wdg.grid_plan.setMode("bounds")
        self.mda_widget.tab_wdg.grid_plan._mode_bounds_radio.toggle()

        if DEBUGGING:
            self.mmc.enableDebugLog(True)

        self.debug(
            "GUI CREATED",
            f"MM config: {mm_config}",
            "MDA save directory: E:/",
            "Grid plan mode: bounds",
        )

    def add_settings_widget(self) -> None:
        """Dock the custom OPM settings widget into the Micro-Manager GUI."""
        self.opm_settings_widget = OPMSettingsV2(config_path=self.config_path)
        dock_widget = QDockWidget("OPM Settings", self.win)
        dock_widget.setWidget(self.opm_settings_widget)
        dock_widget.setObjectName("OPMConfigurator")
        self.win.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_widget)
        dock_widget.setFloating(False)

    def initialize_hardware(self) -> None:
        """Instantiate OPM hardware controllers and put them in startup states."""
        # ------------------------------------------------------------------------------#
        # AO mirror
        # ------------------------------------------------------------------------------#

        self.opm_ao_mirror = AOMirror(
            wfc_config_file_path=Path(self.config["AOMirror"]["wfc_config_path"]),
            haso_config_file_path=Path(self.config["AOMirror"]["haso_config_path"]),
            interaction_matrix_file_path=Path(
                self.config["AOMirror"]["wfc_correction_path"]
            ),
            system_flat_file_path=Path(self.config["AOMirror"]["wfc_flat_path"]),
            n_modes=32,
            n_positions=1,
            modes_to_ignore=[],
            simulate=self.simulate_hardware,
        )
        self.opm_ao_mirror.apply_system_flat_voltage()

        self.debug(
            "AO MIRROR INITIALIZED",
            f"System flat: {self.config['AOMirror']['wfc_flat_path']}",
            "Startup state: system flat voltage applied",
        )

        # ------------------------------------------------------------------------------#
        # NIDAQ
        # ------------------------------------------------------------------------------#

        self.opm_nidaq = OPMNIDAQ(
            name=str(self.config["NIDAQ"]["name"]),
            scan_type=str(self.config["NIDAQ"]["scan_type"]),
            exposure_ms=float(self.config["Camera"]["exposure_ms"]),
            laser_blanking=bool(self.config["acq_config"]["DAQ"]["laser_blanking"]),
            image_mirror_calibration=float(
                str(self.config["NIDAQ"]["image_mirror_calibration"])
            ),
            projection_mirror_calibration=float(
                str(self.config["NIDAQ"]["projection_mirror_calibration"])
            ),
            image_mirror_neutral_v=float(
                str(self.config["NIDAQ"]["image_mirror_neutral_v"])
            ),
            projection_mirror_neutral_v=float(
                str(self.config["NIDAQ"]["projection_mirror_neutral_v"])
            ),
            image_mirror_step_um=float(
                str(self.config["NIDAQ"]["image_mirror_step_um"])
            ),
            verbose=bool(self.config["NIDAQ"]["verbose"]),
            simulate=self.simulate_hardware,
        )
        self.opm_nidaq.reset()

        self.debug(
            "NIDAQ INITIALIZED",
            f"Name: {self.config['NIDAQ']['name']}",
        )

        # ------------------------------------------------------------------------------#
        # O2-O3 laser shutter
        # ------------------------------------------------------------------------------#

        self.opm_picard_shutter = PicardShutter(
            int(self.config["O2O3-autofocus"]["shutter_id"]),
            verbose=True,
            simulate=self.simulate_hardware,
        )
        self.opm_picard_shutter.closeShutter()

        self.debug(
            "O2-O3 SHUTTER INITIALIZED",
            f"Shutter id: {self.config['O2O3-autofocus']['shutter_id']}",
            "Startup state: closed",
        )

        # ------------------------------------------------------------------------------#
        # Fluidics
        # ------------------------------------------------------------------------------#

        if bool(self.config["OPM"].get("fluidics_enabled", False)):
            self.ob1_controller = OB1Controller(
                port=str(self.config["OB1"]["port"]),
                to_OB1_pin=int(self.config["OB1"]["to_OB1_pin"]),
                from_OB1_pin=int(self.config["OB1"]["from_OB1_pin"]),
                simulate=self.simulate_hardware,
            )
            self.debug("FLUIDICS INITIALIZED", "OB1 controller enabled")
        else:
            self.debug("FLUIDICS SKIPPED", "config['OPM']['fluidics_enabled'] is false")

    def connect_signals(self) -> None:
        """Connect Micro-Manager and OPM widget signals to controller methods."""
        # Changes to the mm config
        self.mmc.events.configSet.connect(self.update_live_state)
        self.update_live_state()

        # Changes to the selected AO mirror state
        self.opm_settings_widget.widgets["AO"][
            "mirror_state"
        ].currentIndexChanged.connect(self.update_ao_mirror_state)

        # Connect live preview
        self.mmc.events.continuousSequenceAcquisitionStarting.connect(
            self.setup_preview_mode_callback
        )

        # Replace mda execution function
        self.mda_widget.execute_mda = self.custom_execute_mda

        self.debug(
            "SIGNALS CONNECTED",
            "mmc.events.configSet -> update_live_state",
            "AO mirror_state currentIndexChanged -> update_ao_mirror_state",
            "continuousSequenceAcquisitionStarting -> setup_preview_mode_callback",
            "mda_widget.execute_mda -> custom_execute_mda",
        )

    def update_live_state(self, device_name=None, property_name=None) -> None:
        """Apply live camera, DAQ, and channel state after MM config changes.

        The callback may temporarily stop continuous sequence acquisition while
        the DAQ waveform and camera settings are updated, then restart preview.

        Parameters
        ----------
        device_name : str or None
            Micro-Manager device that emitted the configuration change.
        property_name : str or None
            Device property that changed.
        """
        self.reload_config()
        daq_config = self.config["acq_config"]["DAQ"]
        camera_roi = self.config["acq_config"]["camera_roi"]

        # Ignore shutter signals
        if device_name == self.mmc.getShutterDevice() and property_name == "State":
            return

        self.debug(
            "LIVE STATE TRIGGERED",
            f"Device name: {device_name}",
            f"Property name: {property_name}",
        )

        if "OPM-live-mode" not in set(self.mmc.getLoadedDevices()):
            self.debug(
                "LIVE STATE SKIPPED",
                "The active Micro-Manager configuration has no OPM-live-mode device.",
            )
            return

        # ------------------------------------------------------------------------------#
        # OPM live modes
        # ------------------------------------------------------------------------------#

        opm_mode = self.mmc.getProperty("OPM-live-mode", "Label")
        if opm_mode == "0-Standard":
            scan_type = "2d"
        elif opm_mode == "1-Projection":
            scan_type = "projection"
        else:
            self.warning("UNKNOWN LIVE MODE", f"OPM-live-mode: {opm_mode}")
            return

        # ------------------------------------------------------------------------------#
        # Update the DAQ state
        # ------------------------------------------------------------------------------#

        # Stop / restart live preview
        restart_sequence = False
        if self.mmc.isSequenceRunning():
            self.mmc.stopSequenceAcquisition()
            restart_sequence = True

        # Stop the DAQ playback
        if self.opm_nidaq.running():
            self.opm_nidaq.stop_waveform_playback()

        # Get the current camera exposure
        exposure_ms = round(
            float(self.mmc.getProperty(self.config["Camera"]["camera_id"], "Exposure")),
            0,
        )

        # Get the current active laser channel, limits to one channel at a time
        image_mirror_range_um = np.round(
            float(self.mmc.getProperty("ImageGalvoMirrorRange", "Position")),
            0,
        )

        # Create the channel states list needed for the DAQ waveform
        active_channel_id = self.mmc.getProperty("Laser", "Label")
        channel_states = [
            active_channel_id == channel_id
            for channel_id in self.config["OPM"]["channel_ids"]
        ]

        # Define the y cropping and check for valid exposures
        if scan_type == "projection":
            crop_y = int(image_mirror_range_um / self.mmc.getPixelSizeUm())
            if exposure_ms < MIN_PROJECTION_EXPOSURE:
                self.warning(
                    "PROJECTION EXPOSURE TOO LOW",
                    f"Requested exposure: {exposure_ms} ms",
                    f"Defualting to {DEFUALT_PROJECTION_EXPOSURE}ms",
                )
                exposure_ms = DEFUALT_PROJECTION_EXPOSURE
        else:
            crop_y = int(self.mmc.getProperty("ImageCameraCrop", "Label"))

        # Set the acquisition parameters
        self.opm_nidaq.set_acquisition_params(
            scan_type=scan_type,
            channel_states=channel_states,
            image_mirror_range_um=image_mirror_range_um,
            exposure_ms=exposure_ms,
            laser_blanking=daq_config["laser_blanking"],
        )

        # ------------------------------------------------------------------------------#
        # Update the Camera state
        # ------------------------------------------------------------------------------#

        # Set / check camera ROI
        if crop_y != self.mmc.getROI()[-1]:
            self.mmc.clearROI()
            self.mmc.waitForDevice(str(self.config["Camera"]["camera_id"]))
            self.mmc.setROI(
                int(camera_roi["center_x"] - camera_roi["crop_x"] // 2),
                int(camera_roi["center_y"] - crop_y // 2),
                int(camera_roi["crop_x"]),
                crop_y,
            )
            self.mmc.waitForDevice(str(self.config["Camera"]["camera_id"]))

            self.debug("CAMERA ROI UPDATED", f"Crop Y: {crop_y}")

        # Set / check camera exposure
        if self.mmc.getExposure() != exposure_ms:
            self.mmc.setProperty(
                str(self.config["Camera"]["camera_id"]),
                "Exposure",
                exposure_ms,
            )
            self.mmc.waitForDevice(str(self.config["Camera"]["camera_id"]))
            self.debug("CAMERA EXPOSURE UPDATED", f"Exposure: {exposure_ms} ms")

        self.debug(
            "LIVE STATE APPLIED",
            f"Sequence was running: {restart_sequence}",
            f"Scan type: {scan_type}",
            f"Channel states: {channel_states}",
            f"Image mirror range: {image_mirror_range_um} um",
            f"Exposure: {exposure_ms} ms",
            f"Laser blanking: {daq_config['laser_blanking']}",
        )

        # Start live preview
        if restart_sequence:
            self.mmc.startContinuousSequenceAcquisition()
            self.debug("LIVE PREVIEW RESTARTED")

    def update_ao_mirror_state(self) -> None:
        """Apply the AO mirror state selected in the OPM settings widget."""
        self.reload_config()
        ao_mirror_state = self.config["acq_config"]["AO"]["mirror_state"]

        if "system" in ao_mirror_state:
            self.opm_ao_mirror.apply_system_flat_voltage()
        elif "optimized" in ao_mirror_state:
            self.opm_ao_mirror.apply_optimized_voltage()
        elif "factory" in ao_mirror_state:
            self.opm_ao_mirror.apply_factory_flat_voltage()
        elif "zeros" in ao_mirror_state:
            self.opm_ao_mirror.apply_zeros_voltage()

        self.debug("AO MIRROR STATE UPDATED", f"Mirror state: {ao_mirror_state}")

    def setup_preview_mode_callback(self) -> None:
        """Program the OPM NIDAQ waveform before continuous preview starts."""
        if self.opm_nidaq.running():
            self.opm_nidaq.stop_waveform_playback()

        self.update_live_state()
        if any(self.opm_nidaq.channel_states):
            self.opm_nidaq.clear_tasks()
            self.opm_nidaq.generate_waveforms()
            self.opm_nidaq.program_daq_waveforms()
            self.opm_nidaq.start_waveform_playback()

            self.debug(
                "PREVIEW DAQ REPROGRAMMED",
                f"Channel states: {self.opm_nidaq.channel_states}",
            )

    def custom_execute_mda(self, output: Path | str | None) -> None:
        """Route the MDA widget action to normal MDA or OPM custom events.

        Non-zarr outputs run through the default pymmcore-plus engine. OPM zarr
        outputs and optimize-now modes build custom events and use ``OPMEngineV2``.

        Parameters
        ----------
        output : Path, str, or None
            Requested MDA output or ``"memory"`` for an in-memory acquisition.
        """
        self.reload_config()

        opm_mode = self.config["acq_config"]["opm_mode"]
        ao_mode = self.config["acq_config"]["AO"]["ao_mode"]
        o2o3_mode = self.config["acq_config"]["o2o3_mode"]
        fluidics_mode = self.config["acq_config"]["fluidics"]
        optimize_now = ("now" in ao_mode) or ("now" in o2o3_mode)

        if output == "memory":
            print("No output path supplied")
            output = None

        output = Path(output) if isinstance(output, str) else output
        opm_acquisition = self.is_opm_acquisition(output, optimize_now)

        if not opm_acquisition:
            self.info(
                "STANDARD MDA ACQUISITION",
                f"Output: {output}",
                "Engine: pymmcore-plus MDAEngine",
            )
            self.mmc.register_mda_engine(MDAEngine(self.mmc))
            self.mmc.run_mda(self.mda_widget.value(), output=output)
            return

        self.info(
            "OPM ACQUISITION",
            f"Output: {output}",
            f"Optimize now: {optimize_now}",
        )
        if not optimize_now and not output:
            self.warning(
                "ACQUISITION NOT STARTED",
                "Must set acquisition path to execute acquisition",
            )
            return

        if output:
            output = self.timestamped_output_path(output)

        if ("none" not in fluidics_mode) and not optimize_now:
            if not self.confirm_fluidics_ready():
                return
            self.info("FLUIDICS ACCEPTED", "ESI sequence confirmed by user")

        self.debug(
            "OPM ACQUISITION SETTINGS",
            f"OPM mode: {opm_mode}",
            f"AO mode: {ao_mode}",
            f"O2O3 mode: {o2o3_mode}",
            f"Fluidics mode: {fluidics_mode}",
            f"Output path: {output}",
        )

        opm_events, handler = self.create_opm_events(optimize_now, opm_mode, output)

        if opm_events is None:
            self.warning("ACQUISITION NOT STARTED", "OPM events are empty")
            return

        self.debug(
            "OPM EVENTS READY",
            f"Event count: {len(opm_events)}",
            f"Handler: {handler}",
            "Engine: OPMEngineV2",
        )
        self.mmc.register_mda_engine(
            OPMEngineV2(
                self.mmc,
                self.config_path,
                simulate_hardware=self.simulate_hardware,
            )
        )
        self.mmc.run_mda(opm_events, output=handler)

    def is_opm_acquisition(self, output: Path | None, optimize_now: bool) -> bool:
        """Determine whether an MDA request should use OPM custom events.

        Parameters
        ----------
        output : Path or None
            Requested acquisition output.
        optimize_now : bool
            Whether an immediate autofocus or AO optimization was requested.

        Returns
        -------
        bool
            ``True`` when the OPM event builder should handle the request.
        """
        if output is None:
            if optimize_now:
                return True
            else:
                return False
        if optimize_now:
            return True
        return len(output.suffixes) == 1 and output.suffix == ".zarr"

    def timestamped_output_path(self, output: Path) -> Path:
        """Create a timestamped OPM acquisition output path.

        Parameters
        ----------
        output : Path
            User-selected output path.

        Returns
        -------
        Path
            Output path inside a newly created timestamped directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir = output.parent / Path(f"{timestamp}_{output.stem}")
        new_dir.mkdir(exist_ok=True)
        return new_dir / Path(output.name)

    def confirm_fluidics_ready(self) -> bool:
        """Ask whether the external ESI/fluidics sequence is ready.

        Returns
        -------
        bool
            ``True`` when the user confirms readiness.
        """
        from PyQt6.QtWidgets import QMessageBox

        response = QMessageBox.information(
            self.mda_widget,
            "!WARNING! ESI MUST BE RUNNING!",
            "IS ESI SEQUENCE LOADED AND STARTED?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        return response is QMessageBox.StandardButton.Yes

    def create_opm_events(
        self,
        optimize_now: bool,
        opm_mode: str,
        output: Path | None,
    ):
        """Build custom OPM events for the selected acquisition mode.

        Parameters
        ----------
        optimize_now : bool
            Whether to build immediate optimization events.
        opm_mode : str
            Selected OPM acquisition mode.
        output : Path or None
            Acquisition output path.

        Returns
        -------
        tuple
            Event list and output handler, or ``(None, None)`` when no valid
            acquisition can be built.
        """
        if optimize_now:
            return setup_optimizenow(mmc=self.mmc, config=self.config)

        if not output:
            return None, None

        sequence = self.mda_widget.value()
        if "timelapse" in opm_mode:
            opm_events, handler = setup_timelapse(
                mmc=self.mmc,
                config=self.config,
                sequence=sequence,
                output=output,
            )
        elif "stage" in opm_mode:
            opm_events, handler = setup_stagescan(
                mmc=self.mmc,
                config=self.config,
                sequence=sequence,
                output=output,
            )
        elif "mirror" in opm_mode:
            opm_events, handler = setup_mirrorscan(
                mmc=self.mmc,
                config=self.config,
                sequence=sequence,
                output=output,
            )
        elif "projection" in opm_mode:
            opm_events, handler = setup_projection(
                mmc=self.mmc,
                config=self.config,
                sequence=sequence,
                output=output,
            )
        else:
            self.warning("UNKNOWN OPM ACQUISITION MODE", f"OPM mode: {opm_mode}")
            return None, None

        self.opm_ao_mirror.output_path = output.parents[0]
        return opm_events, handler


def launch_opm_app(
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    mm_config: Path | str | None | Literal[False] = None,
    mmcore=None,
    exec_app: bool = True,
    simulate_hardware: bool | None = None,
) -> MicroManagerGUI:
    """Launch pymmcore-gui and attach OPM after layout restoration.

    Parameters
    ----------
    config_path : Path or str
        OPM JSON configuration path.
    mm_config : Path, str, False, or None
        Micro-Manager configuration passed to ``create_mmgui``.
    mmcore : CMMCorePlus or None
        Existing core instance to reuse.
    exec_app : bool
        Whether to enter the Qt event loop.
    simulate_hardware : bool or None
        Override for external hardware simulation.

    Returns
    -------
    MicroManagerGUI
        Configured Micro-Manager main window.
    """
    _install_excepthook()
    controller = OPMAppController(
        Path(config_path),
        mmcore=mmcore,
        mm_config=mm_config,
        exec_app=exec_app,
        simulate_hardware=simulate_hardware,
    )
    return controller.run()


def main(config_path: Path | str | None = None) -> int:
    """Run the Micro-Manager GUI with the OPM controller.

    Parameters
    ----------
    config_path : Path, str, or None
        OPM JSON configuration path.

    Returns
    -------
    int
        Process exit status.
    """
    launch_opm_app(config_path=config_path or DEFAULT_CONFIG_PATH)
    return 0


def _install_excepthook() -> None:
    """Install a custom excepthook that does not raise sys.exit()."""
    if hasattr(sys, "_original_excepthook_"):
        return
    sys._original_excepthook_ = sys.excepthook  # type: ignore
    sys.excepthook = ndv_excepthook


def rich_print_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    """Render an exception using Rich.

    Parameters
    ----------
    exc_type : type[BaseException]
        Exception class.
    exc_value : BaseException
        Raised exception instance.
    exc_traceback : TracebackType or None
        Associated traceback.
    """
    import psygnal
    from rich.console import Console
    from rich.traceback import Traceback

    tb = Traceback.from_exception(
        exc_type,
        exc_value,
        exc_traceback,
        suppress=[psygnal],
        max_frames=100 if IS_FROZEN else 10,
        show_locals=True,
    )
    Console(stderr=True).print(tb)


def _print_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    """Print an exception with Rich when available.

    Parameters
    ----------
    exc_type : type[BaseException]
        Exception class.
    exc_value : BaseException
        Raised exception instance.
    exc_traceback : TracebackType or None
        Associated traceback.
    """
    try:
        rich_print_exception(exc_type, exc_value, exc_traceback)
    except ImportError:
        traceback.print_exception(exc_type, value=exc_value, tb=exc_traceback)


EXCEPTION_LOG: list[ExcTuple] = []


def ndv_excepthook(
    exc_type: type[BaseException],
    exc_value: BaseException,
    tb: TracebackType | None,
) -> None:
    """Record, display, and emit an unhandled GUI exception.

    Parameters
    ----------
    exc_type : type[BaseException]
        Exception class.
    exc_value : BaseException
        Raised exception instance.
    tb : TracebackType or None
        Associated traceback.
    """
    EXCEPTION_LOG.append((exc_type, exc_value, tb))
    _print_exception(exc_type, exc_value, tb)
    if sig := getattr(QApplication.instance(), "exceptionRaised", None):
        sig.emit(exc_value)
    if not tb:
        return

    if (
        (debugpy := sys.modules.get("debugpy"))
        and debugpy.is_client_connected()
        and ("pydevd" in sys.modules)
    ):  # pragma: no cover
        with suppress(Exception):
            import threading

            import pydevd  # pyright: ignore[reportMissingImports]

            if (py_db := pydevd.get_global_debugger()) is None:
                return

            py_db = cast("pydevd.PyDB", py_db)
            thread = threading.current_thread()
            additional_info = py_db.set_additional_thread_info(thread)
            additional_info.is_tracing += 1

            try:
                arg = (exc_type, exc_value, tb)
                py_db.stop_on_unhandled_exception(py_db, thread, additional_info, arg)
            finally:
                additional_info.is_tracing -= 1
    elif os.getenv("MMGUI_DEBUG_EXCEPTIONS"):
        import pdb

        pdb.post_mortem(tb)

    if os.getenv("MMGUI_EXIT_ON_EXCEPTION"):
        print("\nMMGUI_EXIT_ON_EXCEPTION is set, exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
