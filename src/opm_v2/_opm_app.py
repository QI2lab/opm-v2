"""qi2lab modified version of the launching script for pymmcore-gui.

qi2lab specific changes start on ~ line 112.

Change Log:
2025/02: Initial version of the script.
"""

from __future__ import annotations

# TODO AO getting values from gui instead of json.
import argparse
import importlib
import importlib.util
import json
import os
import sys
import traceback
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from pymmcore_gui import MicroManagerGUI, WidgetAction, __version__
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QDockWidget
from superqt.utils import WorkerBase

from opm_v2._update_config_widget import OPMSettings
from opm_v2.engine.opm_engine import OPMEngine
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
if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    ExcTuple = tuple[type[BaseException], BaseException, TracebackType | None]


APP_NAME = "Micro-Manager GUI"
APP_VERSION = __version__
ORG_NAME = "pymmcore-plus"
ORG_DOMAIN = "pymmcore-plus"
APP_ID = f"{ORG_DOMAIN}.{ORG_NAME}.{APP_NAME}.{APP_VERSION}"
RESOURCES = Path(__file__).parent / "resources"
ICON = RESOURCES / ("icon.ico" if sys.platform.startswith("win") else "logo.png")
IS_FROZEN = getattr(sys, "frozen", False)


class MMQApplication(QApplication):
    exceptionRaised = pyqtSignal(BaseException)

    def __init__(self, argv: list[str]) -> None:
        if sys.platform == "darwin" and not argv[0].endswith("mmgui"):
            # Make sure the app name in the Application menu is `mmgui`
            # which is taken from the basename of sys.argv[0]; we use
            # a copy so the original value is still available at sys.argv
            argv[0] = "napari"

        super().__init__(argv)
        self.setApplicationName("Micro-Manager GUI")
        self.setWindowIcon(QIcon(str(ICON)))

        self.setApplicationName(APP_NAME)
        self.setApplicationVersion(APP_VERSION)
        self.setOrganizationName(ORG_NAME)
        self.setOrganizationDomain(ORG_DOMAIN)
        if os.name == "nt" and not IS_FROZEN:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)  # type: ignore

        self.aboutToQuit.connect(WorkerBase.await_workers)


def parse_args(args: Sequence[str] = ()) -> argparse.Namespace:
    if not args:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Enter string")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Config file to load",
        nargs="?",
    )
    return parser.parse_args(args)


def main() -> None:
    """Run the Micro-Manager GUI."""
    args = parse_args()

    app = MMQApplication(sys.argv)
    _install_excepthook()

    win = MicroManagerGUI()
    win.setWindowIcon(QIcon(str(ICON)))
    win.showMaximized()

    splsh = "_PYI_SPLASH_IPC" in os.environ and importlib.util.find_spec("pyi_splash")
    if splsh:  # pragma: no cover
        import pyi_splash  # pyright: ignore [reportMissingModuleSource]

        pyi_splash.update_text("UI Loaded ...")
        pyi_splash.close()

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # ----------------Begin custom qi2lab code for running OPM control----------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    # load OPM configuration file
    config_path = (
        Path(args.config)
        if args.config
        else Path(r"C:\Users\qi2lab\Documents\github\opm_v2\opm_config.json")
    )
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    # Load the OPM widget for interacting with the configuration.
    OPMwidget = OPMSettings(config_path=config_path)
    dock_widget = QDockWidget("OPM Settings", win)
    dock_widget.setWidget(OPMwidget)
    dock_widget.setObjectName("OPMConfigurator")
    win.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_widget)
    dock_widget.setFloating(False)

    # --------------------------------------------------------------------------------
    # Initialize Custom hardware drivers
    # --------------------------------------------------------------------------------

    # Load DM in system flat state
    opmAOmirror = AOMirror(
        wfc_config_file_path=Path(config["AOMirror"]["wfc_config_path"]),
        haso_config_file_path=Path(config["AOMirror"]["haso_config_path"]),
        interaction_matrix_file_path=Path(config["AOMirror"]["wfc_correction_path"]),
        system_flat_file_path=Path(config["AOMirror"]["wfc_flat_path"]),
        n_modes=32,
        n_positions=1,
        modes_to_ignore=[],
    )

    # Load NIDAQ
    daq = config["NIDAQ"]
    opmNIDAQ = OPMNIDAQ(
        name=str(daq["name"]),
        scan_type=str(daq["scan_type"]),
        exposure_ms=float(config["Camera"]["exposure_ms"]),
        laser_blanking=bool(daq["laser_blanking"]),
        image_mirror_calibration=float(str(daq["image_mirror_calibration"])),
        projection_mirror_calibration=float(str(daq["projection_mirror_calibration"])),
        image_mirror_neutral_v=float(str(daq["image_mirror_neutral_v"])),
        projection_mirror_neutral_v=float(str(daq["projection_mirror_neutral_v"])),
        image_mirror_step_um=float(str(daq["image_mirror_step_um"])),
        verbose=bool(daq["verbose"]),
    )
    opmNIDAQ.reset()

    # Load o2o3 auto-focus shutter
    opmPicardShutter = PicardShutter(
        int(config["O2O3-autofocus"]["shutter_id"]), verbose=True
    )
    opmPicardShutter.closeShutter()

    # Load OB1 Controller for fluidics control
    ob1_kenobi = OB1Controller()

    # --------------------------------------------------------------------------------
    # Load MMC instance and MMC configuration
    # --------------------------------------------------------------------------------

    mmc = win.mmcore
    mmc.loadSystemConfiguration(Path(config["OPM"]["mm_config_path"]))
    mda_widget = win.get_widget(WidgetAction.MDA_WIDGET)
    mda_widget.save_info.save_dir.setText(r"E:/")
    mda_widget.tab_wdg.grid_plan.setMode("bounds")
    mda_widget.tab_wdg.grid_plan._mode_bounds_radio.toggle()
    stage_widget = win.get_widget(WidgetAction.STAGE_CONTROL)
    config_widget = win.get_widget(WidgetAction.CONFIG_GROUPS)

    if DEBUGGING:
        mmc.enableDebugLog(True)

    # --------------------------------------------------------------------------------
    # Methods for updating live state
    # --------------------------------------------------------------------------------

    def update_config():
        """Updates config from file on disk"""
        with open(config_path, "r") as config_file:
            new_config = json.load(config_file)
        config.update(new_config)

    def apply_camera_state(
        camera_id: str, exposure_ms: float = None, camera_roi: list[int] = None
    ):
        """_summary_

        Parameters
        ----------
        camera_id : str
            _description_
        exposure_ms : float, optional
            Exposure in ms, by default None
        camera_roi : _type_, optional
            [center x, center y, crop x, crop y], by default None
        """
        if camera_roi:
            # Update camera ROI
            _ = mmc.getROI()
            mmc.clearROI()
            mmc.waitForDevice(camera_id)
            mmc.setROI(
                int(camera_roi[0] - camera_roi[2] // 2),
                int(camera_roi[1] - camera_roi[3] // 2),
                int(camera_roi[2]),
                int(camera_roi[3]),
            )
            mmc.waitForDevice(str(camera_id))
        if exposure_ms:
            # Update camera exposure
            if mmc.getExposure() != exposure_ms:
                mmc.setProperty(str(camera_id), "Exposure", exposure_ms)
                mmc.waitForDevice(str(camera_id))

    def apply_daq_state(
        scan_type: str = None,
        channel_states: list[bool] = None,
        exposure_ms: float = None,
        image_mirror_range_um: float = None,
    ):

        opmNIDAQ_update = OPMNIDAQ.instance()
        if opmNIDAQ_update.running():
            opmNIDAQ_update.stop_waveform_playback()

        opmNIDAQ_update.set_acquisition_params(
            scan_type=scan_type,
            channel_states=channel_states,
            image_mirror_range_um=image_mirror_range_um,
            exposure_ms=exposure_ms,
        )

        opmNIDAQ_update.start_waveform_playback()

    def update_live_state(group=None, preset=None):
        """Update microscope states and values upon changes to in the GUI

        Parameters
        ----------
        device_name : str, optional
            signal source device name, by default None
        property_name : str, optional
            signal source propery name, by default None
        """
        if group == mmc.getShutterDevice() and preset == "State":
            return

        update_config()
        opm_mode = mmc.getProperty("OPM-live-mode", "Label")
        camera_id = config["Camera"]["camera_id"]

        restart_sequence = False
        if mmc.isSequenceRunning():
            mmc.stopSequenceAcquisition()
            restart_sequence = True

        if group == "ImageGalvoRange" and "Standard" in opm_mode:
            return

        elif group == "Channel":
            active_id = mmc.getProperty("Laser", "Label")
            ids = config["OPM"]["channel_ids"]
            channel_states = [ch == active_id for ch in ids]
            apply_daq_state(channel_states=channel_states)

        elif group == "Camera-Exposure":
            exposure_ms = round(float(mmc.getProperty(camera_id, "Exposure")), 2)
            if "Projection" in opm_mode and exposure_ms < 50:
                exposure_ms = 100
                print("--- Update Live State: Projection exposure set below 50ms! ----")
            apply_daq_state(exposure_ms=exposure_ms)

        elif group == "OPM-live-mode":
            if preset == "Projection":
                exposure_ms = round(float(mmc.getProperty(camera_id, "Exposure")), 2)
                if exposure_ms < 50:
                    exposure_ms = 100
                image_mirror_range_um = np.round(
                    float(mmc.getProperty("ImageGalvoMirrorRange", "Position")), 0
                )
                camera_roi = list(config["acq_config"]["camera_roi"])
                camera_roi[3] = int(image_mirror_range_um / mmc.getPixelSizeUm())
                apply_camera_state(exposure_ms=exposure_ms,camera_roi=camera_roi)
                apply_daq_state(
                    scan_type="projection", image_mirror_range_um=image_mirror_range_um
                )

            elif preset == "Standard":
                scan_type = "2d"
                apply_daq_state(scan_type=scan_type)

        elif group == "ImageGalvoRange":
            image_mirror_range_um = np.round(
                float(mmc.getProperty("ImageGalvoMirrorRange", "Position")), 0
            )
            apply_daq_state(image_mirror_range_um=image_mirror_range_um)

        elif group == "camera_roi":
            camera_roi = list(config["acq_config"]["camera_roi"])
            if "Projection" in opm_mode:
                image_mirror_range_um = np.round(
                    float(mmc.getProperty("ImageGalvoMirrorRange", "Position")), 0
                )
                camera_roi[3] = int(image_mirror_range_um / mmc.getPixelSizeUm())
            apply_camera_state(camera_id=camera_id, camera_roi=camera_roi)

        if restart_sequence:
            mmc.startContinuousSequenceAcquisition()

    def update_ao_mirror_state():
        """Update the mirror positions to reflect GUI settings"""
        update_config()
        ao_mirror_state = config["acq_config"]["AO"]["mirror_state"]
        AOMirror_update = AOMirror.instance()
        if ao_mirror_state == "system flat":
            AOMirror_update.apply_system_flat_voltage()
        elif ao_mirror_state == "last optimized":
            AOMirror_update.apply_optimized_voltage()
        elif ao_mirror_state == "factory flat":
            AOMirror_update.apply_factory_flat_voltage()
        elif ao_mirror_state == "zeros":
            AOMirror_update.apply_zeros_voltage()

    def setup_preview_mode_callback():
        """Callback to intercept preview mode and setup the OPM.

        This function parses the various configuration groups and creates
        the appropriate NIDAQ waveforms for the selected OPM mode and channel.
        """
        # get instance of opmnidaq here
        opmNIDAQ_setup_preview = OPMNIDAQ.instance()

        if opmNIDAQ_setup_preview.running():
            opmNIDAQ_setup_preview.stop_waveform_playback()

        # check if any channels are active. If not, don't setup DAQ.
        update_live_state()

        if any(opmNIDAQ_setup_preview.channel_states):
            # Check OPM mode and set up NIDAQ accordingly
            opmNIDAQ_setup_preview.clear_tasks()
            opmNIDAQ_setup_preview.generate_waveforms()
            opmNIDAQ_setup_preview.program_daq_waveforms()
            opmNIDAQ_setup_preview.start_waveform_playback()

    # --------------------------------------------------------------------------------
    # Connect GU changes to update functions
    # --------------------------------------------------------------------------------
    mmc.events.continuousSequenceAcquisitionStarting.connect(
        setup_preview_mode_callback
    )
    mmc.events.configSet.connect(update_live_state)
    for key in OPMwidget.widgets["camera_roi"].keys():
        OPMwidget.widgets["camera_roi"][key].editingFinished.connect(
            lambda: update_live_state("camera_roi")
        )
    OPMwidget.widgets["AO"]["mirror_state"].currentIndexChanged.connect(
        update_ao_mirror_state
    )

    # --------------------------------------------------------------------------------
    # Method for executing MDAs
    # --------------------------------------------------------------------------------

    def custom_execute_mda(output: Path | str | None) -> None:
        """Custom execute_mda method that modifies the sequence before running it.

        This function parses the various configuration groups and the MDA sequence.
        It then creates a new MDA sequence based on the configuration settings.
        Importantly, we add custom metadata events that trigger the custom parts
        of our acquistion engine.

        Parameters
        ----------
        output : Path | str | None
            The output path for the MDA sequence.
        """
        opm_events, handler = None, None
        if output is not None:
            if isinstance(output, str):
                output = Path(output)

        # --------------------------------------------------------------------#
        # Get the acquisition settings from configuration on disk
        # --------------------------------------------------------------------#

        update_config()
        opm_mode = config["acq_config"]["opm_mode"]
        ao_mode = config["acq_config"]["AO"]["ao_mode"]
        o2o3_mode = config["acq_config"]["O2O3-autofocus"]["o2o3_mode"]
        fluidics_mode = config["acq_config"]["fluidics"]

        # --------------------------------------------------------------------#
        # Validate acquisition settings
        # --------------------------------------------------------------------#

        if ("now" in ao_mode) or ("now" in o2o3_mode):
            optimize_now = True
            if not output:
                new_output = None
        else:
            optimize_now = False

        if not (optimize_now) and not (output):
            print("Must set acquisition path to excecute acquisition")
            return

        if output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_dir = output.parent / Path(f"{timestamp}_{output.stem}")
            new_dir.mkdir(exist_ok=True)
            new_output = new_dir / Path(output.name)
            output = new_output

        if ("none" not in fluidics_mode) and not (optimize_now):
            # load dialog to have user verify ESI is running.
            # TODO: ad an entry for the number of rounds.
            from PyQt6.QtWidgets import QMessageBox

            response = QMessageBox.information(
                mda_widget,
                "!WARNING! ESI MUST BE RUNNING!",
                "IS ESI SEQUENCE LOADED AND STARTED?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if response is not QMessageBox.StandardButton.Yes:
                return
            else:
                print("ESI Sequence accepted")

        if DEBUGGING:
            print(
                "-----------------------------------------------------",
                "\nQi2lab OPM scan acquisition selected:",
                f"\n  opm_mode: {opm_mode}",
                f"\n  ao_mode: {ao_mode}",
                f"\n  O2O3_mode: {o2o3_mode}",
                f"\n  fluidics_mode: {fluidics_mode}",
                f"\n  output_path: {new_output}\n",
                "-----------------------------------------------------",
            )

        # --------------------------------------------------------------------#
        # Get event structure for requested acquisition type
        # --------------------------------------------------------------------#

        if optimize_now:
            opm_events, handler = setup_optimizenow(mmc, config, output)
        else:
            if output:
                if "timelapse" in opm_mode:
                    opm_events, handler = setup_timelapse(
                        mmc=mmc,
                        config=config,
                        sequence=mda_widget.value(),
                        output=output,
                    )
                elif "stage" in opm_mode:
                    opm_events, handler = setup_stagescan(
                        mmc=mmc,
                        config=config,
                        sequence=mda_widget.value(),
                        output=output,
                    )
                elif "mirror" in opm_mode:
                    opm_events, handler = setup_mirrorscan(
                        mmc=mmc,
                        config=config,
                        sequence=mda_widget.value(),
                        output=output,
                    )
                elif "projection" in opm_mode:
                    opm_events, handler = setup_projection(
                        mmc=mmc,
                        config=config,
                        sequence=mda_widget.value(),
                        output=output,
                    )
                # tell AO mirror class where to save mirror information
                opmAOmirror_local = AOMirror.instance()
                opmAOmirror_local.output_path = output.parents[0]

        if opm_events is None:
            print("OPM events empty, acquisition not started!")
            return
        else:
            mda_widget._mmc.run_mda(opm_events, output=handler)

    # modify the method on the instance
    mda_widget.execute_mda = custom_execute_mda
    # Register the custom OPM MDA engine with mmc
    mmc.mda.set_engine(OPMEngine(mmc, config_path))
    # TODO: Modify this logic to check and set the MDA engine with the option to run
    #       the defualt engine -> MDAEngine

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # -----------------End custom qi2lab code for running OPM control-----------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    app.exec()


# ------------------- Custom excepthook -------------------


def _install_excepthook() -> None:
    """Install a custom excepthook that does not raise sys.exit().

    This is necessary to prevent the application from closing when an exception
    is raised.
    """
    if hasattr(sys, "_original_excepthook_"):
        return
    sys._original_excepthook_ = sys.excepthook  # type: ignore
    sys.excepthook = ndv_excepthook


def rich_print_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
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
    try:
        rich_print_exception(exc_type, exc_value, exc_traceback)
    except ImportError:
        traceback.print_exception(exc_type, value=exc_value, tb=exc_traceback)


# This log list is used by the ExceptionLog widget
# Be aware that it's currently possible for that widget to clear this list.
# If an immutable record of exceptions is needed, additional logic will be required.
EXCEPTION_LOG: list[ExcTuple] = []


def ndv_excepthook(
    exc_type: type[BaseException], exc_value: BaseException, tb: TracebackType | None
) -> None:
    EXCEPTION_LOG.append((exc_type, exc_value, tb))
    _print_exception(exc_type, exc_value, tb)
    if sig := getattr(QApplication.instance(), "exceptionRaised", None):
        sig.emit(exc_value)
    if not tb:
        return

    # if we're running in a vscode debugger, let the debugger handle the exception
    if (
        (debugpy := sys.modules.get("debugpy"))
        and debugpy.is_client_connected()
        and ("pydevd" in sys.modules)
    ):  # pragma: no cover
        with suppress(Exception):
            import threading

            import pydevd  # pyright: ignore [reportMissingImports]

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
    # otherwise, if MMGUI_DEBUG_EXCEPTIONS is set, drop into pdb
    elif os.getenv("MMGUI_DEBUG_EXCEPTIONS"):
        import pdb

        pdb.post_mortem(tb)

    # after handling the exception, exit if MMGUI_EXIT_ON_EXCEPTION is set
    if os.getenv("MMGUI_EXIT_ON_EXCEPTION"):
        print("\nMMGUI_EXIT_ON_EXCEPTION is set, exiting.")
        sys.exit(1)
