"""qi2lab modified version of the launching script for pymmcore-gui.

qi2lab specific changes start on ~ line 112.

Change Log:
2025/02: Initial version of the script.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
import traceback
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication
from superqt.utils import WorkerBase

from pymmcore_gui import MicroManagerGUI,WidgetAction, __version__
import json

import numpy as np
from useq import MDAEvent, CustomAction
from types import MappingProxyType as mappingproxy

from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.engine.OPMEngine import OPMENGINE

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
    # args = parse_args()

    app = MMQApplication(sys.argv)
    _install_excepthook()

    win = MicroManagerGUI()
    win.setWindowIcon(QIcon(str(ICON)))
    win.showMaximized()
    win.show()

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

    # load OPM NIDAQ and OPM AO mirror classes
    opmNIDAQ = OPMNIDAQ()
    opmNIDAQ.reset()
    
    # load hardware configuration file
    config_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\opm_config.json")
    with open(config_path, 'r') as _:
        config = json.load(_)
    
    # Start the mirror in the flat_position position.
    opmAOmirror = AOMirror(
        wfc_config_file_path = Path(config["AOMirror"]["wfc_config_path"]),
        haso_config_file_path = Path(config["AOMirror"]["haso_config_path"]),
        interaction_matrix_file_path = Path(config["AOMirror"]["wfc_correction_path"]),
        flat_positions_file_path = Path(config["AOMirror"]["wfc_flat_path"]),
        n_modes = 32,
        n_positions=1,
        modes_to_ignore = []
    )
    
    opmAOmirror.set_mirror_flat()

    # grab mmc instance and load OPM config file
    mmc = win.mmcore
    mmc.loadSystemConfiguration(Path(config["mm_config_path"]))      

    # grab handle to the Stage widget
    stage_widget = win.get_widget(WidgetAction.STAGE_CONTROL)

    # grab handle to the MDA widget and define custom execute_mda method
    # in our method, the MDAEvents are modified before running the sequence
    mda_widget = win.get_widget(WidgetAction.MDA_WIDGET)
    
    def custom_execute_mda(output: Path | str | object | None) -> None:
        """Custom execute_mda method that modifies the sequence before running it.
        

        This function parses the various configuration groups and the MDA sequence.
        It then creates a new MDA sequence based on the configuration settings.
        Importantly, we add custom metadata events that trigger the custom parts 
        of our acquistion engine.

        Parameters
        ----------
        output : Path | str | object | None
            The output path for the MDA sequence.
        """

        # get image galvo mirror range and step size
        image_mirror_range_um = np.round(float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),2)
        if "0.4" in mmc.getProperty("ImageGalvoMirrorStep", "Label"):
            image_mirror_step_um = 0.4
        elif "0.8" in mmc.getProperty("ImageGalvoMirrorStep", "Label"):
            image_mirror_step_um = 0.8
        elif "1.6" in mmc.getProperty("ImageGalvoMirrorStep", "Label"):
            image_mirror_step_um = 1.6

        # get AO mode
        if "System-correction" in mmc.getProperty("AO-mode", "Label"):
            AO_mode = 'System-correction'
        elif "Before-each-XYZ" in mmc.getProperty("AO-mode", "Label"):
            AO_mode = 'Before-each-xyz'
        elif "Before-every-acq" in mmc.getProperty("AO-mode", "Label"):
            AO_mode = "Before-every-acq"
        elif "Optimize-now" in mmc.getProperty("AO-mode", "Label"):
            AO_mode = "Optimize-now"

        # get O2-O3 focus mode
        if "Initial-only" in mmc.getProperty("O2O3focus-mode", "Label"):
            O2O3_mode = "Initial-only"
        elif "Before-each-XYZ" in mmc.getProperty("O2O3focus-mode", "Label"):
            O2O3_mode = "Before-each-xyz"
        elif "Before-each-T" in mmc.getProperty("O2O3focus-mode", "Label"):
            O2O3_mode = "Before-each-t"
        elif "After-30min" in mmc.getProperty("O2O3focus-mode", "Label"):
            O2O3_mode = "After-30min"
        elif "None" in mmc.getProperty("O2O3focus-mode", "Label"):
            O2O3_mode = "None"

        # Get the current MDAsequence and convert to dictionary 
        sequence = mda_widget.value()
        sequence_dict = json.loads(sequence.model_dump_json())

        # extract the revelant portions for qi2lab-OPM
        stage_positions = sequence_dict["stage_positions"]
        n_stage_pos = len(stage_positions)
        
        grid_plan = sequence_dict["grid_plan"]
        print(grid_plan)
        
        time_plan = sequence_dict["time_plan"]
        if time_plan is not None:
            n_time_steps = time_plan["loops"]
            time_interval = time_plan["interval"]
        else:
            n_time_steps = 1
            time_interval = 0
        
        AO_exposure_ms = 500.
    
        channels = sequence_dict["channels"]
        channel_names = ["405nm","488nm","561nm","637nm","730nm"]
        active_channels = [False,False,False,False,False]
        exposure_channels = [0.,0.,0.,0.,0.,0.]
        laser_powers = [0.,0.,0.,0.,0.]
        for chan_idx, channel in enumerate(channels):
            if channel_names[chan_idx] in channel['config']:
                active_channels[chan_idx] = True
                exposure_channels[chan_idx] = channel['exposure'] # need to change to np.round(float(mmc.getProperty("")))
                laser_powers[chan_idx] = np.round(float(mmc.getProperty(laser_box_name,"PowerSetpoint (%)")),1)

        if len(set(exposure_channels)) == 1:
            interleaved_acq = True
        else:
            interleaved_acq = False
        n_active_channels = sum(active_channels)
        
        if "On" in mmc.getProperty("LaserBlanking","Label"):
            laser_blanking = True
        elif "Off" in mmc.getProperty("LaserBlanking","Label"):
            laser_blanking = False
        else:
            laser_blanking = True

        n_scan_steps = int(np.ceil(image_mirror_range_um/image_mirror_step_um))

        O2O3_exposure_ms = 10.

        opm_events: list[MDAEvent] = []

        # run O2-O3 autofocus before acquisition starts
        exposure_event = MDAEvent(
            exposure = O2O3_exposure_ms
        )
        opm_events.append(exposure_event)
        O2O3_event = MDAEvent(
            action=CustomAction(
                name="O2O3-autfocus",
                data = {
                    "O2O3-mode" : "autofocus",
                    "camera-exposure_ms" : 10
                }
            )
        )
        opm_events.append(O2O3_event)

        # check OPM mode and create CustomAction DAQ event
        if "Projection" in mmc.getProperty("OPM-mode", "Label"):
            n_scan_steps = 1
            interleaved_acq = False
            daq_mode = "projection"
          
            DAQ_event = MDAEvent(
                action=CustomAction(
                    name="DAQ-projection",
                    data = {
                        'DAQ-mode' : daq_mode,
                        'DAQ-image_mirror_step_um' : float(image_mirror_step_um),
                        'DAQ-image_mirror_range_um' : float(image_mirror_range_um),
                        'DAQ-active_channels' : active_channels,
                        "DAQ-exposure_channels_ms": exposure_channels,
                        "DAQ-interleaved" : interleaved_acq,
                        "DAQ-laser_powers" : [0,1,2,3,5],
                        "DAQ-blanking" : bool(True),
                        "camera-crop" : [0,0,2048,2048]
                    }
                )
            )
        elif  "Mirror" in mmc.getProperty("OPM-mode", "Label"):
            daq_mode = "mirror_sweep"
            DAQ_event = MDAEvent(
                action=CustomAction(
                    name="DAQ-mirror",
                    data = {
                        'DAQ-mode' : daq_mode,
                        'DAQ-image_mirror_step_um' : float(image_mirror_step_um),
                        'DAQ-image_mirror_range_um' : float(image_mirror_range_um),
                        'DAQ-active_channels' : active_channels,
                        "DAQ-exposure_channels_ms": exposure_channels,
                        "DAQ-interleaved" : interleaved_acq,
                        "DAQ-laser_powers" : [0,1,2,3,5],
                        "DAQ-blanking" : bool(True),  
                        "camera-crop": [0,0,2048,2048]
                    }
                )
            )

        # Create AO event
        AO_event = MDAEvent(
            exposure = AO_exposure_ms,
            action=CustomAction(
                name="AO-projection",
                data = {
                    'AO-opm_mode': str('projection'),
                    'AO-active_channels_bool': [False,True,False,False,False],
                    "AO-laser_power" : [0,75,0,0,0],
                    'AO-exposure_ms': AO_exposure_ms, # replace with AO exposure time
                    'AO-mode': str('dct'),
                    'AO-iterations': int(5),
                    "AO-image_mirror_range_um" : float(50.),
                    "AO-image_mirror_step_um" : float(0.4),
                    "AO-blanking": bool(True),
                    "camera-crop": [0,0,2048,2048]
                }
            )
        )

        # Flags to help ensure sequence-able events are kept together 
        need_to_setup_DAQ = True
        need_to_setup_stage = True

        # setup nD mirror-based AO-OPM acquisition event structure
        for time_idx in range(n_time_steps):
            # Check if autofocus before each timepoint and not initial-only mode
            if O2O3_mode == "Before-each-t" and not(O2O3_mode == "Initial-only"):
                opm_events.append(O2O3_event)
            for pos_idx in range(n_stage_pos):
                if need_to_setup_stage:
                    stage_event = MDAEvent(
                        x_pos = stage_positions[pos_idx]['x'],
                        y_pos = stage_positions[pos_idx]['y'],
                        z_pos = stage_positions[pos_idx]['z'],
                    )
                    opm_events.append(stage_event)
                    if n_stage_pos > 1:
                        need_to_setup_stage = True
                    else:
                        need_to_setup_stage = False
                # Check if autofocus before each XYZ position and not initial-only mode
                if O2O3_mode == "Before-each-xyz" and not(O2O3_mode == "Initial-only"):
                    opm_events.append(O2O3_event)
                # Check if run AO opt. before each XYZ on first time we see this position
                if AO_mode == "Before-each-xyz" and time_idx == 0:
                    need_to_setup_DAQ = True
                    opm_events.append(AO_event)
                # Otherwise, run AO opt. before every acquisition. COSTLY in time and photons!
                elif AO_mode == "Before-every-acq":
                    need_to_setup_DAQ = True
                    opm_events.append(AO_event)
                # Setup DAQ for acquisition
                # NOTE: should be smart about this. We should only setup the DAQ as needed.
                if need_to_setup_DAQ:
                    need_to_setup_DAQ = False
                    opm_events.append(DAQ_event)
                # Finally, handle acquiring images. 
                # These events are passed through to the normal MDAEngine and *should* be sequenced. 
                if interleaved_acq:
                    for scan_idx in range(n_scan_steps):
                        for chan_idx in range(n_active_channels):
                            image_event = MDAEvent(
                                exposure=np.unique(exposure_channels),
                                channel = channel_names,
                                x_pos = stage_positions[pos_idx]['x'],
                                y_pos = stage_positions[pos_idx]['y'],
                                z_pos = stage_positions[pos_idx]['z'],
                                index=mappingproxy({
                                    't': time_idx, 
                                    'p': pos_idx, 
                                    'c': chan_idx, 
                                    'z': scan_idx
                                }),
                                metadta = {
                                        'DAQ-mode' : daq_mode,
                                        'DAQ-image_mirror_step_um' : float(image_mirror_step_um),
                                        'DAQ-image_mirror_range_um' : float(image_mirror_range_um),
                                        'DAQ-active_channels' : active_channels,
                                        "DAQ-exposure_channels_ms": exposure_channels,
                                        "DAQ-interleaved" : interleaved_acq,
                                        "DAQ-laser_powers" : [0,1,2,3,5],
                                        "DAQ-blanking" : bool(True),
                                        "camera-crop" : [0,0,2048,2048]
                                }
                            )
                            opm_events.append(image_event)
                else:
                    for chan_idx in range(n_active_channels):
                        if active_channels[chan_idx]:
                            for scan_idx in range(n_scan_steps):
                                image_event = MDAEvent(
                                    exposure = exposure_channels[chan_idx],
                                    channel = channel_names,
                                    x_pos = stage_positions[pos_idx]['x'],
                                    y_pos = stage_positions[pos_idx]['y'],
                                    z_pos = stage_positions[pos_idx]['z'],
                                    index=mappingproxy({
                                        't': time_idx, 
                                        'p': pos_idx, 
                                        'c': chan_idx, 
                                        'z': scan_idx
                                    }),
                                    metadta = {
                                        'DAQ-mode' : daq_mode,
                                        'DAQ-image_mirror_step_um' : float(image_mirror_step_um),
                                        'DAQ-image_mirror_range_um' : float(image_mirror_range_um),
                                        'DAQ-active_channels' : active_channels,
                                        "DAQ-exposure_channels_ms": exposure_channels,
                                        "DAQ-interleaved" : interleaved_acq,
                                        "DAQ-laser_powers" : [0,1,2,3,5],
                                        "DAQ-blanking" : bool(True),
                                        "camera-crop" : [0,0,2048,2048]
                                    }
                                )
                                opm_events.append(image_event)

        # print(opm_events)
        
        # elif "Stage" in mmc.getProperty("OPM-mode", "Label"):
        #     print("stage mode")

        # # check fluidics mode
        # if "None" in mmc.getProperty("Fluidics-mode", "Label"):
        #     print("No fluidics")
        # elif "Thin-16bit" in mmc.getProperty("Fluidics-mode", "Label"):
        #     print("Thin 16bit fluidics")
        # elif "Thin-22bit" in mmc.getProperty("Fluidics-mode", "Label"):
        #     print("Thin 22bit fluidics")
        # elif "Thick-16bit" in mmc.getProperty("Fluidics-mode", "Label"):
        #     print("Thick 16bit fluidics")
        # elif "Thick-2bit" in mmc.getProperty("Fluidics-mode", "Label"):
        #     print("Thick 22bit fluidics")

        mda_widget._mmc.run_mda(opm_events, output=output)

    # modify the method on the instance
    mda_widget.execute_mda = custom_execute_mda

    # This section sets up a callback to intercept the preview mode 
    # and setup the OPM accordingly.
    def setup_preview_mode_callback():
        """Callback to intercept preview mode and setup the OPM.
        
        This function parses the various configuration groups and creates
        the appropiate NIDAQ waveforms for the selected OPM mode and channel.
        """

        # get instance of opmnidaq here
        opmNIDAQ_local = OPMNIDAQ.instance()

        # get image galvo mirror range and step size
        image_mirror_range_um = np.round(float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),2)
        print(image_mirror_range_um)
        if "0.4" in mmc.getProperty("ImageGalvoMirrorStep", "Label"):
            image_mirror_step_um = 0.4
        elif "0.8" in mmc.getProperty("ImageGalvoMirrorStep", "Label"):
            image_mirror_step_um = 0.8
        elif "1.6" in mmc.getProperty("ImageGalvoMirrorStep", "Label"):
            image_mirror_step_um = 1.6        
        
        # get active channel    
        active_channel = mmc.getProperty("LED", "Label")
       
        exposure_ms = np.round(float(mmc.getProperty("OrcaFusionBT", "Exposure")),2)        
        channel_states = [False,False,False,False,False]
        for ch_i, ch_str in enumerate(["405nm", "488nm", "561nm", "637nm", "730nm"]):
            if active_channel==ch_str:
                channel_states[ch_i] = True
        if "On" in mmc.getProperty("LaserBlanking","Label"):
            laser_blanking = True
        elif "Off" in mmc.getProperty("LaserBlanking","Label"):
            laser_blanking = False
        else:
            laser_blanking = True
        
        # check if any channels are active. If not, don't setup DAQ.
        if any(channel_states):
            # Check OPM mode and set up NIDAQ accordingly
            opmNIDAQ_local.stop_waveform_playback()
            opmNIDAQ_local.clear_tasks()
            if "Projection" in mmc.getProperty("OPM-mode", "Label"):
                _ = opmNIDAQ_local.set_acquisition_params(
                    scan_type="projection",
                    channel_states=channel_states,
                    image_mirror_step_size_um=image_mirror_step_um,
                    image_mirror_sweep_um=image_mirror_range_um,
                    laser_blanking=laser_blanking,
                    exposure_ms=exposure_ms
                )

                opmNIDAQ_local.generate_waveforms()
            else:
                opmNIDAQ_local.set_acquisition_params(
                    scan_type="2d",
                    channel_states=channel_states,
                    laser_blanking=laser_blanking
                    )
                opmNIDAQ_local.generate_waveforms()
        
            opmNIDAQ_local.prepare_waveform_playback()
            opmNIDAQ_local.start_waveform_playback()
        else:
            opmNIDAQ_local.stop_waveform_playback()
            opmNIDAQ_local.clear_tasks()
            
    # Connect the above callback to the event that a continuous sequence is starting
    # Because callbacks are blocking, our custom setup code is called before the preview mode starts. 
    mmc.events.continuousSequenceAcquisitionStarting.connect(setup_preview_mode_callback)

    # Register the custom OPM MDA engine with mmc
    mmc.mda.set_engine(OPMENGINE(mmc))

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
        
    