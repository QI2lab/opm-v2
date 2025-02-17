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

    # load hardware configuration file
    config_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\opm_config_20250216.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    
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

    # load OPM NIDAQ and OPM AO mirror classes
    opmNIDAQ = OPMNIDAQ(
        name = str(config["NIDAQ"]["name"]),
        scan_type = str(config["NIDAQ"]["scan_type"]),
        exposure_ms = float(config["Camera"]["exposure_ms"]),
        laser_blanking = bool(config["NIDAQ"]["laser_blanking"]),
        image_mirror_calibration = float(str(config["NIDAQ"]["image_mirror_calibration"])),
        projection_mirror_calibration = float(str(config["NIDAQ"]["projection_mirror_calibration"])),
        image_mirror_step_size_um = float(str(config["NIDAQ"]["image_mirror_step_size_um"])),
        verbose = bool(config["NIDAQ"]["verbose"])
    )
    opmNIDAQ.reset()


    # grab mmc instance and load OPM config file
    mmc = win.mmcore
    mmc.loadSystemConfiguration(Path(config["mm_config_path"]))
    mmc.setProperty("OrcaFusionBT", "Exposure", float(config["Camera"]["exposure_ms"]))


    def update_state(config_name: str, config_state: str):

        opmNIDAQ_update_state = OPMNIDAQ.instance()

        restart_sequence = False
        if mmc.isSequenceRunning():
            mmc.stopSequenceAcquisition()
            restart_sequence = True

        if opmNIDAQ_update_state.running():
            opmNIDAQ_update_state.stop_waveform_playback()

        exposure_ms = np.round(float(mmc.getProperty("OrcaFusionBT", "Exposure")),0)        

        if config_name == "CameraCrop-Y":

            ROI_size_y = int(mmc.getProperty("ImageCameraCrop","Label"))

            mmc.setROI(
                config["Camera"]["camera_center_x"],
                config["Camera"]["camera_center_y"],
                config["Camera"]["camera_crop_x"],
                ROI_size_y
            )

        elif config_name == "ImageGalvoMirrorRange":
            image_mirror_range_um = np.round(float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),0)
            opmNIDAQ_update_state.image_mirror_sweep_um = image_mirror_range_um

        elif config_name == "ImageGalvoMirrorStep":
            image_mirror_step_um = np.round(float(mmc.getProperty("ImageGalvoMirrorStep", "Label"),0))
            opmNIDAQ_update_state.image_mirror_step_size_um = image_mirror_step_um

        elif config_name == "LED":
            active_channel = mmc.getProperty("LED", "Label")
            channel_states = [False,False,False,False,False]
            for ch_i, ch_str in enumerate(["405nm", "488nm", "561nm", "637nm", "730nm"]):
                if active_channel==ch_str:
                    channel_states[ch_i] = True
            opmNIDAQ_update_state.channel_states = channel_states

        elif config_name == "OPM-mode":
            opm_mode = mmc.getProperty("OPM-mode")
            
            active_channel = mmc.getProperty("LED", "Label")
            channel_states = [False,False,False,False,False]
            for ch_i, ch_str in enumerate(["405nm", "488nm", "561nm", "637nm", "730nm"]):
                if active_channel==ch_str:
                    channel_states[ch_i] = True
            image_mirror_sweep_um = np.round(float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),0)
            image_mirror_step_um = np.round(float(mmc.getProperty("ImageGalvoMirrorStep", "Label"),0))
            if "On" in mmc.getProperty("LaserBlanking","Label"):
                laser_blanking = True
            elif "Off" in mmc.getProperty("LaserBlanking","Label"):
                laser_blanking = False
            else:
                laser_blanking = True

            if opm_mode == "0-Standard" or opm_mode == "2-Stage":
                opmNIDAQ_update_state.set_acquisition_params(
                    scan_type="2d",
                    channel_states=channel_states,
                    image_mirror_step_size_um=image_mirror_step_um,
                    image_mirror_sweep_um=image_mirror_sweep_um,
                    laser_blanking=laser_blanking,
                    exposure_ms=exposure_ms,
                )
            elif opm_mode == "1-projection":
                opmNIDAQ_update_state.set_acquisition_params(
                    scan_type="projection",
                    channel_states=channel_states,
                    image_mirror_step_size_um=image_mirror_step_um,
                    image_mirror_sweep_um=image_mirror_sweep_um,
                    laser_blanking=laser_blanking,
                    exposure_ms=exposure_ms,
                )

        mmc.setProperty("OrcaFusionBT", "Exposure", exposure_ms)

        if restart_sequence:
            mmc.startContinuousSequenceAcquisition()

    mmc.events.configSet.connect(update_state)


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

        opmAOmirror_local = AOMirror.instance()

        # get image galvo mirror range and step size
        image_mirror_range_um = np.round(float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),0)
        image_mirror_step_um = np.round(float(mmc.getProperty("ImageGalvoMirrorStep", "Label"),0))

        # get AO mode
        if "System-correction" in mmc.getProperty("AO-mode", "Label"):
            AO_mode = "System-correction"
        elif "Before-each-XYZ" in mmc.getProperty("AO-mode", "Label"):
            AO_mode = "Before-each-xyz"
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
        opmAOmirror_local.n_positions = n_stage_pos
                
        time_plan = sequence_dict["time_plan"]
        if time_plan is not None:
            n_time_steps = time_plan["loops"]
            time_interval = time_plan["interval"]
        else:
            n_time_steps = 1
            time_interval = 0
    
        channels = sequence_dict["channels"]
        channel_names = ["405nm","488nm","561nm","637nm","730nm"]
        active_channels = [False,False,False,False,False]
        exposure_channels = [0.,0.,0.,0.,0.,0.]
        laser_powers = [0.,0.,0.,0.,0.]
        for chan_idx, channel in enumerate(channels):
            if channel_names[chan_idx] in channel["config"]:
                active_channels[chan_idx] = True
                exposure_channels[chan_idx] = channel["exposure"]
                laser_powers[chan_idx] = float(
                    mmc.getConfigState(
                        "Coherent-Scientific Remote",
                        "Laser-"+str(channel_names[chan_idx])+"-power"
                    )
                )

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

        # reload hardware configuration file before setting up acq
        config_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\opm_config_20250216.json")
        with open(config_path, "r") as config_file:
            updated_config = json.load(config_file)

        opm_events: list[MDAEvent] = []

        # run O2-O3 autofocus before acquisition starts
        O2O3_event = MDAEvent(
            action=CustomAction(
                name="O2O3-autofocus",
                data = {
                    "Camera" : {                    
                        "exposure_ms" : float(updated_config["O2O3-autofocus"]["exposure_ms"]),
                        "camera_crop" : [
                            updated_config["Camera"]["camera_center_x"],
                            updated_config["Camera"]["camera_center_y"],
                            updated_config["Camera"]["camera_crop_x"],
                            updated_config["O2O3-autofocus"]["camera_crop_y"]
                        ]
                    }
                }
            )
        )
        opm_events.append(O2O3_event)

        # check OPM mode and create CustomAction DAQ event
        if "Projection" in mmc.getProperty("OPM-mode", "Label"):
            n_scan_steps = 1
            interleaved_acq = False
            daq_mode = "projection"
            laser_blanking = True
          
            DAQ_event = MDAEvent(
                action=CustomAction(
                    name="DAQ-projection",
                    data = {
                        "DAQ" : {
                            "mode" : daq_mode,
                            "image_mirror_step_um" : float(image_mirror_step_um),
                            "image_mirror_range_um" : float(image_mirror_range_um),
                            "active_channels" : active_channels,
                            "interleaved" : interleaved_acq,
                            "laser_powers" : laser_powers,
                            "blanking" : laser_blanking, 
                        },
                        "Camera" : {
                            "camera_crop" : [
                                updated_config["Camera"]["camera_center_x"],
                                updated_config["Camera"]["camera_center_y"],
                                updated_config["Camera"]["camera_crop_x"],
                                int(mmc.getProperty("ImageCameraCrop","Label"))
                            ]
                        }
                    }
                )
            )
        elif  "Mirror" in mmc.getProperty("OPM-mode", "Label"):
            daq_mode = "mirror_sweep"
            DAQ_event = MDAEvent(
                action=CustomAction(
                    name="DAQ-mirror",
                    data = {
                        "DAQ" : {
                            "mode" : daq_mode,
                            "image_mirror_step_um" : float(image_mirror_step_um),
                            "image_mirror_range_um" : float(image_mirror_range_um),
                            "active_channels" : active_channels,
                            "interleaved" : interleaved_acq,
                            "laser_powers" : laser_powers,
                            "blanking" : laser_blanking,
                        },
                        "Camera" : {
                            "camera_crop" : [
                                updated_config["Camera"]["camera_center_x"],
                                updated_config["Camera"]["camera_center_y"],
                                updated_config["Camera"]["camera_crop_x"],
                                int(mmc.getProperty("ImageCameraCrop","Label"))
                            ]
                        }
                    }
                )
            )

        # setup AO using values in config.json
        if not(AO_mode == "Optimize-now"):
            # Create AO event
            AO_event = MDAEvent(
                exposure = float(updated_config["AO-projection"]["exposure_ms"]),
                action=CustomAction(
                    name="AO-projection",
                    data = {
                        "AO" : {
                            "opm_mode": str("projection"),
                            "active_channels": list(map(bool,updated_config["AO-projection"]["active_channels"])),
                            "laser_power" : list(float(updated_config["AO-projection"]["laser_power"])),
                            "mode": str(updated_config["AO-projection"]["mode"]),
                            "iterations": int(updated_config["AO-projection"]["iterations"]),
                            "image_mirror_step_um" : float(updated_config["AO-projection"]["image_mirror_step_um"]),
                            "image_mirror_range_um" : float(updated_config["AO-projection"]["image_mirror_range_um"]),
                            "blanking": bool(True),
                        },
                        "Camera" : {
                            "exposure_ms": float(updated_config["AO-projection"]["exposure_ms"]),
                            "camera_crop" : [
                                updated_config["Camera"]["camera_center_x"],
                                updated_config["Camera"]["camera_center_y"],
                                updated_config["Camera"]["camera_crop_x"],
                                int(updated_config["AO-projection"]["camera_crop_y"])
                            ]
                        }
                    }
                )
            )

        # setup AO using values in GUI
        else:
            active_channel = mmc.getProperty("LED", "Label")
            AO_exposure_ms = np.round(float(mmc.getProperty("OrcaFusionBT", "Exposure")),0)        
            AO_channel_states = [False,False,False,False,False]
            AO_laser_powers = [0.,0.,0.,0.,0.]
            for chan_idx, chan_str in enumerate(["405nm", "488nm", "561nm", "637nm", "730nm"]):
                if active_channel==chan_str:
                    AO_channel_states[chan_idx] = True
                    AO_laser_powers[chan_idx] = float(
                        mmc.getConfigState(
                            "Coherent-Scientific Remote",
                            "Laser-"+str(active_channel)+"-power"
                        )
                    )
            
            # Create AO event
            AO_event = MDAEvent(
                exposure = AO_exposure_ms,
                action=CustomAction(
                    name="AO-projection",
                    data = {
                        "AO" : {
                            "opm_mode": str("projection"),
                            "active_channels": AO_channel_states,
                            "laser_power" : AO_laser_powers,
                            "exposure_ms": AO_exposure_ms,
                            "mode": str(updated_config["AO-projection"]["mode"]),
                            "iterations": int(updated_config["AO-projection"]["iterations"]),
                            "image_mirror_step_um" : float(image_mirror_step_um),
                            "image_mirror_range_um" : float(image_mirror_range_um),
                            "blanking": bool(True),
                        },
                        "Camera" : {
                            "camera_crop" : [
                                updated_config["Camera"]["camera_center_x"],
                                updated_config["Camera"]["camera_center_y"],
                                updated_config["Camera"]["camera_crop_x"],
                                int(mmc.getProperty("ImageCameraCrop","Label"))
                            ]
                        }
                    }
                )
            )
            
            updated_config["AO-projection"]["active_channels"] = AO_channel_states
            updated_config["AO-projection"]["laser_power"] = AO_laser_powers
            updated_config["AO-projection"]["exposure_ms"] = AO_exposure_ms
            updated_config["AO-projection"]["image_mirror_step_um"] = image_mirror_step_um
            updated_config["AO-projection"]["image_mirror_range_um"] = image_mirror_range_um
            updated_config["AO-projection"]["camera_crop_y"] = int(mmc.getProperty("ImageCameraCrop","Label"))

            with open(config_path, "w") as file:
                json.dump(updated_config, file, indent=4)

            opm_events.append(O2O3_event)
            mda_widget._mmc.run_mda(opm_events, output=output)

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
                        x_pos = stage_positions[pos_idx]["x"],
                        y_pos = stage_positions[pos_idx]["y"],
                        z_pos = stage_positions[pos_idx]["z"],
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
                    current_AO_event = AO_event.model_copy()
                    current_AO_event.action.data["AO"]["pos_idx"] = pos_idx
                    current_AO_event.action.data["AO"]["apply_existing"] = False
                    opm_events.append(current_AO_event)
                # Apply correction for this position if time_idx > 0
                elif AO_mode == "Before-each-xyz" and time_idx > 0:
                    need_to_setup_DAQ = True
                    current_AO_event = AO_event.model_copy()
                    current_AO_event.action.data["AO"]["pos_idx"] = pos_idx
                    current_AO_event.action.data["AO"]["apply_existing"] = True
                    opm_events.append(current_AO_event)
                # Otherwise, run AO opt. before every acquisition. COSTLY in time and photons!
                elif AO_mode == "Before-every-acq":
                    need_to_setup_DAQ = True
                    current_AO_event = AO_event.model_copy()
                    current_AO_event.action.data["AO"]["pos_idx"] = pos_idx
                    current_AO_event.action.data["AO"]["apply_existing"] = False
                    opm_events.append(current_AO_event)
                # Finally, handle acquiring images. 
                # These events are passed through to the normal MDAEngine and *should* be sequenced. 
                if interleaved_acq:
                    # Setup DAQ for acquisition
                    # NOTE: should be smart about this. We should only setup the DAQ as needed.
                    if need_to_setup_DAQ:
                        need_to_setup_DAQ = False
                        opm_events.append(DAQ_event)
                    for scan_idx in range(n_scan_steps):
                        for chan_idx in range(n_active_channels):
                            image_event = MDAEvent(
                                exposure=np.unique(exposure_channels),
                                channel = channel_names,
                                x_pos = stage_positions[pos_idx]["x"],
                                y_pos = stage_positions[pos_idx]["y"],
                                z_pos = stage_positions[pos_idx]["z"],
                                index=mappingproxy({
                                    "t": time_idx, 
                                    "p": pos_idx, 
                                    "c": chan_idx, 
                                    "z": scan_idx
                                }),
                                metadata = {
                                    "DAQ" : {
                                        "mode" : daq_mode,
                                        "image_mirror_step_um" : float(image_mirror_step_um),
                                        "image_mirror_range_um" : float(image_mirror_range_um),
                                        "active_channels" : active_channels,
                                        "exposure_channels_ms": exposure_channels,
                                        "interleaved" : interleaved_acq,
                                        "laser_powers" : laser_powers,
                                        "blanking" : laser_blanking,
                                    },
                                    "Camera" : {
                                        "exposure_ms" : float(exposure_channels[chan_idx]),
                                        "camera_center_x" : updated_config["Camera"]["camera_center_x"],
                                        "camera_center_y" : updated_config["Camera"]["camera_center_y"],
                                        "camera_crop_x" : updated_config["Camera"]["camera_crop_x"],
                                        "camera_crop_y" : int(mmc.getProperty("ImageCameraCrop","Label"))
                                    },
                                    "OPM" : {
                                        "angle_deg" : float(updated_config["OPM"]["angle_deg"]),
                                        "camera_stage_orientation" : str(updated_config["OPM"]["camera_stage_orientation"]),
                                        "camera_mirror_orientation" : str(updated_config["OPM"]["camera_mirror_orientation"])
                                    }
                                }
                            )
                            opm_events.append(image_event)
                else:
                    temp_channels = [False,False,False,False,False]
                    for chan_idx in range(n_active_channels):
                        if active_channels[chan_idx]:
                            if need_to_setup_DAQ:
                                need_to_setup_DAQ = True
                                current_DAQ_event = DAQ_event.model_copy()
                                temp_channels[chan_idx] = True
                                current_DAQ_event.action.data["DAQ"]["active_channels"] = temp_channels
                                current_DAQ_event.action.data["Camera"]["exposure_ms"] = float(exposure_channels[chan_idx])
                                opm_events.append(current_DAQ_event)
                            for scan_idx in range(n_scan_steps):
                                image_event = MDAEvent(
                                    exposure = exposure_channels[chan_idx],
                                    channel = channel_names,
                                    x_pos = stage_positions[pos_idx]["x"],
                                    y_pos = stage_positions[pos_idx]["y"],
                                    z_pos = stage_positions[pos_idx]["z"],
                                    index=mappingproxy({
                                        "t": time_idx, 
                                        "p": pos_idx, 
                                        "c": chan_idx, 
                                        "z": scan_idx
                                    }),
                                    metadata = {
                                        "DAQ" : {
                                            "mode" : daq_mode,
                                            "image_mirror_step_um" : float(image_mirror_step_um),
                                            "image_mirror_range_um" : float(image_mirror_range_um),
                                            "active_channels" : active_channels,
                                            "exposure_channels_ms": exposure_channels,
                                            "interleaved" : interleaved_acq,
                                            "laser_powers" : laser_powers,
                                            "blanking" : laser_blanking,
                                        },
                                        "Camera" : {
                                            "exposure_ms" : float(exposure_channels[chan_idx]),
                                            "camera_center_x" : updated_config["Camera"]["camera_center_x"],
                                            "camera_center_y" : updated_config["Camera"]["camera_center_y"],
                                            "camera_crop_x" : updated_config["Camera"]["camera_crop_x"],
                                            "camera_crop_y" : int(mmc.getProperty("ImageCameraCrop","Label"))
                                        },
                                        "OPM" : {
                                            "angle_deg" : float(updated_config["OPM"]["angle_deg"]),
                                            "camera_stage_orientation" : str(updated_config["OPM"]["camera_stage_orientation"]),
                                            "camera_mirror_orientation" : str(updated_config["OPM"]["camera_mirror_orientation"])
                                        }
                                    }
                                )
                                opm_events.append(image_event)

        print(opm_events)
        
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

    # Register the custom OPM MDA engine with mmc
    mmc.mda.set_engine(OPMENGINE(mmc))

    # This section sets up a callback to intercept the preview mode 
    # and setup the OPM accordingly.
    def setup_preview_mode_callback():
        """Callback to intercept preview mode and setup the OPM.
        
        This function parses the various configuration groups and creates
        the appropiate NIDAQ waveforms for the selected OPM mode and channel.
        """

        # get instance of opmnidaq here
        opmNIDAQ_setup_preview = OPMNIDAQ.instance()

        if opmNIDAQ_setup_preview.running():
            opmNIDAQ_setup_preview.stop_waveform_playback()
            
        # check if any channels are active. If not, don't setup DAQ.
        if any(opmNIDAQ_setup_preview.channel_states):
            # Check OPM mode and set up NIDAQ accordingly
            opmNIDAQ_setup_preview.clear_tasks()
            opmNIDAQ_setup_preview.generate_waveforms()
            opmNIDAQ_setup_preview.prepare_waveform_playback()
            opmNIDAQ_setup_preview.start_waveform_playback()

            
    # Connect the above callback to the event that a continuous sequence is starting
    # Because callbacks are blocking, our custom setup code is called before the preview mode starts. 
    mmc.events.continuousSequenceAcquisitionStarting.connect(setup_preview_mode_callback)



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
        
    