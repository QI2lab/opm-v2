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
from opm_v2.hardware.ElveFlow import OB1Controller
from opm_v2.hardware.PicardShutter import PicardShutter
from opm_v2.engine.OPMEngine import OPMENGINE
from pymmcore_plus.mda.handlers import TensorStoreHandler

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
    config_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\opm_config_20250218.json")
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


    # Initialize ElveFlow OB1 Controller
    opmOB1 = OB1Controller(
        port=config["OB1"]["port"],
        to_OB1_pin=config["OB1"]["to_OB1_pin"],
        from_OB1_pin=config["OB1"]["from_OB1_pin"]
    )
    
    opmPicardShutter = PicardShutter(int(config["O2O3-autofocus"]["shutter_id"]))
    opmPicardShutter.closeShutter()
    
    # grab mmc instance and load OPM config file
    mmc = win.mmcore
    mmc.loadSystemConfiguration(Path(config["mm_config_path"]))
    
    # Enforce config's defualt properties
    mmc.setProperty(
        config["Camera"]["camera_id"],
        "Exposure",
        float(config["Camera"]["exposure_ms"])
    )
    mmc.waitForDevice(str(config["Camera"]["camera_id"]))
    
    mmc.clearROI()
    mmc.waitForDevice(str(config["Camera"]["camera_id"]))
    
    mmc.setROI(
        config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
        config["Camera"]["camera_center_y"] - int(config["Camera"]["camera_crop_y"]//2),
        config["Camera"]["camera_crop_x"],
        config["Camera"]["camera_crop_y"]
    )
    mmc.waitForDevice(str(config["Camera"]["camera_id"]))

    def set_projection_roi(_image_mirror_sweep_um):
        # Calculate the the number of Y pixels for the scan range
        roi_height_um = _image_mirror_sweep_um
        roi_height_px = int((roi_height_um / mmc.getPixelSizeUm()) * np.cos(np.deg2rad(30)))
        # TODO: Verify the image stretch factor using SIMpatterns

        # Set the camera ROI to the image sweep range
        if not(roi_height_px == mmc.getROI()[-1]): 
            current_roi = mmc.getROI()
            print(current_roi)
            mmc.clearROI()
            mmc.waitForDevice(str(config["Camera"]["camera_id"]))
            mmc.setROI(
                config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                config["Camera"]["camera_center_y"] - int(roi_height_px//2),
                config["Camera"]["camera_crop_x"],
                roi_height_px
        )
        mmc.waitForDevice(str(config["Camera"]["camera_id"]))
        return roi_height_px
    
    def update_state(*signal_args):
        """
        Update microscope states and values upon changes to in the GUI
        """
        # Stop the DAQ playback if running.
        opmNIDAQ_update_state = OPMNIDAQ.instance()
        restart_sequence = False
        if mmc.isSequenceRunning():
            mmc.stopSequenceAcquisition()
            restart_sequence = True
            
        if opmNIDAQ_update_state.running():
            opmNIDAQ_update_state.stop_waveform_playback()
        
        # get the current operating mode
        # if we are in any mode besides Standard, we must regenerate the waveforms.
        # current_mode = mmc.getProperty("")
        
        # Check the signal arguments to determine which property or config state was changed.
        if len(signal_args)==3:
            # Extract changed property names and values   
            property_name = signal_args[0]
            property_label = signal_args[1]
            property_value = signal_args[2]
            
            if property_name=="OrcaFusionBT" and property_label=="Exposure":
                _exposure_ms = round(float(property_value), 0)
                opmNIDAQ_update_state.exposure_ms = _exposure_ms
            elif property_name == "ImageGalvoMirrorRange":
                _image_mirror_range_um = np.round(float(property_value), 0)
                opmNIDAQ_update_state.image_mirror_sweep_um = _image_mirror_range_um

        elif len(signal_args)==2:
            # update configurations
            config_name = signal_args[0]
            config_state = signal_args[1]
            
            if config_name == "OPM-Mode":
                # If the OPM-Mode changes, update and re-generate the waveforms
                opm_mode = config_state
                                
                # Get DAQ the current exposure
                _exposure_ms = round(float(mmc.getProperty(config["Camera"]["camera_id"], "Exposure")), 0)
                
                # Define the current channel states
                active_channel = mmc.getProperty("LED", "Label")
                _channel_states = [False,False,False,False,False]
                for ch_i, ch_str in enumerate(config["OPM"]["channel_ids"]):
                    if active_channel==ch_str:
                        _channel_states[ch_i] = True
                        
                # Get the current ImageGalvoMirror parameters
                _image_mirror_sweep_um = np.round(float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),0)
                _image_mirror_step_um = np.round(float(mmc.getProperty("ImageGalvoMirrorStep", "Label").split("-")[0]),2)
                
                # Get the current LaserBlanking State
                if mmc.getProperty("LaserBlanking", "Label")=="On":
                    _laser_blanking = True
                else:
                    _laser_blanking = False
                
                # Get the current camera sensor mode
                current_sensor_mode = mmc.getProperty(config["Camera"]["camera_id"], "SENSOR MODE")
                
                # Define the current scan type based on opm-mode
                if "Standard" in opm_mode:
                    _scan_type = "2d"
                    if current_sensor_mode=="PROGRESSIVE":
                        # Configure the camera to run in full chip
                        mmc.setProperty(config["Camera"]["camera_id"], "SENSOR MODE", "AREA")
                        mmc.waitForDevice(str(config["Camera"]["camera_id"]))
    
                elif "Stage" in opm_mode:
                    # TODO
                    _scan_type = "2d"
                    if current_sensor_mode=="PROGRESSIVE":
                        # Configure the camera to run in full chip
                        mmc.setProperty(config["Camera"]["camera_id"], "SENSOR MODE", "AREA")
                        mmc.waitForDevice(str(config["Camera"]["camera_id"]))
                    
                elif "Projection" in opm_mode:
                    _scan_type = "projection"
                    
                    # Calculate the the number of Y pixels for the scan range
                    roi_height_um = _image_mirror_sweep_um
                    roi_height_px = int((roi_height_um / mmc.getPixelSizeUm()) * np.cos(np.deg2rad(30)))
                    # TODO: Verify the image stretch factor using SIMpatterns
            
                    # Set the camera ROI to the image sweep range
                    if not(roi_height_px == mmc.getROI()[-1]): 
                        current_roi = mmc.getROI()
                        print(current_roi)
                        mmc.clearROI()
                        mmc.waitForDevice(str(config["Camera"]["camera_id"]))
                        mmc.setROI(
                            config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                            config["Camera"]["camera_center_y"] - int(roi_height_px//2),
                            config["Camera"]["camera_crop_x"],
                            roi_height_px
                    )
                    mmc.waitForDevice(str(config["Camera"]["camera_id"]))
                    
                    # Configure the camera to run in light sheet mode
                    # mmc.setProperty(config["Camera"]["camera_id"], "SENSOR MODE", "PROGRESSIVE")
                    # mmc.waitForDevice(str(config["Camera"]["camera_id"]))
                    
                    # # Configure the line readout time, units==ms
                    # mmc.setProperty(config["Camera"]["camera_id"], "INTERNAL LINE INTERVAL", config["Camera"]["line_readout_ms"])
                    # mmc.waitForDevice(str(config["Camera"]["camera_id"]))
                    
                    
                    # Configure the exposure
                    # line_readout_time = float(mmc.getProperty(config["Camera"]["camera_id"], "INTERNAL LINE INTERVAL"))
                    # print(f"Line readout {float(line_readout_time)}")
                    # _exposure_ms = roi_height_px * line_readout_time
                    # print(f"calculated exp: {_exposure_ms}")
                    # _exposure_ms =  round(float(mmc.getProperty(config["Camera"]["camera_id"], "Exposure")), 1)
                    # print(f"from property expected exposure: {_exposure_ms}")
                    # _exposure_ms = round(float(mmc.getExposure()), 1)
                    # print(f"from property exposure: {_exposure_ms}")
                    
                    
                elif "Mirror" in opm_mode:
                    _scan_type = "mirror"
                
                    if current_sensor_mode=="PROGRESSIVE":
                        # Configure the camera to run in full chip
                        mmc.setProperty(config["Camera"]["camera_id"], "SENSOR MODE", "AREA")
                        mmc.waitForDevice(str(config["Camera"]["camera_id"]))
                        
                # Set the DAQ acquisition params.
                opmNIDAQ_update_state.set_acquisition_params(
                    scan_type=_scan_type,
                    channel_states=_channel_states,
                    image_mirror_step_size_um=_image_mirror_step_um,
                    image_mirror_sweep_um=_image_mirror_sweep_um,
                    laser_blanking=_laser_blanking,
                    exposure_ms=_exposure_ms,
                )
                # opmNIDAQ_update_state.clear_tasks()
                # opmNIDAQ_update_state.generate_waveforms()
                # opmNIDAQ_update_state.prepare_waveform_playback()
                
            elif config_name == "Channel":
                active_channel = config_state
                channel_states = [False,False,False,False,False]
                for ch_i, ch_str in enumerate(config["OPM"]["channel_ids"]):
                    if active_channel==ch_str:
                        channel_states[ch_i] = True
                opmNIDAQ_update_state.channel_states = channel_states
            elif config_name == "ImageGalvoMirrorStep":
                image_mirror_step_um =  float(config_state.split("-")[0]) 
                opmNIDAQ_update_state.image_mirror_step_size_um = image_mirror_step_um
            elif config_name == "Camera-CropY":
                camera_crop_y = int(config_state.split("-")[0])
                if not(camera_crop_y == mmc.getROI()[-1]): 
                    mmc.clearROI()
                    mmc.waitForDevice(str(config["Camera"]["camera_id"]))
                    mmc.setROI(
                        config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                        config["Camera"]["camera_center_y"] - int(camera_crop_y//2),
                        config["Camera"]["camera_crop_x"],
                        camera_crop_y
                    )
                    mmc.waitForDevice(str(config["Camera"]["camera_id"]))
        
        
        # Restart acquisition if needed
        if restart_sequence:
            opmNIDAQ_update_state.start_waveform_playback()
            mmc.startContinuousSequenceAcquisition()
            
    # Connect changes in gui fields to the update_state method.            
    mmc.events.propertyChanged.connect(update_state)
    mmc.events.configSet.connect(update_state)
    
    # grab handle to the Stage widget
    # stage_widget = win.get_widget(WidgetAction.STAGE_CONTROL)

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
        opmNIDAQ_custom = OPMNIDAQ.instance()

        #--------------------------------------------------------------------#
        # Get the acquisition parameters
        #--------------------------------------------------------------------#
        
        # get image galvo mirror range and step size
        image_mirror_range_um = np.round(float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),0)
        image_mirror_step_um = np.round(float(mmc.getProperty("ImageGalvoMirrorStep", "Label").split("-")[0]),2)
        
        # use OPMNIDAQ class calculation for number of scan steps to ensure consistency
        opmNIDAQ_custom.set_acquisition_params(
            scan_type="mirror",
            channel_states = [False,False,False,False,False],
            image_mirror_step_size_um=image_mirror_step_um,
            image_mirror_sweep_um=image_mirror_range_um,
            laser_blanking=True,
            exposure_ms=100.
        )
        n_scan_steps = opmNIDAQ_custom.n_scan_steps

        # get AO mode / interval
        AO_mode = ""
        if "System-correction" in mmc.getProperty("AO-mode", "Label"):
            AO_mode = "System-correction"
        elif "Before-each-XYZ" in mmc.getProperty("AO-mode", "Label"):
            AO_mode = "Before-each-xyz"
        elif "Before-every-acq" in mmc.getProperty("AO-mode", "Label"):
            AO_mode = "Before-every-acq"
        elif "Optimize-now" in mmc.getProperty("AO-mode", "Label"):
            AO_mode = "Optimize-now"

        # get O2-O3 focus mode / interval
        if "Initial-only" in mmc.getProperty("O2O3focus-mode", "Label"):
            O2O3_mode = "Initial-only"
        elif "Before-each-XYZ" in mmc.getProperty("O2O3focus-mode", "Label"):
            O2O3_mode = "Before-each-xyz"
        elif "Before-each-t" in mmc.getProperty("O2O3focus-mode", "Label"):
            O2O3_mode = "Before-each-t"
        elif "After-30min" in mmc.getProperty("O2O3focus-mode", "Label"):
            O2O3_mode = "After-30min"
        elif "None" in mmc.getProperty("O2O3focus-mode", "Label"):
            O2O3_mode = "None"

        # Get the current MDAsequence and convert to dictionary 
        sequence = mda_widget.value()
        sequence_dict = json.loads(sequence.model_dump_json())
        
        # extract the revelant portions for qi2lab-OPM
        # Stage positions
        stage_positions = sequence_dict["stage_positions"]
        n_stage_pos = len(stage_positions)
        opmAOmirror_local.n_positions = n_stage_pos
        
        # timelapse values 
        time_plan = sequence_dict["time_plan"]
        if time_plan is not None:
            try:
                n_time_steps = time_plan["loops"]
            except Exception:
                n_time_steps = 1
            time_interval = time_plan["interval"]
        else:
            n_time_steps = 1
            time_interval = 0

        # channels, state and exposures
        channels = sequence_dict["channels"]
        channel_names = config["OPM"]["channel_ids"]
        active_channels = [False] * len(channel_names) #[False,False,False,False,False]
        exposure_channels = [0.] * len(channel_names) #[0.,0.,0.,0.,0.,0.]
        laser_powers = [0.] * len(channel_names) # [0.,0.,0.,0.,0.]
        for channel in channels:
            # determine current channel idx
            ch_id = channel["config"]
            ch_idx = config["OPM"]["channel_ids"].index(ch_id)
            
            # update active channel and powers
            active_channels[ch_idx] = True
            exposure_channels[ch_idx] = channel["exposure"]
            laser_powers[ch_idx] = float(
                mmc.getProperty(
                    "Coherent-Scientific Remote",
                    config["Lasers"]["laser_names"][ch_idx] + " - PowerSetpoint (%)"
                    )
                )

        active_channel_exps = [_exp for _, _exp in zip(active_channels, exposure_channels) if _]

        # TODO need to understand the interleave logic
        if len(set(active_channel_exps)) == 1:
            interleaved_acq = True
        else:
            interleaved_acq = False
        n_active_channels = sum(active_channels)
        
        # laser blanking
        if "On" in mmc.getProperty("LaserBlanking","Label"):
            laser_blanking = True
        elif "Off" in mmc.getProperty("LaserBlanking","Label"):
            laser_blanking = False
        else:
            laser_blanking = True

        # reload hardware configuration file before setting up acq
        with open(config_path, "r") as config_file:
            updated_config = json.load(config_file)

        
        #--------------------------------------------------------------------#
        # Create event structure
        #--------------------------------------------------------------------#
        
        opm_events: list[MDAEvent] = []

        # run O2-O3 autofocus before acquisition starts
        O2O3_event = MDAEvent(
            action=CustomAction(
                name="O2O3-autofocus",
                data = {
                    "Camera" : {                    
                        "exposure_ms" : float(updated_config["O2O3-autofocus"]["exposure_ms"]),
                        "camera_crop" : [
                            updated_config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                            updated_config["Camera"]["camera_center_y"] - int(config["O2O3-autofocus"]["camera_crop_y"]//2),
                            updated_config["Camera"]["camera_crop_x"],
                            updated_config["O2O3-autofocus"]["camera_crop_y"]
                        ]
                    }
                }
            )
        )
        if O2O3_mode=="Initial-only":
            opm_events.append(O2O3_event)
        
        #--------------------------------------------------------------------#
        # Create CustomAction DAQ events for Projection and Mirror Sweep modes
        #--------------------------------------------------------------------#
        # check OPM mode and create 
        # TODO Catch cases where opm mode are not proejction or mirror
        opm_mode = mmc.getProperty("OPM-mode", "Label")
        if "Projection" in opm_mode:
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
                            "exposure_channels" : exposure_channels,
                            "camera_crop" : [
                                updated_config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                                updated_config["Camera"]["camera_center_y"] - int(int(mmc.getProperty("ImageCameraCrop","Label"))//2),
                                updated_config["Camera"]["camera_crop_x"],
                                int(mmc.getProperty("ImageCameraCrop","Label")),
                            ]
                        }
                    }
                )
            )
        elif "Mirror" in opm_mode:
            daq_mode = "mirror"
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
                            "exposure_channels" : exposure_channels,
                            "camera_crop" : [
                                updated_config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                                updated_config["Camera"]["camera_center_y"] - int(int(mmc.getProperty("ImageCameraCrop","Label"))//2),
                                updated_config["Camera"]["camera_crop_x"],
                                int(mmc.getProperty("ImageCameraCrop","Label"))
                            ]
                        }
                    }
                )
            )
        
        #--------------------------------------------------------------------#
        # Create CustomAction events for running AO
        #--------------------------------------------------------------------#
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
                            "laser_power" : list(float(_) for _ in updated_config["AO-projection"]["laser_power"]),
                            "mode": str(updated_config["AO-projection"]["mode"]),
                            "iterations": int(updated_config["AO-projection"]["iterations"]),
                            "image_mirror_step_um" : float(updated_config["AO-projection"]["image_mirror_step_um"]),
                            "image_mirror_range_um" : float(updated_config["AO-projection"]["image_mirror_range_um"]),
                            "blanking": bool(True),
                        },
                        "Camera" : {
                            "exposure_ms": float(updated_config["AO-projection"]["exposure_ms"]),
                            "camera_crop" : [
                                updated_config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                                updated_config["Camera"]["camera_center_y"] - int(int(updated_config["AO-projection"]["camera_crop_y"])//2),
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
            for chan_idx, chan_str in enumerate(config["OPM"]["channel_ids"]):
                if active_channel==chan_str:
                    AO_channel_states[chan_idx] = True
                    AO_laser_powers[chan_idx] = float(
                        mmc.getProperty(
                            config["Lasers"]["name"],
                            str(config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)"
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
                                updated_config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                                updated_config["Camera"]["camera_center_y"] - int(int(mmc.getProperty("ImageCameraCrop","Label"))//2),
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
            # mda_widget._mmc.run_mda(opm_events, output=output)
        
        #--------------------------------------------------------------------#
        # Create MDAevents for time and positions
        #--------------------------------------------------------------------#
        # Flags to help ensure sequence-able events are kept together 
        need_to_setup_DAQ = True
        need_to_setup_stage = True

        # setup nD mirror-based AO-OPM acquisition event structure
        for time_idx in range(n_time_steps):
            # Check if autofocus before each timepoint and not initial-only mode
            if O2O3_mode == "Before-each-t" and not(O2O3_mode == "Initial-only"):
                opm_events.append(O2O3_event)
            # Check for multi-position acq.
            for pos_idx in range(n_stage_pos):
                if need_to_setup_stage:
                    stage_event = MDAEvent(
                        action=CustomAction(
                            name="Stage-Move",
                            data = {
                                "Stage" : {
                                    "x_pos" : stage_positions[pos_idx]["x"],
                                    "y_pos" : stage_positions[pos_idx]["y"],
                                    "z_pos" : stage_positions[pos_idx]["z"],
                                }
                            }
                        )
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
                    current_AO_event = MDAEvent(**AO_event.model_dump())
                    current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                    current_AO_event.action.data["AO"]["apply_existing"] = False
                    opm_events.append(current_AO_event)
                # Apply correction for this position if time_idx > 0
                elif AO_mode == "Before-each-xyz" and time_idx > 0:
                    need_to_setup_DAQ = True
                    current_AO_event = MDAEvent(**AO_event.model_dump())
                    current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                    current_AO_event.action.data["AO"]["apply_existing"] = True
                    opm_events.append(current_AO_event)
                # Otherwise, run AO opt. before every acquisition. COSTLY in time and photons!
                elif AO_mode == "Before-every-acq":
                    need_to_setup_DAQ = True
                    current_AO_event = MDAEvent(**AO_event.model_dump())
                    current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
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
                    # create camera events
                    for scan_idx in range(n_scan_steps):
                        for chan_idx in range(n_active_channels):
                            image_event = MDAEvent(
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
                                        "camera_center_x" : updated_config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                                        "camera_center_y" : updated_config["Camera"]["camera_center_y"] - int(int(mmc.getProperty("ImageCameraCrop","Label"))//2),
                                        "camera_crop_x" : updated_config["Camera"]["camera_crop_x"],
                                        "camera_crop_y" : int(mmc.getProperty("ImageCameraCrop","Label"))
                                    },
                                    "OPM" : {
                                        "angle_deg" : float(updated_config["OPM"]["angle_deg"]),
                                        "camera_stage_orientation" : str(updated_config["OPM"]["camera_stage_orientation"]),
                                        "camera_mirror_orientation" : str(updated_config["OPM"]["camera_mirror_orientation"])
                                    },
                                    "Stage" : {
                                        "x_pos" : stage_positions[pos_idx]["x"],
                                        "y_pos" : stage_positions[pos_idx]["y"],
                                        "z_pos" : stage_positions[pos_idx]["z"],
                                    }
                                }
                            )
                            opm_events.append(image_event)
                else:
                    for chan_idx, chan_bool in enumerate(active_channels):
                        temp_channels = [False,False,False,False,False]
                        if chan_bool:
                            if need_to_setup_DAQ:
                                need_to_setup_DAQ = True
                                current_DAQ_event = MDAEvent(**DAQ_event.model_dump())
                                temp_channels[chan_idx] = True
                                current_DAQ_event.action.data["DAQ"]["active_channels"] = temp_channels
                                current_DAQ_event.action.data["Camera"]["exposure_channels"] = exposure_channels
                                opm_events.append(current_DAQ_event)
                            for scan_idx in range(n_scan_steps):
                                image_event = MDAEvent(
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
                                            "exposure_ms" : exposure_channels[chan_idx],
                                            "camera_center_x" : updated_config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                                            "camera_center_y" : updated_config["Camera"]["camera_center_y"] - int(int(mmc.getProperty("ImageCameraCrop","Label"))//2),
                                            "camera_crop_x" : updated_config["Camera"]["camera_crop_x"],
                                            "camera_crop_y" : int(mmc.getProperty("ImageCameraCrop","Label"))
                                        },
                                        "OPM" : {
                                            "angle_deg" : float(updated_config["OPM"]["angle_deg"]),
                                            "camera_stage_orientation" : str(updated_config["OPM"]["camera_stage_orientation"]),
                                            "camera_mirror_orientation" : str(updated_config["OPM"]["camera_mirror_orientation"])
                                        },
                                        "Stage" : {
                                            "x_pos" : stage_positions[pos_idx]["x"],
                                            "y_pos" : stage_positions[pos_idx]["y"],
                                            "z_pos" : stage_positions[pos_idx]["z"],
                                        }
                                    }
                                )
                                opm_events.append(image_event)

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

        handler = TensorStoreHandler(path= Path(output) / Path("data.zarr"), delete_existing=True)
        mda_widget._mmc.run_mda(opm_events, output=handler)

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
            print("DAQ is running, stopping now.")
            opmNIDAQ_setup_preview.stop_waveform_playback()
        
        
        
        # Update device states from the configuration entries
        # opmNIDAQ_setup_preview.set_acquisition_params(
        #     scan_type=,
        #     channel_states=,
        #     image_mirror_step_size_um=,
        #     image_mirror_sweep_um=,
        #     laser_blanking=,
        #     exposure_ms=
        # )
        
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
        
    