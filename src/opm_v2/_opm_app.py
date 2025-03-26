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
import os
import sys
import traceback
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast
from datetime import datetime

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
from opm_v2.engine.opm_engine import OPMEngine
from opm_v2.engine.setup_events import setup_stagescan
from opm_v2.handlers.opm_mirror_handler import OPMMirrorHandler



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

    # load microscope configuration file
    config_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\opm_config_20250304.json")
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
        modes_to_ignore = [],
        output_path= config["AOMirror"]["output_path"]
    )
    
    opmAOmirror.set_mirror_positions_flat()

    # load OPM NIDAQ and OPM AO mirror classes
    opmNIDAQ = OPMNIDAQ(
        name = str(config["NIDAQ"]["name"]),
        scan_type = str(config["NIDAQ"]["scan_type"]),
        exposure_ms = float(config["Camera"]["exposure_ms"]),
        laser_blanking = bool(config["NIDAQ"]["laser_blanking"]),
        image_mirror_calibration = float(str(config["NIDAQ"]["image_mirror_calibration"])),
        projection_mirror_calibration = float(str(config["NIDAQ"]["projection_mirror_calibration"])),
        image_mirror_neutral_v = float(str(config["NIDAQ"]["image_mirror_neutral_um"])),
        projection_mirror_neutral_v = float(str(config["NIDAQ"]["projection_mirror_neutral_um"])),
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
    
    # Initialize and close alignment laser shutter
    opmPicardShutter = PicardShutter(int(config["O2O3-autofocus"]["shutter_id"]))
    opmPicardShutter.closeShutter()
    
    # grab mmc instance and load OPM config file
    mmc = win.mmcore
    mmc.loadSystemConfiguration(Path(config["mm_config_path"]))
    
    # Enforce config's default properties
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

    def calculate_projection_crop(image_mirror_range_um):
        # Calculate the the number of Y pixels for the scan range
        roi_height_um = image_mirror_range_um # * np.cos(np.deg2rad(30))
        roi_height_px = int(roi_height_um / mmc.getPixelSizeUm())
        return roi_height_px
    
    def update_state(device_name=None, property_name=None):
        """
        Update microscope states and values upon changes to in the GUI
        """
        
        # Ignore updates from AutoShutter
        if device_name == mmc.getShutterDevice() and property_name == "State":
            # print(f"preventing shutter update: {device_name}, {property_name}")
            return
           
        # Get the selected imaging mode
        opm_mode = mmc.getProperty("OPM-mode", "Label")
        
        # Define the scan type based on imaging mode
        if "Standard" in opm_mode:
            _scan_type = "2d"
        elif "Stage" in opm_mode:
            _scan_type = "2d"
        elif "Projection" in opm_mode:
            _scan_type = "projection"
        elif "Mirror" in opm_mode:
            _scan_type = "mirror"
        #--------------------------------------------------------------------#
        # Stop DAQ playback
        opmNIDAQ_update_state = OPMNIDAQ.instance()
        restart_sequence = False
        if mmc.isSequenceRunning():
            mmc.stopSequenceAcquisition()
            # print("stoping sequence`")
            restart_sequence = True
            
        if opmNIDAQ_update_state.running():
            opmNIDAQ_update_state.stop_waveform_playback()
            # if not (_scan_type == opmNIDAQ_update_state.scan_type):
            #     opmNIDAQ.clear_tasks()
            
        #--------------------------------------------------------------------#
        # Grab gui values 
        #--------------------------------------------------------------------#
        
        # Get the gui exposure
        _exposure_ms = round(float(mmc.getProperty(config["Camera"]["camera_id"], "Exposure")), 0)
        
        # Get the current ImageGalvoMirror parameters
        _image_mirror_range_um = np.round(float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),0)
        _image_mirror_step_um = np.round(float(mmc.getProperty("ImageGalvoMirrorStep", "Label").split("-")[0]),2)
        
            # reload hardware configuration file before setting up events
        with open(config_path, "r") as config_file:
            updated_config = json.load(config_file)
            
        _projection_calibration = updated_config["NIDAQ"]["projection_mirror_calibration"]
        opmNIDAQ_update_state.projection_mirror_calibration = _projection_calibration
        
        # Get the selected active channel and populate channel_states
        _active_channel_id = mmc.getProperty("LED", "Label")
        _channel_states = [False] * len(config["OPM"]["channel_ids"])
        for ch_i, ch_str in enumerate(config["OPM"]["channel_ids"]):
            # print({_active_channel_id})
            if _active_channel_id==ch_str:
                # print(f"ch_str:{ch_str}")
                _channel_states[ch_i] = True
        # print(_channel_states)
        # Get the current LaserBlanking State
        if mmc.getProperty("LaserBlanking", "Label")=="On":
            _laser_blanking = True
        else:
            _laser_blanking = False
        
        # Get the current gui crop value (not used in Projection)
        gui_camera_crop_y = int(mmc.getProperty("ImageCameraCrop","Label"))
            
        # The Camera's current sensor mode 
        # TODO
        # current_sensor_mode = mmc.getProperty(config["Camera"]["camera_id"], "SENSOR MODE")
        
        #--------------------------------------------------------------------#
        # Reinforce camera state
        #--------------------------------------------------------------------#
        
        # Configure the camera sensor mode (switches for projection mode, TODO)
        if _scan_type=="projection":
            # if not (current_sensor_mode=="PROGRESSIVE"):
            #     mmc.setProperty(config["Camera"]["camera_id"], "SENSOR MODE", "PROGRESSIVE")
            #     mmc.waitForDevice(str(config["Camera"]["camera_id"]))
            # print("here in update projection")
            
            # Crop the chip down to determined size
            camara_crop_y = calculate_projection_crop(_image_mirror_range_um)
            if not (camara_crop_y == mmc.getROI()[-1]): 
                current_roi = mmc.getROI()
                mmc.clearROI()
                mmc.waitForDevice(str(config["Camera"]["camera_id"]))
                mmc.setROI(
                    config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                    config["Camera"]["camera_center_y"] - int(camara_crop_y//2),
                    config["Camera"]["camera_crop_x"],
                    camara_crop_y
                )
                mmc.waitForDevice(str(config["Camera"]["camera_id"]))
            
        else :
            # if not (current_sensor_mode=="AREA"):
            #     mmc.setProperty(config["Camera"]["camera_id"], "SENSOR MODE", "AREA")
            # #     mmc.waitForDevice(str(config["Camera"]["camera_id"]))
            # print("here in update else")
                
            # Crop the chip down to gui size
            if not (gui_camera_crop_y == mmc.getROI()[-1]): 
                current_roi = mmc.getROI()
                mmc.clearROI()
                mmc.waitForDevice(str(config["Camera"]["camera_id"]))
                mmc.setROI(
                    config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                    config["Camera"]["camera_center_y"] - int(gui_camera_crop_y//2),
                    config["Camera"]["camera_crop_x"],
                    gui_camera_crop_y
                )
                mmc.waitForDevice(str(config["Camera"]["camera_id"]))
                
        # Set the camera exposure
        mmc.setProperty(str(config["Camera"]["camera_id"]),"Exposure",_exposure_ms)
        mmc.waitForDevice(str(config["Camera"]["camera_id"]))
        
        #--------------------------------------------------------------------#
        # update DAQ values and prepare new waveform
        #--------------------------------------------------------------------#
        
        opmNIDAQ_update_state.set_acquisition_params(
            scan_type=_scan_type,
            channel_states=_channel_states,
            image_mirror_step_size_um=_image_mirror_step_um,
            image_mirror_range_um=_image_mirror_range_um,
            laser_blanking=_laser_blanking,
            exposure_ms=_exposure_ms,
        )
        
        #--------------------------------------------------------------------#
        # Restart acquisition if needed
        #--------------------------------------------------------------------#
        
        if restart_sequence:
            opmNIDAQ_update_state.start_waveform_playback()
            mmc.startContinuousSequenceAcquisition()
            
    # Connect changes in gui fields to the update_state method.            
    # mmc.events.propertyChanged.connect(update_state)
    mmc.events.configSet.connect(update_state)
    
    # grab handle to the Stage widget
    # stage_widget = win.get_widget(WidgetAction.STAGE_CONTROL)

    # grab handle to the MDA widget and define custom execute_mda method
    # in our method, the MDAEvents are modified before running the sequence
    mda_widget = win.get_widget(WidgetAction.MDA_WIDGET)
    # mda_widget.channels.setChannelGroups({"Channel":["405nm", "488nm", "561nm", "637nm", "730nm"]})
    # open an issue for this, when printing the mda channel sequence it has the group still set to AO_mode, but config is a channel.

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
        
        # reload hardware configuration file before setting up events
        with open(config_path, "r") as config_file:
            updated_config = json.load(config_file)
        
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        # Get the acquisition parameters from configuration properties
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
                    
        # Get the imaging mode from GUI
        opm_mode = mmc.getProperty("OPM-mode", "Label")
        
        # get AO mode / interval
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
        elif "Optimize-now" in mmc.getProperty("O2O3focus-mode", "Label"):
            O2O3_mode = "Optimize-now"
        elif "None" in mmc.getProperty("O2O3focus-mode", "Label"):
            O2O3_mode = "None"

        # check fluidics mode
        if "None" in mmc.getProperty("Fluidics-mode", "Label"):
            FP_mode = "None"
            FP_num_rounds = 1
        elif "Thin-16bit" in mmc.getProperty("Fluidics-mode", "Label"):
            FP_mode = "thin_16bit"
            FP_num_rounds = 16
        elif "Thin-22bit" in mmc.getProperty("Fluidics-mode", "Label"):
            FP_mode = "thin_22bit"
            FP_num_rounds = 22
        elif "Thick-16bit" in mmc.getProperty("Fluidics-mode", "Label"):
            FP_mode = "thick_16bit"
            FP_num_rounds = 16
        elif "Thick-2bit" in mmc.getProperty("Fluidics-mode", "Label"):
            FP_mode = "thick_22bit"
            FP_num_rounds = 22
            
        # If running a fluidics experiment, make sure the user has ESI running
        if not FP_mode=="None":
            # load dialog to interrupt user.
            from PyQt6.QtWidgets import QMessageBox
            response = QMessageBox.information(
                mda_widget, 
                'WARNING ! ! ! ESI MUST BE RUNNING ! ! !',
                'IS ESI SEQUENCE LOADED AND STARTED?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if response is not QMessageBox.StandardButton.Yes:
                return
            else:
                print("ESI Sequence accepted")
            
       #--------------------------------------------------------------------#
        # Validate acquisition settings
        #--------------------------------------------------------------------#
        
        print(
            f"OPM mode:{opm_mode}\n",
            f"AO mode:{AO_mode}\n",
            f"AF mode:{O2O3_mode}\n",
            f"FP mode:{FP_mode}\n",
        )
        
        # Check if running an Optimize-now acq.
        if ("Optimize-now" in AO_mode) or ("Optimize-now" in O2O3_mode):
            optimize_now = True
        else: 
            optimize_now = False
            
        if not(optimize_now) and not(output):
            print("Must set MDA save path!")
            return
        
        # only projection and mirror modes are available if not optimizing now
        if ("Stage" in opm_mode):
            opm_events, handler = setup_stagescan(
                mmc = mmc,
                config = updated_config,
                sequence = mda_widget.value(),
                output = output,
                FP_mode = FP_mode,
                FP_num_rounds = 11
                )
            
            # tell AO mirror class where to save mirror information
            opmAOmirror_local.output_path = output.parents[0]

            # run MDA with our event structure and modified tensorstore handler 
            mda_widget._mmc.run_mda(opm_events, output=handler)

        elif ("Standard" in opm_mode) and not(optimize_now):
            print("OPM-mode must be 'Projection' or 'Mirror' or 'Stage'")   
            return
                
        if ("Stage" not in opm_mode):
            # try to get camera conversion factor information
            try:
                offset = mmc.getProperty(
                    config["Camera"]["camera_id"],
                    "CONVERSION FACTOR OFFSET"
                )
                e_to_ADU = mmc.getProperty(
                    config["Camera"]["camera_id"],
                    "CONVERSION FACTOR COEFF"
                )
            except Exception:
                offset = 0.
                e_to_ADU = 1.      
            
            #--------------------------------------------------------------------#
            #--------------------------------------------------------------------#
            #--------------------------------------------------------------------#
            # Get the event structure parameters and or AO values
            # - For imaging: timepoints, stage positions, channels
            # - For optimization: run o2o3-af?, ao params from config
            #--------------------------------------------------------------------#
            #--------------------------------------------------------------------#
            #--------------------------------------------------------------------#
            
            # Create daq channel lists
            channel_names = config["OPM"]["channel_ids"]
            active_channels = [False] * len(channel_names) 
            exposure_channels = [0.] * len(channel_names) 
            laser_powers = [0.] * len(channel_names)
            
            now = datetime.now()
            timestamp = f"{now.year:4d}{now.month:2d}{now.day:2d}_{now.hour:2d}{now.minute:2d}{now.second:2d}"

            #--------------------------------------------------------------------#
            #--------------------------------------------------------------------#
            # If optimize now, ignore MDA widget and run_mda now.
            #--------------------------------------------------------------------#
            #--------------------------------------------------------------------#
            
            if optimize_now:
                opm_events: list[MDAEvent] = []
                
                if "Optimize-now" in O2O3_mode:                
                    print("Running O2O3 Autofocus Now\n")
                    
                    o2o3_event = MDAEvent(
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
                    opm_events.append(o2o3_event)
                    
                if "Optimize-now" in AO_mode:
                    
                    print("Running AO optimization Now\n")
                    
                    # get image galvo mirror range
                    image_mirror_range_um = np.round(
                        float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),0
                        )
                    
                    # setup AO using values in the config widget, NOT the MDA widget
                    AO_channel_states = [False] * len(channel_names) 
                    AO_channel_powers = [0.] * len(channel_names)
                    AO_active_channel_id = mmc.getProperty("LED", "Label")
                    AO_exposure_ms = np.round(float(mmc.getProperty("OrcaFusionBT", "Exposure")),0)
                    AO_image_mirror_range_um = float(image_mirror_range_um)
                    AO_camera_crop_y = int(calculate_projection_crop(AO_image_mirror_range_um))
                    AO_iterations = int(updated_config["AO-projection"]["iterations"])
                    AO_metric = str(updated_config["AO-projection"]["mode"])
                    AO_save_path = Path(str(updated_config["AO-projection"]["optimize_now_path"])) / Path(f"{timestamp}_ao_optimizeNOW")
                    
                    output = Path(str(updated_config["AO-projection"]["optimize_now_path"]))
                    
                    # Set the active channel in the daq channel list
                    for chan_idx, chan_str in enumerate(config["OPM"]["channel_ids"]):
                        if AO_active_channel_id==chan_str:
                            AO_channel_states[chan_idx] = True
                            AO_channel_powers[chan_idx] = float(
                                mmc.getProperty(
                                    config["Lasers"]["name"],
                                    str(config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)"
                                )
                            )
                            
                    # check to make sure there exist a laser power > 0
                    if sum(AO_channel_powers)==0:
                        print("All AO lasers set to 0!")
                        return
                    
                    # Make sure a reasonable expsoure is choosen
                    if AO_exposure_ms<50:
                        print(f"Camera exposure should be increased! : {AO_exposure_ms}")
                        return
                    
                    # Update the AO configuration with the GUI values
                    updated_config["AO-projection"]["channel_states"] = AO_channel_states
                    updated_config["AO-projection"]["channel_powers"] = AO_channel_powers
                    updated_config["AO-projection"]["exposure_ms"] = AO_exposure_ms
                    updated_config["AO-projection"]["image_mirror_range_um"] = AO_image_mirror_range_um
                    updated_config["AO-projection"]["camera_crop_y"] = AO_camera_crop_y
                    
                    with open(config_path, "w") as file:
                        json.dump(updated_config, file, indent=4)
                    
                    ao_optimize_event = MDAEvent(
                        exposure = AO_exposure_ms,
                        action=CustomAction(
                            name="AO-projection",
                            data = {
                                "AO" : {
                                    "opm_mode": str("projection"),
                                    "channel_states": AO_channel_states,
                                    "channel_powers" : AO_channel_powers,
                                    "mode": AO_metric,
                                    "iterations": AO_iterations,
                                    "image_mirror_range_um" : AO_image_mirror_range_um,
                                    "blanking": bool(True),
                                    "apply_existing": bool(False),
                                    "pos_idx":int(0),
                                    "output_path":AO_save_path
                                },
                                "Camera" : {
                                    "exposure_ms": AO_exposure_ms,
                                    "camera_crop" : [
                                        updated_config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                                        updated_config["Camera"]["camera_center_y"] - int(AO_camera_crop_y//2),
                                        updated_config["Camera"]["camera_crop_x"],
                                        AO_camera_crop_y
                                    ]
                                }
                            }
                        )
                    )
                    opm_events.append(ao_optimize_event)
                    opmAOmirror_local.output_path = AO_save_path
                    
                # Run optimize now events
                mda_widget._mmc.run_mda(opm_events, output=None)

            else:
                #----------------------------------------------------------------#
                #----------------------------------------------------------------#
                # Setup acquisition timepoints, positions, and channels:
                # - Grab values from MDA widget
                # - AO settings from configuration file
                # - Stage positions are opm_mode dependent
                #----------------------------------------------------------------#
                #----------------------------------------------------------------#
                
                sequence = mda_widget.value()
                sequence_dict = json.loads(sequence.model_dump_json())
                mda_stage_positions = sequence_dict["stage_positions"]
                mda_grid_plan = sequence_dict["grid_plan"]
                mda_channels = sequence_dict["channels"]        
                mda_time_plan = sequence_dict["time_plan"]
                mda_z_plan = sequence_dict["z_plan"]
                
                #----------------------------------------------------------------#
                # Compile the active channel info and validate powers
                #----------------------------------------------------------------#
                if not(mda_channels):
                    print("Must select channels to use in MDA widget")
                    return
                elif not("Channel" in mda_channels[0]["group"]):
                    print("Must select channels to use in MDA widget")
                    return
                
                channel_names = config["OPM"]["channel_ids"]
                channel_states = [False] * len(channel_names) 
                channel_exposures_ms = [0.] * len(channel_names) 
                channel_powers = [0.] * len(channel_names) 
                
                if mmc.getProperty("LaserBlanking", "Label")=="On":
                    laser_blanking = True
                else:
                    laser_blanking = False
                    
                # Validate mda channels is set to channels and not running optimize now.
                if not(mda_channels):
                    print("Must select channels to use in MDA widget")
                    return
                elif not("Channel" in mda_channels[0]["group"]):
                    print("Must select channels to use in MDA widget")
                    return
            
                # Iterate through MDA checked channels and update active lasers list
                for mda_ch in mda_channels:
                    # Get the matching channel idx active lasers list
                    ch_id = mda_ch["config"]
                    ch_idx = config["OPM"]["channel_ids"].index(ch_id)
                    
                    # update active channel and powers
                    channel_states[ch_idx] = True
                    channel_exposures_ms[ch_idx] = mda_ch["exposure"]
                    channel_powers[ch_idx] = float(
                        mmc.getProperty(
                            "Coherent-Scientific Remote",
                            config["Lasers"]["laser_names"][ch_idx] + " - PowerSetpoint (%)"
                            )
                        )
                    
                n_active_channels = sum(channel_states)
                active_channel_names = [_name for _, _name in zip(channel_states, channel_names) if _]
                
                # Interleave only available if all channels have the same exposure.
                active_channel_exps = [_exp for _, _exp in zip(channel_states, channel_exposures_ms) if _]
                if len(set(active_channel_exps))==1:
                    interleaved_acq = True
                else:
                    interleaved_acq = False
                        
                # Check to make sure there exist a laser power > 0
                if sum(channel_powers)==0:
                    print("All lasers set to 0!")
                    return
                
                #--------------------------------------------------------------------#
                # Generate time points
                #--------------------------------------------------------------------#
                
                # If running a fluidics experiment, the number of time points are the number of rounds.
                if FP_mode != "None": 
                    n_time_steps = FP_num_rounds
                    time_interval = 0
                    
                # If the time plan is selected, generate a timelapse 
                elif mda_time_plan is not None:
                    n_time_steps = mda_time_plan["loops"]
                    time_interval = mda_time_plan["interval"]
                
                # If no time plan is given, assume a single time point
                else:
                    n_time_steps = 1
                    time_interval = 0
                    
                #----------------------------------------------------------------#
                # Get configuration properities
                #----------------------------------------------------------------#
                
                # get the camera crop value
                camera_crop_y = int(mmc.getProperty("ImageCameraCrop","Label"))
                
                # get image galvo mirror range and step size
                image_mirror_range_um = np.round(
                    float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),0
                    )
                image_mirror_step_um = np.round(
                    float(mmc.getProperty("ImageGalvoMirrorStep", "Label").split("-")[0]),2
                    )
                
                # use OPMNIDAQ class calculation for number of scan steps to ensure consistency
                opmNIDAQ_custom.set_acquisition_params(
                    scan_type="mirror",
                    channel_states = [False,False,False,False,False],
                    image_mirror_step_size_um=image_mirror_step_um,
                    image_mirror_range_um=image_mirror_range_um,
                    laser_blanking=True,
                    exposure_ms=100.
                )
                n_scan_steps = opmNIDAQ_custom.n_scan_steps
                
                #--------------------------------------------------------------------#
                # Generate stage positions
                #--------------------------------------------------------------------#
                
                stage_positions = []
                
                if ("Mirror" in opm_mode) or ("Projection" in opm_mode):
                    # If only the stage positions tab is selected, use the MDA positions tab
                    if (mda_stage_positions is not None) and (mda_grid_plan is None):
                        for stage_pos in mda_stage_positions:
                            stage_positions.append({
                                        'x': float(stage_pos['x']),
                                        'y': float(stage_pos['y']),
                                        'z': float(stage_pos['z'])
                                    })
                            
                    # If the grid tab is selected, generate positions from the grid plan
                    elif mda_grid_plan is not None:
                        #----------------------------------------------------------------#
                        # Get the z positions
                        # - First check the z-plan
                        # - Then use the singular stage position in the stage plan if it exist
                        # - Lastly use the current stage position
                        #----------------------------------------------------------------#
                        
                        # Check if multiple z plans are setup in MDA
                        if mda_z_plan is not None:
                            max_z_pos = float(mda_z_plan["top"])
                            min_z_pos = float(mda_z_plan["bottom"])
                            step_z = 0.6 * float(mmc.getProperty("ImageCameraCrop","Label") ) * np.sin((np.pi/180.)*float(config["OPM"]["angle_deg"])) * mmc.getPixelSizeUm()
                            if min_z_pos > max_z_pos:
                                step_z = -1 * step_z
                            num_z_pos = int(np.ceil(np.abs((max_z_pos - min_z_pos) / step_z)))
                            
                        # if no z plan is selected, check if only 1 position exists in the stage position plan
                        elif (mda_stage_positions is not None):
                            if len([_p for _p in mda_stage_positions])==1:
                                temp = []
                                for stage_pos in mda_stage_positions:
                                    temp.append({
                                        'x': float(stage_pos['x']),
                                        'y': float(stage_pos['y']),
                                        'z': float(stage_pos['z'])
                                    })
                                min_z_pos = temp[0]["z"]
                                step_z = 0
                                num_z_pos = 1
                        
                        # If no other z positions are set, resort to the current z position 
                        else:
                            min_z_pos = mmc.getZPosition()
                            step_z = 0
                            num_z_pos = 1
                        
                        #----------------------------------------------------------------# 
                        # Generate stage position list using the grid's generated positions
                        #----------------------------------------------------------------#
                        
                        # iterate through sequence to grab stage positions from grid plan
                        stage_positions_array = []
                        for event in sequence:
                            json_event = json.loads(event.model_dump_json())
                            stage_positions_array.append([
                                float(json_event['z_pos']),
                                float(json_event['y_pos']),
                                float(json_event['x_pos'])]
                            )
                        stage_positions_array = np.asarray(stage_positions_array, dtype=np.float32)

                        # Define our own snake pattern, assumes a 20% overlap.
                        max_y_pos = np.max(stage_positions_array[:,1])
                        min_y_pos = np.min(stage_positions_array[:,1])
                        max_x_pos = np.max(stage_positions_array[:,2])
                        min_x_pos = np.min(stage_positions_array[:,2])
                        num_x_pos = int(np.ceil(np.abs(max_x_pos - min_x_pos) / (0.6 * config["Camera"]["camera_crop_x"]*mmc.getPixelSizeUm())))
                        num_y_pos = int(np.ceil(np.abs(max_y_pos - min_y_pos) / (0.6 * image_mirror_range_um)))
                        step_x = (max_x_pos - min_x_pos) / num_x_pos
                        step_y = (max_y_pos - min_y_pos) / num_y_pos
                        
                        # Generate stage positions in a snake like pattern
                        for x_pos in range(num_x_pos):
                            # Even rows (left to right)
                            if x_pos % 2 == 0:                          
                                y_range = range(num_y_pos)
                            # Odd rows (right to left)
                            else:  
                                y_range = range(num_y_pos - 1, -1, -1)
                            
                            # populate x positions
                            for y_pos in y_range:
                                for z_pos in range(num_z_pos):
                                    stage_positions.append({
                                        "x": float(np.round(min_x_pos + x_pos * step_x, 2)),
                                        "y": float(np.round(min_y_pos + y_pos * step_y, 2)),
                                        "z": float(np.round(min_z_pos + z_pos * step_z, 2))
                                    })
                        
                    # if neither positions or grid plan are selected, use the current position
                    elif (mda_grid_plan is None) and (mda_stage_positions is None):
                        current_stage_position = {
                            'x': float(mmc.getXPosition()),
                            'y': float(mmc.getYPosition()),
                            'z': float(mmc.getZPosition())
                        }
                        stage_positions.append({
                            'x': float(mmc.getXPosition()),
                            'y': float(mmc.getYPosition()),
                            'z': float(mmc.getZPosition())
                        })
                        # TODO
                        # return
                        
                    # update the wfc mirror positions array shape
                    n_stage_pos = len(stage_positions)
                    opmAOmirror_local.n_positions = n_stage_pos

                elif "Stage" in opm_mode:
                    pass
                    # TODO
                    
                #----------------------------------------------------------------#
                # Define AO settings from configuration file
                #----------------------------------------------------------------#
                AO_exposure_ms = round(float(updated_config["AO-projection"]["exposure_ms"]),0)
                AO_channel_states = list(map(bool,updated_config["AO-projection"]["channel_states"]))
                AO_channel_powers = list(float(_) for _ in updated_config["AO-projection"]["channel_powers"])
                AO_camera_crop_y = int(calculate_projection_crop(updated_config["AO-projection"]["image_mirror_range_um"]))
                AO_image_mirror_range_um = float(updated_config["AO-projection"]["image_mirror_range_um"])
                AO_save_path = Path(output).parent / Path(f"{timestamp}_ao_optimize")
                AO_iterations = int(updated_config["AO-projection"]["iterations"])
                AO_metric = str(updated_config["AO-projection"]["mode"])
                
                print(
                    f"Running {opm_mode} imaging acquisitions:\n",
                    f"timepoints:{n_time_steps}, stage positions: {n_stage_pos}, channels: {active_channel_names}"
                    )
                #----------------------------------------------------------------#
                # Get configuration properities
                #----------------------------------------------------------------#
                # get the camera crop value
                camera_crop_y = int(mmc.getProperty("ImageCameraCrop","Label"))
                
                # get image galvo mirror range and step size
                image_mirror_range_um = np.round(
                    float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),0
                    )
                image_mirror_step_um = np.round(
                    float(mmc.getProperty("ImageGalvoMirrorStep", "Label").split("-")[0]),2
                    )
                
                # use OPMNIDAQ class calculation for number of scan steps to ensure consistency
                opmNIDAQ_custom.set_acquisition_params(
                    scan_type="mirror",
                    channel_states = [False,False,False,False,False],
                    image_mirror_step_size_um=image_mirror_step_um,
                    image_mirror_range_um=image_mirror_range_um,
                    laser_blanking=True,
                    exposure_ms=100.
                )
                n_scan_steps = opmNIDAQ_custom.n_scan_steps
                
                #--------------------------------------------------------------------#
                #--------------------------------------------------------------------#
                #--------------------------------------------------------------------#
                # Create CustomAction events
                # - O2O3 autofocus
                # - AO mirror optimization
                # - Fluidics program (Not Optimize only)
                # - DAQ hardware setup (Not Optimize only)
                # - Image events (to be sequenced, Not Optimize only)
                #--------------------------------------------------------------------#
                #--------------------------------------------------------------------#
                #--------------------------------------------------------------------#
                    
                #--------------------------------------------------------------------#
                # Create CustomAction autofocus O2O3 events
                #--------------------------------------------------------------------#
                
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
                
                #--------------------------------------------------------------------#
                # Create CustomAction events for running AO optimization
                #--------------------------------------------------------------------#
                
                AO_event = MDAEvent(
                    exposure = AO_exposure_ms,
                    action=CustomAction(
                        name="AO-projection",
                        data = {
                            "AO" : {
                                "opm_mode": str("projection"),
                                "channel_states": AO_channel_states,
                                "channel_powers" : AO_channel_powers,
                                "mode": AO_metric,
                                "iterations": AO_iterations,
                                "image_mirror_range_um" : AO_image_mirror_range_um,
                                "blanking": bool(True),
                                "apply_existing": bool(False),
                                "pos_idx":int(0),
                                "output_path":AO_save_path
                            },
                            "Camera" : {
                                "exposure_ms": AO_exposure_ms,
                                "camera_crop" : [
                                    updated_config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                                    updated_config["Camera"]["camera_center_y"] - int(AO_camera_crop_y//2),
                                    updated_config["Camera"]["camera_crop_x"],
                                    AO_camera_crop_y
                                ]
                            }
                        }
                    )
                )
                
                #--------------------------------------------------------------------#
                # Create CustomAction Fluidics program
                #--------------------------------------------------------------------#

                # Create Fluidics program event
                FP_event = MDAEvent(
                    # exposure = AO_exposure_ms,
                    action=CustomAction(
                        name="Fluidics",
                        data = {
                            "Fluidics" : {
                                "mode" : FP_mode,
                                "round" : int(0)
                            }
                            }
                    )
                )
                
                #--------------------------------------------------------------------#
                # Create CustomAction DAQ event for Projection and Mirror imaging modes
                #--------------------------------------------------------------------#

                # Get daq waveform parameters
                if "Projection" in opm_mode:
                    n_scan_steps = 1
                    interleaved_acq = False
                    daq_mode = "projection"
                    laser_blanking = True
                    event_name = "DAQ-projection"
                    camera_crop_y = calculate_projection_crop(image_mirror_range_um)
                    
                elif "Mirror" in opm_mode:
                    daq_mode = "mirror"
                    event_name = "DAQ-mirror"
                    
                # create DAQ hardware setup event
                DAQ_event = MDAEvent(
                    action=CustomAction(
                        name=event_name,
                        data = {
                            "DAQ" : {
                                "mode" : daq_mode,
                                "image_mirror_step_um" : float(image_mirror_step_um),
                                "image_mirror_range_um" : float(image_mirror_range_um),
                                "channel_states" : active_channels,
                                "channel_powers" : laser_powers,
                                "interleaved" : interleaved_acq,
                                "blanking" : laser_blanking, 
                            },
                            "Camera" : {
                                "exposure_channels" : exposure_channels,
                                "camera_crop" : [
                                    updated_config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                                    updated_config["Camera"]["camera_center_y"] - int(camera_crop_y//2),
                                    updated_config["Camera"]["camera_crop_x"],
                                    int(camera_crop_y),
                                ]
                            }
                        }
                    )
                )
                
                #----------------------------------------------------------------#
                #----------------------------------------------------------------#
                # Else populate a {t, p, c, z} event sequence
                # - Run fliudics, if present
                # - Run O2O3 AF before time points or positions
                # - Run AO before each acq or positions
                # - Program the daq
                # - Acquire images
                #----------------------------------------------------------------#
                #----------------------------------------------------------------#
                
                opm_events: list[MDAEvent] = []
                    
                # Flags to help ensure sequence-able events are kept together 
                need_to_setup_DAQ = True
                need_to_setup_stage = True
                
                # Check if running AF initially to run the autofocus
                if O2O3_mode=="Initial-only":
                    opm_events.append(O2O3_event)
                
                # setup nD mirror-based AO-OPM acquisition event structure
                for time_idx in range(n_time_steps):
                    # TODO Clarify how the acquisition should run. 
                    # Right now, the first round is run manually in ESI, and then the imaging is setup afterwards. 
                    # This offsets the number of rounds and if running fluidics, we acquire the first round images then run fluidics at the second time point.
                    # Check if fluidics active
                    if not(FP_mode=="None") and not(time_idx==0):
                        current_FP_event = MDAEvent(**FP_event.model_dump())
                        current_FP_event.action.data["Fluidics"]["round"] = int(time_idx)
                        opm_events.append(current_FP_event)
                    
                    # Check if AF before each time point
                    if O2O3_mode == "Before-each-t":
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
                        if O2O3_mode == "Before-each-xyz":
                            opm_events.append(O2O3_event)
                            
                        # Check if run AO opt. before each XYZ on first time we see this position
                        if (AO_mode == "Before-each-xyz") and (time_idx == 0):
                            need_to_setup_DAQ = True
                            current_AO_event = MDAEvent(**AO_event.model_dump())
                            current_AO_event.action.data["AO"]["output_path"] = Path(output).parent / Path(f"pos_{pos_idx}_ao_optimize")
                            current_AO_event.action.data["AO"]["pos_idx"] = int(pos_idx)
                            current_AO_event.action.data["AO"]["apply_existing"] = False
                            opm_events.append(current_AO_event)
                            
                        # Apply correction for this position if time_idx > 0
                        elif (AO_mode == "Before-each-xyz") and (time_idx > 0):
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
                            # The DAQ waveform can be repeated for all time and spatial positions
                            if need_to_setup_DAQ:
                                need_to_setup_DAQ = True
                                opm_events.append(MDAEvent(**DAQ_event.model_dump()))
                                
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
                                                "current_channel" : active_channel_names[chan_idx]
                                            },
                                            "Camera" : {
                                                "exposure_ms" : float(exposure_channels[chan_idx]),
                                                "camera_center_x" : updated_config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                                                "camera_center_y" : updated_config["Camera"]["camera_center_y"] - int(camera_crop_y//2),
                                                "camera_crop_x" : updated_config["Camera"]["camera_crop_x"],
                                                "camera_crop_y" : int(camera_crop_y),
                                                "offset" : float(offset),
                                                "e_to_ADU": float(e_to_ADU)
                                            },
                                            "OPM" : {
                                                "angle_deg" : float(updated_config["OPM"]["angle_deg"]),
                                                "camera_Zstage_orientation" : str(updated_config["OPM"]["camera_Zstage_orientation"]),
                                                "camera_XYstage_orientation" : str(updated_config["OPM"]["camera_XYstage_orientation"]),
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
                            # Mirror scan each channel separately
                            for chan_idx, chan_bool in enumerate(active_channels):
                                temp_channels = [False,False,False,False,False]
                                if chan_bool:
                                    if need_to_setup_DAQ:
                                        # The DAQ has to be updated for every channel
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
                                                    "current_channel" : updated_config["OPM"]["channel_ids"][chan_idx]
                                                },
                                                "Camera" : {
                                                    "exposure_ms" : exposure_channels[chan_idx],
                                                    "camera_center_x" : updated_config["Camera"]["camera_center_x"] - int(config["Camera"]["camera_crop_x"]//2),
                                                    "camera_center_y" : updated_config["Camera"]["camera_center_y"] - int(camera_crop_y//2),
                                                    "camera_crop_x" : updated_config["Camera"]["camera_crop_x"],
                                                    "camera_crop_y" : int(camera_crop_y),
                                                    "offset" : float(offset),
                                                    "e_to_ADU": float(e_to_ADU)
                                                },
                                                "OPM" : {
                                                    "angle_deg" : float(updated_config["OPM"]["angle_deg"]),
                                                    "camera_Zstage_orientation" : str(updated_config["OPM"]["camera_Zstage_orientation"]),
                                                    "camera_XYstage_orientation" : str(updated_config["OPM"]["camera_XYstage_orientation"]),
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

                # Check if path ends if .zarr. If so, use our OutputHandler
                if len(Path(output).suffixes) == 1 and Path(output).suffix == ".zarr":

                    if "Mirror" in opm_mode:
                        # Create dictionary of maximum axes sizes.
                        indice_sizes = {
                            't' : int(np.maximum(1,n_time_steps)),
                            'p' : int(np.maximum(1,n_stage_pos)),
                            'c' : int(np.maximum(1,n_active_channels)),
                            'z' : int(np.maximum(1,n_scan_steps))
                        }
                    elif "Projection" in opm_mode:
                        # TODO: need to test structure for acquiring projection images
                        # Create dictionary of maximum axes sizes.
                        indice_sizes = {
                            't' : int(np.maximum(1,n_time_steps)),
                            'p' : int(np.maximum(1,n_stage_pos)),
                            'c' : int(np.maximum(1,n_active_channels))
                        }
                        
                    # Setup modified tensorstore handler
                    handler = OPMMirrorHandler(
                        path=Path(output),
                        indice_sizes=indice_sizes,
                        delete_existing=True
                        )
                        
                    print(f"Using Qi2lab handler,\n  indices: {indice_sizes}")
                        
                else:
                    # If not, use built-in handler based on suffix
                    print("Using default handler")
                    handler = Path(output)
                    
                # tell AO mirror class where to save mirror information
                opmAOmirror_local.output_path = output.parents[0]

                # run MDA with our event structure and modified tensorstore handler 
                mda_widget._mmc.run_mda(opm_events, output=handler)

    # modify the method on the instance
    mda_widget.execute_mda = custom_execute_mda

    # Register the custom OPM MDA engine with mmc
    mmc.mda.set_engine(OPMEngine(mmc,config_path))

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
        if any(opmNIDAQ_setup_preview.channel_states):
            # Check OPM mode and set up NIDAQ accordingly
            opmNIDAQ_setup_preview.clear_tasks()
            opmNIDAQ_setup_preview.generate_waveforms()
            opmNIDAQ_setup_preview.program_daq_waveforms()
            opmNIDAQ_setup_preview.start_waveform_playback()

            
    # Connect the above callback to the event that a continuous sequence is starting
    # Because callbacks are blocking, our custom setup code is called before the preview mode starts. 
    mmc.events.continuousSequenceAcquisitionStarting.connect(setup_preview_mode_callback)


    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # -----------------End custom qi2lab code for running OPM control-----------------
    # --------------------------------------------------------------------------------
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