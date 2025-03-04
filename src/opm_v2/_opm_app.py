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

    # load hardware configuration file
    config_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\opm_config_20250228.json")
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
    
    opmAOmirror.set_mirror_positions_flat()

    # load OPM NIDAQ and OPM AO mirror classes
    opmNIDAQ = OPMNIDAQ(
        name = str(config["NIDAQ"]["name"]),
        scan_type = str(config["NIDAQ"]["scan_type"]),
        exposure_ms = float(config["Camera"]["exposure_ms"]),
        laser_blanking = bool(config["NIDAQ"]["laser_blanking"]),
        image_mirror_calibration = float(str(config["NIDAQ"]["image_mirror_calibration"])),
        projection_mirror_calibration = float(str(config["NIDAQ"]["projection_mirror_calibration"])),
        image_mirror_neutral_um = float(str(config["NIDAQ"]["image_mirror_neutral_um"])),
        projection_mirror_neutral_um = float(str(config["NIDAQ"]["projection_mirror_neutral_um"])),
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
    
    def update_state():
        """
        Update microscope states and values upon changes to in the GUI
        """
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
        
        # Get the selected active channel and populate channel_states
        _active_channel_id = mmc.getProperty("LED", "Label")
        _channel_states = [False] * len(config["OPM"]["channel_ids"])
        for ch_i, ch_str in enumerate(config["OPM"]["channel_ids"]):
            if _active_channel_id==ch_str:
                _channel_states[ch_i] = True
        
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
            #     mmc.waitForDevice(str(config["Camera"]["camera_id"]))
                
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
        # opmNIDAQ_update_state.generate_waveforms()
        # opmNIDAQ_update_state.prepare_waveform_playback()
        
        #--------------------------------------------------------------------#
        # Restart acquisition if needed
        #--------------------------------------------------------------------#
        
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
            print("No fluidics")
        elif "Thin-16bit" in mmc.getProperty("Fluidics-mode", "Label"):
            FP_mode = "thin_16bit"
            FP_num_rounds = 16
            print("Thin 16bit fluidics")
        elif "Thin-22bit" in mmc.getProperty("Fluidics-mode", "Label"):
            FP_mode = "thin_22bit"
            FP_num_rounds = 22
            print("Thin 22bit fluidics")
        elif "Thick-16bit" in mmc.getProperty("Fluidics-mode", "Label"):
            FP_mode = "thick_16bit"
            FP_num_rounds = 16
            print("Thick 16bit fluidics")
        elif "Thick-2bit" in mmc.getProperty("Fluidics-mode", "Label"):
            FP_mode = "thick_22bit"
            FP_num_rounds = 22
            print("Thick 22bit fluidics")        
            
        # get image galvo mirror range and step size
        image_mirror_range_um = np.round(float(mmc.getProperty("ImageGalvoMirrorRange", "Position")),0)
        image_mirror_step_um = np.round(float(mmc.getProperty("ImageGalvoMirrorStep", "Label").split("-")[0]),2)
        
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

        # Get the current MDAsequence and convert to dictionary 
        sequence = mda_widget.value()
        sequence_dict = json.loads(sequence.model_dump_json())
        print(sequence_dict)
        # {'metadata': {'pymmcore_widgets': {'version': '0.9.1', 'save_dir': 'G:\\20250303_opm_ao_testing', 'save_name': 'test_938.zarr', 'format': 'tiff-sequence', 'should_save': True}}, 'axis_order': ['p'], 'stage_positions': [{'x': 634.54, 'y': -2864.91, 'z': -8089.17, 'name': None, 'sequence': None}], 'grid_plan': None, 'channels': [], 'time_plan': None, 'z_plan': None, 'autofocus_plan': None, 'keep_shutter_open_across': []}
        # extract the relevant portions for qi2lab-OPM
        # Stage positions
        # Extract stage positions from either (1) list of stage positions or (2) grid plan
        # For grid plan to work correctly, the microscope pixel size needs to be set
        mda_stage_positions = sequence_dict["stage_positions"]
        if mda_stage_positions is not None:
            if len(mda_stage_positions) == 1:
                # TODO is where is the grid setup
                grid_plan = sequence_dict["grid_plan"]
                
                if grid_plan is not None:
                    print(grid_plan.keys())
                    print(grid_plan)
                    stage_positions = []
                    for event in sequence:
                        json_event = json.loads(event.model_dump_json())
                        stage_positions.append({
                            'x': float(json_event['x_pos']),
                            'y': float(json_event['y_pos']),
                            'z': float(json_event['z_pos'])
                        })
                else:
                    stage_positions = [{
                        'x': mda_widget._mmc.getXPosition(),
                        'y': mda_widget._mmc.getYPosition(),
                        'z': mda_widget._mmc.getPosition(),
                    }]
            else:
                stage_positions = []
                for stage_pos in mda_stage_positions:
                    stage_positions.append({
                                'x': float(stage_pos['x']),
                                'y': float(stage_pos['y']),
                                'z': float(stage_pos['z'])
                            })
        else:
            grid_plan = sequence_dict["grid_plan"]
            if grid_plan is not None:
                stage_positions = []
                for event in sequence:
                    json_event = json.loads(event.model_dump_json())
                    stage_positions.append({
                        'x': float(json_event['x_pos']),
                        'y': float(json_event['y_pos']),
                        'z': float(json_event['z_pos'])
                    })
            else:
                return
        n_stage_pos = len(stage_positions)
        opmAOmirror_local.n_positions = n_stage_pos
        
        # timelapse values
        if not (FP_mode == "None"): 
            n_time_steps = FP_num_rounds
            time_interval = 0
        else:
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

        # if AO_mode is not optimize now, grab values from MDA window
        # if not("Optimize-now" in AO_mode):
        # Define channels, state and exposures from MDA
        mda_channels = sequence_dict["channels"] # list of checked channels in mda
        channel_names = config["OPM"]["channel_ids"] # list of all channel ids
        active_channels = [False] * len(channel_names) 
        exposure_channels = [0.] * len(channel_names) 
        laser_powers = [0.] * len(channel_names) 
        
        # Iterate through MDA checked channels and update active lasers list
        for mda_ch in mda_channels:
            # Get the matching channel idx active lasers list
            ch_id = mda_ch["config"]
            ch_idx = config["OPM"]["channel_ids"].index(ch_id)
            
            # update active channel and powers
            active_channels[ch_idx] = True
            exposure_channels[ch_idx] = mda_ch["exposure"]
            laser_powers[ch_idx] = float(
                mmc.getProperty(
                    "Coherent-Scientific Remote",
                    config["Lasers"]["laser_names"][ch_idx] + " - PowerSetpoint (%)"
                    )
                )

        n_active_channels = sum(active_channels)
        # Create lists containing only the active channel names and exposures
        active_channel_names = [_name for _, _name in zip(active_channels, channel_names) if _]
        active_channel_exps = [_exp for _, _exp in zip(active_channels, exposure_channels) if _]
    
        # Interleave only available if all channels have the same exposure.
        if len(set(active_channel_exps)) == 1:
            interleaved_acq = True
        else:
            interleaved_acq = False
    
        # laser blanking
        if "On" in mmc.getProperty("LaserBlanking","Label"):
            laser_blanking = True
        elif "Off" in mmc.getProperty("LaserBlanking","Label"):
            laser_blanking = False
        else:
            laser_blanking = True
        
        # get the gui y crop value
        camera_crop_y = int(mmc.getProperty("ImageCameraCrop","Label"))
        
        # reload hardware configuration file before setting up acq
        with open(config_path, "r") as config_file:
            updated_config = json.load(config_file)

        # Get the imaging mode from GUI
        opm_mode = mmc.getProperty("OPM-mode", "Label")
        
        # only projection and mirror modes are available
        if "Projection" not in opm_mode and "Mirror" not in opm_mode:
            raise ValueError("OPM-mode must be 'Projection' or 'Mirror'")        
        
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
        # Create CustomAction DAQ event for Projection and Mirror imaging modes
        #--------------------------------------------------------------------#

        # Get daq waveform parameters
        if "Projection" in opm_mode:
            n_scan_steps = 1
            interleaved_acq = False
            daq_mode = "projection"
            laser_blanking = True
            event_name = "DAQ-projection"
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
                        "active_channels" : active_channels,
                        "interleaved" : interleaved_acq,
                        "laser_powers" : laser_powers,
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
        
        #--------------------------------------------------------------------#
        # Create CustomAction events for running AO optimization
        #--------------------------------------------------------------------#
        now = datetime.now()
        timestamp = f"{now.year:4d}{now.month:2d}{now.day:2d}_{now.hour:2d}{now.minute:2d}{now.second:2d}"
        
        # setup AO using values in config.json
        if not(AO_mode == "Optimize-now"):
            AO_exposure_ms = round(float(updated_config["AO-projection"]["exposure_ms"]),0)
            AO_active_channels = list(map(bool,updated_config["AO-projection"]["active_channels"]))
            AO_laser_powers = list(float(_) for _ in updated_config["AO-projection"]["laser_power"])
            AO_camera_crop_y = int(calculate_projection_crop(updated_config["AO-projection"]["image_mirror_range_um"]))
            AO_image_mirror_range_um = float(updated_config["AO-projection"]["image_mirror_range_um"])
            AO_image_mirror_step_um = float(updated_config["AO-projection"]["image_mirror_step_um"])
            AO_iterations = int(updated_config["AO-projection"]["iterations"])
            AO_metric = str(updated_config["AO-projection"]["mode"])
            AO_save_path = Path(output).parent / Path(f"{timestamp}_ao_optimize")
            
        
        # setup AO using values in GUI config
        else:
            active_channel_id = mmc.getProperty("LED", "Label")
            AO_exposure_ms = np.round(float(mmc.getProperty("OrcaFusionBT", "Exposure")),0)
            AO_active_channels = [False] * len(channel_names) 
            AO_laser_powers = [0.] * len(channel_names)
            AO_image_mirror_range_um = float(image_mirror_range_um)
            AO_image_mirror_step_um = float(image_mirror_step_um)
            AO_camera_crop_y = int(calculate_projection_crop(image_mirror_range_um))
            AO_iterations = int(updated_config["AO-projection"]["iterations"])
            AO_metric = str(updated_config["AO-projection"]["mode"])
            AO_save_path = Path(str(updated_config["AO-projection"]["optimize_now_path"])) / Path(f"{timestamp}_ao_optimizeNOW")
        
            # Set the active channel 
            for chan_idx, chan_str in enumerate(config["OPM"]["channel_ids"]):
                if active_channel_id==chan_str:
                    AO_active_channels[chan_idx] = True
                    AO_laser_powers[chan_idx] = float(
                        mmc.getProperty(
                            config["Lasers"]["name"],
                            str(config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)"
                        )
                    )
            
            # Update the AO configuration with this run's values
            updated_config["AO-projection"]["active_channels"] = AO_active_channels
            updated_config["AO-projection"]["laser_power"] = AO_laser_powers
            updated_config["AO-projection"]["exposure_ms"] = AO_exposure_ms
            updated_config["AO-projection"]["image_mirror_step_um"] = AO_image_mirror_step_um
            updated_config["AO-projection"]["image_mirror_range_um"] = AO_image_mirror_range_um
            updated_config["AO-projection"]["camera_crop_y"] = AO_camera_crop_y

            with open(config_path, "w") as file:
                json.dump(updated_config, file, indent=4)
          
        # Create AO event
        AO_event = MDAEvent(
            exposure = AO_exposure_ms,
            action=CustomAction(
                name="AO-projection",
                data = {
                    "AO" : {
                        "opm_mode": str("projection"),
                        "active_channels": AO_active_channels,
                        "laser_powers" : AO_laser_powers,
                        "mode": AO_metric,
                        "iterations": AO_iterations,
                        "image_mirror_step_um" : AO_image_mirror_step_um,
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
        # Create event structure
        #--------------------------------------------------------------------#
        
        opm_events: list[MDAEvent] = []
        
        if not FP_mode=="None":
            # load dialog to interupt user.
            from PyQt6.QtWidgets import QMessageBox
            # this blocks the main thread until the dialog is closed
            response = QMessageBox.information(
                mda_widget,  # parent
                'WARNING ! ! ! ! FLUIDICS MUST BE RUNNING ! ! !',
                'IS ESI SEQUENCE LOADED AND STARTED?',
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            )

            #  `response` is which button was clicked
            if response is not QMessageBox.StandardButton.Ok:
                return
            else:
                print("ESI Sequence accepted")
                # opm_events.append(FP_event)
                
        # run O2-O3 autofocus if only initially
        if O2O3_mode=="Initial-only":
            opm_events.append(O2O3_event)
               
        if AO_mode == "Optimize-now":
            opm_events.append(AO_event)
            mda_widget._mmc.run_mda(opm_events, output=None)
            
            need_to_setup_stage = False
            need_to_setup_DAQ = False
            n_time_steps = 0
        else:
            
            #--------------------------------------------------------------------#
            # Create MDAevents for time and positions
            #--------------------------------------------------------------------#
            # Flags to help ensure sequence-able events are kept together 
            need_to_setup_DAQ = True
            need_to_setup_stage = True

        # setup nD mirror-based AO-OPM acquisition event structure
        for time_idx in range(n_time_steps):
            # Check if fluidics active
            if not(FP_mode=="None") and not(time_idx==0):
                
                # time_idx + 1 becuase the first round (round 0) was done manually
                current_FP_event = MDAEvent(**FP_event.model_dump())
                current_FP_event.action.data["Fluidics"]["round"] = int(time_idx+1)
                opm_events.append(current_FP_event)
            
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
                    if need_to_setup_DAQ:
                        # The DAQ waveform can be repeated for all time and space points
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
                            print("in mda")
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

            # elif "Stage" in mmc.getProperty("OPM-mode", "Label"):
            #     print("stage mode")

        # Check if path ends if .zarr. If so, use our OutputHandler
        if len(Path(output).suffixes) == 1 and Path(output).suffix == ".zarr":
            # Create dictionary of maximum axes sizes.
            indice_sizes = {
                't' : int(np.maximum(1,n_time_steps)),
                'p' : int(np.maximum(1,n_stage_pos)),
                'c' : int(np.maximum(1,n_active_channels)),
                'z' : int(np.maximum(1,n_scan_steps))
            }
            print(indice_sizes)
            # Setup modified tensorstore handler
            handler = OPMMirrorHandler(
                path=Path(output),
                indice_sizes=indice_sizes,
                delete_existing=True
                )
            print("using our handler")
        # If not, use built-in handler based on suffix
        else:
            handler = Path(output)

        # run MDA with our event structure and modified tensorstore handler 
        mda_widget._mmc.run_mda(opm_events, output=handler)

        # tell AO mirror class where to save mirror information
        opmAOmirror_local.output_path = output.parents[0]

    # modify the method on the instance
    mda_widget.execute_mda = custom_execute_mda

    # Register the custom OPM MDA engine with mmc
    mmc.mda.set_engine(OPMEngine(mmc,config_path))

    # This section sets up a callback to intercept the preview mode 
    # and setup the OPM accordingly.
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
        
    