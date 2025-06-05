"""OPM pymmcore-plus MDA Engine

TO DO: Fix init so we only have one instance of OPMNIDAQ, OPMAOMIRROR, and config is not global.

Change Log:
2025-02-07: New version that includes all possible modes
"""
from datetime import datetime
from PyQt6.QtCore import QThread

from pymmcore_plus.mda import MDAEngine
from useq import MDAEvent, MDASequence, CustomAction
from typing import TYPE_CHECKING, Iterable
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.utils.elveflow_control import run_fluidic_program
from pymmcore_plus.metadata import (
    FrameMetaV1,
    SummaryMetaV1
)
from numpy.typing import NDArray
from opm_v2.utils.sensorless_ao import run_ao_optimization, run_ao_grid_mapping
from opm_v2.utils.autofocus_remote_unit import manage_O3_focus
import json
from pathlib import Path
import numpy as np
from time import sleep, perf_counter
import logging

logging.getLogger("pymmcore-plus")

DEBUGGING = True
class OPMEngine(MDAEngine):
    def __init__(self, mmc, config_path: Path, use_hardware_sequencing: bool = True) -> None:

        super().__init__(mmc, use_hardware_sequencing)

        self.opmDAQ = OPMNIDAQ.instance()
        self.AOMirror = AOMirror.instance()
        self.execute_stage_scan = False
        self.start_time = None
        self.elapsed_time = None
        
        with open(config_path, "r") as config_file:
            self._config = json.load(config_file)

    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Setup state of system (hardware, etc.) before an MDA is run.

        This method is called once at the beginning of a sequence.
        (The sequence object needn't be used here if not necessary)
        """
        self.start_time = perf_counter()
        self.elapsed_time = 0
        self._mmc.setCircularBufferMemoryFootprint(16000)
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
            data_dict = event.action.data

            if action_name == "O2O3-autofocus":
                # Stop DAQ playback
                if self.opmDAQ.running():
                    self.opmDAQ.stop_waveform_playback()
                
                # Setup camera properties
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
                    self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                self._mmc.setProperty(
                    str(self._config["Camera"]["camera_id"]), 
                    "Exposure", 
                    np.round(float(data_dict["Camera"]["exposure_ms"]),0)
                )
                self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                
            elif action_name == "Stage-Move":
                #--------------------------------------------------------#
                # Move stage to position 
                stage_move_speed = self._config['OPM']['stage_move_speed']
                self._mmc.setProperty(self._mmc.getXYStageDevice(),"MotorSpeedX-S(mm/s)",stage_move_speed)
                self._mmc.setProperty(self._mmc.getXYStageDevice(),"MotorSpeedY-S(mm/s)",stage_move_speed)
                self._mmc.setPosition(np.round(float(data_dict["Stage"]["z_pos"]),2))
                self._mmc.waitForDevice(self._mmc.getFocusDevice())
                target_x = np.round(float(data_dict["Stage"]["x_pos"]),2) 
                target_y = np.round(float(data_dict["Stage"]["y_pos"]),2)
                current_x, current_y = self._mmc.getXYPosition()
                old_x = current_x
                old_y = current_y
                self._mmc.setXYPosition(target_x,target_y)
                counter = 0
                # Move stage and wait until we are within 1um of the target position.
                while not(np.isclose(current_x, target_x, 0., 1.0)) or not(np.isclose(current_y, target_y, 0., 1.0)):
                    sleep(.5)
                    current_x, current_y = self._mmc.getXYPosition()
                    if old_x == current_x and old_y == current_y:
                        counter = counter + 1
                        if DEBUGGING:
                            print(
                                "Stage move stationary!",
                                f"\ncurrent_x:{current_x} current_y:{current_y}",
                                f"\ntarget_x:{target_x} target_y{target_y}"
                            )
                    else:
                        old_x = current_x
                        old_y = current_y
                    if counter >= 5:
                        break
                    
            elif action_name == "ASI-setupscan":
                #--------------------------------------------------------#
                # Setup PLC controller for TTL output to stage sync signal
                plcName = self._config["PLC"]["name"] # 'PLogic:E:36'
                propPosition = self._config["PLC"]["position"] # 'PointerPosition'
                propCellConfig = self._config["PLC"]["cellconfig"] # 'EditCellConfig'
                addrOutputBNC1 = int(self._config["PLC"]["pin"]) # 33 # BNC1 on the PLC front panel
                addrStageSync = int(self._config["PLC"]["signalid"]) # 46  # TTL5 on Tiger backplane = stage sync signal
                self._mmc.setProperty(plcName, propPosition, addrOutputBNC1)
                self._mmc.setProperty(plcName, propCellConfig, addrStageSync)
                
                #--------------------------------------------------------#
                # Set scan axis speed
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "MotorSpeedX-S(mm/s)",
                    np.round(data_dict["ASI"]["scan_axis_speed_mm_s"],4)
                )    
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "MotorSpeedY-S(mm/s)",
                    self._config['OPM']['stage_move_speed']
                )    

                #--------------------------------------------------------#
                # Set scan axis to true 1D scan with no backlash
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "ScanPattern",
                    "Raster"
                )
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "ScanSlowAxis",
                    "Null (1D scan)"
                )
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "ScanFastAxis",
                    "1st axis"
                )
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "ScanSettlingTime(ms)",
                    3000
                )
                
                #--------------------------------------------------------#
                # Set scan axis start/end positions
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "ScanFastAxisStartPosition(mm)",
                    np.round(data_dict["ASI"]["scan_axis_start_mm"],2)
                )
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "ScanFastAxisStopPosition(mm)",
                    np.round(data_dict["ASI"]["scan_axis_end_mm"],2)
                )
                
                
                #--------------------------------------------------------#
                # Set the scan state
                self._mmc.setProperty(
                    self._mmc.getXYStageDevice(),
                    "ScanState",
                    "Idle"
                )

                if DEBUGGING:
                    actual_speed_x = float(
                        self._mmc.getProperty(
                            self._mmc.getXYStageDevice(),
                            "MotorSpeedX-S(mm/s)"
                        )
                    )
                    print(
                        "\nScan positions:",
                        f"\n  start: {
                            self._mmc.getProperty(
                                self._mmc.getXYStageDevice(),
                                "ScanFastAxisStartPosition(mm)"
                            )}",
                        f"\n  end: {
                            self._mmc.getProperty(
                                self._mmc.getXYStageDevice(),
                                "ScanFastAxisStopPosition(mm)"
                            )}",
                        f"\n  Scan settling time: {
                            self._mmc.getProperty(
                                self._mmc.getXYStageDevice(),
                                "ScanSettlingTime(ms)"
                            )}",
                        f"\n  actual speed: {actual_speed_x}",
                        f"\n  requested speed: {np.round(data_dict["ASI"]["scan_axis_speed_mm_s"],4)}", 
                        f"\n  Do stage speeds match: {actual_speed_x==np.round(data_dict["ASI"]["scan_axis_speed_mm_s"],4)}"                     
                    )
                
                # put camera into external START trigger mode
                self._mmc.setProperty(self._config["Camera"]["camera_id"],"Trigger","START")
                self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                while not(self._mmc.getProperty(self._config["Camera"]["camera_id"],"Trigger") == "START"):
                    sleep(0.1)
                    self._mmc.setProperty(self._config["Camera"]["camera_id"],"Trigger","START")    
                    self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                
                self._mmc.setProperty(self._config["Camera"]["camera_id"],"TriggerPolarity","POSITIVE")
                self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                while not(self._mmc.getProperty(self._config["Camera"]["camera_id"],"TriggerPolarity") == "POSITIVE"):
                    sleep(0.1)
                    self._mmc.setProperty(self._config["Camera"]["camera_id"],"TriggerPolarity","POSITIVE")
                    self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                    
                
                self._mmc.setProperty(self._config["Camera"]["camera_id"],"TRIGGER SOURCE","EXTERNAL")
                self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                while not(self._mmc.getProperty(self._config["Camera"]["camera_id"],"TRIGGER SOURCE") == "EXTERNAL"):
                    sleep(.1)
                    self._mmc.setProperty(self._config["Camera"]["camera_id"],"TRIGGER SOURCE","EXTERNAL")
                    self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))

                # ready for a stage scan
                self.execute_stage_scan = True
                                
            elif action_name == "AO-optimize":
                #--------------------------------------------------------#
                # apply optimized mirror position
                if data_dict["AO"]["apply_existing"]:
                    pass
                
                #--------------------------------------------------------#
                # Set hardware state to run adaptive optics
                else:
                    # Clear DAQ tasks to re-program
                    self.opmDAQ.clear_tasks()
                    
                    # Setup camera properties
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
                    self._mmc.setProperty(
                        str(self._config["Camera"]["camera_id"]), 
                        "Exposure", 
                        np.round(float(data_dict["Camera"]["exposure_ms"]),0)
                    )
                    self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                    
                    # Set laser powers
                    for chan_idx, chan_bool in enumerate(data_dict["AO"]["channel_states"]):
                        if chan_bool:
                            self._mmc.setProperty(
                                self._config["Lasers"]["name"],
                                str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                                float(data_dict["AO"]["channel_powers"][chan_idx])
                            )
                        else:
                            self._mmc.setProperty(
                                self._config["Lasers"]["name"],
                                str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                                0.0
                            )
                                              
            elif action_name == "AO-grid":
                #--------------------------------------------------------#
                # apply optimized position
                if data_dict["AO"]["apply_ao_map"]:
                    print("\nAO: Applying existing mirror position\n\n")
                    pass
                
                #--------------------------------------------------------#
                # run adaptive optics over a grid of positions.                            
                else:
                    # Clear DAQ tasks to re-program
                    self.opmDAQ.clear_tasks()
                    
                    # Setup camera properties
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
                    self._mmc.setProperty(
                        str(self._config["Camera"]["camera_id"]), 
                        "Exposure", 
                        np.round(float(data_dict["Camera"]["exposure_ms"]),0)
                    )
                    self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                    
                    # Set laser powers
                    for chan_idx, chan_bool in enumerate(data_dict["AO"]["ao_dict"]["channel_states"]):
                        if chan_bool:
                            self._mmc.setProperty(
                                self._config["Lasers"]["name"],
                                str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                                float(data_dict["AO"]["ao_dict"]["channel_powers"][chan_idx])
                            )
                        else:
                            self._mmc.setProperty(
                                self._config["Lasers"]["name"],
                                str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                                0.0
                            )
                    
                    # Set ASI stage speed for moves
                    self._mmc.setProperty(
                        self._mmc.getXYStageDevice(),
                        "MotorSpeedX-S(mm/s)",
                        0.15
                    )    
                    self._mmc.setProperty(
                        self._mmc.getXYStageDevice(),
                        "MotorSpeedY-S(mm/s)",
                        0.15
                    )
            
            elif action_name == "DAQ":
                #--------------------------------------------------------#
                # Update daq waveform values and setup daq for playback
                self.opmDAQ.stop_waveform_playback()
                self.opmDAQ.clear_tasks()
                
                for chan_idx, chan_bool in enumerate(data_dict["DAQ"]["active_channels"]):
                    if chan_bool:
                        self._mmc.setProperty(
                            self._config["Lasers"]["name"],
                            str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                            float(data_dict["DAQ"]["channel_powers"][chan_idx])
                        )
                        exposure_ms = np.round(float(data_dict["Camera"]["exposure_channels"][chan_idx]),2)
                    else:
                        self._mmc.setProperty(
                            self._config["Lasers"]["name"],
                            str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                            0.0
                        )
                        
                if str(data_dict["DAQ"]["mode"]) == "stage":
                    self.opmDAQ.set_acquisition_params(
                        scan_type = "stage",
                        channel_states = data_dict["DAQ"]["channel_states"],
                        laser_blanking = bool(data_dict["DAQ"]["blanking"]),
                        exposure_ms = exposure_ms
                    )
                elif str(data_dict["DAQ"]["mode"]) == "projection":
                    self.opmDAQ.set_acquisition_params(
                        scan_type =  "projection",
                        channel_states = data_dict["DAQ"]["active_channels"],
                        image_mirror_range_um = float(data_dict["DAQ"]["image_mirror_range_um"]),
                        laser_blanking = bool(data_dict["DAQ"]["blanking"]),
                        exposure_ms = exposure_ms
                    )
                elif str(data_dict["DAQ"]["mode"]) == "mirror":
                    self.opmDAQ.set_acquisition_params(
                        scan_type = "mirror",
                        channel_states = data_dict["DAQ"]["channel_states"],
                        image_mirror_step_um = float(data_dict["DAQ"]["image_mirror_step_um"]),
                        image_mirror_range_um = float(data_dict["DAQ"]["image_mirror_range_um"]),
                        laser_blanking = bool(data_dict["DAQ"]["blanking"]),
                        exposure_ms = exposure_ms
                    )
                elif str(data_dict["DAQ"]["mode"]) == "2d":
                    self.opmDAQ.set_acquisition_params(
                        scan_type = "2d",
                        channel_states = data_dict["DAQ"]["channel_states"],
                        laser_blanking = bool(data_dict["DAQ"]["blanking"]),
                        exposure_ms = exposure_ms
                    )
                self.opmDAQ.generate_waveforms()
                self.opmDAQ.program_daq_waveforms()
                
                #--------------------------------------------------------#
                # Setup camera properties
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
                    self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                
                self._mmc.setProperty(
                    str(self._config["Camera"]["camera_id"]), 
                    "Exposure", 
                    exposure_ms
                )
                self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                
                # Wait for MM core
                self._mmc.waitForSystem()
                
                if DEBUGGING:
                    print(
                        "Camera Exposures:",
                        f"Actual: {np.round(self._mmc.getExposure(),2)}",
                        f"Requested: {exposure_ms}",
                    )
        else:
            super().setup_event(event)
            
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
            data_dict = event.action.data

            if action_name == "O2O3-autofocus":
                manage_O3_focus(self._config["O2O3-autofocus"]["O3_stage_name"], verbose=DEBUGGING)
                    
            elif action_name == "AO-optimize":               
                if data_dict["AO"]["apply_existing"]:
                    self.AOMirror.set_mirror_positions_from_array(int(data_dict["AO"]["pos_idx"]))
                    if DEBUGGING:
                        print(
                            '\nAO: updating mirror with existing positions:',
                            f'\n  pos: {int(data_dict["AO"]["pos_idx"])}',
                            f'\n  positions: {self.AOMirror.wfc_positions_array[int(data_dict["AO"]["pos_idx"])]}'
                        )
                else:
                    run_ao_optimization(
                        image_mirror_range_um=float(data_dict["AO"]["image_mirror_range_um"]),
                        exposure_ms=float(data_dict["Camera"]["exposure_ms"]),
                        channel_states=data_dict["AO"]["channel_states"],
                        metric_to_use=data_dict["AO"]["metric"],
                        daq_mode=data_dict["AO"]["daq_mode"],
                        num_iterations=int(data_dict["AO"]["iterations"]),
                        init_delta_range=float(data_dict["AO"]["modal_delta"]),
                        delta_range_alpha_per_iter=float(data_dict["AO"]["modal_alpha"]),
                        save_dir_path=data_dict["AO"]["output_path"],
                        verbose=DEBUGGING
                    )
                    try:
                        self.AOMirror.wfc_positions_array[int(data_dict["AO"]["pos_idx"]),:] = self.AOMirror.current_positions.copy()
                        if DEBUGGING:
                            print(
                                '\nAO: Saving positions to array:',
                                f'\n  pos_idx: {int(data_dict["AO"]["pos_idx"])}',
                                f'\n  mirror positions: {self.AOMirror.wfc_positions_array[int(data_dict["AO"]["pos_idx"]),:]}'
                            )
                    except Exception:
                        print("\nAO: Not setting ao positions array")
                        
            elif action_name == "AO-grid":    
                if data_dict["AO"]["apply_ao_map"]:
                    self.AOMirror.set_mirror_positions_from_array(int(data_dict["AO"]["pos_idx"]))
                    if DEBUGGING:
                        print(
                            '\nAO: updating mirror with existing positions:',
                            f'\n  pos: {int(data_dict["AO"]["pos_idx"])}',
                            f'\n  positions: {self.AOMirror.wfc_positions_array[int(data_dict["AO"]["pos_idx"])]}'
                        )
                else:
                    run_ao_grid_mapping(
                        stage_positions = data_dict["AO"]["stage_positions"],
                        ao_dict = data_dict["AO"]["ao_dict"],
                        num_tile_positions = data_dict["AO"]["num_tile_positions"],
                        num_scan_positions = data_dict["AO"]["num_scan_positions"],
                        save_dir_path = data_dict["AO"]["output_path"],
                        verbose = DEBUGGING,
                    )
                                       
            elif action_name == "DAQ":
                self.opmDAQ.start_waveform_playback()
                
            elif action_name == "Fluidics":
                print("\nSending ttl pulse to OB1 to CLEAVE and apply READOUTS")
                run_fluidic_program(True)
            
            elif action_name == "Timelapse":
                interval = data_dict['plan']['interval']
                self.elapsed_time = perf_counter() - self.start_time
                sleep_time = interval - self.elapsed_time
                if sleep_time<0:
                    sleep_time = 0
                    if DEBUGGING:
                        print(
                            '\nImaging did not finish before next timepoint!'
                        )
                QThread.sleep(int(interval - self.elapsed_time))
                self.start_time = perf_counter() 
                
                if DEBUGGING:
                    print(
                        '\nTimelapse:',
                        f"elapsed: {self.elapsed_time}",
                        f"start time: {self.start_time}",
                        f"requested interval: {interval}",
                        f'sleep time: {sleep_time}'
                    )

        else:
            result = super().exec_event(event)
            return result
        
    def teardown_event(self, event):
        if isinstance(event.action, CustomAction):
            self._mmc.clearCircularBuffer()
        super().teardown_event(event)
        
    def teardown_sequence(self, sequence: MDASequence) -> None:
        if DEBUGGING:
            print("Acq finished, tearing down.")
        
        # Shut down DAQ
        self.opmDAQ.clear_tasks()
        self.opmDAQ.reset()

        if DEBUGGING:
            print("Daq reset")

        # Put camera back into internal mode
        self._mmc.setProperty(self._config["Camera"]["camera_id"],"TriggerPolarity","POSITIVE")
        self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
        self._mmc.setProperty(self._config["Camera"]["camera_id"],"TRIGGER SOURCE","INTERNAL")
        self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))

        self._mmc.setProperty(self._mmc.getXYStageDevice(),"MotorSpeedX-S(mm/s)",0.2)
        self._mmc.setProperty(self._mmc.getXYStageDevice(),"MotorSpeedY-S(mm/s)",0.2)
                
        # Set all lasers to zero emission
        for laser in self._config["Lasers"]["laser_names"]:
            self._mmc.setProperty(
                self._config["Lasers"]["name"],
                laser + " - PowerSetpoint (%)",
                0.0
            )
        
        # save mirror positions array
        self.AOMirror.save_wfc_positions_array()
        self._mmc.clearCircularBuffer()

        super().teardown_sequence(sequence)