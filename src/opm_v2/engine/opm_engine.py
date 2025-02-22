"""OPM pymmcore-plus MDA Engine

TO DO: Fix init so we only have one instance of OPMNIDAQ, OPMAOMIRROR, and config is not global.

Change Log:
2025-02-07: New version that includes all possible modes
"""

from pymmcore_plus.mda import MDAEngine
from useq import MDAEvent, MDASequence, CustomAction
from typing import TYPE_CHECKING, Iterable
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.AOMirror import AOMirror
from pymmcore_plus.metadata import (
    FrameMetaV1,
    PropertyValue,
    SummaryMetaV1,
    frame_metadata,
    summary_metadata,
)
from numpy.typing import NDArray
from opm_v2.utils.sensorless_ao import run_ao_optimization
from opm_v2.utils.autofocus_remote_unit import manage_O3_focus
import json
from pathlib import Path
import numpy as np



class OPMEngine(MDAEngine):
    def __init__(self, mmc, config_path: Path, use_hardware_sequencing: bool = True) -> None:

        super().__init__(mmc, use_hardware_sequencing)

        self.opmDAQ = OPMNIDAQ.instance()
        self.AOMirror = AOMirror.instance()

        with open(config_path, "r") as config_file:
            self._config = json.load(config_file)

    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Setup state of system (hardware, etc.) before an MDA is run.

        This method is called once at the beginning of a sequence.
        (The sequence object needn't be used here if not necessary)
        """
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
                # Stop DAQ playback and clear 
                self.opmDAQ.stop_waveform_playback()
                self.opmDAQ.clear_tasks()
                self.opmDAQ.reset()               
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
                self._mmc.setPosition(np.round(float(data_dict["Stage"]["z_pos"]),2))
                self._mmc.waitForDevice(self._mmc.getFocusDevice())
                self._mmc.setXYPosition(
                    np.round(float(data_dict["Stage"]["x_pos"]),2),
                    np.round(float(data_dict["Stage"]["y_pos"]),2)
                )
                self._mmc.waitForDevice(self._mmc.getXYStageDevice())
                                
            elif action_name == "AO-projection":
                
                if data_dict["AO"]["apply_existing"]:
                    print("load AO-projection\n")
                    pass
                else:
                    self.opmDAQ.stop_waveform_playback()
                    self.opmDAQ.clear_tasks()
                    
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
                    
                    for chan_idx, chan_bool in enumerate(data_dict["AO"]["active_channels"]):
                        if chan_bool:
                            self._mmc.setProperty(
                                self._config["Lasers"]["name"],
                                str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                                float(data_dict["AO"]["laser_powers"][chan_idx])
                            )
                        else:
                            self._mmc.setProperty(
                                self._config["Lasers"]["name"],
                                str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                                0.0
                            )
            
            elif "DAQ" in action_name:
                self.opmDAQ.stop_waveform_playback()
                self.opmDAQ.clear_tasks()
                self._mmc.clearROI()
                self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                self._mmc.setROI(
                    data_dict["Camera"]["camera_crop"][0],
                    data_dict["Camera"]["camera_crop"][1],
                    data_dict["Camera"]["camera_crop"][2],
                    data_dict["Camera"]["camera_crop"][3],
                )
                self._mmc.waitForDevice(str(self._config["Camera"]["camera_id"]))
                
                for chan_idx, chan_bool in enumerate(data_dict["DAQ"]["active_channels"]):
                    if chan_bool:
                        self._mmc.setProperty(
                            self._config["Lasers"]["name"],
                            str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                            float(data_dict["DAQ"]["laser_powers"][chan_idx])
                        )
                        exposure_ms = np.round(float(data_dict["Camera"]["exposure_channels"][chan_idx]),0)
                    else:
                        self._mmc.setProperty(
                            self._config["Lasers"]["name"],
                            str(self._config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                            0.0
                        )
                self._mmc.setProperty(
                    str(self._config["Camera"]["camera_id"]), 
                    "Exposure", 
                    exposure_ms
                )
                self.opmDAQ.set_acquisition_params(
                    scan_type = str(data_dict["DAQ"]["mode"]),
                    channel_states = data_dict["DAQ"]["active_channels"],
                    image_mirror_step_size_um = float(data_dict["DAQ"]["image_mirror_step_um"]),
                    image_mirror_range_um = float(data_dict["DAQ"]["image_mirror_range_um"]),
                    laser_blanking = bool(data_dict["DAQ"]["blanking"]),
                    exposure_ms = exposure_ms
                )
                
            elif action_name == "Fluidics":
                print(action_name)
        else:
            super().setup_event(event)
                                        
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
                manage_O3_focus(self._config["O2O3-autofocus"]["O3_stage_name"])
                
            elif action_name == "AO-projection":               
                if data_dict["AO"]["apply_existing"]:
                    wfc_positions_to_use = self.AOMirror.wfc_positions_array[int(data_dict["AO"]["pos_idx"])]
                    self.AOMirror.set_mirror_positions(wfc_positions_to_use)
                else:
                    self.AOMirror.output_path = data_dict["AO"]["output_path"]
                    run_ao_optimization(
                        image_mirror_step_size_um=float(data_dict["AO"]["image_mirror_step_um"]),
                        image_mirror_range_um=float(data_dict["AO"]["image_mirror_range_um"]),
                        exposure_ms=float(data_dict["Camera"]["exposure_ms"]),
                        channel_states=data_dict["AO"]["active_channels"],
                        num_iterations=int(data_dict["AO"]["iterations"]),
                        save_dir_path=data_dict["AO"]["output_path"],
                        verbose=True
                    )
                    self.AOMirror.wfc_positions_array[int(data_dict["AO"]["pos_idx"]),:] = self.AOMirror.current_positions.copy()
            elif "DAQ" in action_name:
                self.opmDAQ.generate_waveforms()
                self.opmDAQ.prepare_waveform_playback()
                self.opmDAQ.start_waveform_playback()
            # elif action_name == "Fluidics":
            #     print(action_name)
        else:
            result = super().exec_event(event)
            return result
        
    def teardown_event(self, event):
        super().teardown_event(event)
        
    def teardown_sequence(self, sequence: MDASequence) -> None:
        super().teardown_sequence(sequence)
        
        # Shut down DAQ
        self.opmDAQ.stop_waveform_playback()
        self.opmDAQ.clear_tasks()
        self.opmDAQ.reset()
        
        # Set all lasers to zero emission
        for laser in self._config["Lasers"]["laser_names"]:
            self._mmc.setProperty(
                self._config["Lasers"]["name"],
                laser + " - PowerSetpoint (%)",
                0.0
            )
            
        self.AOMirror.save_acq_positions()
