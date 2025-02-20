"""OPM py_mmcore-plus MDA Engine

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


config_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\opm_config_20250218.json")
with open(config_path, "r") as config_file:
    config = json.load(config_file)

class OPMENGINE(MDAEngine):
   
    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Setup state of system (hardware, etc.) before an MDA is run.

        This method is called once at the beginning of a sequence.
        (The sequence object needn't be used here if not necessary)
        """

        print(sequence)
        super().setup_sequence(sequence)

    def setup_event(self, event: MDAEvent) -> None:
        """Prepare state of system (hardware, etc.) for `event`.

        This method is called before each event in the sequence. It is
        responsible for preparing the state of the system for the event.
        The engine should be in a state where it can call `exec_event`
        without any additional preparation.
        """

        opmDAQ_setup = OPMNIDAQ.instance()

        if isinstance(event.action,CustomAction):

            action_name = event.action.name
            data_dict = event.action.data

            if action_name == "O2O3-autofocus":
                opmDAQ_setup.stop_waveform_playback()
                opmDAQ_setup.clear_tasks()               
                self._mmc.clearROI()
                self._mmc.waitForDevice("OrcaFusionBT")
                self._mmc.setROI(
                    data_dict["Camera"]["camera_crop"][0],
                    data_dict["Camera"]["camera_crop"][1],
                    data_dict["Camera"]["camera_crop"][2],
                    data_dict["Camera"]["camera_crop"][3],
                )
                self._mmc.waitForDevice("OrcaFusionBT")
                self._mmc.setProperty(
                    "OrcaFusionBT", 
                    "Exposure", 
                    np.round(float(data_dict["Camera"]["exposure_ms"]),0)
                )
                self._mmc.waitForDevice("OrcaFusionBT")
                
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
                    opmDAQ_setup.stop_waveform_playback()
                    opmDAQ_setup.clear_tasks()
                    self._mmc.clearROI()
                    self._mmc.waitForDevice("OrcaFusionBT")
                    self._mmc.setROI(
                        data_dict["Camera"]["camera_crop"][0],
                        data_dict["Camera"]["camera_crop"][1],
                        data_dict["Camera"]["camera_crop"][2],
                        data_dict["Camera"]["camera_crop"][3],
                    )
                    self._mmc.setProperty(
                        "OrcaFusionBT", 
                        "Exposure", 
                        np.round(float(data_dict["Camera"]["exposure_ms"]),0)
                    )
                    self._mmc.waitForDevice("OrcaFusionBT")
                    for chan_idx, chan_bool in enumerate(data_dict["AO"]["active_channels"]):
                        if chan_bool:
                            self._mmc.setProperty(
                                config["Lasers"]["name"],
                                str(config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                                float(data_dict["AO"]["laser_power"][chan_idx])
                            )
                        else:
                            self._mmc.setProperty(
                                config["Lasers"]["name"],
                                str(config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                                0.0
                            )
                    self._mmc.waitForDevice("OrcaFusionBT")
                    
            elif action_name == "DAQ-projection":
                opmDAQ_setup.stop_waveform_playback()
                opmDAQ_setup.clear_tasks()
                self._mmc.clearROI()
                self._mmc.waitForDevice("OrcaFusionBT")
                self._mmc.setROI(
                    data_dict["Camera"]["camera_crop"][0],
                    data_dict["Camera"]["camera_crop"][1],
                    data_dict["Camera"]["camera_crop"][2],
                    data_dict["Camera"]["camera_crop"][3],
                )
                self._mmc.waitForDevice("OrcaFusionBT")
                for chan_idx, chan_bool in enumerate(data_dict["DAQ"]["active_channels"]):
                    if chan_bool:
                        self._mmc.setProperty(
                            config["Lasers"]["name"],
                            str(config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                            float(data_dict["DAQ"]["laser_powers"][chan_idx])
                        )
                        exposure_ms = np.round(float(data_dict["Camera"]["exposure_channels"][chan_idx]),0)
                    else:
                        self._mmc.setProperty(
                            config["Lasers"]["name"],
                            str(config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                            0.0
                        )
                self._mmc.setProperty(
                    "OrcaFusionBT", 
                    "Exposure", 
                    exposure_ms
                )
                opmDAQ_setup.set_acquisition_params(
                    scan_type = str(data_dict["DAQ"]["mode"]),
                    channel_states = data_dict["DAQ"]["active_channels"],
                    image_mirror_step_size_um = float(data_dict["DAQ"]["image_mirror_step_um"]),
                    image_mirror_sweep_um = float(data_dict["DAQ"]["image_mirror_range_um"]),
                    laser_blanking = bool(data_dict["DAQ"]["blanking"]),
                    exposure_ms = exposure_ms
                )
                
            elif action_name == "DAQ-mirror":
                opmDAQ_setup.stop_waveform_playback()
                opmDAQ_setup.clear_tasks()
                self._mmc.clearROI()
                self._mmc.waitForDevice("OrcaFusionBT")
                self._mmc.setROI(
                    data_dict["Camera"]["camera_crop"][0],
                    data_dict["Camera"]["camera_crop"][1],
                    data_dict["Camera"]["camera_crop"][2],
                    data_dict["Camera"]["camera_crop"][3],
                )
                self._mmc.waitForDevice("OrcaFusionBT")
                for chan_idx, chan_bool in enumerate(data_dict["DAQ"]["active_channels"]):
                    if chan_bool:
                        self._mmc.setProperty(
                            config["Lasers"]["name"],
                            str(config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                            float(data_dict["DAQ"]["laser_powers"][chan_idx])
                        )
                        exposure_ms = np.round(float(data_dict["Camera"]["exposure_channels"][chan_idx]),0)
                    else:
                        self._mmc.setProperty(
                            config["Lasers"]["name"],
                            str(config["Lasers"]["laser_names"][chan_idx]) + " - PowerSetpoint (%)",
                            0.0
                        )
                self._mmc.setProperty(
                    "OrcaFusionBT", 
                    "Exposure", 
                    exposure_ms
                )
                opmDAQ_setup.set_acquisition_params(
                    scan_type = str(data_dict["DAQ"]["mode"]),
                    channel_states = data_dict["DAQ"]["active_channels"],
                    image_mirror_step_size_um = float(data_dict["DAQ"]["image_mirror_step_um"]),
                    image_mirror_sweep_um = float(data_dict["DAQ"]["image_mirror_range_um"]),
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

        opmDAQ_exec = OPMNIDAQ.instance()
        # opmAOmirror_exec = AOMirror.instance()

        if isinstance(event.action,CustomAction):
   
            action_name = event.action.name
            data_dict = event.action.data

            if action_name == "O2O3-autofocus":
                manage_O3_focus(config["O2O3-autofocus"]["O3_stage_name"])
            # elif action_name == "AO-projection":
            #     if data_dict["AO"]["apply_existing"]:
            #         wfc_positions_to_use = opmAOmirror_exec.wfc_positions_array[int(data_dict["AO"]["pos_idx"])]
            #         opmAOmirror_exec.update_mirror_positions(wfc_positions_to_use)
            #     else:
            #         run_ao_optimization(
            #             image_mirror_step_size_um=float(data_dict["AO"]["image_mirror_step_um"]),
            #             image_mirror_sweep_um=float(data_dict["AO"]["image_mirror_range_um"]),
            #             exposure_ms=float(data_dict["AO"]["exposure_ms"]),
            #             channel_states=data_dict["AO"]["active_channels"],
            #             num_iterations=int(data_dict["AO"]["iterations"]),
            #             pos_idx=int(data_dict["AO"]["pos_idx"])
            #         )
            #         opmAOmirror_exec.wfc_positions_array[int(data_dict["AO"]["pos_idx"]),:] = opmAOmirror_exec.current_positions().copy()
            elif action_name == "DAQ-projection" or action_name == "DAQ-mirror":
                opmDAQ_exec.generate_waveforms()
                opmDAQ_exec.prepare_waveform_playback()
                opmDAQ_exec.start_waveform_playback()
            # elif action_name == "Fluidics":
            #     print(action_name)
        else:
            result = super().exec_event(event)
            return result
        
    def teardown_event(self, event):
        
        super().teardown_event(event)
        
    def teardown_sequence(self, sequence: MDASequence) -> None:

        opmDAQ_teardown = OPMNIDAQ.instance()
        opmDAQ_teardown.stop_waveform_playback()
        opmDAQ_teardown.clear_tasks()

        super().teardown_sequence(sequence)