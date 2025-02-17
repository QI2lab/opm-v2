"""OPM pymmcore-plus MDA Engine

TO DO: link this up with generating MDAEvents in the modified _opm_app.py

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

class OPMENGINE(MDAEngine):
   
    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Setup state of system (hardware, etc.) before an MDA is run.

        This method is called once at the beginning of a sequence.
        (The sequence object needn't be used here if not necessary)
        """
        print(sequence)
        super().setup_sequence(sequence)
        
        """
        check MDASequence for active laser channels
        check MDASequence for mirror vs stage vs projection vs fluidics-stage mode 
            alternatively, user can define this using a demo state device and we can check that
        if mirror:
            check MDASequence for relative z-range
            setup imaging galvo waveforms
            setup digital blanking waveforms
            start playback
        elif projection:
            check MDASequence for relative z-range
            setup imaging galvo and projection galvo waveforms
            setup digital blanking waveforms
            start playback
        elif stage:
            check XY stage settings for min/max setup
            check Z positions for coverslip tilt
            calculate scan length and z positions
            setup digital blanking waveforms
            set camera to external trigger mode
        elif AO-projection:
            check MDASequence for relative z-range
            setup imaging galvo and projection galvo waveforms
            setup digital blanking waveforms
            start playback
        elif fluidics-stage:
            load fluidics program
            check XY stage settings for min/max setup
            check Z positions for coverslip tilt
            calculate scan length and z positions
            setup digital blanking waveforms
            set camera to external trigger mode
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
        channel_names = ["405nm","488nm","561nm","637nm","730nm"]

        if isinstance(event.action,CustomAction):

            action_name = event.action.name
            data_dict = event.action.data

            if action_name == "O2O3-autofocus":
                self.mmc.setROI(
                    data_dict["Camera"]["camera_crop"][0],
                    data_dict["Camera"]["camera_crop"][1],
                    data_dict["Camera"]["camera_crop"][2],
                    data_dict["Camera"]["camera_crop"][3],
                )
                self.mmc.waitForDevice("OrcaFusionBT")
                self.mmc.setProperty(
                    "OrcaFusionBT", 
                    "Exposure", 
                    float(data_dict["Camera"]["exposure_ms"])
                )
                self.mmc.waitForDevice("OrcaFusionBT")
                if opmDAQ_setup.running():
                    opmDAQ_setup.stop_waveform_playback()
            elif action_name == "AO-projection":
                if data_dict["AO"]["apply_existing"]:
                    pass
                else:
                    self.mmc.setROI(
                        data_dict["Camera"]["camera_crop"][0],
                        data_dict["Camera"]["camera_crop"][1],
                        data_dict["Camera"]["camera_crop"][2],
                        data_dict["Camera"]["camera_crop"][3],
                    )
                    self.mmc.setProperty(
                        "OrcaFusionBT", 
                        "Exposure", 
                        float(data_dict["Camera"]["exposure_ms"])
                    )
                    self.mmc.waitForDevice("OrcaFusionBT")
                    self.mmc.setConfig(
                        "Laser-"+str(data_dict["AO"]["active_channel"])+"-power",
                        float(data_dict["AO"]["laser_power"])
                    )
                    self.mmc.waitForDevice("OrcaFusionBT")
            elif action_name == "DAQ-projection":
                self.mmc.setROI(
                    data_dict["Camera"]["camera_crop"][0],
                    data_dict["Camera"]["camera_crop"][1],
                    data_dict["Camera"]["camera_crop"][2],
                    data_dict["Camera"]["camera_crop"][3],
                )
                self.mmc.waitForDevice("OrcaFusionBT")
                for chan_idx, channel in enumerate(channel_names):
                    self.mmc.setConfig(
                        "Laser-"+str(channel)+"-power",
                        float(data_dict["DAQ"]["laser_powers"][chan_idx])
                    )
                opmDAQ_setup.set_acquisition_params(
                    scan_type = str(data_dict["DAQ"]["mode"]),
                    channel_states = data_dict["DAQ"]["active_channels"],
                    image_mirror_step_size_um = float(data_dict["DAQ"]["image_mirror_step_um"]),
                    image_mirror_sweep_um = float(data_dict["DAQ"]["image_mirror_range_um"]),
                    laser_blanking = bool(data_dict["DAQ"]["blanking"]),
                    exposure_ms = float(data_dict["Camera"]["exposure_ms"])
                )
            elif action_name == "DAQ-mirror":
                self.mmc.setROI(
                    data_dict["Camera"]["camera_crop"][0],
                    data_dict["Camera"]["camera_crop"][1],
                    data_dict["Camera"]["camera_crop"][2],
                    data_dict["Camera"]["camera_crop"][3],
                )
                self.mmc.waitForDevice("OrcaFusionBT")
                for chan_idx, channel in enumerate(channel_names):
                    self.mmc.setConfig(
                        "Laser-"+str(channel)+"-power",
                        float(data_dict["DAQ"]["laser_powers"][chan_idx])
                    )
                opmDAQ_setup.set_acquisition_params(
                    scan_type = str(data_dict["DAQ"]["mode"]),
                    channel_states = data_dict["DAQ"]["active_channels"],
                    image_mirror_step_size_um = float(data_dict["DAQ"]["image_mirror_step_um"]),
                    image_mirror_sweep_um = float(data_dict["DAQ"]["image_mirror_range_um"]),
                    laser_blanking = bool(data_dict["DAQ"]["blanking"]),
                    exposure_ms = float(data_dict["Camera"]["exposure_ms"])
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
        opmAOmirror_exec = AOMirror.instance()

        if isinstance(event.action,CustomAction):
            action_name = event.action.name
            data_dict = event.action.data

            if action_name == "O2O3-autofocus":
                # execute autofocus
                print(action_name)
            elif action_name == "AO-projection":
                if data_dict["AO"]["apply_existing"]:
                    wfc_positions_to_use = opmAOmirror_exec.wfc_positions_array[int(data_dict["AO"]["pos_idx"])]
                    opmAOmirror_exec.update_mirror_positions(wfc_positions_to_use)
                else:
                    run_ao_optimization(
                        image_mirror_step_size_um=float(data_dict["AO"]["image_mirror_step_um"]),
                        image_mirror_sweep_um=float(data_dict["AO"]["image_mirror_range_um"]),
                        exposure_ms=float(data_dict["AO"]["exposure_ms"]),
                        channel_states=data_dict["AO"]["active_channels"],
                        num_iterations=int(data_dict["AO"]["iterations"]),
                        pos_idx=int(data_dict["AO"]["pos_idx"])
                    )
            elif action_name == "DAQ-projection" or action_name == "DAQ-mirror":
                opmDAQ_exec.generate_waveforms()
                opmDAQ_exec.prepare_waveform_playback()
                opmDAQ_exec.start_waveform_playback()
            elif action_name == "Fluidics":
                print(action_name)
        else:
            result = super().exec_event(event)
            return result
        
    def teardown_event(self, event):

        super().teardown_event(event)
        
    def teardown_sequence(self, sequence: MDASequence) -> None:

        opmDAQ_teardown = OPMNIDAQ.instance()
        if opmDAQ_teardown.running():
            opmDAQ_teardown.stop_waveform_playback()
            opmDAQ_teardown.clear_tasks()

        super().teardown_sequence(sequence)