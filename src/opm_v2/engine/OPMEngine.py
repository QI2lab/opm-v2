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

        if isinstance(event.action,CustomAction):
            action_type = event.action.name

            if action_type == "O2O3-autofocus":
                print(action_type)
                pass
            elif action_type == "AO-projection":
                print(action_type)
                pass
            elif action_type == "DAQ-projection":
                print(action_type)
                pass
            elif action_type == "DAQ-mirror":
                print(action_type)
                pass
            elif action_type == "Fluidics":
                print(action_type)
                pass
        else:
            super().setup_event(event)
                            
        """
        if mirror:
            move to new XY position if requested
            move to new Z position if requested
        elif projection:
            move to new XY position if requested
            move to new Z position if requested
        elif stage:
            move to next XYZ position
            run O2-O3 autofocus
            setup ASI controller for stage scan at this position
            setup ASI triggering for stage scan
            start digital waveform playback
            ensure camera is in external trigger mode
        elif AO-projection:
            move to new XY position if requested
            move to new Z position if requested
        elif fluidics-stage:
            execute this round's fluidics program
            run O2-O3 autofocus
            if round == 0:
                run AO-projection for all Z positions and starting XY positions. Store AO corrections.
            apply AO correction for this area
            run O2-O3 autofocus
            setup ASI controller for stage scan at this position
            setup ASI triggering for stage scan
            start digital waveform playback
            ensure camera is in external trigger mode
        """
            
    def exec_event(self, event: MDAEvent) -> Iterable[tuple[NDArray, MDAEvent, FrameMetaV1]]:
        """Execute `event`.

        This method is called after `setup_event` and is responsible for
        executing the event. The default assumption is to acquire an image,
        but more elaborate events will be possible.
        """

        if isinstance(event.action,CustomAction):
            if event.action.name == "AO-projection":
                data_dict = event.action.data
                run_ao_optimization(
                        image_mirror_step_size_um=float(data_dict["AO-image_mirror_step_um"]),
                        image_mirror_sweep_um=float(data_dict["AO-image_mirror_range_um"]),
                        exposure_ms=float(data_dict["AO-exposure_ms"][1]),
                        channel_states=data_dict["AO-active_channels_bool"],
                        num_iterations=data_dict["AO-iterations"]
                    )
        else:
            result = super().exec_event(event)
            return result
        
        """
        if stage or fluidics-stage:
            Start ASI stage scan
        if mirror or projection:
            capture image
        """


    def teardown_event(self, event):
        """
        if mirror:
            do nothing
        elif projection:
            do nothing
        elif stage:
            stop digital playback
        elif AO-projection:
            do nothing
        """

        super().teardown_event(event)
        
    def teardown_sequence(self, sequence: MDASequence) -> None:

        """
        if mirror:
            stop playback
        elif projection:
            stop playback
        elif stage:
            set camera back to internal trigger mode
        elif AO-projection:
            stop playback
        """

        super().teardown_sequence(sequence)