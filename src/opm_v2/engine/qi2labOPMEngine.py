"""OPM pymmcore-plus MDA Engine

TO DO: everything but fluidics. For AO-projection, might need to have a function that 
generates the MDA events for the expected number of images to snap based on AO settings.
Should be able to use demo devices to setup the AO algorithm and grab those settings.

What I don't quite understand is how to make the callback so that it modifies the MDASequence 
when the MDA is started. I'll figure it out.

Change Log:
2025-02-07: New version that includes all possible modes
"""

from pymmcore_plus.mda import MDAEngine
from useq import MDAEvent, MDASequence, SummaryMetaV1, FrameMetaV1
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from numpy.typing import NDArray

class qi2labOPMEngine(MDAEngine):
    def __init__(self, opmNidaq, opmAOmirror, *args, **kwds):
        """Initialize the engine with the OPM NIDAQ and AO mirror.
        
        Parameters
        ----------
        opmNidaq : OPMNIDAQ
            class that controls the NIDAQ for the OPM system
        opmAOmirror : AOMIRROR
            class that controls the AO mirror for the OPM system
        
        """

        """
        check MDAsequence to see if it needs to be modified
        if mirror:
            pull in the MDASequence and modify it to include the image mirror settings
        elif projection:
            pull in the MDASequence and modify it to include the image mirror settings and the projection mirror settings
        elif stage:
            pull in the MDASequence and modify it to include the stage settings
        elif AO-projection:
            pull in the MDASequence and modify it to include the image mirror settings and the projection mirror settings
            Will also need to figure out how to run AO. Should be in the setup_sequence or setup_event functions
        elif fluidics-stage:
            pull in the MDASequence and modify it to include the fluidics settings, stage settings, and AO settings
        """

        self.opmNIDAQ = opmNidaq
        self.opmAOmirror = opmAOmirror

        return super().__call__(*args, **kwds)
    
    def setup_sequence(self, sequence: MDASequence) -> SummaryMetaV1 | None:
        """Setup state of system (hardware, etc.) before an MDA is run.

        This method is called once at the beginning of a sequence.
        (The sequence object needn't be used here if not necessary)
        """
        
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

    def setup_event(self, event: MDAEvent) -> None:
        """Prepare state of system (hardware, etc.) for `event`.

        This method is called before each event in the sequence. It is
        responsible for preparing the state of the system for the event.
        The engine should be in a state where it can call `exec_event`
        without any additional preparation.
        """

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
        
        """
        if AO-projection:
            run AO update using projection mode.
            Update mirror. Need to figure best strategy here. Probably need to pass in AO controller.
        if stage or fluidics-stage:
            Start ASI stage scan
        if mirror or projection:
            do nothing
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