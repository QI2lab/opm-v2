"""Exercise OPM acquisition failures through the real pymmcore-plus runner."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from pymmcore_plus.core._sequencing import SequencedEvent
from pymmcore_plus.mda import MDAEngine, MDARunner
from pymmcore_plus.mda._runner import FinishReason
from useq import CustomAction, MDAEvent

from opm_v2.engine.opm_custom_events import (
    ACTION_DAQ,
    ACTION_O2O3_AUTOFOCUS,
    ACTION_STAGE_MOVE,
    create_daq_event,
    create_o2o3_autofocus_event,
    create_stage_event,
)
from opm_v2.engine.opm_engine import (
    IncompleteHardwareSequenceError,
    OPMEngineV2,
)


def _runner_engine(
    demo_core,
    opm_config_factory,
    simulated_acquisition_hardware,
    workspace_tmp_path,
) -> OPMEngineV2:
    """Build an OPM engine with real dispatch and deterministic hardware seams.

    Returns
    -------
    OPMEngineV2
        Engine using real OPM event dispatch and mocked device-facing methods.
    """
    config = opm_config_factory(
        mode="stage",
        active_channels=(0,),
        channel_powers=(10.0,),
        channel_exposures_ms=(10.0,),
        camera_shape=(32, 16),
    )
    engine = OPMEngineV2(
        demo_core,
        workspace_tmp_path / "unused-config.json",
        config=config,
        simulate_hardware=True,
    )
    assert engine.opmDAQ is simulated_acquisition_hardware.daq
    engine._tile_retry_prepare = MagicMock()
    engine._tile_setup_events = {
        ACTION_STAGE_MOVE: create_stage_event({"x": 100, "y": 200, "z": 30}),
        ACTION_DAQ: MDAEvent(action=CustomAction(name=ACTION_DAQ, data={})),
    }
    engine._restart_hardware_tile = MagicMock()
    return engine


def _acquisition_events(demo_core) -> tuple[
    MDAEvent,
    MDAEvent,
    SequencedEvent,
    MDAEvent,
    MDAEvent,
]:
    """Return a failed tile followed by the dangerous downstream commands.

    Returns
    -------
    tuple
        Initial stage/DAQ setup, hardware tile, next move, and autofocus event.
    """
    current_x, current_y = demo_core.getXYPosition()
    current_z = demo_core.getZPosition()
    initial_stage = create_stage_event(
        {"x": current_x, "y": current_y, "z": current_z}
    )
    daq = create_daq_event(
        mode="stage",
        channel_states=(True, False, False, False, False),
        channel_powers=(10.0, 0.0, 0.0, 0.0, 0.0),
        channel_exposures_ms=(10.0, 10.0, 10.0, 10.0, 10.0),
        camera_center=(256, 256),
        camera_crop=(32, 16),
    )
    image_events = tuple(
        MDAEvent(index={"p": 0, "z": plane, "c": 0}) for plane in range(3)
    )
    tile = SequencedEvent(events=image_events)
    next_stage = create_stage_event(
        {"x": current_x + 1, "y": current_y + 1, "z": current_z}
    )
    autofocus = create_o2o3_autofocus_event(
        exposure_ms=10,
        camera_center=(100, 200),
        camera_crop=(40, 60),
        camera_id="OrcaFusionBT",
    )
    return initial_stage, daq, tile, next_stage, autofocus


def _short_camera_sequence(tile: SequencedEvent):
    """Return one frame and the missing-frame sentinels emitted upstream.

    Yields
    ------
    tuple or None
        One camera payload followed by two missing-frame sentinels.
    """
    yield ("partial-frame", tile.events[0], {})
    yield None
    yield None


def _timed_out_camera_sequence(tile: SequencedEvent):
    """Return one frame and then reproduce a camera timeout.

    Yields
    ------
    tuple
        The camera payload received before the timeout.

    Raises
    ------
    TimeoutError
        Always, after yielding the partial tile.
    """
    yield ("partial-frame", tile.events[0], {})
    raise TimeoutError("camera sequence stopped before tile completion")


@pytest.mark.parametrize(
    ("failure_kind", "expected_error", "error_match"),
    [
        (
            "missing-frame sentinels",
            IncompleteHardwareSequenceError,
            "expected 3 frames, received 1",
        ),
        (
            "camera timeout",
            TimeoutError,
            "camera sequence stopped before tile completion",
        ),
    ],
)
def test_repeated_stage_scan_failure_aborts_before_move_or_autofocus(
    demo_core,
    opm_config_factory,
    simulated_acquisition_hardware,
    workspace_tmp_path,
    failure_kind,
    expected_error,
    error_match,
) -> None:
    """Reproduce both field failure forms and forbid later hardware commands."""
    engine = _runner_engine(
        demo_core,
        opm_config_factory,
        simulated_acquisition_hardware,
        workspace_tmp_path,
    )
    initial_xy = demo_core.getXYPosition()
    initial_z = demo_core.getZPosition()
    initial_stage, daq, tile, next_stage, autofocus = _acquisition_events(demo_core)
    runner = MDARunner()
    runner.set_engine(engine)
    started: list[MDAEvent] = []
    runner.events.eventStarted.connect(started.append)
    failure = (
        _short_camera_sequence
        if failure_kind == "missing-frame sentinels"
        else _timed_out_camera_sequence
    )

    with (
        patch.object(engine, "setup_event", wraps=engine.setup_event) as setup_event,
        patch.object(
            engine, "teardown_event", wraps=engine.teardown_event
        ) as teardown_event,
        patch.object(
            engine, "teardown_sequence", wraps=engine.teardown_sequence
        ) as teardown_sequence,
        patch.object(
            MDAEngine,
            "exec_event",
            side_effect=[
                failure(tile),
                failure(tile),
            ],
        ),
        pytest.raises(expected_error, match=error_match),
    ):
        runner.run(iter((initial_stage, daq, tile, next_stage, autofocus)))

    assert runner.status.finish_reason is FinishReason.ERRORED
    assert started == [initial_stage, daq, tile]
    assert engine.simulated_custom_actions == [ACTION_STAGE_MOVE, ACTION_DAQ]
    assert demo_core.getXYPosition() == pytest.approx(initial_xy)
    assert demo_core.getZPosition() == pytest.approx(initial_z)
    assert setup_event.call_args_list == [
        call(initial_stage),
        call(daq),
        call(tile),
    ]
    engine._restart_hardware_tile.assert_called_once_with(tile, 1)
    assert teardown_event.call_args_list == [
        call(initial_stage),
        call(daq),
        call(tile),
    ]
    teardown_sequence.assert_called_once()


def test_successful_tile_retry_allows_runner_to_reach_move_and_autofocus(
    demo_core,
    opm_config_factory,
    simulated_acquisition_hardware,
    workspace_tmp_path,
) -> None:
    """Prove the abort assertion is tied to incompleteness, not test wiring."""
    engine = _runner_engine(
        demo_core,
        opm_config_factory,
        simulated_acquisition_hardware,
        workspace_tmp_path,
    )
    initial_stage, daq, tile, next_stage, autofocus = _acquisition_events(demo_core)
    runner = MDARunner()
    runner.set_engine(engine)
    started: list[MDAEvent] = []
    executed_actions: list[str] = []
    runner.events.eventStarted.connect(started.append)
    opm_exec_event = OPMEngineV2.exec_event.__get__(engine)

    def _exec_event(event: MDAEvent):
        if isinstance(event, SequencedEvent):
            return opm_exec_event(event)
        if isinstance(event.action, CustomAction):
            executed_actions.append(event.action.name)
        return ()

    engine.exec_event = _exec_event

    def _complete_camera_sequence():
        for image_event in tile.events:
            yield (f"frame-{image_event.index['z']}", image_event, {})

    with patch.object(
        MDAEngine,
        "exec_event",
        side_effect=[
            _short_camera_sequence(tile),
            _complete_camera_sequence(),
        ],
    ):
        runner.run(iter((initial_stage, daq, tile, next_stage, autofocus)))

    assert runner.status.finish_reason is FinishReason.COMPLETED
    assert started == [initial_stage, daq, tile, next_stage, autofocus]
    assert executed_actions == [
        ACTION_STAGE_MOVE,
        ACTION_DAQ,
        ACTION_STAGE_MOVE,
        ACTION_O2O3_AUTOFOCUS,
    ]
    engine._restart_hardware_tile.assert_called_once_with(tile, 1)
