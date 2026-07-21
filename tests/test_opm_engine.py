"""Unit-test OPM engine dispatch independently of devices and the GUI."""

from __future__ import annotations

import weakref
from threading import Event as ThreadEvent
from unittest.mock import MagicMock, call, patch

import pytest
from pymmcore_plus.mda import MDAEngine, SkipEvent
from useq import MDAEvent, MDASequence

from opm_v2.engine.opm_custom_events import (
    ACTION_FLUIDICS,
    STAGE_MOVE_SPEED_METADATA_KEY,
    create_asi_scan_setup_event,
    create_fluidics_event,
    create_stage_event,
)
from opm_v2.engine.opm_engine import OPMEngineV2


def _isolated_engine() -> OPMEngineV2:
    """Create only the state needed by simulation-only dispatch methods.

    Returns
    -------
    OPMEngineV2
        Uninitialized engine instance with isolated simulated state.
    """
    engine = object.__new__(OPMEngineV2)
    engine.simulate_hardware = True
    engine.simulated_asi_state = {}
    engine.simulated_asi_transitions = []
    engine.simulated_custom_actions = []
    engine.simulated_stage_move_speeds = []
    engine.start_asi_scan_after_camera_sequence = False
    engine._stage_move_count = 0
    engine._safe_stop_requested = ThreadEvent()
    return engine


def test_standard_setup_event_delegates_to_upstream_engine() -> None:
    """Verify ordinary useq image events retain pymmcore-plus setup behavior."""
    engine = _isolated_engine()
    event = MDAEvent(index={"t": 0})

    with patch.object(MDAEngine, "setup_event") as upstream_setup:
        engine.setup_event(event)

    upstream_setup.assert_called_once_with(event)


def test_standard_exec_event_delegates_to_upstream_engine() -> None:
    """Verify ordinary useq image acquisition remains owned by pymmcore-plus."""
    engine = _isolated_engine()
    event = MDAEvent(index={"t": 0})
    upstream_result = (("frame", event, {"camera": "demo"}),)

    with patch.object(MDAEngine, "exec_event", return_value=upstream_result) as execute:
        result = engine.exec_event(event)

    execute.assert_called_once_with(event)
    assert result is upstream_result


def test_safe_stop_waits_for_and_skips_next_software_command() -> None:
    """Leave camera events alone, then cancel before the next custom command."""
    engine = _isolated_engine()
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    image_event = MDAEvent(index={"p": 0})

    assert engine.request_safe_stop()
    assert not engine.request_safe_stop()
    with patch.object(MDAEngine, "setup_event") as upstream_setup:
        engine.setup_event(image_event)

    upstream_setup.assert_called_once_with(image_event)
    mmcore.mda.cancel.assert_not_called()
    assert engine._safe_stop_requested.is_set()

    stage_event = create_stage_event({"x": 10.0, "y": 20.0, "z": 30.0})
    with pytest.raises(SkipEvent, match="OPM STOP before software command") as exc:
        engine.setup_event(stage_event)

    assert exc.value.num_frames == 0
    mmcore.mda.cancel.assert_called_once_with()
    mmcore.setXYPosition.assert_not_called()
    assert not engine._safe_stop_requested.is_set()


def test_asi_setup_waits_for_post_camera_sequence_hook() -> None:
    """Verify ASI coordinates hardware only after camera sequencing has started."""
    engine = _isolated_engine()
    event = create_asi_scan_setup_event(start_mm=1.0, end_mm=2.0, speed_mm_s=0.5)

    engine.setup_event(event)

    assert engine.simulated_asi_state == {
        "scan_axis_start_mm": 1.0,
        "scan_axis_end_mm": 2.0,
        "scan_axis_speed_mm_s": 0.5,
        "scan_state": "Idle",
    }
    assert engine.simulated_asi_transitions == ["Idle"]
    assert engine.start_asi_scan_after_camera_sequence is True

    engine.post_sequence_started(MDAEvent())

    assert engine.simulated_asi_state["scan_state"] == "Running"
    assert engine.simulated_asi_transitions == ["Idle", "Running"]
    assert engine.start_asi_scan_after_camera_sequence is False


def test_asi_hardware_setup_preserves_millimetre_position_precision() -> None:
    """Send sub-0.01 mm stage coordinates to the ASI adapter unchanged."""
    engine = object.__new__(OPMEngineV2)
    engine.simulate_hardware = False
    engine._safe_stop_requested = ThreadEvent()
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.getProperty.side_effect = lambda _device, prop: {
        "MotorSpeedX-S(mm/s)": "0.0067",
        "ScanFastAxisStartPosition(mm)": "3.167460",
        "ScanFastAxisStopPosition(mm)": "3.168460",
        "ScanSettlingTime(ms)": "3000",
    }[prop]
    engine._config = {
        "PLC": {
            "name": "PLogic",
            "position": "PointerPosition",
            "cellconfig": "EditCellConfig",
            "pin": 33,
            "signalid": 46,
        },
        "OPM": {"stage_move_speed": 0.05},
    }
    engine.configure_stage_camera_trigger = MagicMock()
    engine.start_asi_scan_after_camera_sequence = False
    event = create_asi_scan_setup_event(
        start_mm=3.167460,
        end_mm=3.168460,
        speed_mm_s=0.0067,
    )

    engine.setup_event(event)

    assert call(
        "XYStage", "ScanFastAxisStartPosition(mm)", pytest.approx(3.167460)
    ) in mmcore.setProperty.call_args_list
    assert call(
        "XYStage", "ScanFastAxisStopPosition(mm)", pytest.approx(3.168460)
    ) in mmcore.setProperty.call_args_list
    engine.configure_stage_camera_trigger.assert_called_once_with()


def test_preview_stage_move_speed_override_is_restored() -> None:
    """Apply Explorer sequence speeds and restore captured normal values."""
    engine = _isolated_engine()
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    engine._config = {
        "OPM": {"stage_move_speed": 0.05, "circular_buffer_mb": 64}
    }
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.hasProperty.return_value = True
    mmcore.getProperty.side_effect = lambda _device, prop: {
        "MotorSpeedX-S(mm/s)": "0.05",
        "MotorSpeedY-S(mm/s)": "0.08",
    }[prop]
    sequence = MDASequence(
        metadata={
            STAGE_MOVE_SPEED_METADATA_KEY: {
                "move_speed_x_mm_s": 0.2,
                "move_speed_y_mm_s": 0.32,
            }
        }
    )

    with patch.object(MDAEngine, "setup_sequence") as upstream_setup:
        engine.setup_sequence(sequence)

    upstream_setup.assert_called_once_with(sequence)
    assert call("XYStage", "MotorSpeedX-S(mm/s)", 0.2) in (
        mmcore.setProperty.call_args_list
    )
    assert call("XYStage", "MotorSpeedY-S(mm/s)", 0.32) in (
        mmcore.setProperty.call_args_list
    )
    assert engine.simulated_stage_move_speeds == [{"x": 0.2, "y": 0.32}]
    assert engine._stage_speeds_before_sequence == {
        "MotorSpeedX-S(mm/s)": "0.05",
        "MotorSpeedY-S(mm/s)": "0.08",
    }

    engine._restore_stage_after_scan()

    assert call("XYStage", "MotorSpeedX-S(mm/s)", "0.05") in (
        mmcore.setProperty.call_args_list
    )
    assert call("XYStage", "MotorSpeedY-S(mm/s)", "0.08") in (
        mmcore.setProperty.call_args_list
    )
    assert engine._stage_speeds_before_sequence == {}


def test_explorer_teardown_returns_xy_before_restoring_normal_speed() -> None:
    """Allow the upstream XY return at 4x speed before restoring speed/timeout."""
    engine = _isolated_engine()
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    engine._config = {
        "NIDAQ": {"image_mirror_neutral_v": 0.0},
        "Camera": {"camera_id": "Camera"},
    }
    engine.opmDAQ = MagicMock()
    engine.AOMirror = MagicMock()
    engine.AOMirror.output_path = None
    engine._post_teardown = None
    mmcore.hasProperty.return_value = False
    mmcore.getTimeoutMs.return_value = 5000
    sequence = MDASequence(metadata={STAGE_MOVE_SPEED_METADATA_KEY: {}})
    teardown_order: list[str] = []

    with (
        patch.object(
            MDAEngine,
            "teardown_sequence",
            side_effect=lambda _sequence: teardown_order.append("return_xy"),
        ),
        patch.object(
            engine,
            "_restore_stage_after_scan",
            side_effect=lambda: teardown_order.append("restore_speed"),
        ),
    ):
        engine.teardown_sequence(sequence)

    assert teardown_order == ["return_xy", "restore_speed"]
    assert mmcore.setTimeoutMs.call_args_list == [call(120_000), call(5000)]


def test_initial_stage_move_temporarily_uses_extended_timeout() -> None:
    """Give the initial acquisition move time to reach a distant start point."""
    engine = _isolated_engine()
    engine.simulate_hardware = False
    engine._config = {"OPM": {"stage_move_speed": 0.05}}
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getTimeoutMs.return_value = 5000
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.getFocusDevice.return_value = "ZStage"
    mmcore.getXYPosition.side_effect = [(0.0, 0.0), (1000.0, 0.0)]
    mmcore.hasProperty.return_value = False
    event = create_stage_event({"x": 1000.0, "y": 0.0, "z": 25.0})

    engine.setup_event(event)

    assert mmcore.setTimeoutMs.call_args_list == [call(120_000), call(5000)]
    assert call("XYStage") in mmcore.waitForDevice.call_args_list
    assert engine._stage_move_count == 1


def test_adjacent_stage_move_retains_normal_timeout() -> None:
    """Do not extend the timeout for a move that fits the normal tile budget."""
    engine = _isolated_engine()
    engine.simulate_hardware = False
    engine._stage_move_count = 1
    engine._config = {"OPM": {"stage_move_speed": 0.05}}
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getTimeoutMs.return_value = 5000
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.getFocusDevice.return_value = "ZStage"
    mmcore.getXYPosition.side_effect = [(0.0, 0.0), (100.0, 0.0)]
    mmcore.hasProperty.return_value = False
    event = create_stage_event({"x": 100.0, "y": 0.0, "z": 25.0})

    engine.setup_event(event)

    mmcore.setTimeoutMs.assert_not_called()
    assert engine._stage_move_count == 2


def test_non_adjacent_stage_move_temporarily_uses_extended_timeout() -> None:
    """Extend a later move whose travel time exceeds the normal tile budget."""
    engine = _isolated_engine()
    engine.simulate_hardware = False
    engine._stage_move_count = 3
    engine._config = {"OPM": {"stage_move_speed": 0.05}}
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getTimeoutMs.return_value = 5000
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.getFocusDevice.return_value = "ZStage"
    mmcore.getXYPosition.side_effect = [(0.0, 0.0), (1000.0, 0.0)]
    mmcore.hasProperty.return_value = False
    event = create_stage_event({"x": 1000.0, "y": 0.0, "z": 25.0})

    engine.setup_event(event)

    assert mmcore.setTimeoutMs.call_args_list == [call(120_000), call(5000)]
    assert engine._stage_move_count == 4


def test_saved_acquisition_teardown_uses_extended_return_timeout() -> None:
    """Extend the timeout for the upstream return after a saved acquisition."""
    engine = _isolated_engine()
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    engine._config = {
        "NIDAQ": {"image_mirror_neutral_v": 0.0},
        "Camera": {"camera_id": "Camera"},
        "Lasers": {"laser_names": []},
    }
    engine.opmDAQ = MagicMock()
    engine.AOMirror = MagicMock()
    engine.AOMirror.output_path = None
    engine._post_teardown = None
    mmcore.hasProperty.return_value = False
    mmcore.getTimeoutMs.return_value = 5000

    with (
        patch.object(MDAEngine, "teardown_sequence") as upstream_teardown,
        patch.object(engine, "_restore_stage_after_scan"),
    ):
        engine.teardown_sequence(MDASequence())

    upstream_teardown.assert_called_once()
    assert mmcore.setTimeoutMs.call_args_list == [call(120_000), call(5000)]


def test_simulated_non_imaging_action_returns_no_camera_frames() -> None:
    """Verify simulated fluidics actions are recorded without fabricating images."""
    engine = _isolated_engine()
    event = create_fluidics_event(total_rounds=2, current_round=1)

    result = engine.exec_event(event)

    assert result == ()
    assert engine.simulated_custom_actions == [ACTION_FLUIDICS]


def test_daq_exposure_validation_uses_only_enabled_channels() -> None:
    """Validate DAQ exposure arrays before any camera or laser operation."""
    payload = {
        "DAQ": {"channel_states": [True, False, True]},
        "Camera": {"exposure_channels": [5.0, 0.0, 7.0]},
    }
    assert OPMEngineV2._active_daq_exposures(payload) == [5.0, 7.0]

    payload["DAQ"]["channel_states"] = [False, False, False]
    with pytest.raises(ValueError, match="no active acquisition channels"):
        OPMEngineV2._active_daq_exposures(payload)

    payload["DAQ"]["channel_states"] = [True, False, False]
    payload["Camera"]["exposure_channels"] = [0.0, 4.0, 4.0]
    with pytest.raises(ValueError, match="greater than 0 ms"):
        OPMEngineV2._active_daq_exposures(payload)

    payload["Camera"]["exposure_channels"] = [4.0]
    with pytest.raises(ValueError, match="equal lengths"):
        OPMEngineV2._active_daq_exposures(payload)
