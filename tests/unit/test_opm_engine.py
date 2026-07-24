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
    engine._stage_speeds_before_sequence = {}
    engine._is_stage_explorer_preview = False
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


def test_stage_move_uses_standard_mda_position_fields() -> None:
    """Expose XYZ to pymmcore-plus while retaining the OPM action marker."""
    event = create_stage_event({"x": 10.123, "y": 20.456, "z": 30.789})

    assert event.x_pos == pytest.approx(10.12)
    assert event.y_pos == pytest.approx(20.46)
    assert event.z_pos == pytest.approx(30.79)
    assert event.action.data["Stage"] == {
        "x_pos": 10.12,
        "y_pos": 20.46,
        "z_pos": 30.79,
    }


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
    property_values = {
        "MotorSpeedX-S(mm/s)": "0.05",
        "MotorSpeedY-S(mm/s)": "0.08",
    }
    mmcore.getProperty.side_effect = lambda _device, prop: property_values[prop]

    def _set_property(_device: str, prop: str, value: object) -> None:
        property_values[prop] = str(value)

    mmcore.setProperty.side_effect = _set_property
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
    mmcore.setTimeoutMs.assert_not_called()
    mmcore.setDeviceTimeoutMs.assert_not_called()
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

    engine._restore_stage_speeds()

    assert call("XYStage", "MotorSpeedX-S(mm/s)", "0.05") in (
        mmcore.setProperty.call_args_list
    )
    assert call("XYStage", "MotorSpeedY-S(mm/s)", "0.08") in (
        mmcore.setProperty.call_args_list
    )
    assert engine._stage_speeds_before_sequence == {}
    mmcore.setTimeoutMs.assert_not_called()


def test_saved_acquisition_restores_pre_sequence_manual_speeds() -> None:
    """Do not leave Stage Explorer at the slow acquisition move speed."""
    engine = _isolated_engine()
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    engine._config = {
        "OPM": {"stage_move_speed": 0.05, "circular_buffer_mb": 64}
    }
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.hasProperty.return_value = True
    property_values = {
        "MotorSpeedX-S(mm/s)": "1.5",
        "MotorSpeedY-S(mm/s)": "1.2",
    }
    mmcore.getProperty.side_effect = lambda _device, prop: property_values[prop]

    def _set_property(_device: str, prop: str, value: object) -> None:
        property_values[prop] = str(value)

    mmcore.setProperty.side_effect = _set_property
    sequence = MDASequence()

    with patch.object(MDAEngine, "setup_sequence"):
        engine.setup_sequence(sequence)

    engine._apply_stage_move_speeds(
        {
            "move_speed_x_mm_s": 0.0067,
            "move_speed_y_mm_s": 0.05,
        }
    )
    engine._restore_stage_speeds()

    assert property_values == {
        "MotorSpeedX-S(mm/s)": "1.5",
        "MotorSpeedY-S(mm/s)": "1.2",
    }
    assert engine._stage_speeds_before_sequence == {}


def test_explorer_teardown_returns_xy_before_restoring_normal_speed() -> None:
    """Allow the upstream XY return at 4x speed before restoring speed/timeout."""
    engine = _isolated_engine()
    engine.simulate_hardware = False
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
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.hasDeviceTimeout.return_value = False
    mmcore.getDeviceTimeoutMs.return_value = 5000
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
            "_restore_stage_speeds",
            side_effect=lambda: teardown_order.append("restore_speed"),
        ),
        patch.object(engine, "_prepare_xy_for_point_move"),
    ):
        engine.teardown_sequence(sequence)

    assert teardown_order == ["return_xy", "restore_speed"]
    mmcore.setTimeoutMs.assert_not_called()
    mmcore.setDeviceTimeoutMs.assert_called_once_with("XYStage", 5000)
    mmcore.unsetDeviceTimeout.assert_called_once_with("XYStage")


def test_explorer_short_return_keeps_accelerated_preview_speed() -> None:
    """Keep the 4x preview speed until even a short return move completes."""
    engine = _isolated_engine()
    engine._is_stage_explorer_preview = True
    engine._stage_move_count = 1
    engine._initial_state = {"xy_position": (10.0, 20.0)}
    engine._stage_speeds_before_sequence = {
        "MotorSpeedX-S(mm/s)": "0.08",
        "MotorSpeedY-S(mm/s)": "0.08",
    }
    engine._config = {"OPM": {"stage_move_speed": 0.05}}
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.getXYPosition.return_value = (0.0, 0.0)
    mmcore.getDeviceTimeoutMs.return_value = 5000

    with patch.object(
        engine,
        "_apply_stage_move_speeds",
        return_value={"x": 0.32, "y": 0.32},
    ) as apply_speeds:
        engine._teardown_return_timeout_ms()

    apply_speeds.assert_called_once_with(
        {
            "move_speed_x_mm_s": 0.32,
            "move_speed_y_mm_s": 0.32,
        }
    )


def test_stage_speed_restore_uses_configured_fallback_without_readback() -> None:
    """Use the configured move speed only when no pre-sequence speed was read."""
    engine = _isolated_engine()
    engine._is_stage_explorer_preview = False
    engine._stage_speeds_before_sequence = {}
    engine._config = {"OPM": {"stage_move_speed": 0.05}}
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.hasProperty.return_value = True

    engine._restore_stage_speeds()

    assert call("XYStage", "MotorSpeedX-S(mm/s)", 0.05) in (
        mmcore.setProperty.call_args_list
    )
    assert call("XYStage", "MotorSpeedY-S(mm/s)", 0.05) in (
        mmcore.setProperty.call_args_list
    )


def test_initial_stage_move_temporarily_uses_extended_timeout() -> None:
    """Use 4x speed and a long timeout to reach a distant start point."""
    engine = _isolated_engine()
    engine.simulate_hardware = False
    engine._config = {"OPM": {"stage_move_speed": 0.05}}
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.getFocusDevice.return_value = "ZStage"
    mmcore.getXYPosition.side_effect = [(0.0, 0.0), (1000.0, 0.0)]
    mmcore.getDeviceTimeoutMs.return_value = 5000
    mmcore.hasDeviceTimeout.return_value = False
    event = create_stage_event({"x": 1000.0, "y": 0.0, "z": 25.0})

    with (
        patch.object(engine, "_prepare_xy_for_point_move"),
        patch.object(MDAEngine, "setup_event") as upstream_setup,
        patch.object(
            engine,
            "_apply_stage_move_speeds",
            return_value={"x": 0.2, "y": 0.2},
        ) as apply_speeds,
    ):
        engine.setup_event(event)

    mmcore.setTimeoutMs.assert_not_called()
    mmcore.setDeviceTimeoutMs.assert_called_once_with("XYStage", 120_000)
    mmcore.unsetDeviceTimeout.assert_called_once_with("XYStage")
    apply_speeds.assert_called_once_with({
        "move_speed_x_mm_s": 0.2,
        "move_speed_y_mm_s": 0.2,
    })
    upstream_setup.assert_called_once_with(event)
    mmcore.setXYPosition.assert_not_called()
    assert engine._stage_move_count == 1


def test_stage_move_timeout_covers_estimated_slow_move_duration() -> None:
    """Do not cap a very long distance-derived move at two minutes."""
    engine = _isolated_engine()
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.getDeviceTimeoutMs.return_value = 5000

    timeout_ms, estimated_s = engine._stage_move_timeout_ms(
        current_x_um=0.0,
        current_y_um=0.0,
        target_x_um=10_000.0,
        target_y_um=0.0,
        speed_x_mm_s=0.05,
        speed_y_mm_s=0.05,
    )

    assert estimated_s == pytest.approx(200.0)
    assert timeout_ms == 202_000


def test_scan_state_cleanup_does_not_conflate_physical_busy() -> None:
    """Stop the ASI scan state machine without polling point-move Busy."""
    engine = _isolated_engine()
    engine.simulate_hardware = False
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.hasProperty.return_value = True
    mmcore.getProperty.side_effect = ["Running", "Idle"]

    engine._stop_active_stage_scan()

    mmcore.setProperty.assert_called_once_with("XYStage", "ScanState", "Idle")
    mmcore.deviceBusy.assert_not_called()
    mmcore.waitForDevice.assert_not_called()


def test_xy_timeout_restores_an_existing_device_override() -> None:
    """Preserve a pre-existing XY-specific timeout instead of changing Core."""
    engine = _isolated_engine()
    engine.simulate_hardware = False
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.hasDeviceTimeout.return_value = True
    mmcore.getDeviceTimeoutMs.return_value = 15_000

    with engine._temporary_xy_stage_timeout(300_000):
        pass

    assert mmcore.setDeviceTimeoutMs.call_args_list == [
        call("XYStage", 300_000),
        call("XYStage", 15_000),
    ]
    mmcore.unsetDeviceTimeout.assert_not_called()
    mmcore.setTimeoutMs.assert_not_called()


def test_failed_initial_move_halts_stage_before_teardown() -> None:
    """Stop and confirm idle before propagating a failed startup move."""
    engine = _isolated_engine()
    engine.simulate_hardware = False
    engine._config = {"OPM": {"stage_move_speed": 0.05}}
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.getFocusDevice.return_value = "ZStage"
    mmcore.getXYPosition.return_value = (0.0, 0.0)
    mmcore.getDeviceTimeoutMs.return_value = 5000
    mmcore.hasDeviceTimeout.return_value = False
    mmcore.waitForDevice.side_effect = [
        None,
        RuntimeError("stage timeout"),
        None,
    ]
    event = create_stage_event({"x": 1000.0, "y": 0.0, "z": 25.0})

    with (
        patch.object(engine, "_prepare_xy_for_point_move"),
        patch.object(
            MDAEngine,
            "setup_event",
            side_effect=RuntimeError("stage timeout"),
        ),
        patch.object(
            engine,
            "_apply_stage_move_speeds",
            return_value={"x": 0.2, "y": 0.2},
        ),
        pytest.raises(RuntimeError, match="stage timeout"),
    ):
        engine.setup_event(event)

    mmcore.stop.assert_called_once_with("XYStage")
    assert mmcore.setDeviceTimeoutMs.call_args_list == [
        call("XYStage", 120_000),
        call("XYStage", 120_000),
    ]
    assert mmcore.unsetDeviceTimeout.call_args_list == [
        call("XYStage"),
        call("XYStage"),
    ]
    mmcore.setTimeoutMs.assert_not_called()


def test_adjacent_stage_move_retains_normal_timeout() -> None:
    """Keep normal speed and timeout for an adjacent tile move."""
    engine = _isolated_engine()
    engine.simulate_hardware = False
    engine._stage_move_count = 1
    engine._config = {"OPM": {"stage_move_speed": 0.05}}
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.getFocusDevice.return_value = "ZStage"
    mmcore.getXYPosition.side_effect = [(0.0, 0.0), (100.0, 0.0)]
    mmcore.getDeviceTimeoutMs.return_value = 5000
    mmcore.hasDeviceTimeout.return_value = False
    event = create_stage_event({"x": 100.0, "y": 0.0, "z": 25.0})

    with (
        patch.object(engine, "_prepare_xy_for_point_move"),
        patch.object(MDAEngine, "setup_event") as upstream_setup,
        patch.object(
            engine,
            "_apply_stage_move_speeds",
            return_value={"x": 0.05, "y": 0.05},
        ) as apply_speeds,
    ):
        engine.setup_event(event)

    mmcore.setTimeoutMs.assert_not_called()
    mmcore.setDeviceTimeoutMs.assert_called_once_with("XYStage", 5000)
    mmcore.unsetDeviceTimeout.assert_called_once_with("XYStage")
    apply_speeds.assert_called_once_with({
        "move_speed_x_mm_s": 0.05,
        "move_speed_y_mm_s": 0.05,
    })
    upstream_setup.assert_called_once_with(event)
    mmcore.setXYPosition.assert_not_called()
    assert engine._stage_move_count == 2


def test_non_adjacent_stage_move_temporarily_uses_extended_timeout() -> None:
    """Use 4x speed and long timeout for a later non-adjacent move."""
    engine = _isolated_engine()
    engine.simulate_hardware = False
    engine._stage_move_count = 3
    engine._config = {"OPM": {"stage_move_speed": 0.05}}
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.getFocusDevice.return_value = "ZStage"
    mmcore.getXYPosition.side_effect = [(0.0, 0.0), (1000.0, 0.0)]
    mmcore.getDeviceTimeoutMs.return_value = 5000
    mmcore.hasDeviceTimeout.return_value = False
    event = create_stage_event({"x": 1000.0, "y": 0.0, "z": 25.0})

    with (
        patch.object(engine, "_prepare_xy_for_point_move"),
        patch.object(MDAEngine, "setup_event") as upstream_setup,
        patch.object(
            engine,
            "_apply_stage_move_speeds",
            return_value={"x": 0.2, "y": 0.2},
        ) as apply_speeds,
    ):
        engine.setup_event(event)

    mmcore.setTimeoutMs.assert_not_called()
    mmcore.setDeviceTimeoutMs.assert_called_once_with("XYStage", 120_000)
    mmcore.unsetDeviceTimeout.assert_called_once_with("XYStage")
    apply_speeds.assert_called_once_with({
        "move_speed_x_mm_s": 0.2,
        "move_speed_y_mm_s": 0.2,
    })
    upstream_setup.assert_called_once_with(event)
    mmcore.setXYPosition.assert_not_called()
    assert engine._stage_move_count == 4


def test_saved_teardown_uses_4x_speed_for_large_return_then_restores() -> None:
    """Accelerate the return-to-start command and restore normal XY speed."""
    engine = _isolated_engine()
    engine.simulate_hardware = False
    engine._initial_state = {"xy_position": (1000.0, 0.0)}
    engine._config = {
        "OPM": {"stage_move_speed": 0.05},
        "NIDAQ": {"image_mirror_neutral_v": 0.0},
        "Camera": {"camera_id": "Camera"},
        "Lasers": {"laser_names": []},
    }
    engine.opmDAQ = MagicMock()
    engine.AOMirror = MagicMock()
    engine.AOMirror.output_path = None
    engine._post_teardown = None
    mmcore = MagicMock()
    engine._mmcore_ref = weakref.ref(mmcore)
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.getXYPosition.return_value = (0.0, 0.0)
    mmcore.getDeviceTimeoutMs.return_value = 5000
    mmcore.hasDeviceTimeout.return_value = False
    mmcore.getProperty.return_value = "0.2"
    speed_properties = {
        "MotorSpeedX-S(mm/s)",
        "MotorSpeedY-S(mm/s)",
    }
    mmcore.hasProperty.side_effect = (
        lambda _device, prop: prop in speed_properties
    )
    order: list[str] = []

    def _set_property(_device: str, prop: str, value: object) -> None:
        order.append(f"{prop}={value}")

    mmcore.setProperty.side_effect = _set_property
    with (
        patch.object(
            MDAEngine,
            "teardown_sequence",
            side_effect=lambda _sequence: order.append("return"),
        ),
        patch.object(engine, "_prepare_xy_for_point_move"),
    ):
        engine.teardown_sequence(MDASequence())

    fast_x = "MotorSpeedX-S(mm/s)=0.2"
    fast_y = "MotorSpeedY-S(mm/s)=0.2"
    normal_x = "MotorSpeedX-S(mm/s)=0.05"
    normal_y = "MotorSpeedY-S(mm/s)=0.05"
    return_idx = order.index("return")
    assert fast_x in order[:return_idx]
    assert fast_y in order[:return_idx]
    assert normal_x in order[return_idx + 1 :]
    assert normal_y in order[return_idx + 1 :]
    mmcore.setTimeoutMs.assert_not_called()
    mmcore.setDeviceTimeoutMs.assert_called_once_with("XYStage", 120_000)
    mmcore.unsetDeviceTimeout.assert_called_once_with("XYStage")


def test_saved_acquisition_teardown_uses_extended_return_timeout() -> None:
    """Extend the timeout for the upstream return after a saved acquisition."""
    engine = _isolated_engine()
    engine.simulate_hardware = False
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
    mmcore.getXYStageDevice.return_value = "XYStage"
    mmcore.getDeviceTimeoutMs.return_value = 5000
    mmcore.hasDeviceTimeout.return_value = False

    with (
        patch.object(MDAEngine, "teardown_sequence") as upstream_teardown,
        patch.object(engine, "_prepare_xy_for_point_move"),
        patch.object(engine, "_restore_stage_speeds"),
    ):
        engine.teardown_sequence(MDASequence())

    upstream_teardown.assert_called_once()
    mmcore.setTimeoutMs.assert_not_called()
    mmcore.setDeviceTimeoutMs.assert_called_once_with("XYStage", 5000)
    mmcore.unsetDeviceTimeout.assert_called_once_with("XYStage")


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
