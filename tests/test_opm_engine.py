"""Unit-test OPM engine dispatch independently of devices and the GUI."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pymmcore_plus.mda import MDAEngine
from useq import MDAEvent

from opm_v2.engine.opm_custom_events import (
    ACTION_FLUIDICS,
    create_asi_scan_setup_event,
    create_fluidics_event,
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
    engine.start_asi_scan_after_camera_sequence = False
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
