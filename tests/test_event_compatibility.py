"""Verify event streams against the executable pre-refactor implementation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from opm_v2.engine import setup_events
from opm_v2.engine.opm_custom_events import ACTION_AO_GRID, ACTION_O2O3_AUTOFOCUS

CONFIGURATION_DIR = Path(__file__).parent / "configurations"
GUI_SCENARIOS = json.loads((CONFIGURATION_DIR / "gui_integration.json").read_text())
COMPATIBILITY = json.loads((CONFIGURATION_DIR / "event_compatibility.json").read_text())


def _timed_scenario(mode: str) -> dict[str, Any]:
    """Return the time-plan scenario executed against both implementations.

    Parameters
    ----------
    mode : str
        OPM acquisition mode.

    Returns
    -------
    dict[str, Any]
        Scenario with a nonzero two-timepoint useq time plan.
    """
    return {
        "mode": mode,
        "active_channels": [0, 2],
        "channel_powers": [11.0, 33.0],
        "channel_exposures_ms": [6.0, 6.0],
        "camera_shape": [64, 32],
        "scan_range_um": 8.0 if mode == "projection" else 4.0,
        "scan_axis_step_um": 2.0,
        "axis_order": {
            "projection": "tpc",
            "mirror": "tpzc",
            "stage": "tpgzc",
            "timelapse": "pztc",
        }[mode],
        "fluidics": "none",
        "o2o3_mode": "none",
        "ao_mode": "none",
        "ao_options": {},
        "laser_blanking": True,
        "time_loops": 2,
        "time_interval": 1.25,
    }


def _scenario(case: dict[str, Any]) -> dict[str, Any]:
    """Resolve a compatibility case into a complete acquisition scenario.

    Parameters
    ----------
    case : dict[str, Any]
        Compatibility case referencing a reusable GUI scenario or timed mode.

    Returns
    -------
    dict[str, Any]
        Independent complete acquisition scenario.
    """
    if timed_mode := case.get("timed_mode"):
        return _timed_scenario(timed_mode)
    scenario = deepcopy(GUI_SCENARIOS[case["source"]])
    scenario.update(deepcopy(case.get("overrides", {})))
    return scenario


def _json_safe(value: Any) -> Any:
    """Convert an event value into a deterministic JSON-compatible value.

    Parameters
    ----------
    value : Any
        Value from a serialized useq event.

    Returns
    -------
    Any
        Recursively normalized JSON-compatible value.
    """
    if isinstance(value, Path):
        return "<PATH>"
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _compatibility_stream(events, removed_bug_actions=()) -> list[dict[str, Any]]:
    """Normalize only the intentional differences from the legacy stream.

    Parameters
    ----------
    events : Sequence[MDAEvent]
        Current ordered event stream.
    removed_bug_actions : Sequence[str]
        Actions absent from the old stream because of a documented legacy bug.

    Returns
    -------
    list[dict[str, Any]]
        Complete ordered events with only standard-field migrations removed.
    """
    removed_actions = {"Timelapse", *removed_bug_actions}
    stream = []
    for event in events:
        serialized = _json_safe(event.model_dump())
        action = serialized.get("action") or {}
        if action.get("name") in removed_actions:
            continue
        serialized.update({
            "exposure": None,
            "x_pos": None,
            "y_pos": None,
            "z_pos": None,
            "min_start_time": None,
            "reset_event_timer": False,
        })
        stream.append(serialized)
    return stream


def _stream_digest(stream: list[dict[str, Any]]) -> str:
    """Return a deterministic digest of every ordered event field.

    Parameters
    ----------
    stream : list[dict[str, Any]]
        Normalized ordered event stream.

    Returns
    -------
    str
        SHA-256 digest of canonical compact JSON.
    """
    payload = json.dumps(stream, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("case_name", tuple(COMPATIBILITY["cases"]))
def test_event_stream_matches_pre_refactor_oracle(
    case_name,
    demo_core,
    workspace_tmp_path,
    simulated_acquisition_hardware,
    opm_config_from_scenario,
    sequence_for_scenario,
    monkeypatch,
) -> None:
    """Compare the complete ordered stream with commit bba09bd's output.

    Parameters
    ----------
    case_name : str
        Named compatibility scenario.
    demo_core : CMMCorePlus
        Core loaded with Micro-Manager demo hardware.
    workspace_tmp_path : Path
        Isolated output parent for AO event paths.
    simulated_acquisition_hardware : SimulatedAcquisitionHardware
        Singleton-backed simulated mirror and DAQ.
    opm_config_from_scenario : Callable
        Reusable scenario configuration builder.
    sequence_for_scenario : Callable
        Reusable standard useq plan builder.
    monkeypatch : pytest.MonkeyPatch
        Patch helper used to isolate event creation from storage.
    """
    case = COMPATIBILITY["cases"][case_name]
    scenario = _scenario(case)
    config = opm_config_from_scenario(scenario)
    builder = {
        "projection": setup_events.setup_projection,
        "mirror": setup_events.setup_mirrorscan,
        "stage": setup_events.setup_stagescan,
        "timelapse": setup_events.setup_timelapse,
    }[scenario["mode"]]
    monkeypatch.setattr(setup_events, "DEBUGGING", False)
    monkeypatch.setattr(
        setup_events, "create_zarr_handler", lambda *_args, **_kwargs: None
    )

    events, handler = builder(
        demo_core,
        config,
        sequence_for_scenario(scenario),
        workspace_tmp_path / "data.zarr",
    )
    stream = _compatibility_stream(events, case.get("removed_bug_actions", ()))

    assert handler is None
    assert len(stream) == case["event_count"]
    assert _stream_digest(stream) == case["sha256"]


@pytest.mark.parametrize(
    ("scenario_name", "action_name", "expected_count"),
    [
        ("mirror_cz_grid", ACTION_AO_GRID, 17),
        ("projection_timepoint_actions", ACTION_O2O3_AUTOFOCUS, 2),
    ],
)
def test_documented_legacy_event_bugs_are_fixed(
    scenario_name,
    action_name,
    expected_count,
    demo_core,
    workspace_tmp_path,
    simulated_acquisition_hardware,
    opm_config_from_scenario,
    sequence_for_scenario,
    split_events,
    monkeypatch,
) -> None:
    """Verify the only event-stream divergences correct legacy failures.

    Parameters
    ----------
    scenario_name : str
        GUI scenario that exposed the legacy bug.
    action_name : str
        Correctly generated action absent from the old stream.
    expected_count : int
        Expected action count in the corrected stream.
    demo_core : CMMCorePlus
        Core loaded with Micro-Manager demo hardware.
    workspace_tmp_path : Path
        Isolated output parent for AO event paths.
    simulated_acquisition_hardware : SimulatedAcquisitionHardware
        Singleton-backed simulated mirror and DAQ.
    opm_config_from_scenario : Callable
        Reusable scenario configuration builder.
    sequence_for_scenario : Callable
        Reusable standard useq plan builder.
    split_events : Callable
        Reusable camera/custom event partitioner.
    monkeypatch : pytest.MonkeyPatch
        Patch helper used to isolate event creation from storage.
    """
    scenario = deepcopy(GUI_SCENARIOS[scenario_name])
    config = opm_config_from_scenario(scenario)
    builder = {
        "projection": setup_events.setup_projection,
        "mirror": setup_events.setup_mirrorscan,
    }[scenario["mode"]]
    monkeypatch.setattr(setup_events, "DEBUGGING", False)
    monkeypatch.setattr(
        setup_events, "create_zarr_handler", lambda *_args, **_kwargs: None
    )

    events, _handler = builder(
        demo_core,
        config,
        sequence_for_scenario(scenario),
        workspace_tmp_path / "data.zarr",
    )
    _image_events, custom_actions = split_events(events)

    assert custom_actions.count(action_name) == expected_count
