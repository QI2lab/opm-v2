"""Unit-test sensorless-AO configuration validation and grid dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pymmcore_plus import CMMCorePlus

from opm_v2.engine.opm_custom_events import create_ao_grid_event
from opm_v2.utils import sensorless_ao


def test_ao_grid_requires_complete_optimization_configuration() -> None:
    """Reject incomplete AO-grid payloads before accessing hardware."""
    with pytest.raises(KeyError, match="metric_acceptance"):
        sensorless_ao.run_ao_grid_mapping(
            ao_dict={},
            stage_positions=[],
            verbose=False,
        )


def test_ao_grid_forwards_metric_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmp_path,
    opm_config_factory,
    demo_core: CMMCorePlus,
) -> None:
    """Forward the GUI metric-acceptance setting into each AO optimization.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing hardware and optimization calls.
    workspace_tmp_path : pathlib.Path
        Workspace-local output directory.
    opm_config_factory : OpmConfigFactory
        Reusable OPM demo-configuration factory.
    demo_core : CMMCorePlus
        Process-wide core instance loaded with Micro-Manager demo stages.
    """
    config = opm_config_factory(mode="projection")
    config["acq_config"]["AO"]["metric_acceptance"] = "optimal"
    ao_dict = create_ao_grid_event(config).action.data["AO"]["ao_dict"]

    mirror = SimpleNamespace(
        positions_modal_array=np.zeros((1, 2)),
        positions_voltage_array=np.zeros((1, 3)),
        current_coeffs=np.asarray([1.0, 2.0]),
        current_voltage=np.asarray([3.0, 4.0, 5.0]),
    )
    calls: list[dict] = []
    monkeypatch.setattr(
        sensorless_ao,
        "CMMCorePlus",
        SimpleNamespace(instance=lambda: demo_core),
    )
    monkeypatch.setattr(sensorless_ao.AOMirror, "instance", lambda: mirror)
    monkeypatch.setattr(
        sensorless_ao,
        "run_ao_optimization",
        lambda **kwargs: calls.append(kwargs),
    )
    completed = sensorless_ao.run_ao_grid_mapping(
        ao_dict=ao_dict,
        stage_positions=[{"x": 0.0, "y": 0.0, "z": 0.0}],
        save_dir_path=workspace_tmp_path,
        verbose=False,
    )

    assert completed
    assert len(calls) == 1
    assert calls[0]["mode_acceptance"] == "optimal"
