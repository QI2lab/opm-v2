"""Integrate sensorless-AO configuration, MMCore, and grid dispatch."""

from __future__ import annotations

import json
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


def test_ao_grid_normalizes_integral_float_counts(
    opm_config_factory,
) -> None:
    """Accept old GUI snapshots while emitting integer event counts."""
    config = opm_config_factory(
        mode="projection",
        updates={
            "acq_config": {
                "AO": {
                    "num_scan_positions": 2.0,
                    "num_tile_positions": 3.0,
                }
            }
        },
    )

    ao_data = create_ao_grid_event(config).action.data["AO"]

    assert ao_data["num_scan_positions"] == 2
    assert type(ao_data["num_scan_positions"]) is int
    assert ao_data["num_tile_positions"] == 3
    assert type(ao_data["num_tile_positions"]) is int


def test_ao_grid_rejects_fractional_counts(opm_config_factory) -> None:
    """Reject invalid counts before starting AO hardware work."""
    config = opm_config_factory(
        mode="projection",
        updates={"acq_config": {"AO": {"num_scan_positions": 2.5}}},
    )

    with pytest.raises(ValueError, match="num_scan_positions"):
        create_ao_grid_event(config)


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
        positions_modal_array=np.asarray([[9.0, 8.0], [0.0, 0.0]]),
        positions_voltage_array=np.asarray(
            [[9.0, 8.0, 7.0], [0.0, 0.0, 0.0]]
        ),
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
        position_indices=[1],
        num_tile_positions=1.0,
        num_scan_positions=1.0,
        save_dir_path=workspace_tmp_path,
        verbose=False,
    )

    assert completed
    assert len(calls) == 1
    assert calls[0]["mode_acceptance"] == "optimal"
    with open(workspace_tmp_path / "ao_stage_positions.json") as file:
        assert json.load(file) == [{"z": 0.0, "y": 0.0, "x": 0.0}]
    np.testing.assert_allclose(
        mirror.positions_modal_array,
        [[9.0, 8.0], [1.0, 2.0]],
    )
    np.testing.assert_allclose(
        mirror.positions_voltage_array,
        [[9.0, 8.0, 7.0], [3.0, 4.0, 5.0]],
    )
