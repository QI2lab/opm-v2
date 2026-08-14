"""Integrate sensorless-AO configuration, MMCore, and grid dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest
from pymmcore_plus import CMMCorePlus

from opm_v2.engine.opm_custom_events import create_ao_grid_event
from opm_v2.utils import sensorless_ao
from opm_v2.utils.position_tools import ao_grid_positions


def _camera_with_transient_snap_failure(image: np.ndarray) -> MagicMock:
    mmc = MagicMock()
    mmc.getCameraDevice.return_value = "OrcaFusionBT"
    mmc.isSequenceRunning.return_value = True
    mmc.hasProperty.return_value = True
    mmc.getAllowedPropertyValues.side_effect = lambda _camera, prop: {
        "Trigger": ("NORMAL", "START"),
        "TriggerPolarity": ("POSITIVE", "NEGATIVE"),
        "TRIGGER SOURCE": ("INTERNAL", "EXTERNAL"),
    }[prop]
    mmc.snap.side_effect = [RuntimeError("Unknown error in the device"), image]
    return mmc


def test_ao_snap_recovery_preserves_projection_waveform() -> None:
    """Reset the camera and retry without replacing projection with a 2D scan."""
    expected = np.arange(12, dtype=np.uint16).reshape(3, 4)
    mmc = _camera_with_transient_snap_failure(expected)
    daq = MagicMock(scan_type="projection")
    daq.programmed.return_value = True
    daq.running.return_value = True

    image = sensorless_ao._snap_ao_image(mmc, daq)

    np.testing.assert_array_equal(image, expected)
    assert mmc.snap.call_count == 2
    mmc.stopSequenceAcquisition.assert_called_once_with()
    assert mmc.setProperty.call_args_list == [
        call("OrcaFusionBT", "Trigger", "NORMAL"),
        call("OrcaFusionBT", "TriggerPolarity", "POSITIVE"),
        call("OrcaFusionBT", "TRIGGER SOURCE", "INTERNAL"),
    ]
    daq.stop_waveform_playback.assert_called_once_with()
    daq.start_waveform_playback.assert_called_once_with()
    daq.clear_tasks.assert_not_called()
    daq.generate_waveforms.assert_not_called()
    assert daq.scan_type == "projection"


def test_ao_snap_recovery_rebuilds_invalid_projection_tasks() -> None:
    """Rebuild invalid tasks from the DAQ's retained projection parameters."""
    mmc = _camera_with_transient_snap_failure(np.ones((2, 2), dtype=np.uint16))
    daq = MagicMock(scan_type="projection")
    daq.programmed.side_effect = (False, True)
    daq.running.return_value = True

    sensorless_ao._snap_ao_image(mmc, daq)

    daq.clear_tasks.assert_called_once_with()
    daq.generate_waveforms.assert_called_once_with()
    daq.program_daq_waveforms.assert_called_once_with()
    daq.start_waveform_playback.assert_called_once_with()
    assert daq.scan_type == "projection"


def test_ao_snap_propagates_after_one_failed_recovery() -> None:
    """Bound recovery to one retry when the camera remains unavailable."""
    mmc = _camera_with_transient_snap_failure(np.ones((2, 2), dtype=np.uint16))
    mmc.snap.side_effect = [RuntimeError("first"), RuntimeError("second")]
    daq = MagicMock(scan_type="projection")
    daq.programmed.return_value = True
    daq.running.return_value = True

    with pytest.raises(RuntimeError, match="after one recovery attempt"):
        sensorless_ao._snap_ao_image(mmc, daq)

    assert mmc.snap.call_count == 2
    daq.start_waveform_playback.assert_called_once_with()


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


@pytest.mark.parametrize(
    ("z_slope_x", "z_slope_y"),
    (
        (0.0, 0.0),
        (0.1, 0.1),
        (0.3, -0.2),
    ),
    ids=("flat", "positive-xy-tilt", "mixed-xy-tilt"),
)
def test_ao_grid_mapping_distributes_results_by_xy_region(
    monkeypatch: pytest.MonkeyPatch,
    workspace_tmp_path,
    opm_config_factory,
    demo_core: CMMCorePlus,
    z_slope_x: float,
    z_slope_y: float,
) -> None:
    """Execute AO optimization and distribute its results by lateral region."""
    config = opm_config_factory(
        mode="projection",
        updates={
            "acq_config": {
                "AO": {
                    "num_scan_positions": 3,
                    "num_tile_positions": 3,
                }
            }
        },
    )
    ao_dict = create_ao_grid_event(config).action.data["AO"]["ao_dict"]
    stage_positions = [
        {
            "x": x,
            "y": y,
            "z": 45.0 + z_slope_x * x + z_slope_y * y,
        }
        for x in (0.0, 1.0, 2.0)
        for y in (0.0, 1.0, 2.0)
    ]
    mirror = SimpleNamespace(
        positions_modal_array=np.full((len(stage_positions), 2), -1.0),
        positions_voltage_array=np.full((len(stage_positions), 3), -1.0),
        current_coeffs=np.zeros(2),
        current_voltage=np.zeros(3),
    )
    optimization_count = 0
    optimization_acceptance: list[str] = []

    def _record_optimization(**kwargs) -> None:
        nonlocal optimization_count
        optimization_count += 1
        optimization_acceptance.append(kwargs["mode_acceptance"])
        mirror.current_coeffs = np.asarray(
            [optimization_count, -optimization_count],
            dtype=float,
        )
        mirror.current_voltage = np.asarray(
            [optimization_count, optimization_count + 0.1, optimization_count + 0.2],
            dtype=float,
        )

    monkeypatch.setattr(
        sensorless_ao,
        "CMMCorePlus",
        SimpleNamespace(instance=lambda: demo_core),
    )
    monkeypatch.setattr(sensorless_ao.AOMirror, "instance", lambda: mirror)
    monkeypatch.setattr(
        sensorless_ao,
        "run_ao_optimization",
        _record_optimization,
    )

    completed = sensorless_ao.run_ao_grid_mapping(
        ao_dict=ao_dict,
        stage_positions=stage_positions,
        position_indices=list(range(len(stage_positions))),
        num_tile_positions=3,
        num_scan_positions=3,
        save_dir_path=workspace_tmp_path,
        verbose=False,
    )

    ao_positions = ao_grid_positions(stage_positions, 3, 3)
    expected_grid_indices = np.asarray([0, 3, 6, 1, 4, 7, 2, 5, 8])
    expected_first_coefficients = expected_grid_indices.astype(float) + 1.0
    assert completed
    assert optimization_count == len(ao_positions)
    assert optimization_acceptance == [
        config["acq_config"]["AO"]["metric_acceptance"]
    ] * len(ao_positions)
    np.testing.assert_allclose(
        mirror.positions_modal_array[:, 0],
        expected_first_coefficients,
    )
    np.testing.assert_allclose(
        mirror.positions_modal_array[:, 1],
        -expected_first_coefficients,
    )
    np.testing.assert_allclose(
        mirror.positions_voltage_array[:, 0],
        expected_first_coefficients,
    )
    np.testing.assert_allclose(
        mirror.positions_voltage_array[:, 1],
        expected_first_coefficients + 0.1,
    )
    np.testing.assert_allclose(
        mirror.positions_voltage_array[:, 2],
        expected_first_coefficients + 0.2,
    )
