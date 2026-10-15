"""Exercise O3 AO event routing, hardware isolation, and saved 3D results."""

import weakref
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import zarr

from opm_v2.engine.opm_custom_events import (
    create_ao_grid_event,
    create_ao_optimize_event,
)
from opm_v2.engine.opm_engine import OPMEngineV2
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.mock_nidaq import MockOPMNIDAQ
from opm_v2.hardware.PicardShutter import PicardShutter
from opm_v2.utils import sensorless_ao as ao
from opm_v2.utils.ao_o3_stack import (
    METRIC_NAME,
    O3StackMetric,
    o2_coordinates,
)


def test_engine_dispatch_passes_o3_settings(monkeypatch, opm_config_factory):
    """Preserve GUI scan settings through the acquisition engine boundary."""
    config = opm_config_factory(
        mode="projection",
        updates={
            "acq_config": {
                "AO": {
                    "metric": METRIC_NAME,
                    "o3_stack_range_um": 6.0,
                }
            }
        },
    )
    core = MagicMock()
    engine = object.__new__(OPMEngineV2)
    engine._mmcore_ref = weakref.ref(core)
    engine._config = config
    engine.AOMirror = AOMirror(simulate=True)
    engine.simulate_hardware = False
    optimize = MagicMock(return_value=True)
    monkeypatch.setattr("opm_v2.engine.opm_engine.run_ao_optimization", optimize)
    engine.exec_event(create_ao_optimize_event(config))
    assert optimize.call_args.kwargs["metric_to_use"] == METRIC_NAME
    assert optimize.call_args.kwargs["o3_stack_settings"]["range_um"] == 6.0
    assert optimize.call_args.kwargs["daq_mode"] == "2d"
    assert not any(optimize.call_args.kwargs["channel_states"])


def test_stop_during_ao_dispatch_cancels_without_completing_history(monkeypatch, opm_config_factory):
    """Deliver the GUI stop event to AO and cancel MDA after clean AO exit."""
    config = opm_config_factory(mode="projection")
    core = MagicMock()
    engine = object.__new__(OPMEngineV2)
    engine._mmcore_ref = weakref.ref(core)
    engine._config = config
    engine.AOMirror = AOMirror(simulate=True)
    engine.simulate_hardware = False
    engine._safe_stop_requested = Event()
    history = MagicMock()
    history.initializer.return_value = {}
    monkeypatch.setattr(engine, "_ao_history", lambda _: history)

    def optimize(**kwargs):
        assert not kwargs["stop_requested"]()
        engine.request_safe_stop()
        assert kwargs["stop_requested"]()
        return False

    monkeypatch.setattr("opm_v2.engine.opm_engine.run_ao_optimization", optimize)
    assert engine.exec_event(create_ao_optimize_event(config)) == ()
    core.mda.cancel.assert_called_once()
    history.complete.assert_not_called()


def test_stop_interrupts_averaged_2d_measurement():
    """Check STOP between individual camera frames, not after the full average."""
    core = MagicMock()
    core.snap.return_value = np.ones((10, 10))
    with pytest.raises(ao.AOCancelled):
        ao.acquire_metric_image(core, "DCT", 50, stop_requested=lambda: core.snap.call_count >= 3)
    assert core.snap.call_count == 3


@pytest.mark.parametrize("factory", [create_ao_optimize_event, create_ao_grid_event])
def test_o3_events_disable_imaging_lasers_and_projection(opm_config_factory, factory):
    """Ignore ordinary AO illumination and projection controls in O3 mode."""
    config = opm_config_factory(
        mode="projection",
        updates={
            "acq_config": {
                "AO": {
                    "metric": METRIC_NAME,
                    "active_channel_power": 0.0,
                    "daq_mode": "projection",
                    "o3_stack_range_um": 6.0,
                    "o3_stack_step_um": 0.5,
                }
            }
        },
    )
    event = factory(config)
    payload = event.action.data["AO"]
    payload = payload.get("ao_dict", payload)
    assert not any(payload["channel_states"])
    assert not any(payload["channel_powers"])
    assert payload["daq_mode"] == "2d"
    assert payload["lightsheet_mode"] == "off"
    assert payload["o3_stack_settings"]["range_um"] == 6.0
    assert (
        payload["o3_stack_settings"]["stage_name"]
        == config["O2O3-autofocus"]["O3_stage_name"]
    )
    assert (
        event.action.data["Camera"]["camera_crop"][3]
        == config["acq_config"]["camera_roi"]["crop_y"]
    )


@pytest.mark.parametrize("cancel_after", [None, 4, 12, 40])
def test_o3_optimization_acquires_real_stacks_and_saves_candidates(
    monkeypatch,
    workspace_tmp_path,
    cancel_after,
):
    """Drive the actual AO loop with synthetic camera planes and simulated devices."""
    mirror = AOMirror(simulate=True)
    daq = MockOPMNIDAQ()
    shutter = PicardShutter(simulate=True)
    shutter.state = "Open"
    opened = MagicMock()
    monkeypatch.setattr(shutter, "openShutter", opened)
    mmc = MagicMock()
    position = [10.0]
    moves = []

    def move(value):
        position[0] = value
        moves.append(value)

    stage = SimpleNamespace(
        getPosition=lambda: position[0], setPosition=move, wait=lambda: None
    )
    mmc.getDeviceObject.return_value = stage
    for name in ("getXPosition", "getYPosition", "getZPosition"):
        getattr(mmc, name).return_value = 0.0
    v, u = np.meshgrid(
        (np.arange(201) - 100) * 0.115, (np.arange(201) - 100) * 0.115, indexing="ij"
    )

    def snap():
        x, y, z = o2_coordinates(u, v, position[0] - 10.0, 30)
        sx = 0.3 + ((float(mirror.current_coeffs[7]) - 0.08) * 3) ** 2
        return 120 + 5000 * np.exp(
            -0.5 * ((x / sx) ** 2 + (y / 0.45) ** 2 + (z / 0.8) ** 2)
        )

    mmc.snap.side_effect = snap
    monkeypatch.setattr(ao, "CMMCorePlus", SimpleNamespace(instance=lambda: mmc))
    monkeypatch.setattr(ao, "_prepare_camera_for_ao_snap", lambda _: "Camera")
    monkeypatch.setattr(ao, "spherical_modes", [7])
    monkeypatch.setattr(ao.time, "sleep", lambda _: None)
    monkeypatch.setattr(mirror, "save_current_state", MagicMock())
    result = ao.run_ao_optimization(
        exposure_ms=5,
        channel_states=[True, False, False, False, False],
        metric_to_use=METRIC_NAME,
        daq_mode="projection",
        num_iterations=1,
        num_mode_samples=3,
        num_averaged_frames=1,
        modes_to_optimize="spherical only",
        starting_coef_delta=0.2,
        starting_mirror_coefficients=np.zeros(32),
        save_dir_path=workspace_tmp_path,
        verbose=False,
        stop_requested=lambda: cancel_after is not None and mmc.snap.call_count >= cancel_after,
    )
    if cancel_after is not None:
        assert result is False
        assert mmc.snap.call_count == cancel_after
        assert position[0] == 10.0
        assert not daq.running()
        np.testing.assert_array_equal(mirror.current_coeffs, np.zeros(32))
        assert not (workspace_tmp_path / "ao_results.zarr").exists()
        return
    assert result is True
    assert position[0] == 10.0
    assert shutter.state == "Closed"
    opened.assert_not_called()
    mmc.setFocusDevice.assert_not_called()
    assert daq.scan_type == "2d"
    assert not daq.digital_waveform.any()
    assert not daq.running()
    saved = zarr.open_group(str(workspace_tmp_path / "ao_results.zarr"), mode="r")
    assert saved["all_images"].ndim == 4
    measured = saved["o3_measured_stacks"]
    diagnostics = saved.attrs["o3_stack"]["measurements"]
    assert measured.shape[0] == len(diagnostics) == 5
    assert measured.shape[1] == 9
    assert mmc.snap.call_count == 45
    assert len(moves) == 50
    assert all(d["valid"] for d in diagnostics)
    assert all(type(edge) is int for edge in diagnostics[0]["crop_y0_y1_x0_x1"])
    assert all(edge > 0 for edge in diagnostics[0]["crop_y0_y1_x0_x1"])
    assert saved.attrs["channel_states"] == [False] * 5
    assert (workspace_tmp_path / "ao_metrics.png").is_file()
    journal = zarr.open_group(
        str(workspace_tmp_path / "o3_measurements.zarr"), mode="r"
    )
    assert len(list(journal.group_keys())) == 5
    assert journal.attrs["o3_start_um"] == 10.0
    assert journal.attrs["o3_return_um"] == 10.0
    assert journal.attrs["return_verified"] is True
    first_probe = journal["measurement_000001"]
    assert first_probe.attrs["kind"] == "probe"
    assert first_probe.attrs["delta"] == -0.2
    assert first_probe.attrs["mode_index"] == 7
    assert first_probe.attrs["iteration"] == 1
    np.testing.assert_array_equal(first_probe["stack"][:], measured[1])
    assert journal["measurement_000004"].attrs["kind"] == "candidate"


@pytest.mark.parametrize("interrupted", [False, True])
def test_stack_is_on_disk_before_fit_even_if_run_fails(
    workspace_tmp_path,
    monkeypatch,
    interrupted,
):
    """Persist measured pixels before an invalid fit or interrupted optimization."""
    output = workspace_tmp_path / "new_run"
    sampler = O3StackMetric(output_path=output)
    sampler.context = {"kind": "probe", "mode_index": 7, "delta": -0.3}
    stage = MagicMock()
    stage.getPosition.return_value = 10.0
    stage.setPosition.side_effect = lambda value: setattr(stage.getPosition, "return_value", value)
    mmc = MagicMock()
    mmc.getDeviceObject.return_value = stage
    image = np.arange(49 * 49, dtype=np.float32).reshape(49, 49)

    def fail_after_checking_disk(stack, offsets):
        saved = zarr.open_group(str(output / "o3_measurements.zarr"), mode="r")
        measurement = saved["measurement_000000"]
        assert measurement.attrs["fit_status"] == "pending"
        np.testing.assert_array_equal(measurement["stack"][:], stack)
        if interrupted:
            raise KeyboardInterrupt()
        raise ValueError("sigma_Z=0.212 um (lower bound 0.212 um)")

    monkeypatch.setattr(sampler, "measure", fail_after_checking_disk)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            sampler.acquire(mmc, lambda: image)
    else:
        _, metric = sampler.acquire(mmc, lambda: image)
        assert np.isnan(metric)
    saved = zarr.open_group(str(output / "o3_measurements.zarr"), mode="r")
    measurement = saved["measurement_000000"]
    assert measurement["stack"].shape[0] == 9
    assert measurement.attrs["delta"] == -0.3
    assert measurement.attrs["mode_index"] == 7
    assert measurement.attrs["o3_start_um"] == 10.0
    assert measurement.attrs["fit_status"] == ("pending" if interrupted else "invalid")
    if not interrupted:
        assert "sigma_Z=" in measurement.attrs["reason"]
    stage.setPosition.assert_called_with(10.0)


def test_failed_o3_return_is_recorded_before_ao_aborts(workspace_tmp_path):
    """Preserve the return target and failed readback for hardware diagnosis."""
    sampler = O3StackMetric(output_path=workspace_tmp_path)
    position = [10.0]

    def move(value):
        if value == 10.0 and position[0] == 12.0:
            return  # Simulate both return commands being ignored.
        position[0] = value

    stage = SimpleNamespace(getPosition=lambda: position[0], setPosition=move, wait=lambda: None)
    mmc = MagicMock()
    mmc.getDeviceObject.return_value = stage
    with pytest.raises(RuntimeError, match="O3 RETURN FAILED"):
        sampler.acquire(mmc, lambda: np.ones((5, 5)))
    saved = zarr.open_group(str(workspace_tmp_path / "o3_measurements.zarr"), mode="r")
    assert saved.attrs["o3_start_um"] == 10.0
    assert saved.attrs["return_verified"] is False
    assert "read back 12.000000" in saved.attrs["return_error"]
