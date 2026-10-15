"""Exercise AO acceptance and recovery without connected hardware."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest
import zarr

from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.utils import sensorless_ao


@pytest.fixture
def ao_loop(monkeypatch):
    """Provide a real simulated mirror with isolated camera and DAQ boundaries.

    Returns
    -------
    SimpleNamespace
        Mirror, mocked acquisition boundaries, and optimization inputs.
    """
    mirror = AOMirror(simulate=True)
    # Use the same workspace-local artifact convention as integration tests;
    # Windows may deny access to pytest's mode-0700 temporary directories.
    output_path = Path.cwd() / ".pytest_tmp" / f"ao_acceptance_{uuid4().hex}"
    output_path.mkdir(parents=True)
    seed = np.zeros_like(mirror.current_coeffs)
    seed[7] = 0.1
    daq = MagicMock()
    daq.running.return_value = True
    mmc = MagicMock()
    for method in ("getXPosition", "getYPosition", "getZPosition"):
        getattr(mmc, method).return_value = 0.0
    monkeypatch.setattr(sensorless_ao.OPMNIDAQ, "instance", lambda: daq)
    monkeypatch.setattr(
        sensorless_ao, "CMMCorePlus", SimpleNamespace(instance=lambda: mmc)
    )
    monkeypatch.setattr(sensorless_ao, "_prepare_camera_for_ao_snap", lambda _: "Demo")
    monkeypatch.setattr(sensorless_ao, "spherical_modes", [7])
    monkeypatch.setattr(sensorless_ao.time, "sleep", lambda _: None)
    for method in ("save_current_state", "save_positions_array"):
        monkeypatch.setattr(mirror, method, MagicMock())
    saved = MagicMock()
    monkeypatch.setattr(sensorless_ao, "save_optimization_results", saved)
    for name in (
        "load_optimization_results",
        "plot_zernike_coeffs",
        "plot_metric_progress",
    ):
        monkeypatch.setattr(sensorless_ao, name, MagicMock())
    real_acquire = sensorless_ao.acquire_metric_image
    real_verify = sensorless_ao._measure_restored_correction
    # Policy-only tests supply a fixed probe/candidate sequence. Exercise the
    # additional camera measurement separately in the restoration tests below.
    verify = MagicMock()
    monkeypatch.setattr(sensorless_ao, "_measure_restored_correction", verify)
    acquire = MagicMock()
    monkeypatch.setattr(sensorless_ao, "acquire_metric_image", acquire)

    return SimpleNamespace(
        mirror=mirror,
        seed=seed,
        daq=daq,
        acquire=acquire,
        real_acquire=real_acquire,
        real_verify=real_verify,
        verify=verify,
        mmc=mmc,
        saved=saved,
        image=np.ones((4, 4)),
        kwargs={
            "exposure_ms": 10.0,
            "channel_states": [True, False, False, False, False],
            "num_iterations": 1,
            "num_mode_samples": 3,
            "modes_to_optimize": "spherical only",
            "starting_coef_delta": 0.2,
            "starting_mirror_coefficients": seed,
            "save_dir_path": output_path,
            "verbose": False,
        },
    )


@pytest.mark.parametrize("restored_metric", [0.4, 1.5, float("nan")])
def test_restoration_measurement_saved_without_changing_history(
    ao_loop, monkeypatch, restored_metric
):
    """Persist fresh images and readbacks without promoting any diagnostic score."""
    monkeypatch.setattr(sensorless_ao, "_measure_restored_correction", ao_loop.real_verify)
    metrics = iter([1.0, 0.2, 0.8, 0.2, 0.7, restored_metric])
    states = []

    def acquire(*args, **kwargs):
        states.append(ao_loop.mirror.current_coeffs.copy())
        return np.full((4, 4), len(states)), next(metrics)

    ao_loop.acquire.side_effect = acquire
    sensorless_ao.run_ao_optimization(
        **ao_loop.kwargs, mode_acceptance="optimal", num_averaged_frames=3
    )
    assert len(states) == 6
    np.testing.assert_array_equal(states[-1], ao_loop.seed)
    saved = ao_loop.saved.call_args.kwargs
    assert saved["optimal_metrics"][0] == 1.0
    np.testing.assert_array_equal(saved["optimal_coeffs"][0], ao_loop.seed)
    assert not saved["update_status"][0]
    group = zarr.open_group(
        str(ao_loop.kwargs["save_dir_path"] / "ao_restore_checks.zarr"), mode="r"
    )["check_000000"]
    np.testing.assert_array_equal(group["image"][:], np.full((4, 4), 6))
    attrs = dict(group.attrs)
    assert attrs["historical_best_metric"] == 1.0
    assert attrs["zero_probe_metric"] == 0.8
    assert attrs["num_averaged_frames"] == 3
    assert attrs["metric_valid"] == bool(np.isfinite(restored_metric))
    if np.isfinite(restored_metric):
        assert attrs["measured_metric"] == restored_metric
    np.testing.assert_array_equal(attrs["requested_voltage"], attrs["reported_voltage_before"])
    np.testing.assert_array_equal(attrs["requested_voltage"], attrs["reported_voltage_after"])
    assert ao_loop.acquire.call_args.kwargs == ao_loop.acquire.call_args_list[0].kwargs


@pytest.mark.parametrize("interval,expected", [(0, []), (2, [2, 4]), (1, [1, 2, 3, 4, 5])])
def test_alignment_autofocus_cadence_preserves_accepted_state(ao_loop, monkeypatch, interval, expected):
    """Count attempted modes across iterations without refocusing after the final mode."""
    from opm_v2.utils import autofocus_remote_unit as af

    monkeypatch.setattr(sensorless_ao, "spherical_modes", [7, 14, 23])
    ao_loop.acquire.side_effect = lambda *a, **kw: (
        ao_loop.image, 1.0 if ao_loop.acquire.call_count == 1 else 0.1
    )
    af_calls = []

    def focus(*args, **kwargs):
        assert kwargs["control_shutter"] is False
        assert kwargs["mmc"] is ao_loop.mmc
        np.testing.assert_array_equal(ao_loop.mirror.current_coeffs, ao_loop.seed)
        af_calls.append(kwargs)
        return 12.5

    monkeypatch.setattr(af, "manage_O3_focus", focus)
    assert sensorless_ao.run_ao_optimization(
        **{**ao_loop.kwargs, "num_iterations": 2}, daq_mode="2d",
        alignment_refocus_every_n_modes=interval, mode_acceptance="optimal",
        num_averaged_frames=51 if interval == 0 else 5,
    )
    records = ao_loop.saved.call_args.kwargs["metadata"]["alignment_refocus"]
    assert [r["after_modes"] for r in records] == expected
    assert len(af_calls) == len(expected)
    assert all(r["best_um"] == 12.5 for r in records)
    if expected:
        journal = zarr.open_group(str(ao_loop.kwargs["save_dir_path"] / "ao_alignment_focus.zarr"), mode="r")
        assert len(list(journal.group_keys())) == len(expected)


def test_alignment_autofocus_stop_keeps_mirror_and_stops_daq(ao_loop, monkeypatch):
    """Propagate cooperative STOP from the nested autofocus routine."""
    from opm_v2.utils import autofocus_remote_unit as af

    monkeypatch.setattr(sensorless_ao, "spherical_modes", [7, 14])
    ao_loop.acquire.side_effect = lambda *a, **kw: (
        ao_loop.image, 1.0 if ao_loop.acquire.call_count == 1 else 0.1
    )
    monkeypatch.setattr(af, "manage_O3_focus", MagicMock(side_effect=sensorless_ao.AOCancelled()))
    assert sensorless_ao.run_ao_optimization(
        **ao_loop.kwargs, daq_mode="2d", alignment_refocus_every_n_modes=1,
        mode_acceptance="optimal",
    ) is False
    np.testing.assert_array_equal(ao_loop.mirror.current_coeffs, ao_loop.seed)
    ao_loop.daq.stop_waveform_playback.assert_called()
    ao_loop.saved.assert_not_called()


@pytest.mark.parametrize("cancel", [False, True])
def test_restoration_verification_failure_stops_with_correction_retained(
    ao_loop, monkeypatch, cancel
):
    """Camera failure and STOP leave the restored command in place and stop DAQ."""
    monkeypatch.setattr(sensorless_ao, "_measure_restored_correction", ao_loop.real_verify)
    failure = sensorless_ao.AOCancelled() if cancel else RuntimeError("Camera failed")
    ao_loop.acquire.side_effect = [
        *[(ao_loop.image, m) for m in [1.0, 0.2, 0.8, 0.2, 0.7]], failure
    ]
    if cancel:
        assert sensorless_ao.run_ao_optimization(**ao_loop.kwargs) is False
    else:
        with pytest.raises(RuntimeError, match="Failed to verify restored"):
            sensorless_ao.run_ao_optimization(**ao_loop.kwargs)
    np.testing.assert_array_equal(ao_loop.mirror.current_coeffs, ao_loop.seed)
    ao_loop.daq.stop_waveform_playback.assert_called()
    ao_loop.saved.assert_not_called()


def test_restoration_journal_preserves_distinct_command_and_readbacks(ao_loop, monkeypatch):
    """Record actual controller reports independently of the requested voltage."""
    mirror = ao_loop.mirror
    assert mirror.set_modal_coefficients(ao_loop.seed)
    requested = mirror.last_requested_voltage.copy()
    reports = iter([requested + 0.001, requested + 0.002])
    monkeypatch.setattr(mirror, "_update_current_state", lambda: setattr(
        mirror, "current_voltage", next(reports)
    ))
    ao_loop.acquire.return_value = (ao_loop.image, 0.5)
    ao_loop.real_verify(
        mirror, ao_loop.mmc, ao_loop.daq, "gauss_2d", 3, {},
        {"step_index": 0, "historical_best_metric": 1.0, "zero_probe_metric": 0.7},
        ao_loop.kwargs["save_dir_path"],
    )
    attrs = zarr.open_group(
        str(ao_loop.kwargs["save_dir_path"] / "ao_restore_checks.zarr"), mode="r"
    )["check_000000"].attrs
    np.testing.assert_array_equal(attrs["requested_voltage"], requested)
    np.testing.assert_array_equal(attrs["reported_voltage_before"], requested + 0.001)
    np.testing.assert_array_equal(attrs["reported_voltage_after"], requested + 0.002)
    assert attrs["max_voltage_readback_error"] == pytest.approx(0.002, abs=1e-7)


@pytest.mark.parametrize("acceptance", ["optimal", "zero"])
def test_zero_probe_refreshes_best_metric_without_a_coefficient_change(
    ao_loop, acceptance
):
    """Retain the user's 0.949010 zero observation when the candidate is worse."""
    metrics = [0.809557, 0.005450, 0.949010, 0.011100, 0.93942]
    images = [np.full((4, 4), i) for i in range(len(metrics))]
    ao_loop.acquire.side_effect = list(zip(images, metrics))
    sensorless_ao.run_ao_optimization(**ao_loop.kwargs, mode_acceptance=acceptance)
    saved = ao_loop.saved.call_args.kwargs
    assert saved["optimal_metrics"][0] == pytest.approx(0.949010)
    np.testing.assert_array_equal(saved["optimal_images"][0], images[2])
    np.testing.assert_array_equal(ao_loop.mirror.current_coeffs, ao_loop.seed)
    assert not saved["update_status"][0]


@pytest.mark.parametrize("reapply_success", [True, False])
def test_optimal_keeps_best_measured_probe_with_matching_coefficients(
    ao_loop,
    monkeypatch,
    reapply_success,
):
    """Prefer a measured peak to an inferior fit, but reject a failed reapplication."""
    metrics = [1.0, 0.7, 1.4, 1.1, 1.3, 0.8, 1.2]
    images = [np.full((4, 4), i) for i in range(len(metrics))]
    ao_loop.acquire.side_effect = list(zip(images, metrics))
    apply = ao_loop.mirror.set_modal_coefficients
    attempts = []

    def command(coefficients):
        attempts.append(np.asarray(coefficients).copy())
        if len(attempts) == 8 and not reapply_success:
            return False
        return apply(coefficients)

    monkeypatch.setattr(ao_loop.mirror, "set_modal_coefficients", command)
    kwargs = {**ao_loop.kwargs, "num_mode_samples": 5}
    sensorless_ao.run_ao_optimization(**kwargs, mode_acceptance="optimal")
    saved = ao_loop.saved.call_args.kwargs
    expected = ao_loop.seed.copy()
    if reapply_success:
        expected[7] -= 0.1
    np.testing.assert_allclose(ao_loop.mirror.current_coeffs, expected)
    np.testing.assert_array_equal(
        saved["optimal_images"][0], images[2 if reapply_success else 3]
    )
    assert saved["optimal_metrics"][0] == pytest.approx(1.4 if reapply_success else 1.1)
    assert bool(saved["update_status"][0]) is reapply_success


def test_even_sweep_includes_actual_zero_delta(ao_loop, monkeypatch):
    """Never treat a positive perturbation as the reference for an even sample count."""
    applied = []
    apply = ao_loop.mirror.set_modal_coefficients

    def command(coefficients):
        applied.append(np.asarray(coefficients).copy())
        return apply(coefficients)

    monkeypatch.setattr(ao_loop.mirror, "set_modal_coefficients", command)
    ao_loop.acquire.side_effect = [
        (ao_loop.image, m) for m in [1, 0.7, 0.8, 1.3, 0.9, 1.1]
    ]
    sensorless_ao.run_ao_optimization(
        **{**ao_loop.kwargs, "num_mode_samples": 4}, mode_acceptance="zero"
    )
    np.testing.assert_array_equal(applied[3], ao_loop.seed)
    assert ao_loop.saved.call_args.kwargs["optimal_metrics"][0] == pytest.approx(1.3)


def test_gaussian_prior_covers_probes_candidates_and_resets_each_run(
    ao_loop, monkeypatch
):
    """Run real Gaussian scoring through the AO loop at two spot locations."""
    monkeypatch.setattr(sensorless_ao, "acquire_metric_image", ao_loop.real_acquire)
    real_metric = sensorless_ao.metric_gauss2d
    centers = []

    def record_metric(image, **kwargs):
        centers.append(kwargs["image_center"])
        return real_metric(image, **kwargs)

    monkeypatch.setattr(sensorless_ao, "metric_gauss2d", record_metric)
    y, x = np.indices((80, 120))
    for cx in (25.0, 75.0):
        images = [
            120 + amplitude * np.exp(-((x - cx) ** 2 + (y - 30) ** 2) / (2 * sigma**2))
            for sigma, amplitude in [
                (2, 5000),
                (3, 350),
                (2, 5000),
                (2.5, 5000),
                (1.8, 5000),
            ]
        ]
        ao_loop.mmc.snap.side_effect = images
        assert sensorless_ao.run_ao_optimization(
            **ao_loop.kwargs,
            metric_to_use="gauss_2d",
            mode_acceptance="accept all",
            num_averaged_frames=1,
        )
        assert len(centers) == (5 if cx == 25 else 10)
        np.testing.assert_allclose(centers[-5:], [(cx, 30)] * 5, atol=0.01)
        assert bool(ao_loop.saved.call_args.kwargs["update_status"][0])


@pytest.mark.parametrize("acceptance", ["accept all", "optimal", "zero"])
@pytest.mark.parametrize(
    "failure", ["probe", "candidate", "probe_metric", "candidate_metric", "fit"]
)
def test_invalid_ao_updates_restore_the_accepted_correction(
    ao_loop, monkeypatch, acceptance, failure
) -> None:
    """Every acceptance policy rejects invalid states and persists the safe state."""
    mirror = ao_loop.mirror
    apply = mirror.set_modal_coefficients
    attempts = []

    def command(coefficients):
        coefficients = np.asarray(coefficients).copy()
        assert np.all(np.isfinite(coefficients))
        attempts.append(coefficients)
        # Initialization, three probes, then the fitted candidate.
        if (failure == "probe" and len(attempts) == 3) or (
            failure == "candidate" and len(attempts) == 5
        ):
            return False
        return apply(coefficients)

    monkeypatch.setattr(mirror, "set_modal_coefficients", command)
    metrics = [1.0, 0.7, 1.0, 0.9, 1.1]
    if failure == "probe_metric":
        metrics[2] = np.nan
    elif failure == "candidate_metric":
        metrics[-1] = np.nan
    elif failure == "fit":
        monkeypatch.setattr(
            sensorless_ao, "quadratic_fit", lambda *_: (np.nan, np.nan, np.nan)
        )
    ao_loop.acquire.side_effect = [(ao_loop.image, metric) for metric in metrics]

    assert sensorless_ao.run_ao_optimization(
        **ao_loop.kwargs, mode_acceptance=acceptance
    )

    expected_snaps = {
        "probe": 2,
        "probe_metric": 3,
        "candidate": 4,
        "fit": 4,
        "candidate_metric": 5,
    }
    assert ao_loop.acquire.call_count == expected_snaps[failure]
    np.testing.assert_array_equal(mirror.current_coeffs, ao_loop.seed)
    np.testing.assert_array_equal(mirror.optimized_modal_coeffs, ao_loop.seed)
    saved = ao_loop.saved.call_args.kwargs
    np.testing.assert_array_equal(saved["update_status"], [False])
    np.testing.assert_array_equal(saved["optimal_coeffs"], [ao_loop.seed])
    np.testing.assert_array_equal(saved["optimal_metrics"], [1.0])
    assert len(saved["all_metrics"]) == 5
    ao_loop.daq.stop_waveform_playback.assert_called()


@pytest.mark.parametrize(
    ("acceptance", "candidate_metric", "accepted"),
    [
        ("accept all", 0.8, True),
        ("optimal", 0.8, False),
        ("zero", 0.8, False),
        ("optimal", 1.1, True),
        ("zero", 1.1, True),
    ],
)
def test_valid_ao_updates_obey_metric_policy(
    ao_loop, acceptance, candidate_metric, accepted
) -> None:
    """Accept-all still permits a finite lower score when the command succeeds."""
    ao_loop.acquire.side_effect = [
        (ao_loop.image, metric) for metric in [1.0, 0.7, 1.0, 0.9, candidate_metric]
    ]

    sensorless_ao.run_ao_optimization(**ao_loop.kwargs, mode_acceptance=acceptance)

    saved = ao_loop.saved.call_args.kwargs
    assert bool(saved["update_status"][0]) is accepted
    if accepted:
        assert ao_loop.mirror.current_coeffs[7] > ao_loop.seed[7]
    else:
        np.testing.assert_array_equal(ao_loop.mirror.current_coeffs, ao_loop.seed)
    np.testing.assert_array_equal(
        ao_loop.mirror.current_coeffs, saved["optimal_coeffs"][-1]
    )


@pytest.mark.parametrize("raises", [False, True])
def test_failed_restoration_stops_ao_without_saving(ao_loop, monkeypatch, raises):
    """Abort instead of continuing acquisition with an unknown mirror state."""
    apply = ao_loop.mirror.set_modal_coefficients
    calls = 0

    def command(coefficients):
        nonlocal calls
        calls += 1
        if calls >= 3:
            if raises and calls > 3:
                raise RuntimeError("Disconnected")
            return False
        return apply(coefficients)

    monkeypatch.setattr(ao_loop.mirror, "set_modal_coefficients", command)
    update_optimized = MagicMock()
    monkeypatch.setattr(ao_loop.mirror, "update_optimized_array", update_optimized)
    ao_loop.acquire.return_value = (ao_loop.image, 1.0)

    with pytest.raises(RuntimeError, match="restore last accepted AO"):
        sensorless_ao.run_ao_optimization(
            **ao_loop.kwargs, mode_acceptance="accept all"
        )

    ao_loop.daq.stop_waveform_playback.assert_called()
    update_optimized.assert_not_called()
    ao_loop.saved.assert_not_called()


def test_nonfinite_initial_metric_stops_before_probing(ao_loop, monkeypatch):
    """Do not start a modal search without a finite baseline metric."""
    apply = MagicMock(wraps=ao_loop.mirror.set_modal_coefficients)
    monkeypatch.setattr(ao_loop.mirror, "set_modal_coefficients", apply)
    ao_loop.acquire.return_value = (ao_loop.image, np.nan)

    with pytest.raises(RuntimeError, match="starting metric"):
        sensorless_ao.run_ao_optimization(**ao_loop.kwargs)

    assert apply.call_count == 1
    ao_loop.daq.stop_waveform_playback.assert_called()
    ao_loop.saved.assert_not_called()


def test_failed_mode_preserves_prior_mode_and_continues(ao_loop, monkeypatch):
    """A failed mode restores the latest accepted state, not the initial flat."""
    monkeypatch.setattr(sensorless_ao, "spherical_modes", [7, 14, 23])
    apply = ao_loop.mirror.set_modal_coefficients
    after_first_mode = None
    first_mode_probes = 0
    failed = False
    restored = False

    def command(coefficients):
        nonlocal failed, restored, first_mode_probes, after_first_mode
        coefficients = np.asarray(coefficients).copy()
        if not failed and coefficients[14] != 0:
            after_first_mode = ao_loop.mirror.current_coeffs.copy()
            failed = True
            return False
        if failed and not restored:
            np.testing.assert_array_equal(coefficients, after_first_mode)
            restored = True
        if not failed and coefficients[7] != ao_loop.seed[7]:
            first_mode_probes += 1
        return apply(coefficients)

    monkeypatch.setattr(ao_loop.mirror, "set_modal_coefficients", command)
    ao_loop.acquire.side_effect = [
        (ao_loop.image, metric)
        for metric in [1.0, 0.7, 1.0, 0.9, 1.1, 0.8, 1.1, 1.0, 1.2]
    ]

    sensorless_ao.run_ao_optimization(**ao_loop.kwargs, mode_acceptance="accept all")

    assert failed and restored
    assert first_mode_probes == 3
    saved = ao_loop.saved.call_args.kwargs
    np.testing.assert_array_equal(saved["update_status"], [True, False, True])
    np.testing.assert_array_equal(saved["optimal_coeffs"][1], after_first_mode)
    assert saved["optimal_coeffs"][-1, 7] == after_first_mode[7]
    assert saved["optimal_coeffs"][-1, 14] == 0.0
    assert saved["optimal_coeffs"][-1, 23] > 0.0
    assert len(saved["all_metrics"]) == 13


def test_failed_final_application_never_publishes_optimized_state(ao_loop, monkeypatch):
    """Do not save an optimized correction if final application is rejected."""
    apply = ao_loop.mirror.set_modal_coefficients
    calls = 0

    def command(coefficients):
        nonlocal calls
        calls += 1
        return False if calls == 6 else apply(coefficients)

    monkeypatch.setattr(ao_loop.mirror, "set_modal_coefficients", command)
    update_optimized = MagicMock()
    monkeypatch.setattr(ao_loop.mirror, "update_optimized_array", update_optimized)
    ao_loop.acquire.side_effect = [
        (ao_loop.image, metric) for metric in [1.0, 0.7, 1.0, 0.9, 1.1]
    ]

    with pytest.raises(RuntimeError, match="restore last accepted AO"):
        sensorless_ao.run_ao_optimization(
            **ao_loop.kwargs, mode_acceptance="accept all"
        )

    ao_loop.daq.stop_waveform_playback.assert_called()
    update_optimized.assert_not_called()
    ao_loop.saved.assert_not_called()


@pytest.mark.parametrize("failure", ["mirror", "camera"])
def test_unexpected_acquisition_failure_restores_accepted_state(
    ao_loop, monkeypatch, failure
):
    """Restore the accepted correction before propagating acquisition errors."""
    if failure == "mirror":
        apply = ao_loop.mirror.set_modal_coefficients
        calls = 0

        def command(coefficients):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("Mirror disconnected")
            return apply(coefficients)

        monkeypatch.setattr(ao_loop.mirror, "set_modal_coefficients", command)
        ao_loop.acquire.return_value = (ao_loop.image, 1.0)
    else:
        ao_loop.acquire.side_effect = [
            (ao_loop.image, 1.0),
            RuntimeError("Camera disconnected"),
        ]

    with pytest.raises(RuntimeError):
        sensorless_ao.run_ao_optimization(**ao_loop.kwargs)

    np.testing.assert_array_equal(ao_loop.mirror.current_coeffs, ao_loop.seed)
    ao_loop.saved.assert_not_called()


def test_failed_starting_state_stops_before_imaging(ao_loop, monkeypatch):
    """Do not silently optimize from another state if the chosen state fails."""
    monkeypatch.setattr(ao_loop.mirror, "apply_optimized_voltage", lambda: False)
    ao_loop.kwargs["starting_mirror_coefficients"] = None

    with pytest.raises(RuntimeError, match="starting AO mirror correction"):
        sensorless_ao.run_ao_optimization(**ao_loop.kwargs)

    ao_loop.acquire.assert_not_called()
    ao_loop.daq.stop_waveform_playback.assert_called()
