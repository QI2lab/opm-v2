"""Validate rotated O3 fits, tail sensitivity, and stage restoration."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from opm_v2.utils.ao_o3_stack import (
    FWHM_PER_SIGMA,
    O3StackMetric,
    O3StackSettings,
    o2_coordinates,
)


def synthetic_stack(widths=(0.3, 0.45, 0.8), secondary=0):
    """Create a spot in O2 coordinates sampled by the tilted camera.

    Returns
    -------
    numpy.ndarray
        Nine through-focus planes with a constant camera background.
    """
    s = O3StackSettings()
    w, v, u = np.meshgrid(
        s.offsets(),
        (np.arange(49) - 24) * s.pixel_size_um,
        (np.arange(49) - 24) * s.pixel_size_um,
        indexing="ij",
    )
    x, y, z = o2_coordinates(u, v, w, s.tilt_deg)
    spot = 5000 * np.exp(
        -0.5 * ((x / widths[0]) ** 2 + (y / widths[1]) ** 2 + (z / widths[2]) ** 2)
    )
    tail = secondary * np.exp(
        -0.5 * (((x - 1.3) / 0.4) ** 2 + (y / 0.5) ** 2 + (z / 0.8) ** 2)
    )
    return 120 + spot + tail


def test_default_scan_and_rotation():
    """Use nine planes across four micrometers and rotate the O3 scan axis."""
    settings = O3StackSettings()
    np.testing.assert_allclose(settings.offsets(), np.arange(-2, 2.01, 0.5))
    np.testing.assert_allclose(o2_coordinates(0, 0, 1, 30), (0, 0.5, np.sqrt(3) / 2))
    np.testing.assert_allclose(o2_coordinates(0, 1, 0, 30), (0, np.sqrt(3) / 2, -0.5))


def test_rotated_fit_recovers_physical_widths_and_does_not_depend_on_brightness():
    """Recover O2 widths without interpolation or per-plane normalization."""
    metric = O3StackMetric()
    first, diagnostic = metric.measure(synthetic_stack())
    np.testing.assert_allclose(
        diagnostic["fwhm_o2_xyz_um"],
        np.array([0.3, 0.45, 0.8]) * FWHM_PER_SIGMA,
        rtol=1e-4,
    )
    dim, _ = metric.measure(120 + (synthetic_stack() - 120) * 0.1)
    assert dim == pytest.approx(first, rel=1e-4)


def test_secondary_peaks_lower_score_with_the_same_sharp_core():
    """The fitted Gaussian cannot hide tails by raising its background."""
    metric = O3StackMetric()
    clean, initial = metric.measure(synthetic_stack())
    weak, weak_diagnostic = metric.measure(synthetic_stack(secondary=1000))
    strong, strong_diagnostic = metric.measure(synthetic_stack(secondary=2500))
    assert 0 < strong < weak < clean
    assert (
        strong_diagnostic["tail_fraction"]
        > weak_diagnostic["tail_fraction"]
        > initial["tail_fraction"]
    )
    assert strong_diagnostic["core_radii_o2_xyz_um"] == initial["core_radii_o2_xyz_um"]
    assert (
        strong_diagnostic["core_center_camera_o3_um"]
        == initial["core_center_camera_o3_um"]
    )


@pytest.mark.parametrize(
    "settings",
    [
        O3StackSettings(step_um=0),
        O3StackSettings(step_um=0.6),
        O3StackSettings(range_um=0.5),
        O3StackSettings(pixel_size_um=0),
    ],
)
def test_invalid_scan_rejected_before_stage_access(settings):
    """Do not move O3 for invalid or asymmetrically sampled scans."""
    mmc = MagicMock()
    with pytest.raises(ValueError):
        O3StackMetric(settings).acquire(mmc, MagicMock())
    mmc.getDeviceObject.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_stack_restores_stage_on_success_and_camera_error(fail):
    """Use the named O3 device and always return to its original position."""
    position = [10.0]
    moves = []

    def move(value):
        position[0] = value
        moves.append(value)

    stage = SimpleNamespace(
        getPosition=lambda: position[0], setPosition=move, wait=lambda: None
    )
    mmc = MagicMock()
    mmc.getDeviceObject.return_value = stage
    images = list(synthetic_stack())
    if fail:
        images[4] = RuntimeError("camera disconnected")
    snap = MagicMock(side_effect=images)
    metric = O3StackMetric()
    if fail:
        with pytest.raises(RuntimeError, match="disconnected"):
            metric.acquire(mmc, snap)
        assert not metric.diagnostics
    else:
        stack, score = metric.acquire(mmc, snap)
        assert np.isfinite(score)
        assert stack.shape[0] == 9
        np.testing.assert_allclose(moves[:-1], np.arange(8, 12.01, 0.5))
        assert metric.diagnostics[0]["stack_index"] == 0
        assert len(metric.measured_stacks) == 1
    assert moves[-1] == 10.0
    mmc.setFocusDevice.assert_not_called()


def test_missing_beam_is_invalid_without_losing_the_measured_stack():
    """Keep the data and explanation for an invalid measurement."""
    stage = MagicMock()
    stage.getPosition.return_value = 10.0
    stage.setPosition.side_effect = lambda value: setattr(
        stage.getPosition, "return_value", value
    )
    mmc = MagicMock()
    mmc.getDeviceObject.return_value = stage
    metric = O3StackMetric()
    _, score = metric.acquire(mmc, lambda: np.full((49, 49), 120))
    assert np.isnan(score)
    assert metric.diagnostics[0]["valid"] is False
    assert "contrast" in metric.diagnostics[0]["reason"]
    stage.setPosition.assert_called_with(10.0)


@pytest.mark.parametrize("failed_returns", [1, 2])
def test_failed_return_is_retried_or_aborts(failed_returns):
    """A stage reporting idle must still reach the original return target."""
    position = [10.0]
    returns = []

    def move(value):
        # The center plane is the first command to 10; later ones are returns.
        if value == 10.0:
            returns.append(value)
            if 1 < len(returns) <= failed_returns + 1:
                return
        position[0] = value

    stage = SimpleNamespace(
        getPosition=lambda: position[0], setPosition=move, wait=lambda: None
    )
    mmc = MagicMock()
    mmc.getDeviceObject.return_value = stage
    sampler = O3StackMetric()
    snap = MagicMock(side_effect=list(synthetic_stack()))
    if failed_returns == 2:
        with pytest.raises(
            RuntimeError, match=r"O3 RETURN FAILED.*read back 12\.000000"
        ):
            sampler.acquire(mmc, snap)
        assert not sampler.diagnostics
    else:
        sampler.acquire(mmc, snap)
        assert position[0] == 10.0
        assert sampler.diagnostics[0]["return_verified"]
    assert len(returns) == 3


def test_return_target_is_fixed_across_stacks():
    """Do not adopt an externally displaced position as the new return target."""
    stage = MagicMock()
    stage.getPosition.return_value = 10.0
    stage.setPosition.side_effect = lambda value: setattr(
        stage.getPosition, "return_value", value
    )
    mmc = MagicMock()
    mmc.getDeviceObject.return_value = stage
    sampler = O3StackMetric()
    sampler.acquire(mmc, MagicMock(side_effect=list(synthetic_stack())))
    stage.getPosition.return_value = 25.0
    sampler.acquire(mmc, MagicMock(side_effect=list(synthetic_stack())))
    assert sampler.start_um == 10.0
    assert stage.getPosition() == 10.0


def test_failed_scan_move_returns_home_before_aborting():
    """Do not acquire images at an unverified stage position."""
    stage = MagicMock()
    stage.getPosition.return_value = 10.0  # All moves ignored by hardware.
    mmc = MagicMock()
    mmc.getDeviceObject.return_value = stage
    snap = MagicMock()
    with pytest.raises(
        RuntimeError, match=r"requested 8\.000000.*read back 10\.000000"
    ):
        O3StackMetric().acquire(mmc, snap)
    snap.assert_not_called()
    stage.setPosition.assert_called_with(10.0)
