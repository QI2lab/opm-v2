"""Check Gaussian AO localization and fits with known single-spot images."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from opm_v2.utils import sensorless_ao as ao


def spot(center=(25.3, 30.2), widths=(2.0, 3.0), amplitude=5000.0):
    """Return a Gaussian on a camera background.

    Returns
    -------
    numpy.ndarray
        Synthetic camera image.
    """
    y, x = np.indices((80, 120))
    return 120 + amplitude * np.exp(
        -(
            (x - center[0]) ** 2 / (2 * widths[0] ** 2)
            + (y - center[1]) ** 2 / (2 * widths[1] ** 2)
        )
    )


def test_dim_broad_probe_uses_initial_spot_prior():
    """A dim aberrated probe scores below the initial spot, without recentering."""
    prior = ao.GaussianSpotPrior()
    initial = spot()
    dim = spot(center=(27.3, 33.2), widths=(7, 9), amplitude=350)
    camera = MagicMock()
    camera.snap.side_effect = [initial, dim]
    _, initial_score = ao.acquire_metric_image(camera, "gauss_2d", gaussian_prior=prior)
    center = prior.center
    _, dim_score = ao.acquire_metric_image(camera, "gauss_2d", gaussian_prior=prior)

    assert dim.max() < prior.threshold
    np.testing.assert_allclose(center, (25.3, 30.2), atol=0.1)
    assert prior.center == center
    assert 0 < dim_score < initial_score / 100


@pytest.mark.parametrize("center", [(25.3, 30.2), (3.3, 4.2), (116.3, 75.2)])
@pytest.mark.parametrize("crop_size", [0, 32])
def test_center_bounds_use_camera_coordinates_even_at_edges(center, crop_size):
    """Clipped crops and full images recover the known Gaussian score."""
    image = spot(center=center)
    score = ao.metric_gauss2d(image, image_center=center, crop_size=crop_size)
    expected = (5000 / image.max()) / (2 * 3) ** 2 * np.exp(-4 * (1 / 5) ** 2)
    assert score == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    "image",
    [spot(widths=(0.3, 0.3)), spot(center=(65, 30)), np.full((80, 120), 120.0)],
    ids=["subpixel-hotspot", "spot-outside-box", "constant-background"],
)
def test_invalid_spots_do_not_produce_an_ao_score(image):
    """Reject narrow noise and fits outside the original spot region."""
    assert np.isnan(ao.metric_gauss2d(image, image_center=(25.3, 30.2)))


def test_missing_initial_spot_has_explicit_error_without_centroid_warning():
    """A weak initial image must not silently seed the camera midpoint."""
    camera = MagicMock()
    camera.snap.return_value = spot(amplitude=350)
    prior = ao.GaussianSpotPrior()
    with pytest.raises(ValueError, match="no spot above threshold 1000"):
        ao.acquire_metric_image(camera, "gauss_2d", gaussian_prior=prior)
    assert prior.center is None


def test_averaged_start_initializes_one_prior_for_all_raw_frames():
    """Threshold the average once, even when one starting frame is dim."""
    camera = MagicMock()
    camera.snap.side_effect = [spot(), spot(amplitude=350)]
    prior = ao.GaussianSpotPrior()
    _, score = ao.acquire_metric_image(
        camera, "gauss_2d", num_averaged_frames=2, gaussian_prior=prior
    )
    assert np.isfinite(score)
    np.testing.assert_allclose(prior.center, (25.3, 30.2), atol=0.1)


def test_invalid_fit_returns_image_and_logs_reason(monkeypatch):
    """Solver failures retain the requested return shape and explain the NaN."""
    log = MagicMock()
    monkeypatch.setattr(ao, "warning", log)
    monkeypatch.setattr(
        ao, "curve_fit", MagicMock(side_effect=RuntimeError("no convergence"))
    )
    score, crop = ao.metric_gauss2d(spot(), return_image=True)
    assert np.isnan(score)
    assert crop.ndim == 2
    assert "no convergence" in log.call_args.args[1]


def test_bound_diagnostic_identifies_width_and_limit(monkeypatch):
    """Explain why a narrow spot is rejected instead of logging a generic bound."""
    log = MagicMock()
    monkeypatch.setattr(ao, "warning", log)
    score = ao.metric_gauss2d(spot(widths=(0.3, 0.3)))
    assert np.isnan(score)
    reason = log.call_args.args[1]
    assert "sigma_x=" in reason
    assert "sigma_y=" in reason
    assert "lower bound" in reason
    assert "FWHM=" in reason


@pytest.mark.parametrize("widths", [(1.0144, 0.9783), (0.9783, 1.0144), (0.5, 0.6)])
def test_spots_with_subpixel_sigma_but_fwhm_above_one_are_valid(widths):
    """Accept focused spots according to full width, including the reported case."""
    image = spot(widths=widths)
    prior = ao.GaussianSpotPrior()
    camera = MagicMock()
    camera.snap.return_value = image
    _, score = ao.acquire_metric_image(camera, "gauss_2d", gaussian_prior=prior)
    sx, sy = widths
    expected = (5000 / image.max()) / (sx * sy) ** 2 * np.exp(
        -4 * ((sx - sy) / (sx + sy)) ** 2
    )
    assert score == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("widths", [(0.4, 1.0), (1.0, 0.4)])
def test_fwhm_below_one_on_either_axis_is_rejected(widths):
    """Enforce the full-width minimum independently on both axes."""
    assert np.isnan(ao.metric_gauss2d(spot(widths=widths)))
