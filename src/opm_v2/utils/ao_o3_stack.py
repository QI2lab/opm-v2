"""O3 through-focus AO measurements in the rotated O2 coordinate frame.

Camera Y is the tilt direction. O3 moves along its own optical axis, so this
scan is a rotation of an orthogonal (camera X, camera Y, O3) sampling grid,
not the shear used for a sample-stage OPM scan. No image resampling is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import zarr
from pymmcore_plus.core import StageDevice
from scipy.optimize import curve_fit

from opm_v2.utils.ao_cancel import check_ao_stop
from opm_v2.utils.metadata_serialization import json_metadata

METRIC_NAME = "gauss_3d_o3"
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))
POSITION_TOLERANCE_UM = 0.05


@dataclass(frozen=True)
class O3StackSettings:
    """Physical scan geometry; camera calibration is in remote-image space."""

    range_um: float = 4.0
    step_um: float = 0.5
    tilt_deg: float = 30.0
    pixel_size_um: float = 0.115
    stage_name: str = "MCL NanoDrive Z Stage"
    crop_half_width_px: int = 48

    def offsets(self) -> np.ndarray:
        """Return a symmetric scan with an exact center plane and endpoints.

        Returns
        -------
        numpy.ndarray
            O3 displacements in micrometers.

        Raises
        ------
        ValueError
            If the scan cannot have at least five evenly spaced centered planes.
        """
        values = (self.range_um, self.step_um, self.tilt_deg, self.pixel_size_um)
        if (
            not np.all(np.isfinite(values))
            or min(self.range_um, self.step_um, self.pixel_size_um) <= 0
        ):
            raise ValueError(
                "O3 range, step, and pixel scale must be positive and finite"
            )
        intervals = self.range_um / self.step_um
        if (
            not np.isclose(intervals, round(intervals))
            or round(intervals) < 4
            or round(intervals) % 2
            or round(intervals) > 100
        ):
            raise ValueError("O3 range/step must be an even integer from 4 to 100")
        return np.linspace(-self.range_um / 2, self.range_um / 2, round(intervals) + 1)


def o2_coordinates(u, v, w, tilt_deg):
    """Rotate camera X/Y and O3 travel into the O2 frame.

    Returns
    -------
    tuple
        Physical O2 X, Y, Z coordinates with the same input units.
    """
    theta = np.deg2rad(tilt_deg)
    return (
        u,
        v * np.cos(theta) + w * np.sin(theta),
        -v * np.sin(theta) + w * np.cos(theta),
    )


@dataclass
class O3StackMetric:
    """One AO run's fixed O3 scan, spatial prior, and measurement diagnostics."""

    settings: O3StackSettings = field(default_factory=O3StackSettings)
    output_path: Path | None = None
    start_um: float | None = None
    bounds: tuple[int, int, int, int] | None = None
    prior_center: np.ndarray | None = None
    core_sigmas: np.ndarray | None = None
    diagnostics: list[dict] = field(default_factory=list)
    measured_stacks: list[np.ndarray] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    _journal: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Create a separate measurement journal without overwriting a prior run."""
        if self.output_path is not None:
            self.output_path = Path(self.output_path)
            self.output_path.mkdir(parents=True, exist_ok=True)
            self._journal = zarr.open_group(
                str(self.output_path / "o3_measurements.zarr"),
                mode="w-",
                attributes=json_metadata({
                    "settings": asdict(self.settings),
                    "axes": ["o3", "camera_y", "camera_x"],
                    "illumination": "manually enabled external alignment beam",
                }),
            )

    def acquire(self, mmc, snap, num_frames=1, stop_requested=None):
        """Acquire and score a stack, returning O3 even when a snap fails.

        Returns
        -------
        tuple
            Cropped stack (plane, Y, X) and its scalar AO score.

        Raises
        ------
        ValueError
            If scan settings or the requested frame average are invalid.
        RuntimeError
            If O3 fails to reach a scan position or return to its origin.
        """
        offsets = self.settings.offsets()
        check_ao_stop(stop_requested)
        if num_frames < 1:
            raise ValueError("O3 stack needs at least one frame per plane")
        stage = mmc.getDeviceObject(self.settings.stage_name, StageDevice)
        stage.wait()
        if self.start_um is None:
            self.start_um = float(stage.getPosition())
        if not np.isfinite(self.start_um):
            raise ValueError("O3 starting position must be finite")
        if self._journal is not None:
            self._journal.attrs.update({
                "o3_start_um": self.start_um,
                "return_verified": False,
            })
        images = []
        positions = []
        try:
            for offset in offsets:
                check_ao_stop(stop_requested)
                positions.append(self._move_verified(stage, self.start_um + offset))
                image = np.mean(
                    [np.asarray(snap(), dtype=np.float32) for _ in range(num_frames)],
                    axis=0,
                )
                images.append(image)
        finally:
            # Never adopt a displaced position as the next stack's return target.
            # Retry a failed return once, then abort AO with the actual readback.
            try:
                try:
                    returned_um = self._move_verified(stage, self.start_um)
                except Exception:
                    returned_um = self._move_verified(stage, self.start_um)
            except Exception as exc:
                if self._journal is not None:
                    self._journal.attrs["return_error"] = str(exc)
                raise RuntimeError(
                    f"O3 RETURN FAILED: target {self.start_um:.6f} um; {exc}"
                ) from exc
            if self._journal is not None:
                self._journal.attrs.update({
                    "return_verified": True,
                    "o3_return_um": returned_um,
                    "return_error": None,
                })
        stack = np.asarray(images, dtype=np.float32)
        if self.bounds is None:
            # A projection is used only to locate the initial ROI, never to score it.
            _, cy, cx = np.unravel_index(np.argmax(stack), stack.shape)
            cy, cx = int(cy), int(cx)
            half = self.settings.crop_half_width_px
            self.bounds = (
                max(0, cy - half),
                min(stack.shape[1], cy + half + 1),
                max(0, cx - half),
                min(stack.shape[2], cx + half + 1),
            )
        y0, y1, x0, x1 = self.bounds
        stack = stack[:, y0:y1, x0:x1].copy()
        measurement = json_metadata({
            **self.context,
            "o3_start_um": self.start_um,
            "o3_positions_um": positions,
            "o3_return_um": returned_um,
            "return_verified": True,
            "crop_y0_y1_x0_x1": list(self.bounds),
            "stack_index": len(self.measured_stacks),
            "fit_status": "pending",
        })
        saved = None
        if self._journal is not None:
            saved = self._journal.create_group(
                f"measurement_{len(self.measured_stacks):06d}", attributes=measurement
            )
            # Persist the exact fit input before invoking the optimizer. Invalid
            # fits and interrupted runs must not discard the observed images.
            saved.create_array("stack", data=stack)
        try:
            score, diagnostic = self.measure(
                stack, np.asarray(positions) - self.start_um
            )
        except (ValueError, RuntimeError) as exc:
            score, diagnostic = np.nan, {"valid": False, "reason": str(exc)}
        diagnostic.update(measurement)
        diagnostic["fit_status"] = "valid" if diagnostic["valid"] else "invalid"
        diagnostic = json_metadata(diagnostic)
        if saved is not None:
            saved.attrs.update(diagnostic)
        self.diagnostics.append(diagnostic)
        self.measured_stacks.append(stack)
        return stack, score

    def _move_verified(self, stage, target_um):
        """Wait for the named O3 device and check its position readback.

        Returns
        -------
        float
            Reported stage position after settling.

        Raises
        ------
        RuntimeError
            If the stage does not report reaching the commanded position.
        """
        target_um = float(target_um)
        stage.setPosition(target_um)
        stage.wait()
        actual_um = float(stage.getPosition())
        if not np.isfinite(actual_um) or abs(actual_um - target_um) > POSITION_TOLERANCE_UM:
            raise RuntimeError(
                f"{self.settings.stage_name}: requested {target_um:.6f} um, "
                f"read back {actual_um:.6f} um "
                f"(tolerance {POSITION_TOLERANCE_UM} um)"
            )
        return actual_um

    def measure(self, stack, offsets=None):
        """Fit a rotated Gaussian and penalize energy outside a fixed core.

        Returns
        -------
        tuple
            Score and fitted widths, center, residual, and energy diagnostics.

        Raises
        ------
        ValueError
            If data, sampling, or the fitted spot is invalid.
        """
        data = np.asarray(stack, dtype=float)
        offsets = self.settings.offsets() if offsets is None else np.asarray(offsets)
        if (
            data.ndim != 3
            or data.shape[0] != len(offsets)
            or not np.all(np.isfinite(data))
            or np.ptp(data) <= 0
            or data.max() <= 0
        ):
            raise ValueError("O3 metric needs a finite 3D spot stack with contrast")
        scale = self.settings.pixel_size_um
        w, v, u = np.meshgrid(
            offsets,
            np.arange(data.shape[1]) * scale,
            np.arange(data.shape[2]) * scale,
            indexing="ij",
        )
        coords = tuple(a.ravel() for a in (u, v, w))
        normalized = data.ravel() / data.max()
        border = np.concatenate((
            data[:, 0, :].ravel(),
            data[:, -1, :].ravel(),
            data[:, :, 0].ravel(),
            data[:, :, -1].ravel(),
        ))
        background = max(float(np.median(border) / data.max()), 0)
        noise = float(
            1.4826 * np.median(np.abs(border - np.median(border))) / data.max()
        )
        signal = np.maximum(normalized - background, 0)
        if signal.max() <= max(5 * noise, 1e-12):
            raise ValueError("No O3 spot above the measured background noise")
        weights = np.where(signal > signal.max() * 0.5, signal, 0)
        seed_center = np.array([np.average(a, weights=weights) for a in coords])
        center = seed_center if self.prior_center is None else self.prior_center
        rotated = o2_coordinates(
            *(a - c for a, c in zip(coords, seed_center)), self.settings.tilt_deg
        )
        floors = np.array([scale, scale, self.settings.step_um]) / FWHM_PER_SIGMA
        sigmas = np.maximum(
            [np.sqrt(np.average(a * a, weights=weights)) * 1.5 for a in rotated],
            floors * 1.5,
        )
        radius = np.array([16 * scale, 16 * scale, self.settings.range_um / 2])
        lower_center = np.maximum(center - radius, [u.min(), v.min(), w.min()])
        upper_center = np.minimum(center + radius, [u.max(), v.max(), w.max()])
        upper_width = np.full(3, max(np.ptp(u), np.ptp(v), np.ptp(w)) * 2)
        lower = np.r_[0, lower_center, floors * (1 + 1e-6), 0]
        upper = np.r_[10, upper_center, upper_width, 1]

        def model(points, amplitude, cu, cv, cw, sx, sy, sz, bg):
            x, y, z = o2_coordinates(
                points[0] - cu, points[1] - cv, points[2] - cw, self.settings.tilt_deg
            )
            return bg + amplitude * np.exp(
                -0.5 * ((x / sx) ** 2 + (y / sy) ** 2 + (z / sz) ** 2)
            )

        initial = np.r_[signal.max(), center, sigmas, max(background, 1e-9)]
        fitted, _ = curve_fit(
            model, coords, normalized, p0=initial, bounds=(lower, upper), maxfev=500
        )
        if not np.all(np.isfinite(fitted)):
            raise ValueError("O3 Gaussian fit returned nonfinite parameters")
        bound_hits = []
        names = (
            "camera_X_center",
            "camera_Y_center",
            "O3_center",
            "sigma_X",
            "sigma_Y",
            "sigma_Z",
        )
        for index, name in enumerate(names, 1):
            for side, bounds in (("lower", lower), ("upper", upper)):
                if np.isclose(fitted[index], bounds[index], atol=1e-5, rtol=0):
                    bound_hits.append(
                        f"{name}={fitted[index]:.6g} um ({side} bound {bounds[index]:.6g} um)"
                    )
        if bound_hits:
            raise ValueError(
                "O3 Gaussian fit reached a bound: " + "; ".join(bound_hits)
            )
        fitted_center, fitted_sigmas = fitted[1:4], fitted[4:7]
        if fitted[0] <= max(5 * noise, 1e-12):
            raise ValueError("Fitted O3 spot amplitude is below the background noise")
        core_center = fitted_center if self.prior_center is None else self.prior_center
        core_sigmas = fitted_sigmas if self.core_sigmas is None else self.core_sigmas
        core_coords = o2_coordinates(
            *(a - c for a, c in zip(coords, core_center)), self.settings.tilt_deg
        )
        core = sum((a / (2 * s)) ** 2 for a, s in zip(core_coords, core_sigmas)) <= 1
        # The Gaussian's fitted offset can absorb unmodeled tails. Measure
        # energy against the independent ROI-border background instead.
        bg_subtracted = normalized - background
        core_energy = max(float(bg_subtracted[core].sum()), 0)
        tail_energy = max(float(bg_subtracted[~core].sum()), 0)
        total = core_energy + tail_energy
        if total <= 0 or core_energy <= 0:
            raise ValueError(
                "No positive background-subtracted energy in the fixed O3 core"
            )
        core_fraction = core_energy / total
        residual = float(
            np.sum((normalized - model(coords, *fitted)) ** 2)
            / max(np.sum(bg_subtracted**2), 1e-12)
        )
        score = float(core_fraction**2 / (np.prod(fitted_sigmas) * (1 + residual)))
        if not np.isfinite(score) or score <= 0:
            raise ValueError("Nonfinite O3 metric")
        if self.prior_center is None:
            self.prior_center = fitted_center.copy()
            self.core_sigmas = fitted_sigmas.copy()
        return score, {
            "valid": True,
            "score": score,
            "center_camera_o3_um": fitted_center.tolist(),
            "fwhm_o2_xyz_um": (fitted_sigmas * FWHM_PER_SIGMA).tolist(),
            "core_fraction": core_fraction,
            "tail_fraction": 1 - core_fraction,
            "relative_squared_residual": residual,
            "amplitude_counts": float(fitted[0] * data.max()),
            "fit_background_counts": float(fitted[-1] * data.max()),
            "energy_background_counts": float(background * data.max()),
            "core_center_camera_o3_um": self.prior_center.tolist(),
            "core_radii_o2_xyz_um": (2 * self.core_sigmas).tolist(),
        }
