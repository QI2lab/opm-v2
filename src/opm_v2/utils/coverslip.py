"""Coverslip-surface calibration shared by the GUI and event planners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

COVERSLIP_METADATA_KEY = "opm_coverslip_plane"


@dataclass(frozen=True)
class CoverslipPlane:
    """A physical-stage plane expressed relative to a well-conditioned origin."""

    origin_x_um: float
    origin_y_um: float
    origin_z_um: float
    slope_x: float
    slope_y: float
    fit_points_um: tuple[tuple[float, float, float], ...] = ()
    rms_error_um: float = 0.0

    def z_at(self, x_um: float, y_um: float) -> float:
        """Return the sample-stage Z predicted at one physical XY coordinate.

        Returns
        -------
        float
            Predicted sample-stage Z in micrometers.
        """
        return float(
            self.origin_z_um
            + self.slope_x * (float(x_um) - self.origin_x_um)
            + self.slope_y * (float(y_um) - self.origin_y_um)
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for useq metadata.

        Returns
        -------
        dict[str, Any]
            Plane parameters and original fit points.
        """
        return {
            "type": "plane",
            "origin_um": {
                "x": self.origin_x_um,
                "y": self.origin_y_um,
                "z": self.origin_z_um,
            },
            "slope_x": self.slope_x,
            "slope_y": self.slope_y,
            "fit_points_um": [list(point) for point in self.fit_points_um],
            "rms_error_um": self.rms_error_um,
        }

    @classmethod
    def from_metadata(cls, value: Mapping[str, Any]) -> CoverslipPlane:
        """Validate and construct a plane from nested useq metadata.

        Returns
        -------
        CoverslipPlane
            Validated plane.

        Raises
        ------
        ValueError
            If required plane fields are absent or invalid.
        """
        if value.get("type", "plane") != "plane":
            raise ValueError("Only planar coverslip calibration is supported")
        origin = value.get("origin_um")
        if not isinstance(origin, Mapping):
            raise ValueError("Coverslip plane metadata has no origin_um mapping")
        try:
            points = tuple(
                (float(point[0]), float(point[1]), float(point[2]))
                for point in value.get("fit_points_um", ())
            )
            plane = cls(
                origin_x_um=float(origin["x"]),
                origin_y_um=float(origin["y"]),
                origin_z_um=float(origin["z"]),
                slope_x=float(value["slope_x"]),
                slope_y=float(value["slope_y"]),
                fit_points_um=points,
                rms_error_um=float(value.get("rms_error_um", 0.0)),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError("Invalid coverslip plane metadata") from exc
        numbers = np.asarray(
            (
                plane.origin_x_um,
                plane.origin_y_um,
                plane.origin_z_um,
                plane.slope_x,
                plane.slope_y,
                plane.rms_error_um,
            )
        )
        if not np.all(np.isfinite(numbers)):
            raise ValueError("Coverslip plane metadata must contain finite values")
        return plane


def fit_coverslip_plane(
    points_um: Sequence[Sequence[float]],
) -> CoverslipPlane:
    """Fit ``z = z0 + sx*(x-x0) + sy*(y-y0)`` to physical XYZ points.

    Returns
    -------
    CoverslipPlane
        Least-squares plane fit.

    Raises
    ------
    ValueError
        If fewer than three valid, non-collinear points are supplied.
    """
    points = np.asarray(points_um, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError("At least three XYZ focus points are required")
    if not np.all(np.isfinite(points)):
        raise ValueError("Coverslip focus points must be finite")

    origin_x, origin_y = np.mean(points[:, :2], axis=0)
    design = np.column_stack(
        (points[:, 0] - origin_x, points[:, 1] - origin_y, np.ones(len(points)))
    )
    coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
        design, points[:, 2], rcond=None
    )
    if rank < 3:
        raise ValueError("Coverslip focus points must not be collinear")
    slope_x, slope_y, origin_z = coefficients
    fitted_z = design @ coefficients
    rms_error = float(np.sqrt(np.mean(np.square(points[:, 2] - fitted_z))))
    return CoverslipPlane(
        origin_x_um=float(origin_x),
        origin_y_um=float(origin_y),
        origin_z_um=float(origin_z),
        slope_x=float(slope_x),
        slope_y=float(slope_y),
        fit_points_um=tuple(tuple(float(value) for value in point) for point in points),
        rms_error_um=rms_error,
    )
