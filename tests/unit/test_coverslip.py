"""Unit tests for coverslip-plane fitting and Stage Explorer calibration state."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from pymmcore_widgets.control._rois.roi_model import RectangleROI
from useq import AbsolutePosition, GridFromEdges, MDASequence

from opm_v2._app import (
    _add_coverslip_focus_point,
    _fit_coverslip_calibration,
    _position_with_coverslip_plane,
    _start_coverslip_calibration,
)
from opm_v2.engine.setup_events import StageExplorerRegion, parse_mda_position_plan
from opm_v2.utils.coverslip import CoverslipPlane, fit_coverslip_plane


def test_coverslip_plane_fit_and_metadata_round_trip() -> None:
    """Recover a physical plane from redundant focus points."""
    points = [
        (0.0, 0.0, 10.0),
        (100.0, 0.0, 11.0),
        (0.0, 100.0, 8.0),
        (100.0, 100.0, 9.0),
    ]

    plane = fit_coverslip_plane(points)
    restored = CoverslipPlane.from_metadata(plane.to_metadata())

    assert restored.slope_x == pytest.approx(0.01)
    assert restored.slope_y == pytest.approx(-0.02)
    assert restored.z_at(25.0, 75.0) == pytest.approx(8.75)
    assert restored.rms_error_um == pytest.approx(0.0, abs=1e-12)


def test_stage_explorer_calibration_uses_standard_live_mode_and_restores() -> None:
    """Collect sample-stage points in Standard mode and attach their plane."""
    controller = SimpleNamespace(info=MagicMock(), warning=MagicMock())
    roi = RectangleROI((0.0, 0.0), (100.0, 100.0))
    manager = SimpleNamespace(selected_rois=MagicMock(return_value=[roi]))
    mmc = MagicMock()
    mmc.mda.is_running.return_value = False
    mmc.getAvailableConfigGroups.return_value = ("OPM-live-mode",)
    mmc.getAvailableConfigs.return_value = ("Standard", "Projection")
    mmc.getCurrentConfig.side_effect = ["Projection", "Standard"]
    mmc.getXYPosition.side_effect = [(10.0, 10.0), (90.0, 10.0), (10.0, 90.0)]
    mmc.getZPosition.side_effect = [9.9, 10.7, 8.3]
    points_visual = MagicMock()
    explorer = SimpleNamespace(
        _mmc=mmc,
        _opm_controller=controller,
        _opm_coverslip_points=[],
        _opm_coverslip_points_visual=points_visual,
        _opm_coverslip_target_roi=None,
        _opm_coverslip_previous_live_preset=None,
        roi_manager=manager,
        window=MagicMock(),
    )

    _start_coverslip_calibration(explorer)
    _add_coverslip_focus_point(explorer)
    _add_coverslip_focus_point(explorer)
    _add_coverslip_focus_point(explorer)
    _fit_coverslip_calibration(explorer)

    assert mmc.setConfig.call_args_list == [
        call("OPM-live-mode", "Standard"),
        call("OPM-live-mode", "Projection"),
    ]
    assert mmc.waitForConfig.call_args_list == mmc.setConfig.call_args_list
    plane = roi._opm_coverslip_plane
    assert plane.slope_x == pytest.approx(0.01)
    assert plane.slope_y == pytest.approx(-0.02)
    assert plane.z_at(50.0, 50.0) == pytest.approx(9.5)
    assert explorer._opm_coverslip_target_roi is None


def test_coverslip_plane_rejects_collinear_xy_points() -> None:
    """Require focus samples that constrain both physical stage axes."""
    with pytest.raises(ValueError, match="must not be collinear"):
        fit_coverslip_plane([(0, 0, 1), (1, 1, 2), (2, 2, 3)])


def test_coverslip_plane_survives_nested_mda_position_serialization() -> None:
    """Carry the per-ROI plane through the native MDA positions model."""
    plane = CoverslipPlane(10.0, 20.0, 30.0, 0.01, -0.02)
    position = _position_with_coverslip_plane(
        AbsolutePosition(
            z=30.0,
            sequence=MDASequence(
                grid_plan=GridFromEdges(
                    left=10.0,
                    right=20.0,
                    top=20.0,
                    bottom=30.0,
                    fov_width=5.0,
                    fov_height=5.0,
                )
            ),
        ),
        plane,
    )

    parsed = parse_mda_position_plan([position.model_dump(mode="json")])

    assert len(parsed) == 1
    assert isinstance(parsed[0], StageExplorerRegion)
    assert parsed[0].coverslip_plane == plane
