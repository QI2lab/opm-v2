"""Unit tests for OPM camera and laboratory position geometry."""

from __future__ import annotations

import numpy as np
import pytest

from opm_v2.utils.position_tools import (
    cam2lab,
    covering_tile_origins,
    lab2cam,
    oblique_camera_extents_um,
    split_stage_scan_bounds,
)


def test_lab_camera_transforms_match_processing_axis_convention() -> None:
    """Match the lab2cam/cam2lab equations used by opm-processing-v2."""
    camera_x = np.asarray([2.0, 4.0])
    camera_y = np.asarray([0.0, 40.0])
    raw_scan = np.asarray([100.0, 100.0])

    lab_x, lab_scan, lab_z = cam2lab(camera_x, camera_y, raw_scan, 30.0)

    assert lab_x == pytest.approx(camera_x)
    assert lab_scan == pytest.approx([100.0, 100.0 + 40.0 * np.cos(np.pi / 6)])
    assert lab_z == pytest.approx([0.0, 20.0])
    roundtrip = lab2cam(lab_x, lab_scan, lab_z, 30.0)
    assert roundtrip[0] == pytest.approx(camera_x)
    assert roundtrip[1] == pytest.approx(camera_y)
    assert roundtrip[2] == pytest.approx(raw_scan)


def test_mirror_tiling_uses_scan_plane_footprint() -> None:
    """Keep the deskewed camera extent out of mirror scan-plane tiling."""
    scan_extent_um, z_extent_um = oblique_camera_extents_um(386, 0.115, 30.0)

    assert scan_extent_um == pytest.approx(38.44286767)
    assert z_extent_um == pytest.approx(22.195)
    assert covering_tile_origins(0.0, 500.0, 200.0, 0.15) == pytest.approx(
        [0.0, 170.0, 340.0]
    )


def test_stage_split_preserves_bounds_and_physical_overlap() -> None:
    """Allow a raw trajectory gap while retaining overlap after deskewing."""
    starts, ends, raw_overlap_um, camera_extent_um = split_stage_scan_bounds(
        0.0,
        250.0,
        100.0,
        30.0,
        386,
        0.115,
        30.0,
    )

    assert starts[0] == pytest.approx(0.0)
    assert ends[-1] == pytest.approx(250.0)
    assert raw_overlap_um == pytest.approx(30.0 - camera_extent_um)
    assert np.diff(starts) == pytest.approx(np.diff(ends))
    assert np.max(ends - starts) <= 100.0
    assert ends[:-1] + camera_extent_um - starts[1:] == pytest.approx([30.0, 30.0])
