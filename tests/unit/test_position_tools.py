"""Unit tests for OPM camera and laboratory position geometry."""

from __future__ import annotations

import numpy as np
import pytest

from opm_v2.utils.position_tools import (
    ao_grid_positions,
    apply_oblique_scan_correction,
    cam2lab,
    covering_tile_origins,
    expand_stage_positions_for_depth,
    lab2cam,
    oblique_camera_extents_um,
    sample_depth_levels_um,
    split_stage_scan_bounds,
)


def test_ao_grid_positions_follow_coverslip_surface_in_both_xy_axes() -> None:
    """Evaluate AO target Z from the complete physical acquisition surface."""
    positions = [
        {
            "x": x,
            "y": y,
            "z": 45.0 + 0.01 * x - 0.02 * y,
        }
        for x in (100.0, 200.0)
        for y in (300.0, 500.0)
    ]

    targets = ao_grid_positions(
        positions,
        num_scan_positions=3,
        num_tile_positions=2,
    )

    assert len(targets) == 6
    for target in targets:
        assert target["z"] == pytest.approx(
            45.0 + 0.01 * target["x"] - 0.02 * target["y"],
            abs=0.01,
        )


def test_single_position_ao_grid_does_not_invent_an_xy_offset() -> None:
    """Keep AO at the acquisition position when no lateral slope is observable."""
    targets = ao_grid_positions(
        [{"x": 100.0, "y": 200.0, "z": 50.0}],
        num_scan_positions=3,
        num_tile_positions=3,
    )

    assert targets == [{"x": 100.0, "y": 200.0, "z": 50.0}]


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


def test_oblique_scan_correction_preserves_lab_xy_across_tilted_z() -> None:
    """Shift raw stage X so tilted-Z positions retain intended lab origins."""
    positions = [
        {"x": 100.0, "y": 200.0, "z": 50.0},
        {"x": 180.0, "y": 250.0, "z": 49.0},
    ]

    corrected = apply_oblique_scan_correction(
        positions,
        angle_deg=30.0,
        camera_zstage_orientation="positive",
    )

    assert [position["lab_scan_um"] for position in corrected] == [100.0, 180.0]
    assert corrected[0]["x"] == pytest.approx(100.0)
    assert corrected[1]["x"] == pytest.approx(180.0 - 1.0 / np.tan(np.pi / 6))
    for position in corrected:
        lab_z_offset = -(position["z"] - 50.0)
        camera_y = lab_z_offset / np.sin(np.pi / 6)
        _, lab_scan, _ = cam2lab(
            0.0,
            camera_y,
            position["x"],
            30.0,
        )
        assert float(lab_scan) == pytest.approx(position["lab_scan_um"])


def test_mirror_tiling_uses_scan_plane_footprint() -> None:
    """Keep the deskewed camera extent out of mirror scan-plane tiling."""
    scan_extent_um, z_extent_um = oblique_camera_extents_um(386, 0.115, 30.0)

    assert scan_extent_um == pytest.approx(38.44286767)
    assert z_extent_um == pytest.approx(22.195)
    assert covering_tile_origins(0.0, 500.0, 200.0, 0.15) == pytest.approx(
        [0.0, 170.0, 340.0]
    )


def test_stage_split_matches_main_overlap_rule() -> None:
    """Add the camera axial footprint to the configured extra overlap."""
    configured_overlap_um = 30.0
    starts, ends, effective_overlap_um, camera_z_extent_um = (
        split_stage_scan_bounds(
        0.0,
        250.0,
        100.0,
        configured_overlap_um,
        386,
        0.115,
        30.0,
        )
    )

    assert starts[0] == pytest.approx(0.0)
    assert ends[-1] == pytest.approx(250.0)
    assert camera_z_extent_um == pytest.approx(386 * 0.115 * np.sin(np.pi / 6))
    assert effective_overlap_um == pytest.approx(
        camera_z_extent_um + configured_overlap_um
    )
    assert np.diff(starts) == pytest.approx(np.diff(ends))
    assert len(starts) == int(np.ceil(250.0 / 100.0))
    assert ends[:-1] - starts[1:] == pytest.approx(
        [effective_overlap_um, effective_overlap_um]
    )


def test_sample_depth_levels_cover_requested_thickness() -> None:
    """Insert enough slab centers to maintain the requested axial overlap."""
    levels = sample_depth_levels_um(0.0, 40.0, 22.195, 0.2)

    assert levels == pytest.approx([0.0, 40 / 3, 80 / 3, 40.0])
    assert float(np.max(np.diff(levels))) <= 22.195 * 0.8 + 1e-12


def test_sample_depth_expansion_is_depth_major_and_orientation_aware() -> None:
    """Keep biological depth separate from its provisional hardware sign."""
    positions = [
        {"x": 1.0, "y": 2.0, "z": 100.0},
        {"x": 3.0, "y": 4.0, "z": 110.0},
    ]

    expanded = expand_stage_positions_for_depth(positions, [0.0, 20.0], "negative")

    assert [(position["x"], position["z"]) for position in expanded] == [
        (1.0, 100.0),
        (3.0, 110.0),
        (1.0, 80.0),
        (3.0, 90.0),
    ]
    assert [position["depth_index"] for position in expanded] == [0, 0, 1, 1]
    assert [position["sample_depth_um"] for position in expanded] == [
        0.0,
        0.0,
        20.0,
        20.0,
    ]


def test_positive_depth_and_oblique_correction_share_one_stage_orientation() -> None:
    """Move deeper toward +Z while compensating the opposite lab displacement."""
    corrected_surface = apply_oblique_scan_correction(
        [
            {
                "x": 100.0,
                "y": 200.0,
                "z": 50.0,
                "scan_reference_z_um": 50.0,
            }
        ],
        angle_deg=30.0,
        camera_zstage_orientation="positive",
    )

    expanded = expand_stage_positions_for_depth(
        corrected_surface,
        [0.0, 10.0],
        "positive",
        angle_deg=30.0,
    )

    assert expanded[1]["z"] == pytest.approx(60.0)
    assert expanded[1]["stage_depth_offset_um"] == pytest.approx(10.0)
    assert expanded[1]["lab_scan_um"] == pytest.approx(100.0)
    assert expanded[1]["x"] == pytest.approx(100.0 + 10.0 / np.tan(np.pi / 6))
