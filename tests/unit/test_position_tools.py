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
    nearest_ao_grid_indices,
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


@pytest.mark.parametrize(
    ("z_slope_x", "z_slope_y"),
    (
        (0.0, 0.0),
        (0.01, 0.0),
        (0.0, -0.02),
        (0.03, -0.02),
    ),
    ids=("flat", "x-tilt", "y-tilt", "xy-tilt"),
)
def test_nearest_ao_grid_indices_partition_positions_in_xy(
    z_slope_x: float,
    z_slope_y: float,
) -> None:
    """Assign the same lateral AO regions regardless of surface Z geometry."""
    stage_positions = [
        {
            "x": x,
            "y": y,
            "z": 45.0 + z_slope_x * x + z_slope_y * y,
        }
        for x in (100.0, 150.0, 200.0)
        for y in (300.0, 400.0, 500.0)
    ]
    ao_positions = ao_grid_positions(
        stage_positions,
        num_scan_positions=3,
        num_tile_positions=3,
    )

    assignments = nearest_ao_grid_indices(stage_positions, ao_positions)

    np.testing.assert_array_equal(
        assignments,
        [0, 3, 6, 1, 4, 7, 2, 5, 8],
    )


@pytest.mark.parametrize(
    ("stage_positions", "ao_positions", "message"),
    (
        ([], [{"x": 0.0, "y": 0.0, "z": 0.0}], "acquisition"),
        ([{"x": 0.0, "y": 0.0, "z": 0.0}], [], "AO-grid"),
    ),
    ids=("no-acquisition-positions", "no-ao-positions"),
)
def test_nearest_ao_grid_indices_require_both_position_sets(
    stage_positions: list[dict[str, float]],
    ao_positions: list[dict[str, float]],
    message: str,
) -> None:
    """Reject undefined AO-region assignments before numerical work."""
    with pytest.raises(ValueError, match=message):
        nearest_ao_grid_indices(stage_positions, ao_positions)


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


@pytest.mark.parametrize(
    ("camera_crop_y", "pixel_size_um", "angle_deg", "configured_overlap_um"),
    (
        (386, 0.115, 30.0, 30.0),
        (512, 0.115, 30.0, 20.0),
        (256, 0.2, 45.0, 12.0),
    ),
    ids=("standard-roi", "large-roi", "forty-five-degrees"),
)
def test_stage_split_preserves_fully_sampled_scan_overlap(
    camera_crop_y: int,
    pixel_size_um: float,
    angle_deg: float,
    configured_overlap_um: float,
) -> None:
    """Leave the configured overlap after accounting for the deskew footprint."""
    starts, ends, effective_overlap_um, camera_scan_extent_um = (
        split_stage_scan_bounds(
            0.0,
            250.0,
            100.0,
            configured_overlap_um,
            camera_crop_y,
            pixel_size_um,
            angle_deg,
        )
    )

    assert starts[0] == pytest.approx(0.0)
    assert ends[-1] == pytest.approx(250.0)
    assert camera_scan_extent_um == pytest.approx(
        camera_crop_y * pixel_size_um * np.cos(np.deg2rad(angle_deg))
    )
    assert effective_overlap_um == pytest.approx(
        camera_scan_extent_um + configured_overlap_um
    )
    assert np.diff(starts) == pytest.approx(np.diff(ends))
    assert len(starts) == int(np.ceil(250.0 / 100.0))
    raw_overlaps_um = ends[:-1] - starts[1:]
    assert raw_overlaps_um == pytest.approx(
        np.full(len(starts) - 1, effective_overlap_um)
    )
    assert raw_overlaps_um - camera_scan_extent_um == pytest.approx(
        np.full(len(starts) - 1, configured_overlap_um)
    )


@pytest.mark.parametrize(
    ("start_um", "end_um", "footprint_um", "overlap_fraction", "count"),
    (
        (0.0, 50.0, 512 * 0.115 * 0.5, 0.15, 2),
        (0.0, 40.0, 22.195, 0.2, 3),
        (10.0, 20.0, 22.195, 0.2, 1),
        (5.0, 5.0, 22.195, 0.2, 1),
    ),
    ids=("fifty-micron-volume", "multiple-slabs", "one-slab", "single-level"),
)
def test_sample_depth_levels_cover_requested_bounds(
    start_um: float,
    end_um: float,
    footprint_um: float,
    overlap_fraction: float,
    count: int,
) -> None:
    """Cover the bounded thickness without adding a slab at the deep edge."""
    levels = sample_depth_levels_um(
        start_um,
        end_um,
        footprint_um,
        overlap_fraction,
    )

    assert len(levels) == count
    assert levels[0] == pytest.approx(start_um)
    if end_um - start_um > footprint_um:
        assert levels[-1] + footprint_um == pytest.approx(end_um)
        assert float(np.max(np.diff(levels))) <= (
            footprint_um * (1.0 - overlap_fraction) + 1e-12
        )


def test_sample_depth_bounds_must_increase_into_sample() -> None:
    """Reject reversed biological-depth bounds instead of silently overscanning."""
    with pytest.raises(ValueError, match="end must be greater"):
        sample_depth_levels_um(50.0, 0.0, 22.195, 0.2)


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
