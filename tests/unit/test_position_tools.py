"""Unit tests for OPM camera and laboratory position geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from opm_v2.utils.position_tools import (
    ao_grid_cells,
    ao_grid_positions,
    apply_oblique_scan_correction,
    cam2lab,
    compose_stage_positions,
    covering_tile_origins,
    expand_stage_positions_for_depth,
    lab2cam,
    nearest_ao_grid_indices,
    oblique_camera_extents_um,
    sample_depth_levels_um,
    split_stage_scan_bounds,
    stage_coordinate_metadata,
    tile_footprints_um,
)


def test_ao_grid_cells_and_centers_cover_explicit_requested_bounds() -> None:
    """Cell edges divide the requested area, independent of sampled origins."""
    bounds = {"x": [100.0, 1000.0], "y": [-120.0, 480.0]}
    cells = ao_grid_cells(bounds, 3, 3)
    expected_cells = np.asarray(
        [
            [x, x + 300.0, y, y + 200.0]
            for y in (-120.0, 80.0, 280.0)
            for x in (100.0, 400.0, 700.0)
        ]
    )
    np.testing.assert_allclose(cells, expected_cells, rtol=0, atol=1e-12)
    targets = ao_grid_positions(
        [{"x": 100.0, "y": -120.0, "z": 50.0}],
        3,
        3,
        bounds_um=bounds,
    )
    assert [(p["x"], p["y"], p["z"]) for p in targets] == [
        (x, y, 50.0) for y in (-20.0, 180.0, 380.0) for x in (250.0, 550.0, 850.0)
    ]


def test_single_long_trajectory_supports_complete_three_by_three_grid() -> None:
    """One trajectory is spatial support, not a limit on AO sampling count."""
    positions = [{"x": 1200.0, "y": 400.0, "z": 80.0}]
    targets = ao_grid_positions(
        positions, 3, 3, bounds_um={"x": [1200.0, 2100.0], "y": [250.0, 550.0]}
    )
    assert len(targets) == 9
    assert [p["x"] for p in targets[:3]] == [1350.0, 1650.0, 1950.0]
    assert [p["y"] for p in targets[::3]] == [300.0, 400.0, 500.0]
    assert [p["z"] for p in targets] == [80.0] * 9


@pytest.mark.parametrize("orientation,stage_sign", [("positive", 1), ("negative", -1)])
def test_single_long_trajectory_uses_authoritative_focus_plane(orientation, stage_sign):
    """Unobserved along-scan tilt comes from the plan at all nine AO centers."""
    bounds = {"x": [100.0, 1000.0], "y": [-100.0, 500.0]}
    reference_z = 50.0
    for depth in (0.0, 11.76, 23.52, 35.28):
        origin_z = reference_z + stage_sign * depth
        positions = [
            {
                "x": 100.0 + stage_sign * (origin_z - reference_z) / np.tan(np.pi / 6),
                "lab_scan_um": 100.0,
                "y": 200.0,
                "z": origin_z,
                "scan_reference_z_um": reference_z,
                "focus_surface_slope_x": 0.01,
                "focus_surface_slope_y": -0.02,
                "sample_depth_um": depth,
                "stage_depth_offset_um": stage_sign * depth,
            }
        ]
        targets = ao_grid_positions(
            positions,
            3,
            3,
            bounds_um=bounds,
            angle_deg=30.0,
            camera_zstage_orientation=orientation,
        )
        assert len(targets) == 9
        for target, (lab_x, y) in zip(
            targets,
            [(x, y) for y in (0.0, 200.0, 400.0) for x in (250.0, 550.0, 850.0)],
            strict=True,
        ):
            expected_z = (
                reference_z
                + 0.01 * (lab_x - 100.0)
                - 0.02 * (y - 200.0)
                + stage_sign * depth
            )
            assert target["z"] == pytest.approx(expected_z, abs=0.0005)
            assert target["lab_scan_um"] == lab_x
            assert target["y"] == y
            assert target["x"] == pytest.approx(
                lab_x + stage_sign * (expected_z - reference_z) / np.tan(np.pi / 6),
                abs=0.0005,
            )
            assert target["sample_depth_um"] == depth
            assert target["stage_depth_offset_um"] == stage_sign * depth


def test_compose_stage_positions_rounds_after_millimeter_conversion() -> None:
    """Retain equivalent six-place mm / three-place um precision on all axes."""
    positions = compose_stage_positions(
        np.asarray([4.6228955078125, 5.265176128]) * 1000.0,
        np.asarray([-0.0913197021484375, 0.0925658162434896]) * 1000.0,
        np.asarray([2.40936056]) * 1000.0,
        z_offset_per_scan_um=0.0012345,
    )
    assert positions == [
        {"x": 4622.896, "y": -91.32, "z": 2409.361},
        {"x": 4622.896, "y": 92.566, "z": 2409.361},
        {"x": 5265.176, "y": -91.32, "z": 2409.362},
        {"x": 5265.176, "y": 92.566, "z": 2409.362},
    ]


def test_ao_grid_positions_follow_coverslip_surface_in_both_xy_axes() -> None:
    """Evaluate AO target Z from the complete physical acquisition surface."""
    positions = [
        {"x": x, "y": y, "z": 45.0 + 0.01 * x - 0.02 * y}
        for x in (100.0, 200.0)
        for y in (300.0, 500.0)
    ]
    targets = ao_grid_positions(
        positions, 3, 2, bounds_um={"x": [80.0, 260.0], "y": [250.0, 550.0]}
    )
    assert len(targets) == 6
    assert [p["x"] for p in targets[:3]] == [110.0, 170.0, 230.0]
    assert [p["y"] for p in targets[::3]] == [325.0, 475.0]
    for target in targets:
        assert target["z"] == pytest.approx(
            45.0 + 0.01 * target["x"] - 0.02 * target["y"], abs=0.0005
        )


@pytest.mark.parametrize("orientation,stage_sign", [("positive", 1), ("negative", -1)])
def test_ao_targets_use_one_depth_and_surface_compensation(orientation, stage_sign):
    """Fit in lab coordinates and transform target commands once at every depth."""
    bounds = {"x": [100.0, 1000.0], "y": [0.0, 600.0]}
    reference_z = 80.0
    for depth in (0.0, 11.76, 23.52, 35.28):
        positions = []
        for lab_x in (100.0, 700.0):
            for y in (0.0, 600.0):
                z = reference_z + 0.01 * (lab_x - 100.0) - 0.02 * y + stage_sign * depth
                positions.append(
                    {
                        "x": lab_x + stage_sign * (z - reference_z) / np.tan(np.pi / 6),
                        "lab_scan_um": lab_x,
                        "y": y,
                        "z": z,
                        "scan_reference_z_um": reference_z,
                    }
                )
        targets = ao_grid_positions(
            positions,
            3,
            3,
            bounds_um=bounds,
            angle_deg=30.0,
            camera_zstage_orientation=orientation,
        )
        for target, (lab_x, y) in zip(
            targets,
            [(x, y) for y in (100.0, 300.0, 500.0) for x in (250.0, 550.0, 850.0)],
            strict=True,
        ):
            z = reference_z + 0.01 * (lab_x - 100.0) - 0.02 * y + stage_sign * depth
            assert target["lab_scan_um"] == lab_x
            assert target["y"] == y
            assert target["z"] == pytest.approx(z, abs=0.0005)
            expected_physical_x = lab_x + stage_sign * (z - reference_z) / np.tan(
                np.pi / 6
            )
            assert target["x"] == pytest.approx(expected_physical_x, abs=0.0005)


def test_compensated_ao_grid_requires_declared_transform() -> None:
    with pytest.raises(ValueError, match=r"(?i)(angle|orientation|transform)"):
        ao_grid_positions(
            [
                {
                    "x": 120.0,
                    "lab_scan_um": 100.0,
                    "y": 0.0,
                    "z": 50.0,
                    "scan_reference_z_um": 40.0,
                }
            ],
            3,
            3,
            bounds_um={"x": [100.0, 1000.0], "y": [-50.0, 50.0]},
        )


@pytest.mark.parametrize("counts", [(0, 3), (3, 0), (-1, 3), (2.5, 3)])
def test_ao_grid_rejects_invalid_dimensions(counts) -> None:
    with pytest.raises(ValueError):
        ao_grid_cells({"x": [0.0, 10.0], "y": [0.0, 10.0]}, *counts)


def test_ao_grid_targets_preserve_three_decimal_micrometer_precision() -> None:
    targets = ao_grid_positions(
        [{"x": 100.0, "y": 200.0, "z": 50.12345}],
        3,
        1,
        bounds_um={"x": [100.0005, 101.0005], "y": [199.9991, 200.9991]},
    )
    assert [p["x"] for p in targets] == [100.167, 100.5, 100.834]
    assert [p["y"] for p in targets] == [200.499] * 3
    assert [p["z"] for p in targets] == [50.123] * 3


def test_largest_overlap_assigns_boundary_crossing_tiles_and_ties() -> None:
    """Origin proximity cannot override physical overlap; ties choose index zero."""
    bounds = {"x": [0.0, 300.0], "y": [0.0, 300.0]}
    cells = ao_grid_cells(bounds, 3, 3)
    positions = [{"x": x, "y": 50.0, "z": 0.0} for x in (90.0, 0.0, 50.0, 0.0)]
    targets = ao_grid_positions(positions, 3, 3, bounds_um=bounds)
    footprints = np.asarray(
        [
            [90.0, 220.0, 10.0, 90.0],  # overlap lengths 10,100,20 -> middle X
            [0.0, 200.0, 0.0, 100.0],  # equal X areas -> first index
            [50.0, 250.0, 50.0, 250.0],  # central cell's 100x100 area wins
            [-20.0, 320.0, -20.0, 320.0],  # all nine equal -> first index
        ]
    )
    assignments = nearest_ao_grid_indices(
        positions,
        targets,
        cell_bounds_um=cells,
        tile_bounds_um=footprints,
    )
    np.testing.assert_array_equal(assignments, [1, 0, 4, 0])
    assert len(set(assignments)) < len(targets)


def test_largest_overlap_is_unchanged_by_physical_z_and_scan_origin() -> None:
    """Declared footprints in the same lab frame determine the assignment."""
    cells = np.asarray([[0.0, 100.0, 0.0, 100.0], [100.0, 200.0, 0.0, 100.0]])
    ao_positions = [
        {"x": 50.0, "y": 50.0, "z": -1000.0},
        {"x": 150.0, "y": 50.0, "z": 1000.0},
    ]
    assignments = nearest_ao_grid_indices(
        [{"x": 1e6, "y": -1e6, "z": -100.0}],
        ao_positions,
        cell_bounds_um=cells,
        tile_bounds_um=[[80.0, 180.0, 20.0, 80.0]],
    )
    np.testing.assert_array_equal(assignments, [1])


def test_nonoverlapping_tile_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"(?i)overlap"):
        nearest_ao_grid_indices(
            [{"x": 20.0, "y": 20.0, "z": 0.0}],
            [{"x": 5.0, "y": 5.0, "z": 0.0}],
            cell_bounds_um=[[0.0, 10.0, 0.0, 10.0]],
            tile_bounds_um=[[20.0, 30.0, 20.0, 30.0]],
        )


@pytest.mark.parametrize(
    "stage_positions,ao_positions,message",
    [
        ([], [{"x": 0.0, "y": 0.0, "z": 0.0}], "acquisition"),
        ([{"x": 0.0, "y": 0.0, "z": 0.0}], [], "AO-grid"),
    ],
)
def test_nearest_ao_grid_indices_require_both_position_sets(
    stage_positions, ao_positions, message
):
    with pytest.raises(ValueError, match=message):
        nearest_ao_grid_indices(
            stage_positions,
            ao_positions,
            cell_bounds_um=[[0.0, 10.0, 0.0, 10.0]],
            tile_bounds_um=[[0.0, 10.0, 0.0, 10.0]],
        )


def test_tile_footprints_include_camera_support_for_forward_and_reverse_scans() -> None:
    positions = [
        {"x": 150.0, "lab_scan_um": 100.0, "y": 50.0, "z": 20.0},
        {"x": 450.0, "lab_scan_um": 400.0, "y": 50.0, "z": 20.0},
    ]
    footprints = tile_footprints_um(
        positions,
        scan_displacements_um=[300.0, -300.0],
        camera_scan_extent_um=20.0,
        tile_width_um=60.0,
    )
    np.testing.assert_allclose(
        footprints,
        [[90.0, 410.0, 20.0, 80.0], [90.0, 410.0, 20.0, 80.0]],
        rtol=0,
        atol=1e-12,
    )


def test_stage_metadata_separates_commanded_physical_and_normalized_scan() -> None:
    position = {
        "x": 482.123456,
        "lab_scan_um": 461.754539,
        "y": -15.123456,
        "z": 2421.12,
        "scan_reference_z_um": 2409.36,
    }
    metadata = stage_coordinate_metadata(
        position,
        angle_deg=30.0,
        camera_zstage_orientation="positive",
        scan_offset_um=-0.4,
    )
    assert metadata["physical_stage_command_um"] == {
        "x": 481.723,
        "y": -15.123,
        "z": 2421.12,
    }
    assert metadata["laboratory_scan_um"]["x"] == 461.355
    assert metadata["stage_readback_um"] is None
    assert metadata["coordinate_transform"]["units"] == "um"


@pytest.mark.parametrize("depth_index", [0, 1, 2, 3])
def test_balmer_requested_grid_and_tile_overlap_at_all_recorded_depths(depth_index):
    """Use recorded plans and calibration, independent of any fused AO overlay."""
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "balmer_ao_geometry.json").read_text()
    )
    bounds = fixture["requested_bounds_um"]
    expected = fixture["independent_expectations"]
    depth = fixture["depths"][depth_index]
    calibration = fixture["calibration"]
    positions = [
        {
            "x": physical_x,
            "lab_scan_um": lab_x,
            "y": y,
            "z": depth["physical_stage_z_um"],
            "scan_reference_z_um": fixture["surface"]["scan_reference_z_um"],
            "depth_index": depth_index,
            "sample_depth_um": depth["sample_depth_um"],
            "stage_depth_offset_um": depth["stage_depth_offset_um"],
        }
        for physical_x, lab_x in zip(
            depth["recorded_physical_stage_x_starts_um"],
            fixture["recorded_lab_scan_starts_um"],
            strict=True,
        )
        for y in fixture["recorded_tile_y_um"]
    ]
    cells = ao_grid_cells(bounds, 3, 3)
    np.testing.assert_allclose(
        cells[:3, :2],
        np.column_stack(
            (
                expected["requested_grid_x_edges_um"][:-1],
                expected["requested_grid_x_edges_um"][1:],
            )
        ),
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        cells[::3, 2:],
        np.column_stack(
            (
                expected["requested_grid_y_edges_um"][:-1],
                expected["requested_grid_y_edges_um"][1:],
            )
        ),
        rtol=0,
        atol=1e-9,
    )
    targets = ao_grid_positions(
        positions,
        3,
        3,
        bounds_um=bounds,
        angle_deg=calibration["angle_deg"],
        camera_zstage_orientation=calibration["camera_Zstage_orientation"],
    )
    assert len(targets) == depth["recorded_ao_measurement_count"] == 9
    assert [p["x"] for p in targets[:3]] == expected[
        "physical_stage_x_grid_centers_rounded_um_by_depth"
    ][depth_index]
    np.testing.assert_allclose(
        [p["lab_scan_um"] for p in targets[:3]],
        np.round(expected["requested_grid_lab_scan_centers_um"], 3),
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        [p["y"] for p in targets[::3]],
        np.round(expected["requested_grid_y_centers_um"], 3),
        rtol=0,
        atol=1e-9,
    )
    assert [p["z"] for p in targets] == [round(depth["physical_stage_z_um"], 3)] * 9
    displacements = [
        end - start
        for start, end in zip(
            fixture["recorded_lab_scan_starts_um"],
            fixture["reconstructed_commanded_lab_scan_ends_um"],
            strict=True,
        )
        for _ in fixture["recorded_tile_y_um"]
    ]
    footprints = tile_footprints_um(
        positions,
        scan_displacements_um=displacements,
        camera_scan_extent_um=expected["camera_scan_extent_um"],
        tile_width_um=expected["camera_tile_extent_um"],
    )
    support = [min(footprints[:, 0]), max(footprints[:, 1])]
    np.testing.assert_allclose(
        support,
        expected["reconstructed_commanded_support_lab_scan_um"],
        rtol=0,
        atol=1e-9,
    )
    assert support[0] < bounds["x"][0]
    assert support[1] > bounds["x"][1]
    assignments = nearest_ao_grid_indices(
        positions,
        targets,
        cell_bounds_um=cells,
        tile_bounds_um=footprints,
    )
    # Explicit physical-area calculation gives 3/2/2 rows per scan trajectory.
    # The middle scan-axis cells are measured but no whole tile selects them.
    np.testing.assert_array_equal(
        assignments, [0, 0, 0, 3, 3, 6, 6, 2, 2, 2, 5, 5, 8, 8]
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
    starts, ends, effective_overlap_um, camera_scan_extent_um = split_stage_scan_bounds(
        0.0,
        250.0,
        100.0,
        configured_overlap_um,
        camera_crop_y,
        pixel_size_um,
        angle_deg,
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
