"""Integrate coverslip-relative depth slabs with OPM mode planners."""

from __future__ import annotations

import numpy as np
import pytest
from useq import AbsolutePosition, CustomAction, GridFromEdges, MDASequence

from opm_v2.engine.opm_custom_events import (
    ACTION_AO_GRID,
    ACTION_AO_MIRROR_UPDATE,
    ACTION_AO_OPTIMIZE,
    ACTION_ASI_SETUP_SCAN,
)
from opm_v2.engine.setup_events import (
    setup_mirrorscan,
    setup_projection,
    setup_stagescan,
)
from opm_v2.utils.coverslip import COVERSLIP_METADATA_KEY, CoverslipPlane


@pytest.mark.parametrize(
    ("mode", "setup", "axis_order"),
    [
        ("projection", setup_projection, "pc"),
        ("mirror", setup_mirrorscan, "pzc"),
        ("stage", setup_stagescan, "pzc"),
    ],
)
@pytest.mark.parametrize("ao_mode", ["none", "at xyz positions"])
def test_opm_modes_repeat_coverslip_corrected_xy_at_sample_depths(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
    mode,
    setup,
    axis_order,
    ao_mode,
) -> None:
    """Apply every depth as an offset from the fitted local coverslip Z."""
    config = opm_config_factory(
        mode=mode,
        active_channels=(0,),
        channel_powers=(10.0,),
        channel_exposures_ms=(10.0,),
        scan_range_um=20.0,
        scan_axis_step_um=2.0,
        updates={
            "acq_config": {
                "AO": {"ao_mode": ao_mode},
                "Positions": {
                    "sample_depth_start_um": 0.0,
                    # This bound is slightly thicker than one simulated
                    # camera footprint in every parameterized OPM mode.
                    "sample_depth_end_um": 17.0,
                },
            }
        },
    )
    plane = CoverslipPlane(100.0, 200.0, 50.0, 0.01, -0.02)
    region = AbsolutePosition(
        z=50.0,
        sequence=MDASequence(
            metadata={COVERSLIP_METADATA_KEY: plane.to_metadata()},
            grid_plan=GridFromEdges(
                left=100.0,
                right=110.0,
                top=200.0,
                bottom=200.0,
                fov_width=50.0,
                fov_height=50.0,
            ),
        ),
    )

    events, handler = setup(
        demo_core,
        config,
        MDASequence(stage_positions=[region], axis_order=axis_order),
        workspace_tmp_path / f"{mode}-depth.ome.zarr",
    )
    image_events, _custom_actions = split_events(events)
    first_image_by_position = {}
    for event in image_events:
        first_image_by_position.setdefault(event.index["p"], event)

    position_count = handler.index_sizes["p"]
    assert position_count % 2 == 0
    base_count = position_count // 2
    if ao_mode == "at xyz positions":
        ao_payloads = [
            event.action.data["AO"]
            for event in events
            if getattr(event.action, "name", None) == ACTION_AO_OPTIMIZE
            and not event.action.data["AO"]["apply_existing"]
        ]
        assert [(p["z_idx"], p["logical_position_idx"]) for p in ao_payloads] == [
            (z_idx, lateral_idx)
            for z_idx in range(2)
            for lateral_idx in range(base_count)
        ]
    if mode == "stage":
        stage_scan_progress = [
            event.action.data["StageScan"]
            for event in events
            if getattr(event.action, "name", None) == ACTION_ASI_SETUP_SCAN
        ]
        assert [item["position_index"] for item in stage_scan_progress] == list(
            range(position_count)
        )
        assert {item["position_count"] for item in stage_scan_progress} == {
            position_count
        }
        assert [item["z_level_index"] for item in stage_scan_progress] == [
            0
        ] * base_count + [1] * base_count
        assert {item["z_level_count"] for item in stage_scan_progress} == {2}

    for base_index in range(base_count):
        surface_event = first_image_by_position[base_index]
        depth_event = first_image_by_position[base_index + base_count]
        assert depth_event.metadata["Stage"]["x_pos"] == pytest.approx(
            surface_event.metadata["Stage"]["x_pos"]
        )
        assert depth_event.metadata["Stage"]["y_pos"] == pytest.approx(
            surface_event.metadata["Stage"]["y_pos"]
        )
        depth_origin_um = depth_event.metadata["Stage"]["sample_depth_um"]
        assert 0.0 < depth_origin_um < 17.0
        # On this microscope, increasing physical Z moves into the sample.
        assert depth_event.metadata["Stage"]["z_pos"] == pytest.approx(
            surface_event.metadata["Stage"]["z_pos"] + depth_origin_um
        )
        assert surface_event.metadata["Stage"]["sample_depth_um"] == 0.0
        assert depth_event.metadata["Stage"]["stage_depth_offset_um"] == pytest.approx(
            depth_origin_um
        )


@pytest.mark.parametrize(
    ("mode", "setup", "axis_order"),
    [
        ("projection", setup_projection, "pc"),
        ("mirror", setup_mirrorscan, "pzc"),
        ("stage", setup_stagescan, "pzc"),
    ],
)
@pytest.mark.parametrize("ao_mode", ["grid at start", "grid at timepoints"])
def test_grid_ao_runs_before_tiles_at_each_sample_depth(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    mode,
    setup,
    axis_order,
    ao_mode,
) -> None:
    """Run an identical AO XY grid immediately before each depth's tiles."""
    config = opm_config_factory(
        mode=mode,
        active_channels=(0,),
        channel_powers=(10.0,),
        channel_exposures_ms=(10.0,),
        scan_range_um=20.0,
        scan_axis_step_um=2.0,
        updates={
            "acq_config": {
                "AO": {"ao_mode": ao_mode},
                "Positions": {
                    "sample_depth_start_um": 0.0,
                    # This bound is slightly thicker than one simulated
                    # camera footprint in every parameterized OPM mode.
                    "sample_depth_end_um": 17.0,
                },
            }
        },
    )
    plane = CoverslipPlane(100.0, 200.0, 50.0, 0.01, -0.02)
    region = AbsolutePosition(
        z=50.0,
        sequence=MDASequence(
            metadata={COVERSLIP_METADATA_KEY: plane.to_metadata()},
            grid_plan=GridFromEdges(
                left=100.0,
                right=110.0,
                top=200.0,
                bottom=200.0,
                fov_width=50.0,
                fov_height=50.0,
            ),
        ),
    )

    events, handler = setup(
        demo_core,
        config,
        MDASequence(stage_positions=[region], axis_order=axis_order),
        workspace_tmp_path / f"{mode}-depth-grid-ao.ome.zarr",
    )
    grid_events = [
        (event_idx, event)
        for event_idx, event in enumerate(events)
        if isinstance(event.action, CustomAction)
        and event.action.name == ACTION_AO_GRID
        and not event.action.data["AO"]["apply_ao_map"]
    ]

    assert len(grid_events) == 2
    first_indices = grid_events[0][1].action.data["AO"]["position_indices"]
    second_indices = grid_events[1][1].action.data["AO"]["position_indices"]
    assert first_indices == list(range(len(first_indices)))
    assert second_indices == list(
        range(len(first_indices), len(first_indices) + len(second_indices))
    )

    first_positions = grid_events[0][1].action.data["AO"]["stage_positions"]
    second_positions = grid_events[1][1].action.data["AO"]["stage_positions"]
    assert [
        (position.get("lab_scan_um", position["x"]), position["y"])
        for position in first_positions
    ] == [
        (position.get("lab_scan_um", position["x"]), position["y"])
        for position in second_positions
    ]
    depth_step_um = second_positions[0]["z"] - first_positions[0]["z"]
    assert 0.0 < depth_step_um < 17.0
    assert [position["z"] for position in second_positions] == pytest.approx(
        [position["z"] + depth_step_um for position in first_positions]
    )

    image_event_indices_by_position: dict[int, list[int]] = {}
    for event_idx, event in enumerate(events):
        if not isinstance(event.action, CustomAction) and "p" in event.index:
            image_event_indices_by_position.setdefault(event.index["p"], []).append(
                event_idx
            )
    apply_events = [
        (event_idx, event)
        for event_idx, event in enumerate(events)
        if isinstance(event.action, CustomAction)
        and event.action.name == ACTION_AO_GRID
        and event.action.data["AO"]["apply_ao_map"]
    ]
    assert [event.action.data["AO"]["pos_idx"] for _, event in apply_events] == list(
        range(handler.index_sizes["p"])
    )
    for apply_event_idx, apply_event in apply_events:
        pos_idx = apply_event.action.data["AO"]["pos_idx"]
        assert apply_event_idx < image_event_indices_by_position[pos_idx][0]
        first_frame, last_frame = (
            image_event_indices_by_position[pos_idx][0],
            image_event_indices_by_position[pos_idx][-1],
        )
        assert not any(
            getattr(event.action, "name", None)
            in {ACTION_AO_GRID, ACTION_AO_OPTIMIZE, ACTION_AO_MIRROR_UPDATE}
            for event in events[first_frame : last_frame + 1]
        )
        payload = apply_event.action.data["AO"]
        assert (
            payload["grid_bounds_um"]
            == grid_events[0][1].action.data["AO"]["grid_bounds_um"]
        )

    first_grid_idx, second_grid_idx = grid_events[0][0], grid_events[1][0]
    assert first_grid_idx < min(
        event_idx
        for event_idx, event in apply_events
        if event.action.data["AO"]["pos_idx"] in first_indices
    )
    assert first_grid_idx < min(
        image_event_indices_by_position[index][0] for index in first_indices
    )
    assert (
        max(image_event_indices_by_position[index][-1] for index in first_indices)
        < second_grid_idx
    )
    assert second_grid_idx < min(
        event_idx
        for event_idx, event in apply_events
        if event.action.data["AO"]["pos_idx"] in second_indices
    )
    assert second_grid_idx < min(
        image_event_indices_by_position[index][0] for index in second_indices
    )
    assert handler.index_sizes["p"] == len(first_indices) + len(second_indices)


def test_single_long_trajectory_retains_full_three_by_three_grid_at_four_depths(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
) -> None:
    """Plan nine cell measurements per slab despite only one scan origin."""
    config = opm_config_factory(
        mode="stage",
        active_channels=(0,),
        channel_powers=(10.0,),
        channel_exposures_ms=(10.0,),
        scan_axis_step_um=50.0,
        updates={
            "acq_config": {
                "AO": {
                    "ao_mode": "grid at start",
                    "num_scan_positions": 3,
                    "num_tile_positions": 3,
                }
            }
        },
    )
    axial_extent = (
        32 * demo_core.getPixelSizeUm() * np.sin(np.deg2rad(config["OPM"]["angle_deg"]))
    )
    config["acq_config"]["Positions"]["sample_depth_end_um"] = float(axial_extent * 3.4)
    bounds = {"bottom": 100.123, "top": 1000.456, "left": 200.0, "right": 240.0}
    events, handler = setup_stagescan(
        demo_core,
        config,
        MDASequence(grid_plan=bounds),
        workspace_tmp_path / "long-trajectory-depth-grid.ome.zarr",
    )
    grids = [
        event.action.data["AO"]
        for event in events
        if getattr(event.action, "name", None) == ACTION_AO_GRID
        and not event.action.data["AO"]["apply_ao_map"]
    ]
    assert len(grids) == 4
    assert [item["z_idx"] for item in grids] == [0, 1, 2, 3]
    expected_bounds = {"x": [100.123, 1000.456], "y": [200.0, 240.0]}
    for payload in grids:
        assert payload["grid_bounds_um"] == expected_bounds
        assert payload["num_scan_positions"] == payload["num_tile_positions"] == 3
        assert len({p["lab_scan_um"] for p in payload["stage_positions"]}) == 1
        assert len(payload["tile_bounds_um"]) == len(payload["position_indices"])
        np.testing.assert_allclose(
            payload["tile_bounds_um"], grids[0]["tile_bounds_um"]
        )
    summary = handler.acquisition_metadata["resolved_spatial_plan"]
    assert summary["resolved_ao_grid_positions"]["count"] == 36
    assert summary["ao_spatial_geometry"]["grid_bounds_um"] == expected_bounds
    assert (
        summary["ao_spatial_geometry"]["commanded_acquired_support_bounds_um"]["x"][1]
        > 1000.456
    )
    assert (
        summary["ao_spatial_geometry"]["planned_retained_sampled_support_bounds_um"][
            "x"
        ][1]
        < summary["ao_spatial_geometry"]["commanded_acquired_support_bounds_um"]["x"][1]
    )
    first_frames = [
        event for event in events if event.index.get("z") == 0 and "p" in event.index
    ]
    for frame in first_frames:
        stage = frame.metadata["Stage"]
        assert stage["stage_readback_um"] is None
        assert stage["laboratory_scan_um"]["x"] == pytest.approx(100.123)
        assert stage["physical_stage_command_um"]["x"] == pytest.approx(
            stage["laboratory_scan_um"]["x"]
            + stage["stage_depth_offset_um"]
            / np.tan(np.deg2rad(config["OPM"]["angle_deg"])),
            abs=0.001,
        )
