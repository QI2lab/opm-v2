"""Integrate coverslip-relative depth slabs with OPM mode planners."""

from __future__ import annotations

import pytest
from useq import AbsolutePosition, CustomAction, GridFromEdges, MDASequence

from opm_v2.engine.opm_custom_events import ACTION_AO_GRID
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
def test_opm_modes_repeat_coverslip_corrected_xy_at_sample_depths(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
    mode,
    setup,
    axis_order,
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
                "Positions": {
                    "sample_depth_start_um": 0.0,
                    "sample_depth_end_um": 1.0,
                }
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
    for base_index in range(base_count):
        surface_event = first_image_by_position[base_index]
        depth_event = first_image_by_position[base_index + base_count]
        assert depth_event.metadata["Stage"]["x_pos"] == pytest.approx(
            surface_event.metadata["Stage"]["x_pos"]
        )
        assert depth_event.y_pos == pytest.approx(surface_event.y_pos)
        # On this microscope, increasing physical Z moves into the sample.
        assert depth_event.z_pos == pytest.approx(surface_event.z_pos + 1.0)
        assert surface_event.metadata["Stage"]["sample_depth_um"] == 0.0
        assert depth_event.metadata["Stage"]["sample_depth_um"] == 1.0
        assert depth_event.metadata["Stage"]["stage_depth_offset_um"] == 1.0


@pytest.mark.parametrize(
    ("mode", "setup", "axis_order"),
    [
        ("projection", setup_projection, "pc"),
        ("mirror", setup_mirrorscan, "pzc"),
        ("stage", setup_stagescan, "pzc"),
    ],
)
def test_grid_ao_runs_before_tiles_at_each_sample_depth(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    mode,
    setup,
    axis_order,
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
                "AO": {"ao_mode": "grid at start"},
                "Positions": {
                    "sample_depth_start_um": 0.0,
                    "sample_depth_end_um": 1.0,
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
    assert [position["z"] for position in second_positions] == pytest.approx(
        [position["z"] + 1.0 for position in first_positions]
    )

    image_event_indices_by_position: dict[int, list[int]] = {}
    for event_idx, event in enumerate(events):
        if not isinstance(event.action, CustomAction) and "p" in event.index:
            image_event_indices_by_position.setdefault(event.index["p"], []).append(
                event_idx
            )
    first_grid_idx, second_grid_idx = grid_events[0][0], grid_events[1][0]
    assert first_grid_idx < min(
        image_event_indices_by_position[index][0] for index in first_indices
    )
    assert max(
        image_event_indices_by_position[index][-1] for index in first_indices
    ) < second_grid_idx
    assert second_grid_idx < min(
        image_event_indices_by_position[index][0] for index in second_indices
    )
    assert handler.index_sizes["p"] == len(first_indices) + len(second_indices)
