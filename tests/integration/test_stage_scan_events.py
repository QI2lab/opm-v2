"""Integrate stage-scan event planning with MMCore demo hardware."""

from __future__ import annotations

import pytest
from useq import AbsolutePosition, GridFromEdges, MDASequence

from opm_v2.engine.opm_custom_events import (
    ACTION_ASI_SETUP_SCAN,
    ACTION_DAQ,
    ACTION_STAGE_MOVE,
)
from opm_v2.engine.setup_events import setup_stagescan
from opm_v2.utils.coverslip import COVERSLIP_METADATA_KEY, CoverslipPlane


@pytest.mark.parametrize(
    "axis_order",
    [
        "tpgzc",
        "tpgcz",
    ],
)
def test_stage_events_follow_hardware_plane_channel_order(
    demo_core,
    workspace_tmp_path,
    axis_order,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
    assert_standard_image_fields,
) -> None:
    """Keep hardware frame order independent of the native MDA axis order.

    Parameters
    ----------
    demo_core : CMMCorePlus
        Core loaded with Micro-Manager demo devices.
    workspace_tmp_path : Path
        Directory used to construct the output handler.
    axis_order : str
        Axis order selected in the standard MDA widget.
    opm_config_factory : OpmConfigFactory
        Reusable simulated configuration factory.
    simulated_acquisition_hardware : SimulatedAcquisitionHardware
        Initialized singleton hardware simulations.
    split_events : Callable
        Event classifier fixture.
    assert_standard_image_fields : Callable
        Reusable standard useq field assertion.
    """
    config = opm_config_factory(mode="stage")
    events, handler = setup_stagescan(
        demo_core,
        config,
        MDASequence(
            grid_plan={"top": 10.0, "left": 0.0, "bottom": 0.0, "right": 0.0},
            axis_order=axis_order,
        ),
        workspace_tmp_path / "stage.ome.zarr",
    )
    image_events, custom_actions = split_events(events)

    assert custom_actions.count(ACTION_DAQ) == 1
    assert custom_actions.count(ACTION_ASI_SETUP_SCAN) == 1
    assert [dict(event.index) for event in image_events] == [
        {"t": 0, "p": 0, "c": channel, "z": plane}
        for plane in range(5)
        for channel in range(2)
    ]
    assert handler.index_sizes == {"t": 1, "p": 1, "c": 2, "z": 5}
    assert handler.acquisition_order == ("t", "p", "z", "c")
    assert_standard_image_fields(image_events)


def test_stage_scan_preserves_micrometre_grid_range_in_asi_millimetres(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
) -> None:
    """Keep a 1 um grid span distinct after conversion to ASI millimetres."""
    config = opm_config_factory(
        mode="stage",
        active_channels=(0, 1, 2),
        channel_powers=(10.0, 20.0, 30.0),
        channel_exposures_ms=(20.0, 20.0, 20.0),
        scan_axis_step_um=0.4,
        updates={
            "acq_config": {
                "stage_scan": {
                    "excess_start_frames": 50,
                    "excess_end_frames": 0,
                }
            }
        },
    )
    events, handler = setup_stagescan(
        demo_core,
        config,
        MDASequence(
            grid_plan={
                "left": 1265.889,
                "top": 3167.460,
                "right": 1265.989,
                "bottom": 3168.460,
            }
        ),
        workspace_tmp_path / "stage-precision.ome.zarr",
    )

    asi_event = next(
        event
        for event in events
        if getattr(event.action, "name", None) == ACTION_ASI_SETUP_SCAN
    )
    assert asi_event.action.data["ASI"]["scan_axis_start_mm"] == pytest.approx(3.167460)
    assert asi_event.action.data["ASI"]["scan_axis_end_mm"] == pytest.approx(3.168460)
    assert handler.index_sizes == {"t": 1, "p": 2, "c": 3, "z": 52}


def test_stage_scan_uses_stage_explorer_left_right_as_physical_fast_x(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
) -> None:
    """Adapt an exported ROI without changing the established stage planner."""
    config = opm_config_factory(
        mode="stage",
        active_channels=(0,),
        channel_powers=(10.0,),
        channel_exposures_ms=(10.0,),
        scan_axis_step_um=2.0,
    )
    region = AbsolutePosition(
        z=7.0,
        name="stage_roi",
        sequence=MDASequence(
            grid_plan=GridFromEdges(
                left=100.0,
                right=110.0,
                top=200.0,
                bottom=200.0,
                fov_width=50.0,
                fov_height=50.0,
            )
        ),
    )

    events, handler = setup_stagescan(
        demo_core,
        config,
        MDASequence(stage_positions=[region], axis_order="pzc"),
        workspace_tmp_path / "stage-explorer.ome.zarr",
    )
    asi_event = next(
        event
        for event in events
        if getattr(event.action, "name", None) == ACTION_ASI_SETUP_SCAN
    )

    assert asi_event.action.data["ASI"]["scan_axis_start_mm"] == pytest.approx(0.1)
    assert asi_event.action.data["ASI"]["scan_axis_end_mm"] == pytest.approx(0.11)
    assert handler.index_sizes == {"t": 1, "p": 1, "c": 1, "z": 5}


def test_stage_scan_applies_exported_plane_at_each_segment_and_tile(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
) -> None:
    """Evaluate both plane slopes at stage-scan software move boundaries."""
    config = opm_config_factory(
        mode="stage",
        active_channels=(0,),
        channel_powers=(10.0,),
        channel_exposures_ms=(10.0,),
        scan_axis_step_um=2.0,
        updates={
            "acq_config": {"stage_scan": {"max_stage_scan_range_um": 100.0}}
        },
    )
    plane = CoverslipPlane(100.0, 200.0, 7.0, 0.01, -0.02)
    region = AbsolutePosition(
        z=7.0,
        sequence=MDASequence(
            metadata={COVERSLIP_METADATA_KEY: plane.to_metadata()},
            grid_plan=GridFromEdges(
                left=100.0,
                right=350.0,
                top=200.0,
                bottom=300.0,
                fov_width=50.0,
                fov_height=50.0,
            ),
        ),
    )

    events, _handler = setup_stagescan(
        demo_core,
        config,
        MDASequence(stage_positions=[region], axis_order="pzc"),
        workspace_tmp_path / "stage-coverslip.ome.zarr",
    )
    moves = [
        event.action.data["Stage"]
        for event in events
        if getattr(event.action, "name", None) == ACTION_STAGE_MOVE
    ]
    unique_moves = {
        (move["x_pos"], move["y_pos"], move["z_pos"]) for move in moves
    }

    assert len({move[0] for move in unique_moves}) > 1
    assert len({move[1] for move in unique_moves}) > 1
    logical_positions = {}
    for event in events:
        if "p" in event.index and event.index.get("z") == 0:
            stage = event.metadata["Stage"]
            logical_positions.setdefault(
                event.index["p"],
                (stage["x_pos"], stage["y_pos"], stage["z_pos"]),
            )
    for x_um, y_um, z_um in logical_positions.values():
        assert z_um == pytest.approx(plane.z_at(x_um, y_um), abs=0.005)


def test_stage_scan_rejects_literal_exported_positions(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
) -> None:
    """Require a region because a point cannot define a hardware scan span."""
    config = opm_config_factory(mode="stage")

    with pytest.raises(ValueError, match="requires ROI regions"):
        setup_stagescan(
            demo_core,
            config,
            MDASequence(stage_positions=[(100.0, 200.0, 7.0)]),
            workspace_tmp_path / "stage-point.ome.zarr",
        )


def test_stage_scan_subdivision_matches_main_overlap_rule(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
) -> None:
    """Use camera axial footprint plus the configured extra overlap."""
    requested_overlap_um = 20.0
    config = opm_config_factory(
        mode="stage",
        active_channels=(0,),
        channel_powers=(10.0,),
        channel_exposures_ms=(10.0,),
        scan_axis_step_um=2.0,
        updates={
            "acq_config": {
                "Positions": {"scan_axis_overlap_um": requested_overlap_um},
                "stage_scan": {"max_stage_scan_range_um": 100.0},
            }
        },
    )
    events, handler = setup_stagescan(
        demo_core,
        config,
        MDASequence(grid_plan={"top": 250.0, "bottom": 0.0, "left": 0.0, "right": 0.0}),
        workspace_tmp_path / "stage-deskew-overlap.ome.zarr",
    )
    asi_events = [
        event
        for event in events
        if getattr(event.action, "name", None) == ACTION_ASI_SETUP_SCAN
    ]
    starts_um = [
        event.action.data["ASI"]["scan_axis_start_mm"] * 1000 for event in asi_events
    ]
    ends_um = [
        event.action.data["ASI"]["scan_axis_end_mm"] * 1000 for event in asi_events
    ]
    assert len(asi_events) == 3
    assert starts_um[0] == pytest.approx(0.0)
    assert ends_um[-1] == pytest.approx(250.0)
    camera_axial_footprint_um = (
        config["acq_config"]["camera_roi"]["crop_y"]
        * demo_core.getPixelSizeUm()
        * 0.5
    )
    effective_overlap_um = requested_overlap_um + camera_axial_footprint_um
    assert [
        end - next_start
        for end, next_start in zip(ends_um[:-1], starts_um[1:])
    ] == pytest.approx(
        [effective_overlap_um, effective_overlap_um],
        abs=0.002,
    )
    assert handler.index_sizes["p"] == 3
