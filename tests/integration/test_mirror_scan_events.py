"""Integrate mirror and timelapse planning with MMCore demo hardware."""

from __future__ import annotations

import numpy as np
import pytest
from useq import AbsolutePosition, GridFromEdges, MDASequence

from opm_v2.engine.opm_custom_events import ACTION_ASI_SETUP_SCAN, ACTION_STAGE_MOVE
from opm_v2.engine.setup_events import setup_mirrorscan, setup_timelapse
from opm_v2.utils.coverslip import COVERSLIP_METADATA_KEY, CoverslipPlane


@pytest.mark.parametrize(
    ("exposures", "axis_order", "camera_indices", "acquisition_order"),
    [
        (
            [10.0, 10.0],
            "tpzc",
            [(plane, channel) for plane in range(2) for channel in range(2)],
            ("t", "p", "z", "c"),
        ),
        (
            [10.0, 20.0],
            "tpcz",
            [(plane, channel) for channel in range(2) for plane in range(2)],
            ("t", "p", "c", "z"),
        ),
    ],
)
def test_mirror_camera_order_follows_interleaving(
    demo_core,
    workspace_tmp_path,
    exposures,
    axis_order,
    camera_indices,
    acquisition_order,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
    assert_standard_image_fields,
) -> None:
    """Build mirror events in each supported channel/Z ordering.

    Parameters
    ----------
    demo_core : CMMCorePlus
        Core loaded with Micro-Manager demo devices.
    workspace_tmp_path : Path
        Directory used to construct the output handler.
    exposures : list[float]
        Active-channel exposures controlling interleaving.
    axis_order : str
        Axis order selected in the standard MDA widget.
    camera_indices : list[tuple[int, int]]
        Expected plane/channel arrival order.
    acquisition_order : tuple[str, ...]
        Expected datastore order.
    opm_config_factory : OpmConfigFactory
        Reusable simulated configuration factory.
    simulated_acquisition_hardware : SimulatedAcquisitionHardware
        Initialized singleton hardware simulations.
    split_events : Callable
        Event classifier fixture.
    assert_standard_image_fields : Callable
        Reusable standard useq field assertion.
    """
    config = opm_config_factory(mode="mirror", channel_exposures_ms=exposures)
    events, handler = setup_mirrorscan(
        demo_core,
        config,
        MDASequence(stage_positions=[(0.0, 0.0, 0.0)], axis_order=axis_order),
        workspace_tmp_path / "mirror.ome.zarr",
    )
    image_events, custom_actions = split_events(events)
    assert [dict(event.index) for event in image_events] == [
        {"t": 0, "p": 0, "c": channel, "z": plane} for plane, channel in camera_indices
    ]
    assert ACTION_ASI_SETUP_SCAN not in custom_actions
    assert_standard_image_fields(image_events)
    assert handler.acquisition_order == acquisition_order


@pytest.mark.parametrize(("scan_steps", "expected_time_chunk"), [(5, 16), (10, 13)])
def test_fast_single_channel_mirror_series_uses_bounded_temporal_chunks(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    scan_steps,
    expected_time_chunk,
) -> None:
    """Batch long, shallow mirror stacks without exceeding the memory budget."""
    config = opm_config_factory(
        mode="mirror",
        active_channels=(1,),
        channel_powers=(10.0,),
        channel_exposures_ms=(1.5,),
        scan_range_um=float(scan_steps),
        scan_axis_step_um=1.0,
    )
    _events, handler = setup_mirrorscan(
        demo_core,
        config,
        MDASequence(
            stage_positions=[(0.0, 0.0, 0.0)],
            time_plan={"interval": 0, "loops": 101},
            axis_order="tpcz",
        ),
        workspace_tmp_path / "fast-mirror.ome.zarr",
    )

    assert handler.max_time_chunk_size == 16
    assert handler.time_chunk_concurrency == scan_steps
    frame = np.empty((128, 1900), dtype=np.uint16)
    assert handler._time_chunk_size(frame) == expected_time_chunk


def test_mirror_retiles_stage_explorer_region_in_physical_stage_axes(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
) -> None:
    """Apply a mirror stack at OPM-derived positions inside an exported ROI."""
    config = opm_config_factory(
        mode="mirror",
        active_channels=(0,),
        channel_powers=(10.0,),
        channel_exposures_ms=(10.0,),
        scan_range_um=4.0,
        scan_axis_step_um=2.0,
    )
    region = AbsolutePosition(
        z=9.0,
        name="mirror_roi",
        sequence=MDASequence(
            grid_plan=GridFromEdges(
                left=100.0,
                right=108.0,
                top=200.0,
                bottom=204.0,
                fov_width=50.0,
                fov_height=50.0,
            )
        ),
    )

    events, handler = setup_mirrorscan(
        demo_core,
        config,
        MDASequence(stage_positions=[region], axis_order="pzc"),
        workspace_tmp_path / "mirror-explorer.ome.zarr",
    )
    image_events, _ = split_events(events)

    x_positions = sorted({event.metadata["Stage"]["x_pos"] for event in image_events})
    scan_footprint_um = config["acq_config"]["DAQ"]["scan_range_um"]
    overlap_fraction = config["acq_config"]["Positions"]["scan_axis_overlap"]
    maximum_stride_um = scan_footprint_um * (1.0 - overlap_fraction)
    strides_um = np.diff(x_positions)
    assert x_positions[0] == pytest.approx(100.0)
    assert x_positions[-1] <= 108.0
    assert x_positions[-1] + scan_footprint_um >= 108.0
    assert np.all(strides_um > 0)
    assert np.max(strides_um) <= maximum_stride_um + 0.01
    assert {event.metadata["Stage"]["y_pos"] for event in image_events} == {200.0}
    assert {event.metadata["Stage"]["z_pos"] for event in image_events} == {9.0}
    assert handler.index_sizes == {"t": 1, "p": 3, "c": 1, "z": 2}


def test_mirror_applies_exported_coverslip_plane(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
) -> None:
    """Predict sample Z at every physical mirror-stack tile coordinate."""
    config = opm_config_factory(
        mode="mirror",
        active_channels=(0,),
        channel_powers=(10.0,),
        channel_exposures_ms=(10.0,),
        scan_range_um=4.0,
        scan_axis_step_um=2.0,
    )
    plane = CoverslipPlane(100.0, 200.0, 9.0, 0.1, 0.2)
    region = AbsolutePosition(
        z=9.0,
        sequence=MDASequence(
            metadata={COVERSLIP_METADATA_KEY: plane.to_metadata()},
            grid_plan=GridFromEdges(
                left=100.0,
                right=108.0,
                top=200.0,
                bottom=204.0,
                fov_width=50.0,
                fov_height=50.0,
            ),
        ),
    )

    events, _handler = setup_mirrorscan(
        demo_core,
        config,
        MDASequence(stage_positions=[region], axis_order="pzc"),
        workspace_tmp_path / "mirror-coverslip.ome.zarr",
    )
    image_events, _custom_actions = split_events(events)
    position_z = {}
    for event in image_events:
        stage = event.metadata["Stage"]
        position_z.setdefault(stage["x_pos"], stage["z_pos"])

    assert position_z == pytest.approx({100.0: 9.0, 102.67: 9.27, 105.34: 9.53})
    stage_events = [
        event
        for event in events
        if getattr(event.action, "name", None) == ACTION_STAGE_MOVE
    ]
    assert any(
        move.x_pos != pytest.approx(logical_x)
        for move, logical_x in zip(stage_events, sorted(position_z), strict=True)
    )


def test_tilted_mirror_tiles_overlap_in_both_lab_axes(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
) -> None:
    """Preserve configured scan and camera-X overlap on a tilted plane."""
    overlap = 0.2
    config = opm_config_factory(
        mode="mirror",
        active_channels=(0,),
        channel_powers=(10.0,),
        channel_exposures_ms=(10.0,),
        scan_range_um=20.0,
        scan_axis_step_um=2.0,
        updates={
            "acq_config": {
                "Positions": {
                    "scan_axis_overlap": overlap,
                    "tile_axis_overlap": overlap,
                }
            }
        },
    )
    plane = CoverslipPlane(100.0, 200.0, 9.0, 0.03, -0.02)
    region = AbsolutePosition(
        z=9.0,
        sequence=MDASequence(
            metadata={COVERSLIP_METADATA_KEY: plane.to_metadata()},
            grid_plan=GridFromEdges(
                left=100.0,
                right=200.0,
                top=200.0,
                bottom=350.0,
                fov_width=50.0,
                fov_height=50.0,
            ),
        ),
    )

    events, _handler = setup_mirrorscan(
        demo_core,
        config,
        MDASequence(stage_positions=[region], axis_order="pzc"),
        workspace_tmp_path / "mirror-tilted-overlap.ome.zarr",
    )
    image_events, _custom_actions = split_events(events)
    first_event_by_position = {}
    for event in image_events:
        first_event_by_position.setdefault(event.index["p"], event)

    logical_scan_origins = np.unique([
        event.metadata["Stage"]["x_pos"]
        for event in first_event_by_position.values()
    ])
    logical_camera_x_origins = np.unique([
        event.metadata["Stage"]["y_pos"]
        for event in first_event_by_position.values()
    ])
    scan_stride_max = 20.0 * (1.0 - overlap)
    camera_x_stride_max = (
        config["acq_config"]["camera_roi"]["crop_x"]
        * demo_core.getPixelSizeUm()
        * (1.0 - overlap)
    )

    assert np.max(np.diff(logical_scan_origins)) <= scan_stride_max + 1e-6
    assert (
        np.max(np.diff(logical_camera_x_origins))
        <= camera_x_stride_max + 1e-6
    )
    reference_z = plane.z_at(100.0, 200.0)
    stage_events = [
        event
        for event in events
        if getattr(event.action, "name", None) == ACTION_STAGE_MOVE
    ]
    for event, move in zip(
        first_event_by_position.values(), stage_events, strict=True
    ):
        stage = event.metadata["Stage"]
        lab_z_offset = -(stage["z_pos"] - reference_z)
        expected_raw_scan = stage["x_pos"] - lab_z_offset / np.tan(np.pi / 6)
        assert move.x_pos == pytest.approx(expected_raw_scan)


def test_timelapse_writer_uses_camera_event_order(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
) -> None:
    """Build timelapse events in the widget-selected frame order.

    Parameters
    ----------
    demo_core : CMMCorePlus
        Core loaded with Micro-Manager demo devices.
    workspace_tmp_path : Path
        Directory used to construct the output handler.
    opm_config_factory : OpmConfigFactory
        Reusable simulated configuration factory.
    simulated_acquisition_hardware : SimulatedAcquisitionHardware
        Initialized singleton hardware simulations.
    split_events : Callable
        Event classifier fixture.
    """
    config = opm_config_factory(mode="timelapse")
    events, handler = setup_timelapse(
        demo_core,
        config,
        MDASequence(
            stage_positions=[(0.0, 0.0, 0.0)],
            time_plan={"interval": 0, "loops": 2},
            axis_order="pztc",
        ),
        workspace_tmp_path / "timelapse.ome.zarr",
    )
    image_events, custom_actions = split_events(events)
    assert [dict(event.index) for event in image_events] == [
        {"t": time, "p": 0, "c": channel, "z": plane}
        for plane in range(2)
        for time in range(2)
        for channel in range(2)
    ]
    assert handler.acquisition_order == ("p", "z", "t", "c")
    assert all(
        event.x_pos is None and event.y_pos is None and event.z_pos is None
        for event in image_events
    )
    assert all(event.metadata["Stage"] == {"x_pos": 0.0, "y_pos": 0.0, "z_pos": 0.0} for event in image_events)
    assert ACTION_ASI_SETUP_SCAN not in custom_actions


def test_timelapse_without_spatial_plan_uses_current_stage_without_command(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
) -> None:
    """Retain the exact current XYZ position without issuing a stage move."""
    demo_core.setXYPosition(123.4567, 234.5678)
    demo_core.setZPosition(34.5678)
    demo_core.waitForSystem()
    current_x, current_y = map(float, demo_core.getXYPosition())
    current_z = float(demo_core.getZPosition())
    config = opm_config_factory(mode="timelapse")

    events, handler = setup_timelapse(
        demo_core,
        config,
        MDASequence(
            time_plan={"interval": 0, "loops": 2},
            axis_order="pztc",
        ),
        workspace_tmp_path / "timelapse-current-position.ome.zarr",
    )
    image_events, custom_actions = split_events(events)

    assert ACTION_STAGE_MOVE not in custom_actions
    assert handler.index_sizes["p"] == 1
    assert image_events
    assert all(
        event.x_pos is None and event.y_pos is None and event.z_pos is None
        for event in image_events
    )
    assert all(
        event.metadata["Stage"]
        == {"x_pos": current_x, "y_pos": current_y, "z_pos": current_z}
        for event in image_events
    )


def test_timelapse_explicit_positions_move_only_at_position_boundaries(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
) -> None:
    """Do not let camera setup re-command Z after explicit stage moves."""
    positions = [(10.0, 20.0, 7.0), (30.0, 40.0, 9.0)]
    config = opm_config_factory(mode="timelapse")

    events, handler = setup_timelapse(
        demo_core,
        config,
        MDASequence(
            stage_positions=positions,
            time_plan={"interval": 0, "loops": 2},
            axis_order="pztc",
        ),
        workspace_tmp_path / "timelapse-explicit-positions.ome.zarr",
    )
    image_events, custom_actions = split_events(events)
    stage_events = [
        event
        for event in events
        if getattr(event.action, "name", None) == ACTION_STAGE_MOVE
    ]

    assert custom_actions.count(ACTION_STAGE_MOVE) == len(positions)
    assert [(event.x_pos, event.y_pos, event.z_pos) for event in stage_events] == positions
    assert handler.index_sizes["p"] == len(positions)
    assert all(
        event.x_pos is None and event.y_pos is None and event.z_pos is None
        for event in image_events
    )
    assert {
        (
            event.metadata["Stage"]["x_pos"],
            event.metadata["Stage"]["y_pos"],
            event.metadata["Stage"]["z_pos"],
        )
        for event in image_events
    } == set(positions)
