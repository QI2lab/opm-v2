"""Component tests for mirror and timelapse generation with demo hardware."""

from __future__ import annotations

import pytest
from useq import AbsolutePosition, GridFromEdges, MDASequence

from opm_v2.engine.opm_custom_events import ACTION_ASI_SETUP_SCAN
from opm_v2.engine.setup_events import setup_mirrorscan, setup_timelapse


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

    # The 4 um mirror sweep is the scan-plane footprint.  At 15% overlap its
    # maximum X stride is 3.4 um, independent of the deskewed volume width.
    x_positions = sorted({event.x_pos for event in image_events})
    assert x_positions == pytest.approx([100.0, 103.4, 106.8])
    assert [right - left for left, right in zip(x_positions, x_positions[1:])] == (
        pytest.approx([3.4, 3.4])
    )
    assert {event.y_pos for event in image_events} == {200.0}
    assert {event.z_pos for event in image_events} == {9.0}
    assert handler.index_sizes == {"t": 1, "p": 3, "c": 1, "z": 2}


def test_timelapse_writer_uses_camera_event_order(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
    assert_standard_image_fields,
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
    assert_standard_image_fields : Callable
        Reusable standard useq field assertion.
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
    assert_standard_image_fields(image_events)
    assert ACTION_ASI_SETUP_SCAN not in custom_actions
