"""Component tests for stage-scan generation with demo hardware."""

from __future__ import annotations

import pytest
from useq import MDASequence

from opm_v2.engine.opm_custom_events import ACTION_ASI_SETUP_SCAN, ACTION_DAQ
from opm_v2.engine.setup_events import setup_stagescan


@pytest.mark.parametrize(
    ("axis_order", "camera_indices", "acquisition_order"),
    [
        (
            "tpgzc",
            [(plane, channel) for plane in range(5) for channel in range(2)],
            ("t", "p", "z", "c"),
        ),
        (
            "tpgcz",
            [(plane, channel) for channel in range(2) for plane in range(5)],
            ("t", "p", "c", "z"),
        ),
    ],
)
def test_stage_events_follow_widget_channel_z_order(
    demo_core,
    workspace_tmp_path,
    axis_order,
    camera_indices,
    acquisition_order,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
    assert_standard_image_fields,
) -> None:
    """Build stage events in each supported channel/Z ordering.

    Parameters
    ----------
    demo_core : CMMCorePlus
        Core loaded with Micro-Manager demo devices.
    workspace_tmp_path : Path
        Directory used to construct the output handler.
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
    config = opm_config_factory(mode="stage")
    events, handler = setup_stagescan(
        demo_core,
        config,
        MDASequence(
            grid_plan={"top": 0.0, "left": 0.0, "bottom": 0.0, "right": 10.0},
            axis_order=axis_order,
        ),
        workspace_tmp_path / "stage.ome.zarr",
    )
    image_events, custom_actions = split_events(events)

    assert custom_actions.count(ACTION_DAQ) == 1
    assert custom_actions.count(ACTION_ASI_SETUP_SCAN) == 1
    assert [dict(event.index) for event in image_events] == [
        {"t": 0, "p": 0, "c": channel, "z": plane} for plane, channel in camera_indices
    ]
    assert handler.index_sizes == {"t": 1, "p": 1, "c": 2, "z": 5}
    assert handler.acquisition_order == acquisition_order
    assert_standard_image_fields(image_events)
