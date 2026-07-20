"""Component tests for projection event generation with demo hardware."""

from __future__ import annotations

import pytest
from useq import MDASequence

from opm_v2.engine.opm_custom_events import ACTION_ASI_SETUP_SCAN, ACTION_DAQ
from opm_v2.engine.setup_events import (
    normalize_ao_mode,
    normalize_autofocus_mode,
    setup_projection,
)


@pytest.mark.parametrize(
    ("normalizer", "widget_label", "event_label"),
    [
        (normalize_ao_mode, "at xyz position", "at xyz positions"),
        (normalize_ao_mode, "grid at timepoints", "grid at timepoints"),
        (normalize_autofocus_mode, "per timepoint", "at timepoints"),
        (normalize_autofocus_mode, "per xyz position", "at xyz positions"),
        (normalize_autofocus_mode, "once at start", "once at start"),
    ],
)
def test_widget_frequency_labels_normalize_for_event_scheduling(
    normalizer,
    widget_label,
    event_label,
) -> None:
    """Map every differing GUI label to its event-scheduler equivalent.

    Parameters
    ----------
    normalizer : Callable[[str], str]
        Mode-specific normalization function.
    widget_label : str
        Label emitted by the custom GUI.
    event_label : str
        Label consumed by existing scheduling branches.
    """
    assert normalizer(widget_label) == event_label


@pytest.mark.parametrize(
    ("active_channels", "positions"),
    [
        ((0,), [(0.0, 0.0, 0.0)]),
        ((0, 1), [(0.0, 0.0, 0.0), (10.0, 20.0, 2.0)]),
        ((0, 2, 4), [(0.0, 0.0, 0.0), (10.0, 20.0, 2.0)]),
    ],
)
def test_projection_events_support_channels_without_asi(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
    assert_standard_image_fields,
    active_channels,
    positions,
) -> None:
    """Build projection events for reusable channel and position cases.

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
    active_channels : tuple[int, ...]
        Enabled canonical channel indices.
    positions : list[tuple[float, float, float]]
        Stage positions supplied by the standard MDA plan.
    """
    channel_count = len(active_channels)
    config = opm_config_factory(
        mode="projection",
        active_channels=active_channels,
        channel_powers=[10.0 * (index + 1) for index in range(channel_count)],
        channel_exposures_ms=[5.0] * channel_count,
        scan_range_um=32.0,
    )
    events, handler = setup_projection(
        demo_core,
        config,
        MDASequence(stage_positions=positions, axis_order="pc"),
        workspace_tmp_path / "projection.ome.zarr",
    )
    image_events, custom_actions = split_events(events)

    assert [dict(event.index) for event in image_events] == [
        {"t": 0, "p": position, "c": channel}
        for position in range(len(positions))
        for channel in range(channel_count)
    ]
    assert all(event.metadata["DAQ"]["mode"] == "projection" for event in image_events)
    assert_standard_image_fields(image_events)
    assert custom_actions.count(ACTION_DAQ) == len(positions)
    assert ACTION_ASI_SETUP_SCAN not in custom_actions
    assert handler.index_sizes == {"t": 1, "p": len(positions), "c": channel_count}
