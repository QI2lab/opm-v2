"""Unit-test the pure useq event-planning helpers."""

from __future__ import annotations

import numpy as np
import pytest
from useq import CustomAction, MDAEvent

from opm_v2.engine.setup_events import (
    apply_timepoint_timing,
    clone_event,
    iter_planned_image_events,
    resolve_ao_spatial_geometry,
)


def test_ao_geometry_uses_requested_bounds_separately_from_camera_support() -> None:
    positions = [{"x": 100.0, "y": 200.0, "z": 0.0}]
    geometry = resolve_ao_spatial_geometry(
        stage_positions=positions,
        mda_positions_plan=None,
        mda_grid_plan={"bottom": 100.0, "top": 1000.0, "left": 180.0, "right": 220.0},
        opm_mode="stage",
        scan_displacement_um=900.0,
        camera_scan_extent_um=20.0,
        tile_width_um=80.0,
    )

    assert geometry["grid_bounds_um"] == {"x": [100.0, 1000.0], "y": [180.0, 220.0]}
    assert geometry["requested_region_bounds_um"] == geometry["grid_bounds_um"]
    assert geometry["commanded_acquired_support_bounds_um"] == {
        "x": [90.0, 1010.0],
        "y": [160.0, 240.0],
    }
    assert geometry["grid_bounds_source"] == "requested_mda_grid_bounds"
    np.testing.assert_allclose(
        geometry["tile_bounds_um"], [[90.0, 1010.0, 160.0, 240.0]]
    )


def test_ao_geometry_declares_degenerate_requested_axis_expansion() -> None:
    geometry = resolve_ao_spatial_geometry(
        stage_positions=[{"x": 100.0, "y": 200.0, "z": 0.0}],
        mda_positions_plan=None,
        mda_grid_plan={"bottom": 100.0, "top": 1000.0, "left": 200.0, "right": 200.0},
        opm_mode="stage",
        scan_displacement_um=900.0,
        camera_scan_extent_um=20.0,
        tile_width_um=80.0,
    )

    assert geometry["requested_region_bounds_um"]["y"] == [200.0, 200.0]
    assert geometry["grid_bounds_um"]["y"] == [160.0, 240.0]
    assert geometry["degenerate_requested_axes_expanded_to_acquired_support"] == ["y"]


def test_literal_position_ao_geometry_uses_declared_full_footprint_fallback() -> None:
    geometry = resolve_ao_spatial_geometry(
        stage_positions=[{"x": 100.0, "lab_scan_um": 50.0, "y": 200.0, "z": 30.0}],
        mda_positions_plan=None,
        mda_grid_plan=None,
        opm_mode="mirror",
        scan_displacement_um=100.0,
        camera_scan_extent_um=20.0,
        tile_width_um=80.0,
    )

    assert geometry["grid_bounds_um"] == {"x": [-10.0, 110.0], "y": [160.0, 240.0]}
    assert geometry["requested_region_bounds_um"] is None
    assert geometry["grid_bounds_source"] == "acquired_support_for_literal_positions"
    assert geometry["coordinate_frame"] == "depth_normalized_laboratory_scan_x_stage_y"


@pytest.mark.parametrize(
    ("axis_order", "expected_indices"),
    [
        (
            ("c", "z"),
            [
                {"c": 0, "z": 0},
                {"c": 0, "z": 1},
                {"c": 0, "z": 2},
                {"c": 1, "z": 0},
                {"c": 1, "z": 1},
                {"c": 1, "z": 2},
            ],
        ),
        (
            ("z", "c"),
            [
                {"c": 0, "z": 0},
                {"c": 1, "z": 0},
                {"c": 0, "z": 1},
                {"c": 1, "z": 1},
                {"c": 0, "z": 2},
                {"c": 1, "z": 2},
            ],
        ),
    ],
)
def test_planned_image_events_follow_useq_axis_order(
    axis_order: tuple[str, ...], expected_indices: list[dict[str, int]]
) -> None:
    """Verify that the native useq iterator determines camera-frame ordering.

    Parameters
    ----------
    axis_order : tuple[str, ...]
        Requested outer-to-inner useq axis order.
    expected_indices : list[dict[str, int]]
        Camera indices expected in delivery order.
    """
    events = list(iter_planned_image_events({"c": 2, "z": 3}, axis_order))

    assert [dict(event.index) for event in events] == expected_indices
    assert all(not isinstance(event.action, CustomAction) for event in events)
    assert all(event.channel is None for event in events)
    assert all(event.x_pos is None and event.z_pos is None for event in events)


def test_planned_image_events_preserve_useq_time_scheduling() -> None:
    """Verify useq supplies time indices, intervals, and timer reset semantics."""
    events = list(
        iter_planned_image_events({"t": 3, "c": 2}, ("t", "c"), time_interval_s=2.5)
    )

    assert [event.min_start_time for event in events] == [0.0, 0.0, 2.5, 2.5, 5.0, 5.0]
    assert [event.reset_event_timer for event in events] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_planned_image_events_reject_incomplete_axis_order() -> None:
    """Verify inconsistent planning inputs fail before event generation."""
    with pytest.raises(ValueError, match="every planned axis"):
        list(iter_planned_image_events({"c": 2, "z": 3}, ("c",)))


def test_apply_timepoint_timing_only_updates_first_timepoint_event() -> None:
    """Verify OPM custom actions do not replace runner-native time scheduling."""
    events = [MDAEvent(), MDAEvent(), MDAEvent()]

    apply_timepoint_timing(
        events,
        start_index=1,
        time_index=2,
        interval_s=1.25,
        reset_event_timer=True,
    )

    assert events[0].min_start_time is None
    assert events[1].min_start_time == 2.5
    assert events[1].reset_event_timer is True
    assert events[2].min_start_time is None


def test_clone_event_deep_copies_custom_action_data() -> None:
    """Verify mutating one scheduled custom action cannot alter its template."""
    template = MDAEvent(
        action=CustomAction(name="DAQ", data={"DAQ": {"channel_states": [True]}})
    )

    cloned = clone_event(template)
    cloned.action.data["DAQ"]["channel_states"][0] = False

    assert template.action.data["DAQ"]["channel_states"] == [True]
    assert cloned.action.data["DAQ"]["channel_states"] == [False]
