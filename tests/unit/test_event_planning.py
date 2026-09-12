"""Unit-test the pure useq event-planning helpers."""

from __future__ import annotations

import pytest
from useq import CustomAction, MDAEvent

from opm_v2.engine.setup_events import (
    apply_timepoint_timing,
    clone_event,
    iter_planned_image_events,
)


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
