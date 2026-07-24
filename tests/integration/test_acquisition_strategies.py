"""Parameterize every OPM mode across native and Stage Explorer plans."""

from __future__ import annotations

import pytest

from opm_v2.engine.setup_events import (
    setup_mirrorscan,
    setup_projection,
    setup_stagescan,
    setup_timelapse,
)

_BUILDERS = {
    "projection": setup_projection,
    "mirror": setup_mirrorscan,
    "stage": setup_stagescan,
    "timelapse": setup_timelapse,
}
_AXIS_ORDERS = {
    "projection": "pc",
    "mirror": "pzc",
    "stage": "pzc",
    "timelapse": "pztc",
}
_STAGE_SINGLE_REGION_STRATEGIES = {
    "stage_explorer_single_no_coverslip",
    "stage_explorer_single_coverslip",
}


def _unsupported_expectation(
    mode: str, strategy_name: str
) -> tuple[type[Exception], str] | None:
    """Return the expected error for an invalid mode/plan combination.

    Returns
    -------
    tuple[type[Exception], str] or None
        Exception type and message pattern, or ``None`` for valid pairings.
    """
    if mode == "stage" and strategy_name.startswith("literal_"):
        return ValueError, "requires ROI regions"
    if mode == "stage" and strategy_name.startswith("stage_explorer_multiple_"):
        return ValueError, "supports exactly one"
    if mode == "timelapse" and strategy_name == "native_grid":
        return IndexError, "list index out of range"
    if mode == "timelapse" and strategy_name.startswith("stage_explorer_"):
        return TypeError, "float"
    return None


def _matching_region(
    x_um: float,
    y_um: float,
    region_expectations,
):
    """Return the region containing a logical stage position.

    Returns
    -------
    tuple
        Region index, physical bounds, and fitted coverslip plane.

    Raises
    ------
    AssertionError
        If the generated position falls outside every exported region.
    """
    for region_index, (bounds, plane) in enumerate(region_expectations):
        x_min, x_max, y_min, y_max, _z_anchor = bounds
        if (
            x_min - 1e-6 <= x_um <= x_max + 1e-6
            and y_min - 1e-6 <= y_um <= y_max + 1e-6
        ):
            return region_index, bounds, plane
    raise AssertionError(
        f"Position ({x_um}, {y_um}) was outside every exported ROI"
    )


def test_every_mode_spatial_plan_combination(
    acquisition_mode,
    spatial_plan_strategy,
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
) -> None:
    """Build every valid strategy and explicitly reject invalid pairings."""
    mode = acquisition_mode
    strategy = spatial_plan_strategy
    sequence, region_expectations = strategy.build(
        axis_order=_AXIS_ORDERS[mode],
        include_time=mode == "timelapse",
    )
    config = opm_config_factory(
        mode=mode,
        active_channels=(0,),
        channel_powers=(10.0,),
        channel_exposures_ms=(10.0,),
        camera_shape=(64, 32),
        scan_range_um=4.0,
        scan_axis_step_um=2.0,
        updates={
            "acq_config": {
                "stage_scan": {"max_stage_scan_range_um": 100.0}
            }
        },
    )
    output = workspace_tmp_path / f"{mode}-{strategy.name}.ome.zarr"
    unsupported_expectation = _unsupported_expectation(mode, strategy.name)
    builder = _BUILDERS[mode]

    if unsupported_expectation is not None:
        exception_type, error_pattern = unsupported_expectation
        with pytest.raises(exception_type, match=error_pattern):
            builder(demo_core, config, sequence, output)
        return

    events, handler = builder(demo_core, config, sequence, output)
    try:
        image_events, _custom_actions = split_events(events)
        assert image_events
        assert handler.index_sizes["p"] >= 1
        assert sorted({event.index["p"] for event in image_events}) == list(
            range(handler.index_sizes["p"])
        )

        if strategy.kind == "literal":
            assert handler.index_sizes["p"] == strategy.position_count
        elif strategy.kind == "stage_explorer":
            represented_regions: set[int] = set()
            first_event_by_position = {}
            for event in image_events:
                first_event_by_position.setdefault(event.index["p"], event)
            for event in first_event_by_position.values():
                stage = event.metadata["Stage"]
                region_index, bounds, plane = _matching_region(
                    float(stage["x_pos"]),
                    float(stage["y_pos"]),
                    region_expectations,
                )
                represented_regions.add(region_index)
                expected_z = (
                    plane.z_at(stage["x_pos"], stage["y_pos"])
                    if strategy.with_coverslip
                    else bounds[4]
                )
                assert stage["z_pos"] == pytest.approx(expected_z, abs=0.01)
            assert represented_regions == set(range(strategy.position_count))

        if mode == "stage":
            assert strategy.name == "native_grid" or (
                strategy.name in _STAGE_SINGLE_REGION_STRATEGIES
            )
    finally:
        handler.close()
