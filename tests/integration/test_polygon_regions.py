"""Exercise polygon exports through OPM spatial planning and event generation."""

import json
from copy import deepcopy

import pytest
from useq import AbsolutePosition, GridFromEdges, GridFromPolygon, MDASequence

from opm_v2.engine.opm_custom_events import ACTION_ASI_SETUP_SCAN
from opm_v2.engine.setup_events import (
    parse_mda_position_plan,
    setup_mirrorscan,
    setup_projection,
    setup_stagescan,
    setup_timelapse,
)
from opm_v2.utils.coverslip import COVERSLIP_METADATA_KEY, CoverslipPlane


@pytest.mark.parametrize("mode,build", [
    ("projection", setup_projection),
    ("mirror", setup_mirrorscan),
    ("stage", setup_stagescan),
    ("timelapse", setup_timelapse),
])
@pytest.mark.parametrize("with_coverslip", [False, True])
def test_concave_polygon_skips_empty_tiles_and_keeps_dense_position_indices(
    mode, build, with_coverslip, demo_core, workspace_tmp_path,
    opm_config_factory, simulated_acquisition_hardware, split_events,
):
    """Retain polygon shape, exposures, coverslip Z, and depth/time ordering."""
    config = opm_config_factory(
        mode=mode, active_channels=(1, 2), channel_powers=(23, 41),
        channel_exposures_ms=(10, 10), camera_shape=(8, 16), scan_range_um=20,
        updates={"acq_config": {
            "DAQ": {"laser_mode": "simultaneous 488/561"},
            "Positions": {"sample_depth_end_um": 17},
            "stage_scan": {"max_stage_scan_range_um": 30},
        }},
    )
    polygon = GridFromPolygon(
        vertices=[(100, 200), (180, 200), (180, 216), (116, 216), (116, 280), (100, 280)],
        fov_width=50, fov_height=50,
    )
    plane = CoverslipPlane(100, 200, 7, 0.01, -0.02)
    metadata = {COVERSLIP_METADATA_KEY: plane.to_metadata()} if with_coverslip else {}

    def sequence(grid):
        return MDASequence(
            stage_positions=[AbsolutePosition(
                z=7, name="concave", sequence=MDASequence(grid_plan=grid, metadata=metadata),
            )],
            time_plan={"interval": 1, "loops": 2},
        )

    source = sequence(polygon)
    restored = MDASequence.model_validate_json(source.model_dump_json())
    original = deepcopy(config)
    events, handler = build(demo_core, config, restored, workspace_tmp_path / "polygon.ome.zarr")
    rectangle = GridFromEdges(left=100, right=180, top=200, bottom=280, fov_width=50, fov_height=50)
    _, rectangle_handler = build(
        demo_core, config, sequence(rectangle), workspace_tmp_path / "rectangle.ome.zarr",
    )
    try:
        assert config == original
        assert 0 < handler.index_sizes["p"] < rectangle_handler.index_sizes["p"]
        images, _ = split_events(events)
        assert {event.index["t"] for event in images} == {0, 1}
        for time_index in (0, 1):
            assert {event.index["p"] for event in images if event.index["t"] == time_index} == set(range(handler.index_sizes["p"]))
        first_images = {}
        for event in images:
            first_images.setdefault((event.index["t"], event.index["p"]), event)
            assert event.metadata["DAQ"]["current_channel"] == "488 + 561"
        for event in first_images.values():
            stage = event.metadata["Stage"]
            x, y = stage["x_pos"], stage["y_pos"]
            assert x < 150 or y < 250  # Exclude the missing upper-right arm.
            surface_z = plane.z_at(x, y) if with_coverslip else 7
            assert stage["z_pos"] == pytest.approx(surface_z + stage.get("stage_depth_offset_um", 0), abs=0.01)
        assert len({event.metadata["Stage"]["sample_depth_um"] for event in first_images.values()}) > 1
        if mode == "stage":
            scans = [event.action.data for event in events if getattr(event.action, "name", None) == ACTION_ASI_SETUP_SCAN]
            assert len(scans) == 2 * handler.index_sizes["p"]
            lengths = [scan["ASI"]["scan_axis_end_mm"] - scan["ASI"]["scan_axis_start_mm"] for scan in scans]
            assert lengths == pytest.approx([lengths[0]] * len(lengths))
            for scan in scans:
                progress = scan["StageScan"]
                assert progress["position_count"] == handler.index_sizes["p"]
                assert scan["ASI"]["scan_axis_start_mm"] == pytest.approx(progress["x_um"] / 1000)
    finally:
        handler.close()
        rectangle_handler.close()


def test_polygon_parser_retains_processed_shape_and_coverslip_metadata():
    """Honor polygon offset and convex hull without mutating saved vertices."""
    polygon = GridFromPolygon(
        vertices=[(10, 20), (30, 20), (10, 40)], offset=2, convex_hull=True,
    )
    plane = CoverslipPlane(10, 20, 7, 0.01, -0.02)
    position = AbsolutePosition(z=7, sequence=MDASequence(
        grid_plan=polygon, metadata={COVERSLIP_METADATA_KEY: plane.to_metadata()},
    ))
    payload = json.loads(position.model_dump_json())
    original = deepcopy(payload)
    region, = parse_mda_position_plan([payload])
    assert payload == original
    assert region.polygon == polygon
    assert region.coverslip_plane == plane
    assert (region.x_min, region.y_min, region.x_max, region.y_max) == pytest.approx(polygon.poly.bounds)


def test_polygon_parser_rejects_empty_processed_region():
    """Reject polygons eroded away before any hardware commands are made."""
    polygon = GridFromPolygon(vertices=[(0, 0), (1, 0), (0, 1)], offset=-10)
    position = AbsolutePosition(z=7, sequence=MDASequence(grid_plan=polygon))
    with pytest.raises(ValueError, match="empty or nonfinite polygon"):
        parse_mda_position_plan([json.loads(position.model_dump_json())])
