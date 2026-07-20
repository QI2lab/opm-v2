from __future__ import annotations

import json

import pytest
from useq import MDASequence

from opm_v2.engine.opm_custom_events import ACTION_ASI_SETUP_SCAN, ACTION_DAQ
from opm_v2.engine.opm_engine import OPMEngineV2
from opm_v2.engine.setup_events import setup_projection


@pytest.mark.parametrize(
    ("active_channels", "positions"),
    [
        ((0,), [(0.0, 0.0, 0.0)]),
        ((0, 1), [(0.0, 0.0, 0.0), (10.0, 20.0, 2.0)]),
        ((0, 2, 4), [(0.0, 0.0, 0.0), (10.0, 20.0, 2.0)]),
    ],
)
def test_projection_events_support_configured_channels_without_triggering_asi(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
    active_channels,
    positions,
) -> None:
    channel_count = len(active_channels)
    config = opm_config_factory(
        mode="projection",
        active_channels=active_channels,
        channel_powers=[10.0 * (index + 1) for index in range(channel_count)],
        channel_exposures_ms=[5.0] * channel_count,
        scan_range_um=32.0,
    )
    sequence = MDASequence(
        stage_positions=positions,
        axis_order="pc",
    )
    events, handler = setup_projection(
        demo_core,
        config,
        sequence,
        workspace_tmp_path / "projection.ome.zarr",
    )
    image_events, custom_actions = split_events(events)

    assert [dict(event.index) for event in image_events] == [
        {"t": 0, "p": position, "c": channel}
        for position in range(len(positions))
        for channel in range(channel_count)
    ]
    assert all(event.metadata["DAQ"]["mode"] == "projection" for event in image_events)
    assert custom_actions.count(ACTION_DAQ) == len(positions)
    assert ACTION_ASI_SETUP_SCAN not in custom_actions
    assert handler.index_sizes == {
        "t": 1,
        "p": len(positions),
        "c": channel_count,
    }


def test_projection_acquires_demo_camera_frames_through_opm_engine(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    read_tensorstore_array,
    split_events,
) -> None:
    config = opm_config_factory(
        mode="projection",
        channel_exposures_ms=(5.0, 5.0),
        camera_shape=(64, 64),
        scan_range_um=32.0,
    )
    config_path = workspace_tmp_path / "opm_demo.json"
    opm_config_factory.write(config, config_path)
    output = workspace_tmp_path / "projection.ome.zarr"
    sequence = MDASequence(stage_positions=[(0.0, 0.0, 0.0), (8.0, 4.0, 2.0)])

    events, handler = setup_projection(demo_core, config, sequence, output)
    engine = OPMEngineV2(demo_core, config_path, use_hardware_sequencing=False)
    demo_core.register_mda_engine(engine)

    demo_core.run_mda(events, output=handler, block=True)

    assert simulated_acquisition_hardware.mirror.current_voltage.shape == (52,)
    assert simulated_acquisition_hardware.daq.simulate
    assert read_tensorstore_array(output / "0" / "0").shape == (1, 2, 32, 64)
    assert read_tensorstore_array(output / "1" / "0").shape == (1, 2, 32, 64)

    image_events, _ = split_events(events)
    for position in range(2):
        stored = json.loads((output / str(position) / "zarr.json").read_text())
        stored_frames = stored["attributes"]["ome_writers"]["frame_metadata"]
        expected_events = [
            event for event in image_events if event.index.get("p", 0) == position
        ]
        assert [frame["event_index"] for frame in stored_frames] == [
            dict(event.index) for event in expected_events
        ]
        assert [frame["event_metadata"] for frame in stored_frames] == [
            dict(event.metadata) for event in expected_events
        ]

    root_metadata = json.loads((output / "zarr.json").read_text())
    opm_metadata = root_metadata["attributes"]["opm_v2"]
    assert opm_metadata["index_sizes"] == {"t": 1, "p": 2, "c": 2}
    assert opm_metadata["acquisition_order"] == ["t", "p", "c"]
    assert opm_metadata["summary_metadata"]["image_infos"][0]["camera_label"] == (
        demo_core.getCameraDevice()
    )
