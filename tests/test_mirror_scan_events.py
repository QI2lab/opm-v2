from __future__ import annotations

import json

import pytest
from useq import MDASequence

from opm_v2.engine.opm_custom_events import ACTION_ASI_SETUP_SCAN
from opm_v2.engine.opm_engine import OPMEngineV2
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
def test_mirror_scan_camera_order_follows_interleaving_setup(
    demo_core,
    workspace_tmp_path,
    exposures,
    axis_order,
    camera_indices,
    acquisition_order,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
) -> None:
    config = opm_config_factory(
        mode="mirror",
        channel_exposures_ms=exposures,
    )
    sequence = MDASequence(
        stage_positions=[(0.0, 0.0, 0.0)],
        axis_order=axis_order,
    )
    events, handler = setup_mirrorscan(
        demo_core,
        config,
        sequence,
        workspace_tmp_path / "mirror.ome.zarr",
    )
    image_events, custom_actions = split_events(events)
    assert [dict(event.index) for event in image_events] == [
        {"t": 0, "p": 0, "c": channel, "z": plane} for plane, channel in camera_indices
    ]
    assert ACTION_ASI_SETUP_SCAN not in custom_actions
    assert handler.acquisition_order == acquisition_order


def test_timelapse_writer_uses_camera_event_order(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
) -> None:
    config = opm_config_factory(mode="timelapse")
    sequence = MDASequence(
        stage_positions=[(0.0, 0.0, 0.0)],
        time_plan={"interval": 0, "loops": 2},
        axis_order="pztc",
    )
    events, handler = setup_timelapse(
        demo_core,
        config,
        sequence,
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
    assert ACTION_ASI_SETUP_SCAN not in custom_actions


@pytest.mark.parametrize(
    (
        "mode",
        "setup_mode",
        "sequence",
        "expected_shape",
        "acquisition_order",
        "daq_mode",
    ),
    [
        (
            "mirror",
            setup_mirrorscan,
            MDASequence(
                stage_positions=[(0.0, 0.0, 0.0)],
                axis_order="tpzc",
            ),
            (1, 2, 2, 32, 64),
            ["t", "p", "z", "c"],
            "mirror",
        ),
        (
            "timelapse",
            setup_timelapse,
            MDASequence(
                stage_positions=[(0.0, 0.0, 0.0)],
                time_plan={"interval": 0, "loops": 2},
                axis_order="pztc",
            ),
            (2, 2, 2, 32, 64),
            ["p", "z", "t", "c"],
            "2d",
        ),
    ],
)
def test_mirror_based_modes_round_trip_generated_event_metadata(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    read_tensorstore_array,
    split_events,
    mode,
    setup_mode,
    sequence,
    expected_shape,
    acquisition_order,
    daq_mode,
) -> None:
    config = opm_config_factory(mode=mode)
    config_path = opm_config_factory.write(
        config,
        workspace_tmp_path / f"{mode}.json",
    )
    output = workspace_tmp_path / f"{mode}.ome.zarr"
    events, handler = setup_mode(demo_core, config, sequence, output)
    engine = OPMEngineV2(demo_core, config_path, use_hardware_sequencing=False)
    demo_core.register_mda_engine(engine)

    demo_core.run_mda(events, output=handler, block=True)

    assert read_tensorstore_array(output / "0").shape == expected_shape
    image_events, _ = split_events(events)
    root_metadata = json.loads((output / "zarr.json").read_text())
    stored_frames = root_metadata["attributes"]["ome_writers"]["frame_metadata"]
    assert [frame["event_index"] for frame in stored_frames] == [
        dict(event.index) for event in image_events
    ]
    assert [frame["event_metadata"] for frame in stored_frames] == [
        dict(event.metadata) for event in image_events
    ]
    assert all(
        frame["event_metadata"]["DAQ"]["mode"] == daq_mode for frame in stored_frames
    )

    opm_metadata = root_metadata["attributes"]["opm_v2"]
    assert opm_metadata["index_sizes"] == handler.index_sizes
    assert opm_metadata["acquisition_order"] == acquisition_order
    assert opm_metadata["summary_metadata"]["image_infos"][0]["camera_label"] == (
        demo_core.getCameraDevice()
    )
