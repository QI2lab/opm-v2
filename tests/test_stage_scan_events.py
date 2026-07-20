from __future__ import annotations

import json

import pytest
from useq import MDASequence

from opm_v2.engine.opm_custom_events import ACTION_ASI_SETUP_SCAN, ACTION_DAQ
from opm_v2.engine.opm_engine import OPMEngineV2
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
def test_stage_scan_events_follow_widget_channel_z_order(
    demo_core,
    workspace_tmp_path,
    axis_order,
    camera_indices,
    acquisition_order,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
) -> None:
    config = opm_config_factory(mode="stage")
    sequence = MDASequence(
        grid_plan={"top": 0.0, "left": 0.0, "bottom": 0.0, "right": 10.0},
        axis_order=axis_order,
    )
    events, handler = setup_stagescan(
        demo_core,
        config,
        sequence,
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


def test_stage_scan_camera_writes_frames_while_simulated_asi_hardware_runs(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    read_tensorstore_array,
    split_events,
) -> None:
    config = opm_config_factory(mode="stage")
    config_path = workspace_tmp_path / "opm_demo_stage.json"
    opm_config_factory.write(config, config_path)
    output = workspace_tmp_path / "stage.ome.zarr"
    sequence = MDASequence(
        grid_plan={"top": 0.0, "left": 0.0, "bottom": 0.0, "right": 10.0},
        axis_order="tpgzc",
    )
    events, handler = setup_stagescan(demo_core, config, sequence, output)
    engine = OPMEngineV2(demo_core, config_path, use_hardware_sequencing=True)
    demo_core.register_mda_engine(engine)

    demo_core.run_mda(events, output=handler, block=True)

    assert engine.simulated_asi_state == {
        "scan_axis_start_mm": 0.0,
        "scan_axis_end_mm": 0.01,
        "scan_axis_speed_mm_s": 0.1,
        "scan_state": "Idle",
    }
    assert engine.simulated_asi_transitions == ["Idle", "Running", "Idle"]
    assert read_tensorstore_array(output / "0").shape == (1, 2, 5, 32, 64)

    image_events, _ = split_events(events)
    root_metadata = json.loads((output / "zarr.json").read_text())
    stored_frames = root_metadata["attributes"]["ome_writers"]["frame_metadata"]
    assert [frame["event_index"] for frame in stored_frames] == [
        dict(event.index) for event in image_events
    ]
    assert [frame["event_metadata"] for frame in stored_frames] == [
        dict(event.metadata) for event in image_events
    ]

    opm_metadata = root_metadata["attributes"]["opm_v2"]
    assert opm_metadata["index_sizes"] == {"t": 1, "p": 1, "c": 2, "z": 5}
    assert opm_metadata["acquisition_order"] == ["t", "p", "z", "c"]
    assert opm_metadata["summary_metadata"]["image_infos"][0]["camera_label"] == (
        demo_core.getCameraDevice()
    )
