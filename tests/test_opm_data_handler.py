from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from useq import MDAEvent, MDASequence

from opm_v2.handlers.opm_data_handler import OpmDataHandler


@pytest.mark.parametrize(
    "acquisition_order",
    [
        ("t", "p", "c", "z"),
        ("t", "p", "z", "c"),
    ],
)
def test_opm_data_handler_round_trips_pixels_and_all_extra_metadata(
    workspace_tmp_path,
    read_tensorstore_array,
    acquisition_order,
) -> None:
    output = workspace_tmp_path / "projection.ome.zarr"
    index_sizes = {"t": 1, "p": 2, "c": 2, "z": 2}
    handler = OpmDataHandler(
        path=output,
        index_sizes=index_sizes,
        delete_existing=True,
        acquisition_order=acquisition_order,
    )
    sequence = MDASequence()
    summary_metadata = {
        "image_infos": [
            {
                "width": 4,
                "height": 3,
                "dtype": "uint16",
                "pixel_size_um": 0.2,
            }
        ],
        "experiment": {
            "name": "metadata-round-trip",
            "output_path": Path("configured/output"),
            "numpy_scalar": np.int64(7),
            "numpy_array": np.asarray([1.25, 2.5], dtype=np.float32),
            "tuple_value": ("projection", np.bool_(True)),
        },
    }
    expected_summary_metadata = {
        "image_infos": [
            {
                "width": 4,
                "height": 3,
                "dtype": "uint16",
                "pixel_size_um": 0.2,
            }
        ],
        "experiment": {
            "name": "metadata-round-trip",
            "output_path": str(Path("configured/output")),
            "numpy_scalar": 7,
            "numpy_array": [1.25, 2.5],
            "tuple_value": ["projection", True],
        },
    }
    expected_frames_by_position = {0: [], 1: []}
    try:
        handler.sequenceStarted(sequence, summary_metadata)

        axis_ranges = {axis: range(index_sizes[axis]) for axis in acquisition_order}
        for coordinates in product(*(axis_ranges[axis] for axis in acquisition_order)):
            index = dict(zip(acquisition_order, coordinates, strict=True))
            position = index["p"]
            channel = index["c"]
            plane = index["z"]
            value = 100 * position + 10 * channel + plane
            event_metadata = {
                "DAQ": {
                    "mode": "projection",
                    "channel_states": [True, True, False, False, False],
                    "channel_power": np.float32(10.0 + channel),
                },
                "Camera": {
                    "name": "DemoCamera",
                    "shape": np.asarray([3, 4], dtype=np.int64),
                },
                "OPM": {"angle_deg": np.float64(30.0)},
            }
            expected_event_metadata = {
                "DAQ": {
                    "mode": "projection",
                    "channel_states": [True, True, False, False, False],
                    "channel_power": 10.0 + channel,
                },
                "Camera": {"name": "DemoCamera", "shape": [3, 4]},
                "OPM": {"angle_deg": 30.0},
            }
            event = MDAEvent(
                index=index,
                exposure=10.0 + channel,
                metadata=event_metadata,
            )
            frame_meta = {
                "runner_time_ms": float(value),
                "exposure_ms": 10.0 + channel,
                "position": {
                    "x": float(position * 10),
                    "y": float(position * 20),
                    "z": float(plane),
                },
            }
            handler.frameReady(
                np.full((3, 4), value, dtype=np.uint16),
                event,
                frame_meta,
            )
            expected_frames_by_position[position].append({
                "event_index": index,
                "delta_t": value / 1000.0,
                "exposure_time": (10.0 + channel) / 1000.0,
                "position_x": float(position * 10),
                "position_y": float(position * 20),
                "position_z": float(plane),
                "event_metadata": expected_event_metadata,
                "storage_index": [0, channel, plane],
            })
    finally:
        handler.sequenceFinished(sequence)

    position_zero = read_tensorstore_array(output / "0" / "0")
    position_one = read_tensorstore_array(output / "1" / "0")
    assert position_zero.shape == (1, 2, 2, 3, 4)
    assert position_one.shape == (1, 2, 2, 3, 4)
    assert np.all(position_zero[0, 1, 1] == 11)
    assert np.all(position_one[0, 1, 1] == 111)

    root_metadata = json.loads((output / "zarr.json").read_text())
    assert root_metadata["attributes"]["opm_v2"] == {
        "index_sizes": index_sizes,
        "acquisition_order": list(acquisition_order),
        "summary_metadata": expected_summary_metadata,
    }

    ome_series = json.loads((output / "OME" / "zarr.json").read_text())
    assert ome_series["attributes"]["ome"]["series"] == ["0", "1"]

    expected_axes = [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    for position in range(2):
        image_metadata = json.loads((output / str(position) / "zarr.json").read_text())
        multiscale = image_metadata["attributes"]["ome"]["multiscales"][0]
        assert multiscale["axes"] == expected_axes
        assert multiscale["datasets"] == [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0, 0.2, 0.2]}
                ],
            }
        ]
        assert (
            image_metadata["attributes"]["ome_writers"]["frame_metadata"]
            == (expected_frames_by_position[position])
        )

        array_metadata = json.loads(
            (output / str(position) / "0" / "zarr.json").read_text()
        )
        assert array_metadata["dimension_names"] == ["t", "c", "z", "y", "x"]
        assert array_metadata["data_type"] == "uint16"
        assert array_metadata["shape"] == [1, 2, 2, 3, 4]
