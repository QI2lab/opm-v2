from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from useq import MDASequence


class FrameRecorder:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.indices: list[dict[str, int]] = []
        self.positions: list[dict[str, float]] = []

    def sequenceStarted(self, _sequence: object, _meta: dict[str, Any]) -> None:
        self.frames.clear()
        self.indices.clear()
        self.positions.clear()

    def frameReady(
        self, image: np.ndarray, event: object, meta: dict[str, Any]
    ) -> None:
        self.frames.append(np.asarray(image).copy())
        self.indices.append(dict(event.index))
        self.positions.append(dict(meta["position"]))

    def sequenceFinished(self, _sequence: object) -> None:
        return None


def test_demo_core_acquires_multichannel_multiplane_data(demo_core: Any) -> None:
    sequence = MDASequence(
        channel_group="Channel",
        channels=[
            {"config": "DAPI", "exposure": 1.0},
            {"config": "FITC", "exposure": 2.0},
        ],
        stage_positions=[(0.0, 0.0, 0.0), (10.0, 20.0, 2.0)],
        z_plan={"range": 2.0, "step": 2.0},
        axis_order="pcz",
    )
    recorder = FrameRecorder()

    demo_core.run_mda(sequence, output=recorder, block=True)

    assert recorder.indices == [
        {"p": position, "c": channel, "z": plane}
        for position in range(2)
        for channel in range(2)
        for plane in range(2)
    ]
    assert len(recorder.frames) == 8
    assert all(frame.dtype == np.uint16 for frame in recorder.frames)
    assert len({frame.shape for frame in recorder.frames}) == 1
    assert recorder.positions[0] == {"x": 0.0, "y": 0.0, "z": -1.0}
    assert recorder.positions[-1] == pytest.approx(
        {"x": 10.0, "y": 20.0, "z": 3.0}, abs=0.01
    )
