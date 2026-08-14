"""Verify fail-safe O2/O3 autofocus camera and MCL-stage handling."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pymmcore_plus.core import StageDevice

from opm_v2.engine.opm_custom_events import create_o2o3_autofocus_event
from opm_v2.utils.autofocus_remote_unit import (
    find_best_O3_focus_metric,
)


def _autofocus_core() -> MagicMock:
    mmc = MagicMock()
    mmc.getFocusDevice.return_value = "ZStage:M:37"
    mmc.isSequenceRunning.return_value = False
    stage = MagicMock()
    stage.getPosition.return_value = 50.0
    mmc.getDeviceObject.return_value = stage
    return mmc


def test_autofocus_event_uses_standard_camera_fields() -> None:
    """Express camera setup through useq fields consumed by MDAEngine."""
    event = create_o2o3_autofocus_event(
        exposure_ms=10,
        camera_center=(100, 200),
        camera_crop=(40, 60),
        camera_id="OrcaFusionBT",
    )

    assert event.exposure == 10.0
    assert event.roi is not None
    assert event.roi.model_dump() == {
        "offset_x": 80,
        "offset_y": 170,
        "width": 40,
        "height": 60,
    }
    assert event.properties == [
        ("OrcaFusionBT", "Trigger", "NORMAL"),
        ("OrcaFusionBT", "TriggerPolarity", "POSITIVE"),
        ("OrcaFusionBT", "TRIGGER SOURCE", "INTERNAL"),
    ]


def test_autofocus_failure_never_reassigns_core_focus_and_cleans_up() -> None:
    """Keep experiment Z selected and return the explicitly addressed MCL stage."""
    mmc = _autofocus_core()
    mmc.snap.side_effect = [RuntimeError("first"), RuntimeError("second")]
    shutter = MagicMock()

    with pytest.raises(RuntimeError, match="after one recovery attempt"):
        find_best_O3_focus_metric(
            mmc,
            shutter,
            "MCL NanoDrive Z Stage",
            verbose=False,
        )

    mmc.setFocusDevice.assert_not_called()
    mmc.getDeviceObject.assert_called_once_with(
        "MCL NanoDrive Z Stage",
        StageDevice,
    )
    shutter.openShutter.assert_called_once_with()
    shutter.closeShutter.assert_called_once_with()
    assert mmc.snap.call_count == 2
    stage = mmc.getDeviceObject.return_value
    assert stage.setPosition.call_args_list[-1].args == (50.0,)
    assert stage.wait.call_count == len(stage.setPosition.call_args_list)
    mmc.setPosition.assert_not_called()
