"""Verify fail-safe O2/O3 autofocus camera and MCL-stage handling."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from pymmcore_plus.core import StageDevice

from opm_v2.engine.opm_custom_events import create_o2o3_autofocus_event
from opm_v2.utils.ao_cancel import AOCancelled
from opm_v2.utils.autofocus_remote_unit import (
    find_best_O3_focus_metric,
    manage_O3_focus,
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
    assert stage.wait.call_count == len(stage.setPosition.call_args_list) + 1
    mmc.setPosition.assert_not_called()


@pytest.mark.parametrize("cancel", [False, True])
def test_alignment_autofocus_never_accesses_shutter(monkeypatch, cancel):
    """Focus using existing illumination, including cancellation cleanup."""
    from opm_v2.utils import autofocus_remote_unit as af

    mmc = _autofocus_core()
    stage = mmc.getDeviceObject.return_value
    position = [50.0]
    stage.getPosition.side_effect = lambda: position[0]
    stage.setPosition.side_effect = lambda value: position.__setitem__(0, value)
    mmc.snap.side_effect = lambda: np.full(
        (8, 8), 2000 * np.exp(-(((position[0] - 50.3) / 0.4) ** 2))
    )
    shutter = MagicMock(side_effect=AssertionError("Must not access shutter"))
    monkeypatch.setattr(af.PicardShutter, "instance", shutter)
    kwargs = {
        "mmc": mmc,
        "control_shutter": False,
        "stop_requested": lambda: cancel and mmc.snap.call_count >= 2,
    }
    if cancel:
        with pytest.raises(AOCancelled):
            manage_O3_focus("O3", **kwargs)
        assert position[0] == 50.0
    else:
        assert manage_O3_focus("O3", **kwargs) == pytest.approx(50.3, abs=0.06)
    shutter.assert_not_called()
    mmc.setExposure.assert_not_called()
    mmc.setFocusDevice.assert_not_called()
