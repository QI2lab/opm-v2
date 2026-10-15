"""Test persistence and migration of AO initialization controls."""

import json
from unittest.mock import patch

import pytest

from opm_v2._update_config_widget import OPMSettingsV2
from opm_v2.utils.ao_policy import PREVIOUS_AO, PREVIOUS_Z, USE_START


@pytest.mark.parametrize(
    "mode,starting,expected",
    [
        ("at xyz positions", "system flat", USE_START),
        ("at xyz positions", "last optimized", PREVIOUS_AO),
        ("grid at start", "system flat", PREVIOUS_Z),
    ],
)
def test_correction_controls_migrate_and_persist(
    qtbot,
    workspace_tmp_path,
    opm_config_factory,
    mode,
    starting,
    expected,
):
    """Load old configs, then save independent policies without moving a mirror."""
    config = opm_config_factory(
        mode="projection",
        updates={
            "acq_config": {
                "AO": {
                    "ao_mode": mode,
                    "mirror_state": starting,
                }
            }
        },
    )
    config["acq_config"]["AO"].pop("starting_correction", None)
    config["acq_config"]["AO"].pop("sequential_correction", None)
    path = workspace_tmp_path / "ao-controls.json"
    path.write_text(json.dumps(config))
    with patch("opm_v2.hardware.AOMirror.AOMirror.instance") as mirror:
        widget = OPMSettingsV2(path)
        qtbot.addWidget(widget)
        assert widget.cmbx_ao_starting_correction.currentText() == starting
        assert widget.cmbx_ao_sequential_correction.currentText() == expected
        widget.cmbx_ao_starting_correction.setCurrentText("system flat")
        widget.cmbx_ao_sequential_correction.setCurrentText(PREVIOUS_AO)
        mirror.assert_not_called()
    saved = json.loads(path.read_text())["acq_config"]["AO"]
    assert saved["starting_correction"] == "system flat"
    assert saved["sequential_correction"] == PREVIOUS_AO
    reopened = OPMSettingsV2(path)
    qtbot.addWidget(reopened)
    assert reopened.cmbx_ao_starting_correction.currentText() == "system flat"
    assert reopened.cmbx_ao_sequential_correction.currentText() == PREVIOUS_AO
