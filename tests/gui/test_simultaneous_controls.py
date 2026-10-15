"""Test the simultaneous illumination selector and Micro-Manager preset."""

import json
from pathlib import Path

import pytest

from opm_v2._update_config_widget import OPMSettingsV2


@pytest.mark.parametrize("mode,pair", [
    ("simultaneous 488/637", (1, 3)),
    ("simultaneous 561/637", (2, 3)),
    ("simultaneous 488/561", (1, 2)),
])
def test_simultaneous_selector_preserves_powers_and_exposures(
    mode, pair,
    qtbot,
    workspace_tmp_path,
    opm_config_factory,
):
    """Persist the selector, show its fixed pair, and leave laser settings intact."""
    config = opm_config_factory(
        mode="mirror",
        active_channels=(0, 1, 3),
        channel_powers=(8, 23, 41),
        channel_exposures_ms=(20, 10, 10),
    )
    path = opm_config_factory.write(config, workspace_tmp_path / "controls.json")
    settings = OPMSettingsV2(path)
    qtbot.addWidget(settings)
    assert settings.cmbx_laser_mode.currentText() == "interleaved"
    settings.cmbx_laser_mode.setCurrentText(mode)
    saved = json.loads(path.read_text())["acq_config"]["DAQ"]
    assert saved["laser_mode"] == mode
    assert saved["channel_states"] == [i in pair for i in range(5)]
    assert saved["channel_powers"] == config["acq_config"]["DAQ"]["channel_powers"]
    assert (
        saved["channel_exposures_ms"]
        == config["acq_config"]["DAQ"]["channel_exposures_ms"]
    )
    assert not settings.chx_405_state.isEnabled()
    assert settings.chx_488_state.isChecked() == (1 in pair)
    assert settings.chx_561_state.isChecked() == (2 in pair)
    assert settings.chx_638_state.isChecked() == (3 in pair)
    reopened = OPMSettingsV2(path)
    qtbot.addWidget(reopened)
    assert reopened.cmbx_laser_mode.currentText() == mode
    reopened.cmbx_laser_mode.setCurrentText("interleaved")
    assert reopened.chx_405_state.isEnabled()
    assert reopened.chx_488_state.isEnabled()


@pytest.mark.parametrize("name", ["488 + 637", "561 + 637", "488 + 561"])
def test_combined_channel_preset_is_valid_and_does_not_set_powers(demo_core, name):
    """Load the actual virtual Laser device definition with the new Channel preset."""
    lines = (
        (Path(__file__).resolve().parents[2] / "OPM_mmgr.cfg").read_text().splitlines()
    )
    demo_core.loadDevice("Laser", "DemoCamera", "DStateDevice")
    position_line = next(
        line for line in lines if line.startswith("Property,Laser,Number of positions,")
    )
    demo_core.setProperty("Laser", "Number of positions", position_line.split(",")[-1])
    demo_core.initializeDevice("Laser")
    for line in lines:
        if line.startswith("Label,Laser,"):
            _, device, index, label = line.split(",")
            demo_core.defineStateLabel(device, int(index), label)
    preset_lines = [
        line for line in lines if line.startswith(f"ConfigGroup,Channel,{name},")
    ]
    assert preset_lines == [f"ConfigGroup,Channel,{name},Laser,Label,{name}"]
    for line in preset_lines:
        _, group, preset, device, prop, value = line.split(",")
        demo_core.defineConfig(group, preset, device, prop, value)
    demo_core.setConfig("Channel", name)
    assert demo_core.getProperty("Laser", "Label") == name


def test_legacy_simultaneous_setting_migrates(qtbot, workspace_tmp_path, opm_config_factory):
    """Preserve the original pair when opening a legacy GUI configuration."""
    config = opm_config_factory(mode="mirror")
    config["acq_config"]["DAQ"]["laser_mode"] = "simultaneous"
    path = opm_config_factory.write(config, workspace_tmp_path / "legacy.json")
    widget = OPMSettingsV2(path)
    qtbot.addWidget(widget)
    assert widget.cmbx_laser_mode.currentText() == "simultaneous 488/637"
    assert json.loads(path.read_text())["acq_config"]["DAQ"]["laser_mode"] == "simultaneous 488/637"
