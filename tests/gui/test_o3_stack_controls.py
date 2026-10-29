"""Test GUI migration, persistence, and event configuration for O3 stacks."""

import json

from opm_v2._update_config_widget import OPMSettingsV2
from opm_v2.engine.opm_custom_events import create_ao_optimize_event
from opm_v2.utils.ao_o3_stack import METRIC_NAME


def test_retired_energy_metric_removed_from_saved_config(
    qtbot, workspace_tmp_path, opm_config_factory
):
    """Old configurations cannot re-enable the removed experimental metric."""
    config = opm_config_factory(mode="projection")
    config["OPM"]["ao_metrics"].append("energy_3d_o3")
    config["acq_config"]["AO"].update(metric="energy_3d_o3", ao_mode="optimize now")
    path = workspace_tmp_path / "retired-metric.json"
    path.write_text(json.dumps(config))
    widget = OPMSettingsV2(path)
    qtbot.addWidget(widget)
    assert widget.cmbx_ao_metric.findText("energy_3d_o3") == -1
    saved = json.loads(path.read_text())
    assert "energy_3d_o3" not in saved["OPM"]["ao_metrics"]
    assert saved["acq_config"]["AO"]["ao_mode"] == "none"
    assert saved["acq_config"]["AO"]["metric"] == METRIC_NAME


def test_o3_stack_controls_migrate_old_config_and_reach_events(
    qtbot,
    workspace_tmp_path,
    opm_config_factory,
):
    """Expose the new metric and save its geometry without changing other AO modes."""
    config = opm_config_factory(mode="projection")
    config["OPM"]["ao_metrics"] = ["DCT", "gauss_2d"]
    config["acq_config"]["AO"]["metric"] = "DCT"
    for key in (
        "o3_stack_range_um",
        "o3_stack_step_um",
        "o3_tilt_deg",
        "o3_pixel_size_um",
    ):
        config["acq_config"]["AO"].pop(key, None)
    path = workspace_tmp_path / "o3-controls.json"
    path.write_text(json.dumps(config))
    widget = OPMSettingsV2(path)
    qtbot.addWidget(widget)
    assert widget.cmbx_ao_metric.findText(METRIC_NAME) >= 0
    widget.cmbx_ao_metric.setCurrentText(METRIC_NAME)
    assert not widget.group_o3_stack.isHidden()
    assert not widget.cmbx_ao_active_channel.isEnabled()
    assert not widget.cmbx_ao_daq_mode.isEnabled()
    assert widget.o3_stack_widgets["o3_stack_range_um"].value() == 4.0
    assert widget.o3_stack_widgets["o3_stack_step_um"].value() == 0.5
    widget.o3_stack_widgets["o3_stack_range_um"].setValue(6.0)
    saved = json.loads(path.read_text())
    payload = create_ao_optimize_event(saved).action.data["AO"]
    assert payload["o3_stack_settings"]["range_um"] == 6.0
    assert payload["o3_stack_settings"]["tilt_deg"] == 30.0
    assert payload["o3_stack_settings"]["pixel_size_um"] == 0.115
    assert not any(payload["channel_states"])
    reopened = OPMSettingsV2(path)
    qtbot.addWidget(reopened)
    assert reopened.cmbx_ao_metric.currentText() == METRIC_NAME
    assert reopened.o3_stack_widgets["o3_stack_range_um"].value() == 6.0
    reopened.cmbx_ao_metric.setCurrentText("gauss_2d")
    assert reopened.group_o3_stack.isHidden()
    assert reopened.cmbx_ao_active_channel.isEnabled()
