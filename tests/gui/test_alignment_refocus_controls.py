"""Exercise alignment autofocus persistence and acquisition gating."""

import json

from opm_v2._update_config_widget import OPMSettingsV2
from opm_v2.engine.opm_custom_events import create_ao_optimize_event


def test_alignment_refocus_gui_roundtrip_and_gating(qtbot, workspace_tmp_path, opm_config_factory):
    """Only manual 2D alignment events enable the saved refocus interval."""
    config = opm_config_factory(mode="2d")
    config["acq_config"]["AO"].update(ao_mode="optimize now", daq_mode="2d", metric="gauss_2d")
    path = workspace_tmp_path / "alignment.json"
    path.write_text(json.dumps(config))
    widget = OPMSettingsV2(path)
    qtbot.addWidget(widget)
    assert widget.spbx_alignment_refocus.value() == 0
    assert widget.spbx_alignment_refocus.isEnabled()
    widget.spbx_alignment_refocus.setValue(4)
    saved = json.loads(path.read_text())
    assert create_ao_optimize_event(saved).action.data["AO"]["alignment_refocus_every_n_modes"] == 4
    reopened = OPMSettingsV2(path)
    qtbot.addWidget(reopened)
    assert reopened.spbx_alignment_refocus.value() == 4
    for key, value in [("ao_mode", "at xyz positions"), ("daq_mode", "projection"), ("metric", "gauss_3d_o3")]:
        modified = json.loads(json.dumps(saved))
        modified["acq_config"]["AO"][key] = value
        assert create_ao_optimize_event(modified).action.data["AO"]["alignment_refocus_every_n_modes"] == 0
    reopened.cmbx_ao_mode.setCurrentText("none")
    assert not reopened.spbx_alignment_refocus.isEnabled()
