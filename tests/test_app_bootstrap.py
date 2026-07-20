from __future__ import annotations

import pytest
from pymmcore_gui._qt.QtWidgets import QApplication, QDockWidget

from opm_v2._app import launch_opm_app
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.PicardShutter import PicardShutter


def test_bootstrap_reuses_demo_core_and_defers_opm_extension_attachment(
    demo_core, workspace_tmp_path, qtbot, offline_icons, opm_config_factory
) -> None:
    config = opm_config_factory(
        mode="stage",
        updates={"OPM": {"simulate_hardware": False}},
    )
    config_path = workspace_tmp_path / "opm_demo.json"
    opm_config_factory.write(config, config_path)

    with pytest.warns(RuntimeWarning, match="not MMQApplication"):
        window = launch_opm_app(
            config_path=config_path,
            mm_config=False,
            mmcore=demo_core,
            exec_app=False,
            simulate_hardware=True,
        )
    qtbot.addWidget(window)

    try:
        assert window.mmcore is demo_core
        assert window.findChild(QDockWidget, "OPMConfigurator") is None

        app = QApplication.instance()
        app.processEvents()
        app.processEvents()
        app.processEvents()

        controller = window.opm_controller
        assert controller.bootstrap_complete
        assert window.findChild(QDockWidget, "OPMConfigurator") is not None
        assert controller.opm_settings_widget is not None
        assert controller.opm_ao_mirror is AOMirror.instance()
        assert controller.opm_nidaq is OPMNIDAQ.instance()
        assert controller.opm_picard_shutter is PicardShutter.instance()
        assert controller.opm_ao_mirror.simulate
        assert controller.opm_nidaq.simulate
        assert controller.opm_picard_shutter.simulate
    finally:
        window.close()
