"""Test deferred OPM application bootstrap with Micro-Manager demo devices."""

from __future__ import annotations

import pytest
from pymmcore_gui._qt.QtCore import Qt
from pymmcore_gui._qt.QtWidgets import QApplication

from opm_v2._app import OPM_WIDGET_KEY, launch_opm_app
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.PicardShutter import PicardShutter


def test_registered_extension_composes_gui_engine_and_hardware_instances(
    demo_core, workspace_tmp_path, qtbot, offline_icons, opm_config_factory
) -> None:
    """Verify the registered extension composes its real application components.

    Parameters
    ----------
    demo_core : CMMCorePlus
        Shared Micro-Manager core loaded with demo devices.
    workspace_tmp_path : Path
        Workspace-local directory for the application configuration.
    qtbot : pytestqt.qtbot.QtBot
        Qt widget lifecycle helper.
    offline_icons : None
        Fixture replacing remote icons with local SVG files.
    opm_config_factory : OpmConfigFactory
        Factory for a simulated OPM configuration.
    """
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
        with pytest.raises(KeyError):
            window.get_widget(OPM_WIDGET_KEY, create=False)

        app = QApplication.instance()
        app.processEvents()
        app.processEvents()
        app.processEvents()

        controller = window.opm_controller
        assert controller.bootstrap_complete
        assert window.get_widget(OPM_WIDGET_KEY, create=False) is (
            controller.opm_settings_widget
        )
        assert window.get_dock_widget(OPM_WIDGET_KEY).objectName() == (
            f"docked_{OPM_WIDGET_KEY}"
        )
        assert controller.opm_settings_widget is not None
        assert controller.mda_widget.execute_mda != controller.custom_execute_mda
        assert controller.opm_ao_mirror is AOMirror.instance()
        assert controller.opm_nidaq is OPMNIDAQ.instance()
        assert controller.opm_picard_shutter is PicardShutter.instance()
        assert controller.opm_ao_mirror.simulate
        assert controller.opm_nidaq.simulate
        assert controller.opm_picard_shutter.simulate

        grid = controller.mda_widget.grid_plan
        controller.mda_widget.tab_wdg.setChecked(grid, True)
        grid.setMode("bounds")
        bounds = grid._core_xy_bounds
        demo_core.setXYPosition(10.0, 20.0)
        demo_core.waitForDevice(demo_core.getXYStageDevice())
        bounds.btn_top.click()
        bounds.btn_left.click()
        assert bounds.top.value() == pytest.approx(10.0, abs=0.02)
        assert bounds.left.value() == pytest.approx(20.0, abs=0.02)

        shutter_button = controller.opm_settings_widget.picard_shutter_button
        assert controller.opm_picard_shutter.state == "Closed"
        assert not shutter_button.isChecked()

        qtbot.mouseClick(shutter_button, Qt.MouseButton.LeftButton)

        assert controller.opm_picard_shutter.state == "Open"
        assert shutter_button.isChecked()
        assert "OPEN" in shutter_button.text()

        controller.opm_picard_shutter.closeShutter()
        qtbot.waitUntil(lambda: not shutter_button.isChecked())
        assert "CLOSED" in shutter_button.text()
    finally:
        window.close()
