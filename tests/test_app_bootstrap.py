"""Test deferred OPM application bootstrap with Micro-Manager demo devices."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pymmcore_gui import WidgetAction
from pymmcore_gui._qt.QtCore import Qt
from pymmcore_gui._qt.QtWidgets import QApplication
from pymmcore_widgets.control._rois.roi_model import RectangleROI

from opm_v2._app import OPM_WIDGET_KEY, launch_opm_app
from opm_v2.engine.setup_events import OPMEventBuilder
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


def test_stage_explorer_export_activates_unambiguous_mda_positions(
    demo_core, workspace_tmp_path, qtbot, offline_icons, opm_config_factory
) -> None:
    """Send preserved ROI bounds and literal centers into the native MDA widget."""
    config_path = opm_config_factory.write(
        opm_config_factory(mode="projection"),
        workspace_tmp_path / "opm_explorer.json",
    )
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
        qtbot.waitUntil(lambda: window.opm_controller.bootstrap_complete, timeout=5000)
        controller = window.opm_controller
        explorer = window.get_widget(WidgetAction.STAGE_EXPLORER)
        roi = RectangleROI(
            (100.0, 200.0),
            (110.0, 205.0),
            text="sample",
            fov_size=explorer._fov_w_h(),
        )
        explorer.roi_manager.add_roi(roi)

        mda_widget = controller.mda_widget
        mda_widget.tab_wdg.setChecked(mda_widget.grid_plan, True)
        replace_button = MagicMock(name="replace_button")
        add_button = MagicMock(name="add_button")
        cancel_button = MagicMock(name="cancel_button")
        message = MagicMock()
        message.addButton.side_effect = [
            replace_button,
            add_button,
            cancel_button,
        ]
        message.clickedButton.return_value = replace_button

        with patch("opm_v2._app.QMessageBox", return_value=message):
            explorer._on_send_to_mda()

        assert mda_widget.tab_wdg.isChecked(mda_widget.stage_positions)
        assert not mda_widget.tab_wdg.isChecked(mda_widget.grid_plan)
        exported = mda_widget.value().stage_positions
        assert len(exported) == 1
        assert exported[0].name.startswith("sample_")
        assert exported[0].sequence is not None
        region = exported[0].sequence.grid_plan
        assert region is not None
        assert region.left == pytest.approx(100.0)
        assert region.right == pytest.approx(110.0)
        assert region.top == pytest.approx(200.0)
        assert region.bottom == pytest.approx(205.0)

        # Send the exact MDAWidget value produced by the real export connection
        # through every requested OPM planner.  This guards the integration seam
        # separately from each mode's component-level coordinate assertions.
        exported_sequence = mda_widget.value()
        for mode in ("projection", "mirror", "stage"):
            controller.config["acq_config"]["opm_mode"] = mode
            events, handler = OPMEventBuilder(
                controller.mmc, controller.config, exported_sequence
            ).build(
                workspace_tmp_path / f"{mode}-explorer.ome.zarr",
                mode=mode,
            )
            assert events
            assert handler is not None
            assert handler.index_sizes["p"] >= 1
            handler.close()

        explorer._send_mode_group.actions()[1].setChecked(True)
        message.addButton.side_effect = [
            replace_button,
            add_button,
            cancel_button,
        ]
        with patch("opm_v2._app.QMessageBox", return_value=message):
            explorer._on_send_to_mda()

        literal = mda_widget.value().stage_positions[0]
        assert literal.sequence is None
        assert literal.x == pytest.approx(105.0)
        assert literal.y == pytest.approx(202.5)
    finally:
        window.close()
