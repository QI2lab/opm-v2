"""Test deferred OPM application bootstrap with Micro-Manager demo devices."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pymmcore_gui import WidgetAction
from pymmcore_gui._qt.QtCore import Qt
from pymmcore_gui._qt.QtWidgets import QApplication
from pymmcore_widgets.control._rois.roi_model import RectangleROI

from opm_v2._app import (
    OPM_WIDGET_KEY,
    OPMAppController,
    _stage_explorer_accelerated_speeds,
    launch_opm_app,
)
from opm_v2.engine.opm_custom_events import (
    ACTION_DAQ,
    ACTION_STAGE_MOVE,
)
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
        pixel_size_um = demo_core.getPixelSizeUm()
        demo_core.setPixelSizeAffine(
            demo_core.getCurrentPixelSizeConfig(),
            (0.0, -pixel_size_um, 0.0, pixel_size_um, 0.0, 0.0),
        )
        explorer._on_pixel_size_affine_changed()
        assert list(explorer._affine_state.system_affine[:2, :2].flat) == pytest.approx(
            [0.0, -pixel_size_um, pixel_size_um, 0.0]
        )
        assert explorer._fov_w_h() == pytest.approx(
            (
                demo_core.getImageHeight() * pixel_size_um,
                demo_core.getImageWidth() * pixel_size_um,
            )
        )
        assert explorer.roi_manager._fov_size == pytest.approx(explorer._fov_w_h())

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


def test_stage_explorer_selected_roi_uses_current_mm_channel_preset(
    demo_core,
    workspace_tmp_path,
    qtbot,
    offline_icons,
    opm_config_factory,
    camera_frame_recorder,
) -> None:
    """Run an unsaved native preview with the current MM channel preset."""
    preview_config = opm_config_factory(
        mode="projection",
        active_channels=(0, 2),
        channel_powers=(13.0, 37.0),
        channel_exposures_ms=(5.0, 9.0),
    )
    config_path = opm_config_factory.write(
        preview_config,
        workspace_tmp_path / "opm_explorer_scan.json",
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
        demo_core.setConfig("Channel", "FITC")
        demo_core.waitForConfig("Channel", "FITC")
        # Match OPM_mmgr.cfg: Channel is a normal Config Groups entry but is not
        # necessarily designated as MMCore's special channel group.
        demo_core.setChannelGroup("")
        explorer = window.get_widget(WidgetAction.STAGE_EXPLORER)
        fov_width, fov_height = explorer._fov_w_h()
        roi = RectangleROI(
            (100.0, 200.0),
            (110.0, 205.0),
            text="direct_scan",
            fov_size=(fov_width, fov_height),
        )
        explorer.roi_manager.add_roi(roi)
        explorer.roi_manager.select_roi(roi)

        controller = window.opm_controller
        with (
            patch.object(controller, "prepare_stage_explorer_preview") as prepare,
            qtbot.waitSignal(demo_core.mda.events.sequenceFinished, timeout=10000),
        ):
            explorer._on_scan_action()

        prepare.assert_called_once_with()
        assert camera_frame_recorder.frames
        assert all("g" in event.index for event in camera_frame_recorder.events)
        assert all(event.channel is None for event in camera_frame_recorder.events)
        assert all(event.exposure is None for event in camera_frame_recorder.events)
        actions = controller.opm_engine.simulated_custom_actions
        assert ACTION_DAQ not in actions
        assert ACTION_STAGE_MOVE not in actions
        assert controller.opm_engine.simulated_laser_powers == {}
        assert (
            controller.config["acq_config"]["DAQ"]
            == preview_config["acq_config"]["DAQ"]
        )
    finally:
        window.close()


def test_stage_explorer_preview_uses_four_times_current_xy_speed() -> None:
    """Annotate every preview move from the actual per-axis stage speeds."""
    mmc = MagicMock()
    mmc.getXYStageDevice.return_value = "XYStage"
    mmc.hasProperty.return_value = True
    mmc.getProperty.side_effect = lambda _device, prop: {
        "MotorSpeedX-S(mm/s)": "0.05",
        "MotorSpeedY-S(mm/s)": "0.08",
    }[prop]
    accelerated = _stage_explorer_accelerated_speeds(mmc)

    assert accelerated == pytest.approx(
        {"move_speed_x_mm_s": 0.2, "move_speed_y_mm_s": 0.32}
    )


def test_stage_explorer_arms_projection_from_mm_config_groups() -> None:
    """Program projection preview from live MM properties, not OPM channels."""
    controller = object.__new__(OPMAppController)
    controller.config = {
        "Camera": {"camera_id": "Camera"},
        "OPM": {
            "channel_ids": ["405nm", "488nm", "561nm", "637nm", "730nm"]
        },
        "acq_config": {
            "DAQ": {"laser_blanking": True},
            "camera_roi": {
                "center_x": 1151,
                "center_y": 1151,
                "crop_x": 1900,
                "crop_y": 386,
            },
        },
    }
    controller.debug = MagicMock()
    controller.warning = MagicMock()
    controller.mmc = MagicMock()
    controller.mmc.getShutterDevice.return_value = "Shutter"
    controller.mmc.getLoadedDevices.return_value = {
        "OPM-live-mode",
        "ImageGalvoMirrorRange",
        "Laser",
        "Camera",
    }
    controller.mmc.isSequenceRunning.return_value = False
    controller.mmc.getPixelSizeUm.return_value = 0.115
    controller.mmc.getROI.return_value = (201, 716, 1900, 869)
    controller.mmc.getExposure.return_value = 300.0
    properties = {
        ("OPM-live-mode", "Label"): "1-Projection",
        ("Camera", "Exposure"): "300",
        ("ImageGalvoMirrorRange", "Position"): "100",
        ("Laser", "Label"): "488nm",
    }
    controller.mmc.getProperty.side_effect = lambda device, prop: properties[
        (device, prop)
    ]

    controller.opm_nidaq = MagicMock()
    controller.opm_nidaq.running.side_effect = [False, False, True]
    controller.opm_nidaq.programmed.return_value = True
    controller.opm_nidaq.channel_states = [False, True, False, False, False]
    controller.opm_nidaq.scan_type = "projection"
    controller.opm_nidaq.exposure_ms = 300.0
    controller.opm_nidaq.image_mirror_range_um = 100.0

    controller.prepare_stage_explorer_preview()

    controller.opm_nidaq.set_acquisition_params.assert_called_once_with(
        scan_type="projection",
        channel_states=[False, True, False, False, False],
        image_mirror_range_um=100.0,
        exposure_ms=300.0,
        laser_blanking=True,
    )
    controller.opm_nidaq.clear_tasks.assert_called_once_with()
    controller.opm_nidaq.generate_waveforms.assert_called_once_with()
    controller.opm_nidaq.program_daq_waveforms.assert_called_once_with()
    controller.opm_nidaq.start_waveform_playback.assert_called_once_with()
