"""Test deferred OPM application bootstrap with Micro-Manager demo devices."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pymmcore_gui import WidgetAction
from pymmcore_gui._qt.QtCore import Qt
from pymmcore_gui._qt.QtWidgets import QApplication
from pymmcore_gui.actions import widget_actions as pymmcore_widget_actions
from pymmcore_widgets import StageExplorer
from pymmcore_widgets.control._rois.roi_model import RectangleROI

from opm_v2._app import (
    _STAGE_EXPLORER_REQUIRED_METHODS,
    OPM_WIDGET_KEY,
    OPMAppController,
    _install_stage_explorer_export_compatibility,
    _stage_explorer_accelerated_speeds,
    _stage_explorer_mouse_double_click,
    _stage_explorer_on_frame_ready,
    _stage_explorer_scratch_output,
    _validate_stage_explorer_compatibility,
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
from opm_v2.utils.coverslip import COVERSLIP_METADATA_KEY, CoverslipPlane


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


@pytest.mark.parametrize(
    "with_coverslip",
    (False, True),
    ids=("without-coverslip", "with-coverslip"),
)
def test_stage_explorer_export_activates_unambiguous_mda_positions(
    demo_core,
    workspace_tmp_path,
    qtbot,
    offline_icons,
    opm_config_factory,
    with_coverslip,
) -> None:
    """Export a real Stage Explorer ROI with and without a fitted plane."""
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
        plane = CoverslipPlane(100.0, 200.0, 7.0, 0.01, -0.02)
        if with_coverslip:
            roi._opm_coverslip_plane = plane
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
        nested_metadata = dict(exported[0].sequence.metadata or {})
        if with_coverslip:
            assert CoverslipPlane.from_metadata(
                nested_metadata[COVERSLIP_METADATA_KEY]
            ) == plane
        else:
            assert COVERSLIP_METADATA_KEY not in nested_metadata

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
            if with_coverslip:
                image_events = [
                    event
                    for event in events
                    if "DAQ" in event.metadata and "Stage" in event.metadata
                ]
                assert image_events
                for event in image_events:
                    if mode == "stage" and event.index.get("z", 0) != 0:
                        # ASI advances physical X during the hardware sequence;
                        # the Z correction is applied at each software boundary.
                        continue
                    stage = event.metadata["Stage"]
                    assert stage["z_pos"] == pytest.approx(
                        plane.z_at(stage["x_pos"], stage["y_pos"]),
                        abs=0.01,
                    )
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


def test_stage_explorer_uses_mirror_footprint_only_while_acquiring(
    demo_core, workspace_tmp_path, qtbot, offline_icons, opm_config_factory
) -> None:
    """Keep preview cells camera-sized until a mirror acquisition starts."""
    config = opm_config_factory(
        mode="mirror",
        camera_shape=(64, 32),
        scan_range_um=4.0,
    )
    config_path = opm_config_factory.write(
        config,
        workspace_tmp_path / "opm_explorer_mirror_footprint.json",
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
        roi = RectangleROI(
            (0.0, 0.0),
            (20.0, 20.0),
            fov_size=explorer.roi_manager._fov_size,
        )
        explorer.roi_manager.add_roi(roi)

        camera_footprint = explorer._fov_w_h()
        preview_overlap, _mode = explorer._toolbar.scan_menu.value()
        assert explorer.roi_manager._fov_size == pytest.approx(camera_footprint)
        assert roi.fov_size == pytest.approx(camera_footprint)
        assert explorer.roi_manager.scan_overlap == pytest.approx(preview_overlap)

        controller._opm_acquisition_active = True
        controller._opm_scan_footprint_active = False
        controller.data_handler = MagicMock()
        controller.data_handler.get_preview_state.return_value = (0, {})
        controller.sync_stage_explorer_acquisition_footprint()
        assert explorer.roi_manager._fov_size == pytest.approx(camera_footprint)
        assert roi.fov_size == pytest.approx(camera_footprint)

        controller.data_handler.get_preview_state.return_value = (1, {})
        controller.sync_stage_explorer_acquisition_footprint()
        mirror_footprint = (4.0, 64 * pixel_size_um)
        assert explorer.roi_manager._fov_size == pytest.approx(mirror_footprint)
        assert roi.fov_size == pytest.approx(mirror_footprint)
        assert explorer.roi_manager.scan_overlap == pytest.approx((15.0, 20.0))
        marker = explorer._stage_pos_marker
        assert marker is not None
        marker_local_size = np.asarray(
            (marker._rect.width, marker._rect.height), dtype=float
        )
        marker_world_size = (
            np.abs(explorer._affine_state.system_affine[:2, :2])
            @ marker_local_size
        )
        assert marker_world_size == pytest.approx(mirror_footprint)

        controller._opm_acquisition_active = False
        controller._opm_scan_footprint_active = False
        controller.refresh_stage_explorer_footprint()
        assert explorer.roi_manager._fov_size == pytest.approx(camera_footprint)
        assert roi.fov_size == pytest.approx(camera_footprint)
        restored_marker_size = np.asarray(
            (marker._rect.width, marker._rect.height), dtype=float
        )
        restored_marker_world_size = (
            np.abs(explorer._affine_state.system_affine[:2, :2])
            @ restored_marker_size
        )
        assert restored_marker_world_size == pytest.approx(camera_footprint)

        for mode in ("projection", "stage"):
            config["acq_config"]["opm_mode"] = mode
            controller._opm_acquisition_active = True
            controller._opm_scan_footprint_active = True
            controller.update_config_snapshot(config)
            assert explorer.roi_manager._fov_size == pytest.approx(
                explorer._fov_w_h()
            )
            assert roi.fov_size == pytest.approx(explorer._fov_w_h())
        controller._opm_acquisition_active = False
        controller._opm_scan_footprint_active = False
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
        updates={
            "OPM": {
                "stage_explorer_scratch_dir": str(
                    workspace_tmp_path / "stage_explorer_scratch"
                )
            }
        },
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
        assert len(controller._stage_explorer_scratch_dirs) == 1
        scratch_path = Path(controller._stage_explorer_scratch_dirs[0].name)
        assert scratch_path.parent == workspace_tmp_path / "stage_explorer_scratch"
        assert (scratch_path / "manifest.json").is_file()
    finally:
        window.close()


def test_stage_explorer_scratch_output_uses_unique_configured_child(
    workspace_tmp_path,
) -> None:
    """Never use the configured scratch parent itself as an overwrite target."""
    scratch_parent = workspace_tmp_path / "explorer_data"
    controller = SimpleNamespace(
        config={"OPM": {"stage_explorer_scratch_dir": str(scratch_parent)}},
        _stage_explorer_scratch_dirs=[],
    )

    first = _stage_explorer_scratch_output(controller)
    second = _stage_explorer_scratch_output(controller)

    try:
        first_path = Path(first.root_path)
        second_path = Path(second.root_path)
        assert first.format.name == "scratch"
        assert first.overwrite
        assert first_path.parent == scratch_parent
        assert second_path.parent == scratch_parent
        assert first_path != second_path
        assert first_path.is_dir()
        assert second_path.is_dir()
    finally:
        for scratch_dir in controller._stage_explorer_scratch_dirs:
            scratch_dir.cleanup()


def test_stage_explorer_preview_uses_four_times_y_speed_for_both_axes() -> None:
    """Use the Y point-move speed for both axes during an ROI preview."""
    mmc = MagicMock()
    mmc.getXYStageDevice.return_value = "XYStage"
    mmc.hasProperty.return_value = True
    mmc.getProperty.side_effect = lambda _device, prop: {
        "MotorSpeedX-S(mm/s)": "0.05",
        "MotorSpeedY-S(mm/s)": "0.08",
    }[prop]
    accelerated = _stage_explorer_accelerated_speeds(mmc)

    assert accelerated == pytest.approx(
        {"move_speed_x_mm_s": 0.32, "move_speed_y_mm_s": 0.32}
    )
    mmc.setProperty.assert_called_once_with(
        "XYStage", "MotorSpeedX-S(mm/s)", 0.08
    )


def test_stage_explorer_double_click_equalizes_speed_then_delegates_upstream() -> None:
    """Preserve upstream move-and-snap after setting X equal to Y."""
    mmc = MagicMock()
    mmc.getXYStageDevice.return_value = "XYStage"
    mmc.hasProperty.return_value = True
    mmc.getProperty.return_value = "0.08"
    original_callback = MagicMock()
    explorer = SimpleNamespace(
        _mmc=mmc,
        _opm_original_on_mouse_double_click=original_callback,
    )
    event = object()

    _stage_explorer_mouse_double_click(explorer, event)

    mmc.setProperty.assert_called_once_with(
        "XYStage", "MotorSpeedX-S(mm/s)", 0.08
    )
    original_callback.assert_called_once_with(event)


def test_stage_explorer_double_click_wrapper_is_installed() -> None:
    """Install the equal-speed wrapper around the upstream callback."""
    _install_stage_explorer_export_compatibility()

    assert (
        pymmcore_widget_actions._StageExplorer._on_mouse_double_click
        is _stage_explorer_mouse_double_click
    )
    assert (
        pymmcore_widget_actions._StageExplorer._opm_original_on_mouse_double_click
        is StageExplorer._on_mouse_double_click
    )


def test_stage_explorer_private_api_is_validated_before_patching() -> None:
    """Fail clearly when a pymmcore-gui update changes the private shim API."""
    incompatible_actions = SimpleNamespace(
        _StageExplorer=type(
            "ChangedStageExplorer",
            (),
            {
                method_name: object()
                for method_name in (
                    set(_STAGE_EXPLORER_REQUIRED_METHODS)
                    - {"_on_scan_action"}
                )
            },
        ),
        _setup_stage_mda_connections=object(),
    )

    with pytest.raises(
        RuntimeError,
        match=r"Unsupported pymmcore-gui.*missing: _on_scan_action",
    ):
        _validate_stage_explorer_compatibility(incompatible_actions)


def test_stage_explorer_ignores_frames_from_regular_opm_acquisition() -> None:
    """Do not add hardware-triggered OPM stack planes to the Explorer mosaic."""
    original_callback = MagicMock()
    controller = SimpleNamespace(_opm_acquisition_active=True)
    explorer = SimpleNamespace(
        _opm_controller=controller,
        _our_mda_running=False,
        _opm_original_on_frame_ready=original_callback,
        window=MagicMock(),
    )
    image = MagicMock(name="image")
    event = MagicMock(name="event")

    _stage_explorer_on_frame_ready(explorer, image, event)

    original_callback.assert_not_called()

    explorer._our_mda_running = True
    _stage_explorer_on_frame_ready(explorer, image, event)

    original_callback.assert_called_once_with(image, event)


def test_regular_opm_temporarily_disconnects_stage_explorer_frames() -> None:
    """Prevent even no-op Explorer callbacks from filling the Qt event queue."""
    controller = object.__new__(OPMAppController)
    frame_ready = MagicMock()
    explorer = SimpleNamespace(_on_frame_ready=MagicMock())
    controller.mmc = SimpleNamespace(
        mda=SimpleNamespace(events=SimpleNamespace(frameReady=frame_ready))
    )
    controller.win = SimpleNamespace(get_widget=MagicMock(return_value=explorer))
    controller._suspended_stage_explorer = None

    controller._set_stage_explorer_frame_updates(False)
    controller._set_stage_explorer_frame_updates(False)
    controller._set_stage_explorer_frame_updates(True)
    controller._set_stage_explorer_frame_updates(True)

    frame_ready.disconnect.assert_called_once_with(explorer._on_frame_ready)
    frame_ready.connect.assert_called_once_with(explorer._on_frame_ready)
    assert controller._suspended_stage_explorer is None


def test_regular_opm_temporarily_stops_stage_explorer_position_polling() -> None:
    """Avoid serial W X/W Y traffic while acquisition code waits on ASI Busy."""
    controller = object.__new__(OPMAppController)
    explorer = SimpleNamespace(poll_stage_position=True)
    controller.win = SimpleNamespace(get_widget=MagicMock(return_value=explorer))
    controller._stage_explorer_polling_was_enabled = None

    controller._set_stage_explorer_position_polling(False)
    controller._set_stage_explorer_position_polling(False)

    assert explorer.poll_stage_position is False
    assert controller._stage_explorer_polling_was_enabled is True

    controller._set_stage_explorer_position_polling(True)
    controller._set_stage_explorer_position_polling(True)

    assert explorer.poll_stage_position is True
    assert controller._stage_explorer_polling_was_enabled is None


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


def test_opm_preview_coalesces_frames_and_restores_native_follow_mode() -> None:
    """Refresh NDV once per timer interval instead of once per camera frame."""
    controller = object.__new__(OPMAppController)
    current_index = {"t": 0, "p": 0, "c": 0, "z": 0}
    data_changed = MagicMock()
    wrapper = SimpleNamespace(data_changed=SimpleNamespace(emit=data_changed))
    viewer = SimpleNamespace(
        data_wrapper=wrapper,
        display_model=SimpleNamespace(current_index=current_index),
    )
    manager = SimpleNamespace(_active_mda_viewer=viewer, _follow_acquisition=True)
    controller.win = SimpleNamespace(_viewers_manager=manager)
    controller.data_handler = MagicMock()
    controller.data_handler.get_preview_state.return_value = (
        12,
        {"t": 0, "p": 2, "c": 1, "z": 37},
    )
    controller._opm_acquisition_active = True
    controller._opm_scan_footprint_active = True
    controller._opm_preview_last_frame = -1
    controller.opm_engine = MagicMock()
    controller._set_stage_explorer_frame_updates = MagicMock()
    controller._set_stage_explorer_position_polling = MagicMock()
    controller.refresh_stage_explorer_footprint = MagicMock()

    controller._on_mda_sequence_started()
    assert not controller._opm_scan_footprint_active
    controller.sync_opm_mda_preview()
    controller.sync_opm_mda_preview()

    assert not manager._follow_acquisition
    controller._set_stage_explorer_frame_updates.assert_called_once_with(False)
    controller._set_stage_explorer_position_polling.assert_called_once_with(False)
    controller.refresh_stage_explorer_footprint.assert_called_once_with()
    assert current_index == {"t": 0, "p": 2, "c": 1, "z": 37}
    data_changed.assert_called_once_with()

    controller._on_mda_sequence_finished()

    assert manager._follow_acquisition
    assert not controller._opm_acquisition_active
    controller._set_stage_explorer_frame_updates.assert_called_with(True)
    controller._set_stage_explorer_position_polling.assert_called_with(True)
    assert controller.refresh_stage_explorer_footprint.call_count == 2
    assert data_changed.call_count == 2
    controller.opm_engine.clear_safe_stop.assert_called_once_with()
