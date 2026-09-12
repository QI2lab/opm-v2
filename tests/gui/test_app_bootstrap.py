"""Test deferred OPM application bootstrap with Micro-Manager demo devices."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from pymmcore_gui import CoreAction, WidgetAction
from pymmcore_gui._qt.QtCore import QEvent, Qt
from pymmcore_gui._qt.QtWidgets import QApplication
from pymmcore_widgets.control._rois.roi_model import RectangleROI
from useq import MDASequence

from opm_v2._app import (
    LIVE_DISPLAY_INTERVAL_MS,
    MAX_LIVE_FRAMES_PER_TICK,
    OPM_WIDGET_KEY,
    OPMAppController,
    _add_coverslip_focus_point,
    _BoundedLivePreviewTimer,
    _configure_live_preview,
    _connect_stage_explorer_to_mda,
    _coverslip_planes_by_roi,
    _fit_coverslip_calibration,
    _prepare_stage_explorer_blended_preview,
    _stage_explorer_accelerated_speeds,
    _stage_explorer_scratch_output,
    _start_coverslip_calibration,
    launch_opm_app,
)
from opm_v2.engine.opm_custom_events import (
    ACTION_DAQ,
    ACTION_STAGE_MOVE,
)
from opm_v2.engine.setup_events import OPMEventBuilder
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.mock_nidaq import MockOPMNIDAQ
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.PicardShutter import PicardShutter
from opm_v2.utils.coverslip import (
    COVERSLIP_METADATA_KEY,
    CoverslipPlane,
)


def test_camera_crop_virtual_device_declares_all_five_states() -> None:
    """Keep the 128-pixel preset inside a valid initialized state device."""
    config_lines = (
        Path(__file__).resolve().parents[2] / "OPM_mmgr.cfg"
    ).read_text(encoding="utf-8").splitlines()
    position_lines = [
        line for line in config_lines if line.startswith("Label,ImageCameraCrop,")
    ]

    assert "Property,ImageCameraCrop,Number of positions,5" in config_lines
    assert len(position_lines) == 5
    assert position_lines[0] == "Label,ImageCameraCrop,0,128"


@pytest.mark.parametrize(
    ("positions_selected", "expected_positions"),
    [(False, ()), (True, ((100.0, 200.0, 300.0),))],
)
def test_timelapse_planning_preserves_positions_tab_intent(
    workspace_tmp_path,
    positions_selected,
    expected_positions,
) -> None:
    """Remove the current-position record synthesized for an unchecked tab."""
    controller = object.__new__(OPMAppController)
    positions_widget = object()
    grid_widget = object()
    tab_widget = MagicMock()
    tab_widget.isChecked.side_effect = lambda widget: (
        positions_selected if widget is positions_widget else False
    )
    controller.mda_widget = SimpleNamespace(
        stage_positions=positions_widget,
        grid_plan=grid_widget,
        tab_wdg=tab_widget,
        value=lambda: MDASequence(
            stage_positions=[(100.0, 200.0, 300.0)],
            time_plan={"interval": 0, "loops": 2},
        ),
    )
    controller.mmc = MagicMock()
    controller.config = {}
    controller.opm_ao_mirror = SimpleNamespace(output_path=None)
    output = workspace_tmp_path / "timelapse.ome.zarr"

    with patch("opm_v2._app.OPMEventBuilder") as builder:
        builder.return_value.build.return_value = (["event"], "handler")
        result = controller.create_opm_events(False, "timelapse", output)

    planned_sequence = builder.call_args.args[2]
    assert (
        tuple(
            (position.x, position.y, position.z)
            for position in planned_sequence.stage_positions
        )
        == expected_positions
    )
    assert result == (["event"], "handler")


def test_run_captures_visible_mode_before_acquisition_dispatch(
    workspace_tmp_path,
) -> None:
    """Use the current controls rather than a stale controller mode snapshot."""
    controller = object.__new__(OPMAppController)
    visible_config = {"acq_config": {"opm_mode": "mirror"}}
    requested_output = workspace_tmp_path / "mirror.ome.zarr"
    controller.opm_settings_widget = SimpleNamespace(value=lambda: visible_config)
    controller.mda_widget = SimpleNamespace(prepare_mda=lambda: requested_output)
    controller.update_config_snapshot = MagicMock()
    controller.custom_execute_mda = MagicMock()

    controller.run_opm_acquisition()

    controller.update_config_snapshot.assert_called_once_with(visible_config)
    controller.custom_execute_mda.assert_called_once_with(requested_output)


def test_inactive_crop_edit_does_not_retile_stage_explorer() -> None:
    """Keep a 128-pixel settings edit from rebuilding inactive ROI grids."""
    controller = object.__new__(OPMAppController)
    controller.config_store = MagicMock()
    controller.config = {"acq_config": {"camera_roi": {"crop_y": 256}}}
    controller.opm_engine = None
    controller._opm_scan_footprint_active = False
    controller.refresh_stage_explorer_footprint = MagicMock()
    updated = {"acq_config": {"camera_roi": {"crop_y": 128}}}

    controller.update_config_snapshot(updated)

    controller.config_store.replace.assert_called_once_with(updated)
    controller.refresh_stage_explorer_footprint.assert_not_called()


def test_live_state_update_ignores_reentrant_camera_crop_callback() -> None:
    """Do not recursively reconfigure Live while changing the camera ROI."""
    controller = object.__new__(OPMAppController)
    controller.debug = MagicMock()
    controller._updating_live_state = False
    controller._update_live_state_once = MagicMock(
        side_effect=lambda *_args: controller.update_live_state(
            "ImageCameraCrop", "Label"
        )
    )

    controller.update_live_state("ImageCameraCrop", "Label")

    controller._update_live_state_once.assert_called_once_with(
        "ImageCameraCrop", "Label"
    )
    assert controller._updating_live_state is False


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
        assert isinstance(controller.opm_nidaq, MockOPMNIDAQ)
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


def test_window_close_releases_application_owned_resources(
    demo_core, workspace_tmp_path, qtbot, offline_icons, opm_config_factory
) -> None:
    """Close writers, scratch data, and external hardware before hiding the GUI."""
    config_path = opm_config_factory.write(
        opm_config_factory(
            mode="projection",
            updates={
                "OPM": {
                    "stage_explorer_scratch_dir": str(
                        workspace_tmp_path / "shutdown_scratch"
                    )
                }
            },
        ),
        workspace_tmp_path / "opm_shutdown.json",
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
    qtbot.waitUntil(lambda: window.opm_controller.bootstrap_complete, timeout=5000)
    controller = window.opm_controller
    scratch = _stage_explorer_scratch_output(controller)
    scratch_path = Path(scratch.root_path)
    scratch_temp = controller._stage_explorer_scratch_dirs[-1]

    daq = controller.opm_nidaq
    shutter = controller.opm_picard_shutter
    mirror = controller.opm_ao_mirror
    with (
        patch.object(daq, "clear_tasks", wraps=daq.clear_tasks) as clear_daq,
        patch.object(shutter, "shutDown", wraps=shutter.shutDown) as close_shutter,
        patch.object(mirror, "disconnect", wraps=mirror.disconnect) as close_mirror,
        patch.object(
            scratch_temp,
            "cleanup",
            wraps=scratch_temp.cleanup,
        ) as cleanup_scratch,
    ):
        assert window.close()

    assert controller._shutdown_complete
    assert not controller._stage_explorer_scratch_dirs
    assert scratch_path.parent.name == "shutdown_scratch"
    cleanup_scratch.assert_called_once_with()
    clear_daq.assert_called_once_with()
    close_shutter.assert_called_once_with()
    close_mirror.assert_called_once_with()

    # A repeated fallback shutdown must not touch released hardware again.
    controller.shutdown()
    clear_daq.assert_called_once_with()
    close_shutter.assert_called_once_with()
    close_mirror.assert_called_once_with()


def test_window_close_waits_for_active_mda_thread(
    demo_core, workspace_tmp_path, qtbot, offline_icons, opm_config_factory
) -> None:
    """Keep the GUI open until a requested acquisition cancellation completes."""
    config_path = opm_config_factory.write(
        opm_config_factory(mode="projection"),
        workspace_tmp_path / "opm_shutdown_wait.json",
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
    qtbot.waitUntil(lambda: window.opm_controller.bootstrap_complete, timeout=5000)
    controller = window.opm_controller
    mda_thread = MagicMock()
    mda_thread.is_alive.return_value = True
    controller._opm_mda_thread = mda_thread

    with patch.object(
        controller.opm_engine,
        "request_safe_stop",
        wraps=controller.opm_engine.request_safe_stop,
    ) as request_stop:
        assert not window.close()

    assert controller._shutdown_requested
    assert not controller._shutdown_complete
    assert controller._shutdown_poll_timer.isActive()
    request_stop.assert_called_once_with()

    mda_thread.is_alive.return_value = False
    controller._continue_shutdown()
    QApplication.instance().processEvents()

    assert controller._shutdown_complete
    assert not controller._shutdown_poll_timer.isActive()
    assert not window.isVisible()


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
        controller._configure_stage_explorer()
        pixel_size_um = demo_core.getPixelSizeUm()
        demo_core.setPixelSizeAffine(
            demo_core.getCurrentPixelSizeConfig(),
            (0.0, -pixel_size_um, 0.0, pixel_size_um, 0.0, 0.0),
        )
        explorer._on_pixel_size_affine_changed()
        assert list(explorer._affine_state.system_affine[:2, :2].flat) == pytest.approx([
            0.0,
            -pixel_size_um,
            pixel_size_um,
            0.0,
        ])
        assert explorer._fov_w_h() == pytest.approx((
            demo_core.getImageHeight() * pixel_size_um,
            demo_core.getImageWidth() * pixel_size_um,
        ))
        assert explorer.roi_manager._fov_size == pytest.approx(explorer._fov_w_h())

        roi = RectangleROI(
            (100.0, 200.0),
            (110.0, 205.0),
            text="sample",
            fov_size=explorer._fov_w_h(),
        )
        plane = CoverslipPlane(100.0, 200.0, 7.0, 0.01, -0.02)
        explorer.roi_manager.add_roi(roi)
        if with_coverslip:
            explorer.roi_manager.select_roi(roi)
            if "OPM-live-mode" not in set(demo_core.getAvailableConfigGroups()):
                demo_core.defineConfigGroup("OPM-live-mode")
                demo_core.defineConfig("OPM-live-mode", "Standard")
                demo_core.defineConfig("OPM-live-mode", "Projection")
            demo_core.setConfig("OPM-live-mode", "Projection")
            calibration_points = (
                (101.0, 201.0, 6.99),
                (109.0, 201.0, 7.07),
                (101.0, 204.0, 6.93),
            )
            _start_coverslip_calibration(explorer)
            for x_um, y_um, z_um in calibration_points:
                demo_core.setXYPosition(x_um, y_um)
                demo_core.setZPosition(z_um)
                demo_core.waitForSystem()
                _add_coverslip_focus_point(explorer)
            _fit_coverslip_calibration(explorer)
            plane = _coverslip_planes_by_roi(explorer)[roi]
            assert plane.slope_x == pytest.approx(0.01, abs=2e-5)
            assert plane.slope_y == pytest.approx(-0.02, abs=2e-5)

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

        with patch(
            "pymmcore_gui.widgets._stage_explorer.QMessageBox",
            return_value=message,
        ):
            explorer._on_send_to_mda()

        assert mda_widget.tab_wdg.isChecked(mda_widget.stage_positions)
        assert not mda_widget.tab_wdg.isChecked(mda_widget.grid_plan)
        positions_path = workspace_tmp_path / f"positions-{with_coverslip}.json"
        mda_widget.stage_positions.save(positions_path)
        mda_widget.stage_positions.setValue(())
        mda_widget.stage_positions.load(positions_path)
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
            assert (
                CoverslipPlane.from_metadata(nested_metadata[COVERSLIP_METADATA_KEY])
                == plane
            )
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
        with patch(
            "pymmcore_gui.widgets._stage_explorer.QMessageBox",
            return_value=message,
        ):
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
        controller._configure_stage_explorer()
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
            np.abs(explorer._affine_state.system_affine[:2, :2]) @ marker_local_size
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
            np.abs(explorer._affine_state.system_affine[:2, :2]) @ restored_marker_size
        )
        assert restored_marker_world_size == pytest.approx(camera_footprint)

        for mode in ("projection", "stage"):
            config["acq_config"]["opm_mode"] = mode
            controller._opm_acquisition_active = True
            controller._opm_scan_footprint_active = True
            controller.update_config_snapshot(config)
            assert explorer.roi_manager._fov_size == pytest.approx(explorer._fov_w_h())
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
        window.opm_controller._configure_stage_explorer()
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
            explorer.toolBar().scan_action.trigger()

        prepare.assert_called_once_with()
        assert camera_frame_recorder.frames
        assert all("g" in event.index for event in camera_frame_recorder.events)
        assert all(event.channel is None for event in camera_frame_recorder.events)
        assert all(event.exposure is None for event in camera_frame_recorder.events)
        actions = controller.opm_engine.simulated_custom_actions
        assert ACTION_DAQ not in actions
        assert ACTION_STAGE_MOVE not in actions
        assert controller.opm_engine.simulated_laser_powers == {}
        assert explorer.blending_enabled is True
        assert explorer.mosaic_tile_count > 0
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

    assert accelerated == pytest.approx({
        "move_speed_x_mm_s": 0.32,
        "move_speed_y_mm_s": 0.32,
    })
    mmc.setProperty.assert_called_once_with("XYStage", "MotorSpeedX-S(mm/s)", 0.08)


def test_stage_explorer_preview_caps_speed_at_adapter_limit() -> None:
    """Publish hardware-valid accelerated speeds in preview metadata."""
    mmc = MagicMock()
    mmc.getXYStageDevice.return_value = "XYStage"
    mmc.hasProperty.return_value = True
    mmc.getProperty.side_effect = lambda _device, prop: {
        "MotorSpeedX-S(mm/s)": "1.2864",
        "MotorSpeedY-S(mm/s)": "1.2864",
    }[prop]
    mmc.hasPropertyLimits.return_value = True
    mmc.getPropertyLowerLimit.return_value = 0.001
    mmc.getPropertyUpperLimit.return_value = 1.2864
    mmc.getAllowedPropertyValues.return_value = ()

    accelerated = _stage_explorer_accelerated_speeds(mmc)

    assert accelerated == pytest.approx({
        "move_speed_x_mm_s": 1.2864,
        "move_speed_y_mm_s": 1.2864,
    })


def test_stage_explorer_speed_property_failure_does_not_disable_interaction() -> None:
    """Keep Explorer connections alive when ASI speed setup is unavailable."""
    controller = SimpleNamespace(
        warning=MagicMock(),
        mmc=MagicMock(),
        _stage_explorer_polling_was_enabled=None,
        _opm_acquisition_active=False,
        prepare_stage_explorer_preview=MagicMock(),
        opm_nidaq=SimpleNamespace(clear_tasks=MagicMock()),
    )
    controller.mmc.isSequenceRunning.return_value = False
    stage_explorer = SimpleNamespace(
        window=MagicMock(return_value=SimpleNamespace(opm_controller=controller)),
        sendToMDARequested=MagicMock(),
        set_scan_handler=MagicMock(),
        set_snap_lifecycle=MagicMock(),
        set_roi_position_transform=MagicMock(),
        interactionError=MagicMock(),
        mda_frame_updates_enabled=True,
        blending_enabled=False,
    )
    mda_widget = MagicMock()

    with (
        patch(
            "opm_v2._app._equalize_stage_explorer_xy_speed",
            side_effect=RuntimeError("ASI property unavailable"),
        ),
        patch("opm_v2._app._install_stage_explorer_coverslip_controls"),
        patch("opm_v2._app._refresh_stage_explorer_display_footprint"),
    ):
        _connect_stage_explorer_to_mda(stage_explorer, mda_widget)

    controller.warning.assert_called_once()
    stage_explorer.sendToMDARequested.connect.assert_called_once()
    stage_explorer.set_scan_handler.assert_called_once()
    stage_explorer.set_snap_lifecycle.assert_called_once()
    assert stage_explorer.blending_enabled is True


def test_stage_explorer_canvas_double_click_moves_and_snaps_demo_hardware(
    demo_core,
    workspace_tmp_path,
    qtbot,
    offline_icons,
    opm_config_factory,
) -> None:
    """Exercise the real canvas signal through async stage motion and snap."""
    config_path = opm_config_factory.write(
        opm_config_factory(mode="projection"),
        workspace_tmp_path / "opm_double_click.json",
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
        explorer = window.get_widget(WidgetAction.STAGE_EXPLORER)
        window.opm_controller._configure_stage_explorer()
        explorer.show()
        qtbot.waitExposed(explorer)
        controller = window.opm_controller
        assert explorer._opm_integration_configured
        assert not hasattr(explorer, "_opm_original_on_mouse_double_click")
        assert explorer._opm_controller is controller
        assert explorer.snap_on_double_click is True
        assert explorer.roi_manager.mode != "create-poly"
        assert not demo_core.isSequenceRunning()
        controller.opm_nidaq.set_acquisition_params(
            scan_type="projection",
            channel_states=[False, True, False, False, False],
            image_mirror_range_um=100.0,
            exposure_ms=300.0,
            laser_blanking=True,
        )
        target_xy = (25.0, 15.0)
        canvas_xy = explorer._stage_viewer.world_to_canvas(target_xy)

        with (
            patch.object(
                controller,
                "prepare_stage_explorer_preview",
                wraps=controller.prepare_stage_explorer_preview,
            ) as prepare_projection,
            patch.object(
                controller.opm_nidaq,
                "start_waveform_playback",
                wraps=controller.opm_nidaq.start_waveform_playback,
            ) as start_projection,
            qtbot.waitSignal(demo_core.events.imageSnapped, timeout=5000),
        ):
            explorer._stage_viewer.canvas.events.mouse_double_click(
                pos=canvas_xy,
                button=1,
            )
            prepare_projection.assert_called_once_with()

        qtbot.waitUntil(lambda: explorer.mosaic_tile_count == 1, timeout=2000)
        start_projection.assert_called_once_with()
        assert explorer.blending_enabled is True
        assert controller.opm_nidaq.scan_type == "projection"
        assert demo_core.getXYPosition() == pytest.approx(target_xy, abs=0.1)
        assert explorer._stage_controller.snap_on_finish is False
        assert not controller.opm_nidaq.running()
        assert not controller.opm_nidaq.programmed()
    finally:
        window.close()


def test_stage_explorer_arms_projection_from_mm_config_groups() -> None:
    """Program projection preview from live MM properties, not OPM channels."""
    controller = object.__new__(OPMAppController)
    controller.config = {
        "Camera": {"camera_id": "Camera"},
        "OPM": {"channel_ids": ["405nm", "488nm", "561nm", "637nm", "730nm"]},
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


def test_projection_settings_identify_stage_explorer_blending_session() -> None:
    """Keep one mosaic only while projection and camera settings are compatible."""
    mmc = MagicMock()
    mmc.getAvailableConfigGroups.return_value = ("Channel",)
    mmc.getChannelGroup.return_value = "Channel"
    mmc.getCurrentConfig.return_value = "FITC"
    mmc.getCameraDevice.return_value = "Camera"
    mmc.getROI.return_value = (0, 0, 1900, 128)
    mmc.getImageWidth.return_value = 1900
    mmc.getImageHeight.return_value = 128
    mmc.getPixelSizeUm.return_value = 0.115
    mmc.getPixelSizeAffine.return_value = (1, 0, 0, 0, 1, 0)
    daq = SimpleNamespace(
        scan_type="projection",
        channel_states=[False, True, False, False, False],
        exposure_ms=5.0,
        image_mirror_range_um=100.0,
    )
    controller = SimpleNamespace(
        mmc=mmc,
        opm_nidaq=daq,
        prepare_stage_explorer_preview=MagicMock(),
        _opm_scan_footprint_active=False,
        config={"acq_config": {"opm_mode": "projection"}},
    )
    explorer = SimpleNamespace(
        _opm_controller=controller,
        camera_fov_size_um=MagicMock(return_value=(218.5, 14.72)),
        begin_blending_session=MagicMock(),
    )

    _prepare_stage_explorer_blended_preview(explorer)
    first_key = explorer.begin_blending_session.call_args.args[0]
    daq.exposure_ms = 7.5
    _prepare_stage_explorer_blended_preview(explorer)
    second_key = explorer.begin_blending_session.call_args.args[0]

    assert first_key != second_key
    assert controller.prepare_stage_explorer_preview.call_count == 2


def test_user_live_request_programs_and_starts_preview_daq() -> None:
    """Drive the production preview callback through the stateful DAQ backend."""
    controller = object.__new__(OPMAppController)
    controller.mmc = MagicMock()
    daq = MockOPMNIDAQ()
    daq.clear_tasks()
    daq.set_acquisition_params(
        scan_type="projection",
        channel_states=[False, True, False, False, False],
        image_mirror_range_um=20.0,
        exposure_ms=10.0,
    )
    controller.opm_nidaq = daq
    live_updates: list[bool] = []
    controller.update_live_state = lambda: live_updates.append(True)
    controller.debug = lambda *_args: None

    controller.setup_preview_mode_callback()

    assert live_updates == [True]
    controller.mmc.clearCircularBuffer.assert_called_once_with()
    assert daq.programmed()
    assert daq.running()
    assert all(task.valid and task.running for task in daq.tasks)


def test_live_preview_avoids_reentrant_qt_event_processing() -> None:
    """Let Qt return from each live-frame timer callback before repainting."""
    preview = SimpleNamespace(process_events_on_update=True)

    _configure_live_preview(SimpleNamespace(widget=lambda: preview))

    assert preview.process_events_on_update is False


def test_live_preview_tick_terminates_while_camera_keeps_refilling() -> None:
    """Never wait for a fast 128-row camera buffer to become empty."""
    core = MagicMock()
    core.getRemainingImageCount.return_value = 10_000
    core.popNextImage.side_effect = range(MAX_LIVE_FRAMES_PER_TICK)
    preview = SimpleNamespace(
        _mmc=core,
        _timer_id=17,
        append=MagicMock(),
    )
    event = SimpleNamespace(
        type=lambda: QEvent.Type.Timer,
        timerId=lambda: 17,
    )
    timer_filter = _BoundedLivePreviewTimer(preview)

    assert timer_filter.eventFilter(preview, event) is True
    assert core.popNextImage.call_count == MAX_LIVE_FRAMES_PER_TICK
    preview.append.assert_called_once_with(MAX_LIVE_FRAMES_PER_TICK - 1)


def test_live_sequence_suspends_and_restores_stage_explorer_polling() -> None:
    """Avoid serial stage polling contention during continuous camera Live."""
    controller = object.__new__(OPMAppController)
    controller.mmc = MagicMock()
    controller.mmc.mda.is_running.return_value = False
    controller._opm_acquisition_active = False
    controller.opm_nidaq = MagicMock()
    controller.opm_nidaq.running.return_value = True
    controller.warning = MagicMock()
    controller._set_stage_explorer_position_polling = MagicMock()

    controller._on_live_sequence_started()
    controller._on_live_sequence_stopped()

    assert controller._set_stage_explorer_position_polling.call_args_list == [
        call(False),
        call(True),
    ]
    controller.opm_nidaq.stop_waveform_playback.assert_called_once_with()
    controller.opm_nidaq.clear_tasks.assert_called_once_with()
    controller.mmc.clearCircularBuffer.assert_called_once_with()


def test_fast_live_exposure_is_coalesced_to_display_rate() -> None:
    """Acquire at camera speed without repainting the GUI at 200 Hz."""
    controller = object.__new__(OPMAppController)
    controller.mmc = MagicMock()
    controller.mmc.isSequenceRunning.return_value = True
    controller.mmc.mda.is_running.return_value = False
    controller.mmc.getExposure.return_value = 2.0
    controller._opm_acquisition_active = False
    preview = SimpleNamespace(
        _timer_id=17,
        killTimer=MagicMock(),
        startTimer=MagicMock(return_value=23),
    )
    controller.win = SimpleNamespace(
        _viewers_manager=SimpleNamespace(
            _current_image_preview=SimpleNamespace(widget=lambda: preview)
        )
    )

    controller._set_live_preview_refresh_interval()

    preview.killTimer.assert_called_once_with(17)
    preview.startTimer.assert_called_once_with(
        LIVE_DISPLAY_INTERVAL_MS,
        Qt.TimerType.PreciseTimer,
    )
    assert preview._timer_id == 23
    assert preview._opm_refresh_interval_ms == LIVE_DISPLAY_INTERVAL_MS


def test_real_live_preview_excludes_stage_explorer_updates(
    demo_core, workspace_tmp_path, qtbot, offline_icons, opm_config_factory
) -> None:
    """Keep an open Explorer idle while the native live viewer consumes frames."""
    config_path = opm_config_factory.write(
        opm_config_factory(mode="projection"),
        workspace_tmp_path / "opm_live_explorer.json",
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
        explorer = window.get_widget(WidgetAction.STAGE_EXPLORER)
        window.opm_controller._configure_stage_explorer()
        polling_before_live = explorer.poll_stage_position
        assert not window.get_action(CoreAction.TOGGLE_LIVE).autoRepeat()
        demo_core.setExposure(2.0)

        for _ in range(3):
            demo_core.startContinuousSequenceAcquisition()
            qtbot.waitUntil(demo_core.isSequenceRunning, timeout=5000)

            assert explorer.poll_stage_position is False
            preview_dock = window._viewers_manager._current_image_preview
            assert preview_dock is not None
            preview = preview_dock.widget()
            assert preview.process_events_on_update is False
            assert isinstance(
                preview._opm_bounded_timer_filter,
                _BoundedLivePreviewTimer,
            )
            qtbot.waitUntil(
                lambda: hasattr(preview, "_opm_refresh_interval_ms"),
                timeout=5000,
            )
            assert preview._opm_refresh_interval_ms == LIVE_DISPLAY_INTERVAL_MS
            qtbot.waitUntil(lambda: preview._timer_id is not None, timeout=5000)

            demo_core.stopSequenceAcquisition()
            qtbot.waitUntil(lambda: not demo_core.isSequenceRunning(), timeout=5000)
            assert preview._timer_id is None
            assert demo_core.getRemainingImageCount() == 0
            assert explorer.poll_stage_position is polling_before_live
            assert not window.opm_controller.opm_nidaq.running()
            assert not window.opm_controller.opm_nidaq.programmed()
    finally:
        if demo_core.isSequenceRunning():
            demo_core.stopSequenceAcquisition()
        qtbot.waitUntil(lambda: not demo_core.isSequenceRunning(), timeout=5000)
        window.close()


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
