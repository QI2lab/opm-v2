"""Prevent closed acquisition previews accumulating across OPM runs."""

from __future__ import annotations

import pytest
from pymmcore_gui._qt.QtCore import QAbstractEventDispatcher
from useq import MDASequence

from opm_v2._app import launch_opm_app


@pytest.mark.parametrize("close_during_acquisition", [False, True])
def test_repeated_acquisitions_release_closed_viewers(
    demo_core,
    workspace_tmp_path,
    qtbot,
    offline_icons,
    opm_config_factory,
    close_during_acquisition,
):
    """Closing previews releases resources while acquisition writing continues."""
    config_path = opm_config_factory.write(
        opm_config_factory(mode="projection"),
        workspace_tmp_path / "viewer_cleanup.json",
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
    controller = window.opm_controller
    manager = window._viewers_manager
    dispatcher = QAbstractEventDispatcher.instance()
    try:
        qtbot.waitUntil(lambda: controller.bootstrap_complete, timeout=5000)
        sequence = MDASequence(
            time_plan={"loops": 5, "interval": 0.1},
            stage_positions=[
                {
                    "x": demo_core.getXPosition(),
                    "y": demo_core.getYPosition(),
                    "z": demo_core.getZPosition(),
                }
            ],
        )
        controller.mda_widget.value = lambda: sequence
        baseline = len(window.dock_manager.dockWidgetsMap())
        for run in range(3):
            controller.custom_execute_mda(
                workspace_tmp_path / f"acquisition_{run}.ome.zarr"
            )
            qtbot.waitUntil(lambda: manager._active_mda_viewer is not None)
            handler = controller.data_handler
            qtbot.waitUntil(lambda: handler.get_view() is not None, timeout=10000)
            if not close_during_acquisition:
                qtbot.waitUntil(
                    lambda: controller._opm_mda_thread is None, timeout=15000
                )
            controller.sync_opm_mda_preview(force=True)
            viewer = manager._active_mda_viewer
            assert viewer.data_wrapper is not None
            dock = next(
                dock
                for dock in window.dock_manager.dockWidgetsMap().values()
                if getattr(dock, "_viewer", None) is viewer
            )
            if close_during_acquisition:
                assert demo_core.mda.is_running()
            with qtbot.waitSignal(dock.destroyed):
                dock.closeDockWidget()
            assert viewer.data_wrapper is None
            assert len(manager) == 0
            assert manager._active_mda_viewer is None
            qtbot.waitUntil(lambda: controller._opm_mda_thread is None, timeout=15000)
            assert handler.is_finalized
            assert handler.get_preview_state()[0] == handler._frame_count

            demo_core.startContinuousSequenceAcquisition()
            qtbot.waitUntil(
                lambda: manager._current_image_preview is not None, timeout=5000
            )
            preview = manager._current_image_preview.widget()
            qtbot.waitUntil(lambda: preview._timer_id is not None)
            assert len(dispatcher.registeredTimers(preview)) == 1
            demo_core.stopSequenceAcquisition()
            assert not dispatcher.registeredTimers(preview)
            # Live creates one reusable preview dock; closed MDA docks are removed.
            assert len(window.dock_manager.dockWidgetsMap()) == baseline + 1
    finally:
        if demo_core.isSequenceRunning():
            demo_core.stopSequenceAcquisition()
        window.close()
