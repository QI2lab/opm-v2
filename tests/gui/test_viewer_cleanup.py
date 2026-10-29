"""Validate OPM-owned cleanup with the stock acquisition viewer manager."""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import useq
from pymmcore_gui import MicroManagerGUI
from pymmcore_gui._qt.QtCore import QEvent
from pymmcore_gui._qt.QtWidgets import QApplication

from opm_v2._viewer_cleanup import install_acquisition_viewer_cleanup


@pytest.fixture
def gui(demo_core, qtbot, offline_icons):
    """Provide a stock window with OPM's cleanup hooks installed.

    Returns
    -------
    MicroManagerGUI
        Window using the upstream viewer manager and OPM cleanup hooks.
    """
    window = MicroManagerGUI(mmcore=demo_core)
    qtbot.addWidget(window)
    install_acquisition_viewer_cleanup(window)
    return window


def _create_viewer(gui, mmcore, data):
    """Create one acquisition viewer through the stock manager's event handler.

    Returns
    -------
    tuple
        Sequence, dock, and viewer created by the upstream manager.
    """
    sequence = useq.MDASequence()
    with patch.object(mmcore.mda, "get_view", return_value=data):
        gui._viewers_manager._on_sequence_started(sequence, {})
    dock = gui.dock_manager.dockWidgetsMap()[f"ndv-{str(sequence.uid)[:8]}"]
    return sequence, dock, gui._viewers_manager._active_mda_viewer


def test_closing_acquisition_tabs_releases_resources(gui, demo_core, qtbot):
    """Release data and graphics after every closed acquisition tab."""
    manager = gui._viewers_manager
    baseline = len(gui.dock_manager.dockWidgetsMap())
    for _ in range(3):
        data = np.zeros((2, 8, 8), dtype=np.uint16)
        data_ref = weakref.ref(data)
        sequence, dock, viewer = _create_viewer(gui, demo_core, data)
        qtbot.waitUntil(viewer._is_idle)
        del data
        with (
            patch.object(viewer._canvas, "close", wraps=viewer._canvas.close) as close,
            qtbot.waitSignal(dock.destroyed),
            qtbot.waitSignal(manager.viewerDestroyed) as destroyed,
        ):
            dock.closeDockWidget()
        assert destroyed.args == [str(sequence.uid)]
        close.assert_called_once_with()
        assert viewer.data_wrapper is None
        assert not viewer._futures
        assert all(not c.handles for c in viewer._lut_controllers.values())
        assert len(manager) == 0
        assert manager._active_mda_viewer is None
        assert manager._view is None
        assert len(gui.dock_manager.dockWidgetsMap()) == baseline
        gc.collect()
        assert data_ref() is None
        # Disposal is safe when called again after dock destruction.
        gui._opm_acquisition_viewer_cleanup._dispose(dock)


def test_closing_older_result_preserves_active_viewer(gui, demo_core, qtbot):
    """Keep the active acquisition connected when an older result closes."""
    manager = gui._viewers_manager
    _, older_dock, older_viewer = _create_viewer(gui, demo_core, np.zeros((8, 8)))
    _, active_dock, active_viewer = _create_viewer(gui, demo_core, np.ones((8, 8)))
    with qtbot.waitSignal(older_dock.destroyed):
        older_dock.closeDockWidget()
    assert older_viewer.data is None
    assert len(manager) == 1
    assert manager._active_mda_viewer is active_viewer
    assert manager._view is active_viewer.data
    with qtbot.waitSignal(active_dock.destroyed):
        active_dock.closeDockWidget()
    assert len(manager) == 0


def test_queued_frame_updates_and_running_reads_ignore_closed_viewer(
    gui, demo_core, qtbot
):
    """Ignore delayed upstream refreshes and results from uncancellable reads."""
    manager = gui._viewers_manager
    _, dock, viewer = _create_viewer(gui, demo_core, np.zeros((2, 8, 8)))
    qtbot.waitUntil(viewer._is_idle)
    # A running worker cannot be cancelled; its late result must be ignored.
    future = Future()
    future.set_running_or_notify_cancel()
    viewer._futures[future] = viewer._current_gen
    future.add_done_callback(viewer._on_data_response_ready)
    manager._on_frame_ready(np.zeros((8, 8)), useq.MDAEvent(index={"t": 1}), {})
    dock.closeDockWidget()
    with (
        patch.object(viewer._canvas, "refresh") as refresh,
        patch.object(viewer._canvas, "add_image") as add_image,
    ):
        future.set_result(None)
        qtbot.wait(30)
    refresh.assert_not_called()
    add_image.assert_not_called()
    assert not viewer._futures
    assert viewer.data is None


def test_window_destruction_disposes_open_viewers(demo_core, qtbot, offline_icons):
    """Release any remaining acquisition resources during window destruction."""
    gui = MicroManagerGUI(mmcore=demo_core)
    install_acquisition_viewer_cleanup(gui)
    _, _, viewer = _create_viewer(gui, demo_core, np.zeros((8, 8)))
    qtbot.waitUntil(viewer._is_idle)
    with qtbot.waitSignal(gui.destroyed):
        gui.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert viewer.data is None
    assert not viewer._futures


def test_closing_viewer_disconnects_stream_coordinates(gui, demo_core, qtbot):
    """Disconnect the strong coordinate callback installed by the stock manager."""

    class DynamicArray(np.ndarray):
        """Array supporting the coordinate signal provided by OME StreamView."""

    # StreamView's notification stores a strong callback reference, so explicitly
    # check the connection owned by the manager is removed on disposal.
    data = np.zeros((8, 8)).view(DynamicArray)
    data.coords_changed = MagicMock()
    _, dock, viewer = _create_viewer(gui, demo_core, data)
    wrapper = viewer.data_wrapper
    data.coords_changed.connect.assert_called_once_with(wrapper.dims_changed)
    with qtbot.waitSignal(dock.destroyed):
        dock.closeDockWidget()
    data.coords_changed.disconnect.assert_called_once_with(wrapper.dims_changed)
    assert not gui._opm_acquisition_viewer_cleanup._viewers


def test_cleanup_installation_is_idempotent(gui):
    """Reconnect bootstrap safely without installing duplicate cleanup owners."""
    cleanup = gui._opm_acquisition_viewer_cleanup
    assert install_acquisition_viewer_cleanup(gui) is cleanup
