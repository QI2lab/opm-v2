"""Release acquisition viewer resources using stock pymmcore-gui hooks."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pymmcore_gui._qt.QtAds import CDockWidget
from pymmcore_gui._qt.QtCore import QObject

if TYPE_CHECKING:
    from ndv import ArrayViewer
    from pymmcore_gui import MicroManagerGUI
    from useq import MDASequence


@dataclass
class _ViewerResources:
    """Resources owned by one acquisition tab."""

    sequence_uid: str
    viewer: Any
    coordinate_connection: tuple[Any, Any] | None


class AcquisitionViewerCleanup(QObject):
    """Dispose closed OPM acquisition tabs without modifying pymmcore-gui.

    Parameters
    ----------
    window : MicroManagerGUI
        Window whose acquisition docks should release resources on close.
    """

    def __init__(self, window: MicroManagerGUI) -> None:
        """Attach cleanup after the stock manager has created each dock."""
        super().__init__(window)
        self._window = window
        self._manager = window._viewers_manager
        self._viewers: dict[CDockWidget, _ViewerResources] = {}
        self._manager.mdaViewerCreated.connect(self._on_viewer_created)
        window.destroyed.connect(self._dispose_all)
        for uid, viewer in self._manager._seq_viewers.items():
            self._track_viewer(viewer, uid)

    def _on_viewer_created(self, viewer: ArrayViewer, sequence: MDASequence) -> None:
        """Configure a newly created acquisition dock after upstream slots run."""
        self._track_viewer(viewer, str(sequence.uid))

    def _track_viewer(self, viewer: ArrayViewer, sequence_uid: str) -> None:
        """Keep the viewer's resources tied to its acquisition dock."""
        dock = next(
            (
                dock
                for dock in self._window.dock_manager.dockWidgetsMap().values()
                if getattr(dock, "_viewer", None) is viewer
            ),
            None,
        )
        if dock is None or dock in self._viewers:
            return
        connection = None
        view, wrapper = viewer.data, viewer.data_wrapper
        if hasattr(view, "coords_changed") and hasattr(wrapper, "dims_changed"):
            # The stock manager installs this strong callback when creating the
            # viewer. Retain its identity so disposal can disconnect it.
            connection = view.coords_changed, wrapper.dims_changed
        self._viewers[dock] = _ViewerResources(sequence_uid, viewer, connection)
        dock.setFeature(dock.DockWidgetFeature.DockWidgetDeleteOnClose, True)
        dock.closed.connect(self._on_dock_closed)

    def _on_dock_closed(self) -> None:
        """Release resources before Qt deletes the dock and its child widgets."""
        if isinstance(dock := self.sender(), CDockWidget):
            self._dispose(dock)

    def _dispose_all(self) -> None:
        """Release open acquisition viewers when the window is destroyed."""
        for dock in list(self._viewers):
            self._dispose(dock)

    def _dispose(self, dock: CDockWidget) -> None:
        """Cancel pending reads and release one acquisition's data and canvases."""
        resources = self._viewers.pop(dock, None)
        if resources is None:
            return
        uid, viewer = resources.sequence_uid, resources.viewer
        if self._manager._seq_viewers.get(uid) is viewer:
            del self._manager._seq_viewers[uid]
        if self._manager._active_mda_viewer is viewer:
            self._manager._active_mda_viewer = None
            self._manager._view = None
        if resources.coordinate_connection is not None:
            signal, callback = resources.coordinate_connection
            with suppress(ValueError, RuntimeError):
                signal.disconnect(callback)

        # NDV.close() only hides the widget. Cancel reads first so late responses
        # are discarded, then release storage references and graphics explicitly.
        viewer._cancel_futures()
        viewer.data = None
        viewer._clear_canvas()
        viewer.close()
        viewer._canvas.close()
        for histogram in viewer._histograms.values():
            histogram.close()
        if histogram := getattr(viewer, "_shared_histogram", None):
            histogram.close()
        if histogram := getattr(viewer, "_precreated_shared_histogram", None):
            histogram.close()
        dock._viewer = None
        self._manager.viewerDestroyed.emit(uid)


def install_acquisition_viewer_cleanup(
    window: MicroManagerGUI,
) -> AcquisitionViewerCleanup:
    """Install one acquisition cleanup owner on an OPM window.

    Returns
    -------
    AcquisitionViewerCleanup
        Existing or newly installed owner of acquisition viewer cleanup.
    """
    cleanup = getattr(window, "_opm_acquisition_viewer_cleanup", None)
    if cleanup is None:
        cleanup = AcquisitionViewerCleanup(window)
        window._opm_acquisition_viewer_cleanup = cleanup
    return cleanup
