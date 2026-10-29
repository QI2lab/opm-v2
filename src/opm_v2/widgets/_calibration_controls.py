"""OPM-owned Stage Explorer support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QObject, Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QMenu, QToolButton
from vispy.scene.visuals import Markers

if TYPE_CHECKING:
    from ._stage_explorer import StageExplorer


class StageExplorerCalibrationControls(QObject):
    """Presentation-only controls for calibrating a surface from XYZ points.

    The object deliberately contains no hardware or fitting logic. Applications
    connect its signals to their controller and provide points back with
    :meth:`set_points`.
    """

    startRequested = Signal()
    addPointRequested = Signal()
    fitRequested = Signal()
    clearRequested = Signal()

    def __init__(
        self,
        explorer: StageExplorer,
        *,
        title: str = "Coverslip",
    ) -> None:
        super().__init__(explorer)
        self._explorer = explorer
        self._points_visual: Markers | None = None

        action = explorer.toolBar().addAction(QIcon(), title)
        button = explorer.toolBar().widgetForAction(action)
        if not isinstance(button, QToolButton):
            raise RuntimeError("Stage Explorer toolbar did not create a tool button")

        menu = QMenu(button)
        menu.addAction("Start calibration").triggered.connect(self.startRequested)
        menu.addAction("Add current XYZ focus point").triggered.connect(
            self.addPointRequested
        )
        menu.addAction("Fit plane and apply to selected ROI").triggered.connect(
            self.fitRequested
        )
        menu.addSeparator()
        menu.addAction("Clear selected ROI plane").triggered.connect(
            self.clearRequested
        )
        button.setMenu(menu)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

    def set_points(self, points: list[tuple[float, float, float]]) -> None:
        """Display calibration points in stage-world coordinates."""
        if self._points_visual is None:
            self._points_visual = Markers(
                parent=self._explorer._stage_viewer.view.scene,
                symbol="cross",
                face_color="#00ffff",
                edge_color="#003333",
                size=12,
                edge_width=2,
                scaling=False,
            )
        if points:
            self._points_visual.set_data(
                pos=np.asarray([(x, y) for x, y, _z in points], dtype=float)
            )
            self._points_visual.visible = True
        else:
            self._points_visual.visible = False
