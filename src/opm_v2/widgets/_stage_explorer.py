"""OPM Stage Explorer extensions over unmodified upstream widgets.

Adapted from pymmcore-widgets and pymmcore-gui (BSD-3-Clause); see the adjacent
LICENSE.pymmcore-widgets and LICENSE.pymmcore-gui files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import useq
from pymmcore_gui._qt.QtCore import Qt, Signal
from pymmcore_gui._qt.QtGui import QAction, QActionGroup
from pymmcore_gui._qt.QtWidgets import (
    QLabel,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from pymmcore_plus import CMMCorePlus
from pymmcore_widgets.control._q_stage_controller import QStageMoveAccumulator
from pymmcore_widgets.control._rois.roi_manager import GRAY, SceneROIManager
from pymmcore_widgets.control._rois.roi_model import ROI, RectangleROI
from pymmcore_widgets.control._stage_explorer._stage_explorer import (
    AffineState,
    ContrastSlider,
    _StagePoller,
)
from pymmcore_widgets.control._stage_explorer._stage_explorer import (
    StageExplorer as BaseStageExplorer,
)
from pymmcore_widgets.control._stage_explorer._stage_explorer import (
    StageExplorerToolbar as BaseStageExplorerToolbar,
)
from pymmcore_widgets.control._stage_explorer._stage_position_marker import (
    StagePositionMarker,
)
from qtpy.QtCore import Slot
from superqt import QIconifyIcon
from useq import OrderMode

from ._stage_viewer import StageViewer

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable

    from pymmcore_gui._qt.QtGui import QCloseEvent
    from vispy.app.canvas import MouseEvent


class StageExplorerToolbar(BaseStageExplorerToolbar):
    """Add OPM's blend control to the upstream toolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.blend_action = QAction(
            QIconifyIcon("mdi:image-filter-center-focus-weak", color=GRAY),
            "Blend Overlaps",
            self,
        )
        self.blend_action.setCheckable(True)
        actions = self.actions()
        self.insertAction(
            actions[actions.index(self.show_grid_action) + 1], self.blend_action
        )


class StageExplorer(BaseStageExplorer):
    """OPM-owned planning, preview, and hardware lifecycle extensions."""

    interactionError = Signal(object)
    sendToMDARequested = Signal(list, bool)

    def __init__(
        self, parent: QWidget | None = None, mmcore: CMMCorePlus | None = None
    ):
        # Upstream construction fixes the viewer and toolbar types. Initialize
        # Qt here to supply OPM's subclasses while inheriting upstream behavior.
        QWidget.__init__(self, parent)
        self.setWindowTitle("Stage Explorer")

        self._mmc = mmcore or CMMCorePlus.instance()
        self._mmc.events.roiSet.connect(self._on_roi_changed)

        self._stage_controller: QStageMoveAccumulator | None = None
        self._set_stage_controller()

        self._stage_viewer = StageViewer(self)
        self._stage_viewer.setCursor(Qt.CursorShape.CrossCursor)
        self.roi_manager = SceneROIManager(self._stage_viewer.canvas)

        # properties
        self._auto_zoom_to_fit: bool = False
        self._snap_on_double_click: bool = True
        self._poll_stage_position: bool = self._has_devices()
        self._our_mda_running: bool = False
        self._mda_frame_updates_enabled = False
        self._display_fov_um: tuple[float, float] | None = None
        self._scan_handler: Callable[[StageExplorer], None] | None = None
        self._snap_prepare: Callable[[], None] | None = None
        self._snap_cleanup: Callable[[], None] | None = None
        self._pending_snap_cleanup: Callable[[], None] | None = None
        self._roi_position_transform: (
            Callable[[useq.AbsolutePosition, ROI], useq.AbsolutePosition] | None
        ) = None

        # background thread for polling stage position
        self._stage_poller = _StagePoller(self._mmc)
        self._stage_poller.positionChanged.connect(self._on_stage_position_polled)

        # marker for stage position (created when a camera is available)
        self._stage_pos_marker: StagePositionMarker | None = None
        self._create_stage_pos_marker()

        # --- cached parameters for efficient affine calculations ---
        self._affine_state = AffineState(self._mmc)

        # toolbar and actions
        self._toolbar = tb = StageExplorerToolbar()
        # (also add the actions from the ROI manager)
        self._toolbar.insertActions(
            tb.delete_rois_action, self.roi_manager.mode_actions.actions()
        )
        # add stage pos label to the toolbar
        self._stage_pos_label = QLabel()
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toolbar.addWidget(spacer)
        self._toolbar.addWidget(self._stage_pos_label)

        # connect actions to methods
        tb.clear_action.triggered.connect(self._on_clear_action)
        tb.zoom_to_fit_action.triggered.connect(self._on_zoom_to_fit_action)
        tb.auto_zoom_to_fit_action.triggered.connect(self._on_auto_zoom_to_fit_action)
        tb.snap_action.triggered.connect(self._on_snap_action)
        tb.poll_stage_action.triggered.connect(self._on_poll_stage_action)
        tb.show_grid_action.triggered.connect(self._on_show_grid_action)
        tb.blend_action.triggered.connect(self._on_blend_action)
        tb.delete_rois_action.triggered.connect(self.roi_manager.clear)
        tb.scan_action.triggered.connect(self._dispatch_scan_action)
        tb.stop_scan_action.triggered.connect(self._on_stop_scan_action)
        tb.marker_mode_action_group.triggered.connect(self._update_marker_mode)
        tb.scan_menu.valueChanged.connect(self._on_scan_options_changed)

        self._contrast_slider = ContrastSlider(self)
        self._contrast_slider.setVisible(False)
        self._contrast_slider.valueChanged.connect(self._on_contrast_slider_changed)

        # main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._toolbar, 0)
        main_layout.addWidget(self._stage_viewer, 1)
        main_layout.addWidget(self._contrast_slider, 0)

        # connections core events
        self._mmc.events.systemConfigurationLoaded.connect(self._on_sys_config_loaded)
        self._mmc.events.imageSnapped.connect(self._on_image_snapped)
        self.mda_frame_updates_enabled = True
        self._mmc.mda.events.sequenceFinished.connect(self._on_sequence_finished)
        self._mmc.events.pixelSizeChanged.connect(self._on_pixel_size_changed)
        self._mmc.events.pixelSizeAffineChanged.connect(
            self._on_pixel_size_affine_changed
        )

        # connections vispy events
        self._stage_viewer.canvas.events.mouse_double_click.connect(
            self._on_mouse_double_click
        )

        # initial setup
        self._on_roi_changed()
        self._toolbar.snap_action.setChecked(self._snap_on_double_click)
        self._update_actions_enabled()
        if self._poll_stage_position:
            self._toolbar.poll_stage_action.trigger()
        self.zoom_to_fit()

        self._send_to_mda_action = self._toolbar.addAction(
            QIconifyIcon("mdi:send", color=GRAY),
            "Send to MDA",
        )
        # move the action right after scan_action
        actions = self._toolbar.actions()
        idx = actions.index(
            self._toolbar.scan_action  # pyright: ignore[reportArgumentType]
        )
        self._toolbar.insertAction(
            actions[idx + 1],
            self._send_to_mda_action,  # pyright: ignore[reportArgumentType]
        )
        send_menu = QMenu(self)
        self._send_mode_group = QActionGroup(send_menu)
        self._send_mode_group.setExclusive(True)
        for label in ("ROIs with Grid Sub-Sequence", "List of Single Positions"):
            act = self._send_mode_group.addAction(label)
            act.setCheckable(True)
        self._send_mode_group.actions()[0].setChecked(True)
        send_menu.addActions(self._send_mode_group.actions())
        self._send_to_mda_action.triggered.connect(self._on_send_to_mda)
        btn = self._toolbar.widgetForAction(
            self._send_to_mda_action  # pyright: ignore[reportArgumentType]
        )
        if isinstance(btn, QToolButton):
            btn.setMenu(send_menu)
            btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        self.mda_frame_updates_enabled = False
        self._finish_pending_snap()
        self._stop_poller()
        super().closeEvent(a0)

    @property
    def mda_frame_updates_enabled(self) -> bool:
        """Whether this explorer receives frames from the global MDA runner."""
        return self._mda_frame_updates_enabled

    @mda_frame_updates_enabled.setter
    def mda_frame_updates_enabled(self, value: bool) -> None:
        """Idempotently connect or disconnect the global MDA frame stream."""
        value = bool(value)
        if value == self._mda_frame_updates_enabled:
            return
        signal = self._mmc.mda.events.frameReady
        if value:
            signal.connect(self._on_frame_ready)
        else:
            try:
                signal.disconnect(self._on_frame_ready)
            except (RuntimeError, TypeError):
                pass
        self._mda_frame_updates_enabled = value

    @property
    def blending_enabled(self) -> bool:
        """Whether incoming images are combined into a feathered mosaic."""
        return self._stage_viewer.blending_enabled

    @blending_enabled.setter
    def blending_enabled(self, value: bool) -> None:
        """Enable or disable normalized feather blending."""
        enabled = bool(value)
        self._stage_viewer.set_blending_enabled(enabled)
        self._toolbar.blend_action.setChecked(enabled)

    @property
    def mosaic_tile_count(self) -> int:
        """The number of unique spatial tiles in the current mosaic."""
        return self._stage_viewer.mosaic_tile_count

    @property
    def mosaic_chunk_count(self) -> int:
        """The number of rendered chunks in the current mosaic."""
        return self._stage_viewer.mosaic_chunk_count

    def set_blending_overlap(self, overlap: float | tuple[float, float]) -> None:
        """Set world-X and world-Y overlap percentages used for feathering."""
        self._stage_viewer.set_blending_overlap(overlap)

    def begin_blending_session(self, key: Hashable | None = None) -> None:
        """Begin or continue a mosaic session for compatible acquisition settings.

        Raises
        ------
        TypeError
            If the session key is not hashable.
        """
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError("Blending session keys must be hashable") from exc
        self._stage_viewer.begin_blending_session(key)

    def camera_fov_size_um(self) -> tuple[float, float]:
        """Return the axis-aligned camera footprint in stage-world micrometers.

        Returns
        -------
        tuple[float, float]
            Camera footprint along the stage X and Y axes.
        """
        return self.pixel_rect_size_um(
            self._mmc.getImageWidth(), self._mmc.getImageHeight()
        )

    def pixel_rect_size_um(
        self, width_px: float, height_px: float
    ) -> tuple[float, float]:
        """Transform a pixel-aligned rectangle into stage-world axis extents.

        Returns
        -------
        tuple[float, float]
            Axis-aligned physical width and height in micrometers.
        """
        image_width = float(width_px)
        image_height = float(height_px)
        linear = np.asarray(self._affine_state.system_affine, dtype=float)[:2, :2]
        world_width = abs(linear[0, 0]) * image_width + abs(linear[0, 1]) * image_height
        world_height = (
            abs(linear[1, 0]) * image_width + abs(linear[1, 1]) * image_height
        )
        return float(world_width), float(world_height)

    def set_mda_capture_active(self, active: bool) -> None:
        """Mark whether the current MDA belongs to this explorer."""
        self._our_mda_running = bool(active)
        if active:
            self.mda_frame_updates_enabled = True

    def set_display_fov_size_um(self, value: tuple[float, float] | None) -> None:
        """Set an optional acquisition footprint used by ROI planning and display.

        Raises
        ------
        ValueError
            If either footprint dimension is non-positive.
        """
        if value is None:
            self._display_fov_um = None
        else:
            width, height = map(float, value)
            if width <= 0 or height <= 0:
                raise ValueError("FOV dimensions must be positive")
            self._display_fov_um = (width, height)
        self._refresh_fov_display()

    def set_scan_handler(self, handler: Callable[[StageExplorer], None] | None) -> None:
        """Set an optional application scan handler without replacing widget methods."""
        self._scan_handler = handler

    def set_snap_lifecycle(
        self,
        *,
        prepare: Callable[[], None] | None = None,
        cleanup: Callable[[], None] | None = None,
    ) -> None:
        """Set synchronous hooks around double-click snaps.

        ``prepare`` runs before stage motion is requested. ``cleanup`` runs after
        the move accumulator has attempted the snap, including failure paths.
        """
        self._snap_prepare = prepare
        self._snap_cleanup = cleanup

    def set_roi_position_transform(
        self,
        transform: Callable[[useq.AbsolutePosition, ROI], useq.AbsolutePosition] | None,
    ) -> None:
        """Set an optional per-ROI transform used by position export."""
        self._roi_position_transform = transform

    def positions_from_rois(
        self,
        rois: Iterable[ROI] | None = None,
        *,
        flatten: bool = False,
        preserve_regions: bool = True,
        z_pos: float | None = None,
        transform: Callable[[useq.AbsolutePosition, ROI], useq.AbsolutePosition]
        | None = None,
    ) -> list[useq.AbsolutePosition]:
        """Build positions from the current ROI planning state.

        This is the shared conversion used by applications that need more than
        the default Stage Explorer scan action.  It preserves single-FOV regions
        when requested and provides a domain-neutral transform hook for per-ROI
        Z models or metadata.

        Returns
        -------
        list[useq.AbsolutePosition]
            Positions or nested region plans transformed for each ROI.
        """
        selected = list(rois) if rois is not None else self.roi_manager.all_rois()
        overlap = self.roi_manager.scan_overlap
        mode = self.roi_manager.scan_mode
        fov_w, fov_h = self._fov_w_h()
        positions: list[useq.AbsolutePosition] = []

        position_transform = transform or self._roi_position_transform

        def _apply(position: useq.AbsolutePosition, roi: ROI) -> useq.AbsolutePosition:
            return (
                position_transform(position, roi)
                if position_transform is not None
                else position
            )

        for roi_index, roi in enumerate(selected):
            position = roi.create_useq_position(
                fov_w,
                fov_h,
                z_pos=z_pos,
                overlap=overlap,
                mode=mode,
            )
            grid_plan = position.sequence.grid_plan if position.sequence else None
            if grid_plan is not None:
                if flatten:
                    base_name = position.name or f"pos_{roi_index:04d}"
                    for grid_index, grid_position in enumerate(grid_plan):
                        flat = useq.AbsolutePosition(
                            x=grid_position.x,
                            y=grid_position.y,
                            z=z_pos,
                            name=f"{base_name}_{grid_index:04d}",
                        )
                        positions.append(_apply(flat, roi))
                else:
                    positions.append(_apply(position, roi))
                continue

            center_x, center_y = roi.center()
            centered = position.model_copy(update={"x": center_x, "y": center_y})
            if flatten or not preserve_regions:
                positions.append(_apply(centered, roi))
                continue

            left, top, right, bottom = roi.bbox()
            overlap_xy = overlap if isinstance(overlap, tuple) else (overlap, overlap)
            if isinstance(roi, RectangleROI):
                region_plan: useq.GridFromEdges | useq.GridFromPolygon
                region_plan = useq.GridFromEdges(
                    top=top,
                    bottom=bottom,
                    left=left,
                    right=right,
                    fov_width=fov_w,
                    fov_height=fov_h,
                    overlap=overlap_xy,
                    mode=mode,
                )
            else:
                region_plan = useq.GridFromPolygon(
                    vertices=list(roi.vertices),
                    fov_width=fov_w,
                    fov_height=fov_h,
                    overlap=overlap_xy,
                    mode=mode,
                )
            nested = position.model_copy(
                update={"sequence": useq.MDASequence(grid_plan=region_plan)}
            )
            positions.append(_apply(nested, roi))

        return positions

    @Slot()
    def _on_roi_changed(self) -> None:
        """Update the ROI manager when a new ROI is set."""
        img_w = self._mmc.getImageWidth()
        img_h = self._mmc.getImageHeight()
        if not img_w or not img_h:
            return
        # by default, vispy add the images from the bottom-left corner. We need to
        # translate by -w/2 and -h/2 so the position corresponds to the center of the
        # images. In addition, this makes sure the rotation (if any) is applied around
        # the center of the image.
        self._half_img_shift = np.eye(4)
        self._half_img_shift[0:2, 3] = (-img_w / 2, -img_h / 2)

        self._refresh_fov_display()

    def _refresh_fov_display(self) -> None:
        """Refresh ROI cells and the position marker from the active footprint."""
        if not self._mmc.getImageWidth() or not self._mmc.getImageHeight():
            return
        footprint = self._fov_w_h()
        self.roi_manager.update_fovs(footprint)

        marker = self._stage_pos_marker
        if marker is None:
            return
        if self._display_fov_um is None:
            marker.set_rect_size(self._mmc.getImageWidth(), self._mmc.getImageHeight())
            return

        linear = np.abs(
            np.asarray(self._affine_state.system_affine, dtype=float)[:2, :2]
        )
        try:
            local_size = np.linalg.solve(linear, np.asarray(footprint, dtype=float))
        except np.linalg.LinAlgError:
            marker.set_rect_size(self._mmc.getImageWidth(), self._mmc.getImageHeight())
            return
        if np.all(np.isfinite(local_size)) and np.all(local_size > 0):
            marker.set_rect_size(float(local_size[0]), float(local_size[1]))

    @Slot()
    def _dispatch_scan_action(self) -> None:
        """Dispatch scanning through an application handler or the normal method."""
        if self._scan_handler is not None:
            self._scan_handler(self)
        else:
            self._on_scan_action()

    @Slot(object)
    def _on_scan_options_changed(self, value: tuple[float, OrderMode]) -> None:
        """Update scan settings on the ROI manager so visuals refresh."""
        overlap, mode = value
        self.roi_manager.set_scan_options(overlap, mode)
        self.set_blending_overlap(overlap)

    def _fov_w_h(self) -> tuple[float, float]:
        """Return the field of view width and height.

        Returns
        -------
        tuple[float, float]
            Active planning footprint in stage-world micrometers.
        """
        return self._display_fov_um or self.camera_fov_size_um()

    def _set_stage_controller(self) -> None:
        previous = self._stage_controller
        if previous is not None:
            try:
                previous.moveFinished.disconnect(self._finish_pending_snap)
            except (RuntimeError, TypeError):
                pass
        self._stage_controller = None
        if xy_dev := self._mmc.getXYStageDevice():
            self._stage_controller = QStageMoveAccumulator.for_device(xy_dev, self._mmc)
            self._stage_controller.moveFinished.connect(self._finish_pending_snap)

    @Slot(float)
    def _on_pixel_size_changed(self, value: float) -> None:
        """Refresh the affine state when pixel size changes."""
        self._affine_state.refresh()
        self._on_roi_changed()

    @Slot()
    def _on_pixel_size_affine_changed(self) -> None:
        """Refresh the affine state when pixel size affine changes."""
        self._affine_state.refresh()
        self._on_roi_changed()

    @Slot(object)
    def _on_mouse_double_click(self, event: MouseEvent) -> None:
        """Move the stage to the clicked position."""
        if not self._mmc.getXYStageDevice() or self._stage_controller is None:
            return
        if self.roi_manager.mode == "create-poly":
            return

        # map the clicked canvas position to the stage position
        x, y = self._stage_viewer.canvas_to_world(event.pos)
        self._finish_pending_snap()
        should_snap = self._snap_on_double_click and not self._mmc.isSequenceRunning()
        if should_snap and self._snap_prepare is not None:
            try:
                self._snap_prepare()
            except Exception as exc:
                should_snap = False
                self.interactionError.emit(exc)
                if self._snap_cleanup is not None:
                    try:
                        self._snap_cleanup()
                    except Exception as cleanup_exc:
                        self.interactionError.emit(cleanup_exc)

        self._pending_snap_cleanup = self._snap_cleanup if should_snap else None
        self._stage_controller.snap_on_finish = should_snap
        self._stage_controller.move_absolute((x, y))

        # update the stage position label
        self._stage_pos_label.setText(f"X: {x:.2f} µm  Y: {y:.2f} µm")

    @Slot()
    def _on_image_snapped(self) -> None:
        """Add the snapped image to the scene."""
        if self._mmc.mda.is_running() or self._mmc.isSequenceRunning():
            return
        # get the snapped image
        img = self._mmc.getImage()
        # get the current stage position
        x, y = self._mmc.getXYPosition()
        self._add_image_and_update_widget(img, x, y)

    @Slot(object, object, object)
    def _on_frame_ready(
        self,
        image: np.ndarray,
        event: useq.MDAEvent,
        metadata: dict | None = None,
    ) -> None:
        """Add the image to the scene when frameReady event is emitted."""
        # TODO: better handle c and z (e.g. multi-channels?, max projection?)
        if not self._mda_frame_updates_enabled:
            return
        if event.x_pos is None or event.y_pos is None:
            return
        self._add_image_and_update_widget(image, event.x_pos, event.y_pos)

    def _finish_pending_snap(self) -> None:
        callback = self._pending_snap_cleanup
        self._pending_snap_cleanup = None
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            self.interactionError.emit(exc)

    @Slot(bool)
    def _on_blend_action(self, checked: bool) -> None:
        """Enable or disable normalized overlap blending."""
        self.blending_enabled = checked

    def _on_scan_action(self) -> None:
        """Override to pass output='memory' so the ndv viewer gets data."""
        if not (active_rois := self.roi_manager.selected_rois()):
            return
        active_roi = active_rois[0]

        overlap, mode = self._toolbar.scan_menu.value()
        if plan := active_roi.create_grid_plan(*self._fov_w_h(), overlap, mode):
            seq = useq.MDASequence(grid_plan=plan)
            if not self._mmc.mda.is_running():
                self._our_mda_running = True
                self._mmc.run_mda(seq, output="memory")

    def _on_send_to_mda(self) -> None:
        checked = self._send_mode_group.checkedAction()
        flatten = checked is not None and checked.text() == "List of Single Positions"
        z_pos = self._mmc.getZPosition()
        positions = self.positions_from_rois(
            flatten=flatten,
            preserve_regions=not flatten,
            z_pos=z_pos,
        )
        if not positions:
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Send to MDA")
        msg.setText("Replace existing stage positions or add to them?")
        replace_btn = msg.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Add", QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is cancel_btn or clicked is None:
            return

        self.sendToMDARequested.emit(positions, clicked is replace_btn)
