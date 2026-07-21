"""Launch the OPM Micro-Manager GUI using an application controller."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import useq
from pymmcore_gui import MicroManagerGUI, WidgetAction, create_mmgui
from pymmcore_gui._qt.QtAds import DockWidgetArea
from pymmcore_gui._qt.QtCore import QTimer
from pymmcore_gui._qt.QtWidgets import QApplication, QMessageBox, QTabBar
from pymmcore_gui.actions import ActionInfo, CoreAction, QCoreAction, WidgetActionInfo
from pymmcore_gui.actions import widget_actions as pymmcore_widget_actions
from pymmcore_plus import CMMCorePlus

from opm_v2._update_config_widget import OPMSettingsV2
from opm_v2.engine.debug_printing import debug, info, warning
from opm_v2.engine.opm_custom_events import STAGE_MOVE_SPEED_METADATA_KEY
from opm_v2.engine.opm_engine import OPMEngineV2
from opm_v2.engine.setup_events import (
    OPMEventBuilder,
    setup_optimizenow,
)
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.ElveFlow import OB1Controller
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.PicardShutter import PicardShutter

DEBUGGING = True
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "opm_config.json"
MIN_PROJECTION_EXPOSURE = 50  # ms
DEFUALT_PROJECTION_EXPOSURE = 150
OPM_WIDGET_KEY = "opm.settings"


def _disconnect_system_configuration_callback(
    mmc: CMMCorePlus, callback
) -> None:
    """Disconnect an action callback even if Qt has deleted the core signaler."""
    try:
        mmc.events.systemConfigurationLoaded.disconnect(callback)
    except (RuntimeError, TypeError):
        # During QApplication teardown the C++ QCoreSignaler may be destroyed
        # before its QAction children.  There is then no live signal to disconnect.
        pass


def _initialize_snap_action_safely(action: QCoreAction) -> None:
    """Initialize the upstream Snap action with deletion-safe cleanup."""
    mmc = action.mmc

    def _on_load() -> None:
        action.setEnabled(bool(mmc.getCameraDevice()))

    mmc.events.systemConfigurationLoaded.connect(_on_load)
    action.destroyed.connect(
        lambda: _disconnect_system_configuration_callback(mmc, _on_load)
    )
    _on_load()


def _initialize_live_action_safely(action: QCoreAction) -> None:
    """Initialize the upstream Live action with deletion-safe cleanup."""
    mmc = action.mmc

    def _on_load() -> None:
        action.setEnabled(bool(mmc.getCameraDevice()))

    mmc.events.systemConfigurationLoaded.connect(_on_load)
    action.destroyed.connect(
        lambda: _disconnect_system_configuration_callback(mmc, _on_load)
    )

    def _on_change() -> None:
        action.setChecked(mmc.isSequenceRunning())

    mmc.events.sequenceAcquisitionStarted.connect(_on_change)
    mmc.events.continuousSequenceAcquisitionStarted.connect(_on_change)
    mmc.events.sequenceAcquisitionStopped.connect(_on_change)
    _on_load()


def _install_safe_core_action_initializers() -> None:
    """Replace two unsafe pymmcore-gui teardown hooks before actions are created."""
    ActionInfo.for_key(CoreAction.SNAP).on_created = _initialize_snap_action_safely
    ActionInfo.for_key(CoreAction.TOGGLE_LIVE).on_created = (
        _initialize_live_action_safely
    )


def _stage_explorer_world_fov_w_h(stage_explorer) -> tuple[float, float]:
    """Return camera FOV extents in physical stage-world X and Y.

    Returns
    -------
    tuple[float, float]
        Axis-aligned physical stage X and Y extents in micrometers.
    """
    image_width = float(stage_explorer._mmc.getImageWidth())
    image_height = float(stage_explorer._mmc.getImageHeight())
    linear = np.asarray(stage_explorer._affine_state.system_affine)[:2, :2]
    world_width = abs(linear[0, 0]) * image_width + abs(linear[0, 1]) * image_height
    world_height = abs(linear[1, 0]) * image_width + abs(linear[1, 1]) * image_height
    return float(world_width), float(world_height)


def _update_stage_explorer_transformed_fov(stage_explorer) -> None:
    """Update ROI tiling after the upstream camera/ROI bookkeeping runs."""
    stage_explorer._opm_original_on_roi_changed()
    stage_explorer.roi_manager.update_fovs(stage_explorer._fov_w_h())


def _refresh_stage_explorer_pixel_size(stage_explorer, value: float) -> None:
    """Refresh the Explorer affine and its physical FOV after a scale change."""
    stage_explorer._opm_original_on_pixel_size_changed(value)
    stage_explorer._on_roi_changed()


def _refresh_stage_explorer_pixel_affine(stage_explorer) -> None:
    """Refresh the Explorer affine and its physical FOV after an axis change."""
    stage_explorer._opm_original_on_pixel_size_affine_changed()
    stage_explorer._on_roi_changed()


def _stage_explorer_region_position(stage_explorer, roi) -> useq.AbsolutePosition:
    """Return one selected ROI as a physical Stage Explorer region.

    Returns
    -------
    useq.AbsolutePosition
        Position containing the selected rectangular ROI grid.

    Raises
    ------
    ValueError
        If a single-FOV ROI is not rectangular.
    """
    overlap, mode = stage_explorer._toolbar.scan_menu.value()
    fov_width, fov_height = stage_explorer._fov_w_h()
    z_position = stage_explorer._mmc.getZPosition()
    position = roi.create_useq_position(
        fov_width,
        fov_height,
        z_pos=z_position,
        overlap=overlap,
        mode=mode,
    )
    if position.sequence and position.sequence.grid_plan:
        return position

    left, top, right, bottom = roi.bbox()
    if type(roi).__name__ != "RectangleROI":
        raise ValueError("OPM Stage Explorer previews require a rectangular ROI.")
    grid_plan = useq.GridFromEdges(
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        fov_width=fov_width,
        fov_height=fov_height,
        overlap=(overlap, overlap),
        mode=mode,
    )
    return position.replace(sequence=useq.MDASequence(grid_plan=grid_plan))


def _stage_explorer_accelerated_speeds(mmc: CMMCorePlus) -> dict[str, float]:
    """Return four times the current XY speeds for an Explorer preview.

    Returns
    -------
    dict[str, float]
        Per-axis speed keys and accelerated speeds in millimeters per second.
    """
    xy_stage = mmc.getXYStageDevice()
    speed_keys = {
        "MotorSpeedX-S(mm/s)": "move_speed_x_mm_s",
        "MotorSpeedY-S(mm/s)": "move_speed_y_mm_s",
    }
    accelerated: dict[str, float] = {}
    for property_name, event_key in speed_keys.items():
        if xy_stage and mmc.hasProperty(xy_stage, property_name):
            accelerated[event_key] = 4.0 * float(
                mmc.getProperty(xy_stage, property_name)
            )

    return accelerated


def _stage_explorer_channel_preset(mmc: CMMCorePlus) -> tuple[str, str]:
    """Return the active preview preset without changing MM config state.

    The OPM hardware configuration defines a normal config group named
    ``Channel`` but does not necessarily designate it as MMCore's special
    channel group.  Prefer the designation when present, then fall back to the
    explicitly defined ``Channel`` group used by the Config Groups widget.

    Returns
    -------
    tuple[str, str]
        Config group name and the preset matching its current device state.

    Raises
    ------
    ValueError
        If no channel group exists or its current device state matches no preset.
    """
    available_groups = set(mmc.getAvailableConfigGroups())
    designated_group = mmc.getChannelGroup()
    if designated_group and designated_group in available_groups:
        channel_group = designated_group
    elif "Channel" in available_groups:
        channel_group = "Channel"
    else:
        raise ValueError(
            "No Micro-Manager channel config group is available for the preview"
        )

    channel_preset = mmc.getCurrentConfig(channel_group)
    if not channel_preset:
        raise ValueError(
            f"Select a preset in the Micro-Manager {channel_group!r} config group "
            "before scanning an ROI"
        )
    return channel_group, channel_preset


def _scan_stage_explorer_roi(stage_explorer) -> None:
    """Scan the selected ROI with the active Micro-Manager channel preset."""
    controller = getattr(stage_explorer, "_opm_controller", None) or getattr(
        stage_explorer.window(), "opm_controller", None
    )
    if controller is None:
        stage_explorer._opm_original_on_scan_action()
        return
    if stage_explorer._mmc.mda.is_running():
        return
    selected_rois = stage_explorer.roi_manager.selected_rois()
    if not selected_rois:
        return

    try:
        position = _stage_explorer_region_position(
            stage_explorer, selected_rois[0]
        )
    except ValueError as exc:
        controller.warning("STAGE EXPLORER PREVIEW", str(exc))
        return

    try:
        channel_group, channel_preset = _stage_explorer_channel_preset(
            controller.mmc
        )

        accelerated = _stage_explorer_accelerated_speeds(controller.mmc)
        metadata = {STAGE_MOVE_SPEED_METADATA_KEY: accelerated}
        sequence = useq.MDASequence(
            metadata=metadata,
            stage_positions=(position,),
        )
        controller.info(
            "STAGE EXPLORER PREVIEW",
            f"Channel preset: {channel_group}/{channel_preset}",
            f"Accelerated stage speeds: {accelerated}",
            "Output: preview only (not saved)",
        )
        controller.data_handler = None
        controller.suspend_live_preview_for_mda()
        controller.prepare_stage_explorer_preview()
        stage_explorer._our_mda_running = True
        controller._opm_mda_thread = controller.mmc.run_mda(
            sequence, output="memory"
        )
    except ValueError as exc:
        controller.warning("STAGE EXPLORER PREVIEW", str(exc))
    except Exception:
        controller.opm_nidaq.clear_tasks()
        controller.restore_live_preview_after_mda(force=True)
        raise


def _send_stage_explorer_rois_to_mda(stage_explorer) -> None:
    """Export Stage Explorer ROIs while preserving regions and literal centers."""
    checked = stage_explorer._send_mode_group.checkedAction()
    flatten = checked is not None and checked.text() == "List of Single Positions"
    overlap, mode = stage_explorer._toolbar.scan_menu.value()
    fov_w, fov_h = stage_explorer._fov_w_h()
    z_pos = stage_explorer._mmc.getZPosition()
    positions: list[useq.AbsolutePosition] = []

    roi_model = stage_explorer.roi_manager.roi_model
    for row in range(roi_model.rowCount()):
        roi = roi_model.index(row).internalPointer()
        position = roi.create_useq_position(
            fov_w,
            fov_h,
            z_pos=z_pos,
            overlap=overlap,
            mode=mode,
        )
        if position.sequence and position.sequence.grid_plan:
            if flatten:
                positions.extend(stage_explorer._flatten_to_single_positions([position]))
            else:
                positions.append(position)
            continue

        center_x, center_y = roi.center()
        if flatten:
            positions.append(position.replace(x=center_x, y=center_y))
            continue

        # ``ROI.create_grid_plan`` intentionally returns None when an ROI fits in
        # one camera FOV.  OPM stage mode still needs the ROI's physical bounds, so
        # retain a one-cell nested region instead of degrading it to a point.
        left, top, right, bottom = roi.bbox()
        if type(roi).__name__ == "RectangleROI":
            grid_plan = useq.GridFromEdges(
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                fov_width=fov_w,
                fov_height=fov_h,
                overlap=(overlap, overlap),
                mode=mode,
            )
        else:
            grid_plan = useq.GridFromPolygon(
                vertices=list(roi.vertices),
                fov_width=fov_w,
                fov_height=fov_h,
                overlap=(overlap, overlap),
                mode=mode,
            )
        positions.append(
            position.replace(sequence=useq.MDASequence(grid_plan=grid_plan))
        )

    if not positions:
        return

    message = QMessageBox(stage_explorer)
    message.setWindowTitle("Send to MDA")
    message.setText("Replace existing stage positions or add to them?")
    replace_button = message.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
    message.addButton("Add", QMessageBox.ButtonRole.AcceptRole)
    cancel_button = message.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    message.exec()
    clicked = message.clickedButton()
    if clicked is cancel_button or clicked is None:
        return
    stage_explorer.sendToMDARequested.emit(positions, clicked is replace_button)


def _connect_stage_explorer_to_mda(stage_explorer=None, mda_widget=None) -> None:
    """Connect ROI export to MDA and select one unambiguous spatial plan."""
    if stage_explorer is None or mda_widget is None:
        return
    stage_explorer._opm_controller = getattr(stage_explorer.window(), "opm_controller", None)

    def _on_send_to_mda(positions: list, clear: bool) -> None:
        if clear:
            mda_widget.stage_positions.setValue(positions)
        else:
            current = list(mda_widget.stage_positions.value() or [])
            mda_widget.stage_positions.setValue(current + positions)

        # Exported positions own the spatial plan.  Leaving the global Grid tab
        # active would make useq clear X/Y and would cause the OPM builders to ignore
        # the exported regions.
        mda_widget.tab_wdg.setChecked(mda_widget.grid_plan, False)
        mda_widget.tab_wdg.setChecked(mda_widget.stage_positions, True)

    stage_explorer.sendToMDARequested.connect(_on_send_to_mda)


def _install_stage_explorer_export_compatibility() -> None:
    """Complete Stage Explorer affine/FOV and export behavior for OPM use."""
    if getattr(pymmcore_widget_actions, "_opm_export_compatibility", False):
        return
    explorer_class = pymmcore_widget_actions._StageExplorer
    explorer_class._opm_original_on_roi_changed = explorer_class._on_roi_changed
    explorer_class._opm_original_on_pixel_size_changed = (
        explorer_class._on_pixel_size_changed
    )
    explorer_class._opm_original_on_pixel_size_affine_changed = (
        explorer_class._on_pixel_size_affine_changed
    )
    explorer_class._opm_original_on_scan_action = explorer_class._on_scan_action
    explorer_class._fov_w_h = _stage_explorer_world_fov_w_h
    explorer_class._on_roi_changed = _update_stage_explorer_transformed_fov
    explorer_class._on_pixel_size_changed = _refresh_stage_explorer_pixel_size
    explorer_class._on_pixel_size_affine_changed = (
        _refresh_stage_explorer_pixel_affine
    )
    explorer_class._on_scan_action = _scan_stage_explorer_roi
    pymmcore_widget_actions._setup_stage_mda_connections = (
        _connect_stage_explorer_to_mda
    )
    explorer_class._on_send_to_mda = _send_stage_explorer_rois_to_mda
    pymmcore_widget_actions._opm_export_compatibility = True


def _create_opm_settings_widget(parent: MicroManagerGUI) -> OPMSettingsV2:
    """Create the registered OPM widget for a Micro-Manager window.

    Parameters
    ----------
    parent : MicroManagerGUI
        Window that owns the widget and its OPM controller.

    Returns
    -------
    OPMSettingsV2
        OPM settings widget managed by pymmcore-gui.
    """
    controller: OPMAppController = parent.opm_controller
    return OPMSettingsV2(controller.config_path, parent=parent)


def _ensure_opm_widget_registered() -> None:
    """Register the OPM settings widget with pymmcore-gui exactly once."""
    _install_safe_core_action_initializers()
    _install_stage_explorer_export_compatibility()
    try:
        WidgetActionInfo.for_key(OPM_WIDGET_KEY)
    except KeyError:
        WidgetActionInfo(
            key=OPM_WIDGET_KEY,
            text="OPM Settings",
            create_widget=_create_opm_settings_widget,
            dock_area=DockWidgetArea.LeftDockWidgetArea,
        )


def enhance_main_window(main_window: MicroManagerGUI) -> OPMSettingsV2:
    """Create and attach the OPM dock using pymmcore-gui widget APIs.

    Parameters
    ----------
    main_window : MicroManagerGUI
        Main window receiving the OPM settings dock.

    Returns
    -------
    OPMSettingsV2
        Attached OPM settings widget.
    """
    _ensure_opm_widget_registered()
    widget = main_window.get_widget(OPM_WIDGET_KEY)
    opm_dock = main_window.get_dock_widget(OPM_WIDGET_KEY)
    try:
        mda_dock = main_window.get_dock_widget(WidgetAction.MDA_WIDGET)
    except Exception:
        mda_dock = None
    else:
        if opm_dock.dockAreaWidget() is not mda_dock.dockAreaWidget():
            main_window.dock_manager.addDockWidgetTabToArea(
                opm_dock, mda_dock.dockAreaWidget()
            )
        mda_dock.raise_()
    return widget


class ConfigStore:
    """Provide a reloadable wrapper around an OPM JSON configuration.

    Parameters
    ----------
    path : Path
        Path to the JSON configuration file.
    """

    def __init__(self, path: Path) -> None:
        """Create a config store and immediately load the JSON file.

        Parameters
        ----------
        path : Path
            Path to the JSON configuration file.
        """
        self.path = Path(path)
        self.data: dict = {}
        self.reload()

    def reload(self) -> dict:
        """Reload the JSON configuration in place.

        Returns
        -------
        dict
            Shared configuration dictionary updated from disk.
        """
        with open(self.path) as config_file:
            new_config = json.load(config_file)
        self.data.clear()
        self.data.update(new_config)
        return self.data

    def replace(self, config: dict) -> dict:
        """Replace the active configuration with an isolated GUI snapshot.

        Parameters
        ----------
        config : dict
            Configuration emitted by the OPM settings widget.

        Returns
        -------
        dict
            Shared controller dictionary updated in place.
        """
        self.data.clear()
        self.data.update(deepcopy(config))
        return self.data


class OPMAppController:
    """Manage MMGUI, OPM hardware, live-state updates, and MDA execution.

    Parameters
    ----------
    config_path : Path
        Path to the OPM JSON configuration file.
    """

    debug = staticmethod(partial(debug, enabled=DEBUGGING))
    info = staticmethod(info)
    warning = staticmethod(warning)

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        *,
        mmcore=None,
        mm_config: Path | str | Literal[False] | None = None,
        exec_app: bool = True,
        simulate_hardware: bool | None = None,
    ) -> None:
        """Load configuration and prepare placeholders for app-owned objects.

        Heavy GUI and hardware objects are initialized later by ``run()`` so the
        controller can be constructed without immediately touching hardware.

        Parameters
        ----------
        config_path : Path
            Path to the OPM JSON configuration file.
        mmcore : CMMCorePlus or None
            Existing core instance to reuse, or ``None`` to use MMGUI's instance.
        mm_config : Path, str, False, or None
            Micro-Manager configuration passed to ``create_mmgui``. ``False``
            preserves the state of a supplied core.
        exec_app : bool
            Whether to enter the Qt application event loop.
        simulate_hardware : bool or None
            Override for external OPM hardware simulation.
        """
        self.config_store = ConfigStore(config_path)
        self.config = self.config_store.data
        self.config_path = self.config_store.path
        self._provided_mmcore = mmcore
        self._mm_config = mm_config
        self._exec_app = exec_app
        configured_simulation = bool(
            self.config.get("OPM", {}).get("simulate_hardware", False)
        )
        self.simulate_hardware = (
            configured_simulation
            if simulate_hardware is None
            else bool(simulate_hardware)
        )

        self.win = None
        self.mmc = None
        self.mda_widget = None
        self.opm_settings_widget = None
        self.opm_ao_mirror = None
        self.opm_nidaq = None
        self.opm_picard_shutter = None
        self.ob1_controller = None
        self.opm_engine = None
        self.data_handler = None
        self._picard_state_timer = None
        self._mda_preview_timer = None
        self._opm_mda_thread = None
        self._suspended_live_preview = None
        self._live_action_was_enabled = None
        self.bootstrap_complete = False

    def run(self) -> MicroManagerGUI:
        """Create the GUI, schedule OPM bootstrap, and optionally start Qt.

        Returns
        -------
        MicroManagerGUI
            Configured Micro-Manager main window.
        """
        self.create_gui()
        QTimer.singleShot(0, partial(self._schedule_extension_step_two))
        if self._exec_app and (app := QApplication.instance()) is not None:
            app.exec()
        return self.win

    def _schedule_extension_step_two(self) -> None:
        """Wait one more event-loop turn for pymmcore-gui layout restoration."""
        QTimer.singleShot(0, self.finish_bootstrap)

    def finish_bootstrap(self) -> None:
        """Attach OPM UI and hardware after pymmcore-gui restores its layout."""
        self.initialize_hardware()
        self.opm_settings_widget = enhance_main_window(self.win)
        self.opm_engine = OPMEngineV2(
            self.mmc,
            self.config_path,
            simulate_hardware=self.simulate_hardware,
            config=self.config,
            post_teardown=self.prepare_live_preview_after_mda,
        )
        self.mmc.register_mda_engine(self.opm_engine)
        self.connect_signals()
        self.bootstrap_complete = True

    def create_gui(self) -> None:
        """Create Micro-Manager GUI using its public bootstrap and widget APIs."""
        _ensure_opm_widget_registered()
        mm_config = self._mm_config
        if mm_config is None:
            mm_config = (
                False
                if self._provided_mmcore is not None
                else Path(self.config["OPM"]["mm_config_path"])
            )
        self.win = create_mmgui(
            mm_config=mm_config,
            mmcore=self._provided_mmcore,
            exec_app=False,
        )
        self.mmc = self.win.mmcore
        self.win.opm_controller = self

        self.mda_widget = self.win.get_widget(WidgetAction.MDA_WIDGET)
        self.disable_native_mda_channels()
        self.configure_transposed_mda_grid_bounds()

        if DEBUGGING:
            self.mmc.enableDebugLog(True)

        self.debug(
            "GUI CREATED",
            f"MM config: {mm_config}",
        )

    def disable_native_mda_channels(self) -> None:
        """Disable native MDA channels because OPM settings own channel setup."""
        tabs = self.mda_widget.tab_wdg
        channels = tabs.channels
        channel_index = tabs.indexOf(channels)
        explanation = "Configure acquisition channels in OPM Settings."

        tabs.setChecked(channels, False)
        channels.setEnabled(False)
        tabs.setTabText(channel_index, "Channels (use OPM Settings)")
        tabs.setTabToolTip(channel_index, explanation)
        tabs.setTabEnabled(channel_index, False)

        channel_checkbox = tabs.tabBar().tabButton(
            channel_index, QTabBar.ButtonPosition.LeftSide
        )
        if channel_checkbox is not None:
            channel_checkbox.setChecked(False)
            channel_checkbox.setEnabled(False)
            channel_checkbox.setToolTip(explanation)

    def configure_transposed_mda_grid_bounds(self) -> None:
        """Map camera-oriented grid controls onto the physical XY stage axes."""
        grid_widget = self.mda_widget.grid_plan
        bounds_control = getattr(grid_widget, "_core_xy_bounds", None)
        if bounds_control is None:
            self.warning(
                "MDA GRID AXIS MAPPING",
                "Could not find the native XY bounds control.",
            )
            return

        button_mapping = {
            "btn_top": (True, None),
            "btn_bottom": (False, None),
            "btn_left": (None, True),
            "btn_right": (None, False),
            "btn_top_left": (True, True),
            "btn_top_right": (True, False),
            "btn_bottom_left": (False, True),
            "btn_bottom_right": (False, False),
        }
        for button_name, (top, left) in button_mapping.items():
            button = getattr(bounds_control, button_name)
            try:
                button.clicked.disconnect()
            except (RuntimeError, TypeError):
                pass
            button.clicked.connect(
                partial(
                    self._mark_or_visit_transposed_grid_bound,
                    bounds_control,
                    top=top,
                    left=left,
                )
            )

        self.debug(
            "MDA GRID AXIS MAPPING",
            "Top/bottom records physical X (ASI fast axis).",
            "Left/right records physical Y (tile axis).",
        )

    def _mark_or_visit_transposed_grid_bound(
        self,
        bounds_control,
        _checked: bool = False,
        *,
        top: bool | None = None,
        left: bool | None = None,
    ) -> None:
        """Mark or visit one camera-oriented bound using swapped stage axes."""
        xy_stage = self.mmc.getXYStageDevice()
        current_x = float(self.mmc.getXPosition(xy_stage))
        current_y = float(self.mmc.getYPosition(xy_stage))

        if bounds_control.go_middle.isChecked():
            target_x = (
                current_x
                if top is None
                else float(
                    bounds_control.top.value()
                    if top
                    else bounds_control.bottom.value()
                )
            )
            target_y = (
                current_y
                if left is None
                else float(
                    bounds_control.left.value()
                    if left
                    else bounds_control.right.value()
                )
            )
            self.mmc.setXYPosition(xy_stage, target_x, target_y)
            self.mmc.waitForDevice(xy_stage)
            return

        if top is not None:
            (bounds_control.top if top else bounds_control.bottom).setValue(current_x)
        if left is not None:
            (bounds_control.left if left else bounds_control.right).setValue(current_y)

        self.debug(
            "MDA GRID BOUND UPDATED",
            f"top / bottom (physical X): "
            f"{bounds_control.top.value()} / {bounds_control.bottom.value()}",
            f"left / right (physical Y): "
            f"{bounds_control.left.value()} / {bounds_control.right.value()}",
        )

    def initialize_hardware(self) -> None:
        """Instantiate OPM hardware controllers and put them in startup states."""
        # ------------------------------------------------------------------------------#
        # AO mirror
        # ------------------------------------------------------------------------------#

        self.opm_ao_mirror = AOMirror(
            wfc_config_file_path=Path(self.config["AOMirror"]["wfc_config_path"]),
            haso_config_file_path=Path(self.config["AOMirror"]["haso_config_path"]),
            interaction_matrix_file_path=Path(
                self.config["AOMirror"]["wfc_correction_path"]
            ),
            system_flat_file_path=Path(self.config["AOMirror"]["wfc_flat_path"]),
            n_modes=32,
            n_positions=1,
            modes_to_ignore=[],
            simulate=self.simulate_hardware,
        )
        self.opm_ao_mirror.apply_system_flat_voltage()

        self.debug(
            "AO MIRROR INITIALIZED",
            f"System flat: {self.config['AOMirror']['wfc_flat_path']}",
            "Startup state: system flat voltage applied",
        )

        # ------------------------------------------------------------------------------#
        # NIDAQ
        # ------------------------------------------------------------------------------#

        self.opm_nidaq = OPMNIDAQ(
            name=str(self.config["NIDAQ"]["name"]),
            scan_type=str(self.config["NIDAQ"]["scan_type"]),
            exposure_ms=float(self.config["Camera"]["exposure_ms"]),
            laser_blanking=bool(self.config["acq_config"]["DAQ"]["laser_blanking"]),
            image_mirror_calibration=float(
                str(self.config["NIDAQ"]["image_mirror_calibration"])
            ),
            projection_mirror_calibration=float(
                str(self.config["NIDAQ"]["projection_mirror_calibration"])
            ),
            image_mirror_neutral_v=float(
                str(self.config["NIDAQ"]["image_mirror_neutral_v"])
            ),
            projection_mirror_neutral_v=float(
                str(self.config["NIDAQ"]["projection_mirror_neutral_v"])
            ),
            image_mirror_step_um=float(
                str(self.config["NIDAQ"]["image_mirror_step_um"])
            ),
            verbose=bool(self.config["NIDAQ"]["verbose"]),
            simulate=self.simulate_hardware,
        )
        self.opm_nidaq.reset()

        self.debug(
            "NIDAQ INITIALIZED",
            f"Name: {self.config['NIDAQ']['name']}",
        )

        # ------------------------------------------------------------------------------#
        # O2-O3 laser shutter
        # ------------------------------------------------------------------------------#

        self.opm_picard_shutter = PicardShutter(
            int(self.config["O2O3-autofocus"]["shutter_id"]),
            verbose=True,
            simulate=self.simulate_hardware,
        )
        self.opm_picard_shutter.closeShutter()

        self.debug(
            "O2-O3 SHUTTER INITIALIZED",
            f"Shutter id: {self.config['O2O3-autofocus']['shutter_id']}",
            "Startup state: closed",
        )

        # ------------------------------------------------------------------------------#
        # Fluidics
        # ------------------------------------------------------------------------------#

        if bool(self.config["OPM"].get("fluidics_enabled", False)):
            self.ob1_controller = OB1Controller(
                port=str(self.config["OB1"]["port"]),
                to_OB1_pin=int(self.config["OB1"]["to_OB1_pin"]),
                from_OB1_pin=int(self.config["OB1"]["from_OB1_pin"]),
                simulate=self.simulate_hardware,
            )
            self.debug("FLUIDICS INITIALIZED", "OB1 controller enabled")
        else:
            self.debug("FLUIDICS SKIPPED", "config['OPM']['fluidics_enabled'] is false")

    def connect_signals(self) -> None:
        """Connect Micro-Manager and OPM widget signals to controller methods."""
        self.opm_settings_widget.settings_changed.connect(self.update_config_snapshot)
        self.opm_settings_widget.run_requested.connect(self.run_opm_acquisition)
        self.opm_settings_widget.picard_shutter_requested.connect(
            self.set_picard_shutter_open
        )
        self.sync_picard_shutter_button()
        self._picard_state_timer = QTimer(self.win)
        self._picard_state_timer.setInterval(100)
        self._picard_state_timer.timeout.connect(self.sync_picard_shutter_button)
        self._picard_state_timer.start()

        # The OPM writer owns its live data stream rather than registering as
        # pymmcore-plus's native sink.  Attach that stream to the MDA viewer as
        # soon as its first frame establishes the array shape.
        self._mda_preview_timer = QTimer(self.win)
        self._mda_preview_timer.setInterval(50)
        self._mda_preview_timer.timeout.connect(self.sync_opm_mda_preview)
        self._mda_preview_timer.timeout.connect(self.sync_live_button_state)
        self._mda_preview_timer.timeout.connect(self.restore_live_preview_after_mda)
        self._mda_preview_timer.start()

        # Changes to the mm config
        self.mmc.events.configSet.connect(self.update_live_state)
        self.update_live_state()

        # Changes to the selected AO mirror state
        self.opm_settings_widget.widgets["AO"][
            "mirror_state"
        ].currentIndexChanged.connect(self.update_ao_mirror_state)

        # Connect live preview
        self.mmc.events.continuousSequenceAcquisitionStarting.connect(
            self.setup_preview_mode_callback
        )

        self.debug(
            "SIGNALS CONNECTED",
            "OPM settings -> in-memory acquisition snapshot",
            "OPM run button -> OPM acquisition",
            "Picard shutter button -> O2-O3 shutter state",
            "mmc.events.configSet -> update_live_state",
            "AO mirror_state currentIndexChanged -> update_ao_mirror_state",
            "continuousSequenceAcquisitionStarting -> setup_preview_mode_callback",
        )

    def sync_opm_mda_preview(self) -> None:
        """Attach the active OPM writer stream to the native MDA viewer."""
        handler = self.data_handler
        if handler is None:
            return

        viewers_manager = getattr(self.win, "_viewers_manager", None)
        viewer = getattr(viewers_manager, "_active_mda_viewer", None)
        if viewer is None or viewer.data_wrapper is not None:
            return

        view = handler.get_view()
        if view is None:
            return

        viewer.data = view
        wrapper = viewer.data_wrapper
        if hasattr(view, "coords_changed") and hasattr(wrapper, "dims_changed"):
            view.coords_changed.connect(wrapper.dims_changed)
        self.info(
            "OPM MDA PREVIEW READY",
            "Displaying the complete dimensional OPM acquisition",
        )

    def sync_live_button_state(self) -> None:
        """Clear a stale live-button highlight after an MDA camera sequence."""
        if self.mmc.mda.is_running() or self.mmc.isSequenceRunning():
            return

        live_action = self.win.get_action(CoreAction.TOGGLE_LIVE)
        if live_action.isChecked():
            live_action.setChecked(False)

    def suspend_live_preview_for_mda(self) -> None:
        """Prevent the live preview timer from consuming MDA camera frames."""
        if self.mmc.isSequenceRunning():
            self.mmc.stopSequenceAcquisition()

        live_action = self.win.get_action(CoreAction.TOGGLE_LIVE)
        self._live_action_was_enabled = live_action.isEnabled()
        live_action.setChecked(False)
        live_action.setEnabled(False)
        viewers_manager = getattr(self.win, "_viewers_manager", None)
        preview_dock = getattr(viewers_manager, "_current_image_preview", None)
        preview = preview_dock.widget() if preview_dock is not None else None
        if preview is None:
            return

        preview._on_streaming_stop()
        try:
            self.mmc.events.sequenceAcquisitionStarted.disconnect(
                preview._on_streaming_start
            )
        except (RuntimeError, TypeError):
            return
        self._suspended_live_preview = preview

    def restore_live_preview_after_mda(self, force: bool = False) -> None:
        """Reconnect a suspended live preview after the OPM MDA thread exits."""
        if not force and (
            self._opm_mda_thread is None or self._opm_mda_thread.is_alive()
        ):
            return

        preview = self._suspended_live_preview
        if preview is not None:
            self.mmc.events.sequenceAcquisitionStarted.connect(
                preview._on_streaming_start
            )
        if self._live_action_was_enabled is not None:
            self.win.get_action(CoreAction.TOGGLE_LIVE).setEnabled(
                self._live_action_was_enabled
            )
        self._suspended_live_preview = None
        self._live_action_was_enabled = None
        self._opm_mda_thread = None

    def set_picard_shutter_open(self, is_open: bool) -> None:
        """Apply a Picard shutter state requested by the settings widget.

        Parameters
        ----------
        is_open : bool
            Whether the user requested the shutter to be open.
        """
        if self.opm_picard_shutter is None:
            self.opm_settings_widget.set_picard_shutter_open(False)
            return

        try:
            if is_open:
                self.opm_picard_shutter.openShutter()
            else:
                self.opm_picard_shutter.closeShutter()
        finally:
            self.sync_picard_shutter_button()

        self.debug(
            "PICARD SHUTTER UPDATED",
            f"State: {self.opm_picard_shutter.state}",
        )

    def sync_picard_shutter_button(self) -> None:
        """Reflect the current Picard shutter state in the settings widget."""
        if self.opm_settings_widget is None or self.opm_picard_shutter is None:
            return
        state_name = str(self.opm_picard_shutter.state).rsplit(".", maxsplit=1)[-1]
        is_open = state_name.casefold() == "open"
        if self.opm_settings_widget.picard_shutter_button.isChecked() != is_open:
            self.opm_settings_widget.set_picard_shutter_open(is_open)

    def update_config_snapshot(self, config: dict) -> None:
        """Accept an immutable-at-dispatch snapshot from the OPM widget.

        Parameters
        ----------
        config : dict
            Complete OPM configuration represented by current GUI controls.
        """
        self.config_store.replace(config)
        if self.opm_engine is not None:
            self.opm_engine.set_config(self.config)

    def update_live_state(self, device_name=None, property_name=None) -> None:
        """Apply live camera, DAQ, and channel state after MM config changes.

        The callback may temporarily stop continuous sequence acquisition while
        the DAQ waveform and camera settings are updated, then restart preview.

        Parameters
        ----------
        device_name : str or None
            Micro-Manager device that emitted the configuration change.
        property_name : str or None
            Device property that changed.
        """
        daq_config = self.config["acq_config"]["DAQ"]
        camera_roi = self.config["acq_config"]["camera_roi"]

        # Ignore shutter signals
        if device_name == self.mmc.getShutterDevice() and property_name == "State":
            return

        self.debug(
            "LIVE STATE TRIGGERED",
            f"Device name: {device_name}",
            f"Property name: {property_name}",
        )

        if "OPM-live-mode" not in set(self.mmc.getLoadedDevices()):
            self.debug(
                "LIVE STATE SKIPPED",
                "The active Micro-Manager configuration has no OPM-live-mode device.",
            )
            return

        # ------------------------------------------------------------------------------#
        # OPM live modes
        # ------------------------------------------------------------------------------#

        opm_mode = self.mmc.getProperty("OPM-live-mode", "Label")
        if opm_mode == "0-Standard":
            scan_type = "2d"
        elif opm_mode == "1-Projection":
            scan_type = "projection"
        else:
            self.warning("UNKNOWN LIVE MODE", f"OPM-live-mode: {opm_mode}")
            return

        # ------------------------------------------------------------------------------#
        # Update the DAQ state
        # ------------------------------------------------------------------------------#

        # Stop / restart live preview
        restart_sequence = False
        if self.mmc.isSequenceRunning():
            self.mmc.stopSequenceAcquisition()
            restart_sequence = True

        # Stop the DAQ playback
        if self.opm_nidaq.running():
            self.opm_nidaq.stop_waveform_playback()

        # Get the current camera exposure
        exposure_ms = round(
            float(self.mmc.getProperty(self.config["Camera"]["camera_id"], "Exposure")),
            1,
        )

        # Get the current active laser channel, limits to one channel at a time
        image_mirror_range_um = np.round(
            float(self.mmc.getProperty("ImageGalvoMirrorRange", "Position")),
            0,
        )

        # Create the channel states list needed for the DAQ waveform
        active_channel_id = self.mmc.getProperty("Laser", "Label")
        channel_states = [
            active_channel_id == channel_id
            for channel_id in self.config["OPM"]["channel_ids"]
        ]

        # Define the y cropping and check for valid exposures
        if scan_type == "projection":
            crop_y = int(image_mirror_range_um / self.mmc.getPixelSizeUm())
            if exposure_ms < MIN_PROJECTION_EXPOSURE:
                self.warning(
                    "PROJECTION EXPOSURE TOO LOW",
                    f"Requested exposure: {exposure_ms} ms",
                    f"Defualting to {DEFUALT_PROJECTION_EXPOSURE}ms",
                )
                exposure_ms = DEFUALT_PROJECTION_EXPOSURE
        else:
            crop_y = int(self.mmc.getProperty("ImageCameraCrop", "Label"))

        # Set the acquisition parameters
        self.opm_nidaq.set_acquisition_params(
            scan_type=scan_type,
            channel_states=channel_states,
            image_mirror_range_um=image_mirror_range_um,
            exposure_ms=exposure_ms,
            laser_blanking=daq_config["laser_blanking"],
        )

        # ------------------------------------------------------------------------------#
        # Update the Camera state
        # ------------------------------------------------------------------------------#

        # Set / check camera ROI
        if crop_y != self.mmc.getROI()[-1]:
            self.mmc.clearROI()
            self.mmc.waitForDevice(str(self.config["Camera"]["camera_id"]))
            self.mmc.setROI(
                int(camera_roi["center_x"] - camera_roi["crop_x"] // 2),
                int(camera_roi["center_y"] - crop_y // 2),
                int(camera_roi["crop_x"]),
                crop_y,
            )
            self.mmc.waitForDevice(str(self.config["Camera"]["camera_id"]))

            self.debug("CAMERA ROI UPDATED", f"Crop Y: {crop_y}")

        # Set / check camera exposure
        if self.mmc.getExposure() != exposure_ms:
            self.mmc.setProperty(
                str(self.config["Camera"]["camera_id"]),
                "Exposure",
                exposure_ms,
            )
            self.mmc.waitForDevice(str(self.config["Camera"]["camera_id"]))
            self.debug("CAMERA EXPOSURE UPDATED", f"Exposure: {exposure_ms} ms")

        self.debug(
            "LIVE STATE APPLIED",
            f"Sequence was running: {restart_sequence}",
            f"Scan type: {scan_type}",
            f"Channel states: {channel_states}",
            f"Image mirror range: {image_mirror_range_um} um",
            f"Exposure: {exposure_ms} ms",
            f"Laser blanking: {daq_config['laser_blanking']}",
        )

        # Start live preview
        if restart_sequence:
            self.mmc.startContinuousSequenceAcquisition()
            self.debug("LIVE PREVIEW RESTARTED")

    def update_ao_mirror_state(self) -> None:
        """Apply the AO mirror state selected in the OPM settings widget."""
        ao_mirror_state = self.config["acq_config"]["AO"]["mirror_state"]

        if "system" in ao_mirror_state:
            self.opm_ao_mirror.apply_system_flat_voltage()
        elif "optimized" in ao_mirror_state:
            self.opm_ao_mirror.apply_optimized_voltage()
        elif "factory" in ao_mirror_state:
            self.opm_ao_mirror.apply_factory_flat_voltage()
        elif "zeros" in ao_mirror_state:
            self.opm_ao_mirror.apply_zeros_voltage()

        self.debug("AO MIRROR STATE UPDATED", f"Mirror state: {ao_mirror_state}")

    def setup_preview_mode_callback(self) -> None:
        """Program the OPM NIDAQ waveform before continuous preview starts."""
        if self.opm_nidaq.running():
            self.opm_nidaq.stop_waveform_playback()

        self.update_live_state()
        if any(self.opm_nidaq.channel_states):
            if not self.opm_nidaq.programmed():
                self.opm_nidaq.clear_tasks()
                self.opm_nidaq.generate_waveforms()
                self.opm_nidaq.program_daq_waveforms()
                self.debug(
                    "PREVIEW DAQ REPROGRAMMED",
                    f"Channel states: {self.opm_nidaq.channel_states}",
                )
            else:
                self.debug("PREVIEW DAQ ALREADY PREPARED")
            self.opm_nidaq.start_waveform_playback()

    def prepare_stage_explorer_preview(self) -> None:
        """Arm the current config-group preview waveform for an Explorer MDA.

        Raises
        ------
        RuntimeError
            If the active Micro-Manager channel preset produces no DAQ channel
            or if the waveform cannot be programmed or started.
        """
        if self.opm_nidaq.running():
            self.opm_nidaq.stop_waveform_playback()

        # This reads the live Micro-Manager device properties: Channel, camera
        # exposure, image-galvo range, and OPM-live-mode.  It deliberately does
        # not use the OPM acquisition widget's channel/power/exposure arrays.
        self.update_live_state()
        if not any(self.opm_nidaq.channel_states):
            raise RuntimeError(
                "The active Micro-Manager Channel preset enables no OPM laser"
            )

        self.opm_nidaq.clear_tasks()
        self.opm_nidaq.generate_waveforms()
        self.opm_nidaq.program_daq_waveforms()
        if not self.opm_nidaq.programmed():
            raise RuntimeError("Could not program the Stage Explorer preview DAQ")

        self.opm_nidaq.start_waveform_playback()
        if not self.opm_nidaq.running():
            raise RuntimeError("Could not start the Stage Explorer preview DAQ")

        self.debug(
            "STAGE EXPLORER PREVIEW DAQ ARMED",
            f"Scan type: {self.opm_nidaq.scan_type}",
            f"Channel states: {self.opm_nidaq.channel_states}",
            f"Exposure: {self.opm_nidaq.exposure_ms} ms",
            f"Image mirror range: {self.opm_nidaq.image_mirror_range_um} um",
        )

    def prepare_live_preview_after_mda(self) -> None:
        """Prepare stopped live-preview DAQ tasks on the MDA worker thread."""
        self.update_live_state()
        if not any(self.opm_nidaq.channel_states):
            return

        self.opm_nidaq.clear_tasks()
        self.opm_nidaq.generate_waveforms()
        self.opm_nidaq.program_daq_waveforms()
        if self.opm_nidaq.programmed():
            self.debug(
                "LIVE PREVIEW DAQ PREPARED",
                "Waveform is stopped and ready for the next Live request.",
            )

    def run_opm_acquisition(self) -> None:
        """Prepare the standard MDA plan and dispatch it through OPM controls."""
        output = self.mda_widget.prepare_mda()
        if output is False:
            return
        self.custom_execute_mda(output)

    def custom_execute_mda(self, output: Path | str | None) -> None:
        """Build and run OPM events from the current in-memory GUI snapshot.

        Parameters
        ----------
        output : Path, str, or None
            Requested MDA output or ``"memory"`` for an in-memory acquisition.
        """
        opm_mode = self.config["acq_config"]["opm_mode"]
        ao_mode = self.config["acq_config"]["AO"]["ao_mode"]
        o2o3_mode = self.config["acq_config"]["o2o3_mode"]
        fluidics_mode = self.config["acq_config"]["fluidics"]
        optimize_now = ("now" in ao_mode) or ("now" in o2o3_mode)

        if optimize_now:
            immediate_actions = []
            if "now" in o2o3_mode:
                immediate_actions.append("O2-O3")
            if "now" in ao_mode:
                immediate_actions.append("AO")
            self.info(
                "OPTIMIZE NOW OVERRIDES ACQUISITION",
                f"Only optimizing {' and '.join(immediate_actions)}",
                "No OPM images will be acquired",
            )

        if output == "memory":
            print("No output path supplied")
            output = None

        output = Path(output) if isinstance(output, str) else output
        self.info(
            "OPM ACQUISITION",
            f"Output: {output}",
            f"Optimize now: {optimize_now}",
        )
        if not optimize_now and not output:
            self.warning(
                "ACQUISITION NOT STARTED",
                "Must set acquisition path to execute acquisition",
            )
            return

        if output:
            output = self.timestamped_output_path(output)

        if ("none" not in fluidics_mode) and not optimize_now:
            if not self.confirm_fluidics_ready():
                return
            self.info("FLUIDICS ACCEPTED", "ESI sequence confirmed by user")

        self.debug(
            "OPM ACQUISITION SETTINGS",
            f"OPM mode: {opm_mode}",
            f"AO mode: {ao_mode}",
            f"O2O3 mode: {o2o3_mode}",
            f"Fluidics mode: {fluidics_mode}",
            f"Output path: {output}",
        )

        opm_events, handler = self.create_opm_events(optimize_now, opm_mode, output)
        self.data_handler = handler

        if opm_events is None:
            self.warning("ACQUISITION NOT STARTED", "OPM events are empty")
            return

        self.debug(
            "OPM EVENTS READY",
            f"Event count: {len(opm_events)}",
            f"Handler: {handler}",
            "Engine: OPMEngineV2",
        )
        self.opm_engine.set_config(self.config)
        self.suspend_live_preview_for_mda()
        try:
            self._opm_mda_thread = self.mmc.run_mda(opm_events, output=handler)
        except Exception:
            self.restore_live_preview_after_mda(force=True)
            raise

    def timestamped_output_path(self, output: Path) -> Path:
        """Create a timestamped OPM acquisition output path.

        Parameters
        ----------
        output : Path
            User-selected output path.

        Returns
        -------
        Path
            Output path inside a newly created timestamped directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = output.name
        if output_name.endswith(".ome.zarr"):
            parent_stem = output_name.removesuffix(".ome.zarr")
        else:
            parent_stem = output.stem
        new_dir = output.parent / Path(f"{timestamp}_{parent_stem}")
        new_dir.mkdir(exist_ok=True)
        return new_dir / Path(output.name)

    def confirm_fluidics_ready(self) -> bool:
        """Ask whether the external ESI/fluidics sequence is ready.

        Returns
        -------
        bool
            ``True`` when the user confirms readiness.
        """
        response = QMessageBox.information(
            self.mda_widget,
            "!WARNING! ESI MUST BE RUNNING!",
            "IS ESI SEQUENCE LOADED AND STARTED?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        return response is QMessageBox.StandardButton.Yes

    def create_opm_events(
        self,
        optimize_now: bool,
        opm_mode: str,
        output: Path | None,
    ):
        """Build custom OPM events for the selected acquisition mode.

        Parameters
        ----------
        optimize_now : bool
            Whether to build immediate optimization events.
        opm_mode : str
            Selected OPM acquisition mode.
        output : Path or None
            Acquisition output path.

        Returns
        -------
        tuple
            Event list and output handler, or ``(None, None)`` when no valid
            acquisition can be built.
        """
        if optimize_now:
            return setup_optimizenow(mmc=self.mmc, config=self.config)

        if not output:
            return None, None

        sequence = self.mda_widget.value()
        supported_modes = ("timelapse", "stage", "mirror", "projection")
        if not any(mode in opm_mode for mode in supported_modes):
            self.warning("UNKNOWN OPM ACQUISITION MODE", f"OPM mode: {opm_mode}")
            return None, None
        opm_events, handler = OPMEventBuilder(
            self.mmc, self.config, sequence
        ).build(output=output, mode=opm_mode)

        self.opm_ao_mirror.output_path = output.parents[0]
        return opm_events, handler


def launch_opm_app(
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    mm_config: Path | str | Literal[False] | None = None,
    mmcore=None,
    exec_app: bool = True,
    simulate_hardware: bool | None = None,
) -> MicroManagerGUI:
    """Launch pymmcore-gui and attach OPM after layout restoration.

    Parameters
    ----------
    config_path : Path or str
        OPM JSON configuration path.
    mm_config : Path, str, False, or None
        Micro-Manager configuration passed to ``create_mmgui``.
    mmcore : CMMCorePlus or None
        Existing core instance to reuse.
    exec_app : bool
        Whether to enter the Qt event loop.
    simulate_hardware : bool or None
        Override for external hardware simulation.

    Returns
    -------
    MicroManagerGUI
        Configured Micro-Manager main window.
    """
    controller = OPMAppController(
        Path(config_path),
        mmcore=mmcore,
        mm_config=mm_config,
        exec_app=exec_app,
        simulate_hardware=simulate_hardware,
    )
    return controller.run()


def main(config_path: Path | str | None = None) -> int:
    """Run the Micro-Manager GUI with the OPM controller.

    Parameters
    ----------
    config_path : Path, str, or None
        OPM JSON configuration path.

    Returns
    -------
    int
        Process exit status.
    """
    launch_opm_app(config_path=config_path or DEFAULT_CONFIG_PATH)
    return 0


if __name__ == "__main__":
    main()
