"""Test custom OPM GUI controls and configured end-to-end acquisitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pymmcore_gui._qt.QtCore import Qt
from pymmcore_gui._qt.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox
from useq import MDASequence

from opm_v2._app import launch_opm_app
from opm_v2._update_config_widget import OPMSettingsV2
from opm_v2.engine.opm_custom_events import (
    ACTION_AO_GRID,
    ACTION_AO_OPTIMIZE,
    ACTION_ASI_SETUP_SCAN,
    ACTION_FLUIDICS,
    ACTION_O2O3_AUTOFOCUS,
)
from opm_v2.engine.opm_engine import OPMEngineV2


def _read_config(path: Path) -> dict[str, Any]:
    """Read an OPM configuration from disk.

    Parameters
    ----------
    path : Path
        JSON configuration path.

    Returns
    -------
    dict[str, Any]
        Parsed OPM configuration.
    """
    return json.loads(path.read_text())


def _read_json_after_close(path: Path, qtbot) -> dict[str, Any]:
    """Wait for a finalized JSON document to become readable on Windows.

    Parameters
    ----------
    path : Path
        JSON document written during ome-writers finalization.
    qtbot : pytestqt.qtbot.QtBot
        Qt test helper used to process queued finalization callbacks.

    Returns
    -------
    dict[str, Any]
        Parsed finalized JSON document.
    """
    documents: list[dict[str, Any]] = []

    def json_is_available() -> bool:
        """Attempt one non-blocking read of the finalized document.

        Returns
        -------
        bool
            Whether a complete JSON document was read.
        """
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        documents[:] = [document]
        return True

    qtbot.waitUntil(json_is_available, timeout=5000)
    return documents[0]


def _nested_value(config: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return a value from a nested configuration mapping.

    Parameters
    ----------
    config : dict[str, Any]
        OPM configuration mapping.
    keys : tuple[str, ...]
        Ordered keys leading to the requested value.

    Returns
    -------
    Any
        Value stored at the nested key path.
    """
    value: Any = config
    for key in keys:
        value = value[key]
    return value


def _combo_items(combo: QComboBox) -> list[str]:
    """Return every selectable string in a combo box.

    Parameters
    ----------
    combo : QComboBox
        Combo box to inspect.

    Returns
    -------
    list[str]
        Selectable values in display order.
    """
    return [combo.itemText(index) for index in range(combo.count())]


def _channel_widgets(settings: OPMSettingsV2) -> list[tuple[Any, ...]]:
    """Return each custom channel's slider, value, exposure, and state controls.

    Parameters
    ----------
    settings : OPMSettingsV2
        Custom OPM settings widget.

    Returns
    -------
    list[tuple[Any, ...]]
        Controls ordered according to the configuration channel arrays.
    """
    return [
        (
            settings.sldr_405_power,
            settings.spbx_405_power,
            settings.spbx_405_exp,
            settings.chx_405_state,
        ),
        (
            settings.sldr_488_power,
            settings.spbx_488_power,
            settings.spbx_488_exp,
            settings.chx_488_state,
        ),
        (
            settings.sldr_561_power,
            settings.spbx_561_power,
            settings.spbx_561_exp,
            settings.chx_561_state,
        ),
        (
            settings.sldr_638_power,
            settings.spbx_638_power,
            settings.spbx_638_exp,
            settings.chx_638_state,
        ),
        (
            settings.sldr_705_power,
            settings.spbx_705_power,
            settings.spbx_705_exp,
            settings.chx_705_state,
        ),
    ]


def test_custom_widget_loads_reusable_configuration_without_resetting_it(
    workspace_tmp_path, qtbot, opm_config_factory
) -> None:
    """Verify that every persisted custom control is restored on construction."""
    config = opm_config_factory(
        mode="timelapse",
        active_channels=(0, 2),
        channel_powers=(11.0, 33.0),
        channel_exposures_ms=(6.0, 8.0),
        scan_range_um=8.0,
        scan_axis_step_um=1.25,
        updates={
            "acq_config": {
                "o2o3_mode": "per timepoint",
                "AO": {
                    "active_channel_id": "561nm",
                    "channel_id": "561nm",
                    "ao_mode": "once at start",
                    "mirror_state": "zeros",
                    "metric": "brightness",
                    "metric_acceptance": "accept all",
                    "modes_to_optimize": "high order first",
                    "daq_mode": "projection",
                    "lightsheet_mode": "off",
                    "num_averaged_frames": 7,
                    "num_scan_positions": 3,
                    "num_tile_positions": 4,
                },
                "stage_scan": {"max_stage_scan_range_um": 321.0},
            }
        },
    )
    config_path = opm_config_factory.write(
        config, workspace_tmp_path / "widget_restore.json"
    )
    settings = OPMSettingsV2(config_path)
    qtbot.addWidget(settings)

    assert settings.cmbx_opm_mode.currentText() == "timelapse"
    assert settings.cmbx_o2o3_mode.currentText() == "per timepoint"
    assert settings.cmbx_ao_active_channel.currentText() == "561nm"
    assert settings.cmbx_ao_mode.currentText() == "once at start"
    assert settings.cmbx_ao_mirror.currentText() == "zeros"
    assert settings.cmbx_ao_metric.currentText() == "brightness"
    assert settings.cmbx_ao_accept.currentText() == "accept all"
    assert settings.cmbx_ao_modes.currentText() == "high order first"
    assert settings.cmbx_ao_daq_mode.currentText() == "projection"
    assert settings.cmbx_ao_camera_mode.currentText() == "off"
    assert settings.spbx_averaged_frames.value() == 7
    assert settings.spbx_num_scan_positions.value() == 3
    assert settings.spbx_num_tile_positions.value() == 4
    assert settings.spbx_stage_image_range.value() == 321.0
    assert settings.spbx_scan_step_size.value() == 1.25

    persisted = _read_config(config_path)
    assert persisted["acq_config"]["DAQ"]["channel_states"] == [
        True,
        False,
        True,
        False,
        False,
    ]
    assert persisted["acq_config"]["DAQ"]["channel_powers"] == [
        11.0,
        0.0,
        33.0,
        0.0,
        0.0,
    ]
    assert persisted["acq_config"]["DAQ"]["channel_exposures_ms"] == [
        6.0,
        0.0,
        8.0,
        0.0,
        0.0,
    ]


def test_every_custom_combo_box_option_persists(
    workspace_tmp_path, qtbot, opm_config_factory
) -> None:
    """Exercise every enumerated option exposed by the custom OPM widget."""
    config = opm_config_factory(mode="stage")
    config_path = opm_config_factory.write(
        config, workspace_tmp_path / "widget_combos.json"
    )
    settings = OPMSettingsV2(config_path)
    qtbot.addWidget(settings)

    combo_specs = [
        (settings.cmbx_opm_mode, ("acq_config", "opm_mode"), str),
        (settings.cmbx_o2o3_mode, ("acq_config", "o2o3_mode"), str),
        (settings.cmbx_fluidics_mode, ("acq_config", "fluidics"), str),
        (settings.cmbx_ao_mode, ("acq_config", "AO", "ao_mode"), str),
        (settings.cmbx_ao_mirror, ("acq_config", "AO", "mirror_state"), str),
        (settings.cmbx_ao_metric, ("acq_config", "AO", "metric"), str),
        (
            settings.cmbx_ao_accept,
            ("acq_config", "AO", "metric_acceptance"),
            str,
        ),
        (
            settings.cmbx_ao_modes,
            ("acq_config", "AO", "modes_to_optimize"),
            str,
        ),
        (settings.cmbx_ao_daq_mode, ("acq_config", "AO", "daq_mode"), str),
        (
            settings.cmbx_ao_active_channel,
            ("acq_config", "AO", "active_channel_id"),
            str,
        ),
        (
            settings.cmbx_ao_camera_mode,
            ("acq_config", "AO", "lightsheet_mode"),
            str,
        ),
    ]
    assert {
        name
        for name in vars(settings)
        if name.startswith("cmbx_") and isinstance(getattr(settings, name), QComboBox)
    } == {
        "cmbx_opm_mode",
        "cmbx_o2o3_mode",
        "cmbx_fluidics_mode",
        "cmbx_laser_blanking",
        "cmbx_ao_active_channel",
        "cmbx_ao_mode",
        "cmbx_ao_mirror",
        "cmbx_ao_metric",
        "cmbx_ao_accept",
        "cmbx_ao_modes",
        "cmbx_ao_daq_mode",
        "cmbx_ao_camera_mode",
    }
    for combo, config_keys, transform in combo_specs:
        for index, option in enumerate(_combo_items(combo)):
            combo.setCurrentIndex(index)
            assert _nested_value(_read_config(config_path), config_keys) == transform(
                option
            )

    settings.cmbx_fluidics_mode.setCurrentText("none")
    for index, option in enumerate(_combo_items(settings.cmbx_laser_blanking)):
        settings.cmbx_laser_blanking.setCurrentIndex(index)
        persisted = _read_config(config_path)
        assert persisted["acq_config"]["DAQ"]["laser_blanking"] is (option == "on")


def test_every_custom_numeric_channel_and_boolean_control_persists(
    workspace_tmp_path, qtbot, opm_config_factory
) -> None:
    """Exercise both bounds of every custom numeric and channel control."""
    config = opm_config_factory(mode="stage")
    config_path = opm_config_factory.write(
        config, workspace_tmp_path / "widget_numeric.json"
    )
    settings = OPMSettingsV2(config_path)
    qtbot.addWidget(settings)

    numeric_specs = {
        "spbx_active_channel_power": ("acq_config", "AO", "active_channel_power"),
        "spbx_ao_exposure": ("acq_config", "AO", "exposure_ms"),
        "spbx_ao_mirror_range": ("acq_config", "AO", "image_mirror_range_um"),
        "spbx_num_iterations": ("acq_config", "AO", "num_iterations"),
        "spbx_num_mode_samples": ("acq_config", "AO", "num_mode_samples"),
        "spbx_mode_delta": ("acq_config", "AO", "mode_delta"),
        "spbx_mode_alpha": ("acq_config", "AO", "mode_alpha"),
        "spbx_metric_precision": ("acq_config", "AO", "metric_precision"),
        "spbx_averaged_frames": ("acq_config", "AO", "num_averaged_frames"),
        "spbx_num_scan_positions": ("acq_config", "AO", "num_scan_positions"),
        "spbx_num_tile_positions": ("acq_config", "AO", "num_tile_positions"),
        "spbx_readout_time": ("acq_config", "AO", "readout_ms"),
        "spbx_mirror_image_range": ("acq_config", "DAQ", "scan_range_um"),
        "spbx_proj_image_range": ("acq_config", "DAQ", "scan_range_um"),
        "spbx_scan_step_size": ("acq_config", "DAQ", "scan_axis_step_um"),
        "spbx_stage_image_range": (
            "acq_config",
            "stage_scan",
            "max_stage_scan_range_um",
        ),
        "spbx_excess_start_frames": (
            "acq_config",
            "stage_scan",
            "excess_start_frames",
        ),
        "spbx_excess_end_frames": (
            "acq_config",
            "stage_scan",
            "excess_end_frames",
        ),
        "spbx_stage_slope_x": ("acq_config", "Positions", "coverslip_slope_x"),
        "spbx_stage_slope_y": ("acq_config", "Positions", "coverslip_slope_y"),
        "spbx_roi_center_x": ("acq_config", "camera_roi", "center_x"),
        "spbx_roi_center_y": ("acq_config", "camera_roi", "center_y"),
        "spbx_roi_crop_x": ("acq_config", "camera_roi", "crop_x"),
        "spbx_roi_crop_y": ("acq_config", "camera_roi", "crop_y"),
    }
    channel_spin_names = {
        f"spbx_{wavelength}_{kind}"
        for wavelength in ("405", "488", "561", "638", "705")
        for kind in ("power", "exp")
    }
    all_spin_names = {
        name
        for name in vars(settings)
        if name.startswith("spbx_")
        and isinstance(getattr(settings, name), (QSpinBox, QDoubleSpinBox))
    }
    assert set(numeric_specs) | channel_spin_names == all_spin_names

    for widget_name, config_keys in numeric_specs.items():
        control = getattr(settings, widget_name)
        for value in (control.minimum(), control.maximum()):
            control.setValue(value)
            assert (
                _nested_value(_read_config(config_path), config_keys) == control.value()
            )

    expected_slider_names = {
        f"sldr_{wavelength}_power" for wavelength in ("405", "488", "561", "638", "705")
    }
    assert {
        name for name in vars(settings) if name.startswith("sldr_")
    } == expected_slider_names

    for channel_index, (slider, power, exposure, checkbox) in enumerate(
        _channel_widgets(settings)
    ):
        for value in (slider.minimum(), slider.maximum()):
            slider.setValue(value)
            persisted = _read_config(config_path)["acq_config"]["DAQ"]
            assert power.value() == value
            assert persisted["channel_powers"][channel_index] == value
        for control, config_key in (
            (power, "channel_powers"),
            (exposure, "channel_exposures_ms"),
        ):
            for value in (control.minimum(), control.maximum()):
                control.setValue(value)
                persisted = _read_config(config_path)["acq_config"]["DAQ"]
                assert persisted[config_key][channel_index] == control.value()
        for checked in (False, True):
            checkbox.setChecked(checked)
            persisted = _read_config(config_path)["acq_config"]["DAQ"]
            assert persisted["channel_states"][channel_index] is checked


def _sequence_for_scenario(mode: str, scenario: dict[str, Any]) -> MDASequence:
    """Build the standard sequence envelope consumed by an OPM mode.

    This does not test the standard MDA widget; it supplies only the plans that
    the custom OPM event builder reads.

    Parameters
    ----------
    mode : str
        OPM acquisition mode.
    scenario : dict[str, Any]
        Reusable integration scenario.

    Returns
    -------
    MDASequence
        Sequence containing the required position, grid, and time plans.
    """
    if mode == "stage":
        return MDASequence(
            grid_plan={"top": 0.0, "left": 0.0, "bottom": 0.0, "right": 10.0},
            axis_order=scenario["axis_order"],
        )
    kwargs: dict[str, Any] = {
        "stage_positions": [(0.0, 0.0, 0.0)],
        "axis_order": scenario["axis_order"],
    }
    if mode == "timelapse":
        kwargs["time_plan"] = {"interval": 0, "loops": scenario["time_loops"]}
    return MDASequence(**kwargs)


@pytest.mark.parametrize(
    ("ao_mode", "o2o3_mode", "expected_action"),
    [
        ("optimize now", "none", ACTION_AO_OPTIMIZE),
        ("none", "optimize now", ACTION_O2O3_AUTOFOCUS),
    ],
)
def test_immediate_gui_actions_reach_the_simulated_engine(
    ao_mode,
    o2o3_mode,
    expected_action,
    demo_core,
    workspace_tmp_path,
    qtbot,
    offline_icons,
    opm_config_factory,
) -> None:
    """Dispatch each non-imaging GUI action through the registered engine.

    Parameters
    ----------
    ao_mode : str
        Immediate AO option under test.
    o2o3_mode : str
        Immediate O2/O3 option under test.
    expected_action : str
        Custom action expected in the simulated engine state.
    demo_core : CMMCorePlus
        Core loaded with Micro-Manager demo devices.
    workspace_tmp_path : Path
        Directory for the isolated configuration.
    qtbot : pytestqt.qtbot.QtBot
        Qt interaction and signal-wait helper.
    offline_icons : None
        Local icon provider for deterministic GUI construction.
    opm_config_factory : OpmConfigFactory
        Reusable simulated configuration factory.
    """
    config = opm_config_factory(
        mode="projection",
        updates={
            "acq_config": {
                "o2o3_mode": o2o3_mode,
                "AO": {
                    "ao_mode": ao_mode,
                    "save_dir_path": str(workspace_tmp_path),
                },
            }
        },
    )
    config_path = opm_config_factory.write(config, workspace_tmp_path / "actions.json")
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
        settings = controller.opm_settings_widget
        settings.cmbx_ao_mode.setCurrentText(ao_mode)
        settings.cmbx_o2o3_mode.setCurrentText(o2o3_mode)
        controller.mda_widget.prepare_mda = lambda: "memory"

        with qtbot.waitSignal(demo_core.mda.events.sequenceFinished, timeout=5000):
            qtbot.mouseClick(settings.run_button, Qt.MouseButton.LeftButton)

        assert controller.data_handler is None
        assert controller.opm_engine.simulated_custom_actions == [expected_action]
        assert not list(workspace_tmp_path.rglob("*.zarr"))
    finally:
        window.close()


def test_custom_gui_selection_reaches_engine_and_tensorstore(
    gui_integration_scenario,
    demo_core,
    workspace_tmp_path,
    qtbot,
    offline_icons,
    opm_config_factory,
    read_tensorstore_array,
    camera_frame_recorder,
) -> None:
    """Run every configured OPM mode and option through persisted pixels."""
    scenario = gui_integration_scenario
    mode = scenario["mode"]
    ao_options = scenario["ao_options"]
    config = opm_config_factory(
        mode=mode,
        active_channels=scenario["active_channels"],
        channel_powers=scenario["channel_powers"],
        channel_exposures_ms=scenario["channel_exposures_ms"],
        camera_shape=tuple(scenario["camera_shape"]),
        scan_range_um=scenario["scan_range_um"],
        scan_axis_step_um=scenario["scan_axis_step_um"],
        updates={
            "acq_config": {
                "o2o3_mode": scenario["o2o3_mode"],
                "fluidics": scenario["fluidics"],
                "AO": {"ao_mode": scenario["ao_mode"], **ao_options},
                "DAQ": {"laser_blanking": scenario["laser_blanking"]},
                "stage_scan": {
                    "excess_start_frames": scenario.get("excess_start_frames", 0),
                    "excess_end_frames": scenario.get("excess_end_frames", 0),
                },
            }
        },
    )
    config_path = opm_config_factory.write(
        config, workspace_tmp_path / f"{scenario['name']}_gui.json"
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
        settings = controller.opm_settings_widget
        # pymmcore-plus recommends disabling hardware sequencing for demo scenarios.
        # Stage mode retains it so the camera-start/ASI-start hook is exercised.
        controller.opm_engine.use_hardware_sequencing = mode == "stage"

        alternate_mode = "mirror" if mode == "stage" else "stage"
        settings.cmbx_opm_mode.setCurrentText(alternate_mode)
        settings.cmbx_opm_mode.setCurrentText(mode)
        combo_values = {
            settings.cmbx_o2o3_mode: scenario["o2o3_mode"],
            settings.cmbx_fluidics_mode: scenario["fluidics"],
            settings.cmbx_ao_mode: scenario["ao_mode"],
            settings.cmbx_ao_mirror: ao_options["mirror_state"],
            settings.cmbx_ao_metric: ao_options["metric"],
            settings.cmbx_ao_accept: ao_options["metric_acceptance"],
            settings.cmbx_ao_modes: ao_options["modes_to_optimize"],
            settings.cmbx_ao_daq_mode: ao_options["daq_mode"],
            settings.cmbx_ao_active_channel: ao_options["active_channel_id"],
            settings.cmbx_ao_camera_mode: ao_options["lightsheet_mode"],
        }
        for combo, value in combo_values.items():
            alternate_index = (combo.findText(value) + 1) % combo.count()
            combo.setCurrentIndex(alternate_index)
            combo.setCurrentText(value)
        settings.cmbx_laser_blanking.setCurrentText(
            "on" if scenario["laser_blanking"] else "off"
        )
        settings.spbx_mirror_image_range.setValue(
            scenario["scan_range_um"] + scenario["scan_axis_step_um"]
        )
        settings.spbx_mirror_image_range.setValue(scenario["scan_range_um"])
        settings.spbx_scan_step_size.setValue(
            max(settings.spbx_scan_step_size.minimum(), 1.0)
        )
        settings.spbx_scan_step_size.setValue(scenario["scan_axis_step_um"])

        expected_states = [False] * len(config["OPM"]["channel_ids"])
        expected_powers = [0.0] * len(expected_states)
        expected_exposures = [0.0] * len(expected_states)
        configured_channels = dict(
            zip(
                scenario["active_channels"],
                zip(
                    scenario["channel_powers"],
                    scenario["channel_exposures_ms"],
                    strict=True,
                ),
                strict=True,
            )
        )
        for channel_index, (_, power, exposure, checkbox) in enumerate(
            _channel_widgets(settings)
        ):
            checkbox.setChecked(False)
            power.setValue(0.0)
            exposure.setValue(0.0)
            if channel_index in configured_channels:
                channel_power, channel_exposure = configured_channels[channel_index]
                power.setValue(channel_power)
                exposure.setValue(channel_exposure)
                checkbox.setChecked(True)
                expected_states[channel_index] = True
                expected_powers[channel_index] = channel_power
                expected_exposures[channel_index] = channel_exposure

        settings.spbx_roi_crop_x.setValue(scenario["camera_shape"][0])
        settings.spbx_roi_crop_y.setValue(scenario["camera_shape"][1])
        settings.spbx_excess_start_frames.setValue(
            scenario.get("excess_start_frames", 0)
        )
        settings.spbx_excess_end_frames.setValue(scenario.get("excess_end_frames", 0))

        persisted = _read_config(config_path)
        assert persisted["acq_config"]["opm_mode"] == mode
        assert (
            persisted["acq_config"]["DAQ"]["scan_range_um"] == scenario["scan_range_um"]
        )
        assert (
            persisted["acq_config"]["DAQ"]["scan_axis_step_um"]
            == scenario["scan_axis_step_um"]
        )
        assert persisted["acq_config"]["DAQ"]["channel_states"] == expected_states
        assert persisted["acq_config"]["DAQ"]["channel_powers"] == expected_powers
        assert (
            persisted["acq_config"]["DAQ"]["channel_exposures_ms"] == expected_exposures
        )
        assert (
            persisted["acq_config"]["camera_roi"]["crop_x"]
            == scenario["camera_shape"][0]
        )
        assert (
            persisted["acq_config"]["camera_roi"]["crop_y"]
            == scenario["camera_shape"][1]
        )
        assert persisted["acq_config"]["o2o3_mode"] == scenario["o2o3_mode"]
        assert persisted["acq_config"]["fluidics"] == scenario["fluidics"]
        for option, value in {
            "ao_mode": scenario["ao_mode"],
            **ao_options,
        }.items():
            assert persisted["acq_config"]["AO"][option] == value

        sequence = _sequence_for_scenario(mode, scenario)
        controller.mda_widget.value = lambda: sequence
        requested_output = workspace_tmp_path / "data.zarr"
        controller.confirm_fluidics_ready = lambda: True
        controller.mda_widget.prepare_mda = lambda: requested_output
        with qtbot.waitSignal(demo_core.mda.events.sequenceFinished, timeout=30000):
            qtbot.mouseClick(settings.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(
            lambda: controller.data_handler.is_finalized,
            timeout=5000,
        )

        matches = list(
            workspace_tmp_path.glob(
                f"*_{requested_output.stem}/{requested_output.name}"
            )
        )
        assert len(matches) == 1
        output = matches[0]
        stored_array = read_tensorstore_array(output / "0")
        assert tuple(stored_array.shape) == tuple(scenario["expected_shape"])
        assert isinstance(demo_core.mda.engine, OPMEngineV2)

        root_metadata = _read_json_after_close(output / "zarr.json", qtbot)
        opm_metadata = root_metadata["attributes"]["opm_v2"]
        assert (
            opm_metadata["acquisition_order"] == scenario["expected_acquisition_order"]
        )
        storage_axes = [
            axis["name"]
            for axis in root_metadata["attributes"]["ome"]["multiscales"][0]["axes"]
            if axis["name"] not in {"y", "x"}
        ]
        camera_frame_recorder.assert_matches_store(stored_array, storage_axes)
        stored_frames = root_metadata["attributes"]["ome_writers"]["frame_metadata"]
        assert stored_frames
        assert len(stored_frames) == len(camera_frame_recorder.frames)
        stored_config = opm_metadata["configuration"]
        assert stored_config["acq_config"] == controller.config["acq_config"]
        assert {frame["event_metadata"]["DAQ"]["mode"] for frame in stored_frames} == {
            scenario["expected_daq_mode"]
        }
        mirror_is_sequential = (
            mode == "mirror" and len(set(scenario["channel_exposures_ms"])) > 1
        )

        def expected_frame_daq(frame, values):
            """Return full or single-channel DAQ values for a stored frame.

            Parameters
            ----------
            frame : dict[str, Any]
                Stored ome-writers frame metadata.
            values : list[Any]
                Full GUI-selected DAQ values.

            Returns
            -------
            list[Any]
                Values representing the hardware state for this frame.
            """
            if not mirror_is_sequential:
                return values
            current_channel = frame["event_metadata"]["DAQ"]["current_channel"]
            channel_index = config["OPM"]["channel_ids"].index(current_channel)
            result = [False if isinstance(value, bool) else 0.0 for value in values]
            result[channel_index] = values[channel_index]
            return result

        assert all(
            frame["event_metadata"]["DAQ"]["channel_states"]
            == expected_frame_daq(frame, expected_states)
            for frame in stored_frames
        )
        if mode == "stage":
            stage_speed_mm_s = demo_core.mda.engine.simulated_asi_state[
                "scan_axis_speed_mm_s"
            ]
            actual_exposure_ms = (
                scenario["scan_axis_step_um"]
                / 1000
                / stage_speed_mm_s
                / sum(expected_states)
                * 1000
            )
            stored_exposures = [actual_exposure_ms] * len(expected_states)
        else:
            stored_exposures = expected_exposures
        assert all(
            frame["event_metadata"]["DAQ"]["exposure_channels_ms"]
            == expected_frame_daq(frame, stored_exposures)
            for frame in stored_frames
        )
        assert all(
            frame["event_metadata"]["DAQ"]["laser_powers"]
            == expected_frame_daq(frame, expected_powers)
            for frame in stored_frames
        )
        assert {
            frame["event_metadata"]["DAQ"]["blanking"] for frame in stored_frames
        } == {scenario["laser_blanking"] or scenario["fluidics"] != "none"}
        assert {
            frame["event_metadata"]["DAQ"]["current_channel"] for frame in stored_frames
        } == {
            config["OPM"]["channel_ids"][channel]
            for channel in scenario["active_channels"]
        }
        assert {
            frame["event_metadata"]["Camera"]["exposure_ms"] for frame in stored_frames
        } == (
            {actual_exposure_ms}
            if mode == "stage"
            else set(scenario["channel_exposures_ms"])
        )
        assert all(
            frame["event_metadata"]["Camera"]["camera_crop_x"]
            == scenario["camera_shape"][0]
            for frame in stored_frames
        )
        expected_crop_y = (
            int(scenario["scan_range_um"] / round(demo_core.getPixelSizeUm(), 3))
            if mode == "projection"
            else scenario["camera_shape"][1]
        )
        assert all(
            frame["event_metadata"]["Camera"]["camera_crop_y"] == expected_crop_y
            for frame in stored_frames
        )
        actions = controller.opm_engine.simulated_custom_actions
        if mode == "stage":
            assert ACTION_ASI_SETUP_SCAN in actions
            timepoints = int(
                scenario["fluidics"] if scenario["fluidics"] != "none" else 1
            )
            assert controller.opm_engine.simulated_asi_transitions == (
                ["Idle", "Running"] * timepoints + ["Idle"]
            )
            assert {
                frame["event_metadata"]["OPM"]["excess_scan_start_positions"]
                for frame in stored_frames
            } == {scenario.get("excess_start_frames", 0)}
            assert {
                frame["event_metadata"]["OPM"]["excess_scan_end_positions"]
                for frame in stored_frames
            } == {scenario.get("excess_end_frames", 0)}
        else:
            assert ACTION_ASI_SETUP_SCAN not in actions
            assert controller.opm_engine.simulated_asi_state == {}
        if scenario["fluidics"] not in {"none", "1"}:
            assert ACTION_FLUIDICS in actions
        if scenario["o2o3_mode"] != "none":
            assert ACTION_O2O3_AUTOFOCUS in actions
        if "grid" in scenario["ao_mode"]:
            assert ACTION_AO_GRID in actions
        elif scenario["ao_mode"] != "none":
            assert ACTION_AO_OPTIMIZE in actions
    finally:
        window.close()
