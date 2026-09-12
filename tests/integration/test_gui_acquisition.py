"""Integrate persisted GUI selections with the engine and TensorStore."""

from __future__ import annotations

from tests.gui.test_opm_gui import (
    run_custom_gui_selection_reaches_engine_and_tensorstore,
)


def test_custom_gui_selection_reaches_engine_and_tensorstore(
    gui_integration_scenario,
    demo_core,
    workspace_tmp_path,
    qtbot,
    offline_icons,
    opm_config_factory,
    read_tensorstore_array,
    camera_frame_recorder,
    opm_config_from_scenario,
    sequence_for_scenario,
) -> None:
    """Run every configured OPM mode and option through persisted pixels."""
    run_custom_gui_selection_reaches_engine_and_tensorstore(
        gui_integration_scenario,
        demo_core,
        workspace_tmp_path,
        qtbot,
        offline_icons,
        opm_config_factory,
        read_tensorstore_array,
        camera_frame_recorder,
        opm_config_from_scenario,
        sequence_for_scenario,
    )
