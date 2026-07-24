"""Expose the shared fixtures needed by cross-component tests."""

from tests import _fixtures

assert_standard_image_fields = _fixtures.assert_standard_image_fields
acquisition_mode = _fixtures.acquisition_mode
camera_frame_recorder = _fixtures.camera_frame_recorder
demo_core = _fixtures.demo_core
gui_integration_scenario = _fixtures.gui_integration_scenario
offline_icons = _fixtures.offline_icons
opm_config_factory = _fixtures.opm_config_factory
opm_config_from_scenario = _fixtures.opm_config_from_scenario
read_tensorstore_array = _fixtures.read_tensorstore_array
reset_acquisition_singletons = _fixtures.reset_acquisition_singletons
sequence_for_scenario = _fixtures.sequence_for_scenario
simulated_acquisition_hardware = _fixtures.simulated_acquisition_hardware
spatial_plan_strategy = _fixtures.spatial_plan_strategy
split_events = _fixtures.split_events
workspace_tmp_path = _fixtures.workspace_tmp_path
