"""Expose only the shared fixtures needed by Qt and application tests."""

from tests import _fixtures

camera_frame_recorder = _fixtures.camera_frame_recorder
demo_core = _fixtures.demo_core
offline_icons = _fixtures.offline_icons
opm_config_factory = _fixtures.opm_config_factory
reset_acquisition_singletons = _fixtures.reset_acquisition_singletons
workspace_tmp_path = _fixtures.workspace_tmp_path
