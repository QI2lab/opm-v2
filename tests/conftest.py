"""Provide reusable configuration, hardware, and storage fixtures for OPM tests."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch
from pathlib import Path

os.environ.setdefault("PYTEST_RUNNING", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import tensorstore as ts
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.core import _mmcore_plus
from useq import CustomAction, MDAEvent

import opm_v2.hardware.AOMirror as ao_module
import opm_v2.hardware.APump as pump_module
import opm_v2.hardware.ElveFlow as ob1_module
import opm_v2.hardware.OPMNIDAQ as daq_module
import opm_v2.hardware.PicardShutter as shutter_module
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ

PROJECT_ROOT = Path(__file__).parents[1]
CANONICAL_CONFIG_PATH = PROJECT_ROOT / "opm_config.json"
GUI_INTEGRATION_CONFIG_PATH = (
    PROJECT_ROOT / "tests" / "configurations" / "gui_integration.json"
)


def _deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
    """Recursively update a nested configuration dictionary.

    Parameters
    ----------
    target : dict[str, Any]
        Configuration dictionary modified in place.
    updates : Mapping[str, Any]
        Nested values to merge into the target.
    """
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _channel_values(
    channel_count: int,
    active_channels: Sequence[int],
    active_values: Sequence[float],
) -> list[float]:
    """Expand values for active channels into a complete channel array.

    Parameters
    ----------
    channel_count : int
        Total number of configured OPM channels.
    active_channels : Sequence[int]
        Indices of channels receiving values.
    active_values : Sequence[float]
        Values corresponding to the active channel indices.

    Returns
    -------
    list[float]
        Full channel array with inactive entries set to zero.

    Raises
    ------
    ValueError
        If values and channel indices differ in length or an index is invalid.
    """
    if len(active_channels) != len(active_values):
        raise ValueError("Each active channel requires one configured value")
    values = [0.0] * channel_count
    for channel, value in zip(active_channels, active_values, strict=True):
        if not 0 <= channel < channel_count:
            raise ValueError(f"Channel index {channel} is outside 0:{channel_count}")
        values[channel] = float(value)
    return values


@dataclass(frozen=True)
class OpmConfigFactory:
    """Build independent OPM demo configurations from the canonical template.

    Parameters
    ----------
    template : dict[str, Any]
        Canonical OPM configuration copied for every scenario.
    camera_id : str
        Micro-Manager demo camera device name.
    camera_width : int
        Demo camera width in pixels.
    camera_height : int
        Demo camera height in pixels.
    """

    template: dict[str, Any]
    camera_id: str
    camera_width: int
    camera_height: int

    def __call__(
        self,
        *,
        mode: str,
        active_channels: Sequence[int] = (0, 1),
        channel_powers: Sequence[float] = (10.0, 20.0),
        channel_exposures_ms: Sequence[float] = (10.0, 10.0),
        camera_shape: tuple[int, int] = (64, 32),
        scan_range_um: float = 4.0,
        scan_axis_step_um: float = 2.0,
        updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a configured, isolated demo acquisition dictionary.

        Parameters
        ----------
        mode : str
            OPM acquisition mode.
        active_channels : Sequence[int]
            Canonical channel indices enabled for the scenario.
        channel_powers : Sequence[float]
            Power for each enabled channel.
        channel_exposures_ms : Sequence[float]
            Exposure in milliseconds for each enabled channel.
        camera_shape : tuple[int, int]
            Camera crop width and height in pixels.
        scan_range_um : float
            Image-mirror scan range in micrometers.
        scan_axis_step_um : float
            Scan-axis sampling step in micrometers.
        updates : Mapping[str, Any] or None
            Arbitrary nested overrides applied after reusable defaults.

        Returns
        -------
        dict[str, Any]
            Independent configuration bound to Micro-Manager demo hardware.
        """
        config = deepcopy(self.template)
        channel_count = len(config["OPM"]["channel_ids"])
        powers = _channel_values(channel_count, active_channels, channel_powers)
        exposures = _channel_values(
            channel_count, active_channels, channel_exposures_ms
        )
        states = [channel in active_channels for channel in range(channel_count)]
        crop_x, crop_y = camera_shape
        _deep_update(
            config,
            {
                "OPM": {"simulate_hardware": True, "fluidics_enabled": False},
                "Camera": {"camera_id": self.camera_id},
                "acq_config": {
                    "opm_mode": mode,
                    "o2o3_mode": "none",
                    "fluidics": "none",
                    "camera_roi": {
                        "center_x": self.camera_width // 2,
                        "center_y": self.camera_height // 2,
                        "crop_x": crop_x,
                        "crop_y": crop_y,
                    },
                    "AO": {"ao_mode": "none"},
                    "DAQ": {
                        "channel_states": states,
                        "channel_powers": powers,
                        "channel_exposures_ms": exposures,
                        "scan_range_um": scan_range_um,
                        "scan_axis_step_um": scan_axis_step_um,
                        "laser_blanking": True,
                    },
                    "Positions": {
                        "coverslip_slope_x": 0.0,
                        "coverslip_slope_y": 0.0,
                        "coverslip_max_dz": 2.0,
                        "scan_axis_overlap_um": 0.0,
                        "tile_axis_overlap": 0.2,
                        "z_axis_overlap": 0.2,
                    },
                    "stage_scan": {
                        "max_stage_scan_range_um": 1000.0,
                        "excess_start_frames": 0,
                        "excess_end_frames": 0,
                    },
                },
            },
        )
        if updates:
            _deep_update(config, updates)
        return config

    @staticmethod
    def write(config: Mapping[str, Any], path: Path) -> Path:
        """Serialize a generated configuration and return its path.

        Parameters
        ----------
        config : Mapping[str, Any]
            Generated OPM configuration.
        path : Path
            Destination JSON path.

        Returns
        -------
        Path
            Destination path containing the serialized configuration.
        """
        path.write_text(json.dumps(config, indent=4))
        return path


@dataclass(frozen=True)
class SimulatedAcquisitionHardware:
    """Hold simulated hardware that participates in OPM acquisitions.

    Parameters
    ----------
    mirror : AOMirror
        Singleton-backed simulated adaptive-optics mirror.
    daq : OPMNIDAQ
        Singleton-backed simulated waveform controller.
    """

    mirror: AOMirror
    daq: OPMNIDAQ


@pytest.fixture
def workspace_tmp_path(request: pytest.FixtureRequest) -> Path:
    """Return a deterministic workspace-local path for acquisition artifacts.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Active test request used to derive a scenario-specific path.

    Returns
    -------
    Path
        Workspace-local artifact directory.
    """
    test_name = request.node.name.replace("[", "_").replace("]", "_")
    path = Path.cwd() / "pytest_workspace" / test_name
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def offline_icons(workspace_tmp_path: Path) -> Iterator[None]:
    """Provide deterministic local SVGs for pymmcore-gui widget construction.

    Parameters
    ----------
    workspace_tmp_path : Path
        Test artifact directory in which icons are created.

    Yields
    ------
    None
        Control while icon lookups are patched to local files.
    """
    svg_dir = workspace_tmp_path / "icons"
    svg_dir.mkdir(exist_ok=True)
    counter = 0

    def mock_svg_path(*_key: str, color: str | None = None, **_kwargs) -> Path:
        """Create and return a deterministic local SVG path.

        Parameters
        ----------
        *_key : str
            Ignored icon lookup keys.
        color : str or None
            Optional SVG fill color.
        **_kwargs : Any
            Ignored icon lookup options.

        Returns
        -------
        Path
            Newly created local SVG path.
        """
        nonlocal counter
        svg_file = svg_dir / f"icon_{counter}.svg"
        counter += 1
        svg_file.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            f'<rect width="24" height="24" fill="{color or "currentColor"}"/>'
            "</svg>"
        )
        return svg_file

    with (
        patch("pymmcore_widgets.control._stage_widget.svg_path", mock_svg_path),
        patch("superqt.iconify.svg_path", mock_svg_path),
    ):
        yield


@pytest.fixture
def demo_core() -> Iterator[CMMCorePlus]:
    """Provide the shared core loaded with Micro-Manager's real demo devices.

    Yields
    ------
    CMMCorePlus
        Process-wide core instance loaded with Micro-Manager demo hardware.
    """
    _mmcore_plus._instance = None
    core = CMMCorePlus.instance()
    core.loadSystemConfiguration()
    try:
        yield core
    finally:
        try:
            core.waitForSystem()
        except Exception:
            pass
        core.unloadAllDevices()
        _mmcore_plus._instance = None


@pytest.fixture
def opm_config_factory(demo_core: CMMCorePlus) -> OpmConfigFactory:
    """Provide a factory for isolated configurations bound to demo hardware.

    Parameters
    ----------
    demo_core : CMMCorePlus
        Core instance supplying demo camera properties.

    Returns
    -------
    OpmConfigFactory
        Scenario configuration factory.
    """
    return OpmConfigFactory(
        template=json.loads(CANONICAL_CONFIG_PATH.read_text()),
        camera_id=demo_core.getCameraDevice(),
        camera_width=demo_core.getImageWidth(),
        camera_height=demo_core.getImageHeight(),
    )


@pytest.fixture(scope="session")
def gui_integration_configurations() -> dict[str, dict[str, Any]]:
    """Load reusable custom-widget acquisition scenarios.

    Returns
    -------
    dict[str, dict[str, Any]]
        Named mode configurations and their expected datastore results.
    """
    return json.loads(GUI_INTEGRATION_CONFIG_PATH.read_text())


@pytest.fixture(autouse=True)
def reset_acquisition_singletons() -> Iterator[None]:
    """Isolate the process-wide OPM acquisition hardware instances per test.

    Yields
    ------
    None
        Control while all hardware singleton slots belong to the active test.
    """
    ao_module._instance_mirror = None
    pump_module._instance_pump = None
    ob1_module._instance_ob1 = None
    daq_module._instance_daq = None
    shutter_module._instance_shutter = None
    yield
    ao_module._instance_mirror = None
    pump_module._instance_pump = None
    ob1_module._instance_ob1 = None
    daq_module._instance_daq = None
    shutter_module._instance_shutter = None


@pytest.fixture
def simulated_acquisition_hardware() -> SimulatedAcquisitionHardware:
    """Provide singleton-backed simulated AO mirror and DAQ controllers.

    Returns
    -------
    SimulatedAcquisitionHardware
        Simulated controllers used by event generation and acquisition.
    """
    hardware = SimulatedAcquisitionHardware(
        mirror=AOMirror(simulate=True),
        daq=OPMNIDAQ(simulate=True),
    )
    assert AOMirror.instance() is hardware.mirror
    assert OPMNIDAQ.instance() is hardware.daq
    return hardware


@pytest.fixture
def split_events() -> Callable[[Sequence[MDAEvent]], tuple[list[MDAEvent], list[str]]]:
    """Partition camera events from custom-action names.

    Returns
    -------
    Callable
        Function returning camera events and ordered custom-action names.
    """

    def _split(events: Sequence[MDAEvent]) -> tuple[list[MDAEvent], list[str]]:
        """Return camera events and ordered custom-action names.

        Parameters
        ----------
        events : Sequence[MDAEvent]
            Generated OPM acquisition events.

        Returns
        -------
        tuple[list[MDAEvent], list[str]]
            Camera events and custom-action names.
        """
        image_events = [
            event for event in events if not isinstance(event.action, CustomAction)
        ]
        custom_actions = [
            event.action.name
            for event in events
            if isinstance(event.action, CustomAction)
        ]
        return image_events, custom_actions

    return _split


@pytest.fixture
def read_tensorstore_array() -> Callable[[Path], Any]:
    """Provide a reader for arrays written by the TensorStore OME backend.

    Returns
    -------
    Callable
        Function reading an array from a TensorStore dataset path.
    """

    def _read(path: Path) -> Any:
        """Read an array from a TensorStore dataset path.

        Parameters
        ----------
        path : Path
            TensorStore dataset path.

        Returns
        -------
        Any
            Materialized TensorStore array.
        """
        store = ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(path)},
            },
            open=True,
        ).result()
        return store.read().result()

    return _read
