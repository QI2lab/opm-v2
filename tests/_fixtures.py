"""Provide fixtures shared by GUI and integration tests."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ.setdefault("PYTEST_RUNNING", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import tensorstore as ts
from pymmcore_plus import CMMCorePlus
from useq import AbsolutePosition, CustomAction, GridFromEdges, MDAEvent, MDASequence

from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.APump import APump
from opm_v2.hardware.ElveFlow import OB1Controller
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.PicardShutter import PicardShutter
from opm_v2.utils.coverslip import COVERSLIP_METADATA_KEY, CoverslipPlane

PROJECT_ROOT = Path(__file__).parents[1]
TEST_RUN_ID = uuid.uuid4().hex[:6]
CANONICAL_CONFIG_PATH = PROJECT_ROOT / "opm_config.json"
GUI_INTEGRATION_CONFIG_PATH = (
    PROJECT_ROOT / "tests" / "configurations" / "gui_integration.json"
)
GUI_INTEGRATION_CONFIGURATIONS = json.loads(GUI_INTEGRATION_CONFIG_PATH.read_text())


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
                    "AO": {
                        "ao_mode": "none",
                        "mirror_state": "system flat",
                        "metric": "DCT",
                        "mode_delta": 0.15,
                        "mode_alpha": 0.5,
                        "num_iterations": 2,
                        "num_mode_samples": 3,
                        "num_scan_positions": 1,
                        "num_tile_positions": 1,
                        "image_mirror_range_um": 150.0,
                        "active_channel_id": "405nm",
                        "active_channel_power": 10.0,
                        "exposure_ms": 150.0,
                        "readout_ms": 4.868,
                        "metric_precision": 6,
                        "metric_acceptance": "zero",
                        "num_averaged_frames": 1,
                    },
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
                        "sample_depth_start_um": 0.0,
                        "sample_depth_end_um": 0.0,
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


@dataclass(frozen=True)
class SpatialPlanStrategy:
    """Describe one native-MDA or Stage Explorer spatial input strategy.

    Parameters
    ----------
    name : str
        Stable pytest identifier.
    kind : str
        One of ``literal``, ``grid``, or ``stage_explorer``.
    position_count : int
        Number of literal positions or exported Stage Explorer regions.
    with_coverslip : bool
        Whether exported regions carry fitted coverslip-plane metadata.
    """

    name: str
    kind: str
    position_count: int
    with_coverslip: bool = False

    def _stage_explorer_positions(
        self,
    ) -> tuple[list[AbsolutePosition], list[tuple[tuple[float, ...], CoverslipPlane]]]:
        """Build exported ROI positions and their expected physical planes.

        Returns
        -------
        tuple
            Absolute positions plus ``(bounds, plane)`` expectations.
        """
        positions: list[AbsolutePosition] = []
        expectations: list[tuple[tuple[float, ...], CoverslipPlane]] = []
        for index in range(self.position_count):
            x_min = 100.0 + 60.0 * index
            x_max = x_min + 20.0
            y_min = 200.0 + 40.0 * index
            y_max = y_min + 20.0
            z_anchor = 50.0 + 5.0 * index
            plane = CoverslipPlane(
                x_min,
                y_min,
                z_anchor,
                0.01,
                -0.02,
            )
            metadata = (
                {COVERSLIP_METADATA_KEY: plane.to_metadata()}
                if self.with_coverslip
                else {}
            )
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            center_z = (
                plane.z_at(center_x, center_y)
                if self.with_coverslip
                else z_anchor
            )
            positions.append(
                AbsolutePosition(
                    z=center_z,
                    name=f"roi_{index}",
                    sequence=MDASequence(
                        metadata=metadata,
                        grid_plan=GridFromEdges(
                            left=x_min,
                            right=x_max,
                            top=y_min,
                            bottom=y_max,
                            fov_width=50.0,
                            fov_height=50.0,
                        ),
                    ),
                )
            )
            expectations.append(((x_min, x_max, y_min, y_max, z_anchor), plane))
        return positions, expectations

    def build(
        self,
        *,
        axis_order: str,
        include_time: bool,
    ) -> tuple[
        MDASequence,
        list[tuple[tuple[float, ...], CoverslipPlane]],
    ]:
        """Build an independent useq sequence for this strategy.

        Parameters
        ----------
        axis_order : str
            Axis order accepted by the selected OPM builder.
        include_time : bool
            Whether to include the time plan required by timelapse mode.

        Returns
        -------
        tuple
            Sequence and Stage Explorer region expectations.

        Raises
        ------
        ValueError
            If a statically defined fixture contains an unknown strategy kind.
        """
        sequence_kwargs: dict[str, Any] = {"axis_order": axis_order}
        expectations: list[tuple[tuple[float, ...], CoverslipPlane]] = []
        if include_time:
            sequence_kwargs["time_plan"] = {"interval": 0.1, "loops": 2}

        if self.kind == "literal":
            positions = [
                (10.0 + 20.0 * index, 20.0 + 10.0 * index, 30.0 + index)
                for index in range(self.position_count)
            ]
            sequence_kwargs["stage_positions"] = positions
        elif self.kind == "grid":
            sequence_kwargs["grid_plan"] = {
                "left": 100.0,
                "right": 120.0,
                "top": 220.0,
                "bottom": 200.0,
            }
        elif self.kind == "stage_explorer":
            positions, expectations = self._stage_explorer_positions()
            sequence_kwargs["stage_positions"] = positions
        else:  # pragma: no cover - fixture definitions are static
            raise ValueError(f"Unknown spatial strategy: {self.kind}")

        return MDASequence(**sequence_kwargs), expectations


SPATIAL_PLAN_STRATEGIES = (
    SpatialPlanStrategy("literal_single", "literal", 1),
    SpatialPlanStrategy("literal_multiple", "literal", 2),
    SpatialPlanStrategy("native_grid", "grid", 0),
    SpatialPlanStrategy(
        "stage_explorer_single_no_coverslip",
        "stage_explorer",
        1,
    ),
    SpatialPlanStrategy(
        "stage_explorer_single_coverslip",
        "stage_explorer",
        1,
        with_coverslip=True,
    ),
    SpatialPlanStrategy(
        "stage_explorer_multiple_no_coverslip",
        "stage_explorer",
        2,
    ),
    SpatialPlanStrategy(
        "stage_explorer_multiple_coverslip",
        "stage_explorer",
        2,
        with_coverslip=True,
    ),
)


@dataclass
class CameraFrameRecorder:
    """Record frames and useq events emitted by the Micro-Manager camera.

    Attributes
    ----------
    frames : list[np.ndarray]
        Independent copies of frames in camera delivery order.
    events : list[MDAEvent]
        Events paired with the delivered frames.
    """

    frames: list[np.ndarray] = field(default_factory=list)
    events: list[MDAEvent] = field(default_factory=list)

    def frameReady(self, frame: np.ndarray, event: MDAEvent, _meta: Any) -> None:
        """Record one camera-delivered frame and its event.

        Parameters
        ----------
        frame : np.ndarray
            Image emitted by pymmcore-plus.
        event : MDAEvent
            useq event associated with the image.
        _meta : Any
            Frame metadata emitted with the image.
        """
        self.frames.append(np.asarray(frame).copy())
        self.events.append(event.model_copy(deep=True))

    def assert_matches_store(
        self, array: np.ndarray, storage_axes: Sequence[str]
    ) -> None:
        """Assert every delivered camera frame occupies its indexed store position.

        Parameters
        ----------
        array : np.ndarray
            Materialized TensorStore OME-Zarr array.
        storage_axes : Sequence[str]
            OME axes preceding the array's spatial ``y`` and ``x`` axes.
        """
        assert len(self.frames) == len(self.events)
        assert len(self.frames) == int(np.prod(array.shape[:-2]))
        for frame, event in zip(self.frames, self.events, strict=True):
            omitted_axes = set(event.index) - set(storage_axes)
            assert all(int(event.index[axis]) == 0 for axis in omitted_axes)
            output_index = tuple(int(event.index[axis]) for axis in storage_axes)
            np.testing.assert_array_equal(array[output_index], frame)


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
    node_hash = hashlib.sha1(request.node.nodeid.encode()).hexdigest()[:8]
    short_name = f"{test_name[:36]}_{node_hash}"
    path = Path.cwd() / ".pytest_tmp" / TEST_RUN_ID / short_name
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
    core = CMMCorePlus()
    core.loadSystemConfiguration()
    try:
        yield core
    finally:
        try:
            core.waitForSystem()
        except Exception:
            pass
        core.unloadAllDevices()


@pytest.fixture
def camera_frame_recorder(demo_core: CMMCorePlus) -> Iterator[CameraFrameRecorder]:
    """Record all camera frames emitted during one acquisition test.

    Parameters
    ----------
    demo_core : CMMCorePlus
        Core whose MDA runner emits camera frames.

    Yields
    ------
    CameraFrameRecorder
        Recorder connected through the standard pymmcore-plus MDA signal.
    """
    recorder = CameraFrameRecorder()
    demo_core.mda.events.frameReady.connect(recorder.frameReady)
    try:
        yield recorder
    finally:
        demo_core.mda.events.frameReady.disconnect(recorder.frameReady)


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


@pytest.fixture(params=("projection", "mirror", "stage", "timelapse"))
def acquisition_mode(request: pytest.FixtureRequest) -> str:
    """Parameterize every top-level OPM imaging mode.

    Returns
    -------
    str
        Selected acquisition mode.
    """
    return str(request.param)


@pytest.fixture(
    params=SPATIAL_PLAN_STRATEGIES,
    ids=lambda strategy: strategy.name,
)
def spatial_plan_strategy(
    request: pytest.FixtureRequest,
) -> SpatialPlanStrategy:
    """Parameterize every supported native and Stage Explorer plan shape.

    Returns
    -------
    SpatialPlanStrategy
        Independent strategy definition used to build a useq sequence.
    """
    return request.param


@pytest.fixture(
    params=tuple(GUI_INTEGRATION_CONFIGURATIONS),
    ids=tuple(GUI_INTEGRATION_CONFIGURATIONS),
)
def gui_integration_scenario(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Return one independent, named GUI-to-data acquisition scenario.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Parameterized request naming a scenario in the shared JSON document.

    Returns
    -------
    dict[str, Any]
        Scenario values plus the stable ``name`` used for artifacts.
    """
    scenario = deepcopy(GUI_INTEGRATION_CONFIGURATIONS[str(request.param)])
    scenario["name"] = str(request.param)
    return scenario


@pytest.fixture
def opm_config_from_scenario(
    opm_config_factory: OpmConfigFactory,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Build an OPM configuration from a reusable acquisition scenario.

    Parameters
    ----------
    opm_config_factory : OpmConfigFactory
        Canonical demo-configuration factory.

    Returns
    -------
    Callable[[Mapping[str, Any]], dict[str, Any]]
        Function converting one scenario mapping into an isolated configuration.
    """

    def _build(scenario: Mapping[str, Any]) -> dict[str, Any]:
        """Build one scenario configuration.

        Parameters
        ----------
        scenario : Mapping[str, Any]
            Acquisition mode and custom-widget selections.

        Returns
        -------
        dict[str, Any]
            Independent OPM configuration for the scenario.
        """
        return opm_config_factory(
            mode=str(scenario["mode"]),
            active_channels=scenario["active_channels"],
            channel_powers=scenario["channel_powers"],
            channel_exposures_ms=scenario["channel_exposures_ms"],
            camera_shape=tuple(scenario["camera_shape"]),
            scan_range_um=float(scenario["scan_range_um"]),
            scan_axis_step_um=float(scenario["scan_axis_step_um"]),
            updates={
                "acq_config": {
                    "o2o3_mode": scenario["o2o3_mode"],
                    "fluidics": scenario["fluidics"],
                    "AO": {
                        "ao_mode": scenario["ao_mode"],
                        **scenario.get("ao_options", {}),
                    },
                    "DAQ": {"laser_blanking": scenario["laser_blanking"]},
                    "stage_scan": {
                        "excess_start_frames": scenario.get("excess_start_frames", 0),
                        "excess_end_frames": scenario.get("excess_end_frames", 0),
                    },
                }
            },
        )

    return _build


@pytest.fixture
def sequence_for_scenario() -> Callable[[Mapping[str, Any]], MDASequence]:
    """Build standard useq plans consumed by OPM scenario tests.

    Returns
    -------
    Callable[[Mapping[str, Any]], MDASequence]
        Function producing the scenario's MDA sequence envelope.
    """

    def _build(scenario: Mapping[str, Any]) -> MDASequence:
        """Build one scenario sequence.

        Parameters
        ----------
        scenario : Mapping[str, Any]
            Acquisition scenario containing mode, axis, and optional time plans.

        Returns
        -------
        MDASequence
            Standard useq sequence consumed by the OPM event builder.
        """
        if scenario["mode"] == "stage":
            kwargs: dict[str, Any] = {
                "grid_plan": {
                    "top": 10.0,
                    "left": 0.0,
                    "bottom": 0.0,
                    "right": 0.0,
                },
                "axis_order": scenario["axis_order"],
            }
        else:
            kwargs = {
                "stage_positions": [(0.0, 0.0, 0.0)],
                "axis_order": scenario["axis_order"],
            }
        if scenario.get("time_loops"):
            kwargs["time_plan"] = {
                "interval": scenario.get("time_interval", 0),
                "loops": scenario["time_loops"],
            }
        return MDASequence(**kwargs)

    return _build


@pytest.fixture(autouse=True)
def reset_acquisition_singletons() -> Iterator[None]:
    """Isolate the process-wide OPM acquisition hardware instances per test.

    Yields
    ------
    None
        Control while all hardware singleton slots belong to the active test.
    """
    for hardware_class in (
        AOMirror,
        APump,
        OB1Controller,
        OPMNIDAQ,
        PicardShutter,
    ):
        hardware_class.reset_instance()
    yield
    for hardware_class in (
        AOMirror,
        APump,
        OB1Controller,
        OPMNIDAQ,
        PicardShutter,
    ):
        hardware_class.reset_instance()


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
def assert_standard_image_fields() -> Callable[[Sequence[MDAEvent]], None]:
    """Provide reusable assertions for fields executed by upstream MDAEngine.

    Returns
    -------
    Callable[[Sequence[MDAEvent]], None]
        Assertion checking standard position and exposure fields against OPM
        metadata for each camera event.
    """

    def _assert(events: Sequence[MDAEvent]) -> None:
        """Check standard useq fields on camera events.

        Parameters
        ----------
        events : Sequence[MDAEvent]
            Camera events generated by an OPM mode builder.
        """
        stage_scan_x: dict[int, float] = {}
        for event in events:
            stage = event.metadata["Stage"]
            camera = event.metadata["Camera"]
            if event.metadata["DAQ"]["mode"] != "stage":
                assert event.x_pos == stage["x_pos"]
            else:
                # ASI advances X in hardware; the standard event retains the
                # scan's starting X so MDAEngine never moves it between frames.
                assert event.x_pos is not None
                position_index = int(event.index.get("p", 0))
                stage_scan_x.setdefault(position_index, float(event.x_pos))
                assert event.x_pos == stage_scan_x[position_index]
            assert event.y_pos == stage["y_pos"]
            assert event.z_pos == stage["z_pos"]
            assert event.exposure == camera["exposure_ms"]

    return _assert


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
