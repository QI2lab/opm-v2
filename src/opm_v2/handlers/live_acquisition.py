"""Publish acquisition plans and lifecycle events for live processing."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO
from uuid import uuid4

if TYPE_CHECKING:
    from useq import MDAEvent


LIVE_ACQUISITION_SCHEMA = "opm_v2.live_acquisition"
LIVE_ACQUISITION_SCHEMA_VERSION = "1.0"
LIFECYCLE_TERMINAL_EVENTS = frozenset({"completed", "canceled", "errored"})
_INDEX_AXES = frozenset({"t", "p", "c", "z"})
_REQUIRED_EVENT_AXES = frozenset({"t", "p", "c"})
_ORIENTATION_KEYS = (
    "camera_XYstage_orientation",
    "camera_Zstage_orientation",
    "camera_mirror_orientation",
)


def acquisition_sidecar_paths(data_path: Path) -> tuple[Path, Path]:
    """Return manifest and lifecycle-log paths for an OME-Zarr store.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        ``(<stem>.manifest.json, <stem>.log.jsonl)`` next to the store.
    """
    name = data_path.name
    for suffix in (".ome.zarr", ".zarr"):
        if name.endswith(suffix):
            stem = name.removesuffix(suffix)
            break
    else:  # pragma: no cover - production paths are validated by the builder
        stem = data_path.stem
    return (
        data_path.parent / f"{stem}.manifest.json",
        data_path.parent / f"{stem}.log.jsonl",
    )


def build_live_acquisition_manifest(
    *,
    data_path: Path,
    index_sizes: Mapping[str, int],
    acquisition_order: Sequence[str],
    events: Sequence[MDAEvent],
    config: Mapping[str, Any],
    pixel_size_um: float,
) -> dict[str, Any]:
    """Build the versioned live-processing manifest from a completed event plan.

    Returns
    -------
    dict[str, Any]
        JSON-compatible manifest satisfying ``LIVE_PROCESSING_CONTRACT.md``.

    Raises
    ------
    ValueError
        If planned image events do not cover every channel and position.
    """
    image_events = [
        event
        for event in events
        if _REQUIRED_EVENT_AXES.issubset(map(str, event.index))
    ]
    if not image_events:
        raise ValueError("Live acquisition manifest requires planned image events")

    normalized_sizes = {
        axis: int(index_sizes.get(axis, 1)) for axis in ("t", "p", "c", "z")
    }
    channels = _channel_records(config, image_events, normalized_sizes["c"])
    positions = _position_records(image_events, normalized_sizes["p"])
    first_metadata = image_events[0].metadata
    daq = first_metadata.get("DAQ", {})
    camera = first_metadata.get("Camera", {})
    opm = first_metadata.get("OPM", {})
    scan_step = daq.get("scan_axis_step_um", daq.get("image_mirror_step_um"))
    if scan_step is None:
        scan_step = config.get("acq_config", {}).get("DAQ", {}).get(
            "scan_axis_step_um"
        )
    excess_start = int(opm.get("excess_scan_start_positions", 0))
    excess_end = int(opm.get("excess_scan_end_positions", 0))
    orientations = {
        key: str(opm.get(key, config.get("OPM", {}).get(key, "")))
        for key in _ORIENTATION_KEYS
    }
    manifest = {
        "schema": LIVE_ACQUISITION_SCHEMA,
        "schema_version": LIVE_ACQUISITION_SCHEMA_VERSION,
        "acquisition_id": str(uuid4()),
        "data_path": data_path.name,
        "mode": str(config.get("acq_config", {}).get("opm_mode", "")),
        "index_sizes": normalized_sizes,
        "acquisition_order": [str(axis) for axis in acquisition_order],
        "channels": channels,
        "stage_positions_zxy": positions,
        "scan_axis": "x",
        "scan_axis_step_um": _required_float(scan_step, "scan_axis_step_um"),
        "pixel_size_um": float(pixel_size_um),
        "angle_deg": _required_float(opm.get("angle_deg"), "angle_deg"),
        "camera_offset": _required_float(camera.get("offset"), "camera_offset"),
        "camera_e_to_adu": _required_float(
            camera.get("e_to_ADU"), "camera_e_to_adu"
        ),
        "excess_scan_positions": excess_start + excess_end,
        "excess_scan_start_positions": excess_start,
        "excess_scan_end_positions": excess_end,
        "orientations": orientations,
    }
    validate_live_acquisition_manifest(manifest)
    return manifest


def validate_live_acquisition_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the public version-1 live-acquisition manifest schema.

    Raises
    ------
    ValueError
        If a required field is absent or incompatible with schema version 1.0.
    """
    if manifest.get("schema") != LIVE_ACQUISITION_SCHEMA:
        raise ValueError("Unsupported live acquisition schema")
    if manifest.get("schema_version") != LIVE_ACQUISITION_SCHEMA_VERSION:
        raise ValueError("Unsupported live acquisition schema version")
    for key in ("acquisition_id", "data_path", "mode", "scan_axis"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"Manifest field {key!r} must be a non-empty string")

    sizes = manifest.get("index_sizes")
    if not isinstance(sizes, Mapping) or set(sizes) != _INDEX_AXES:
        raise ValueError("index_sizes must contain t, p, c, and z exactly once")
    if any(type(size) is not int or size < 1 for size in sizes.values()):
        raise ValueError("Every acquisition index size must be a positive integer")
    order = manifest.get("acquisition_order")
    if not isinstance(order, list) or len(order) != 4 or set(order) != _INDEX_AXES:
        raise ValueError("acquisition_order must contain t, p, c, and z exactly once")

    channels = manifest.get("channels")
    if not isinstance(channels, list) or len(channels) != sizes["c"]:
        raise ValueError("Manifest must contain one channel record per c index")
    for channel in channels:
        if not isinstance(channel, Mapping) or not isinstance(channel.get("name"), str):
            raise ValueError("Every channel record requires a string name")
        for key in ("wavelength_nm", "exposure_ms", "laser_power"):
            value = channel.get(key)
            if value is not None and not _is_number(value):
                raise ValueError(f"Channel field {key!r} must be numeric or null")

    positions = manifest.get("stage_positions_zxy")
    if not isinstance(positions, list) or len(positions) != sizes["p"]:
        raise ValueError("Manifest must contain one ZXY position per p index")
    if any(
        not isinstance(position, list)
        or len(position) != 3
        or any(not _is_number(value) for value in position)
        for position in positions
    ):
        raise ValueError("Every stage position must contain three numeric ZXY values")

    for key in (
        "scan_axis_step_um",
        "pixel_size_um",
        "angle_deg",
        "camera_offset",
        "camera_e_to_adu",
        "excess_scan_positions",
        "excess_scan_start_positions",
        "excess_scan_end_positions",
    ):
        if not _is_number(manifest.get(key)):
            raise ValueError(f"Manifest field {key!r} must be numeric")
    orientations = manifest.get("orientations")
    if not isinstance(orientations, Mapping) or set(orientations) != set(
        _ORIENTATION_KEYS
    ):
        raise ValueError("Manifest orientations are incomplete")
    if any(not isinstance(value, str) or not value for value in orientations.values()):
        raise ValueError("Every manifest orientation must be a non-empty string")


class LiveAcquisitionPublisher:
    """Own one atomic manifest and one flushed lifecycle log."""

    def __init__(self, data_path: Path, manifest: Mapping[str, Any]) -> None:
        """Publish a validated manifest without starting acquisition lifecycle."""
        self.data_path = Path(data_path)
        self.manifest_path, self.log_path = acquisition_sidecar_paths(self.data_path)
        self.manifest = dict(manifest)
        validate_live_acquisition_manifest(self.manifest)
        self.acquisition_id = str(self.manifest["acquisition_id"])
        self._lock = threading.Lock()
        self._log_file: TextIO | None = None
        self._started = False
        self._terminal_event: str | None = None
        _atomic_write_json(self.manifest_path, self.manifest)

    @property
    def terminal_event(self) -> str | None:
        """Terminal event already published, if any."""
        return self._terminal_event

    def started(self, **diagnostics: Any) -> None:
        """Truncate the lifecycle log and append one flushed ``started`` event."""
        with self._lock:
            if self._started:
                return
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log_path.open("w", encoding="utf-8", newline="\n")
            self._started = True
            self._append_record("started", diagnostics)

    def terminal(self, event: str, **diagnostics: Any) -> None:
        """Append exactly one flushed terminal lifecycle event.

        Raises
        ------
        ValueError
            If ``event`` is not a supported terminal lifecycle value.
        RuntimeError
            If acquisition has not published its ``started`` event.
        """
        if event not in LIFECYCLE_TERMINAL_EVENTS:
            raise ValueError(f"Unknown lifecycle terminal event: {event!r}")
        with self._lock:
            if not self._started:
                raise RuntimeError("Cannot publish a terminal event before started")
            if self._terminal_event is not None:
                return
            self._terminal_event = event
            self._append_record(event, diagnostics)
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

    def _append_record(self, event: str, diagnostics: Mapping[str, Any]) -> None:
        """Write, flush, and fsync one complete JSON Lines record.

        Raises
        ------
        RuntimeError
            If the lifecycle log has not been opened.
        """
        if self._log_file is None:  # pragma: no cover - guarded by callers
            raise RuntimeError("Lifecycle log is not open")
        record = {
            "event": event,
            "acquisition_id": self.acquisition_id,
            "timestamp": _timestamp(),
            **diagnostics,
        }
        self._log_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._log_file.flush()
        os.fsync(self._log_file.fileno())


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write one JSON document through an fsynced same-directory replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(document, temporary, separators=(",", ":"))
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _channel_records(
    config: Mapping[str, Any], image_events: Sequence[MDAEvent], count: int
) -> list[dict[str, Any]]:
    """Return channel records in storage-index order.

    Returns
    -------
    list[dict[str, Any]]
        Channel measurements ordered by the acquisition ``c`` index.

    Raises
    ------
    ValueError
        If planned events do not describe every channel.
    """
    opm = config.get("OPM", {})
    daq = config.get("acq_config", {}).get("DAQ", {})
    names = list(opm.get("channel_ids", ()))
    states = list(daq.get("channel_states", ()))
    powers = list(daq.get("channel_powers", ()))
    exposures = list(daq.get("channel_exposures_ms", ()))
    active_indices = [index for index, active in enumerate(states) if active]
    if len(active_indices) == count and all(index < len(names) for index in active_indices):
        return [
            {
                "name": str(names[index]),
                "wavelength_nm": _wavelength(names[index]),
                "exposure_ms": _optional_float_at(exposures, index),
                "laser_power": _optional_float_at(powers, index),
            }
            for index in active_indices
        ]

    records: list[dict[str, Any] | None] = [None] * count
    for event in image_events:
        channel_index = int(event.index["c"])
        if records[channel_index] is not None:
            continue
        metadata = event.metadata
        name = str(metadata.get("DAQ", {}).get("current_channel", channel_index))
        records[channel_index] = {
            "name": name,
            "wavelength_nm": _wavelength(name),
            "exposure_ms": _optional_float(metadata.get("Camera", {}).get("exposure_ms")),
            "laser_power": None,
        }
    if any(record is None for record in records):
        raise ValueError("Planned events do not describe every acquisition channel")
    return [record for record in records if record is not None]


def _position_records(
    image_events: Sequence[MDAEvent], count: int
) -> list[list[float]]:
    """Return one physical ZXY coordinate in storage-position order.

    Returns
    -------
    list[list[float]]
        Physical positions ordered by the acquisition ``p`` index.

    Raises
    ------
    ValueError
        If planned events do not describe every position.
    """
    positions: list[list[float] | None] = [None] * count
    for event in image_events:
        position_index = int(event.index["p"])
        if positions[position_index] is not None:
            continue
        stage = event.metadata.get("Stage", {})
        positions[position_index] = [
            _required_float(stage.get("z_pos", event.z_pos), "stage z"),
            _required_float(stage.get("x_pos", event.x_pos), "stage x"),
            _required_float(stage.get("y_pos", event.y_pos), "stage y"),
        ]
    if any(position is None for position in positions):
        raise ValueError("Planned events do not describe every acquisition position")
    return [position for position in positions if position is not None]


def _wavelength(name: object) -> float | None:
    """Extract a leading numeric wavelength from a channel name.

    Returns
    -------
    float or None
        Parsed wavelength, if present.
    """
    match = re.search(r"\d+(?:\.\d+)?", str(name))
    return float(match.group()) if match else None


def _optional_float(value: object) -> float | None:
    """Convert a present measurement to float while retaining null values.

    Returns
    -------
    float or None
        Floating-point measurement or ``None``.
    """
    return None if value is None else float(value)


def _optional_float_at(values: Sequence[object], index: int) -> float | None:
    """Return an optional floating-point list entry.

    Returns
    -------
    float or None
        Converted list entry or ``None`` when absent.
    """
    return _optional_float(values[index]) if index < len(values) else None


def _required_float(value: object, field: str) -> float:
    """Convert one required numeric field.

    Returns
    -------
    float
        Converted numeric value.

    Raises
    ------
    ValueError
        If the required value is absent.
    """
    if value is None:
        raise ValueError(f"Live acquisition manifest requires {field}")
    return float(value)


def _is_number(value: object) -> bool:
    """Return whether a value is a JSON number rather than a boolean.

    Returns
    -------
    bool
        Whether the value is numeric and not boolean.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _timestamp() -> str:
    """Return an ISO-8601 UTC timestamp.

    Returns
    -------
    str
        Current UTC timestamp.
    """
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
