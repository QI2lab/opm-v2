"""Verify acquisition-side publication for the live-processing contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from useq import MDAEvent, MDASequence

import opm_v2.handlers.live_acquisition as live_acquisition
from opm_v2.engine.opm_custom_events import ACTION_ASI_SETUP_SCAN, ACTION_DAQ
from opm_v2.engine.setup_events import OPMEventBuilder, setup_mirrorscan
from opm_v2.handlers.live_acquisition import (
    LIVE_ACQUISITION_SCHEMA,
    LIVE_ACQUISITION_SCHEMA_VERSION,
    acquisition_sidecar_paths,
    validate_live_acquisition_manifest,
)
from opm_v2.handlers.opm_data_handler import OpmDataHandler


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    """Read complete non-empty JSON Lines records from a sidecar.

    Returns
    -------
    list[dict[str, Any]]
        Parsed lifecycle records.
    """
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    return [json.loads(line) for line in text.splitlines() if line]


def _contract_manifest(data_path: Path) -> dict[str, Any]:
    """Return a minimal version-1 manifest for lifecycle tests.

    Returns
    -------
    dict[str, Any]
        Schema-compatible acquisition manifest.
    """
    return {
        "schema": LIVE_ACQUISITION_SCHEMA,
        "schema_version": LIVE_ACQUISITION_SCHEMA_VERSION,
        "acquisition_id": "test-acquisition-id",
        "data_path": data_path.name,
        "mode": "mirror",
        "index_sizes": {"t": 1, "p": 1, "c": 1, "z": 1},
        "acquisition_order": ["t", "p", "c", "z"],
        "channels": [
            {
                "name": "488nm",
                "wavelength_nm": 488.0,
                "exposure_ms": 1.5,
                "laser_power": 12.0,
            }
        ],
        "stage_positions_zxy": [[30.0, 100.0, 200.0]],
        "scan_axis": "x",
        "scan_axis_step_um": 0.4,
        "pixel_size_um": 0.115,
        "angle_deg": 30.0,
        "camera_offset": 100.0,
        "camera_e_to_adu": 0.24,
        "excess_scan_positions": 0,
        "excess_scan_start_positions": 0,
        "excess_scan_end_positions": 0,
        "orientations": {
            "camera_XYstage_orientation": "positive",
            "camera_Zstage_orientation": "negative",
            "camera_mirror_orientation": "positive",
        },
    }


def test_planning_atomically_publishes_manifest_before_acquisition(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    monkeypatch,
) -> None:
    """Publish only a complete manifest while leaving the writer untouched."""
    output = workspace_tmp_path / "sample.ome.zarr"
    manifest_path, log_path = acquisition_sidecar_paths(output)
    real_replace = live_acquisition.os.replace
    replacements: list[dict[str, Any]] = []

    def inspect_atomic_replace(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == manifest_path:
            assert source_path.parent == manifest_path.parent
            assert source_path.suffix == ".tmp"
            assert not manifest_path.exists()
            replacements.append(json.loads(source_path.read_text(encoding="utf-8")))
        real_replace(source, destination)

    monkeypatch.setattr(live_acquisition.os, "replace", inspect_atomic_replace)
    config = opm_config_factory(
        mode="mirror",
        active_channels=(1,),
        channel_powers=(12.0,),
        channel_exposures_ms=(2.5,),
        scan_range_um=4.0,
        scan_axis_step_um=2.0,
    )
    events, handler = setup_mirrorscan(
        demo_core,
        config,
        MDASequence(stage_positions=[(100.0, 200.0, 30.0)], axis_order="tpcz"),
        output,
    )

    assert events
    assert len(replacements) == 1
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == replacements[0]
    assert not log_path.exists()
    assert not output.exists()
    assert handler.get_view() is None


def test_planned_manifest_is_schema_compatible(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
) -> None:
    """Publish every required version-1 field in processing-compatible order."""
    output = workspace_tmp_path / "schema.ome.zarr"
    config = opm_config_factory(
        mode="mirror",
        active_channels=(1,),
        channel_powers=(12.0,),
        channel_exposures_ms=(2.5,),
        scan_range_um=4.0,
        scan_axis_step_um=2.0,
    )
    _events, handler = setup_mirrorscan(
        demo_core,
        config,
        MDASequence(stage_positions=[(100.0, 200.0, 30.0)], axis_order="tpcz"),
        output,
    )
    manifest_path, _log_path = acquisition_sidecar_paths(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    validate_live_acquisition_manifest(manifest)
    assert manifest["schema"] == LIVE_ACQUISITION_SCHEMA
    assert manifest["schema_version"] == LIVE_ACQUISITION_SCHEMA_VERSION
    assert manifest["data_path"] == output.name
    assert manifest["mode"] == "mirror"
    assert manifest["index_sizes"] == {"t": 1, "p": 1, "c": 1, "z": 2}
    assert manifest["acquisition_order"] == ["t", "p", "z", "c"]
    assert manifest["channels"] == [
        {
            "name": "488nm",
            "wavelength_nm": 488.0,
            "exposure_ms": 2.5,
            "laser_power": 12.0,
        }
    ]
    assert manifest["stage_positions_zxy"] == [[30.0, 100.0, 200.0]]
    assert manifest["scan_axis_step_um"] == 2.0
    assert manifest["pixel_size_um"] == pytest.approx(demo_core.getPixelSizeUm())
    assert handler.acquisition_metadata["acq_config"]["opm_mode"] == "mirror"


def test_explicit_mirror_selection_overrides_stale_stage_metadata(
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
) -> None:
    """Publish the selected builder mode without mutating its source config."""
    output = workspace_tmp_path / "selected-mirror.ome.zarr"
    config = opm_config_factory(
        mode="stage",
        active_channels=(1,),
        channel_powers=(12.0,),
        channel_exposures_ms=(2.5,),
        scan_range_um=4.0,
        scan_axis_step_um=2.0,
    )

    events, handler = OPMEventBuilder(
        demo_core,
        config,
        MDASequence(stage_positions=[(100.0, 200.0, 30.0)], axis_order="tpcz"),
    ).build(output, mode="mirror")
    manifest_path, _log_path = acquisition_sidecar_paths(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    action_names = [getattr(event.action, "name", None) for event in events]
    daq_events = [
        event
        for event in events
        if getattr(event.action, "name", None) == ACTION_DAQ
    ]
    image_events = [event for event in events if "DAQ" in event.metadata]

    assert daq_events
    assert {event.action.data["DAQ"]["mode"] for event in daq_events} == {"mirror"}
    assert ACTION_ASI_SETUP_SCAN not in action_names
    assert image_events
    assert {event.metadata["DAQ"]["mode"] for event in image_events} == {"mirror"}
    assert manifest["mode"] == "mirror"
    assert handler.acquisition_metadata["acq_config"]["opm_mode"] == "mirror"
    assert config["acq_config"]["opm_mode"] == "stage"
    handler.close()


@pytest.mark.parametrize("terminal_event", ["completed", "canceled", "errored"])
def test_lifecycle_log_flushes_started_and_exactly_one_terminal(
    workspace_tmp_path,
    terminal_event,
) -> None:
    """Keep one pollable log containing lifecycle events and no tile records."""
    output = workspace_tmp_path / f"lifecycle-{terminal_event}.ome.zarr"
    event = MDAEvent(
        index={"t": 0, "p": 0, "c": 0, "z": 0},
        metadata={"DAQ": {"current_channel": "488nm"}},
    )
    handler = OpmDataHandler(
        path=output,
        index_sizes={"t": 1, "p": 1, "c": 1, "z": 1},
        delete_existing=True,
        acquisition_order=("t", "p", "c", "z"),
        events=(event,),
        live_manifest=_contract_manifest(output),
    )
    sequence = MDASequence()
    manifest_path, log_path = acquisition_sidecar_paths(output)
    log_path.write_text('{"event":"stale"}\n', encoding="utf-8")
    handler.sequenceStarted(sequence, {})

    started_records = _read_json_lines(log_path)
    assert [record["event"] for record in started_records] == ["started"]
    assert manifest_path.exists()

    if terminal_event == "completed":
        handler.frameReady(
            np.ones((2, 3), dtype=np.uint16),
            event,
            {"runner_time_ms": 0.0, "exposure_ms": 1.5},
        )
        assert [record["event"] for record in _read_json_lines(log_path)] == [
            "started"
        ]
        handler.sequenceFinished(sequence)
    elif terminal_event == "canceled":
        handler.sequenceCanceled(sequence)
        handler.sequenceFinished(sequence)
    else:
        handler.set_finish_reason_getter(lambda: "errored")
        handler.sequenceFinished(sequence)

    records = _read_json_lines(log_path)
    assert [record["event"] for record in records] == ["started", terminal_event]
    assert {record["acquisition_id"] for record in records} == {
        "test-acquisition-id"
    }
    assert all("timestamp" in record for record in records)
    assert not any("tile" in record["event"] for record in records)

    # Repeated cleanup callbacks cannot append a second terminal record.
    if terminal_event != "canceled":
        handler.sequenceFinished(sequence)
    assert _read_json_lines(log_path) == records
    assert list(output.parent.glob(f"{log_path.stem}*")) == [log_path]
