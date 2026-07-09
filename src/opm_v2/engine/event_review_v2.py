"""Save, load, and validate OPM MDA event structures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from useq import MDAEvent

from opm_v2.engine.debug_printing_v2 import info

REQUIRED_IMAGE_METADATA_KEYS = ("DAQ", "Camera", "OPM", "Stage")


def event_to_jsonable(event: MDAEvent) -> dict:
    """Convert an MDAEvent to a JSON-safe dictionary."""
    try:
        return event.model_dump(mode="json")
    except TypeError:
        return json.loads(json.dumps(event.model_dump(), default=str))


def _event_action_name(event: MDAEvent) -> str | None:
    """Return the custom action name for an event, if present."""
    action = getattr(event, "action", None)
    return getattr(action, "name", None)


def _event_index_dict(event: MDAEvent) -> dict:
    """Return an event index as a normal dict."""
    if event.index is None:
        return {}
    return dict(event.index)


def summarize_event_structure(
    opm_events: list[MDAEvent],
    handler: Any = None,
) -> dict:
    """Summarize event order, action counts, image indices, and metadata keys."""
    action_counts: Counter[str] = Counter()
    image_count = 0
    index_max: dict[str, int] = {}
    index_counts: Counter[str] = Counter()
    metadata_key_counts: Counter[str] = Counter()
    metadata_daq_modes: Counter[str] = Counter()
    metadata_channels: Counter[str] = Counter()
    warnings: list[str] = []

    for event_idx, event in enumerate(opm_events):
        action_name = _event_action_name(event)
        if action_name:
            action_counts[action_name] += 1
            continue

        image_count += 1
        index_dict = _event_index_dict(event)
        index_counts[str(index_dict)] += 1
        for key, value in index_dict.items():
            if isinstance(value, int):
                index_max[key] = max(index_max.get(key, 0), value)

        metadata = event.metadata or {}
        missing_keys = [
            key for key in REQUIRED_IMAGE_METADATA_KEYS if key not in metadata
        ]
        if missing_keys:
            warnings.append(
                f"event {event_idx}: image metadata missing keys {missing_keys}"
            )

        for key in metadata.keys():
            metadata_key_counts[key] += 1

        daq_metadata = metadata.get("DAQ", {})
        if "mode" in daq_metadata:
            metadata_daq_modes[str(daq_metadata["mode"])] += 1
        if "current_channel" in daq_metadata:
            metadata_channels[str(daq_metadata["current_channel"])] += 1

    duplicate_indices = {
        index: count for index, count in index_counts.items() if count > 1
    }
    if duplicate_indices:
        warnings.append(f"duplicate image indices found: {duplicate_indices}")

    handler_summary = None
    if handler is not None:
        handler_summary = {
            "type": type(handler).__name__,
            "path": str(getattr(handler, "path", "")),
            "indice_sizes": getattr(handler, "indice_sizes", None),
        }

    return {
        "event_count": len(opm_events),
        "image_count": image_count,
        "custom_action_count": sum(action_counts.values()),
        "action_counts": dict(action_counts),
        "index_max": index_max,
        "metadata_key_counts": dict(metadata_key_counts),
        "metadata_daq_modes": dict(metadata_daq_modes),
        "metadata_channels": dict(metadata_channels),
        "handler": handler_summary,
        "warnings": warnings,
    }


def validate_event_structure(opm_events: list[MDAEvent], handler: Any = None) -> dict:
    """Return a structured validation report for an OPM event list."""
    summary = summarize_event_structure(opm_events, handler)
    errors: list[str] = []

    if summary["image_count"] == 0:
        errors.append("No image events found.")

    if not summary["metadata_key_counts"]:
        errors.append("No image metadata found.")

    if handler is not None and summary["handler"]:
        indice_sizes = summary["handler"].get("indice_sizes")
        if isinstance(indice_sizes, dict):
            for axis, max_index in summary["index_max"].items():
                size = indice_sizes.get(axis)
                if size is not None and max_index >= int(size):
                    errors.append(
                        f"Index axis {axis} has max {max_index}, "
                        f"but handler size is {size}."
                    )

    return {
        "ok": not errors and not summary["warnings"],
        "errors": errors,
        "summary": summary,
    }


def save_event_structure_review(
    opm_events: list[MDAEvent],
    filepath: Path,
    handler: Any = None,
    include_events: bool = True,
) -> dict:
    """Save an event review JSON file and return the validation report."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    report = validate_event_structure(opm_events, handler)
    payload = {"review": report}
    if include_events:
        payload["events"] = [
            {"event_index": idx, **event_to_jsonable(event)}
            for idx, event in enumerate(opm_events)
        ]

    with open(filepath, "w") as file:
        json.dump(payload, file, indent=2, default=str)

    return report


def load_event_structure_review(filepath: Path) -> dict:
    """Load a saved event review JSON file."""
    with open(filepath, "r") as file:
        return json.load(file)


def print_event_structure_review(review: dict) -> None:
    """Print a compact event review summary to the console."""
    report = review.get("review", review)
    summary = report.get("summary", {})
    lines = [
        f"OK: {report.get('ok')}",
        f"Events: {summary.get('event_count')}",
        f"Images: {summary.get('image_count')}",
        f"Custom actions: {summary.get('custom_action_count')}",
        f"Action counts: {summary.get('action_counts')}",
        f"Index max: {summary.get('index_max')}",
        f"DAQ modes: {summary.get('metadata_daq_modes')}",
        f"Channels: {summary.get('metadata_channels')}",
    ]
    for error in report.get("errors", []):
        lines.append(f"ERROR: {error}")
    for warning in summary.get("warnings", []):
        lines.append(f"WARNING: {warning}")
    info("OPM EVENT STRUCTURE REVIEW", *lines)
