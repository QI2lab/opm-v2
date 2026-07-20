"""Write OPM acquisitions through ome-writers' TensorStore backend."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ome_writers import (
    AcquisitionSettings,
    Dimension,
    OMEStream,
    OmeZarrFormat,
    create_stream,
)

if TYPE_CHECKING:
    from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1
    from useq import MDAEvent, MDASequence


class OpmDataHandler:
    """Write indexed acquisitions using the public ome-writers stream API.

    Parameters
    ----------
    path : str or PathLike[str]
        Destination ``.zarr`` or ``.ome.zarr`` path.
    index_sizes : Mapping[str, int]
        Size of each indexed acquisition axis.
    delete_existing : bool
        Whether ome-writers may replace an existing destination.
    acquisition_order : Sequence[str] or None
        Frame-arrival axis order. Defaults to ``index_sizes`` insertion order.
    """

    def __init__(
        self,
        *,
        path: str | PathLike[str],
        index_sizes: Mapping[str, int],
        delete_existing: bool = False,
        acquisition_order: Sequence[str] | None = None,
    ) -> None:
        """Initialize an indexed TensorStore-backed acquisition writer.

        Parameters
        ----------
        path : str or PathLike[str]
            Destination ``.zarr`` or ``.ome.zarr`` path.
        index_sizes : Mapping[str, int]
            Size of each indexed acquisition axis.
        delete_existing : bool
            Whether ome-writers may replace an existing destination.
        acquisition_order : Sequence[str] or None
            Frame-arrival axis order.

        Raises
        ------
        ValueError
            If the indexed shape is empty or the acquisition order is invalid.
        """
        self.path = Path(path)
        self.index_sizes = {
            str(axis): int(size) for axis, size in index_sizes.items() if int(size) > 0
        }
        if not self.index_sizes:
            raise ValueError("index_sizes must contain at least one positive axis size")
        self.acquisition_order = tuple(acquisition_order or self.index_sizes)
        if set(self.acquisition_order) != set(self.index_sizes):
            raise ValueError("acquisition_order must contain every indexed axis once")
        self.delete_existing = bool(delete_existing)
        self._stream: OMEStream | None = None
        self._summary_meta: dict[str, Any] = {}
        self._next_frame = 0
        self._frame_count = int(np.prod(tuple(self.index_sizes.values())))
        self._is_finalized = False

    @property
    def indice_sizes(self) -> dict[str, int]:
        """Indexed shape used by event-review tooling.

        Returns
        -------
        dict[str, int]
            Copy of the acquisition axis sizes.
        """
        return dict(self.index_sizes)

    @property
    def is_finalized(self) -> bool:
        """Whether sequence finalization completed successfully.

        Returns
        -------
        bool
            ``True`` after the active ome-writers stream has been finalized.
        """
        return self._is_finalized

    def sequenceStarted(
        self, _sequence: MDASequence, meta: SummaryMetaV1 | dict[str, Any]
    ) -> None:
        """Reset writer state when an MDA sequence starts.

        Parameters
        ----------
        _sequence : MDASequence
            Sequence announced by pymmcore-plus.
        meta : SummaryMetaV1 or dict[str, Any]
            Acquisition summary metadata.
        """
        self.close()
        self._is_finalized = False
        self._summary_meta = dict(meta or {})
        self._next_frame = 0

    def frameReady(self, frame: np.ndarray, event: MDAEvent, meta: FrameMetaV1) -> None:
        """Append a camera frame at its indexed output position.

        Parameters
        ----------
        frame : numpy.ndarray
            Two-dimensional image delivered by the camera.
        event : MDAEvent
            Acquisition event associated with the frame.
        meta : FrameMetaV1
            Per-frame metadata emitted by pymmcore-plus.

        Raises
        ------
        ValueError
            If the frame is not two-dimensional or arrives out of order.
        """
        image = np.asarray(frame)
        if image.ndim != 2:
            raise ValueError(f"OPM frames must be 2D; received shape {image.shape}")
        if self._stream is None:
            self._stream = self._create_stream(image)

        target_frame = self._flat_event_index(event)
        if target_frame < self._next_frame:
            raise ValueError(
                f"Event index {dict(event.index)} arrived after its output position"
            )
        if target_frame > self._next_frame:
            self._stream.skip(frames=target_frame - self._next_frame)

        self._stream.append(image, frame_metadata=self._frame_metadata(event, meta))
        self._next_frame = target_frame + 1

    def sequenceFinished(self, _sequence: MDASequence) -> None:
        """Close the writer after successful sequence completion.

        Parameters
        ----------
        _sequence : MDASequence
            Completed sequence.
        """
        self.close()
        self._is_finalized = True

    def sequenceCanceled(self, _sequence: MDASequence) -> None:
        """Close the writer after sequence cancellation.

        Parameters
        ----------
        _sequence : MDASequence
            Canceled sequence.
        """
        self.close()
        self._is_finalized = True

    def close(self) -> None:
        """Flush and close the active ome-writers stream."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _create_stream(self, frame: np.ndarray) -> OMEStream:
        """Create and describe the ome-writers TensorStore stream.

        Parameters
        ----------
        frame : numpy.ndarray
            First camera frame, used to establish shape and data type.

        Returns
        -------
        OMEStream
            Configured writable OME stream.
        """
        image_info = next(iter(self._summary_meta.get("image_infos", ())), {})
        pixel_size_um = image_info.get("pixel_size_um")
        height, width = frame.shape
        dimensions = [
            self._dimension(axis, self.index_sizes[axis])
            for axis in self.acquisition_order
        ]
        dimensions.extend([
            Dimension(
                name="y",
                count=height,
                chunk_size=height,
                type="space",
                unit="micrometer",
                scale=pixel_size_um,
            ),
            Dimension(
                name="x",
                count=width,
                chunk_size=width,
                type="space",
                unit="micrometer",
                scale=pixel_size_um,
            ),
        ])
        settings = AcquisitionSettings(
            root_path=str(self.path),
            dimensions=dimensions,
            dtype=str(frame.dtype),
            format=OmeZarrFormat(
                backend="tensorstore",
                suffix=(
                    ".ome.zarr" if self.path.name.endswith(".ome.zarr") else ".zarr"
                ),
            ),
            storage_order="ome",
            overwrite=self.delete_existing,
        )
        stream = create_stream(settings)
        stream.set_global_metadata(
            "opm_v2",
            {
                "index_sizes": dict(self.index_sizes),
                "acquisition_order": list(self.acquisition_order),
                "summary_metadata": _json_safe(self._summary_meta),
            },
        )
        return stream

    def _dimension(self, axis: str, count: int) -> Dimension:
        """Create an ome-writers dimension for an acquisition axis.

        Parameters
        ----------
        axis : str
            useq axis identifier.
        count : int
            Number of positions along the axis.

        Returns
        -------
        Dimension
            Described ome-writers dimension.
        """
        kwargs: dict[str, Any] = {"name": axis, "count": count, "chunk_size": 1}
        if axis == "p":
            kwargs["type"] = "position"
        elif axis == "c":
            kwargs["type"] = "channel"
        elif axis == "t":
            kwargs.update(type="time", unit="second")
        elif axis == "z":
            kwargs.update(type="space", unit="micrometer")
        else:
            kwargs["type"] = "other"
        return Dimension(**kwargs)

    def _flat_event_index(self, event: MDAEvent) -> int:
        """Convert an event's multidimensional index to stream order.

        Parameters
        ----------
        event : MDAEvent
            Camera event containing useq indices.

        Returns
        -------
        int
            Flat frame index in the writer stream.

        Raises
        ------
        IndexError
            If an event index falls outside a configured dimension.
        """
        indices = tuple(
            int(event.index.get(axis, 0)) for axis in self.acquisition_order
        )
        shape = tuple(self.index_sizes[axis] for axis in self.acquisition_order)
        for axis, index, size in zip(
            self.acquisition_order, indices, shape, strict=True
        ):
            if not 0 <= index < size:
                raise IndexError(
                    f"Event axis {axis!r} index {index} exceeds size {size}"
                )
        return int(np.ravel_multi_index(indices, shape))

    @staticmethod
    def _frame_metadata(event: MDAEvent, meta: FrameMetaV1) -> dict[str, Any]:
        """Convert camera and event metadata to ome-writers fields.

        Parameters
        ----------
        event : MDAEvent
            Event that produced the camera frame.
        meta : FrameMetaV1
            Frame metadata supplied by pymmcore-plus.

        Returns
        -------
        dict[str, Any]
            JSON-safe per-frame metadata.
        """
        result: dict[str, Any] = {
            "event_index": dict(event.index),
            "delta_t": float(meta.get("runner_time_ms", 0.0)) / 1000.0,
            "exposure_time": float(
                meta.get(
                    "exposure_ms", event.exposure if event.exposure is not None else 0
                )
            )
            / 1000.0,
        }
        if position := meta.get("position"):
            result.update({
                f"position_{axis}": float(value)
                for axis, value in position.items()
                if axis in "xyz" and value is not None
            })
        if event.metadata:
            result["event_metadata"] = _json_safe(event.metadata)
        return result


def _json_safe(value: Any) -> Any:
    """Convert a value to a JSON-compatible object.

    Parameters
    ----------
    value : Any
        Arbitrary metadata value.

    Returns
    -------
    Any
        Equivalent value containing only JSON-compatible types.
    """
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json", exclude_unset=True))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value
