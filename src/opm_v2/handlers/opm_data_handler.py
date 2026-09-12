"""Write OPM acquisitions through ome-writers' TensorStore backend."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from os import PathLike
from pathlib import Path
from threading import Condition
from time import monotonic
from typing import TYPE_CHECKING, Any

import numpy as np
from ome_writers import (
    AcquisitionSettings,
    OMEStream,
    OmeZarrFormat,
    Position,
    create_stream,
    dims_from_standard_axes,
)

from opm_v2.engine.debug_printing import info
from opm_v2.engine.opm_custom_events import TILE_RETRY_ATTEMPT_METADATA_KEY
from opm_v2.handlers.live_acquisition import LiveAcquisitionPublisher

if TYPE_CHECKING:
    from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1
    from useq import MDAEvent, MDASequence


TIMELAPSE_CHUNK_MEMORY_BUDGET_BYTES = 64 * 1024**2
TILE_RETRY_STORAGE_DRAIN_TIMEOUT_S = 120.0


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
    events : Sequence[MDAEvent] or None
        Prepared camera events used to retain channel labels, stage coordinates,
        and physical axis scales in OME metadata.
    acquisition_metadata : Mapping[str, Any] or None
        Complete acquisition configuration to store with the OME-Zarr root.
    """

    def __init__(
        self,
        *,
        path: str | PathLike[str],
        index_sizes: Mapping[str, int],
        delete_existing: bool = False,
        acquisition_order: Sequence[str] | None = None,
        events: Sequence[MDAEvent] | None = None,
        acquisition_metadata: Mapping[str, Any] | None = None,
        max_time_chunk_size: int = 1,
        time_chunk_concurrency: int | None = None,
        live_manifest: Mapping[str, Any] | None = None,
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
        events : Sequence[MDAEvent] or None
            Prepared camera events describing semantic dimension coordinates.
        acquisition_metadata : Mapping[str, Any] or None
            Complete acquisition configuration persisted as global metadata.
        max_time_chunk_size : int
            Maximum number of adjacent timepoints written in one Zarr chunk.
        time_chunk_concurrency : int or None
            Number of temporal chunks included in the 64 MiB memory budget.
        live_manifest : Mapping[str, Any] or None
            Validated acquisition manifest published after event planning.

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
        if len(self.acquisition_order) != len(self.index_sizes) or set(
            self.acquisition_order
        ) != set(self.index_sizes):
            raise ValueError("acquisition_order must contain every indexed axis once")
        self._events = tuple(events or ())
        self._acquisition_metadata = dict(acquisition_metadata or {})
        self._max_time_chunk_size = max(1, int(max_time_chunk_size))
        self._time_chunk_concurrency = max(
            1,
            int(
                time_chunk_concurrency
                if time_chunk_concurrency is not None
                else self.index_sizes.get("c", 1)
            ),
        )
        self._resolved_time_chunk_size = 1
        self.delete_existing = bool(delete_existing)
        self._stream: OMEStream | None = None
        self._view: Any | None = None
        self._summary_meta: dict[str, Any] = {}
        self._next_frame = 0
        self._latest_event_index: dict[str, int] = {}
        self._frame_count = int(np.prod(tuple(self.index_sizes.values())))
        self._is_finalized = False
        self._was_canceled = False
        self._finish_reason_getter: Callable[[], object] | None = None
        self._tile_retry_range: tuple[int, int] | None = None
        self._tile_retry_metadata_slots: dict[
            tuple[int, tuple[int, ...]], int
        ] = {}
        self._write_condition = Condition()
        self._write_error: Exception | None = None
        self._publisher = (
            LiveAcquisitionPublisher(self.path, live_manifest)
            if live_manifest is not None
            else None
        )

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
    def acquisition_metadata(self) -> dict[str, Any]:
        """A copy of the metadata that will be stored at the Zarr root."""
        return dict(self._acquisition_metadata)

    @property
    def is_finalized(self) -> bool:
        """Whether sequence finalization completed successfully.

        Returns
        -------
        bool
            ``True`` after the active ome-writers stream has been finalized.
        """
        return self._is_finalized

    @property
    def was_canceled(self) -> bool:
        """Whether the most recent sequence ended by cancellation.

        Returns
        -------
        bool
            ``True`` only after ``sequenceCanceled`` is received.
        """
        return self._was_canceled

    @property
    def max_time_chunk_size(self) -> int:
        """Configured upper bound for temporal Zarr batching."""
        return self._max_time_chunk_size

    @property
    def time_chunk_concurrency(self) -> int:
        """Number of temporal chunk buffers included in the memory budget."""
        return self._time_chunk_concurrency

    def set_finish_reason_getter(self, getter: Callable[[], object]) -> None:
        """Provide access to the active MDA runner's completion reason.

        Parameters
        ----------
        getter : Callable[[], object]
            Callback returning a value such as ``FinishReason.ERRORED``.
        """
        self._finish_reason_getter = getter

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
        self._was_canceled = False
        self._summary_meta = dict(meta or {})
        self._next_frame = 0
        self._latest_event_index = {}
        self._view = None
        self._tile_retry_range = None
        self._tile_retry_metadata_slots = {}
        with self._write_condition:
            self._write_error = None
        if self._publisher is not None:
            self._publisher.started(
                frames_expected=self._frame_count,
                data_path=self.path.name,
            )
        info(
            "OPM IMAGE ACQUISITION STARTED",
            f"Expected frames: {self._frame_count}",
            f"Output: {self.path}",
        )

    def frameReady(self, frame: np.ndarray, event: MDAEvent, meta: FrameMetaV1) -> None:
        """Append a camera frame and publish a terminal error on write failure.

        Parameters
        ----------
        frame : numpy.ndarray
            Two-dimensional image delivered by the camera.
        event : MDAEvent
            Acquisition event associated with the frame.
        meta : FrameMetaV1
            Per-frame metadata emitted by pymmcore-plus.

        """
        try:
            self._write_frame(frame, event, meta)
        except Exception as exc:
            with self._write_condition:
                self._write_error = exc
                self._write_condition.notify_all()
            self._publish_terminal(
                "errored",
                frames_saved=self._next_frame,
                frames_expected=self._frame_count,
                error=str(exc),
            )
            raise
        else:
            with self._write_condition:
                self._write_condition.notify_all()

    def _write_frame(
        self, frame: np.ndarray, event: MDAEvent, meta: FrameMetaV1
    ) -> None:
        """Write one validated frame at its planned output position.

        Raises
        ------
        ValueError
            If the frame is not two-dimensional or arrives out of order.
        IndexError
            If the event lies outside the configured acquisition shape.
        """
        image = np.asarray(frame)
        if image.ndim != 2:
            raise ValueError(f"OPM frames must be 2D; received shape {image.shape}")
        if self._stream is None:
            self._stream = self._create_stream(image)
            # The acquisition dimensions are known before the first frame.  A
            # fixed-shape view gives NDV its sliders immediately and avoids a
            # coordinate-expansion callback for every newly reached plane.
            self._view = self._stream.view(dynamic_shape=False, strict=False)

        target_frame = self._flat_event_index(event)
        if target_frame >= self._frame_count:
            raise IndexError(
                f"Event index {dict(event.index)} exceeds configured frame count"
            )
        retry_attempt = int(
            event.metadata.get(TILE_RETRY_ATTEMPT_METADATA_KEY, 0)
        )
        if target_frame < self._next_frame:
            if retry_attempt <= 0 or not self._is_tile_retry_frame(target_frame):
                raise ValueError(
                    f"Event index {dict(event.index)} arrived after its output position"
                )
            self._rewrite_frame(
                image,
                event,
                self._frame_metadata(event, meta),
                target_frame,
            )
            self._latest_event_index = {
                ("p" if str(axis) == "g" else str(axis)): int(index)
                for axis, index in event.index.items()
            }
            self._finish_tile_retry_if_last(target_frame)
            return
        if target_frame > self._next_frame:
            self._stream.skip(frames=target_frame - self._next_frame)

        self._stream.append(image, frame_metadata=self._frame_metadata(event, meta))
        self._next_frame = target_frame + 1
        self._latest_event_index = {
            ("p" if str(axis) == "g" else str(axis)): int(index)
            for axis, index in event.index.items()
        }
        self._finish_tile_retry_if_last(target_frame)

    def prepare_tile_retry(
        self,
        events: Sequence[MDAEvent],
        attempt: int,
        received_frames: int | None = None,
    ) -> None:
        """Prepare the current OME-Zarr tile for complete in-place replacement.

        Frames already appended for a failed tile are flushed before the retry
        begins.  Retried frames then overwrite their original pixels and
        metadata at the same OME-Zarr coordinates; frames that were never
        received by the failed attempt continue through the normal append path.

        Parameters
        ----------
        events : Sequence[MDAEvent]
            Every camera event in the hardware-triggered tile being restarted.
        attempt : int
            One-based retry attempt number.
        received_frames : int or None
            Number of frames emitted by the failed hardware attempt.  The MDA
            runner dispatches output handlers asynchronously, so retry setup
            must wait for those already-emitted callbacks before snapshotting
            the metadata slots that will be replaced.

        Raises
        ------
        RuntimeError
            If the active writer cannot safely perform indexed replacement.
        ValueError
            If the tile does not occupy one contiguous acquisition-order range.
        """
        if not events:
            raise ValueError("Cannot retry an empty camera tile")

        flat_indices = [self._flat_event_index(event) for event in events]
        first = min(flat_indices)
        last = max(flat_indices)
        if sorted(flat_indices) != list(range(first, last + 1)):
            raise ValueError(
                "Retried camera tile must be contiguous in acquisition order"
            )

        if received_frames is not None:
            if not 0 <= received_frames <= len(events):
                raise ValueError(
                    "received_frames must lie within the retried tile; received "
                    f"{received_frames} for {len(events)} planned frames"
                )
            self._wait_for_saved_frame_count(first + received_frames)

        self._tile_retry_range = (first, last)
        self._tile_retry_metadata_slots = {}
        if self._stream is None:
            if self._next_frame > first:  # pragma: no cover - invalid lifecycle
                raise RuntimeError(
                    "OME stream is unavailable after frames were already saved"
                )
            info(
                "OPM TILE STORAGE REWRITE",
                f"Retry attempt: {attempt}",
                f"Frame range: {first}-{last}",
                "No failed-pass frames reached storage; retry will append all frames",
            )
            return

        backend = self._stream._backend
        futures = getattr(backend, "_futures", None)
        if futures is None:
            raise RuntimeError(
                "Tile replacement requires the TensorStore OME-Zarr backend"
            )
        # Complete every write from the failed attempt before issuing
        # replacement writes to the same array coordinates.
        while futures:
            futures.pop(0).result()

        chunk_buffers = getattr(backend, "_chunk_buffers", None)
        if chunk_buffers:
            # A camera timeout can leave part of a temporal chunk in memory.
            # Those frames belong to the failed pass and must not later be
            # flushed over the restarted tile.  Complete chunks are already on
            # disk and are replaced as the retry traverses them again.
            for chunk_buffer in chunk_buffers:
                chunk_buffer.flush_all_partial()

        metadata_slots: dict[tuple[int, tuple[int, ...]], int] = {}
        retry_positions = {
            self._storage_route(event)[0] for event in events
        }
        for position_index in retry_positions:
            group_path = backend._image_group_paths[position_index]
            mirror = backend._meta_mirrors[group_path]
            for slot, frame_metadata in enumerate(mirror._frame_metadata):
                storage_index = tuple(
                    int(index) for index in frame_metadata["storage_index"]
                )
                metadata_slots[(position_index, storage_index)] = slot

        self._tile_retry_metadata_slots = metadata_slots
        info(
            "OPM TILE STORAGE REWRITE",
            f"Retry attempt: {attempt}",
            f"Frame range: {first}-{last}",
            "Every previously saved frame in this tile will be replaced",
        )

    def _wait_for_saved_frame_count(self, expected_next_frame: int) -> None:
        """Wait for asynchronous writer callbacks emitted before tile failure.

        Raises
        ------
        RuntimeError
            If a queued writer callback failed.
        TimeoutError
            If the output-handler relay did not drain within the bounded wait.
        """
        deadline = monotonic() + TILE_RETRY_STORAGE_DRAIN_TIMEOUT_S
        with self._write_condition:
            while (
                self._next_frame < expected_next_frame
                and self._write_error is None
            ):
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for queued OME-Zarr writes before tile "
                        f"retry: saved through frame {self._next_frame}, expected "
                        f"{expected_next_frame}"
                    )
                self._write_condition.wait(remaining)

            if self._write_error is not None:
                raise RuntimeError(
                    "OME-Zarr writer failed before tile retry preparation"
                ) from self._write_error

    def _is_tile_retry_frame(self, target_frame: int) -> bool:
        """Return whether a repeated frame belongs to the active tile retry.

        Returns
        -------
        bool
            Whether the frame lies inside the active replacement range.
        """
        if self._tile_retry_range is None:
            return False
        first, last = self._tile_retry_range
        return first <= target_frame <= last

    def _finish_tile_retry_if_last(self, target_frame: int) -> None:
        """Release rewrite bookkeeping after the tile's final frame is stored."""
        if (
            self._tile_retry_range is not None
            and target_frame == self._tile_retry_range[1]
        ):
            self._tile_retry_range = None
            self._tile_retry_metadata_slots = {}

    def _storage_route(self, event: MDAEvent) -> tuple[int, tuple[int, ...]]:
        """Map an event index to the active backend's position and storage index.

        Returns
        -------
        tuple
            Position-array index and storage-order array index.

        Raises
        ------
        RuntimeError
            If no OME stream is active.
        """
        if self._stream is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("OME stream is not active")
        settings = self._stream._settings
        dimensions = settings.dimensions[:-2]
        position_axis = settings.position_dimension_index
        acquisition_indices = [
            int(event.index.get(dimension.name, 0))
            for axis, dimension in enumerate(dimensions)
            if axis != position_axis
        ]
        permutation = settings.storage_index_permutation
        if permutation is not None:
            acquisition_indices = [
                acquisition_indices[index] for index in permutation
            ]
        position_index = (
            int(event.index.get(dimensions[position_axis].name, 0))
            if position_axis is not None
            else 0
        )
        return position_index, tuple(acquisition_indices)

    def _rewrite_frame(
        self,
        image: np.ndarray,
        event: MDAEvent,
        frame_metadata: dict[str, Any],
        target_frame: int,
    ) -> None:
        """Replace one frame and its metadata without advancing stream order.

        Raises
        ------
        RuntimeError
            If the stream is inactive or no original metadata slot exists.
        """
        if self._stream is None:  # pragma: no cover - guarded by frameReady
            raise RuntimeError("OME stream is not active")
        position_index, storage_index = self._storage_route(event)
        backend = self._stream._backend
        slot_key = (position_index, storage_index)
        try:
            metadata_slot = self._tile_retry_metadata_slots[slot_key]
        except KeyError as exc:
            raise RuntimeError(
                "No original metadata slot exists for retried frame "
                f"{target_frame} at position {position_index}, "
                f"storage index {storage_index}"
            ) from exc

        backend.write(
            position_index,
            storage_index,
            image,
            frame_metadata=None,
        )
        group_path = backend._image_group_paths[position_index]
        mirror = backend._meta_mirrors[group_path]
        replacement = {**frame_metadata, "storage_index": storage_index}
        with mirror._lock:
            mirror._frame_metadata[metadata_slot] = replacement
            mirror._dirty = True

    def sequenceFinished(self, _sequence: MDASequence) -> None:
        """Close the writer after successful sequence completion.

        Parameters
        ----------
        _sequence : MDASequence
            Completed sequence.

        """
        try:
            self.close()
        except Exception as exc:
            self._publish_terminal(
                "errored",
                frames_saved=self._next_frame,
                frames_expected=self._frame_count,
                error=str(exc),
            )
            raise
        # pymmcore-plus emits sequenceCanceled followed by sequenceFinished.
        # A cooperative STOP intentionally leaves the planned array incomplete,
        # so cancellation owns finalization and must not trigger the missing-frame
        # error used for unexpectedly truncated acquisitions.
        if self._was_canceled:
            return
        finish_reason = (
            self._finish_reason_getter()
            if self._finish_reason_getter is not None
            else None
        )
        if str(finish_reason).casefold().endswith("errored"):
            self._is_finalized = False
            self._publish_terminal(
                "errored",
                frames_saved=self._next_frame,
                frames_expected=self._frame_count,
            )
            info(
                "OPM IMAGE ACQUISITION ERRORED",
                f"Frames saved: {self._next_frame} of {self._frame_count}",
                "See the preceding acquisition exception for the root cause",
            )
            return
        if self._next_frame != self._frame_count:
            error = RuntimeError(
                "OPM acquisition finished with "
                f"{self._next_frame} of {self._frame_count} expected frames"
            )
            self._publish_terminal(
                "errored",
                frames_saved=self._next_frame,
                frames_expected=self._frame_count,
                error=str(error),
            )
            raise error
        self._is_finalized = True
        self._publish_terminal(
            "completed",
            frames_saved=self._next_frame,
            frames_expected=self._frame_count,
        )
        info(
            "OPM IMAGE ACQUISITION COMPLETE",
            f"Frames saved: {self._next_frame}",
            f"Output: {self.path}",
        )

    def sequenceCanceled(self, _sequence: MDASequence) -> None:
        """Close the writer after sequence cancellation.

        Parameters
        ----------
        _sequence : MDASequence
            Canceled sequence.
        """
        try:
            self.close()
        except Exception as exc:
            self._publish_terminal(
                "errored",
                frames_saved=self._next_frame,
                frames_expected=self._frame_count,
                error=str(exc),
            )
            raise
        self._is_finalized = False
        self._was_canceled = True
        self._publish_terminal(
            "canceled",
            frames_saved=self._next_frame,
            frames_expected=self._frame_count,
        )
        info(
            "OPM IMAGE ACQUISITION CANCELED",
            f"Frames saved: {self._next_frame} of {self._frame_count}",
            f"Output: {self.path}",
        )

    def _publish_terminal(self, event: str, **diagnostics: Any) -> None:
        """Publish one terminal lifecycle event when sidecars are enabled."""
        if self._publisher is not None:
            self._publisher.terminal(event, **diagnostics)

    def close(self) -> None:
        """Flush and close the active ome-writers stream."""
        if self._stream is not None:
            backend = self._stream._backend
            # TensorStore waits for futures before the base backend flushes
            # partial buffered chunks.  Flush them first so cancellation and
            # non-divisible time series cannot leave a write running after the
            # arrays are released.
            if getattr(backend, "_chunk_buffers", None):
                backend._finalize_chunk_buffers()
                futures = getattr(backend, "_futures", None)
                while futures:
                    futures.pop().result()
            self._stream.close()
            self._stream = None

    def get_view(self) -> Any | None:
        """Return a live array view for the native MDA preview widget.

        Returns
        -------
        Any or None
            Dynamic OME stream view after the first frame, otherwise ``None``.
        """
        return self._view

    def get_preview_state(self) -> tuple[int, dict[str, int]]:
        """Return the latest saved-frame count and dimensional index for NDV.

        Returns
        -------
        tuple[int, dict[str, int]]
            Number of frames appended and a copy of the latest event index.
        """
        return self._next_frame, dict(self._latest_event_index)

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
        standard_sizes = {
            axis: self._semantic_axis_value(axis) for axis in self.acquisition_order
        }
        standard_sizes.update({"y": height, "x": width})
        dimensions = dims_from_standard_axes(standard_sizes)
        self._resolved_time_chunk_size = self._time_chunk_size(frame)
        dimensions = [
            dimension.model_copy(update={"chunk_size": self._resolved_time_chunk_size})
            if dimension.name == "t"
            else dimension
            for dimension in dimensions
        ]
        scales = {
            "t": self._axis_scale("t"),
            "z": self._axis_scale("z"),
            "y": pixel_size_um,
            "x": pixel_size_um,
        }
        dimensions = [
            dimension.model_copy(update={"scale": scales[dimension.name]})
            if scales.get(dimension.name) is not None
            else dimension
            for dimension in dimensions
        ]
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
        self._set_global_metadata(stream)
        return stream

    def _set_global_metadata(self, stream: OMEStream) -> None:
        """Update acquisition-level OME-Zarr metadata on an active stream."""
        stream.set_global_metadata(
            "opm_v2",
            {
                "index_sizes": dict(self.index_sizes),
                "acquisition_order": list(self.acquisition_order),
                "summary_metadata": _json_safe(self._summary_meta),
                "configuration": _json_safe(self._acquisition_metadata),
                "storage_backend": "tensorstore",
                "time_chunk_size": self._resolved_time_chunk_size,
            },
        )

    def _time_chunk_size(self, frame: np.ndarray) -> int:
        """Resolve temporal batching without allowing large memory spikes.

        Returns
        -------
        int
            Number of adjacent timepoints stored in one Zarr chunk.
        """
        timepoints = self.index_sizes.get("t", 1)
        bytes_per_timepoint = max(
            1,
            int(frame.nbytes) * self._time_chunk_concurrency,
        )
        memory_limited_size = max(
            1,
            TIMELAPSE_CHUNK_MEMORY_BUDGET_BYTES // bytes_per_timepoint,
        )
        return min(
            self._max_time_chunk_size,
            timepoints,
            memory_limited_size,
        )

    def _semantic_axis_value(self, axis: str) -> int | list[str | Position]:
        """Return semantic coordinates for one indexed axis.

        Parameters
        ----------
        axis : str
            useq axis identifier.

        Returns
        -------
        int or list[str or Position]
            Count, channel labels, or physical stage positions.
        """
        count = self.index_sizes[axis]
        if axis == "c":
            labels = [str(index) for index in range(count)]
            for event in self._events:
                if "c" in event.index:
                    label = event.metadata.get("DAQ", {}).get("current_channel")
                    if label is not None:
                        labels[int(event.index["c"])] = str(label)
            return labels
        if axis == "p":
            positions = [Position(name=str(index)) for index in range(count)]
            for event in self._events:
                if "p" not in event.index:
                    continue
                index = int(event.index["p"])
                stage = event.metadata.get("Stage", {})
                positions[index] = Position(
                    name=str(index),
                    x_coord=_optional_float(stage.get("x_pos")),
                    y_coord=_optional_float(stage.get("y_pos")),
                    z_coord=_optional_float(stage.get("z_pos")),
                )
            return positions
        return count

    def _axis_scale(self, axis: str) -> float | None:
        """Infer a physical axis scale from prepared OPM camera events.

        Parameters
        ----------
        axis : str
            Indexed axis name.

        Returns
        -------
        float or None
            Scale in the axis unit, when represented by the event metadata.
        """
        if axis == "z":
            for event in self._events:
                daq = event.metadata.get("DAQ", {})
                for key in ("scan_axis_step_um", "image_mirror_step_um"):
                    if daq.get(key) is not None:
                        return abs(float(daq[key]))
        if axis == "t":
            times = sorted({
                float(event.min_start_time)
                for event in self._events
                if event.min_start_time is not None
            })
            if len(times) > 1:
                return min(
                    later - earlier
                    for earlier, later in zip(times, times[1:], strict=False)
                    if later > earlier
                )
        return None

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


def _optional_float(value: Any) -> float | None:
    """Convert an optional coordinate to a floating-point value.

    Parameters
    ----------
    value : Any
        Coordinate value or ``None``.

    Returns
    -------
    float or None
        Floating-point coordinate when one was supplied.
    """
    return None if value is None else float(value)
