from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import warnings
from itertools import product
from os import PathLike
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pymmcore_plus.metadata.serialize import json_dumps, json_loads
from pymmcore_plus.mda.handlers._util import position_sizes
from pymmcore_plus.mda.handlers import TensorStoreHandler

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal, TypeAlias

    import tensorstore as ts
    import useq
    from typing_extensions import Self  # py311

    from pymmcore_plus.metadata import FrameMetaV1, SummaryMetaV1

    TsDriver: TypeAlias = Literal["zarr", "zarr3", "n5", "neuroglancer_precomputed"]
    EventKey: TypeAlias = frozenset[tuple[str, int]]

FRAME_DIM = "frame"

class OPMMirrorHandler(TensorStoreHandler):
    """qi2lab tensorstore handler for writing mirror-driven OPM MDA sequences.

    Modified class from original code to handle all hardware-triggered acquisitions
    used in the qi2lab OPM. Because our events do not collapse into an `MDASequence`,
    we have to set up the structure of the store differently than the normal dynamic
    check done in the standard `TensorStoreHandler`.

    The only difference is this handler requires a dictionary with the maximum indice
    for 't', 'p', 'c', 'z'.

    For example,
    ```python
    indice_sizes = {
        't' : np.max(1, n_time_steps),
        'p' : np.max(1, n_stage_pos),
        'c' : np.max(1, n_active_channels),
        'z' : np.max(1, n_scan_steps)
    }
    handler = OPMMirrorHandler(
        path=Path("c:/example_ts.zarr"),
        indice_sizes=indice_sizes,
        delete_existing=True
    )
    ```
    """
    def __init__(
        self,
        *,
        driver: TsDriver = "zarr",
        indice_sizes: dict,
        kvstore: str | dict | None = "memory://",
        path: str | PathLike | None = None,
        delete_existing: bool = False,
        spec: Mapping | None = None,
    ) -> None:
        super().__init__(
            driver=driver,
            kvstore=kvstore,
            path=path,
            delete_existing=delete_existing,
            spec=spec,
        )
        self._indice_sizes = indice_sizes

    def new_store(
        self, frame: np.ndarray, seq: useq.MDASequence | None, meta: FrameMetaV1
    ) -> ts.Future[ts.TensorStore]:
        # Use our overridden get_shape_chunks_labels (which ignores seq)
        shape, chunks, labels = self.get_shape_chunks_labels(frame.shape, seq)
        # Do not assign _nd_storage as done in the parent
        return self._ts.open(
            self.get_spec(),
            create=True,
            delete_existing=self.delete_existing,
            dtype=self._ts.dtype(frame.dtype),
            shape=shape,
            chunk_layout=self._ts.ChunkLayout(chunk_shape=chunks),
            domain=self._ts.IndexDomain(labels=labels),
        )

    def get_shape_chunks_labels(
        self, frame_shape: tuple[int, ...], seq: useq.MDASequence | None
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
        # Custom implementation using provided _indice_sizes, ignoring seq.
        max_sizes = dict(self._indice_sizes)
        # Remove axes with length 0 and unpack keys and sizes.
        labels, sizes = zip(*(x for x in max_sizes.items() if x[1]))
        full_shape: tuple[int, ...] = (*sizes, *frame_shape)
        chunks = [1] * len(full_shape)
        chunks[-len(frame_shape):] = frame_shape
        labels = (*labels, "y", "x")
        return full_shape, tuple(chunks), labels

    def finalize_metadata(self) -> None:
        """Finalize and flush metadata to storage with updated zarr write."""
        if not (store := self._store) or not store.kvstore:
            return  # pragma: no cover

        metadata = {"frame_metadatas": [m[1] for m in self.frame_metadatas]}
        if not self._nd_storage:
            metadata["frame_indices"] = [
                (tuple(dict(k).items()), v)  # type: ignore
                for k, v in self._frame_indices.items()
            ]
        
        if self.ts_driver.startswith("zarr"):
            store.kvstore.write(".zattrs", json_dumps(metadata).decode("utf-8")).result()
        elif self.ts_driver == "n5":  # pragma: no cover
            attrs = json_loads(store.kvstore.read("attributes.json").result().value)
            attrs.update(metadata)
            store.kvstore.write("attributes.json", json_dumps(attrs).decode("utf-8"))
