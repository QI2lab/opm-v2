"""Shared file readers and writers for OPM command-line scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorstore as ts


def load_json(path: Path) -> Any:
    """Load a JSON document.

    Parameters
    ----------
    path : Path
        JSON document path.

    Returns
    -------
    Any
        Parsed JSON value.
    """
    return json.loads(path.read_text())


def save_json(value: Any, path: Path) -> Path:
    """Write a JSON document, creating its parent directory.

    Parameters
    ----------
    value : Any
        JSON-serializable value.
    path : Path
        Destination document path.

    Returns
    -------
    Path
        Written destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4))
    return path


def load_tensorstore(path: Path) -> np.ndarray:
    """Materialize a Zarr v3 array through TensorStore.

    Parameters
    ----------
    path : Path
        Path to a Zarr v3 array node, such as ``dataset.zarr/0``.

    Returns
    -------
    numpy.ndarray
        Materialized array.
    """
    store = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
        },
        open=True,
    ).result()
    return np.asarray(store.read().result())
