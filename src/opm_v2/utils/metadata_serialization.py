"""Convert scientific metadata to the native JSON types used by Zarr."""

import json
from pathlib import Path

import numpy as np


def _json_default(value):
    """Convert NumPy values and filesystem paths for the JSON encoder.

    Returns
    -------
    object
        Native scalar, list, or path string.

    Raises
    ------
    TypeError
        If the value has no supported metadata representation.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_metadata(value: dict) -> dict:
    """Validate and copy nested metadata into native JSON types before writing.

    Returns
    -------
    dict
        Serializable metadata, without modifying the caller's arrays or dicts.
    """
    return json.loads(json.dumps(value, default=_json_default))
