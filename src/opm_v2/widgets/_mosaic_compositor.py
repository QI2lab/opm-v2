"""OPM-owned Stage Explorer support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

type Overlap = float | tuple[float, float]
type ChunkKey = tuple[int, int]
type TileKey = tuple[int, int]


class IncompatibleMosaicError(ValueError):
    """Raised when an image cannot be added to the current mosaic session."""


@dataclass(frozen=True, slots=True)
class MosaicChunk:
    """A rendered mosaic chunk and its origin in common pixel coordinates."""

    data: np.ndarray
    pixel_origin: tuple[int, int]


@dataclass(slots=True)
class _Tile:
    image: np.ndarray
    x: int
    y: int

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass(slots=True)
class _ChunkAccumulator:
    image: np.ndarray
    weight_sum: np.ndarray
    x0: int
    y0: int
    x1: int = 0
    y1: int = 0


class MosaicCompositor:
    """Incrementally compose translated images with normalized feather weights.

    Images in one session must have the same shape and affine linear component.
    Translation is converted into the common image-pixel coordinate system. A frame
    at an existing spatial location replaces the prior frame, which keeps repeated
    timelapse or channel visits from accumulating indefinitely.
    """

    def __init__(self, *, chunk_size: int = 1024, overlap: Overlap = 0.0) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._chunk_size = int(chunk_size)
        self._overlap = _normalize_overlap(overlap)
        self._base_transform: np.ndarray | None = None
        self._image_shape: tuple[int, ...] | None = None
        self._tiles: dict[TileKey, _Tile] = {}
        self._chunks: dict[ChunkKey, _ChunkAccumulator] = {}
        self._weight_cache: dict[tuple[int, int], np.ndarray] = {}

    @property
    def tile_count(self) -> int:
        """The number of unique spatial tiles in the mosaic."""
        return len(self._tiles)

    @property
    def chunk_count(self) -> int:
        """The number of occupied mosaic chunks."""
        return len(self._chunks)

    @property
    def overlap(self) -> tuple[float, float]:
        """World-X and world-Y feather overlap percentages."""
        return self._overlap

    def clear(self) -> None:
        """Remove every tile and reset the session geometry."""
        self._base_transform = None
        self._image_shape = None
        self._tiles.clear()
        self._chunks.clear()
        self._weight_cache.clear()

    def set_overlap(self, overlap: Overlap) -> dict[ChunkKey, MosaicChunk | None]:
        """Set feather overlap and rebuild occupied chunks if it changed.

        Returns
        -------
        dict
            Rebuilt chunks keyed by spatial chunk index.
        """
        normalized = _normalize_overlap(overlap)
        if normalized == self._overlap:
            return {}
        self._overlap = normalized
        self._weight_cache.clear()
        tiles = list(self._tiles.values())
        self._chunks.clear()
        affected: set[ChunkKey] = set()
        for tile in tiles:
            affected.update(self._accumulate_new_tile(tile))
        return self._chunk_updates(affected)

    def add_image(
        self, image: np.ndarray, transform: np.ndarray
    ) -> dict[ChunkKey, MosaicChunk | None]:
        """Add or replace an image and return changed rendered chunks.

        Parameters
        ----------
        image : np.ndarray
            A two-dimensional grayscale or three-dimensional multichannel image.
        transform : np.ndarray
            A conventional row-major 4x4 transform from image pixels to stage world.

        Returns
        -------
        dict
            Rendered chunks changed by adding or replacing the image.
        """
        array = np.asarray(image)
        matrix = np.asarray(transform, dtype=float)
        self._validate_image(array)
        self._validate_transform(matrix)

        if self._base_transform is None:
            self._base_transform = matrix.copy()
            self._image_shape = tuple(array.shape)
        else:
            self._validate_session(array, matrix)

        tile_x, tile_y = self._pixel_offset(matrix)
        key = (tile_x, tile_y)
        tile = _Tile(np.array(array, copy=True), tile_x, tile_y)
        previous = self._tiles.get(key)
        if previous is None:
            affected = self._accumulate_new_tile(tile)
        else:
            affected = self._replace_tile(previous, tile)
        self._tiles[key] = tile
        return self._chunk_updates(affected)

    def chunk_transform(self, pixel_origin: tuple[int, int]) -> np.ndarray:
        """Return the row-major world transform for a rendered chunk.

        Returns
        -------
        numpy.ndarray
            Affine transform from chunk pixels to stage-world coordinates.

        Raises
        ------
        RuntimeError
            If no image has established the mosaic's coordinate system.
        """
        if self._base_transform is None:
            raise RuntimeError("The mosaic has no base transform")
        shift = np.eye(4)
        shift[0:2, 3] = pixel_origin
        return self._base_transform @ shift

    def _validate_image(self, image: np.ndarray) -> None:
        if image.ndim not in (2, 3):
            raise ValueError("Mosaic images must be 2D or 3D")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Mosaic images must not be empty")

    def _validate_transform(self, transform: np.ndarray) -> None:
        if transform.shape != (4, 4):
            raise ValueError("Transform must be a 4x4 matrix")
        if not np.all(np.isfinite(transform)):
            raise ValueError("Transform must contain only finite values")
        if not np.allclose(transform[-1], (0, 0, 0, 1)):
            raise ValueError("Transform must be a conventional row-major affine")
        if abs(np.linalg.det(transform[:2, :2])) < np.finfo(float).eps:
            raise ValueError("Transform linear component must be invertible")

    def _validate_session(self, image: np.ndarray, transform: np.ndarray) -> None:
        assert self._base_transform is not None
        if tuple(image.shape) != self._image_shape:
            raise IncompatibleMosaicError("Image shape changed within mosaic session")
        if not np.allclose(
            transform[:2, :2],
            self._base_transform[:2, :2],
            rtol=1e-6,
            atol=1e-9,
        ):
            raise IncompatibleMosaicError("Image affine changed within mosaic session")

    def _pixel_offset(self, transform: np.ndarray) -> tuple[int, int]:
        assert self._base_transform is not None
        linear = self._base_transform[:2, :2]
        world_delta = transform[:2, 3] - self._base_transform[:2, 3]
        pixel_delta = np.linalg.solve(linear, world_delta)
        return int(np.rint(pixel_delta[0])), int(np.rint(pixel_delta[1]))

    def _chunk_keys(self, tile: _Tile) -> set[ChunkKey]:
        size = self._chunk_size
        x_first = tile.x // size
        x_last = (tile.x + tile.width - 1) // size
        y_first = tile.y // size
        y_last = (tile.y + tile.height - 1) // size
        return {
            (chunk_x, chunk_y)
            for chunk_x in range(x_first, x_last + 1)
            for chunk_y in range(y_first, y_last + 1)
        }

    def _accumulate_new_tile(self, tile: _Tile) -> set[ChunkKey]:
        affected = self._chunk_keys(tile)
        weights = self._weight_for_shape(tile.height, tile.width)
        for chunk_key in affected:
            chunk = self._chunks.get(chunk_key)
            if chunk is None:
                chunk = self._new_chunk(tile.image)
                self._chunks[chunk_key] = chunk
            tile_y, tile_x, chunk_y, chunk_x = self._intersection(tile, chunk_key)
            values = tile.image[tile_y, tile_x].astype(np.float32, copy=False)
            contribution_weights = weights[tile_y, tile_x]
            old_weights = chunk.weight_sum[chunk_y, chunk_x]
            new_weights = old_weights + contribution_weights
            current = chunk.image[chunk_y, chunk_x]
            current_values = np.nan_to_num(current, nan=0.0)
            if values.ndim == 2:
                numerator = current_values * old_weights + values * contribution_weights
                np.divide(numerator, new_weights, out=current)
            else:
                old_weights_rgb = old_weights[..., np.newaxis]
                contribution_rgb = contribution_weights[..., np.newaxis]
                numerator = current_values * old_weights_rgb + values * contribution_rgb
                np.divide(numerator, new_weights[..., np.newaxis], out=current)
            old_weights[...] = new_weights
            self._expand_bounds(chunk, chunk_y, chunk_x)
        return affected

    def _replace_tile(self, previous: _Tile, tile: _Tile) -> set[ChunkKey]:
        affected = self._chunk_keys(tile)
        weights = self._weight_for_shape(tile.height, tile.width)
        for chunk_key in affected:
            chunk = self._chunks[chunk_key]
            tile_y, tile_x, chunk_y, chunk_x = self._intersection(tile, chunk_key)
            old_values = previous.image[tile_y, tile_x].astype(np.float32, copy=False)
            new_values = tile.image[tile_y, tile_x].astype(np.float32, copy=False)
            contribution_weights = weights[tile_y, tile_x]
            denominator = chunk.weight_sum[chunk_y, chunk_x]
            current = chunk.image[chunk_y, chunk_x]
            if current.ndim == 2:
                current += (
                    (new_values - old_values) * contribution_weights / denominator
                )
            else:
                current += (
                    (new_values - old_values)
                    * contribution_weights[..., np.newaxis]
                    / denominator[..., np.newaxis]
                )
        return affected

    def _new_chunk(self, image: np.ndarray) -> _ChunkAccumulator:
        size = self._chunk_size
        channels = () if image.ndim == 2 else (image.shape[2],)
        return _ChunkAccumulator(
            image=np.full((size, size, *channels), np.nan, dtype=np.float32),
            weight_sum=np.zeros((size, size), dtype=np.float32),
            x0=size,
            y0=size,
        )

    def _intersection(
        self, tile: _Tile, chunk_key: ChunkKey
    ) -> tuple[slice, slice, slice, slice]:
        size = self._chunk_size
        chunk_x0 = chunk_key[0] * size
        chunk_y0 = chunk_key[1] * size
        chunk_x1 = chunk_x0 + size
        chunk_y1 = chunk_y0 + size
        x0 = max(tile.x, chunk_x0)
        y0 = max(tile.y, chunk_y0)
        x1 = min(tile.x + tile.width, chunk_x1)
        y1 = min(tile.y + tile.height, chunk_y1)
        return (
            slice(y0 - tile.y, y1 - tile.y),
            slice(x0 - tile.x, x1 - tile.x),
            slice(y0 - chunk_y0, y1 - chunk_y0),
            slice(x0 - chunk_x0, x1 - chunk_x0),
        )

    def _expand_bounds(
        self, chunk: _ChunkAccumulator, chunk_y: slice, chunk_x: slice
    ) -> None:
        assert chunk_x.start is not None and chunk_x.stop is not None
        assert chunk_y.start is not None and chunk_y.stop is not None
        chunk.x0 = min(chunk.x0, chunk_x.start)
        chunk.y0 = min(chunk.y0, chunk_y.start)
        chunk.x1 = max(chunk.x1, chunk_x.stop)
        chunk.y1 = max(chunk.y1, chunk_y.stop)

    def _chunk_updates(
        self, chunk_keys: set[ChunkKey]
    ) -> dict[ChunkKey, MosaicChunk | None]:
        updates: dict[ChunkKey, MosaicChunk | None] = {}
        size = self._chunk_size
        for chunk_key in chunk_keys:
            chunk = self._chunks[chunk_key]
            origin = (
                chunk_key[0] * size + chunk.x0,
                chunk_key[1] * size + chunk.y0,
            )
            updates[chunk_key] = MosaicChunk(
                chunk.image[chunk.y0 : chunk.y1, chunk.x0 : chunk.x1], origin
            )
        return updates

    def _weight_for_shape(self, height: int, width: int) -> np.ndarray:
        cache_key = (height, width)
        cached = self._weight_cache.get(cache_key)
        if cached is not None:
            return cached
        feather_x, feather_y = self._feather_pixels(height, width)
        x_weights = _axis_weights(width, feather_x)
        y_weights = _axis_weights(height, feather_y)
        weights = np.asarray(
            y_weights[:, np.newaxis] * x_weights[np.newaxis, :], dtype=np.float32
        )
        self._weight_cache[cache_key] = weights
        return weights

    def _feather_pixels(self, height: int, width: int) -> tuple[int, int]:
        overlap_x, overlap_y = self._overlap
        fallback = np.array(
            [width * overlap_x / 100, height * overlap_y / 100], dtype=float
        )
        if self._base_transform is None:
            local = fallback
        else:
            extents = np.abs(self._base_transform[:2, :2])
            world_size = extents @ np.array([width, height], dtype=float)
            desired = world_size * np.array([overlap_x, overlap_y]) / 100
            try:
                local = np.linalg.solve(extents, desired)
            except np.linalg.LinAlgError:
                local = fallback
            if not np.all(np.isfinite(local)) or np.any(local < 0):
                local = fallback
        return (
            min(width, max(0, int(np.rint(local[0])))),
            min(height, max(0, int(np.rint(local[1])))),
        )


def _normalize_overlap(overlap: Overlap) -> tuple[float, float]:
    if isinstance(overlap, tuple):
        if len(overlap) != 2:
            raise ValueError("overlap must contain world-X and world-Y percentages")
        x_value, y_value = overlap
    else:
        x_value = y_value = overlap
    return max(0.0, float(x_value)), max(0.0, float(y_value))


def _axis_weights(length: int, feather: int) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    if feather <= 0:
        return weights
    ramp = np.arange(1, feather + 1, dtype=np.float32) / (feather + 1)
    weights[:feather] = np.minimum(weights[:feather], ramp)
    weights[-feather:] = np.minimum(weights[-feather:], ramp[::-1])
    return weights
