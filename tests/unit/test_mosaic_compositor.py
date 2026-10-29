"""Validate OPM mosaic blending and replacement semantics."""

from __future__ import annotations

import numpy as np
import pytest

from opm_v2.widgets._mosaic_compositor import (
    IncompatibleMosaicError,
    MosaicCompositor,
)


def _transform(x: float = 0, y: float = 0) -> np.ndarray:
    matrix = np.eye(4)
    matrix[0:2, 3] = (x, y)
    return matrix


def _only_chunk(updates: dict) -> np.ndarray:
    chunks = [chunk for chunk in updates.values() if chunk is not None]
    assert len(chunks) == 1
    return chunks[0].data


def test_single_tile_is_not_attenuated_by_feather_weights() -> None:
    """Validate single tile is not attenuated by feather weights."""
    compositor = MosaicCompositor(chunk_size=32, overlap=(50, 50))
    image = np.arange(24, dtype=np.uint16).reshape(4, 6)

    result = _only_chunk(compositor.add_image(image, _transform()))

    np.testing.assert_allclose(result, image)


def test_overlap_is_normalized_feather_blend() -> None:
    """Validate overlap is normalized feather blend."""
    compositor = MosaicCompositor(chunk_size=32, overlap=(50, 0))
    left = np.zeros((2, 4), dtype=np.uint16)
    right = np.full((2, 4), 100, dtype=np.uint16)
    compositor.add_image(left, _transform())

    result = _only_chunk(compositor.add_image(right, _transform(x=2)))

    expected = np.array([0, 0, 100 / 3, 200 / 3, 100, 100], dtype=np.float32)
    np.testing.assert_allclose(result, np.tile(expected, (2, 1)), rtol=1e-6)


def test_blending_is_independent_of_tile_arrival_order() -> None:
    """Validate blending is independent of tile arrival order."""
    anchor = np.full((2, 4), 25, dtype=np.uint8)
    left = np.zeros((2, 4), dtype=np.uint8)
    right = np.full((2, 4), 100, dtype=np.uint8)

    first = MosaicCompositor(chunk_size=64, overlap=(50, 0))
    first.add_image(anchor, _transform(x=-10))
    first.add_image(left, _transform())
    first_result = _only_chunk(first.add_image(right, _transform(x=2)))

    second = MosaicCompositor(chunk_size=64, overlap=(50, 0))
    second.add_image(anchor, _transform(x=-10))
    second.add_image(right, _transform(x=2))
    second_result = _only_chunk(second.add_image(left, _transform()))

    np.testing.assert_allclose(first_result, second_result)


def test_repeated_spatial_tile_replaces_previous_frame() -> None:
    """Validate repeated spatial tile replaces previous frame."""
    compositor = MosaicCompositor(chunk_size=16, overlap=25)
    transform = _transform(x=4, y=8)

    for value in range(1000):
        updates = compositor.add_image(
            np.full((4, 6), value, dtype=np.uint16), transform
        )

    assert compositor.tile_count == 1
    assert compositor.chunk_count == 1
    np.testing.assert_allclose(_only_chunk(updates), 999)


def test_replacement_updates_existing_overlap_without_reaccumulating() -> None:
    """Validate replacement updates existing overlap without reaccumulating."""
    compositor = MosaicCompositor(chunk_size=16, overlap=(50, 0))
    left_transform = _transform()
    compositor.add_image(np.zeros((2, 4), dtype=np.uint16), left_transform)
    compositor.add_image(np.full((2, 4), 100, dtype=np.uint16), _transform(x=2))

    result = _only_chunk(
        compositor.add_image(np.full((2, 4), 50, dtype=np.uint16), left_transform)
    )

    np.testing.assert_allclose(result[:, 2], 200 / 3, rtol=1e-6)


def test_new_tile_does_not_recompose_historical_tile_images() -> None:
    """Validate new tile does not recompose historical tile images."""
    compositor = MosaicCompositor(chunk_size=16, overlap=(50, 0))
    compositor.add_image(np.zeros((2, 4), dtype=np.uint16), _transform())
    # The accumulated chunk is sufficient when a new spatial tile arrives. This
    # sentinel makes the test fail if historical raw tiles are traversed again.
    compositor._tiles[(0, 0)].image = None  # type: ignore[assignment]

    updates = compositor.add_image(
        np.full((2, 4), 100, dtype=np.uint16), _transform(x=2)
    )

    np.testing.assert_allclose(_only_chunk(updates)[:, -1], 100, rtol=1e-6)


def test_chunks_remain_sparse_across_negative_and_positive_offsets() -> None:
    """Validate chunks remain sparse across negative and positive offsets."""
    compositor = MosaicCompositor(chunk_size=4)
    image = np.ones((2, 2), dtype=np.uint8)
    compositor.add_image(image, _transform())
    compositor.add_image(image, _transform(x=-6, y=-6))
    compositor.add_image(image, _transform(x=8, y=8))

    assert compositor.tile_count == 3
    assert compositor.chunk_count == 3


def test_uncovered_pixels_inside_a_chunk_are_transparent_nan() -> None:
    """Validate uncovered pixels inside a chunk are transparent nan."""
    compositor = MosaicCompositor(chunk_size=32)
    image = np.ones((2, 2), dtype=np.uint8)
    compositor.add_image(image, _transform())

    result = _only_chunk(compositor.add_image(image, _transform(x=8)))

    assert np.isnan(result[:, 2:8]).all()
    np.testing.assert_allclose(result[:, :2], 1)
    np.testing.assert_allclose(result[:, 8:], 1)


def test_rgb_images_are_blended_per_channel() -> None:
    """Validate rgb images are blended per channel."""
    compositor = MosaicCompositor(chunk_size=16, overlap=(50, 0))
    left = np.zeros((2, 4, 3), dtype=np.uint8)
    right = np.empty((2, 4, 3), dtype=np.uint8)
    right[...] = (30, 60, 90)
    compositor.add_image(left, _transform())

    result = _only_chunk(compositor.add_image(right, _transform(x=2)))

    np.testing.assert_allclose(result[:, -1], ((30, 60, 90),) * 2, rtol=1e-6)
    np.testing.assert_allclose(result[:, 2], ((10, 20, 30),) * 2, rtol=1e-6)


def test_changed_image_geometry_requires_a_new_session() -> None:
    """Validate changed image geometry requires a new session."""
    compositor = MosaicCompositor()
    compositor.add_image(np.zeros((2, 4), dtype=np.uint8), _transform())

    with pytest.raises(IncompatibleMosaicError, match="shape changed"):
        compositor.add_image(np.zeros((3, 4), dtype=np.uint8), _transform(x=2))

    rotated = _transform(x=2)
    rotated[:2, :2] = ((0, -1), (1, 0))
    with pytest.raises(IncompatibleMosaicError, match="affine changed"):
        compositor.add_image(np.zeros((2, 4), dtype=np.uint8), rotated)


def test_world_axis_overlap_maps_through_rotated_affine() -> None:
    """Validate world axis overlap maps through rotated affine."""
    compositor = MosaicCompositor(overlap=(50, 0))
    rotated = _transform()
    rotated[:2, :2] = ((0, -1), (1, 0))
    compositor.add_image(np.ones((8, 4), dtype=np.uint8), rotated)

    # World X is local image Y after a 90-degree rotation.
    assert compositor._feather_pixels(8, 4) == (0, 4)


def test_stage_translation_preserves_calibrated_affine_direction() -> None:
    """Validate stage translation preserves calibrated affine direction."""
    compositor = MosaicCompositor(chunk_size=32)
    base = _transform(x=100, y=200)
    base[:2, :2] = ((-2, 0), (0, 3))
    compositor.add_image(np.ones((2, 2), dtype=np.uint8), base)
    moved = base.copy()
    moved[0:2, 3] += (-8, 6)

    compositor.add_image(np.ones((2, 2), dtype=np.uint8), moved)

    assert (4, 2) in compositor._tiles
    np.testing.assert_allclose(compositor.chunk_transform((4, 2)), moved)


def test_clear_resets_tiles_chunks_and_session_geometry() -> None:
    """Validate clear resets tiles chunks and session geometry."""
    compositor = MosaicCompositor()
    compositor.add_image(np.ones((2, 2), dtype=np.uint8), _transform())

    compositor.clear()

    assert compositor.tile_count == 0
    assert compositor.chunk_count == 0
    compositor.add_image(np.ones((3, 3), dtype=np.uint8), _transform(x=20))
    assert compositor.tile_count == 1
