"""OPM mosaic rendering over the stock StageViewer.

Adapted from pymmcore-widgets (BSD-3-Clause); see LICENSE.pymmcore-widgets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymmcore_widgets.control._stage_explorer._stage_viewer import (
    StageViewer as BaseStageViewer,
)
from vispy import scene
from vispy.scene.visuals import Image

from ._mosaic_compositor import (
    IncompatibleMosaicError,
    MosaicChunk,
    MosaicCompositor,
    Overlap,
)

if TYPE_CHECKING:
    from collections.abc import Hashable


class StageViewer(BaseStageViewer):
    """Keep feather blending and mosaic storage inside OPM."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._blending_enabled = False
        self._blend_session_key = None
        self._mosaic = MosaicCompositor()
        self._mosaic_visuals = {}

    @property
    def blending_enabled(self) -> bool:
        """Whether incoming images are composed into a feathered mosaic."""
        return self._blending_enabled

    @property
    def mosaic_tile_count(self) -> int:
        """The number of unique spatial tiles in the blended mosaic."""
        return self._mosaic.tile_count

    @property
    def mosaic_chunk_count(self) -> int:
        """The number of rendered chunks in the blended mosaic."""
        return self._mosaic.chunk_count

    def set_blending_enabled(self, enabled: bool) -> None:
        """Enable or disable normalized feather blending.

        Changing renderer clears existing images so blended and independently drawn
        visuals are never mixed in one view.
        """
        enabled = bool(enabled)
        if enabled == self._blending_enabled:
            return
        self.clear()
        self._blend_session_key = None
        self._blending_enabled = enabled

    def set_blending_overlap(self, overlap: Overlap) -> None:
        """Set world-X and world-Y overlap percentages for feather blending."""
        updates = self._mosaic.set_overlap(overlap)
        if self._blending_enabled:
            self._apply_mosaic_updates(updates)

    def begin_blending_session(self, key: Hashable | None = None) -> None:
        """Begin or continue a mosaic session identified by ``key``.

        A changed key clears the display before the next frame. Applications can use
        this to prevent images with incompatible channels or acquisition settings from
        being blended together.
        """
        if key == self._blend_session_key:
            return
        self.clear()
        self._blend_session_key = key

    def add_image(self, img: np.ndarray, transform: np.ndarray | None = None) -> None:
        """Add an image to the scene with the given transform.

        Parameters
        ----------
        img : np.ndarray
            The image to add to the scene. It should be a (Y, X) or (Y, X, 3) array.
        transform : np.ndarray | None
            The transform to apply to the image. It should be a 4x4 matrix.
            If None, the image will be added with the identity transform.
            The transformation is indented to be calculated elsewhere (in higher level
            widgets) based on, e.g., the stage position, pixel size, configuration
            affine, etc.  This is a relatively low-level, direct function.
        """
        row_transform, vispy_transform = _normalize_transform(transform)

        if self._blending_enabled:
            try:
                updates = self._mosaic.add_image(img, row_transform)
            except IncompatibleMosaicError:
                self._clear_image_visuals()
                self._mosaic.clear()
                updates = self._mosaic.add_image(img, row_transform)
            self._apply_mosaic_updates(updates)
            return

        # add the image to the scene with the transform
        # texture_format="auto" uses GPUScaledTexture2D so that clim changes
        # only update a shader uniform instead of re-uploading the texture.
        frame = Image(
            img,
            cmap=self._cmap.to_vispy(),
            parent=self.view.scene,
            clim="auto" if self._clims is None else self._clims,
            texture_format="auto",
        )
        # keep the added image on top of the others
        frame.order = min(child.order for child in self._get_images()) - 1
        frame.transform = scene.MatrixTransform(matrix=vispy_transform)

    def clear(self) -> None:
        """Clear the scene."""
        self._clear_image_visuals()
        self._mosaic.clear()

    def _clear_image_visuals(self) -> None:
        """Detach image visuals while retaining non-image scene decorations."""
        for child in reversed(self.view.scene.children):
            if isinstance(child, Image):
                child.parent = None
        self._mosaic_visuals.clear()

    def _apply_mosaic_updates(
        self, updates: dict[tuple[int, int], MosaicChunk | None]
    ) -> None:
        for chunk_key, chunk in updates.items():
            visual = self._mosaic_visuals.get(chunk_key)
            if chunk is None:
                if visual is not None:
                    visual.parent = None
                    del self._mosaic_visuals[chunk_key]
                continue

            matrix = self._mosaic.chunk_transform(chunk.pixel_origin).T
            if visual is None:
                visual = Image(
                    chunk.data,
                    cmap=self._cmap.to_vispy(),
                    parent=self.view.scene,
                    clim="auto" if self._clims is None else self._clims,
                    texture_format="auto",
                )
                self._mosaic_visuals[chunk_key] = visual
            else:
                visual.set_data(chunk.data)
            visual.transform = scene.MatrixTransform(matrix=matrix)


def _normalize_transform(
    transform: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conventional row-major and VisPy transform matrices.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Conventional and transposed VisPy affine matrices.

    Raises
    ------
    ValueError
        If the transform is not a 4-by-4 affine matrix.
    """
    if transform is None:
        row_major = np.eye(4)
    else:
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("Transform must be a 4x4 matrix.")
        if np.allclose(matrix[-1], (0, 0, 0, 1)):
            row_major = matrix
        elif np.allclose(matrix[:, -1], (0, 0, 0, 1)):
            row_major = matrix.T
        else:
            raise ValueError("Transform must be an affine 4x4 matrix.")
    return row_major, row_major.T
