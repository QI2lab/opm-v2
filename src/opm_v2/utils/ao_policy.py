"""Choose AO initial conditions within one acquisition."""

from collections.abc import Hashable

import numpy as np
from numpy.typing import NDArray

USE_START = "use starting correction"
PREVIOUS_AO = "previous AO result"
PREVIOUS_Z = "corresponding position in previous Z layer"
SEQUENTIAL_CORRECTIONS = [USE_START, PREVIOUS_AO, PREVIOUS_Z]
STARTING_CORRECTIONS = ["system flat", "last optimized", "factory flat", "zeros"]


def correction_settings(config: dict) -> dict[str, str]:
    """Resolve explicit settings or preserve legacy AO initialization behavior.

    Returns
    -------
    dict[str, str]
        Starting and sequential correction settings.

    Raises
    ------
    ValueError
        If either setting is unknown.
    """
    starting = config.get(
        "starting_correction", config.get("mirror_state", "system flat")
    )
    default = (
        PREVIOUS_Z
        if "grid" in config.get("ao_mode", "")
        else PREVIOUS_AO
        if starting == "last optimized"
        else USE_START
    )
    sequential = config.get("sequential_correction", default)
    if starting not in STARTING_CORRECTIONS:
        raise ValueError(f"Invalid AO starting correction: {starting}")
    if sequential not in SEQUENTIAL_CORRECTIONS:
        raise ValueError(f"Invalid AO sequential correction: {sequential}")
    return {"starting_correction": starting, "sequential_correction": sequential}


class AOCorrectionHistory:
    """Own a fixed starting correction and completed results for one acquisition."""

    def __init__(self, mirror, config: dict) -> None:
        self.settings = correction_settings(config)
        starting = self.settings["starting_correction"]
        self.reference_state = {
            "system flat": "system_flat",
            "factory flat": "factory_flat",
            "zeros": "zeros_voltage",
            "last optimized": getattr(
                mirror, "optimized_reference_state", "system_flat"
            ),
        }[starting]
        self._seed = np.array(
            mirror.optimized_modal_coeffs
            if starting == "last optimized"
            else np.zeros_like(mirror.current_coeffs),
            copy=True,
        )
        self._latest: NDArray | None = None
        self._positions: dict[tuple[int, int, Hashable], NDArray] = {}

    def initializer(
        self, *, time_idx: int = 0, z_idx: int = 0, position: Hashable = 0
    ) -> dict:
        """Select a copied seed and provenance for an optimization.

        Returns
        -------
        dict
            Keyword arguments accepted by ``run_ao_optimization``.

        Raises
        ------
        RuntimeError
            If the preceding layer has no matching completed correction.
        """
        policy = self.settings["sequential_correction"]
        coefficients = self._seed
        source = "acquisition starting correction"
        if policy == PREVIOUS_AO and self._latest is not None:
            coefficients = self._latest
            source = PREVIOUS_AO
        elif policy == PREVIOUS_Z and z_idx > 0:
            key = (time_idx, z_idx - 1, position)
            if key not in self._positions:
                raise RuntimeError(f"Missing previous-Z AO correction for {key}")
            coefficients = self._positions[key]
            source = PREVIOUS_Z
        return {
            "starting_mirror_coefficients": coefficients.copy(),
            "starting_reference_state": self.reference_state,
            "starting_coefficients_source": source,
            "initialization_metadata": {
                **self.settings,
                "source": source,
                "reference_state": self.reference_state,
                "time_idx": time_idx,
                "z_idx": z_idx,
                "logical_position": position,
            },
        }

    def complete(
        self,
        coefficients: NDArray,
        *,
        time_idx: int = 0,
        z_idx: int = 0,
        position: Hashable = 0,
    ) -> None:
        """Record only a successfully completed optimization.

        Raises
        ------
        ValueError
            If completed coefficients are nonfinite or have the wrong shape.
        """
        coefficients = np.asarray(coefficients)
        if coefficients.shape != self._seed.shape or not np.all(
            np.isfinite(coefficients)
        ):
            raise ValueError("Invalid completed AO coefficients")
        self._latest = coefficients.copy()
        if self.settings["sequential_correction"] == PREVIOUS_Z:
            # Retain only this timepoint's current and preceding layers.
            self._positions = {
                key: value
                for key, value in self._positions.items()
                if key[0] == time_idx and key[1] >= z_idx - 1
            }
            self._positions[(time_idx, z_idx, position)] = coefficients.copy()
