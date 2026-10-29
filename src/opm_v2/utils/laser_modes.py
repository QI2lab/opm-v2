"""Shared names and validation for OPM laser illumination modes."""

import math
from collections.abc import Sequence

INTERLEAVED = "interleaved"
SIMULTANEOUS = "simultaneous 488/637"
SIMULTANEOUS_561 = "simultaneous 561/637"
SIMULTANEOUS_488_561 = "simultaneous 488/561"
COMBINED_CHANNEL = "488 + 637"
SIMULTANEOUS_CHANNELS = ("488nm", "637nm")
COMBINED_CHANNELS = {
    COMBINED_CHANNEL: SIMULTANEOUS_CHANNELS,
    "561 + 637": ("561nm", "637nm"),
    "488 + 561": ("488nm", "561nm"),
}
SIMULTANEOUS_PRESETS = {
    SIMULTANEOUS: COMBINED_CHANNEL,
    SIMULTANEOUS_561: "561 + 637",
    SIMULTANEOUS_488_561: "488 + 561",
}


def normalize_laser_mode(mode: str | None) -> str | None:
    """Map the legacy simultaneous setting to its original pair.

    Returns
    -------
    str
        Explicit laser mode.
    """
    return SIMULTANEOUS if mode == "simultaneous" else mode


def is_simultaneous(mode: str | None) -> bool:
    """Identify simultaneous illumination.

    Returns
    -------
    bool
        Whether the setting selects a simultaneous pair.
    """
    return normalize_laser_mode(mode) in SIMULTANEOUS_PRESETS


def laser_mode_for_channel(channel: str) -> str:
    """Resolve a stored channel name.

    Returns
    -------
    str
        Corresponding illumination mode.
    """
    return next(
        (mode for mode, name in SIMULTANEOUS_PRESETS.items() if name == channel),
        INTERLEAVED,
    )


def validate_interleaved_exposures(
    exposures: Sequence[float],
    names: Sequence[str] | None = None,
) -> None:
    """Require identical positive exposures for a camera-clocked channel cycle.

    Raises
    ------
    ValueError
        If an active exposure is invalid or differs from the others.
    """
    values = [float(value) for value in exposures]
    if (
        not values
        or any(not math.isfinite(value) or value <= 0 for value in values)
        or any(value != values[0] for value in values)
    ):
        labels = (
            names
            if names is not None
            else [f"channel {i + 1}" for i in range(len(values))]
        )
        settings = ", ".join(
            f"{name}: {value:g} ms" for name, value in zip(labels, values, strict=True)
        )
        raise ValueError(
            "Interleaved acquisition requires equal, positive exposure times for "
            "all enabled channels because the camera clocks the laser and mirror waveform. "
            f"Set the enabled channels to the same exposure. Current settings: {settings}"
        )


def simultaneous_channel_indices(
    channel_names: Sequence[str], mode: str = SIMULTANEOUS
) -> tuple[int, int]:
    """Locate the physical lasers contributing to the combined channel.

    Returns
    -------
    tuple[int, int]
        Physical channel indices for the selected laser pair.

    Raises
    ------
    ValueError
        If either required wavelength is absent.
    """
    names = list(channel_names)
    pair = COMBINED_CHANNELS[SIMULTANEOUS_PRESETS[normalize_laser_mode(mode)]]
    if any(name not in names for name in pair):
        raise ValueError(
            f"Simultaneous mode requires configured {pair[0]} and {pair[1]} channels"
        )
    return names.index(pair[0]), names.index(pair[1])


def validate_simultaneous_exposures(
    exposures: Sequence[float], mode: str = SIMULTANEOUS
) -> None:
    """Require one finite, positive exposure shared by both physical lasers.

    Raises
    ------
    ValueError
        If the two exposures differ or are not finite and positive.
    """
    if (
        len(exposures) != 2
        or any(
            not math.isfinite(float(value)) or float(value) <= 0 for value in exposures
        )
        or not math.isclose(
            float(exposures[0]), float(exposures[1]), rel_tol=0, abs_tol=1e-9
        )
    ):
        raise ValueError(
            f"Simultaneous {SIMULTANEOUS_PRESETS[normalize_laser_mode(mode)]} "
            "requires equal, positive exposure settings for both lasers."
        )
