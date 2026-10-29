"""Test acquisition-scoped AO initialization without physical hardware."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from opm_v2.utils.ao_policy import (
    PREVIOUS_AO,
    PREVIOUS_Z,
    USE_START,
    AOCorrectionHistory,
    correction_settings,
)


def _mirror():
    """Build mirror state without a device connection.

    Returns
    -------
    SimpleNamespace
        Mutable current and optimized states.
    """
    return SimpleNamespace(
        current_coeffs=np.array([0.2, 0.3]),
        optimized_modal_coeffs=np.array([0.4, 0.5]),
        optimized_reference_state="factory_flat",
        update_positions_array=MagicMock(),
    )


@pytest.mark.parametrize(
    "mode,starting,expected",
    [
        ("at xyz positions", "system flat", USE_START),
        ("at xyz positions", "last optimized", PREVIOUS_AO),
        ("grid at start", "system flat", PREVIOUS_Z),
        ("grid at timepoints", "last optimized", PREVIOUS_Z),
    ],
)
def test_legacy_policy_migration(mode, starting, expected):
    """Preserve existing behavior until explicit policies are saved."""
    settings = correction_settings({"ao_mode": mode, "mirror_state": starting})
    assert settings == {
        "starting_correction": starting,
        "sequential_correction": expected,
    }
    assert (
        correction_settings({**settings, "mirror_state": "zeros", "ao_mode": "none"})
        == settings
    )


@pytest.mark.parametrize(
    "starting,reference",
    [
        ("system flat", "system_flat"),
        ("factory flat", "factory_flat"),
        ("zeros", "zeros_voltage"),
        ("last optimized", "factory_flat"),
    ],
)
def test_starting_snapshot_is_independent_of_later_mirror_updates(starting, reference):
    """Capture coefficients and reference once, with no alias to mutable state."""
    mirror = _mirror()
    history = AOCorrectionHistory(
        mirror,
        {
            "starting_correction": starting,
            "sequential_correction": USE_START,
        },
    )
    expected = (
        mirror.optimized_modal_coeffs.copy()
        if starting == "last optimized"
        else np.zeros(2)
    )
    mirror.optimized_modal_coeffs[:] = 99
    mirror.optimized_reference_state = "zeros_voltage"
    history.complete(np.array([1.0, 2.0]))
    seed = history.initializer()
    assert seed["starting_reference_state"] == reference
    np.testing.assert_array_equal(seed["starting_mirror_coefficients"], expected)
    seed["starting_mirror_coefficients"][:] = -99
    np.testing.assert_array_equal(
        history.initializer()["starting_mirror_coefficients"], expected
    )


def test_previous_ao_chains_across_positions_and_timepoints():
    """The most recent completed run is shared within this acquisition only."""
    mirror = _mirror()
    settings = {
        "starting_correction": "system flat",
        "sequential_correction": PREVIOUS_AO,
    }
    history = AOCorrectionHistory(mirror, settings)
    np.testing.assert_array_equal(
        history.initializer()["starting_mirror_coefficients"], [0, 0]
    )
    history.complete(np.array([1, 2]), position=3)
    np.testing.assert_array_equal(
        history.initializer(time_idx=1, position=9)["starting_mirror_coefficients"],
        [1, 2],
    )
    next_acquisition = AOCorrectionHistory(mirror, settings)
    np.testing.assert_array_equal(
        next_acquisition.initializer()["starting_mirror_coefficients"], [0, 0]
    )


def test_previous_z_matches_positions_and_resets_at_new_timepoint():
    """Keep lateral histories separate and require an immediate predecessor."""
    history = AOCorrectionHistory(
        _mirror(),
        {
            "starting_correction": "system flat",
            "sequential_correction": PREVIOUS_Z,
        },
    )
    history.complete(np.array([1, 2]), position="a")
    np.testing.assert_array_equal(
        history.initializer(position="b")["starting_mirror_coefficients"], [0, 0]
    )
    history.complete(np.array([3, 4]), position="b")
    for position, expected in (("a", [1, 2]), ("b", [3, 4])):
        np.testing.assert_array_equal(
            history.initializer(z_idx=1, position=position)[
                "starting_mirror_coefficients"
            ],
            expected,
        )
    with pytest.raises(RuntimeError, match="Missing previous-Z"):
        history.initializer(z_idx=2, position="a")
    with pytest.raises(RuntimeError, match="Missing previous-Z"):
        history.initializer(time_idx=1, z_idx=1, position="a")
    np.testing.assert_array_equal(
        history.initializer(time_idx=1, position="a")["starting_mirror_coefficients"],
        [0, 0],
    )
