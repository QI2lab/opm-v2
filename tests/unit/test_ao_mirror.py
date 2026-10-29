"""Test adaptive-optics mirror behavior with its in-memory backend."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from opm_v2.hardware.AOMirror import AOMirror


@pytest.mark.parametrize("simulate", [True, False])
@pytest.mark.parametrize("invalid_voltage", [np.nan, np.inf, -np.inf, 0.99, -0.99, 1.1])
def test_invalid_voltages_never_reach_mirror(
    simulate: bool, invalid_voltage: float
) -> None:
    """Reject invalid voltages before hardware access or simulated state changes."""
    mirror = AOMirror(simulate=True)
    before = mirror.current_voltage.copy()
    requested = before.copy()
    requested[0] = invalid_voltage
    mirror.simulate = simulate
    mirror.wfc = MagicMock()

    assert mirror.set_mirror_voltage(requested) is False

    mirror.wfc.move_to_absolute_positions.assert_not_called()
    np.testing.assert_array_equal(mirror.current_voltage, before)


@pytest.mark.parametrize("simulate", [True, False])
@pytest.mark.parametrize("invalid_coefficient", [np.nan, np.inf, -np.inf])
def test_nonfinite_modal_coefficients_do_not_change_mirror(
    simulate: bool, invalid_coefficient: float
) -> None:
    """Reject nonfinite coefficients before WaveKit conversion or state mutation."""
    mirror = AOMirror(simulate=True)
    before = mirror.current_coeffs.copy()
    requested = before.copy()
    requested[0] = invalid_coefficient
    mirror.simulate = simulate
    mirror.modal_coeff = MagicMock()
    mirror.set_mirror_voltage = MagicMock()

    assert mirror.set_modal_coefficients(requested) is False

    mirror.modal_coeff.set_coefs_values.assert_not_called()
    mirror.set_mirror_voltage.assert_not_called()
    np.testing.assert_array_equal(mirror.current_coeffs, before)


def test_simulated_overvoltage_preserves_accepted_coefficients() -> None:
    """Keep both coefficient and voltage state intact when a command is rejected."""
    mirror = AOMirror(simulate=True)
    accepted = np.full(mirror.current_coeffs.shape, 0.1, dtype=np.float32)
    assert mirror.set_modal_coefficients(accepted)
    before_voltage = mirror.current_voltage.copy()
    requested = accepted.copy()
    requested[0] = 100.0

    assert mirror.set_modal_coefficients(requested) is False

    np.testing.assert_array_equal(mirror.current_coeffs, accepted)
    np.testing.assert_array_equal(mirror.current_voltage, before_voltage)


def test_last_optimized_restores_the_saved_accepted_correction() -> None:
    """Use one coefficient field for saving and restoring the optimized state."""
    mirror = AOMirror(simulate=True)
    accepted = np.full(mirror.current_coeffs.shape, 0.1, dtype=np.float32)
    assert mirror.set_modal_coefficients(accepted)
    mirror.update_optimized_array()
    accepted_voltage = mirror.current_voltage.copy()
    assert mirror.apply_system_flat_voltage()

    assert mirror.apply_optimized_voltage()

    np.testing.assert_array_equal(mirror.current_coeffs, accepted)
    np.testing.assert_array_equal(mirror.optimized_modal_coeffs, accepted)
    np.testing.assert_array_equal(mirror.current_voltage, accepted_voltage)


def test_simulated_ao_mirror_preserves_position_corrections() -> None:
    """Verify saved modal corrections round-trip through mirror positions."""
    mirror = AOMirror(simulate=True, n_modes=4, n_positions=2)
    correction = np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32)

    mirror.set_reference_state("system_flat")
    assert mirror.set_modal_coefficients(correction) is True
    mirror.update_positions_array(1)
    saved_voltage = mirror.current_voltage.copy()

    assert AOMirror.instance() is mirror
    np.testing.assert_allclose(mirror.current_coeffs, correction)
    assert np.any(saved_voltage != mirror.system_flat_voltage)

    mirror.apply_system_flat_voltage()
    assert np.all(mirror.current_coeffs == 0)
    assert mirror.apply_positions_array(1) is True
    np.testing.assert_allclose(mirror.current_coeffs, correction)
    np.testing.assert_allclose(mirror.current_voltage, saved_voltage)


def test_deformable_mirror_arrays_are_written_with_neutral_names(
    tmp_path: Path,
) -> None:
    """Persist both voltage and modal amplitudes for a non-optimized state."""
    mirror = AOMirror(simulate=True, n_modes=4, n_positions=2)
    mirror.output_path = tmp_path
    mirror.positions_modal_array[0] = np.asarray([0.1, -0.2, 0.3, -0.4])

    mirror.save_positions_array(prefix="ao_position")

    modal_path = tmp_path / "ao_position_mode_amplitude.json"
    voltage_path = tmp_path / "ao_position_voltage.json"
    assert modal_path.is_file()
    assert voltage_path.is_file()
    np.testing.assert_allclose(
        json.loads(modal_path.read_text(encoding="utf-8"))[0],
        [0.1, -0.2, 0.3, -0.4],
    )


def test_physical_ao_mirror_disconnect_is_idempotent() -> None:
    """Release the WaveKit connection only once during layered teardown."""
    mirror = object.__new__(AOMirror)
    mirror.simulate = False
    mirror._wfc_connected = True
    mirror.wfc = MagicMock()

    mirror.disconnect()
    mirror.disconnect()

    mirror.wfc.disconnect.assert_called_once_with()
    assert not mirror._wfc_connected


@pytest.mark.parametrize("reference", ["factory_flat", "zeros_voltage", "system_flat"])
def test_optimized_correction_restores_its_original_reference(reference):
    """Last optimized retains the baseline used when its coefficients were found."""
    mirror = AOMirror(simulate=True)
    mirror.set_reference_state(reference)
    coefficients = np.zeros_like(mirror.current_coeffs)
    coefficients[7] = 0.15
    assert mirror.set_modal_coefficients(coefficients)
    mirror.update_optimized_array()
    assert mirror.apply_system_flat_voltage()
    assert mirror.apply_optimized_voltage()
    assert mirror._reference_state == reference
    np.testing.assert_allclose(mirror.current_coeffs, coefficients)
