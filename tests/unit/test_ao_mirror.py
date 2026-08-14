"""Test adaptive-optics mirror behavior with its in-memory backend."""

from unittest.mock import MagicMock

import numpy as np

from opm_v2.hardware.AOMirror import AOMirror


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
