"""Test NI-DAQ waveform generation with its in-memory backend."""

from __future__ import annotations

import numpy as np
import pytest

from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ


def test_simulated_projection_waveform_runs_through_singleton() -> None:
    """Verify projection waveforms and playback through the shared DAQ instance."""
    with pytest.raises(RuntimeError, match="initialized before instance"):
        OPMNIDAQ.instance()

    daq = OPMNIDAQ(
        simulate=True,
        exposure_ms=10.0,
        image_mirror_calibration=0.04,
        projection_mirror_calibration=0.005,
    )

    daq.set_acquisition_params(
        scan_type="projection",
        channel_states=[True, False, True, False, False],
        image_mirror_range_um=20.0,
        laser_blanking=True,
        exposure_ms=10.0,
    )
    daq.generate_waveforms()
    daq.program_daq_waveforms()
    daq.start_waveform_playback()

    assert OPMNIDAQ.instance() is daq
    with pytest.raises(RuntimeError, match="already initialized"):
        OPMNIDAQ(simulate=True)
    assert daq.running() is True
    assert daq.scan_type == "projection"
    assert daq.channel_states == [True, False, True, False, False]
    assert daq.digital_waveform.shape == (4, 8)
    assert daq.analog_waveform.shape == (101, 2)
    np.testing.assert_allclose(daq.analog_waveform[0], [-0.4, 0.05])
    np.testing.assert_allclose(daq.analog_waveform[-2], [0.4, -0.05])
    np.testing.assert_allclose(daq.analog_waveform[-1], [-0.4, 0.05])

    daq.set_mirror_neutral_position(image_mirror_v=0.25, projection_mirror_v=-0.5)
    daq.reset_ao_channels()
    assert daq.mirror_neutral_positions == (0.25, -0.5)

    daq.stop_waveform_playback()
    assert daq.running() is False
