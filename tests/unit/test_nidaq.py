"""Test NI-DAQ waveform generation with its in-memory backend."""

from __future__ import annotations

import numpy as np
import pytest

from opm_v2.hardware.mock_nidaq import (
    MockDAQRoutingError,
    MockDAQTaskError,
    MockOPMNIDAQ,
)
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ


def test_simulated_projection_waveform_runs_through_singleton() -> None:
    """Verify projection waveforms and playback through the shared DAQ instance."""
    with pytest.raises(RuntimeError, match="initialized before instance"):
        OPMNIDAQ.instance()

    daq = MockOPMNIDAQ(
        name="MockDev",
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
        MockOPMNIDAQ()
    assert daq.running() is True
    assert len(daq.tasks) == 3
    assert all(task.valid and task.running for task in daq.tasks)
    assert daq.reserved_routes
    assert all(route.startswith("/MockDev/") for route in daq.reserved_routes)
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
    assert all(task.valid and not task.running for task in daq.tasks)


def test_mock_daq_reset_invalidates_tasks_and_blocks_routing_until_ready() -> None:
    """Model the NI reset window and reject stale task handles."""
    daq = MockOPMNIDAQ(exposure_ms=10.0)
    daq.set_acquisition_params(
        scan_type="projection",
        channel_states=[True, False, False, False, False],
        image_mirror_range_um=20.0,
        exposure_ms=10.0,
    )
    daq.generate_waveforms()
    daq.program_daq_waveforms()
    stale_tasks = daq.tasks

    daq.hold_next_reset()
    daq.reset()

    assert daq.reset_in_progress
    assert not daq.tasks
    assert not daq.programmed()
    assert all(not task.valid for task in stale_tasks)
    with pytest.raises(MockDAQTaskError, match="invalid or cleared"):
        stale_tasks[0].start(current_generation=0)

    daq.generate_waveforms()
    with pytest.raises(MockDAQRoutingError, match="-89130"):
        daq.program_daq_waveforms()

    daq.complete_reset()
    daq.program_daq_waveforms()
    daq.start_waveform_playback()

    assert daq.running()
    assert all(task.valid and task.running for task in daq.tasks)
