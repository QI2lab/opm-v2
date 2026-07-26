"""Exercise OPM tile recovery against simulated Micro-Manager hardware."""

from __future__ import annotations

from opm_v2.engine.opm_engine import OPMEngineV2


def test_camera_snap_recovery_round_trip_on_simulated_hardware(
    demo_core,
    opm_config_factory,
    simulated_acquisition_hardware,
    workspace_tmp_path,
) -> None:
    """Snap directly without routing a reset DAQ through GUI Live setup."""
    config = opm_config_factory(
        mode="stage",
        active_channels=(0,),
        channel_powers=(10.0,),
        channel_exposures_ms=(10.0,),
        camera_shape=(32, 16),
    )
    engine = OPMEngineV2(
        demo_core,
        workspace_tmp_path / "unused-config.json",
        config=config,
        simulate_hardware=True,
    )
    daq = simulated_acquisition_hardware.daq
    daq.set_acquisition_params(
        scan_type="stage",
        channel_states=[True, False, False, False, False],
        exposure_ms=10.0,
    )
    daq.generate_waveforms()
    daq.hold_next_reset()
    daq.reset()
    program_attempts_before_snap = daq.operation_log.count("program:attempt")
    continuous_start_events: list[bool] = []

    def _record_continuous_start() -> None:
        continuous_start_events.append(True)
        daq.program_daq_waveforms()

    demo_core.events.continuousSequenceAcquisitionStarting.connect(
        _record_continuous_start
    )

    assert not demo_core.isSequenceRunning()
    try:
        engine._snap_camera_for_retry()
    finally:
        demo_core.events.continuousSequenceAcquisitionStarting.disconnect(
            _record_continuous_start
        )

    assert not demo_core.isSequenceRunning()
    assert demo_core.getRemainingImageCount() == 0
    assert continuous_start_events == []
    assert daq.operation_log.count("program:attempt") == program_attempts_before_snap
    assert daq.reset_in_progress
    daq.complete_reset()
    assert daq is engine.opmDAQ
