"""Test ElveFlow OB1 triggering with its in-memory backend."""

from __future__ import annotations

from opm_v2.hardware.ElveFlow import OB1Controller


def test_simulated_ob1_completes_trigger_handshake_in_memory() -> None:
    """Verify a simulated OB1 trigger completes its acknowledgment handshake."""
    controller = OB1Controller(simulate=True)

    try:
        assert OB1Controller.instance() is controller
        controller.init_board()
        controller.trigger_OB1(pulse_duration=0.5)
        controller.wait_for_OB1()

        assert controller.trigger_count == 1
        assert controller.last_pulse_duration == 0.5
        assert controller._from_OB1_pin_high is False
    finally:
        controller.close_board()
