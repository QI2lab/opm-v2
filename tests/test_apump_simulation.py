from __future__ import annotations

from opm_v2.hardware.APump import APump


def test_simulated_apump_tracks_flow_state_without_opening_a_serial_port() -> None:
    pump = APump({"simulate_pump": True, "verbose": False})

    try:
        assert pump.identification == "SIMULATED APUMP"
        assert pump.serial is None
        assert APump.instance() is pump

        pump.startFlow(12.5, direction="Reverse")
        flowing = pump.getStatus()
        assert flowing[:4] == ("Flowing", 12.5, "Reverse", "Remote")
        assert any(entry.startswith("BUFFERED") for entry in pump.command_log)

        pump.stopFlow()
        assert pump.getStatus()[0] == "Stopped"
    finally:
        pump.close()
