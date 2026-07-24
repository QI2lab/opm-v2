"""Test Picard shutter behavior with its in-memory backend."""

from __future__ import annotations

from opm_v2.hardware.PicardShutter import PicardShutter


def test_simulated_picard_shutter_tracks_open_and_closed_states() -> None:
    """Verify simulated shutter commands update the shared shutter state."""
    shutter = PicardShutter(shutter_id=712, simulate=True)

    try:
        assert PicardShutter.instance() is shutter
        assert shutter.is_connected
        assert shutter.state == "Closed"

        shutter.openShutter()
        assert shutter.state == "Open"

        shutter.closeShutter()
        assert shutter.state == "Closed"
    finally:
        shutter.shutDown()
