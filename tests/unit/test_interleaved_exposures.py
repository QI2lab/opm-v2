"""Reject exposures that split camera-clocked interleaved scans."""

import pytest

from opm_v2.engine.setup_events import get_channel_plan
from opm_v2.utils.laser_modes import validate_interleaved_exposures


@pytest.mark.parametrize(
    "values", [[20, 5, 5], [5, 5.001], [float("nan")], [float("inf")], [0], [-1]]
)
def test_invalid_interleaved_exposures(values):
    """Reject differences before rounding, and nonphysical exposures."""
    with pytest.raises(ValueError, match="Interleaved acquisition requires equal"):
        validate_interleaved_exposures(values)


@pytest.mark.parametrize("values", [[5], [5, 5], [20, 20, 20]])
def test_valid_interleaved_exposures(values):
    """Accept a single channel and equal multichannel exposures."""
    validate_interleaved_exposures(values)


def test_saved_configuration_requires_equal_enabled_exposures():
    """Reject the failing 20/5/5-ms acquisition with actionable channel names."""
    config = {
        "OPM": {"channel_ids": ["405nm", "488nm", "561nm", "637nm", "730nm"]},
        "acq_config": {
            "DAQ": {
                "channel_states": [True, True, True, False, False],
                "channel_powers": [15, 15, 15, 0, 0],
                "channel_exposures_ms": [20, 5, 5, 0, 0],
                "laser_mode": "interleaved",
            }
        },
    }
    with pytest.raises(ValueError, match="405nm: 20 ms, 488nm: 5 ms, 561nm: 5 ms"):
        get_channel_plan(config, "DAQ", round_active_exposures=True)
    config["acq_config"]["DAQ"]["channel_exposures_ms"] = [20, 20, 20, 0, 99]
    assert get_channel_plan(config, "DAQ")["active_exposures_ms"] == [20, 20, 20]
