"""Exercise combined laser acquisition planning, timing, and powers."""

import json
from copy import deepcopy

import numpy as np
import pytest
from useq import GridFromEdges, MDASequence

from opm_v2.engine.opm_custom_events import ACTION_DAQ
from opm_v2.engine.opm_engine import OPMEngineV2
from opm_v2.engine.setup_events import (
    get_channel_plan,
    setup_mirrorscan,
    setup_projection,
    setup_stagescan,
    setup_timelapse,
)
from opm_v2.handlers.live_acquisition import acquisition_sidecar_paths


@pytest.mark.parametrize(
    "mode,build",
    [
        ("projection", setup_projection),
        ("mirror", setup_mirrorscan),
        ("stage", setup_stagescan),
        ("timelapse", setup_timelapse),
    ],
)
@pytest.mark.parametrize("laser_mode,pair,name", [
    ("simultaneous", (1, 3), "488 + 637"),
    ("simultaneous 488/637", (1, 3), "488 + 637"),
    ("simultaneous 561/637", (2, 3), "561 + 637"),
    ("simultaneous 488/561", (1, 2), "488 + 561"),
])
def test_simultaneous_planning_and_engine_powers(
    laser_mode, pair, name,
    mode,
    build,
    demo_core,
    workspace_tmp_path,
    opm_config_factory,
    simulated_acquisition_hardware,
    split_events,
):
    """Store one combined channel while driving both lasers at independent powers."""
    first, second = pair
    canonical_mode = "simultaneous 488/637" if laser_mode == "simultaneous" else laser_mode
    config = opm_config_factory(
        mode=mode,
        active_channels=(0, first, second, 4),
        channel_powers=(5, 23, 41, 6),
        channel_exposures_ms=(7, 10, 10, 20),
        updates={"acq_config": {"DAQ": {"laser_mode": laser_mode}}},
    )
    original = deepcopy(config)
    plan = get_channel_plan(config, "DAQ")
    assert config == original
    assert plan["active_count"] == 1
    assert plan["active_names"] == [name]
    assert plan["states"] == [i in pair for i in range(5)]
    assert plan["powers"] == [23 if i == first else 41 if i == second else 0 for i in range(5)]
    sequence = MDASequence(
        grid_plan=GridFromEdges(bottom=0, top=4, left=0, right=0),
        time_plan={"interval": 0, "loops": 1},
    )
    output = workspace_tmp_path / "combined.ome.zarr"
    events, handler = build(demo_core, config, sequence, output)
    images, _ = split_events(events)
    assert handler.index_sizes["c"] == 1
    assert images
    for event in images:
        assert event.index["c"] == 0
        assert event.metadata["DAQ"]["current_channel"] == name
        assert event.metadata["DAQ"]["channel_states"] == plan["states"]
        assert event.metadata["DAQ"]["laser_powers"] == plan["powers"]
        assert event.metadata["DAQ"]["laser_mode"] == canonical_mode

    engine = OPMEngineV2(
        demo_core,
        workspace_tmp_path / "unused.json",
        config=config,
        simulate_hardware=True,
    )
    daq_events = [
        event for event in events if getattr(event.action, "name", None) == ACTION_DAQ
    ]
    assert daq_events
    for event in daq_events:
        engine.setup_event(event)
        waveform = engine.opmDAQ.digital_waveform
        np.testing.assert_array_equal(waveform[:, first], waveform[:, second])
        assert waveform[::2, first].all()
        assert not waveform[:, [i for i in range(5) if i not in pair]].any()
        assert engine.simulated_laser_powers[config["Lasers"]["laser_names"][first]] == 23
        assert engine.simulated_laser_powers[config["Lasers"]["laser_names"][second]] == 41
    manifest_path, _ = acquisition_sidecar_paths(output)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["channels"][0]["name"] == name
    assert manifest["channels"][0]["wavelength_nm"] is None
    assert manifest["channels"][0]["laser_powers"] == {
        config["OPM"]["channel_ids"][first]: 23,
        config["OPM"]["channel_ids"][second]: 41,
    }


@pytest.mark.parametrize("exposures", [(10, 11), (0, 0), (float("nan"), 10)])
@pytest.mark.parametrize("laser_mode,pair", [
    ("simultaneous", (1, 3)),
    ("simultaneous 561/637", (2, 3)),
    ("simultaneous 488/561", (1, 2)),
])
def test_simultaneous_rejects_invalid_exposures(opm_config_factory, exposures, laser_mode, pair):
    """Reject incompatible exposures before creating events or commanding hardware."""
    config = opm_config_factory(
        mode="projection",
        active_channels=pair,
        channel_exposures_ms=exposures,
        updates={"acq_config": {"DAQ": {"laser_mode": laser_mode}}},
    )
    with pytest.raises(ValueError, match="equal, positive exposure"):
        get_channel_plan(config, "DAQ")
