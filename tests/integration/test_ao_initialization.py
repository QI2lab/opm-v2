"""Exercise AO initialization through acquisition planning and engine dispatch."""

import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pymmcore_plus.mda import MDAEngine
from useq import MDASequence

from opm_v2.engine.opm_custom_events import create_ao_optimize_event
from opm_v2.engine.opm_engine import OPMEngineV2
from opm_v2.engine.setup_events import annotate_ao_positions
from opm_v2.utils.ao_policy import PREVIOUS_AO


def _mirror():
    """Build mirror state without a device connection.

    Returns
    -------
    SimpleNamespace
        Mutable current and optimized states.
    """
    return SimpleNamespace(
        current_coeffs=np.array([0.2, 0.3]),
        optimized_modal_coeffs=np.array([0.4, 0.5]),
        optimized_reference_state="factory_flat",
        update_positions_array=MagicMock(),
    )


def test_alignment_refocus_event_reaches_engine(opm_config_factory):
    """Forward the alignment-only interval and configured O3 device through dispatch."""
    config = opm_config_factory(mode="2d")
    config["acq_config"]["AO"].update(
        ao_mode="optimize now", daq_mode="2d", metric="gauss_2d",
        alignment_refocus_every_n_modes=3,
    )
    config["O2O3-autofocus"]["O3_stage_name"] = "Alignment O3"
    engine = object.__new__(OPMEngineV2)
    engine.AOMirror = _mirror()
    engine.simulate_hardware = False
    with patch("opm_v2.engine.opm_engine.run_ao_optimization", return_value=True) as optimize:
        engine.exec_event(create_ao_optimize_event(config))
    assert optimize.call_args.kwargs["alignment_refocus_every_n_modes"] == 3
    assert optimize.call_args.kwargs["o3_stack_settings"]["stage_name"] == "Alignment O3"


def test_logical_position_annotations_follow_depth_shift(opm_config_factory):
    """Stage X shifts with depth while logical tile identities remain stable."""
    config = opm_config_factory(mode="projection")
    positions = [
        {"x": x + 5 * z, "lab_scan_um": x, "y": 0, "z": z * 10}
        for z in range(2)
        for x in (0, 100)
    ]
    events = [create_ao_optimize_event(config) for _ in positions]
    for idx, event in enumerate(events):
        event.action.data["AO"]["pos_idx"] = idx
    annotate_ao_positions(events, positions)
    assert [
        (e.action.data["AO"]["z_idx"], e.action.data["AO"]["logical_position_idx"])
        for e in events
    ] == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_engine_resets_chain_at_each_sequence_and_does_not_publish_failure(
    opm_config_factory,
):
    """Run real engine dispatch across acquisitions with a failing AO boundary."""
    config = opm_config_factory(
        mode="projection",
        updates={
            "acq_config": {
                "AO": {
                    "ao_mode": "at xyz positions",
                    "starting_correction": "system flat",
                    "sequential_correction": PREVIOUS_AO,
                }
            }
        },
    )
    mirror = _mirror()
    core = MagicMock()
    engine = object.__new__(OPMEngineV2)
    engine._mmcore_ref = weakref.ref(core)
    engine._config = config
    engine.AOMirror = mirror
    engine.simulate_hardware = True
    engine._capture_stage_speeds = MagicMock()
    engine._apply_stage_move_speeds = MagicMock()
    starts = []

    def optimize(**kwargs):
        starts.append(kwargs["starting_mirror_coefficients"].copy())
        mirror.current_coeffs = kwargs["starting_mirror_coefficients"] + 1
        return True

    event = create_ao_optimize_event(config)
    with (
        patch.object(MDAEngine, "setup_sequence"),
        patch(
            "opm_v2.engine.opm_engine.run_ao_optimization",
            side_effect=optimize,
        ) as optimize_mock,
    ):
        engine.setup_sequence(MDASequence())
        engine.simulate_hardware = False
        engine.exec_event(event)
        engine.exec_event(event)
        optimize_mock.side_effect = RuntimeError("camera failed")
        with pytest.raises(RuntimeError, match="camera failed"):
            engine.exec_event(event)
        np.testing.assert_array_equal(
            engine._ao_corrections.initializer()["starting_mirror_coefficients"], [2, 2]
        )
        optimize_mock.side_effect = optimize
        # setup_sequence is also the boundary after cancellation/restart.
        engine.simulate_hardware = True
        engine.setup_sequence(MDASequence())
        engine.simulate_hardware = False
        engine.exec_event(event)
    np.testing.assert_array_equal(starts, [[0, 0], [1, 1], [0, 0]])


@pytest.mark.parametrize(
    "policy",
    [
        "use starting correction",
        "previous AO result",
        "corresponding position in previous Z layer",
    ],
)
def test_per_position_dispatch_uses_matching_history(opm_config_factory, policy):
    """Exercise all initialization policies through actual individual AO dispatch."""
    from opm_v2.utils.ao_policy import PREVIOUS_Z

    config = opm_config_factory(
        mode="projection",
        updates={
            "acq_config": {
                "AO": {
                    "starting_correction": "system flat",
                    "sequential_correction": policy,
                }
            }
        },
    )
    mirror = _mirror()
    engine = object.__new__(OPMEngineV2)
    engine.AOMirror = mirror
    engine.simulate_hardware = False
    starts = []

    def optimize(**kwargs):
        starts.append(kwargs["starting_mirror_coefficients"].copy())
        mirror.current_coeffs = np.full(2, len(starts), dtype=float)
        return True

    with patch("opm_v2.engine.opm_engine.run_ao_optimization", side_effect=optimize):
        for time_idx in range(2):
            for z_idx in range(2):
                for position in range(2):
                    event = create_ao_optimize_event(config)
                    event.action.data["AO"].update(
                        time_idx=time_idx,
                        z_idx=z_idx,
                        logical_position_idx=position,
                        pos_idx=z_idx * 2 + position,
                    )
                    engine.exec_event(event)
    expected = (
        list(range(8))
        if policy == PREVIOUS_AO
        else [0, 0, 1, 2, 0, 0, 5, 6]
        if policy == PREVIOUS_Z
        else [0] * 8
    )
    np.testing.assert_array_equal(np.asarray(starts)[:, 0], expected)
