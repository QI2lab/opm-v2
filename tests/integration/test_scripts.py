"""Exercise command-line scripts through reusable simulated workflows."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
import zarr
from useq import MDAEvent, MDASequence

from opm_v2.handlers.opm_data_handler import OpmDataHandler
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.utils import sensorless_ao
from opm_v2.utils.script_io import load_tensorstore
from scripts.ao_grid_mapping import main as map_ao_grid
from scripts.ao_positions import main as select_ao_grid
from scripts.load_ao_optimization_results import (
    main as inspect_ao_results,
)
from scripts.load_ao_optimization_results import (
    summarize_results,
)
from scripts.load_test_acquisition import main as inspect_acquisition
from scripts.load_test_acquisition import summarize_array
from scripts.make_gifs_from_data import main as make_gif
from scripts.parse_ao_results import plot_results
from scripts.parse_grid_ao_results import main as plot_ao_grid
from scripts.stage_positions import main as generate_positions


def test_aborted_ao_can_save_positions_before_output_directory_exists(workspace_tmp_path):
    """Save teardown sidecars even when AO failed before creating its folder."""
    mirror = AOMirror(simulate=True)
    mirror.output_path = workspace_tmp_path / "not_created" / "ao_run"
    mirror.save_positions_array(prefix="ao_position")
    voltages = json.loads((mirror.output_path / "ao_position_voltage.json").read_text())
    coefficients = json.loads(
        (mirror.output_path / "ao_position_mode_amplitude.json").read_text()
    )
    np.testing.assert_array_equal(voltages, mirror.positions_voltage_array)
    np.testing.assert_array_equal(coefficients, mirror.positions_modal_array)


@pytest.mark.parametrize("per_mode", [True, False], ids=["current", "legacy"])
def test_six_iteration_ao_save_load_and_plots(workspace_tmp_path, monkeypatch, per_mode):
    """Plot all six iterations and the actual final coefficients of each."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    iterations, modes, samples = 6, 2, 3
    count = iterations * modes
    coefficients = np.arange(count * 32, dtype=float).reshape(count, 32) / 1000
    iteration_coefficients = coefficients.reshape(iterations, modes, 32)[:, -1]
    sensorless_ao.save_optimization_results(
        all_images=np.zeros((1 + count * (samples + 1), 2, 2), dtype=np.uint16),
        all_metrics=np.arange(1 + count * (samples + 1), dtype=float),
        optimal_images=np.zeros((count, 2, 2), dtype=np.uint16),
        optimal_metrics=np.arange(count, dtype=float),
        optimal_coeffs=coefficients if per_mode else iteration_coefficients,
        starting_coeffs=np.zeros(32),
        starting_metric=0.0,
        starting_image=np.zeros((2, 2), dtype=np.uint16),
        update_status=np.ones(count, dtype=bool),
        metadata={
            "num_iterations": np.int64(iterations),
            "num_mode_samples": samples,
            "modes_to_optimize": [3, 7],
            "metric_name": "gauss_2d",
            "o3_stack": {"measurements": [{
                "crop_y0_y1_x0_x1": [np.int64(50), np.int64(147), np.int64(52), np.int64(149)],
                "valid": np.bool_(True), "metric": np.float32(1.25),
                "positions": np.array([8.0, 8.5, 9.0]),
            }]},
        },
        save_dir_path=workspace_tmp_path,
    )
    results = sensorless_ao.load_optimization_results(workspace_tmp_path / "ao_results.zarr")
    assert results["metadata"]["o3_stack"]["measurements"][0] == {
        "crop_y0_y1_x0_x1": [50, 147, 52, 149], "valid": True,
        "metric": 1.25, "positions": [8.0, 8.5, 9.0],
    }
    closed_figures = []
    real_close = plt.close

    def record_close(fig=None):
        if isinstance(fig, Figure):
            closed_figures.append(fig)
        real_close(fig)

    monkeypatch.setattr(plt, "close", record_close)
    sensorless_ao.plot_zernike_coeffs(results, sensorless_ao.mode_names, workspace_tmp_path)
    sensorless_ao.plot_metric_progress(
        results, sensorless_ao.mode_names, save_dir_path=workspace_tmp_path
    )
    assert (workspace_tmp_path / "ao_zernike_coeffs.png").is_file()
    assert (workspace_tmp_path / "ao_metrics.png").is_file()
    assert len(closed_figures) == 2
    coefficient_ax = closed_figures[0].axes[0]
    for mode in range(32):
        for iteration in range(iterations):
            point = coefficient_ax.collections[32 + mode * iterations + iteration]
            assert point.get_offsets()[0, 0] == iteration_coefficients[iteration, mode]
    metric_lines = closed_figures[1].axes[0].lines
    assert len(metric_lines) == 2 * iterations
    np.testing.assert_array_equal(metric_lines[-1].get_ydata(), [10, 11])


def test_invalid_metadata_does_not_replace_existing_ao_results(workspace_tmp_path):
    """Reject unsupported metadata before opening an existing store for overwrite."""
    path = workspace_tmp_path / "ao_results.zarr"
    original = zarr.open_group(str(path), mode="w", attributes={"preserved": True})
    original.create_array("image", data=np.arange(4))
    arrays = {name: np.zeros(1) for name in (
        "all_images", "all_metrics", "optimal_images", "optimal_metrics",
        "optimal_coeffs", "starting_coeffs", "starting_metric", "starting_image",
        "update_status",
    )}
    with pytest.raises(TypeError, match="not JSON serializable"):
        sensorless_ao.save_optimization_results(
            **arrays, metadata={"unsupported": object()}, save_dir_path=workspace_tmp_path,
        )
    restored = zarr.open_group(str(path), mode="r")
    assert restored.attrs["preserved"] is True
    np.testing.assert_array_equal(restored["image"][:], np.arange(4))


def test_position_scripts_share_one_configurable_pipeline(
    workspace_tmp_path: Path,
) -> None:
    """Generate positions, select an AO grid, and map the acquisition to it.

    Parameters
    ----------
    workspace_tmp_path : Path
        Workspace-local directory for generated JSON documents.
    """
    positions_path = workspace_tmp_path / "positions.json"
    common_arguments = [
        "--x",
        "0",
        "100",
        "--y",
        "0",
        "50",
        "--z",
        "0",
        "10",
        "--camera-shape",
        "10",
        "10",
        "--pixel-size-um",
        "1",
        "--angle-deg",
        "30",
        "--tile-overlap",
        "0",
        "--max-stage-range-um",
        "60",
        "--scan-overlap-um",
        "10",
        "--coverslip-slope-x",
        "0.01",
    ]
    assert generate_positions([str(positions_path), *common_arguments]) == 0
    positions = json.loads(positions_path.read_text())
    assert len(positions) == 36

    ao_path = workspace_tmp_path / "ao_positions.json"
    selection_arguments = ["--scan-samples", "2", "--tile-samples", "2"]
    assert (
        select_ao_grid([str(positions_path), str(ao_path), *selection_arguments]) == 0
    )
    ao_positions = json.loads(ao_path.read_text())
    assert len(ao_positions) == 12

    mapping_path = workspace_tmp_path / "ao_mapping.json"
    assert map_ao_grid([str(positions_path), str(ao_path), str(mapping_path)]) == 0
    mapping = json.loads(mapping_path.read_text())
    assert len(mapping) == len(positions)
    assert set(mapping) == set(range(len(ao_positions)))
    for position, ao_index in zip(positions, mapping, strict=True):
        squared_distances = [
            (position["x"] - candidate["x"]) ** 2
            + (position["y"] - candidate["y"]) ** 2
            for candidate in ao_positions
        ]
        assert squared_distances[ao_index] == min(squared_distances)


def test_data_scripts_read_real_opm_tensorstore_and_create_gif(
    workspace_tmp_path: Path,
) -> None:
    """Write with OpmDataHandler, inspect with TensorStore, and create a GIF.

    Parameters
    ----------
    workspace_tmp_path : Path
        Workspace-local directory for acquisition artifacts.
    """
    output = workspace_tmp_path / "script_data.ome.zarr"
    sequence = MDASequence()
    handler = OpmDataHandler(
        path=output,
        index_sizes={"c": 2, "z": 2},
        acquisition_order=("c", "z"),
        delete_existing=True,
    )
    handler.sequenceStarted(
        sequence,
        {"image_infos": [{"width": 5, "height": 4, "dtype": "uint16"}]},
    )
    try:
        for channel in range(2):
            for plane in range(2):
                value = channel * 100 + plane * 25
                handler.frameReady(
                    np.full((4, 5), value, dtype=np.uint16),
                    MDAEvent(index={"c": channel, "z": plane}),
                    {"runner_time_ms": float(value)},
                )
    finally:
        handler.sequenceFinished(sequence)

    array_path = output / "0"
    data = load_tensorstore(array_path)
    assert summarize_array(data) == {
        "shape": (2, 2, 4, 5),
        "dtype": "uint16",
        "minimum": 0.0,
        "maximum": 125.0,
    }
    positions_path = workspace_tmp_path / "data_positions.json"
    positions_path.write_text(json.dumps([{"x": 0, "y": 0, "z": 0}]))
    assert (
        inspect_acquisition([str(array_path), "--positions", str(positions_path)]) == 0
    )

    gif_path = workspace_tmp_path / "acquisition.gif"
    assert make_gif([str(array_path), str(gif_path), "--fps", "5"]) == 0
    frames = imageio.mimread(gif_path)
    assert len(frames) == 4
    assert frames[0].shape[:2] == (4, 5)


def test_current_ao_scripts_load_discover_and_plot_results(
    workspace_tmp_path: Path,
) -> None:
    """Create a current AO store and process it with every current AO script.

    Parameters
    ----------
    workspace_tmp_path : Path
        Workspace-local directory for AO result artifacts.
    """
    result_directory = workspace_tmp_path / "ao_grid" / "position_0"
    result_directory.mkdir(parents=True, exist_ok=True)
    mode_count = len(sensorless_ao.mode_names)
    sensorless_ao.save_optimization_results(
        all_images=np.arange(9 * 16, dtype=np.uint16).reshape(9, 4, 4),
        all_metrics=np.linspace(1.0, 2.0, 9),
        optimal_images=np.zeros((2, 4, 4), dtype=np.uint16),
        optimal_metrics=np.asarray([1.5, 2.0]),
        optimal_coeffs=np.zeros((1, mode_count)),
        starting_coeffs=np.zeros(mode_count),
        starting_metric=1.0,
        starting_image=np.zeros((4, 4), dtype=np.uint16),
        update_status=np.asarray([True, True]),
        metadata={
            "num_iterations": 1,
            "num_mode_samples": 3,
            "modes_to_optimize": [0, 1],
            "modes_to_optimize_names": ["Piston", "Tilt"],
            "metric_name": "DCT entropy",
        },
        save_dir_path=result_directory,
    )
    results_path = result_directory / "ao_results.zarr"
    results = sensorless_ao.load_optimization_results(results_path)
    assert summarize_results(results) == {
        "image_count": 9,
        "metric_count": 9,
        "iterations": 1,
        "mode_samples": 3,
        "modes_to_optimize": [0, 1],
    }
    assert inspect_ao_results([str(results_path)]) == 0

    direct_plots = workspace_tmp_path / "direct_ao_plots"
    loaded_results = plot_results(results_path, direct_plots)
    assert loaded_results["metadata"]["metric_name"] == "DCT entropy"
    assert (direct_plots / "ao_metrics.png").is_file()
    assert (direct_plots / "ao_zernike_coeffs.png").is_file()

    grid_plots = workspace_tmp_path / "grid_ao_plots"
    assert (
        plot_ao_grid([str(workspace_tmp_path / "ao_grid"), "--output", str(grid_plots)])
        == 0
    )
    assert (grid_plots / "grid_0" / "ao_metrics.png").is_file()
    assert (grid_plots / "grid_0" / "ao_zernike_coeffs.png").is_file()
