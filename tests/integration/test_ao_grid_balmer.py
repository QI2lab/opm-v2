"""Run the recorded Balmer AO plan through an entirely fake stage and mirror."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np

from opm_v2.utils import sensorless_ao


def test_balmer_full_grid_mapping_at_four_depths_without_controller(
    monkeypatch, workspace_tmp_path
):
    """Visit all cells and update only each depth's global whole-tile states."""
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "balmer_ao_geometry.json").read_text()
    )
    calibration = fixture["calibration"]
    independent = fixture["independent_expectations"]
    bounds = fixture["requested_bounds_um"]
    reference_z = fixture["surface"]["scan_reference_z_um"]
    expected_assignments = np.asarray([0, 0, 0, 3, 3, 6, 6, 2, 2, 2, 5, 5, 8, 8])
    tiles_per_depth = 14
    total_tiles = tiles_per_depth * len(fixture["depths"])
    positions = [
        {
            "x": physical_x,
            "lab_scan_um": lab_x,
            "y": y,
            "z": depth["physical_stage_z_um"],
            "scan_reference_z_um": reference_z,
            "depth_index": depth["depth_index"],
            "sample_depth_um": depth["sample_depth_um"],
            "stage_depth_offset_um": depth["stage_depth_offset_um"],
        }
        for depth in fixture["depths"]
        for physical_x, lab_x in zip(
            depth["recorded_physical_stage_x_starts_um"],
            fixture["recorded_lab_scan_starts_um"],
            strict=True,
        )
        for y in fixture["recorded_tile_y_um"]
    ]
    # Build support directly from the recorded trajectory endpoints and optical
    # calibration, independently of the production footprint/assignment helpers.
    half_scan = (
        calibration["camera_crop_y"]
        * calibration["pixel_size_um"]
        * np.cos(np.deg2rad(calibration["angle_deg"]))
        / 2
    )
    half_tile = calibration["camera_crop_x"] * calibration["pixel_size_um"] / 2
    footprints = np.asarray(
        [
            [start - half_scan, end + half_scan, y - half_tile, y + half_tile]
            for start, end in zip(
                fixture["recorded_lab_scan_starts_um"],
                fixture["reconstructed_commanded_lab_scan_ends_um"],
                strict=True,
            )
            for y in fixture["recorded_tile_y_um"]
        ]
    )
    spatial_geometry = {
        "coordinate_frame": "depth_normalized_laboratory_scan_x_stage_y",
        "units": "um",
        "grid_bounds_source": "requested_stage_explorer_region_bounds",
        "requested_region_bounds_um": bounds,
        "grid_bounds_um": bounds,
        "commanded_acquired_support_bounds_um": {
            "x": independent["reconstructed_commanded_support_lab_scan_um"],
            "y": independent["support_y_um"],
        },
        "tile_support_definition": "commanded trajectory endpoints plus centered camera support",
    }
    ao_dict = {
        "channel_states": [True],
        "daq_mode": "2d",
        "exposure_ms": 10.0,
        "image_mirror_range_um": 0.0,
        "iterations": 1,
        "metric": "DCT",
        "metric_acceptance": "zero",
        "metric_precision": 6,
        "mirror_state": "system flat",
        "modal_alpha": 0.5,
        "modal_delta": 0.1,
        "modes_to_optimize": "all",
        "num_averaged_frames": 1,
        "num_mode_samples": 3,
    }
    mirror = SimpleNamespace(
        positions_modal_array=np.full((total_tiles, 2), -999.0),
        positions_voltage_array=np.full((total_tiles, 3), -999.0),
        current_coeffs=np.zeros(2),
        current_voltage=np.zeros(3),
    )
    command = {"x": 0.0, "y": 0.0, "z": 0.0}
    stage = MagicMock()
    stage.getFocusDevice.return_value = "fake_focus"
    stage.setPosition.side_effect = lambda z: command.update(z=float(z))
    stage.setXYPosition.side_effect = lambda x, y: command.update(
        x=float(x), y=float(y)
    )
    # Measured readbacks differ from the commanded targets but satisfy the
    # production settling tolerance. They must remain separate in the artifact.
    stage.getXYPosition.side_effect = lambda: (command["x"] + 0.1, command["y"] + 0.1)
    stage.getZPosition.side_effect = lambda: command["z"] + 0.1
    optimization_starts = []

    def optimize(**kwargs):
        optimization_starts.append(
            np.asarray(kwargs["starting_mirror_coefficients"]).copy()
        )
        result_number = len(optimization_starts)
        mirror.current_coeffs = np.asarray([result_number, -result_number], dtype=float)
        mirror.current_voltage = np.asarray(
            [result_number, result_number + 0.1, result_number + 0.2], dtype=float
        )
        return True

    monkeypatch.setattr(
        sensorless_ao, "CMMCorePlus", SimpleNamespace(instance=lambda: stage)
    )
    monkeypatch.setattr(
        sensorless_ao, "AOMirror", SimpleNamespace(instance=lambda: mirror)
    )
    monkeypatch.setattr(sensorless_ao, "run_ao_optimization", optimize)
    expected_modal = mirror.positions_modal_array.copy()
    expected_voltage = mirror.positions_voltage_array.copy()
    previous = None
    expected_all_xy = []
    expected_all_z = []
    for depth in fixture["depths"]:
        depth_index = depth["depth_index"]
        indices = list(
            range(depth_index * tiles_per_depth, (depth_index + 1) * tiles_per_depth)
        )
        output = workspace_tmp_path / f"z_{depth_index}"
        output.mkdir()
        modal_before = mirror.positions_modal_array.copy()
        voltage_before = mirror.positions_voltage_array.copy()
        result = sensorless_ao.run_ao_grid_mapping(
            ao_dict=ao_dict,
            stage_positions=[positions[i] for i in indices],
            position_indices=indices,
            num_scan_positions=3,
            num_tile_positions=3,
            save_dir_path=output,
            verbose=False,
            time_idx=0,
            z_idx=depth_index,
            previous_grid_coefficients=None
            if previous is None
            else previous.modal_coefficients,
            previous_grid_reference_state=None
            if previous is None
            else previous.reference_state,
            grid_bounds_um=bounds,
            tile_bounds_um=footprints,
            angle_deg=calibration["angle_deg"],
            camera_zstage_orientation=calibration["camera_Zstage_orientation"],
            spatial_geometry=spatial_geometry,
        )
        assert result is not None
        assert result.modal_coefficients.shape == (9, 2)
        assert result.actuator_positions.shape == (9, 3)
        assert len(optimization_starts) == (depth_index + 1) * 9
        expected_starts = (
            np.zeros((9, 2)) if previous is None else previous.modal_coefficients
        )
        np.testing.assert_array_equal(
            np.stack(optimization_starts[-9:]), expected_starts
        )
        expected_numbers = depth_index * 9 + expected_assignments + 1
        expected_modal[indices] = np.column_stack((expected_numbers, -expected_numbers))
        expected_voltage[indices] = np.column_stack(
            (expected_numbers, expected_numbers + 0.1, expected_numbers + 0.2)
        )
        np.testing.assert_array_equal(mirror.positions_modal_array, expected_modal)
        np.testing.assert_array_equal(mirror.positions_voltage_array, expected_voltage)
        outside = [i for i in range(total_tiles) if i not in indices]
        np.testing.assert_array_equal(
            mirror.positions_modal_array[outside], modal_before[outside]
        )
        np.testing.assert_array_equal(
            mirror.positions_voltage_array[outside], voltage_before[outside]
        )
        artifact = json.loads((output / "ao_grid_geometry.json").read_text())
        assert artifact["position_indices"] == indices
        assert artifact["tile_ao_indices"] == expected_assignments.tolist()
        assert artifact["correction_scope"] == "entire_tile"
        assert artifact["assignment"] == "largest_xy_overlap_area"
        assert artifact["tie_break"] == "lowest_grid_index_y_major_x_minor"
        assert artifact["coordinate_frame"] == spatial_geometry["coordinate_frame"]
        assert artifact["units"] == "um"
        assert (
            artifact["spatial_geometry"]["grid_bounds_source"]
            == spatial_geometry["grid_bounds_source"]
        )
        for axis in ("x", "y"):
            np.testing.assert_allclose(
                artifact["grid_bounds_um"][axis],
                np.round(bounds[axis], 3),
                rtol=0,
                atol=1e-9,
            )
        commands = json.loads((output / "ao_stage_positions.json").read_text())
        assert len(commands) == 9
        expected_xy = [
            (x, round(y, 3))
            for y in independent["requested_grid_y_centers_um"]
            for x in independent["physical_stage_x_grid_centers_rounded_um_by_depth"][
                depth_index
            ]
        ]
        expected_z = round(depth["physical_stage_z_um"], 3)
        assert [(p["x"], p["y"]) for p in commands] == expected_xy
        assert [p["z"] for p in commands] == [expected_z] * 9
        expected_all_xy.extend(expected_xy)
        expected_all_z.extend([expected_z] * 9)
        for idx, coordinate in enumerate(artifact["ao_positions"]):
            physical = coordinate["physical_stage_command_um"]
            assert physical == {
                "x": expected_xy[idx][0],
                "y": expected_xy[idx][1],
                "z": expected_z,
            }
            assert coordinate["laboratory_scan_um"]["x"] == round(
                independent["requested_grid_lab_scan_centers_um"][idx % 3], 3
            )
            assert coordinate["laboratory_scan_um"]["z"] == -depth["sample_depth_um"]
            for axis in ("x", "y", "z"):
                assert coordinate["stage_readback_um"][axis] == round(
                    physical[axis] + 0.1, 3
                )
            assert coordinate["coordinate_transform"]["units"] == "um"
            assert (
                coordinate["coordinate_transform"]["scan_reference_z_um"] == reference_z
            )
        previous = result
    assert total_tiles == 56
    assert len(optimization_starts) == 36
    assert stage.setXYPosition.call_args_list == [
        call(x, y) for x, y in expected_all_xy
    ]
    assert stage.setPosition.call_args_list == [call(z) for z in expected_all_z]
    assert stage.waitForDevice.call_args_list == [call("fake_focus")] * 36
