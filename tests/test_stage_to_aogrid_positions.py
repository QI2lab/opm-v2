import numpy as np

# Example list of stage positions
stage_positions = [
    {"z": 0.1, "y": 0, "x": 1},
    {"z": 0.2, "y": 0, "x": 2},
    {"z": 0.3, "y": 0, "x": 3},
    {"z": 0.4, "y": 0, "x": 4},
    {"z": 0.5, "y": 0, "x": 5},
    {"z": 5.1, "y": 0, "x": 1},
    {"z": 5.2, "y": 0, "x": 2},
    {"z": 5.3, "y": 0, "x": 3},
    {"z": 5.4, "y": 0, "x": 4},
    {"z": 5.5, "y": 0, "x": 5},
    {"z": 0.1, "y": 10, "x": 1},
    {"z": 0.2, "y": 10, "x": 2},
    {"z": 0.3, "y": 10, "x": 3},
    {"z": 0.4, "y": 10, "x": 4},
    {"z": 0.5, "y": 10, "x": 5},
    {"z": 5.1, "y": 10, "x": 1},
    {"z": 5.2, "y": 10, "x": 2},
    {"z": 5.3, "y": 10, "x": 3},
    {"z": 5.4, "y": 10, "x": 4},
    {"z": 5.5, "y": 10, "x": 5}
]

stage_positions_array = np.array(
    [
        (pos["z"], pos["y"], pos["x"]) for pos in stage_positions
    ]
)

# Extract unique x values
scan_axis_positions = np.unique(stage_positions_array[:, 2])
ao_scan_axis_positions = scan_axis_positions - np.diff(scan_axis_positions)[0]/2
tile_axis_positions = np.unique(stage_positions_array[:, 1])
ao_tile_axis_position = np.mean(tile_axis_positions)

ao_stage_positions = []
for scan_tile, scan_tile_pos in enumerate(ao_scan_axis_positions):    
    z_tile_positions = np.unique(
        stage_positions_array[stage_positions_array[:, 2] == scan_axis_positions[scan_tile]][:, 0]
    )
    num_z_tiles = len(z_tile_positions)
    for z_pos in z_tile_positions:
        ao_stage_positions.append(
            {
                "z": z_pos,
                "y": ao_tile_axis_position,
                "x": scan_tile_pos
            }
        )
        
# Save AO optimization results here
num_ao_pos = len(ao_stage_positions)
ao_grid_wfc_coeffs = np.zeros((num_ao_pos, 10))
ao_grid_wfc_positions = np.zeros((num_ao_pos, 15))

# Run AO optimization for each stage position
for pos_idx in range(num_ao_pos):
    
    ao_grid_wfc_coeffs[pos_idx] = [pos_idx+3]*10
    ao_grid_wfc_positions[pos_idx] = [pos_idx+6]*15

# Map ao_grid_wfc_coeffs to experiment stage positions.
position_wfc_coeffs = np.zeros((len(stage_positions), 10))
position_wfc_positions = np.zeros((len(stage_positions), 15))

for pos_idx in range(len(stage_positions)):
    z = stage_positions[pos_idx]["z"]
    y = stage_positions[pos_idx]["y"]
    x = stage_positions[pos_idx]["x"]
    
    # get the scan axis
    scan_axis_idx = np.where(scan_axis_positions==x)[0][0]
    z_tile_positions = np.unique(
        stage_positions_array[stage_positions_array[:, 2] == scan_axis_positions[scan_axis_idx]][:, 0]
    )
    
    z_axis_idx = np.argmin(np.abs(z_tile_positions-z))
    ao_grid_idx = 2*scan_axis_idx + z_axis_idx
    position_wfc_positions[pos_idx] = ao_grid_wfc_positions[ao_grid_idx]
    position_wfc_coeffs[pos_idx] = ao_grid_wfc_coeffs[ao_grid_idx]


    
    # position_wfc_coeffs[pos_idx] = ao_grid_wfc_coeffs[ao_pos_idx]
    # position_wfc_positions[pos_idx] = ao_grid_wfc_positions[ao_pos_idx]
    
    