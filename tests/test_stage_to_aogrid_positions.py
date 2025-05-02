import numpy as np
num_scan_positions = 3
num_tile_positions = 3
verbose=True
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
# Extract unique positions along each axis
tile_axis_positions = np.unique(stage_positions_array[:, 1])
scan_axis_positions = np.unique(stage_positions_array[:, 2])

num_z_positions = np.unique(
    stage_positions_array[stage_positions_array[:, 2] == scan_axis_positions[0]][:, 0]
).shape[0]

if len(scan_axis_positions)==1 or num_scan_positions==1:
    scan_axis_delta = 0
    ao_scan_axis_positions = np.asarray([np.mean(scan_axis_positions)])
else:
    if num_scan_positions>len(scan_axis_positions)+1:
        num_scan_positions=len(scan_axis_positions)+1
    scan_tile_delta = scan_axis_positions[1] - scan_axis_positions[0]
    scan_axis_min = scan_axis_positions[0]
    scan_axis_max = scan_axis_positions[-1]
    scan_axis_range = np.abs(scan_axis_max - scan_axis_min)
    scan_axis_delta = scan_axis_range / num_scan_positions
    ao_scan_axis_positions = np.linspace(
        scan_axis_min,
        scan_axis_max,
        num_scan_positions+2,
        endpoint=True
    )[1:-1]
    
if len(tile_axis_positions)==1 or num_tile_positions==1:
    tile_axis_delta = 0
    ao_tile_axis_positions = np.asarray([np.mean(tile_axis_positions)])
else:
    if num_tile_positions>len(tile_axis_positions)+1:
        num_tile_positions=len(tile_axis_positions)+1
    tile_axis_delta = tile_axis_positions[1] - tile_axis_positions[0]
    tile_axis_min = tile_axis_positions[0]
    tile_axis_max = tile_axis_positions[-1]
    tile_axis_range = np.abs(tile_axis_max - tile_axis_min)
    tile_axis_delta = tile_axis_range / num_tile_positions
    ao_tile_axis_positions = np.linspace(
        tile_axis_min, 
        tile_axis_max,
        num_tile_positions+2, 
        endpoint=True
    )[1:-1]

# compile AO stage positions to visit, visit XY positions before stepping in Z
ao_stage_positions = []
for z_idx in range(num_z_positions):
    for tile_idx in range(num_tile_positions):
        for scan_idx in range(num_scan_positions):
            scan_pos_filter = np.ceil(ao_scan_axis_positions[scan_idx] - stage_positions_array[:, 2])==1
            z_tile_positions = np.unique(
                stage_positions_array[scan_pos_filter][:, 0]
            )
            ao_stage_positions.append(
                {
                    "z": np.round(z_tile_positions[z_idx],2),
                    "y": np.round(ao_tile_axis_positions[tile_idx],2),
                    "x": np.round(ao_scan_axis_positions[scan_idx],2)
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

# Convert AO positions and stage positions into structured arrays for efficient lookup
ao_stage_positions_array = np.array([(pos["z"], pos["y"], pos["x"]) for pos in ao_stage_positions])
stage_positions_array = np.array([(pos["z"], pos["y"], pos["x"]) for pos in stage_positions])

for pos_idx, (stage_z, stage_y, stage_x) in enumerate(stage_positions_array):

    target_z = ao_stage_positions_array[:,0][
        int(np.argmin(np.abs(stage_z - ao_stage_positions_array[:, 0])))
        ]
    target_y = ao_stage_positions_array[:,1][
        int(np.argmin(np.abs(stage_y - ao_stage_positions_array[:, 1])))
        ]
    target_x = ao_stage_positions_array[:,2][
        int(np.argmin(np.abs(stage_x - ao_stage_positions_array[:, 2])))
        ]
    
    # Find AO positions with matching z and y
    candidates = ao_stage_positions_array[
        (ao_stage_positions_array[:, 0] == target_z) &
        (ao_stage_positions_array[:, 1] == target_y) &
        (ao_stage_positions_array[:, 2] >= target_x)  # AO x must be >= stage x
    ]
    
    # Compute distances
    distances = np.linalg.norm(candidates - [target_z, stage_y, stage_x], axis=1)
    best_candidate_idx = np.argmin(distances)
    ao_grid_idx = np.where(
        (ao_stage_positions_array[:, 0] == candidates[best_candidate_idx][0]) &
        (ao_stage_positions_array[:, 1] == candidates[best_candidate_idx][1]) &
        (ao_stage_positions_array[:, 2] == candidates[best_candidate_idx][2])
    )[0][0]

    # Assign AO data
    position_wfc_positions[pos_idx] = ao_grid_wfc_positions[ao_grid_idx]
    position_wfc_coeffs[pos_idx] = ao_grid_wfc_coeffs[ao_grid_idx]

    if verbose:
        print(
            f"\n\n AO grid position: {ao_stage_positions[ao_grid_idx]}",
            f"\n Exp. stage position: {stage_positions[pos_idx]}"
        )