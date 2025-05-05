import numpy as np
verbose = True
stage_positions = [
    {'x': -6600.0, 'y': 0.0, 'z': 1900.0}, 
    {'x': -6600.0, 'y': 100.0, 'z': 1900.0}, 
    {'x': -6600.0, 'y': 200.0,'z': 1900.0},
    {'x': -6370.0, 'y': 0.0, 'z': 1900.0}, 
    {'x': -6370.0, 'y': 100.0, 'z': 1900.0}, 
    {'x': -6370.0, 'y': 200.0, 'z': 1900.0},
    {'x': -6600.0, 'y': 0.0, 'z': 1918.75},
    {'x': -6600.0, 'y': 100.0, 'z': 1918.75},
    {'x': -6600.0, 'y': 200.0, 'z': 1918.75},
    {'x': -6370.0, 'y': 0.0, 'z': 1918.75}, 
    {'x': -6370.0, 'y': 100.0, 'z': 1918.75}, 
    {'x':-6370.0, 'y': 200.0, 'z': 1918.75}, 
    {'x': -6600.0, 'y': 0.0, 'z': 1937.5}, 
    {'x': -6600.0, 'y': 100.0, 'z': 1937.5},
    {'x': -6600.0, 'y': 200.0, 'z': 1937.5}, 
    {'x': -6370.0, 'y': 0.0, 'z': 1937.5}, 
    {'x': -6370.0, 'y': 100.0, 'z': 1937.5}, 
    {'x': -6370.0, 'y': 200.0, 'z': 1937.5}, 
    {'x': -6600.0, 'y': 0.0, 'z': 1956.25}, 
    {'x': -6600.0, 'y': 100.0,'z': 1956.25}, 
    {'x': -6600.0, 'y': 200.0, 'z': 1956.25},
    {'x': -6370.0, 'y': 0.0, 'z': 1956.25},
    {'x': -6370.0, 'y': 100.0, 'z': 1956.25},
    {'x': -6370.0, 'y': 200.0, 'z': 1956.25}, 
    {'x': -6600.0, 'y': 0.0, 'z': 1975.0}, 
    {'x': -6600.0, 'y': 100.0, 'z': 1975.0}, 
    {'x': -6600.0, 'y': 200.0, 'z': 1975.0},
    {'x': -6370.0, 'y': 0.0, 'z': 1975.0},
    {'x': -6370.0, 'y': 100.0, 'z': 1975.0},
    {'x': -6370.0, 'y': 200.0, 'z': 1975.0},
    {'x': -6600.0, 'y': 0.0, 'z': 1993.75}, 
    {'x': -6600.0, 'y': 100.0, 'z': 1993.75}, 
    {'x': -6600.0, 'y': 200.0, 'z': 1993.75},
    {'x': -6370.0, 'y': 0.0, 'z': 1993.75},
    {'x': -6370.0, 'y': 100.0, 'z': 1993.75},
    {'x': -6370.0, 'y': 200.0, 'z': 1993.75},
    {'x': -6600.0, 'y': 0.0, 'z': 2012.5}, 
    {'x': -6600.0, 'y': 100.0, 'z': 2012.5},
    {'x': -6600.0, 'y': 200.0, 'z': 2012.5},
    {'x': -6370.0, 'y': 0.0, 'z': 2012.5},
    {'x': -6370.0, 'y': 100.0, 'z': 2012.5},
    {'x': -6370.0, 'y': 200.0, 'z': 2012.5}, 
    {'x': -6600.0, 'y': 0.0, 'z': 2031.25},
    {'x': -6600.0, 'y': 100.0, 'z': 2031.25}, 
    {'x': -6600.0, 'y': 200.0, 'z': 2031.25},
    {'x': -6370.0, 'y': 0.0, 'z': 2031.25}, 
    {'x': -6370.0, 'y': 100.0, 'z': 2031.25},
    {'x': -6370.0, 'y': 200.0, 'z': 2031.25},
    {'x': -6600.0, 'y': 0.0, 'z': 2050.0}, 
    {'x': -6600.0,'y': 100.0, 'z': 2050.0},
    {'x': -6600.0, 'y': 200.0, 'z': 2050.0},
    {'x': -6370.0, 'y': 0.0, 'z': 2050.0}, 
    {'x': -6370.0, 'y': 100.0, 'z': 2050.0},
    {'x': -6370.0, 'y': 200.0,'z': 2050.0}
]
stage_positions_array = np.array(
    [
        (pos["z"], pos["y"], pos["x"]) for pos in stage_positions
    ]
)

# Extract scan positions
scan_axis_positions = np.unique(stage_positions_array[:, 2])
if scan_axis_positions.shape[0]==1:
    print("AO Grid not optimized to work on 1 scan position!")
if scan_axis_positions[0]>scan_axis_positions[1]:
    print("AO grid is only working for scanning in positive directions!")

scan_tile_dx = np.diff(scan_axis_positions)[0]/2
ao_scan_axis_positions = scan_axis_positions + scan_tile_dx

# Extract tile positons, only sample the middle of the tile axis
tile_axis_positions = np.unique(stage_positions_array[:, 1])
ao_tile_axis_position = np.mean(tile_axis_positions)

# Extract the number of z positions per scan tile
num_z_positions = np.unique(
        stage_positions_array[stage_positions_array[:, 2] == scan_axis_positions[0]][:, 0]
    ).shape[0]

# compile AO stage positions to visit, visit XY positions before stepping in Z
ao_stage_positions = []
for z_idx in range(num_z_positions):
    for scan_tile, scan_tile_pos in enumerate(ao_scan_axis_positions):  
        # Extract the unique z positions for this scan tile, use the z_idx
        z_tile_positions = np.unique(
            stage_positions_array[stage_positions_array[:, 2] == scan_axis_positions[scan_tile]][:, 0]
        )
        ao_stage_positions.append(
            {
                "z": z_tile_positions[z_idx],
                "y": ao_tile_axis_position,
                "x": scan_tile_pos
            }
        )
if verbose:
    print(
        "\nAO grid positions:",
        f"\nY tile position: {ao_tile_axis_position}",
        f"\nScan axis dx: {scan_tile_dx}",
        f"\nScan axis positions: {ao_scan_axis_positions}",
        f"\nNumber of z-planes: {num_z_positions}"    
    )
# Save AO optimization results here
num_ao_pos = len(ao_stage_positions)
ao_grid_wfc_coeffs = np.zeros((num_ao_pos, 11))
ao_grid_wfc_positions = np.zeros((num_ao_pos, 11))

# Run AO optimization for each stage position
print("\nGenerating AO map:")
print(f"Number of positions: {num_ao_pos}")

# Visit AO stage positions and measure best WFC positions
for ao_pos_idx in range(num_ao_pos):
    
    if verbose:
        print(f"\nMoving stage to: {ao_stage_positions[ao_pos_idx]}")
    target_x = ao_stage_positions[ao_pos_idx]["x"]
    target_y = ao_stage_positions[ao_pos_idx]["y"]
    target_z = ao_stage_positions[ao_pos_idx]["z"]

    
    
    ao_grid_wfc_coeffs[ao_pos_idx] = np.random.random(11)
    ao_grid_wfc_positions[ao_pos_idx] = np.random.random(11)

# Define a threshold for matching AO grid positions
AO_POSITION_THRESHOLD = 0

# Convert AO positions and stage positions into structured arrays for efficient lookup
ao_stage_positions_array = np.array([(pos["z"], pos["y"], pos["x"]) for pos in ao_stage_positions])
stage_positions_array = np.array([(pos["z"], pos["y"], pos["x"]) for pos in stage_positions])

# Pre-allocate arrays for results
position_wfc_coeffs = np.zeros((len(stage_positions), 11))
position_wfc_positions = np.zeros((len(stage_positions), 11))

# Find the closest AO match per stage position
for pos_idx, (stage_z, stage_y, stage_x) in enumerate(stage_positions_array):
    # Find AO positions within threshold range of the current stage X
    valid_mask = ao_stage_positions_array[:, 2] - stage_x > AO_POSITION_THRESHOLD
    relevant_ao_positions = ao_stage_positions_array[valid_mask]
    
    if relevant_ao_positions.size == 0:
        print(f"Warning: No AO match found for stage position {stage_positions[pos_idx]}")
        continue  # Skip this position if no match

    # Compute Euclidean distance only among relevant AO positions
    distances = np.linalg.norm(relevant_ao_positions - [stage_z, stage_y, stage_x], axis=1)
    
    # Get index of closest AO position
    ao_grid_idx = np.where(valid_mask)[0][np.argmin(distances)]
    
    # Assign the corresponding AO grid data
    position_wfc_positions[pos_idx] = ao_grid_wfc_positions[ao_grid_idx]
    position_wfc_coeffs[pos_idx] = ao_grid_wfc_coeffs[ao_grid_idx]

    if verbose:
        print(
            f"AO grid position: {ao_stage_positions[ao_grid_idx]}",
            f"Exp. stage position: {stage_positions[pos_idx]}"
        )