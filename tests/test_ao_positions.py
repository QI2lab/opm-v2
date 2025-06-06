from scipy.spatial import cKDTree
import numpy as np
num_scan_positions = 3
num_tile_positions = 3
verbose=True
# Example list of stage positions

stage_positions = [
    {"z": 180, "y": 1350.15, "x": -12990},
    {"z": 196.67, "y": 1350.15, "x": -12990},
    {"z": 213.33, "y": 1350.15, "x": -12990},
    {"z": 230., "y": 1350.15, "x": -12990},
]
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

# Convert to array for easier manipulation
stage_positions_array = np.array(
    [(pos["z"], pos["y"], pos["x"]) for pos in stage_positions]
)

z_vals = stage_positions_array[:, 0]
y_vals = stage_positions_array[:, 1]
x_vals = stage_positions_array[:, 2]

# Define unique sorted coordinates
unique_x = np.unique(x_vals)
unique_y = np.unique(y_vals)
unique_z = np.unique(z_vals)

# AO grid interpolation
ao_x = np.linspace(unique_x.min(), unique_x.max(), num_scan_positions)
ao_y = np.linspace(unique_y.min(), unique_y.max(), num_tile_positions)
ao_z = np.unique(z_vals)  # Use real z-steps (do not interpolate)

# Use a KD-tree to find nearest real stage positions to AO targets
tree = cKDTree(stage_positions_array)

ao_stage_positions = []
for z in ao_z:
    for y in ao_y:
        for x in ao_x:
            # Query nearest real position to (z, y, x)
            dist, idx = tree.query([z, y, x])
            matched = stage_positions_array[idx]
            ao_stage_positions.append({
                "z": round(matched[0], 2),
                "y": round(matched[1], 2),
                "x": round(matched[2], 2)
            })

if verbose:
    for pos in ao_stage_positions:
        print(pos)