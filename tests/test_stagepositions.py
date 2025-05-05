import numpy as np
import matplotlib.pyplot as plt
DEBUG = True

camera_crop_y = 128
camera_crop_x = 1900
coverslip_slope = 0.008
scan_axis_step_um = 0.4
scan_tile_overlap_um = 50
scan_tile_overlap_mm = scan_tile_overlap_um/1000.
tile_axis_overlap = 0.15
opm_angle_scale = np.sin((np.pi/180.)*float(30))
n_active_channels = 2
exposure_ms = 20
# exposure_s = 0.020
pixel_size_um = 0.115
coverslip_max_dz = 3.0

scan_tile_overlap_um = camera_crop_y * opm_angle_scale * pixel_size_um + 20
scan_tile_overlap_mm = scan_tile_overlap_um/1000.
#--------------------------------------------------------------------#
# Grab grid plan extents
min_y_pos = -2500
max_y_pos = 500
min_x_pos = -11201
max_x_pos = -8200
min_z_pos = 116
max_z_pos = 111

# Set grid axes ranges
range_x_um = np.round(np.abs(max_x_pos - min_x_pos),2)
range_y_um = np.round(np.abs(max_y_pos - min_y_pos),2)
range_z_um = np.round(np.abs(max_z_pos - min_z_pos),2)

# Define coverslip bounds, to offset Z positions
cs_min_pos = min_z_pos
cs_max_pos = cs_min_pos + range_x_um * coverslip_slope
cs_range_um = np.round(np.abs(cs_max_pos - cs_min_pos),2)

#--------------------------------------------------------------------#
# Calculate tile steps
z_axis_step_max = (
    camera_crop_y
    * pixel_size_um
    * opm_angle_scale 
    * (1-tile_axis_overlap)
)
tile_axis_step_max = (
    camera_crop_x
    * pixel_size_um
    * (1-tile_axis_overlap)
)

#--------------------------------------------------------------------#
# Correct directions for stage moves
if min_z_pos > max_z_pos:
    z_axis_step_max *= -1
if min_x_pos > max_x_pos:
    min_x_pos, max_x_pos = max_x_pos, min_x_pos
    
#--------------------------------------------------------------------#
# Generate coverslip offset along the scan axis
if coverslip_slope != 0:
    # Set the max scan range using coverslip slope
    stage_scan_max_range = np.abs(coverslip_max_dz / coverslip_slope)
else:
    stage_scan_max_range = 100

if DEBUG: 
    print("Compiles settings to generate positions:")
    print(f'Scan start: {min_x_pos}')
    print(f'Scan end: {max_x_pos}')
    print(f'Tile start: {min_y_pos}')
    print(f'Tile end: {max_y_pos}')
    print(f"Z position minimum:{min_z_pos}")
    print(f"Z position maximum:{max_z_pos}")
    print(f'Coverslip slope: {coverslip_slope}')
    print(f'Coverslip low: {cs_min_pos}')
    print(f'Coverslip high: {cs_max_pos}')
    print(f'Max scan range (CS used?:{coverslip_slope!=0}): {stage_scan_max_range}\n')
    
#--------------------------------------------------------------------#
# calculate scan axis tile locations, units: mm and s

# Break scan range up using max scan range
if stage_scan_max_range >= range_x_um:
    n_scan_positions = 1
    scan_tile_length_um = range_x_um
else:
    n_scan_positions = int(
        np.ceil(range_x_um / (stage_scan_max_range))
    )
    scan_tile_length_um = np.round(
        (range_x_um/n_scan_positions) + (n_scan_positions-1)*(scan_tile_overlap_um/(n_scan_positions)),
        2
    )
scan_axis_step_mm = scan_axis_step_um / 1000. # unit: mm
scan_axis_start_mm = min_x_pos / 1000. # unit: mm
scan_axis_end_mm = max_x_pos / 1000. # unit: mm
scan_tile_length_mm = scan_tile_length_um / 1000. # unit: mm

scan_axis_start_pos_mm = np.full(n_scan_positions, scan_axis_start_mm)
scan_axis_end_pos_mm = np.full(n_scan_positions, scan_axis_end_mm)
for ii in range(n_scan_positions):
    scan_axis_start_pos_mm[ii] = scan_axis_start_mm + ii * (scan_tile_length_mm - scan_tile_overlap_mm)
    scan_axis_end_pos_mm[ii] = scan_axis_start_pos_mm[ii] + scan_tile_length_mm
    
scan_axis_start_pos_mm = np.round(scan_axis_start_pos_mm,2)
scan_axis_end_pos_mm = np.round(scan_axis_end_pos_mm,2)
scan_tile_length_w_overlap_mm = np.round(np.abs(scan_axis_end_pos_mm[0]-scan_axis_start_pos_mm[0]),2)
scan_axis_positions = np.rint(scan_tile_length_w_overlap_mm / scan_axis_step_mm).astype(int)
scan_axis_speed = np.round(scan_axis_step_mm / exposure_s / n_active_channels,5) 

if DEBUG: 
    print(f"scan speed values: {scan_axis_step_mm}, {exposure_s}, {n_active_channels}")
    print(f"Stage scan positions, units: mm")
    print(f"Is the scan tile w/ overlap the same as the scan tile length?: {scan_tile_length_mm==scan_tile_length_w_overlap_mm}")
    print(f'Number scan tiles: {n_scan_positions}')
    print(f'Scan tile length um: {scan_tile_length_um}')
    print(f'Scan tile overlap um: {scan_tile_overlap_um}')
    print(f'Scan axis start positions: {scan_axis_start_pos_mm}.')
    print(f'Scan axis end positions: {scan_axis_end_pos_mm}.')
    print(f'Scan axis positions: {scan_axis_positions}')
    print(f'Scan tile size: {scan_tile_length_w_overlap_mm}')
    print(f'Scan axis speed (mm/s): {scan_axis_speed}\n')
    print(f'Scan axis speed (mm/s): {scan_axis_speed}\n')

#--------------------------------------------------------------------#
# Generate tile axis positions
n_tile_positions = int(np.ceil(range_y_um / tile_axis_step_max)) + 1
tile_axis_positions = np.round(np.linspace(min_y_pos, max_y_pos, n_tile_positions),2)
if n_tile_positions==1:
    tile_axis_step = 0
else:
    tile_axis_step = tile_axis_positions[1]-tile_axis_positions[0]

if DEBUG:
    print("Tile axis positions units: um")
    print(f"Tile axis positions: {tile_axis_positions}")
    print(f"Num tile axis positions: {n_tile_positions}")
    print(f"Tile axis step: {tile_axis_step}")

#--------------------------------------------------------------------#
# Generate z axis positions, ignoring coverslip slope
n_z_positions = int(np.ceil(range_z_um / z_axis_step_max)) + 1
z_positions = np.round(np.linspace(min_z_pos, max_z_pos, n_z_positions), 2)
if n_z_positions==1:
    z_axis_step_um = 0.
else:
    z_axis_step_um = np.round(z_positions[1] - z_positions[0],2)

# Calculate the stage z change along the scan axis
dz_per_scan_tile = (cs_range_um / n_scan_positions) * np.sign(coverslip_slope)

if DEBUG:
    print("\nZ axis positions, units: um")
    print(f"Z axis positions: {z_positions}")
    print(f"Z axis range: {range_z_um} um")
    print(f"Z axis step: {z_axis_step_um} um")
    print(f"Num z axis positions: {n_z_positions}")
    print(f"Z offset per x-scan-tile: {dz_per_scan_tile} um")

#--------------------------------------------------------------------#
# Generate stage positions 
n_stage_positions = n_scan_positions * n_tile_positions * n_z_positions
stage_positions = []
for z_idx in range(n_z_positions):
    for scan_idx in range(n_scan_positions):
        for tile_idx in range(n_tile_positions):
            stage_positions.append(
                {
                    "x": float(np.round(scan_axis_start_pos_mm[scan_idx]*1000, 2)),
                    "y": float(np.round(tile_axis_positions[tile_idx], 2)),
                    "z": float(np.round(z_positions[z_idx] + dz_per_scan_tile*scan_idx, 2))
                }
            )
if DEBUG:
    print(stage_positions)