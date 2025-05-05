import numpy as np
stage_positions = []
                    
# grab grid plan extents
min_y_pos = 0
max_y_pos = 1000
min_x_pos = 0
max_x_pos = 1000

min_z_pos = 0
max_z_pos = 20
coverslip_slope = 0
camera_crop_y = 386
camera_crop_x = 1900
tile_axis_overlap = 0.2
opm_angle_scale = np.sin(np.deg2rad(30))
pixel_size_um = 0.115
scan_tile_overlap_um = 40
scan_tile_overlap_mm = scan_tile_overlap_um/1000
coverslip_max_dz = 2.0
scan_axis_step_um = 0.4
exposure_s = 0.030
n_active_channels = 1
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
    stage_scan_max_range = coverslip_max_dz / coverslip_slope
else:
    stage_scan_max_range = float(200)  

DEBUG=True
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
scan_tile_z_step_um = np.round(cs_range_um / n_scan_positions, 2)
scan_axis_step_mm = scan_axis_step_um / 1000. # unit: mm
scan_axis_start_mm = min_x_pos / 1000. # unit: mm
scan_axis_end_mm = max_x_pos / 1000. # unit: mm
scan_tile_length_mm = scan_tile_length_um / 1000. # unit: mm

scan_axis_start_pos_mm = np.full(n_scan_positions, scan_axis_start_mm)
scan_axis_end_pos_mm = np.full(n_scan_positions, scan_axis_end_mm)
scan_axis_z_positions = np.full(n_scan_positions, min_z_pos)
for ii in range(n_scan_positions):
    scan_axis_start_pos_mm[ii] = scan_axis_start_mm + ii * (scan_tile_length_mm - scan_tile_overlap_mm)
    scan_axis_end_pos_mm[ii] = scan_axis_start_pos_mm[ii] + scan_tile_length_mm
    scan_axis_z_positions[ii] = min_z_pos + ii * scan_tile_z_step_um
    
scan_axis_start_pos_mm = np.round(scan_axis_start_pos_mm,2)
scan_axis_end_pos_mm = np.round(scan_axis_end_pos_mm,2)
scan_tile_length_w_overlap_mm = np.round(np.abs(scan_axis_end_pos_mm[0]-scan_axis_start_pos_mm[0]),2)
scan_axis_positions = np.rint(scan_tile_length_w_overlap_mm / scan_axis_step_mm).astype(int)
scan_axis_speed = np.round(scan_axis_step_mm / exposure_s / n_active_channels,5) 

if DEBUG: 
    print(f"Stage scan positions, units: mm")
    print(f"Is the scan tile w/ overlap the same as the scan tile length?: {scan_tile_length_mm==scan_tile_length_w_overlap_mm}")
    print(f'Number scan tiles: {n_scan_positions}')
    print(f'Scan tile length um: {scan_tile_length_um}')
    print(f'Scan tile overlap um: {scan_tile_overlap_um}')
    print(f'Scan axis start positions: {scan_axis_start_pos_mm}.')
    print(f'Scan axis end positions: {scan_axis_end_pos_mm}.')
    print(f'Scan axis positions: {scan_axis_positions}')
    print(f'Scan axis coverslip offsets: {scan_tile_z_step_um}')
    print(f'Scan tile size: {scan_tile_length_w_overlap_mm}')
    print(f'Scan axis speed (mm/s): {scan_axis_speed}\n')