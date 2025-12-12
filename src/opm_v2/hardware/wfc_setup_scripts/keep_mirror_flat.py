"""
Set the AO mirror positions while script is running
"""

from pathlib import Path

from opm_v2.hardware.AOMirror import AOMirror

# WFC configuration paths
wfc_config_file_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_configuration_files\WaveFrontCorrector_mirao52-e_0329.dat')
wfc_correction_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_configuration_files\202508026_tilted_interaction_matrix.aoc")
haso_config_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_configuration_files\WFS_HASO4_VIS_7635.dat")
mirror_state_output_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_position_files')

# WFC positions file
# wfc_flat_file_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_position_files\20250826_laser_tilted_curvature_closed_loop_output.wcs')
# wfc_flat_file_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_position_files\20250826_laser_tilted_curvature_with_tilt_closed_loop_output.wcs')
# wfc_flat_file_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_position_files\20250826_laser_tilted_no_curvature_closed_loop_output.wcs')
# wfc_flat_file_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_position_files\20250826_laser_tilted_no_curvature_with_tilt_closed_loop_output.wcs')
# wfc_flat_file_path = None # Use None for flat (zero) positions
# wfc_flat_file_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_setup_scripts\20250827_141952_laser_brightness_brightness_from_wfs_opt_using_max\20250827_brightness_from_wfs_opt_max_wfc_voltage.wcs')
wfc_flat_file_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_setup_scripts\20250827_153811_laser_brightness_brightness_from_wfs_opt_using_max\20250827_brightness_from_wfs_opt_max_wfc_voltage.wcs')
# Load ao_mirror controller using the given positions file
ao_mirror = AOMirror(wfc_config_file_path = wfc_config_file_path,
                    haso_config_file_path = haso_config_file_path,
                    interaction_matrix_file_path = wfc_correction_file_path,
                    system_flat_file_path = wfc_flat_file_path,
                    mirror_flat_file_path = None,
                    coeff_file_path = None,
                    output_path = mirror_state_output_path,
                    n_modes = 32,
                    modes_to_ignore = [])

print('AO mirror set to positions . . .')
print('type exit to end\n')

while True:
    answer = input("Type 'exit' to quit: ")
    if answer.strip().lower() == 'exit':
        print("Exiting...")
        break