"""
Set the AO mirror positions while script is running
"""

from pathlib import Path
from opm_v2.hardware.AOMirror import AOMirror

# WFC configuration paths
wfc_config_file_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\WaveFrontCorrector_mirao52-e_0329.dat')
wfc_correction_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\correction_data_backup_starter.aoc")
haso_config_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\WFS_HASO4_VIS_7635.dat")
mirror_state_output_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_position_files')

# WFC positions file, use None for zeros
wfc_flat_file_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_setup\20250818_154141_laser_optimization\20250818_laser_wfc_voltage.wcs')

# Load ao_mirror controller using the given positions file
ao_mirror = AOMirror(wfc_config_file_path = wfc_config_file_path,
                    haso_config_file_path = haso_config_file_path,
                    interaction_matrix_file_path = wfc_correction_file_path,
                    system_flat_file_path = wfc_flat_file_path,
                    mirror_flat_file_path = None,
                    coeff_file_path = None,
                    control_mode= 'voltage',
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