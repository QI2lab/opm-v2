import json
import time
from pathlib import Path
from typing import List

import numpy as np
import wavekit_py as wkpy
from numpy.typing import NDArray

DEBUGGING = True
MIRROR_SETTLE_MS = 100

haso_config_file_path = r"C:/Users/qi2lab/Documents/github/opm_v2/src/opm_v2/hardware/wfc_configuration_files/WFS_HASO4_VIS_7635.dat"
wfc_config_file_path =  r"C:/Users/qi2lab/Documents/github/opm_v2/src/opm_v2/hardware/wfc_configuration_files/WaveFrontCorrector_mirao52-e_0329.dat"
interaction_matrix_file_path = "C:/Users/qi2lab/Documents/github/opm_v2/src/opm_v2/hardware/wfc_configuration_files/202508026_tilted_interaction_matrix.aoc"
coeff_file_path = None
system_flat_file_path = r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_configuration_files\20250827_brightness_from_wfs_opt_max_wfc_voltage.wcs"
n_modes = 32
modes_to_ignore = None
n_positions = 1
output_path = None
focus_mode = 1.65


#---------------------------------------------#
# Start wfc 
#---------------------------------------------#

# Wavefront corrector and set objects
wfc = wkpy.WavefrontCorrector(
    config_file_path = str(wfc_config_file_path)
)
wfc_set = wkpy.WavefrontCorrectorSet(wavefrontcorrector = wfc)
wfc.connect(True)

# create corrdata manager object and compute command matrix
corr_data_manager = wkpy.CorrDataManager(
    haso_config_file_path = str(haso_config_file_path),
    interaction_matrix_file_path = str(interaction_matrix_file_path)
)
corr_data_manager.set_command_matrix_prefs(32,True)
corr_data_manager.compute_command_matrix()

wfc.set_temporization(MIRROR_SETTLE_MS)

# create the configuration object
haso_config, haso_specs, _ = wkpy.HasoConfig.get_config(config_file_path=str(haso_config_file_path))

# construct pupil dimensions from haso specs
pupil_dimensions = wkpy.dimensions(haso_specs.nb_subapertures,haso_specs.ulens_step)

# initiate an empty pupil
pupil = wkpy.Pupil(dimensions=pupil_dimensions, value=False)
pupil_buffer = corr_data_manager.get_greatest_common_pupil()
pupil.set_data(pupil_buffer)

center, radius = wkpy.ComputePupil.fit_zernike_pupil(
    pupil,
    wkpy.E_PUPIL_DETECTION.AUTOMATIC,
    wkpy.E_PUPIL_COVERING.CIRCUMSCRIBED,
    False
)
# create modal coeff object
modal_coef = wkpy.ModalCoef(modal_type=wkpy.E_MODAL.ZERNIKE)
modal_coef.set_zernike_prefs(
    zernike_normalisation=wkpy.E_ZERNIKE_NORM.RMS, 
    nb_zernike_coefs_total=32, 
    coefs_to_filter=modes_to_ignore,
    projection_pupil=wkpy.ZernikePupil_t(
        center,
        radius
    )
)  
# update modal data with zero coeffs.
modal_coef.set_data(
    coef_array = np.zeros(n_modes),
    index_array = np.arange(1, 32+1, 1),
    pupil = pupil
) 
#---------------------------------------------#
# Set up wfc positions and mirror position tracking
#---------------------------------------------#

if system_flat_file_path is not None:
    system_flat_voltage = np.asarray(
        wfc.get_positions_from_file(str(system_flat_file_path))
    )
# wfc.move_to_absolute_positions(system_flat_voltage)
wfc.move_to_absolute_positions(np.asarray(wfc_set.get_flat_mirror_positions()))
print(wfc.get_current_positions()==wfc_set.get_flat_mirror_positions())

# update modal data with zero coeffs.
test_coefs = np.full(32, 0.1)
modal_coef.set_data(
    coef_array = test_coefs,
    index_array = np.arange(1, 32+1, 1),
    pupil = pupil
) 

# create a new haso_slope from the new modal coefficients
haso_slopes = wkpy.HasoSlopes(
    modalcoef = modal_coef, 
    config_file_path=str(haso_config_file_path)
)

deltas = corr_data_manager.compute_delta_command_from_delta_slopes(delta_slopes=haso_slopes)

print(deltas)
