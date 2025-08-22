#! /usr/bin/python

""" 
Generate the interaction matrix. Copied from wavekitpy examples.
- Requires both WFC and WFS be connected
- Modify output path prefix for different correction callibrations
"""

import wavekit_py as wkpy
from pathlib import Path
import ndv
import numpy as np
import tifffile as tf


"""Setup configuration paths
"""
wfs_config_file_path = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\WFS_HASO4_VIS_7635.dat"
)
wfc_config_file_path = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\WaveFrontCorrector_mirao52-e_0329.dat"
)
# Set the output path
output_prefix = '202508021_tilted'
root_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\interaction_matrices")
output_image_path = root_path / Path(output_prefix + r'_interaction_images.tiff')
output_file_path = root_path / Path(output_prefix + "_interaction_matrix.aoc")

"""Instantiate objects
"""
camera = wkpy.Camera(config_file_path = str(wfs_config_file_path))
wavefrontcorrector = wkpy.WavefrontCorrector(config_file_path = str(wfc_config_file_path))
corr_data_manager = wkpy.CorrDataManager(
    haso_config_file_path = str(wfs_config_file_path),
    wfc_config_file_path = str(wfc_config_file_path)
    )

"""Connect to hardware
"""
camera.connect()
wavefrontcorrector.connect(True)

exposure_duration = 7000
camera.set_parameter_value('exposure_duration_us',exposure_duration)
print('Exposure duration requested : ' + str(exposure_duration) + 'us')
print('Exposure duration applied : '   + str(camera.get_parameter_value('exposure_duration_us')) + 'us')

push_pull_value = 0.5
corr_data_manager.set_calibration_prefs(push_pull_value)
size = corr_data_manager.get_calibration_matrix_size()
print('Calibration matrix size : ('+str(size.X)+' x '+str(size.Y)+')')

images = []

"""HasoSlopes array to store computed slopes
"""
specs = corr_data_manager.get_specifications()
hasoslopes_array = []
for i in range(2 * specs.nb_actuators):
    hasoslopes_array.append(
        wkpy.HasoSlopes(
            dimensions = specs.dimensions,
            serial_number = specs.haso_serial_number
            )
        )

"""Calibration process
"""
start_subpupil = wkpy.uint2D(20,16)
for i in range(specs.nb_actuators):
    prefs = corr_data_manager.get_actuator_prefs(i)
    if(prefs.validity==wkpy.E_ACTUATOR_CONDITIONS.VALID):
        push, pull = corr_data_manager.get_calibration_commands(i)
        """Get push interaction slopes
        """
        wavefrontcorrector.move_to_absolute_positions(push)
        image = camera.snap_raw_image()
        images.append(image.get_data().astype(np.uint16))
        hasoslopes_array[(2 * i)] = wkpy.HasoSlopes(
            image = image,
            config_file_path = str(wfs_config_file_path),
            start_subpupil = start_subpupil
            )
        """Get pull interaction slopes
        """
        wavefrontcorrector.move_to_absolute_positions(pull)
        image = camera.snap_raw_image()
        images.append(image.get_data().astype(np.uint16))
        hasoslopes_array[(2 * i) + 1] = wkpy.HasoSlopes(
            image = image,
            config_file_path = str(wfs_config_file_path),
            start_subpupil = start_subpupil
            )
        print('Calibration process experiment for actuator ' + str(i+1) + '/' + str(specs.nb_actuators) + ' succeed.')

"""Compute interaction matrix
"""
corr_data_manager.compute_interaction_matrix(hasoslopes_array)
corr_data_manager.save_backup_file(str(output_file_path), 'Tilted')
print(f'Correction data saved to file {str(output_file_path)} in Examples directory.')

"""Set computation prefs
"""
nb_kept_modes = 32
corr_data_manager.set_command_matrix_prefs(
    nb_kept_modes,
    False
    )
corr_data_manager.compute_command_matrix()
"""Get vector of singular values
"""
influence_array = corr_data_manager.get_diagnostic_singular_vector()
for i in range(nb_kept_modes):
    print('Singular value at index '+str(i)+': '+str(influence_array[i]))

camera.stop()
camera.disconnect()

ndv.imshow(np.stack(images))

tf.imwrite(output_image_path, images)

del camera
del wavefrontcorrector
del corr_data_manager
del hasoslopes_array
del image
