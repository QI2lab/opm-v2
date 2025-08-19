
""" WARNING ! This example is Hardware dependant.
Please ensure that hardware is connected,
And use your own Haso configuration file and acquisition system.
"""
#----

"""Import Imagine Optic's Python interface
"""
import wavekit_py as wkpy
from pathlib import Path
import numpy as np
from tifffile import imwrite
import time

"""Get configuration file path
"""
wfs_config_file_path = Path(r"C:\Program Files (x86)\Imagine Optic\Configuration Files\WFS_HASO4_VIS_7635.dat")
wfc_config_file_path = Path(r"C:\Program Files (x86)\Imagine Optic\Configuration Files\WaveFrontCorrector_mirao52-e_0329.dat")

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
wavefrontcorrector.connect(True)
wavefrontcorrector.move_to_absolute_positions(np.full(wavefrontcorrector.nb_actuators, 0))

push_pull_value = 0.5
corr_data_manager.set_calibration_prefs(push_pull_value)
size = corr_data_manager.get_calibration_matrix_size()
print('Calibration matrix size : ('+str(size.X)+' x '+str(size.Y)+')')

specs = corr_data_manager.get_specifications()

# images = []
start_subpupil = wkpy.uint2D(20,15)
for i in range(specs.nb_actuators):
    prefs = corr_data_manager.get_actuator_prefs(i)
    if(prefs.validity==wkpy.E_ACTUATOR_CONDITIONS.VALID):
        push, pull = corr_data_manager.get_calibration_commands(i)
        """Get push interaction slopes
        """
        wavefrontcorrector.move_to_absolute_positions(push)
        time.sleep(2)
        wavefrontcorrector.move_to_absolute_positions(pull)
        time.sleep(2)
        print('Calibration process experiment for actuator ' + str(i+1) + '/' + str(specs.nb_actuators) + ' succeed.')

wavefrontcorrector.disconnect()

del wavefrontcorrector
