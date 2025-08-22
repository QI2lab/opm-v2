import msvcrt

#! /usr/bin/python
# TODO: Update paths
#----
""" WARNING ! This example is Hardware dependant.
Please ensure that hardware is connected,
And use your own Haso configuration file and acquisition system.
"""
""" WARNING ! If you have no Correction data backup file,
Please exit this example and run the correction_data example
"""
#----

print('WARNING ! If you have no Correction data backup file, \nPlease exit this example and run the correction_data example')

"""Import Imagine Optic's Python interface
"""
import os, sys
sys.path.append('./../..')
import wavekit_py as wkpy
from pathlib import Path
import numpy as np

"""Get configuration file path
"""
wfs_config_file_path = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\WFS_HASO4_VIS_7635.dat"
)
wfc_config_file_path = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\WaveFrontCorrector_mirao52-e_0329.dat"
)
wfc_correction_file_path = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\20250808_laser_interaction_matrix.aoc"
)
output_prefix = '20250808_laser'
output_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration") / Path(output_prefix + "_closed_loop_output.wcs")

print('Haso configuration file path set to :' + str(wfs_config_file_path))
print('Corrector configuration file path set to :' + str(wfc_config_file_path))
print('Correction data backup file path set to :' + str(wfc_correction_file_path))
print('Output file set to :' + str(output_file_path))


"""Create camera, wavefrontcorrector and corrdatamanager objects
"""
camera = wkpy.Camera(config_file_path = str(wfs_config_file_path))
wavefront_corrector = wkpy.WavefrontCorrector(config_file_path = str(wfc_config_file_path))
corr_data_manager = wkpy.CorrDataManager(haso_config_file_path = str(wfs_config_file_path), interaction_matrix_file_path = str(wfc_correction_file_path))
"""Connect hardware
"""
camera_exposure = 8000
camera.connect()
camera.set_parameter_value('exposure_duration_us', camera_exposure)
wavefront_corrector.connect(True)
if output_file_path.exists():
    best_positions = np.asarray(wavefront_corrector.get_positions_from_file(str(output_file_path)),dtype=np.float32)
    print(f"Initial wavefront corrector positions: {wavefront_corrector.get_current_positions()}")
    wavefront_corrector.move_to_absolute_positions(best_positions)
    print(f"Starting wavefront corrector positions: {wavefront_corrector.get_current_positions()}")
else:
    wavefront_corrector.move_to_absolute_positions(np.zeros(52, dtype=np.float32))

"""Acquire an image
"""
image = camera.snap_raw_image()
start_subpupil = wkpy.uint2D(20,16)
ref_hasoslopes = wkpy.HasoSlopes(
    image = image,
    config_file_path = str(wfs_config_file_path),
    start_subpupil = start_subpupil
    )
wkpy.SlopesPostProcessor.apply_filter(
    ref_hasoslopes,
    True,
    True,
    True,
    False,
    False,
    False
    )
wavefront_corrector.set_temporization(100)

corr_data_manager.set_command_matrix_prefs(
    32,
    False
    )
corr_data_manager.compute_command_matrix()

loop_smoothing = wkpy.LoopSmoothing(level = "HIGH")
gain = 0.8
best_value = None
best_commands = []

"""Loop
"""
try :
    while True:
        image = camera.snap_raw_image()
        hasoslopes = wkpy.HasoSlopes(
            image = image,
            config_file_path = str(wfs_config_file_path),
            start_subpupil = start_subpupil
            )
        delta_hasoslopes = wkpy.SlopesPostProcessor.apply_substractor(hasoslopes, ref_hasoslopes)
        phase = wkpy.Phase(
            hasoslopes = delta_hasoslopes,
            type_ = wkpy.E_COMPUTEPHASESET.ZONAL,
            filter_ = [True, True, True, False, False]
            )
        

        if best_value == None:
            print ('Starting RMS value : ' + str(np.round(phase.get_statistics().rms,5)) + '\t(Press any key or Ctrl+C to stop the closed loop)')
            best_value = np.round(phase.get_statistics().rms,5)
        else:
            if np.round(phase.get_statistics().rms,5) < best_value:
                best_value = np.round(phase.get_statistics().rms,5)
                print('New best value : ' + str(best_value))
                best_commands = wavefront_corrector.get_current_positions()
    
        delta_commands, applied_gain = corr_data_manager.compute_closed_loop_iteration(
            delta_hasoslopes,
            False,
            loop_smoothing,
            gain
            )
        wavefront_corrector.move_to_relative_positions(
            delta_commands
            )

        if msvcrt.kbhit():
            break
except KeyboardInterrupt:
    pass

print(best_commands)
print(wavefront_corrector.get_current_positions())
wavefront_corrector.move_to_absolute_positions(np.asarray(best_commands, dtype=np.float32))
wavefront_corrector.save_current_positions_to_file(str(output_file_path))

"""End of loop
"""
camera.stop()
camera.disconnect()
del loop_smoothing
del phase
del delta_hasoslopes
del ref_hasoslopes
del hasoslopes
del image
del corr_data_manager
del wavefront_corrector
del camera
