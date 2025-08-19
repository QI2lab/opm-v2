from pathlib import Path
import matplotlib.pyplot as plt
from opm_v2.utils import sensorless_ao as ao
import numpy as np


showfig = False
root_dir = Path("/home/steven/Documents/qi2lab/projects/local_working_files/OPM/autophagy_full_run_ao_results")
import zarr
# process grid acquisitions, need to parse mulitle directories by defualt
grid_results = []
for _d in root_dir.iterdir():
    zarr_path = [p for p in _d.glob('*ao_results.zarr')][0]
    grid_results.append(ao.load_optimization_results(zarr_path))    

# Combine grid data to take place of iterations
if 'grid' in _d.name:
    optimal_coefficients = []
    all_metrics = []
    for ao_results in grid_results:
        optimal_coefficients.append(ao_results['coefficients_per_iteration'])
        all_metrics.append(ao_results['all_metrics'])
        
        zernike_mode_names = ao_results['mode_names']
        modes_to_optimize = ao_results['modes_to_optimize']
        save_dir_path = None
        showfig = True

    optimal_coefficients = np.asarray(optimal_coefficients).reshape([len(grid_results), len(zernike_mode_names)])
    all_metrics = np.asarray(all_metrics)


    
    ao.plot_metric_progress(
        all_metrics = all_metrics,
        modes_to_optimize = modes_to_optimize,
        num_iterations = len(grid_results),
        zernike_mode_names = zernike_mode_names,
        save_dir_path = root_dir,
        show_fig = True,
    )
    ao.plot_zernike_coeffs(
        optimal_coefficients=optimal_coefficients,
        zernike_mode_names=zernike_mode_names,
        save_dir_path=root_dir,
        show_fig=True,
        x_range=0.1
    )
