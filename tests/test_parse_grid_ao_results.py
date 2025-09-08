from pathlib import Path
import matplotlib.pyplot as plt
from opm_v2.utils import sensorless_ao as ao
import numpy as np


showfig = False
root_dir = Path("/home/steven/Documents/qi2lab/projects/local_working_files/OPM/opm_ao/in_IB_AO_ao_results")

import zarr

# process grid acquisitions, need to parse mulitle directories by defualt
grid_results = []
for _d in root_dir.iterdir():
    zarr_path = [p for p in _d.glob('*ao_results.zarr')]
    
    if len(zarr_path)==0:
        pass
    else:
        grid_results.append(ao.load_optimization_results(zarr_path[0]))    

    """  
    ao_results = {
        "all_images":all_images,
        "all_metrics":all_metrics,
        "metrics_per_iteration":metrics_per_iteration,
        "images_per_iteration":images_per_iteration,
        "optimal_coeffs":optimal_coeffs,
        "modes_to_optimize":modes_to_optimize,
        # "optimized_phase":optimized_phase,
        "mode_names":zernike_mode_names,
    }
    ao_results.update(results.attrs)
    
    """
# Combine grid data to take place of iterations
if 'grid' in _d.name:
    optimal_coeffs = []
    all_metrics = []
    for ao_results in grid_results:
        
        zernike_mode_names = ao_results['mode_names']
        modes_to_optimize = ao_results['modes_to_optimize']
        save_dir_path = None
        showfig = True
        
        ao.plot_metric_progress(
            all_metrics = all_metrics,
            modes_to_optimize = modes_to_optimize,
            num_iterations = optimal_coeffs.shape[1],
            zernike_mode_names = zernike_mode_names,
            save_dir_path = root_dir,
            show_fig = True,
        )
        ao.plot_zernike_coeffs(
            optimal_coeffs=optimal_coeffs,
            zernike_mode_names=zernike_mode_names,
            save_dir_path=root_dir,
            show_fig=True,
            x_range=0.1
        )

        optimal_coeffs.append(ao_results['optimal_coeffs'])
        all_metrics.append(ao_results['all_metrics'])
        
        
        
    optimal_coeffs = np.asarray(optimal_coeffs).reshape([len(grid_results), len(zernike_mode_names)])
    all_metrics = np.asarray(all_metrics)


    num_iterations = optimal_coeffs.shape[1]
    
    ao.plot_metric_progress(
        all_metrics = all_metrics,
        modes_to_optimize = modes_to_optimize,
        num_iterations = optimal_coeffs.shape[1],
        zernike_mode_names = zernike_mode_names,
        save_dir_path = root_dir,
        show_fig = True,
    )
    ao.plot_zernike_coeffs(
        optimal_coeffs=optimal_coeffs,
        zernike_mode_names=zernike_mode_names,
        save_dir_path=root_dir,
        show_fig=True,
        x_range=0.1
    )
