from pathlib import Path
import matplotlib.pyplot as plt
from opm_v2.utils import sensorless_ao as ao
import numpy as np
import zarr


showfig = False
data_path = Path(
    r'/home/steven/Documents/qi2lab/projects/local_working_files/OPM/opm_ao/ao_results.zarr'
)

results = ao.load_optimization_results(data_path)

best_metrics = results['best_metrics']
all_images = results['all_images']
all_metrics = results['all_metrics']
metrics_per_iteration = results['metrics_per_iteration']
images_per_iteration = results['images_per_iteration']
coeff_per_iteration = results['coefficients_per_iteration']
modes_to_optimize = results['modes_to_optimize']
zernike_mode_names = results['mode_names']
modes_to_use_names = [zernike_mode_names[i] for i in modes_to_optimize]
num_iterations = results['num_iterations']
num_modes = len(modes_to_optimize)
num_metrics = len(all_metrics)
samples_per_mode = len(all_metrics) // num_iterations / len(modes_to_optimize)

print(
    f'samples per mode: {samples_per_mode}\n'
    f'number of iterations: {num_iterations}\n'
    f'number of metrics: {len(all_metrics)}\n'
)

ao.plot_metric_progress(
    all_metrics = all_metrics,
    modes_to_optimize = modes_to_optimize,
    num_iterations = num_iterations,
    zernike_mode_names = zernike_mode_names,
    save_dir_path = data_path,
    show_fig = True,
)

ao.plot_zernike_coeffs(
    coefficients_per_iteration=coeff_per_iteration,
    num_iterations=num_iterations,
    zernike_mode_names=zernike_mode_names,
    save_dir_path=data_path,
    show_fig=True,
    x_range=0.1
)
