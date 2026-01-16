from pathlib import Path
import matplotlib.pyplot as plt
from opm_v2.utils import sensorless_ao as ao
import numpy as np
import zarr


showfig = True
data_path = Path(
    r'/home/steven/Documents/qi2lab/projects/local_working_files/OPM/opm_ao/ao_results/20251222_125338_ao_optimizeNOW/ao_results.zarr'
    # "/home/steven/Documents/qi2lab/projects/local_working_files/OPM/opm_ao/ao_results/20251222_130948_ao_optimizeNOW/ao_results.zarr"
)

results = ao.load_optimization_results(data_path)

all_images = results['all_images']
all_metrics = results['all_metrics']
metrics_per_iteration = results['metrics_per_iteration']
images_per_iteration = results['images_per_iteration']

optimal_coeffs = results['optimal_coeffs']
modes_to_optimize = results['modes_to_optimize']
zernike_mode_names = results['mode_names']
modes_to_use_names = [zernike_mode_names[i] for i in modes_to_optimize]
num_iterations = results['num_iterations']
num_modes = len(modes_to_optimize)
num_metrics = len(all_metrics)
samples_per_mode = np.ceil((len(all_metrics)-1) // num_iterations / len(modes_to_optimize)).astype(int)


# Reshape into: (iterations, modes, samples)
# metrics = np.array(all_metrics[1:])

# metrics_by_iteration = np.zeros(
#     [int(num_iterations), int(num_modes), int(samples_per_mode)]
# )
# idx = 1
# for kk in range(num_iterations):
#     for jj in range(num_modes):
#         for ii in range(samples_per_mode):
#             metrics_by_iteration[kk, jj, ii] = all_metrics[idx]
#             idx += 1
            
            
# metrics_by_iteration = metrics.reshape(
#     num_iterations, num_modes, int(samples_per_mode)
# )

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
    show_fig = showfig,
)

ao.plot_zernike_coeffs(
    optimal_coeffs=optimal_coeffs,
    num_iterations=num_iterations,
    zernike_mode_names=zernike_mode_names,
    save_dir_path=data_path,
    show_fig=showfig,
    x_range=0.1
)



import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def show_inter_plot(images,
                    norm: bool = True):
    # Create a new figure for each directory
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)  # Adjust layout to fit slider

    # Display the first image
    img_display = ax.imshow(images[0, :, :]/np.max(images[0, :, :]), cmap="gray", aspect='equal')
    # Slider axis
    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])  # Positioning the slider
    slider = Slider(slider_ax, "Image", 0, len(images) - 1, valinit=0, valstep=1)

    # Function to update the displayed image
    def update(val):
        img_display.set_data(images[int(slider.val), :, :]/np.max(images[int(slider.val), :, :]))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

show_inter_plot(images_per_iteration)