from pathlib import Path
import matplotlib.pyplot as plt
from opm_v2.utils import sensorless_ao as ao
import numpy as np
from numpy.typing import NDArray
from typing import List, Optional
import zarr
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

showfig = False
root_dir = Path(
    "/home/steven/Documents/qi2lab/projects/local_working_files/OPM/opm_ao/ao_results/after_adding_IB_wAO_ao_results"
)


def load_optimization_results(results_path: Path):
    """Load optimization results from a Zarr store.
    
    NOTE: This method is modified for the depreciated saved results
          There are different keys!

    Parameters
    ----------
    results_path : Path
        Path to the Zarr directory containing the data.
    """
    # Open the Zarr store
    store = zarr.DirectoryStore(str(results_path))
    results = zarr.open(store)
    print(results.items())
    images_per_iteration = results["images_per_iteration"][:]
    images_per_mode = results["images_per_mode"][:]
    metrics_per_iteration = results["metrics_per_iteration"][:]
    metrics_per_mode = results["metrics_per_mode"][:]
    coefficients_per_iteration = results["coefficients_per_iteration"][:]

    modes_to_optimize = results["modes_to_optimize"][:]
    zernike_mode_names = [
        name.decode("utf-8") for name in results["zernike_mode_names"][:]
    ]
    
    ao_results = {
        "metrics_per_iteration":metrics_per_iteration,
        "images_per_iteration":images_per_iteration,
        "images_per_mode":images_per_mode,
        "modes_to_optimize":modes_to_optimize,
        "metrics_per_mode":metrics_per_mode,
        "coefficients_per_iteration":coefficients_per_iteration,
        "mode_names":zernike_mode_names,
    }
    ao_results.update(results.attrs)

    return ao_results


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


def plot_metric_progress(
    metrics_per_mode,
    modes_to_optimize: List[int],
    zernike_mode_names: List[str],
    grid_id,
    save_dir_path: Path = None,
    show_fig: Optional[bool] = False
) -> None:
    """Plot the metric magnitude throughout optimization

    Parameters
    ----------
    all_metrics : NDArray
    num_iterations : float
    modes_to_optimize : List[int]
    zernike_mode_names : List[str]
    save_dir_path : Path, optional
        Path to save figure, by default None
    show_fig : Optional[bool], optional
        whether to display figure, by default False
    """
    import matplotlib.pyplot as plt
        
    plt.rcParams.update(
        {
            "font.size": 14,        # base font size
            "axes.titlesize": 18,   # title size
            "axes.labelsize": 15,   # x/y label size
            "xtick.labelsize": 15,  # x-tick label size
            "ytick.labelsize": 15,  # y-tick label size
            "legend.fontsize": 14   # legend size
        }
    )
    num_modes = len(modes_to_optimize)
    modes_to_use_names = [zernike_mode_names[i] for i in modes_to_optimize]
    num_iterations = metrics_per_mode.shape[0]//num_modes
    metrics_per_mode = metrics_per_mode[1:]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and markers for each iteration
    colors = ["b", "g", "r", "c", "m"]
    markers = ["x", "o", "^", "s", "*"]

    # Loop over iterations and plot each series
    for ii in range(num_iterations):
        print(ii)
        print(num_modes)
        current_metrics = metrics_per_mode[
            int(ii*(num_modes)):int((num_modes)*(ii+1))
        ]
        ax.plot(
            np.linspace(0, num_modes-1, num_modes),
            current_metrics,
            color=colors[ii],
            label=f"iteration {ii}", 
            marker=markers[ii],
            linestyle="--", 
            linewidth=1
        )

    ax.set(
        title="Metric Progress per Iteration",
        ylabel="Metric",
        xticks=np.arange(len(modes_to_use_names))
    )
    ax.set_xticklabels(modes_to_use_names, rotation=60, ha="right", fontsize=16) 
    ax.legend()
    plt.tight_layout()
    
    if show_fig:
        plt.show()
    if save_dir_path:
        fig.savefig(save_dir_path / Path(f"ao_metrics_{grid_id}.png"))


# result = load_optimization_results(root_dir)

# images_per_mode = result["images_per_mode"]
# metrics_per_mode = result["metrics_per_mode"]
# zernike_mode_names = result['mode_names']
# modes_to_optimize = result['modes_to_optimize']
# save_dir_path = None
# showfig = True
# plot_metric_progress(
#     metrics_per_mode= metrics_per_mode,
#     modes_to_optimize = modes_to_optimize,
#     zernike_mode_names = zernike_mode_names,
#     save_dir_path = root_dir,
#     grid_id=0,
#     show_fig = True,
# )
# show_inter_plot(images_per_mode)


# process grid acquisitions, need to parse mulitle directories by default
grid_results = []
root_dir = Path("/home/steven/Documents/qi2lab/projects/local_working_files/OPM/opm_ao/ao_results/after_adding_IB_wAO_ao_results")
for _d in root_dir.iterdir():
    zarr_path = [p for p in _d.glob('*ao_results.zarr')]
    
    if len(zarr_path)==0:
        pass
    else:
        grid_results.append(load_optimization_results(zarr_path[0]))    

for ii, result in enumerate(grid_results):
    images_per_mode = result["images_per_mode"]
    metrics_per_mode = result["metrics_per_mode"]
    zernike_mode_names = result['mode_names']
    modes_to_optimize = result['modes_to_optimize']
    images_per_iteration = result['images_per_iteration']
    save_dir_path = None
    showfig = True
    show_inter_plot(images_per_iteration)
    
    
    
# Combine grid data to take place of iterations
# if 'grid' in _d.name:
#     optimal_coeffs = []
#     all_metrics = [] 
    
#     for ao_results in grid_results:

        
    #     ao.plot_zernike_coeffs(
    #         optimal_coeffs=optimal_coeffs,
    #         zernike_mode_names=zernike_mode_names,
    #         save_dir_path=root_dir,
    #         show_fig=True,
    #         x_range=0.1
    #     )

    #     optimal_coeffs.append(ao_results['optimal_coeffs'])
    #     all_metrics.append(ao_results['all_metrics'])
        
        
        
    # optimal_coeffs = np.asarray(optimal_coeffs).reshape([len(grid_results), len(zernike_mode_names)])
    # all_metrics = np.asarray(all_metrics)


    # num_iterations = optimal_coeffs.shape[1]
    
    # ao.plot_metric_progress(
    #     all_metrics = all_metrics,
    #     modes_to_optimize = modes_to_optimize,
    #     num_iterations = optimal_coeffs.shape[1],
    #     zernike_mode_names = zernike_mode_names,
    #     save_dir_path = root_dir,
    #     show_fig = True,
    # )
    # ao.plot_zernike_coeffs(
    #     optimal_coeffs=optimal_coeffs,
    #     zernike_mode_names=zernike_mode_names,
    #     save_dir_path=root_dir,
    #     show_fig=True,
    #     x_range=0.1
    # )
