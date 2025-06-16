"""Sensorless adaptive optics.

TO DO:
- Load interaction matrix from disk
- Set and get Zernike mode amplitudes from mirror
- Might need HASO functions to do this, since Zernike need to be composed given the pupil

2024/12 DPS initial work
"""
from pymmcore_plus import CMMCorePlus
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Sequence, List
from scipy.fftpack import dct
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from pathlib import Path
from tifffile import imwrite
import zarr
from time import sleep
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ

DEBUGGING = False
METRIC_PRECISION = 4

spherical_modes = [7,14,23]
spherical_modes_first =  [7,14,23,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
all_modes = [3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

mode_names = [
    "Vert. Tilt",
    "Horz. Tilt",
    "Defocus",
    "Vert. Asm.",
    "Oblq. Asm.",
    "Vert. Coma",
    "Horz. Coma",
    "3rd Spherical",
    "Vert. Tre.",
    "Horz. Tre.",
    "Vert. 5th Asm.",
    "Oblq. 5th Asm.",
    "Vert. 5th Coma",
    "Horz. 5th Coma",
    "5th Spherical",
    "Vert. Tetra.",
    "Oblq. Tetra.",
    "Vert. 7th Tre.",
    "Horz. 7th Tre.",
    "Vert. 7th Asm.",
    "Oblq. 7th Asm.",
    "Vert. 7th Coma",
    "Horz. 7th Coma",
    "7th Spherical",
    "Vert. Penta.",
    "Horz. Penta.",
    "Vert. 9th Tetra.",
    "Oblq. 9th Tetra.",
    "Vert. 9th Tre.",
    "Horz. 9th Tre.",
    "Vert. 9th Asm.",
    "Oblq. 9th Asm.",
]

#-------------------------------------------------#
# AO optimization
#-------------------------------------------------#

def run_ao_optimization(
    exposure_ms: float,
    channel_states: List[bool],
    metric_to_use: Optional[str] = "shannon_dct",
    daq_mode: Optional[str] = "projection",
    image_mirror_range_um: float = 100,
    psf_radius_px: Optional[float] = 2,
    num_iterations: Optional[int] = 3,
    num_mode_steps: Optional[int] = 3,
    init_delta_range: Optional[float] = 0.35,
    delta_range_alpha_per_iter: Optional[float] = 0.60,
    modes_to_optimize: Optional[List[int]] = all_modes,
    save_dir_path: Optional[Path] = None,
    verbose: Optional[bool] = True,
    ):
    
    #---------------------------------------------#
    # Create hardware controller instances
    #---------------------------------------------#
    opmNIDAQ_local = OPMNIDAQ.instance()
    aoMirror_local = AOMirror.instance()
    mmc = CMMCorePlus.instance()
    
    # Enforce camera exposure
    mmc.setProperty("OrcaFusionBT", "Exposure", float(exposure_ms))
    mmc.waitForDevice("OrcaFusionBT")
    
    #---------------------------------------------#
    # Setup Zernike modal coeff arrays
    #---------------------------------------------#
    # TODO: Do we start from system flat or current mirror state?
    # aoMirror_local.apply_system_flat_voltage()
    starting_modal_coeffs = aoMirror_local.current_coeffs.copy() # coeff before optimizationmode 
    active_zern_modes = starting_modal_coeffs.copy() # modified coeffs to be or are applied to mirror
    optimized_modal_coeffs = starting_modal_coeffs.copy() # mode coeffs after running all iterations

    if verbose:
        print(
            f'\n AO optimization starting mirror modal amplitudes:\n{aoMirror_local.current_zernikes}'
        )
        
    #----------------------------------------------
    # Setup for saving AO results
    #----------------------------------------------
    if save_dir_path:
        if verbose:
            print(f"Saving AO results at:\n  {save_dir_path}\n")
        all_images = []
        all_metrics = []
        all_mode_coeffs = []
        images_per_iteration = []
        metrics_per_iteration = [] 
        coeffs_per_iteration = []
    
    #---------------------------------------------#
    # setup the daq for selected mode
    #---------------------------------------------#
    if "projection" not in daq_mode:
        image_mirror_range_um = None    
    opmNIDAQ_local.stop_waveform_playback()
    opmNIDAQ_local.clear_tasks()
    opmNIDAQ_local.set_acquisition_params(
        scan_type=daq_mode,
        channel_states=channel_states,
        image_mirror_range_um=image_mirror_range_um,
        laser_blanking=True,
        exposure_ms=exposure_ms
    )
    opmNIDAQ_local.generate_waveforms()
    opmNIDAQ_local.program_daq_waveforms()
    opmNIDAQ_local.start_waveform_playback()
    
    #---------------------------------------------#
    # Start AO optimization
    #---------------------------------------------#   
    if verbose:
        print(f"\nStarting A.O. optimization using {metric_to_use} metric")
    
    # Snap an image and calculate the starting metric.
    starting_image = mmc.snap()
    
    if "DCT" in metric_to_use:
        starting_metric = metric_shannon_dct(
            image=starting_image,
            psf_radius_px=psf_radius_px,
            crop_size=None
            )
    # TODO: test other information based metrics  
    elif "fourier_ratio" in metric_to_use:
        starting_metric = metric_shannon_dct(
            image=starting_image,
            psf_radius_px=psf_radius_px,
            crop_size=None
            )
    elif "localize_gauss_2d" in metric_to_use:        
        starting_metric = metric_localize_gauss2d(
            image=starting_image
            )  
    else:
        print(f"Warning: AO metric '{metric_to_use}' not supported. Exiting function.")
        return  
    
    # update saved results
    if save_dir_path:
        all_images.append(starting_image.copy())
        all_metrics.append(starting_metric)
        all_mode_coeffs.append(starting_modal_coeffs.copy())
        
    # initialize delta range
    delta_range = init_delta_range
        
    # Start AO iterations 
    for k in range(num_iterations): 
        if k==0:       
            current_metric = starting_metric
            current_image = starting_image.copy()
            current_modal_coeffs = starting_modal_coeffs.copy()
            
        if save_dir_path:
            images_per_iteration.append(current_image)
            metrics_per_iteration.append(current_metric) 
            coeffs_per_iteration.append(current_modal_coeffs)

        # Iterate over modes to optimize
        for mode in modes_to_optimize:
            if verbose:
                print(
                    f'\nAO iteration: {k+1} / {num_iterations}',
                    f'\n    Perturbing mirror with Zernike mode: {mode+1}'
                    )
                
            metrics = []
            deltas = np.linspace(-delta_range, delta_range, num_mode_steps)
            for delta in deltas:
                # Create an array for applying deltas and sending to the mirror
                active_modal_coeffs = current_modal_coeffs.copy()
                active_modal_coeffs[mode] += delta
                
                # Apply Zernike pertubation mode to the mirror
                success = aoMirror_local.set_modal_coefficients(active_modal_coeffs)
                
                if not(success):
                    print('\n    ---- Setting mirror coefficients failed! ----')
                    metric = 0
                    image = np.zeros_like(starting_image)
                                            
                else:
                    # Acquire image and calculate metric
                    if not opmNIDAQ_local.running():
                        opmNIDAQ_local.start_waveform_playback()
                    image = mmc.snap()
                    
                    if "DCT" in metric_to_use:
                        metric = metric_shannon_dct(
                            image=image,
                            psf_radius_px=psf_radius_px,
                            crop_size=None
                            )  
                    elif "localize_gauss_2d" in metric_to_use:        
                        metric = metric_localize_gauss2d(
                            image=image
                            )

                    if metric==np.nan:
                        if verbose:
                            print('\n    ---- Metric failed == NAN ----')
                        success = False
                        metric = float(np.nan_to_num(metric))
                    
                    metric = np.round(metric, METRIC_PRECISION)
                    
                    if verbose:
                        print(f'      Delta={delta:.3f}, Metric = {metric:.6}, Success = {success}')

                    if DEBUGGING:
                        imwrite(Path(f"g:/ao/ao_{mode}_{delta}.tiff"),image)
                    
                metrics.append(metric)
                if save_dir_path:
                    all_images.append(image)
                    all_metrics.append(metric)
                    all_mode_coeffs.append(active_modal_coeffs)

            #---------------------------------------------#
            # Fit metrics to determine optimal metric
            #---------------------------------------------#   
            # Do not accept any changes if not success
            if success:
                try:                  
                    # Test if metric samples have a peak to fit
                    is_increasing = all(x < y for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:]))
                    is_decreasing = all(x > y for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:]))
                    if is_increasing or is_decreasing:
                        raise Exception(f'Test metrics are monotonic and linear: {metrics}')
                    
                    # elif a >=0:
                    #     raise Exception(f'Test metrics have a positive curvature: {metrics}')
                    
                    # compare to booth opt_delta
                    optimal_delta = -delta_range * (metrics[2] - metrics[0]) / (2*metrics[0] + 2*metrics[2] - 4*metrics[1])
                    
                    # Optimal metric is at the peak of quadratic 
                    # popt = quadratic_fit(deltas, metrics)
                    # a, b, c = popt
                    # optimal_delta = -b / (2 * a)
                    # if verbose:
                    #     print(
                    #         f'\n    ++++ our delta: {optimal_delta} ++++'
                    #         f'\n    ++++ Booth delta {booth_delta} ++++'
                    #     )
                    
                    # Reject metric if it is outside the test range.
                    if (optimal_delta>delta_range) or (optimal_delta<-delta_range):
                        raise Exception(f'Result outside of range: opt_delta = {optimal_delta:.4f}')
                    
                    if verbose:
                        print(f'\n    Metric maximum at delta = {optimal_delta:4f}')
                                
                except Exception as e:
                    optimal_delta = 0
                    if verbose:
                        print(f'\n    ---- Exception in fit occurred, no changes accepted! ----\n      e:{e}')
            else:
                optimal_delta = 0
                if verbose:
                    print('\n    ---- Error occurred in metric, no changes accepted! ----')
           
            #---------------------------------------------#
            # Apply the kept optimized mirror modal coeffs
            #---------------------------------------------#  
            # apply the new modal coefficient to the mirror and update current coefficients
            current_modal_coeffs[mode] = current_modal_coeffs[mode] + optimal_delta
            _ = aoMirror_local.set_modal_coefficients(current_modal_coeffs)
            
            """Loop back to top and do the next mode until all modes are done"""
        
        #---------------------------------------------#
        # After all modes, reduce the delta range for the next iteration
        #---------------------------------------------# 
        if num_iterations > 1:
            delta_range *= delta_range_alpha_per_iter
            if verbose:
                print(f"   Reduced sweep range to {delta_range:.4f}")
            
        """Loop back to top and do the next iteration"""
        
    #---------------------------------------------#
    # After all the iterations the mirror state will be optimized
    #---------------------------------------------# 
    aoMirror_local.update_optimized_array()
    if verbose:
        print(
            f"\n++++ Starting Zernike mode amplitude: ++++ \n{starting_modal_coeffs}",
            f"\n++++ Final optimized Zernike mode amplitude: ++++ \n{current_modal_coeffs}"
            )
    
    # Stop the DAQ programming
    opmNIDAQ_local.stop_waveform_playback()
    
    if save_dir_path:
        all_images = np.asarray(all_images)
        all_metrics = np.asarray(all_metrics)
        all_mode_coeffs = np.asarray(all_mode_coeffs)
        images_per_iteration = np.asarray(images_per_iteration)
        metrics_per_iteration = np.asarray(metrics_per_iteration)
        coeffs_per_iteration = np.asarray(coeffs_per_iteration)
        
        # save and produce
        save_optimization_results(
            all_images,
            all_metrics,
            images_per_iteration,
            metrics_per_iteration,
            coeffs_per_iteration,
            modes_to_optimize,
            save_dir_path
        )        
        # plot_zernike_coeffs(
        #     coeffs_per_iteration,
        #     mode_names,
        #     save_dir_path=save_dir_path
        # )        
        # plot_metric_progress(
        #     all_metrics,
        #     num_iterations,
        #     modes_to_optimize,
        #     mode_names,
        #     save_dir_path
        # )

#-------------------------------------------------#
# Plotting functions
#-------------------------------------------------#

def plot_zernike_coeffs(
    optimal_coefficients: ArrayLike,
    zernike_mode_names: ArrayLike,
    save_dir_path: Optional[Path] = None,
    show_fig: Optional[bool] = False
):
    """_summary_

    Parameters
    ----------
    optimal_coefficients : ArrayLike
        _description_
    save_dir_path : Path
        _description_
    showfig : bool
        _description_
    """
    import matplotlib.pyplot as plt
    import matplotlib
    if not show_fig:
        matplotlib.use('Agg')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Define colors and markers for each iteration
    colors = ['b', 'g', 'r', 'c', 'm']  
    markers = ['x', 'o', '^', 's', '*']  

    # populate plots
    for i in range(len(zernike_mode_names)):
        for j in range(optimal_coefficients.shape[0]):
            marker_style = markers[j % len(markers)]
            ax.scatter(
                optimal_coefficients[j, i], i, 
                color=colors[j % len(colors)],
                s=125, 
                marker=marker_style
            )  
        ax.axhline(y=i, linestyle="--", linewidth=1, color='k')
        
    # Plot a vertical line at 0 for reference
    ax.axvline(0, color='k', linestyle='-', linewidth=1)

    # Customize the plot
    ax.set_yticks(np.arange(len(zernike_mode_names)))
    ax.set_yticklabels(zernike_mode_names)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Zernike mode coefficients at each iteration")
    ax.set_xlim(-0.350, 0.350)

    # Add a legend for time points
    ax.legend(
        [f'Iteration: {i+1}' for i in range(optimal_coefficients.shape[0])], 
        loc='upper right'
    )

    # Remove grid lines
    ax.grid(False)

    plt.tight_layout()
    if show_fig:
        plt.show()
    if save_dir_path:
        fig.savefig(save_dir_path / Path("ao_zernike_coeffs.png"))

def plot_metric_progress(
    all_metrics: ArrayLike,
    num_iterations: ArrayLike,
    modes_to_optimize: List[int],
    zernike_mode_names: List[str],
    save_dir_path: Optional[Path] = None,
    show_fig: Optional[bool] = False
):
    """_summary_

    Parameters
    ----------
    metrics_per_iteration : ArrayLike
        N_iter x N_modes array of the metric value per mode
    modes_to_optmize : List[int]
        _description_
    zernike_mode_names : List[str]
        _description_
    save_dir_path : Optional[Path], optional
        _description_, by default None
    show_fig : Optional[bool], optional
        _description_, by default False
    """   
    import matplotlib.pyplot as plt
    import matplotlib
    if not show_fig:
        matplotlib.use('Agg')
    
    metrics_per_mode = np.reshape(
        all_metrics[1::1], # Only show the  
        (num_iterations, len(modes_to_optimize))
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and markers for each iteration
    colors = ['b', 'g', 'r', 'c', 'm']
    markers = ['x', 'o', '^', 's', '*']

    # Loop over iterations and plot each series
    for ii, series in enumerate(metrics_per_mode):
        ax.plot(
            series,
            color=colors[ii],
            label=f"iteration {ii}", 
            marker=markers[ii],
            linestyle="--", 
            linewidth=1
        )

    # Set the x-axis to correspond to the modes_to_optimize
    mode_labels = [zernike_mode_names[i] for i in modes_to_optimize]
    ax.set_xticks(np.arange(len(mode_labels))) 
    ax.set_xticklabels(mode_labels, rotation=60, ha="right", fontsize=16) 

    # Customize the plot
    ax.set_ylabel("Metric", fontsize=16)
    ax.set_title("Optimal Metric Progress per Iteration", fontsize=18)

    ax.legend(fontsize=15)
    
    plt.tight_layout()
    
    if show_fig:
        plt.show()
    if save_dir_path:
        fig.savefig(save_dir_path / Path("ao_metrics.png"))

def plot_2d_localization_fit_summary(
    fit_results,
    img,
    coords_2d,
    save_dir_path: Path = None,
    showfig: bool = False
):
    """_summary_

    Parameters
    ----------
    fit_results : _type_
        _description_
    img : _type_
        _description_
    coords_2d : _type_
        _description_
    save_dir_path : Path, optional
        _description_, by default None
    showfig : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    # imports
    from localize_psf.fit_psf import sxy2na
    from localize_psf.localize import plot_bead_locations
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    to_keep = fit_results["to_keep"]
    sxy = fit_results["fit_params"][to_keep, 4]
    amp = fit_results["fit_params"][to_keep, 0]
    bg = fit_results["fit_params"][to_keep, 6]
    centers = fit_results["fit_params"][to_keep][:, (3, 2, 1)]
    cx = centers[:,2]
    cy = centers[:,1]

    width_ratios=[1,0.7,0.1,0.1,0.1]
    height_ratios=[1,0.1,0.5,0.5,0.5,0.5,0.5]
    figh_sum = plt.figure(figsize=(10,8))
    grid_sum = figh_sum.add_gridspec(
        nrows=len(height_ratios),
        ncols=len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0.2,
        wspace=0.3
    )

    ax_proj_sxy = figh_sum.add_subplot(grid_sum[0,:2])
    ax_cmap_i_sxy = figh_sum.add_subplot(grid_sum[0,2])
    ax_cmap_sxy = figh_sum.add_subplot(grid_sum[0,4])
    figh_sum = plot_bead_locations(
        img,
        centers,
        weights=[fit_results["fit_params"][to_keep, 4]],
        color_lists=["autumn"],
        color_limits=[[0.05,0.5]],
        cbar_labels=[r"$\sigma_{xy}$"],
        title="Max intensity projection with Sxy",
        coords=coords_2d,
        gamma=0.5,
        axes=[ax_proj_sxy, ax_cmap_i_sxy, ax_cmap_sxy]
    )
    ax_proj_sxy.set_title(
        f"Sxy: mean={np.mean(sxy):.3f}, median={np.median(sxy):.3f}; NA (median):{sxy2na(0.473, np.median(sxy)):.2f}"
    )

    # Create axes for plotting x, y specific results
    ax_sxy_cx = figh_sum.add_subplot(grid_sum[3,0])
    ax_sxy_cy = figh_sum.add_subplot(
        grid_sum[3,1:],
        sharey=ax_sxy_cx
    )
    ax_amp_cx = figh_sum.add_subplot(grid_sum[4,0],sharex=ax_sxy_cx)
    ax_amp_cy = figh_sum.add_subplot(
        grid_sum[4,1:],
        sharey=ax_amp_cx,
        sharex=ax_sxy_cy
    )
    ax_bg_cx = figh_sum.add_subplot(grid_sum[5,0],sharex=ax_sxy_cx)
    ax_bg_cy = figh_sum.add_subplot(
        grid_sum[5,1:],
        sharey=ax_bg_cx,
        sharex=ax_sxy_cy
    )
    ax_sxy_cx.set_ylabel(r"$\sigma_{xy}$ ($\mu m$)")
    ax_amp_cx.set_ylabel("amplitude")
    ax_bg_cx.set_ylabel("background")
    ax_bg_cx.set_xlabel(r"$C_x$ $\mu m$")
    ax_bg_cy.set_xlabel(r"$C_y$ $\mu m$")
    for ax in [ax_sxy_cy,ax_amp_cy,ax_bg_cy]:
        ax.tick_params(labelleft=False)
    for ax in [ax_sxy_cx,ax_sxy_cy,ax_amp_cx,ax_amp_cy]:
        ax.tick_params(labelbottom=False)

    # Set limits for visualizing sz
    if max(amp)>65000:
        amp_max = 15000
    else:
        amp_max = np.max(amp)*1.1
    ax_sxy_cx.set_ylim(0,1.0)
    ax_sxy_cy.set_ylim(0,1.0)
    ax_amp_cx.set_ylim(0, amp_max)
    ax_amp_cy.set_ylim(0, amp_max)
    ax_bg_cx.set_ylim(0, amp_max)
    ax_bg_cy.set_ylim(0, amp_max)
    ax_sxy_cx.set_xlim(0,img.shape[1]*0.115)
    ax_sxy_cy.set_xlim(0,img.shape[0]*0.115)
    ax_amp_cx.set_xlim(0,img.shape[1]*0.115)
    ax_amp_cy.set_xlim(0,img.shape[0]*0.115)
    ax_bg_cx.set_xlim(0,img.shape[1]*0.115)
    ax_bg_cy.set_xlim(0,img.shape[0]*0.115)
    # Plot directional results
    ax_sxy_cx.plot(cx, sxy, c="b", marker=".", markersize=3, linestyle="none")
    ax_sxy_cy.plot(cy, sxy, c="b", marker=".", markersize=3, linestyle="none")
    ax_amp_cx.plot(cx, amp, c="b", marker=".", markersize=3, linestyle="none")
    ax_amp_cy.plot(cy, amp, c="b", marker=".", markersize=3, linestyle="none")
    ax_bg_cx.plot(cx, bg, c="b", marker=".", markersize=3, linestyle="none")
    ax_bg_cy.plot(cy, bg, c="b", marker=".", markersize=3, linestyle="none")

    if showfig:
        figh_sum.show()
        plt.show()
    else:
        plt.close(figh_sum)
    if save_dir_path:
        figh_sum.savefig(
            save_dir_path / Path("ao_localization_results.png"),
            dpi=150
        )
    
    figh_sum = None
    del figh_sum
    return None

#-------------------------------------------------#
# Functions for preparing data
#-------------------------------------------------#

def get_image_center(image: ArrayLike, threshold: float) -> Tuple[int, int]:
    """
    Calculate the center of an image using a thresh-holded binary mask.

    Parameters
    ----------
    image : ArrayLike
        2D image array.
    threshold : float
        Intensity threshold for binarization.

    Returns
    -------
    center : Tuple[int, int]
        Estimated center coordinates (x, y).
    """
    try:
        binary_image = image > threshold
        center = center_of_mass(binary_image)
        center = tuple(map(int, center))
    except Exception:
        center = (image.shape[1]//2, image.shape[0]//2)
    return center

def get_cropped_image(
    image: ArrayLike,
    crop_size: int,
    center: Tuple[int, int]
) -> ArrayLike:
    """
    Extract a square region from an image centered at a given point.

    Parameters
    ----------
    image : ArrayLike
        Input 2D or 3D image.
    crop_size : int
        Half-width of the cropping region.
    center : Tuple[int, int]
        Center coordinates (x, y) of the crop.

    Returns
    -------
    cropped_image : ArrayLike
        Cropped region from the input image.
    """
    if len(image.shape) == 3:
        x_min, x_max = max(center[0] - crop_size, 0), min(center[0] + crop_size, image.shape[1])
        y_min, y_max = max(center[1] - crop_size, 0), min(center[1] + crop_size, image.shape[2])
        cropped_image = image[:, x_min:x_max, y_min:y_max]
    else:
        x_min, x_max = max(center[0] - crop_size, 0), min(center[0] + crop_size, image.shape[0])
        y_min, y_max = max(center[1] - crop_size, 0), min(center[1] + crop_size, image.shape[1])
        cropped_image = image[x_min:x_max, y_min:y_max]
    return cropped_image

#-------------------------------------------------#
# Functions for fitting and calculations
#-------------------------------------------------#

def gauss2d(
    coords_xy: ArrayLike,
    amplitude: float,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
    offset: float
) -> ArrayLike:
    """
    Generates a 2D Gaussian function for curve fitting.

    Parameters
    ----------
    coords_xy : ArrayLike
        Meshgrid coordinates (x, y).
    amplitude : float
        Peak intensity of the Gaussian.
    center_x : float
        X-coordinate of the Gaussian center.
    center_y : float
        Y-coordinate of the Gaussian center.
    sigma_x : float
        Standard deviation along the x-axis.
    sigma_y : float
        Standard deviation along the y-axis.
    offset : float
        Background offset intensity.

    Returns
    -------
    raveled_gauss2d : ArrayLike
        Flattened 2D Gaussian function values.
    """
    x, y = coords_xy
    raveled_gauss2d = (
        offset +
        amplitude * np.exp(
            -(((x - center_x)**2 / (2 * sigma_x**2)) + ((y - center_y)**2 / (2 * sigma_y**2)))
        )
    ).ravel()

    return raveled_gauss2d

def otf_radius(img: ArrayLike, psf_radius_px: float) -> int:
    """
    Computes the optical transfer function (OTF) cutoff frequency.

    Parameters
    ----------
    img : ArrayLike
        2D image.
    psf_radius_px : float
        Estimated point spread function (PSF) radius in pixels.

    Returns
    -------
    cutoff : int
        OTF cutoff frequency in pixels.
    """
    w = min(img.shape)
    psf_radius_px = max(1, np.ceil(psf_radius_px))  # clip all PSF radii below 1 px to 1.
    cutoff = np.ceil(w / (2 * psf_radius_px)).astype(int)

    return cutoff

def normL2(x: ArrayLike) -> float:
    """
    Computes the L2 norm of an n-dimensional array.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    l2norm : float
        L2 norm of the array.
    """
    l2norm = np.sqrt(np.sum(x.flatten() ** 2))

    return l2norm

def shannon(spectrum_2d: ArrayLike, otf_radius: int = 100) -> float:
    """
    Computes the Shannon entropy of an image spectrum within a given OTF radius.

    Parameters
    ----------
    spectrum_2d : ArrayLike
        2D spectrum of an image (e.g., from DCT or FFT).
    otf_radius : int, optional
        OTF support radius in pixels (default is 100).

    Returns
    -------
    entropy : float
        Shannon entropy of the spectrum.
    """
    h, w = spectrum_2d.shape
    y, x = np.ogrid[:h, :w]

    # Circular mask centered at (0,0) for DCT
    support = (x**2 + y**2) < otf_radius**2

    spectrum_values = np.abs(spectrum_2d[support])
    total_energy = np.sum(spectrum_values)

    if total_energy == 0:
        return 0  # Avoid division by zero

    probabilities = spectrum_values / total_energy
    entropy = -np.sum(probabilities * np.log2(probabilities, where=(probabilities > 0)))
    metric = np.log10(entropy)
    return metric

def dct_2d(image: ArrayLike, cutoff: int = 100) -> ArrayLike:
    """
    Computes the 2D discrete cosine transform (DCT) of an image with a cutoff.

    Parameters
    ----------
    image : ArrayLike
        2D image array.
    cutoff : int, optional
        OTF radius cutoff in pixels (default is 100).

    Returns
    -------
    dct_2d : ArrayLike
        Transformed image using DCT.
    """
    dct_2d = dct(dct(image.astype(np.float32), axis=0, norm='ortho'), axis=1, norm='ortho')

    return dct_2d

def quadratic(x: float, a: float, b: float, c: float) -> ArrayLike:
    """
    Quadratic function evaluation at x.

    Parameters
    ----------
    x : float
        Point to evaluate.
    a : float
        x^2 coefficient.
    b : float
        x coefficient.
    c : float
        Offset.

    Returns
    -------
    value : float
        a * x^2 + b * x + c
    """
    return a * x**2 + b * x + c

def quadratic_fit(x: ArrayLike, y: ArrayLike) -> Sequence[float]:
    """
    Quadratic function for curve fitting.

    Parameters
    ----------
    x : ArrayLike
        1D x-axis data.
    y : ArrayLike
        1D y-axis data.

    Returns
    -------
    coeffs : Sequence[float]
        Fitting parameters.
    """
    A = np.vstack([x**2, x, np.ones_like(x)]).T
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]

    return coeffs

#-------------------------------------------------#
# Localization methods to generate ROIs for fitting
#-------------------------------------------------#

def localize_2d_img(
    img, 
    dxy,
    localize_psf_filters = {
        "threshold":3000,
        "amp_bounds":(1000, 30000),
        "sxy_bounds":(0.100, 1.0)
    },
    save_dir_path: Path = None,
    label: str = "", 
    showfig: bool = False,
    verbose: bool = False
):
    """_summary_

    Parameters
    ----------
    img : _type_
        _description_
    dxy : _type_
        _description_
    localize_psf_filters : dict, optional
        _description_, by default { "threshold":3000, "amp_bounds":(1000, 30000), "sxy_bounds":(0.100, 1.0) }
    save_dir_path : Path, optional
        _description_, by default None
    label : str, optional
        _description_, by default ""
    showfig : bool, optional
        _description_, by default False
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    from localize_psf.fit_psf import gaussian3d_psf_model
    from localize_psf.localize import (
        localize_beads_generic,
        get_param_filter,
        get_coords
        )
    
    # Define fitting model and coordinates
    model = gaussian3d_psf_model() 
    coords_3d = get_coords((1,)+img.shape, (1, dxy, dxy))
    coords_2d = get_coords(img.shape, (dxy, dxy))
                           
    # Set fit bounds and parameter filters
    threshold = localize_psf_filters["threshold"]
    amp_bounds = localize_psf_filters["amp_bounds"]
    sxy_bounds = localize_psf_filters["sxy_bounds"]
    fit_dist_max_err = (0, dxy*2) 
    fit_roi_size = (1, dxy*9, dxy*9)
    min_spot_sep = (0, dxy*5)
    dist_boundary_min = (0, 1.0)
        
    param_filter = get_param_filter(
        coords_3d,
        fit_dist_max_err=fit_dist_max_err,
        min_spot_sep=min_spot_sep,
        amp_bounds=amp_bounds,
        dist_boundary_min=dist_boundary_min,
        sigma_bounds=((0,sxy_bounds[0]),(1,sxy_bounds[1]))
        )
        
    # Run localization function
    _, r, _ = localize_beads_generic(
        img,
        (1, dxy, dxy),
        threshold=threshold,
        roi_size=fit_roi_size,
        filter_sigma_small=None,
        filter_sigma_large=None,
        min_spot_sep=min_spot_sep,
        model=model,
        filter=param_filter,
        max_nfit_iterations=100,
        use_gpu_fit=False,
        use_gpu_filter=False,
        return_filtered_images=False,
        fit_filtered_images=False,
        verbose=verbose
        )
    
    if save_dir_path:
        plot_2d_localization_fit_summary(
            r, 
            img,
            coords_2d, 
            save_dir_path / Path(f"localize_psf_summary_{label}.png"),
            showfig
            )
        
    return r

#-------------------------------------------------#
# Functions to calculate image metrics
#-------------------------------------------------#

def metric_brightness(
    image: ArrayLike,
    crop_size: Optional[int] = None,
    threshold: Optional[float] = 100,
    image_center: Optional[int] = None,
    return_image: Optional[bool] = False
) -> float:
    """
    Compute weighted metric for 2D Gaussian.

    Parameters
    ----------
    image : ArrayLike
        2D image.
    threshold : float, optional
        Initial threshold to find spot (default is 100).
    crop_size_px : int, optional
        Crop size in pixels, one side (default is 20).
    image_center : Optional[int], optional
        Center of the image to crop (default is None).
    return_image : Optional[bool], optional
        Whether to return the cropped image (default is False).

    Returns
    -------
    weighted_metric : float
        Weighted metric value.
    """
    if crop_size:
        if image_center is None:
            center = get_image_center(image, threshold)
        else:
            center = image_center
        image = get_cropped_image(image, crop_size, center)

    if len(image.shape) == 3:
        image = np.max(image, axis=0)

    image_perc = np.percentile(image, 90)
    max_pixels = image[image >= image_perc]

    if return_image:
        return np.mean(max_pixels), image
    else:
        return np.mean(max_pixels)

def metric_shannon_dct(
    image: ArrayLike, 
    psf_radius_px: float = 3,
    crop_size: Optional[int] = None,
    threshold: Optional[float] = None,
    image_center: Optional[int] = None,
    return_image: Optional[bool] = False
) -> float:
    """Compute the Shannon entropy metric using DCT.

    Parameters
    ----------
    image : ArrayLike
        2D image.
    psf_radius_px : float, optional
        Estimated point spread function (PSF) radius in pixels (default: 3).
    crop_size : Optional[int], optional
        Crop size for image (default: 501).
    threshold : Optional[float], optional
        Intensity threshold to find the center (default: 100).
    image_center : Optional[int], optional
        Custom image center (default: None).
    return_image : Optional[bool], optional
        Whether to return the image along with the metric (default: False).
    
    Returns
    -------
    entropy_metric : float
        Shannon entropy metric.
    """
    # Crop image if necessary
    if not crop_size:
        crop_size = min(image.shape)-1
        
    if image_center is None:
        center = get_image_center(image, threshold)  # Ensure this function is defined
    else:
        center = image_center
        # Crop image (ensure get_cropped_image is correctly implemented)
        image = get_cropped_image(image, crop_size, center)
    
    # Compute the cutoff frequency based on OTF radius
    cutoff = otf_radius(image, psf_radius_px)

    # Compute DCT
    dct_result = dct_2d(image)

    # Compute Shannon entropy within the cutoff radius
    shannon_dct = shannon(dct_result, cutoff)

    if return_image:
        return shannon_dct, image
    else:
        return shannon_dct

def metric_gauss2d(
    image: ArrayLike,
    crop_size: Optional[int] = None,
    threshold: Optional[float] = 100,
    image_center: Optional[int] = None,
    return_image: Optional[bool]= False
) -> float:
    """Compute weighted metric for 2D gaussian.

    Parameters
    ----------
    image : ArrayLike
        2D image.
    threshold : float, optional
        Initial threshold to find spot (default is 100).
    crop_size_px : int, optional
        Crop size in pixels, one side (default is 20).
    image_center : Optional[int], optional
        Center of the image to crop (default is None).
    return_image : Optional[bool], optional
        Whether to return the cropped image (default is False).

    Returns
    -------
    weighted_metric : float
        Weighted metric value.
    """
    # Optionally crop the image
    if crop_size:    
        if image_center is None:
            center = get_image_center(image, threshold)
        else:
            center = image_center
        # crop image
        image = get_cropped_image(image, crop_size, center)
        
    # normalize image 0-1
    image = image / np.max(image)
    image = image.astype(np.float32)
    
    # create coord. grid for fitting 
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)
    
    # fitting assumes a single bead in FOV....
    initial_guess = (image.max(), image.shape[1] // 2, 
                     image.shape[0] // 2, 5, 5, image.min())
    fit_bounds = [[0,0,0,1.0,1.0,0],
                  [1.5,image.shape[1],image.shape[0],100,100,5000]]
    try:
        popt, pcov = curve_fit(gauss2d, (x, y), image.ravel(), 
                               p0=initial_guess,
                               bounds=fit_bounds,
                               maxfev=1000)
        
        amplitude, center_x, center_y, sigma_x, sigma_y, offset = popt
        weighted_metric = ((1 - np.abs((sigma_x-sigma_y) / (sigma_x+sigma_y))) 
                           + (1 / (sigma_x+sigma_y)) 
                           + np.exp(-1 * (sigma_x+sigma_y-4)**2))
        
        if (weighted_metric <= 0) or (weighted_metric > 100):
            weighted_metric = 1e-12 
    except Exception:
        weighted_metric = 1e-12
        
        
    if return_image:
        return weighted_metric, image
    else:
        return weighted_metric

def metric_localize_gauss2d(image: ArrayLike) -> float:
    """_summary_

    Parameters
    ----------
    image : ArrayLike
        _description_

    Returns
    -------
    float
        _description_
    """
    try:
        fit_results = localize_2d_img(
            image, 
            0.115,
            {"threshold":1000,
            "amp_bounds":(500, 30000),
            "sxy_bounds":(0.100, 1.0)
            },
            save_dir_path = None,
            label = "", 
            showfig = False,
            verbose = False
            )
        
        to_keep = fit_results["to_keep"]
        sxy = fit_results["fit_params"][to_keep, 4]
        metric = 1 / np.median(sxy)
    except Exception as e:
        print(f"2d localization and fit exceptions: {e}")
        metric = 0
        
    return metric

#-------------------------------------------------#
# Helper function for generating grid
#-------------------------------------------------#

def run_ao_grid_mapping(
    stage_positions: List,
    ao_dict: dict,
    num_tile_positions: int = 1,
    num_scan_positions: int = 1,
    save_dir_path: Path = None,
    verbose: bool = False,
) -> bool:
    """_summary_

    Parameters
    ----------
    stage_positions : list
        Experimental stage positions. Optimized for stage scan acquisitions
    ao_dict : dict
        A dictionary containing AO optimization parameters, including:
        - "image_mirror_range_um"
        - "exposure_ms"
        - "channel_states"
        - "metric"
        - "modal_alpha"
        - "iterations"
    save_dir_path : Path, optional
        Path to save AO optimization data. Default is None.
    verbose : bool, optional
        If True, prints additional debugging information. Default is False.

    Returns
    -------
    np.ndarray
        _description_
    """
    aoMirror_local = AOMirror.instance()
    mmc = CMMCorePlus.instance()
    
    stage_positions_array = np.array(
        [
            (pos["z"], pos["y"], pos["x"]) for pos in stage_positions
        ]
    )
    # Extract unique positions along each axis
    tile_axis_positions = np.unique(stage_positions_array[:, 1])
    scan_axis_positions = np.unique(stage_positions_array[:, 2])

    stage_z_positions = np.unique(
        stage_positions_array[stage_positions_array[:, 2] == scan_axis_positions[0]][:, 0]
    )
    num_z_positions = stage_z_positions.shape[0]

    if num_scan_positions==1:
        if len(scan_axis_positions)==1:
            ao_scan_axis_positions = np.asarray(scan_axis_positions+100)
        else:
            ao_scan_axis_positions = np.asarray([np.mean(scan_axis_positions)])
    elif len(scan_axis_positions)==1:
        ao_scan_length = num_scan_positions * 100
        scan_axis_min = scan_axis_positions[0]
        scan_axis_max = scan_axis_min + ao_scan_length
        ao_scan_axis_positions = np.linspace(
            scan_axis_min,
            scan_axis_max,
            num_scan_positions+2,
            endpoint=True
        )[1:-1]
    else:
        if num_scan_positions>len(scan_axis_positions)+1:
            num_scan_positions=len(scan_axis_positions)+1
        scan_axis_min = scan_axis_positions[0]
        scan_axis_max = scan_axis_positions[-1]
        ao_scan_axis_positions = np.linspace(
            scan_axis_min,
            scan_axis_max,
            num_scan_positions+2,
            endpoint=True
        )[1:-1]
        
    if num_tile_positions==1:
        ao_tile_axis_positions = np.asarray([np.mean(tile_axis_positions)])
    elif len(tile_axis_positions)==1:
        ao_tile_axis_positions = tile_axis_positions
        num_tile_positions = 1 
    else:
        if num_tile_positions>len(tile_axis_positions)+1:
            num_tile_positions=len(tile_axis_positions)+1
        tile_axis_min = tile_axis_positions[0]
        tile_axis_max = tile_axis_positions[-1]
        ao_tile_axis_positions = np.linspace(
            tile_axis_min, 
            tile_axis_max,
            num_tile_positions+2, 
            endpoint=True
        )[1:-1]

    # compile AO stage positions to visit, visit XY positions before stepping in Z
    ao_stage_positions = []
    starting_mirror_positions = aoMirror_local.current_voltage.copy()
    for z_idx in range(num_z_positions):
        for tile_idx in range(num_tile_positions):
            for scan_idx in range(num_scan_positions):
                scan_pos_filter = np.ceil(ao_scan_axis_positions[scan_idx] - stage_positions_array[:, 2])==1
                
                if not any(scan_pos_filter):
                    z_tile_positions = stage_z_positions
                else:
                    z_tile_positions = np.unique(
                        stage_positions_array[scan_pos_filter][:, 0]
                    )
                ao_stage_positions.append(
                    {
                        "z": np.round(z_tile_positions[z_idx],2),
                        "y": np.round(ao_tile_axis_positions[tile_idx],2),
                        "x": np.round(ao_scan_axis_positions[scan_idx],2)
                    }
                )

    # Save AO optimization results here
    num_ao_pos = len(ao_stage_positions)
    ao_grid_wfc_coeffs = np.zeros((num_ao_pos, aoMirror_local.positions_modal_array.shape[1]))
    ao_grid_wfc_positions = np.zeros((num_ao_pos, aoMirror_local.positions_voltage_array.shape[1]))
    
    # Run AO optimization for each stage position
    if verbose:
        print(f"\nGenerating AO map, number positions = {num_ao_pos}:")
    
    # Visit AO stage positions and measure best WFC positions
    for ao_pos_idx in range(num_ao_pos):
        if verbose:
            print(f"\nMoving stage to: {ao_stage_positions[ao_pos_idx]}")
            
        target_x = ao_stage_positions[ao_pos_idx]["x"]
        target_y = ao_stage_positions[ao_pos_idx]["y"]
        target_z = ao_stage_positions[ao_pos_idx]["z"]
        
        mmc.setPosition(np.round(target_z,2))
        mmc.waitForDevice(mmc.getFocusDevice())
        
        mmc.setXYPosition(
            np.round(float(target_x),2),
            np.round(float(target_y),2)
        )
        current_x, current_y = mmc.getXYPosition()
        while not(np.isclose(current_x, target_x, 0., 1.0)) or not(np.isclose(current_y, target_y, 0., 1.0)):
            current_x, current_y = mmc.getXYPosition()
            sleep(.5)
        
        current_save_dir = save_dir_path / Path(f"grid_pos_{int(ao_pos_idx)}")
        current_save_dir.mkdir(exist_ok=True)
        run_ao_optimization(
            metric_to_use=ao_dict["metric"],
            daq_mode=ao_dict["daq_mode"],
            image_mirror_range_um=ao_dict["image_mirror_range_um"],
            exposure_ms=ao_dict["exposure_ms"],
            channel_states=ao_dict["channel_states"],
            num_iterations=ao_dict["iterations"],
            init_delta_range=ao_dict["modal_delta"],
            delta_range_alpha_per_iter=ao_dict["modal_alpha"],
            save_dir_path=current_save_dir,
            verbose=verbose
        )
        ao_grid_wfc_coeffs[ao_pos_idx] = aoMirror_local.current_coeffs.copy()
        ao_grid_wfc_positions[ao_pos_idx] = aoMirror_local.current_voltage.copy()

    if verbose:
        print(
            "Optimization complete!",
            "\n Mapping AO grid positions to experimental stage positions. .  ."
        )
        
    # Map ao_grid_wfc_coeffs to experiment stage positions.
    position_wfc_coeffs = np.zeros(aoMirror_local.positions_modal_array.shape)
    position_wfc_positions = np.zeros(aoMirror_local.positions_voltage_array.shape)
        
    # Convert AO positions and stage positions into structured arrays for efficient lookup
    ao_stage_positions_array = np.array([(pos["z"], pos["y"], pos["x"]) for pos in ao_stage_positions])
    stage_positions_array = np.array([(pos["z"], pos["y"], pos["x"]) for pos in stage_positions])

    for pos_idx, (stage_z, stage_y, stage_x) in enumerate(stage_positions_array):
        # Get matching target ao positions
        target_z = ao_stage_positions_array[:,0][
            int(np.argmin(np.abs(stage_z - ao_stage_positions_array[:, 0])))
            ]
        target_y = ao_stage_positions_array[:,1][
            int(np.argmin(np.abs(stage_y - ao_stage_positions_array[:, 1])))
            ]
        target_x = ao_stage_positions_array[:,2][
            int(np.argmin(np.abs(stage_x - ao_stage_positions_array[:, 2])))
            ]
        
        # Find AO positions with matching z and y
        candidates = ao_stage_positions_array[
            (ao_stage_positions_array[:, 0] == target_z) &
            (ao_stage_positions_array[:, 1] == target_y) &
            (ao_stage_positions_array[:, 2] >= target_x)  # AO x must be >= stage x
        ]
        
        # Compute distances
        distances = np.linalg.norm(candidates - [target_z, stage_y, stage_x], axis=1)
        best_candidate_idx = np.argmin(distances)
        ao_grid_idx = np.where(
            (ao_stage_positions_array[:, 0] == candidates[best_candidate_idx][0]) &
            (ao_stage_positions_array[:, 1] == candidates[best_candidate_idx][1]) &
            (ao_stage_positions_array[:, 2] == candidates[best_candidate_idx][2])
        )[0][0]

        
        # Assign AO data
        position_wfc_positions[pos_idx] = ao_grid_wfc_positions[ao_grid_idx]
        position_wfc_coeffs[pos_idx] = ao_grid_wfc_coeffs[ao_grid_idx]

        if verbose:
            print(
                f"\n\n AO grid position: {ao_stage_positions[ao_grid_idx]}",
                f"\n Exp. stage position: {stage_positions[pos_idx]}"
            )
    aoMirror_local.positions_modal_array = position_wfc_coeffs
    aoMirror_local.positions_voltage_array = position_wfc_positions
    return True

#-------------------------------------------------#
# Helper functions for saving optmization results
#-------------------------------------------------#

def save_optimization_results(all_images: ArrayLike,
                              all_metrics: ArrayLike,
                              images_per_iteration: ArrayLike,
                              metrics_per_iteration: ArrayLike,
                              coefficients_per_iteration: ArrayLike,
                              modes_to_optimize: List[int],
                              save_dir_path: Path):
    """_summary_

    Parameters
    ----------
    images_per_mode : ArrayLike
        _description_
    metrics_per_mode : ArrayLike
        _description_
    images_per_iteration : ArrayLike
        _description_
    metrics_per_iteration : ArrayLike
        _description_
    coefficients_per_iteration : ArrayLike
        _description_
    modes_to_optimize : List[int]
        _description_
    save_dir_path : Path
        _description_
    """

    # Create the Zarr directory if it doesn't exist
    store = zarr.DirectoryStore(str(save_dir_path / Path("ao_results.zarr")))
    root = zarr.group(store=store)

    # Create datasets in the Zarr store
    root.create_dataset("all_images", data=all_metrics, overwrite=True)
    root.create_dataset("all_metrics", data=all_metrics, overwrite=True)
    root.create_dataset("images_per_iteration", data=images_per_iteration, overwrite=True)
    root.create_dataset("metrics_per_iteration", data=metrics_per_iteration, overwrite=True)
    root.create_dataset("coefficients_per_iteration", data=coefficients_per_iteration, overwrite=True)
    root.create_dataset("modes_to_optimize", data=modes_to_optimize, overwrite=True)
    root.create_dataset("zernike_mode_names", data=np.array(mode_names, dtype="S"), overwrite=True)

def load_optimization_results(results_path: Path):
    """Load optimization results from a Zarr store.

    Parameters
    ----------
    results_path : Path
        Path to the Zarr directory containing the data.
    """
    # Open the Zarr store
    store = zarr.DirectoryStore(str(results_path))
    results = zarr.open(store)
    
    images_per_mode = results["images_per_mode"][:]
    metrics_per_mode = results["metrics_per_mode"][:]
    images_per_iteration = results["images_per_iteration"][:]
    metrics_per_iteration = results["metrics_per_iteration"][:]
    coefficients_per_iteration = results["coefficients_per_iteration"][:]
    modes_to_optimize = results["modes_to_optimize"][:]
    zernike_mode_names = [name.decode("utf-8") for name in results["zernike_mode_names"][:]]

    ao_results = {
        "images_per_mode":images_per_mode,
        "metrics_per_mode":metrics_per_mode,
        "metrics_per_iteration":metrics_per_iteration,
        "images_per_iteration":images_per_iteration,
        "coefficients_per_iteration":coefficients_per_iteration,
        "modes_to_optimize":modes_to_optimize,
        "mode_names":zernike_mode_names,
    }
    return ao_results

#-------------------------------------------------#
# Run to 'keeps mirror flat'
#-------------------------------------------------#

if __name__ == "__main__":
    """Keeps the mirror in it's flat position
    """
    wfc_config_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\Configuration Files\WaveFrontCorrector_mirao52-e_0329.dat")
    wfc_correction_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\correction_data_backup_starter.aoc")
    haso_config_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\Configuration Files\WFS_HASO4_VIS_7635.dat")
    wfc_flat_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\flat_actuator_positions.wcs")
    wfc_calibrated_flat_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\20250215_tilted_brightness_laser_actuator_positions.wcs")

    # ao_mirror puts the mirror in the flat_position state to start.
    ao_mirror = AOMirror(wfc_config_file_path = wfc_config_file_path,
                         haso_config_file_path = haso_config_file_path,
                         interaction_matrix_file_path = wfc_correction_file_path,
                         system_flat_file_path = wfc_calibrated_flat_path)
    
    input("Press enter to exit . . . ")
    ao_mirror = None



