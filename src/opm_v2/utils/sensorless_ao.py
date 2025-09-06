"""
Sensorless adaptive optics and tools.

2024/12 DPS: initial work
2025/09/05 SJS: updates to synchronize with opm_custom_events and opm_config
"""
from pymmcore_plus import CMMCorePlus
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Sequence, List, Dict
from scipy.fftpack import dct
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from pathlib import Path
import zarr
from time import sleep

try:
    from opm_v2.hardware.AOMirror import AOMirror
    from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
except Exception:
    AOMirror = None
    OPMNIDAQ = None
    
DEBUGGING = False

# Metric global parameters
# METRIC_PRECISION = 5
METRIC_PERC_THRESHOLD = 1.0
MAXIMUM_MODE_DELTA = 0.5
PSF_RADIUS_PX = 3
SIGN_FIGS = 6

# Modes to optimize lists
focusing_modes = [2,7,14,23]
spherical_modes = [7,14,23]
spherical_modes_first =  [7,14,23,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
all_modes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
stationary_modes = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

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
# Modal AO optimization
#-------------------------------------------------#

def get_metric(
    image: ArrayLike,
    metric_to_use: str
) -> float:
    """ Calculate the selected metric on the provided image

    Parameters
    ----------
    image : ArrayLike
        image to measure metric on
    metric_to_use : str
        descriptor for metric to use

    Returns
    -------
    float
        calculated metric
    """
    if metric_to_use not in ['DCT", "localize_gauss_2d", "gauss_2d", "brightness", "fourier_ratio']:
        print(f"Warning: AO metric '{metric_to_use}' not supported. Exiting function.")
        return np.nan
    try:
        if "DCT" in metric_to_use:
            metric = metric_shannon_dct(image, PSF_RADIUS_PX)  * 100
        elif "localize_gauss_2d" in metric_to_use:
            metric = metric_localize_gauss2d(image)
        elif 'gauss_2d' in metric_to_use:
            metric = metric_gauss2d(image)
        elif 'brightness' in metric_to_use:
            metric = metric_brightness(image)
        elif "fourier_ratio" in metric_to_use:
            # TODO: implement fourier ratio metric
            metric = metric_shannon_dct(image, PSF_RADIUS_PX)
    except Exception as e:
        print(f"Warning: AO metric '{metric_to_use}' calculation failed with exception: {e}")
        metric = np.nan
    return metric

def metric_from_fit(a: float, b: float, c: float, delta: float) -> float:
    """Return the optimal metric based on the delta and quadratic fit

    Parameters
    ----------
    a : float
        x2 term of quadratic
    b : float
        x term of quadratic
    c : float
        constant term of quadratic
    delta : float
        Optimal delta to evaluate quadratic at

    Returns
    -------
    float
        metric at the given delta and fit parameters
    """
    return a * delta**2 + b * delta + c

def round_to_sigfigs(x: float, signif_figs: int=SIGN_FIGS) -> float:
    """Round value to given significant figures

    Parameters
    ----------
    x : float
        value to round
    signif_figs : int, optional
        number of significant figures, by default SIGN_FIGS

    Returns
    -------
    float
        given value rounded to significant figures
    """
    if x == 0:
        return 0
    else:
        digits = signif_figs - int(np.floor(np.log10(abs(x)))) - 1
        return np.round(x, digits)

def run_ao_optimization(
    exposure_ms: float,
    channel_states: List[bool],
    metric_to_use: Optional[str] = "DCT",
    daq_mode: Optional[str] = "projection",
    image_mirror_range_um: float = 100,
    num_iterations: Optional[int] = 3,
    num_mode_steps: Optional[int] = 3,
    init_delta_range: Optional[float] = 0.25,
    delta_range_alpha_per_iter: Optional[float] = 0.9,
    metric_precision: Optional[int] = SIGN_FIGS,
    modes_to_optimize: Optional[List[int]] = spherical_modes_first,
    starting_mirror_state: Optional[str] = "system flat",
    save_dir_path: Optional[Path] = None,
    save_prefix: Optional[str] = None,
    verbose: Optional[bool] = True,
) -> bool:
    """Run sensorless adaptive optics

    Parameters
    ----------
    exposure_ms : float
        camera exposure in ms
    channel_states : List[bool]
        channel states list to pass to daq
    metric_to_use : Optional[str], optional
        descriptor of metric to use, by default "DCT"
    daq_mode : Optional[str], optional
        by default "projection"
    image_mirror_range_um : float, optional
        by default 100
    num_iterations : Optional[int], optional
        by default 3
    num_mode_steps : Optional[int], optional
        number of deltas to sample per mode, by default 3
    init_delta_range : Optional[float], optional
        maximum mode coefficient delta to sample, by default 0.25
    delta_range_alpha_per_iter : Optional[float], optional
        factor to reduce delta range by per iteration, by default 0.9
    metric_precision : Optional[int], optional
        number of significant figures to round metrics to, by default SIGN_FIGS
    modes_to_optimize : Optional[List[int]], optional
        list of modes to optimize, by default spherical_modes_first
    starting_mirror_state : Optional[str], optional
        the starting mirror state, either system flat or last optimized, by default "system flat"
    save_dir_path : Optional[Path], optional
        Path to save figures, by default None
    save_prefix : Optional[str], optional
        Path prefix to append for saving the AO mirror state, by default None
    verbose : Optional[bool], optional
        whether to print out updates, by default True

    Returns
    -------
    bool
        Indicates success or not
    """
    
    #---------------------------------------------#
    # Create hardware controller instances
    #---------------------------------------------#
    opmNIDAQ_local = OPMNIDAQ.instance()
    aoMirror_local = AOMirror.instance()
    mmc = CMMCorePlus.instance()
    
    # Enforce camera exposure
    mmc.setProperty("OrcaFusionBT", "Exposure", float(exposure_ms))
    mmc.waitForDevice("OrcaFusionBT")
    
    # get the current stage position
    stage_position = {
        'z': mmc.getZPosition(),
        'y': mmc.getYPosition(),
        'x': mmc.getXPosition()
    }
    
    #---------------------------------------------#
    # Setup metadata
    #---------------------------------------------#
    
    mode_deltas = [
        (init_delta_range - k*init_delta_range*(1-delta_range_alpha_per_iter)) for k in range(num_iterations)
    ]
    metadata = {
        'starting_mirror_state':starting_mirror_state,
        'stage_position':stage_position,
        'opm_mode': daq_mode,
        'metric_name': metric_to_use,
        'num_iterations': num_iterations,
        'num_mode_steps': num_mode_steps,
        'mode_deltas': mode_deltas,
        'modes_to_optimize': modes_to_optimize,
        'channel_states': channel_states,
        'channel_exposure_ms': exposure_ms,
        'image_mirror_range_um': image_mirror_range_um
    }
    
    #---------------------------------------------#
    # Setup lists for images / metrics / coefficients
    #---------------------------------------------#
    
    """Definition of variables
    all_images: ALL images passed into get_metric except optimal measurements.
    all_metrics: ALL measured metrics, except optimal measurements
    all_optimal_metrics: ALL metrics obtained and kept when applying the optimal_delta. 
                         This list is appended after each mode. If no update is applied
                         the previous optimal metric is appended.
    all_optimal_images: ALL images obtained and kept after apply the optimal_delta. If no update
                        is applied, the last optimal image is appended. Does not reset.
    current_optimal_metrics: Metrics obtained and kept when applying the optimal_delta. This
                             list is reset each iteration and appended after each mode. If no 
                             update is applied the previous optimal metric is appended. 
    current_coeffs: Mirror mode coefficients updated when new mode deltas are accepted. 
                    If not update is applied, no chages are made. This list is reset at
                    the start of each iteration.    
    optimal_coeffs: The mirror coefficients at the end of each iteration. Includes the
                    starting mode coefficients. Includes the starting coefficients.
    active_coeffs: An array that holds temporary mirror coefficients that are applied. 
    """
    all_images = []
    all_metrics = []
    all_optimal_metrics = []
    all_optimal_images = []
    current_optimal_metrics = []
    current_coeffs = []
    optimal_coeffs = []
    
    #---------------------------------------------#
    # Set starting mode coeff
    #---------------------------------------------#
    if DEBUGGING:
        print(
            'starting_mirror_state:', starting_mirror_state
        )

    if starting_mirror_state == "system flat":
        aoMirror_local.apply_system_flat_voltage()
        if verbose:
            print(
                '--- AOmirror: Mirror set to system flat positions ----'
            )
    elif starting_mirror_state == 'last optimized':
        aoMirror_local.apply_last_optimized()
        if verbose:
            print(
                '--- AOmirror: Mirror set to last optimized positions ----'
                f'\n  mode coeffs: {aoMirror_local.current_coeffs}'
            )
    else:
        print('Starting mirror state defualted to system flat')
        aoMirror_local.apply_system_flat_voltage()
        
    starting_coeffs = aoMirror_local.current_coeffs.copy()
    
    if verbose:
        print(
            f'\n AO optimization starting mirror modal amplitudes:\n{aoMirror_local.current_zernikes}'
        )
        
    #---------------------------------------------#
    # setup the daq for the selected imaging mode
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
        print(
            f'+++++ Starting A.O. optimization ++++++'
            f'\n   Metric: {metric_to_use}'            
        )

    try:
        starting_image = mmc.snap()
        starting_metric = get_metric(starting_image, metric_to_use)
        starting_metric = round_to_sigfigs(starting_metric, metric_precision)
    except Exception as e:
        print(
            'Failed to get metric!'
            f'{e}')
        return

    
    # Start AO iterations 
    for k in range(num_iterations): 
        # initialize current values
        if k==0:
            all_images.append(starting_image)
            all_metrics.append(starting_metric)
            all_optimal_images.append(starting_image)
            all_optimal_metrics.append(starting_metric)
            current_optimal_metrics.append(starting_metric)
        else:
            current_optimal_metrics = []

        # Get the current mirror state at the start of iteration
        current_coeffs = aoMirror_local.current_coeffs.copy()
        optimal_coeffs.append(aoMirror_local.current_coeffs.copy())
            
        if verbose:
            print(
                '\n-------------------------------------------------------------------'
                '\n-------------------------------------------------------------------'
                f'\nAO iteration: {k+1} / {num_iterations}'
                f'   Modal pertubation amplitude: {mode_deltas[k]:.3f}'
                '\n-------------------------------------------------------------------'
                '\n-------------------------------------------------------------------'
            )

        # Iterate over modes to optimize
        for mode in modes_to_optimize:
            if verbose:
                print(
                    f'\n    ++ Perturbing mirror with Zernike mode: {mode+1} : {mode_names[mode]}\n ++'
                    f'\n    ++ Modal pertubation amplitude: {mode_deltas[k]:.3} ++'
                    )
                
            metrics = []
            deltas = np.linspace(-mode_deltas[k], mode_deltas[k], num_mode_steps)
            for delta in deltas:
                # Apply perturbation to the mirror
                active_coeffs = current_coeffs.copy()
                active_coeffs[mode] += delta
                success = aoMirror_local.set_modal_coefficients(active_coeffs)
                
                if not(success):
                    metric = 0
                    image = np.zeros_like(starting_image)
                else:
                    # Acquire image
                    if not opmNIDAQ_local.running():
                        opmNIDAQ_local.start_waveform_playback()
                    image = mmc.snap()
                    
                    # Measure the image metric
                    try:
                        metric = get_metric(image, metric_to_use)
                        metric = round_to_sigfigs(metric, metric_precision)
                        metrics.append(metric)
                        all_metrics.append(metric)
                        all_images.append(image)
                    except Exception:
                        success = False
                        
                    # Catch bad metric calculations
                    if not(success) or metric==np.nan or metric==0:
                        success = False
                        if verbose:
                            print('\n    ---- Metric failed! ----')
                                        
                if verbose:
                    print(f'    + Delta={delta:.4f}, Metric = {metric:.4}, Success = {success} +')
            
            #---------------------------------------------#
            # Fit metrics to determine optimal metric
            # Skip if there were any problems in metrics
            #---------------------------------------------#   
            if success:
                if k!=0 and len(current_optimal_metrics)==0:
                    current_optimal_metrics.append(metrics[num_mode_steps//2])
                try:                  
                    # Are metrics monotonic, if so use the maximum
                    is_increasing = all(x < y for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:]))
                    is_decreasing = all(x > y for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:]))
                    if is_increasing or is_decreasing:
                        optimal_delta = deltas[np.argmax(metrics)]
                        print(
                            f' --- Metrics increasing/decreasing using maximum: {optimal_delta} ---'
                        )
                        # optimal_delta = 0
                    else:
                        # Fit and use the peak of quadratic fit
                        popt = quadratic_fit(deltas, metrics)
                        a, b, c = popt
                        optimal_delta = -b / (2 * a)
                        
                        if DEBUGGING:
                            # NOTE: Our delta and booth's delta only match if 3 points are used
                            booth_delta = -mode_deltas[k] * (metrics[2] - metrics[0]) / (2*metrics[0] + 2*metrics[2] - 4*metrics[1])
                            optimal_delta = booth_delta
                            if verbose:
                                print(
                                    f'\n    ++   our delta: {optimal_delta}   ++'
                                    f'\n    ++   Booth delta {booth_delta}   ++'
                                )
                                
                        # Reject result if metrics have positive curvature or too large of delta
                        if a >=0:
                            optimal_delta = 0
                            print(
                                f' --- Metrics have positive curvature ---'
                            )
                        elif np.abs(optimal_delta)>MAXIMUM_MODE_DELTA:
                            optimal_delta = 0
                            print(
                                ' --- Optimal delta is outside of range: {optimal_delta:.3f} ---'
                            )
                        
                except Exception as e:
                    optimal_delta = 0
                    if verbose:
                        print(f' --- Exception in fit occurred! ---\n      e:{e}')
            else:
                optimal_delta = 0
                if verbose:
                    print(' --- Error occurred in acquiring image or metric! ---')
           
            #---------------------------------------------#
            # Apply the kept optimized mirror modal coeffs
            #---------------------------------------------# 
            
            if optimal_delta!=0:
                # remeasure metric to verify new position
                active_coeffs = current_coeffs.copy() 
                active_coeffs[mode] = active_coeffs[mode] + optimal_delta
                _ = aoMirror_local.set_modal_coefficients(active_coeffs)

                # Acquire image and metric
                if not opmNIDAQ_local.running():
                        opmNIDAQ_local.start_waveform_playback()
                optimal_image = mmc.snap()
                optimal_metric = get_metric(optimal_image, metric_to_use)
                optimal_metric = round_to_sigfigs(optimal_metric, metric_precision)
                if verbose:
                    print(
                        f'\n   ++ Optimal delta from fit: {optimal_delta:.4f} ++'
                        f'\n   ++ Measured optimal metric: {optimal_metric:.4} ++'
                    )
                
                # Check if the new metric is better than the current optimal metric
                if optimal_metric >= current_optimal_metrics[-1]*METRIC_PERC_THRESHOLD:
                    # Accept the change, and update current mode coeffs
                    update = True
                    current_coeffs[mode] = current_coeffs[mode] + optimal_delta
                    all_optimal_images.append(optimal_image)
                    all_optimal_metrics.append(optimal_metric)
                    current_optimal_metrics.append(optimal_metric)
                else:
                    # Reject the change, keep current mode coeffs
                    update = False
                    all_optimal_images.append(all_optimal_images[-1])
                    all_optimal_metrics.append(all_optimal_metrics[-1])
                    current_optimal_metrics.append(current_optimal_metrics[-1])
                    
                if verbose:
                    print(
                        f'\n   ++ Current optimal metric: {current_optimal_metrics[-1]:.6f} ++'
                    )
            else:
                update = False
            
            # Update the mirror to the current best state
            _ = aoMirror_local.set_modal_coefficients(current_coeffs)   
             
            if verbose:
                print(
                    f'\n   ++ Was mirror updated: {update} ++'
                )
                
            """Loop back to top and do the next mode until all modes are done"""
                        
        """Loop back to top and do the next iteration"""
        
    #---------------------------------------------#
    # After all the iterations the mirror state will be optimized
    #---------------------------------------------# 
    aoMirror_local.update_optimized_array()
    if verbose:
        print(
            f"\n++++ Starting Zernike mode amplitude: ++++ \n{starting_coeffs}",
            f"\n++++ Final optimized Zernike mode amplitude: ++++ \n{current_coeffs}"
            )
    
    # Stop the DAQ programming
    opmNIDAQ_local.stop_waveform_playback()
    
    if save_dir_path:
        if verbose:
            print(f"Saving AO results at:\n  {save_dir_path}\n")
        
        if save_prefix:
            aoMirror_local.save_current_state(save_prefix)
            aoMirror_local.save_positions_array(prefix=save_prefix)
            
        all_images = np.asarray(all_images)
        all_metrics = np.asarray(all_metrics)
        all_optimal_images = np.asarray(all_optimal_images)
        all_optimal_metrics = np.asarray(all_optimal_metrics)
        optimal_coeffs = np.asarray(optimal_coeffs)
        # TODO
        # optimized_phase = aoMirror_local.get_current_phase()
        
        try:
            save_optimization_results(
                all_images=all_images,
                all_metrics=all_metrics,
                images_per_iteration=all_optimal_images,
                metrics_per_iteration=all_optimal_metrics,
                optimal_coeffs=optimal_coeffs,
                # optimized_phase=optimized_phase,
                modes_to_optimize=modes_to_optimize,
                metadata=metadata,
                save_dir_path=save_dir_path
            )        
            plot_zernike_coeffs(
                optimal_coeffs=optimal_coeffs,
                num_iterations=num_iterations,
                zernike_mode_names=mode_names,
                save_dir_path=save_dir_path,
                show_fig=False,
                x_range=0.1
            )        
            plot_metric_progress(
                all_metrics = all_metrics,
                modes_to_optimize = modes_to_optimize,
                num_iterations = num_iterations,
                zernike_mode_names = mode_names,
                save_dir_path = save_dir_path,
                show_fig = False,
            )
        except Exception as e:
            print(
                '\n  ----- Saving result had an exception! -----'
                f'\n {e}'
            )
            
#-------------------------------------------------#
# Plotting functions
#-------------------------------------------------#

def plot_zernike_coeffs(
    optimal_coeffs: ArrayLike,
    num_iterations: int,
    zernike_mode_names: ArrayLike,
    save_dir_path: Optional[Path] = None,
    show_fig: Optional[bool] = False,
    x_range = 0.1
) -> None:
    """Plot the Zernike coefficient values per iteration

    Parameters
    ----------
    optimal_coeffs : ArrayLike
    num_iterations : int
    zernike_mode_names : ArrayLike
    save_dir_path : Optional[Path], optional
        Path to save figure, by default None
    show_fig : Optional[bool], optional
        whether to display figure, by default False
    x_range : float, optional
        the coefficient magnitude to display on axis, by default 0.1
    """
    import matplotlib.pyplot as plt
    import matplotlib
    if not show_fig:
        matplotlib.use('Agg')
    from matplotlib.ticker import FormatStrFormatter
    
    plt.rcParams.update({
        "font.size": 14,        # base font size
        "axes.titlesize": 18,   # title size
        "axes.labelsize": 15,   # x/y label size
        "xtick.labelsize": 15,  # x-tick label size
        "ytick.labelsize": 15,  # y-tick label size
        "legend.fontsize": 14   # legend size
    })
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Define colors and markers for each iteration
    colors = ['b', 'g', 'r', 'c', 'm']  
    markers = ['x', 'o', '^', 's', '*']  

    # populate plots
    for i in range(len(zernike_mode_names)):
        for j in range(num_iterations):
            marker_style = markers[j % len(markers)]
            ax.scatter(
                optimal_coeffs[j, i], i, 
                color=colors[j % len(colors)],
                s=125, 
                marker=marker_style,
                alpha=0.5
            )  
        ax.axhline(y=i, linestyle="--", linewidth=1, color='k')
        
    # Plot a vertical line at 0 for reference
    ax.axvline(0, color='k', linestyle='-', linewidth=1)

    # Customize the plot
    ax.set(
        yticks=np.round(np.arange(len(zernike_mode_names)),1),
        yticklabels=zernike_mode_names,
        xlabel=r'$\vert Z_n \vert$',
        title='Zernike Coefficients per Iteration',
        xlim=(-x_range, x_range)
    )

    # Add a legend for time points
    ax.legend(
        [f'Iter.: {i+1}' for i in range(num_iterations)], 
        loc='upper right'
    )

    # Remove grid lines
    ax.grid(False)

    plt.tight_layout()

    plt.tight_layout()
    if show_fig:
        plt.show()
    if save_dir_path:
        fig.savefig(save_dir_path / Path("ao_zernike_coeffs.png"))

def plot_metric_progress(
    all_metrics: ArrayLike,
    num_iterations: float,
    modes_to_optimize: List[int],
    zernike_mode_names: List[str],
    save_dir_path: Optional[Path] = None,
    show_fig: Optional[bool] = False
) -> None:
    """Plot the metric magnitude throughout optimization

    Parameters
    ----------
    all_metrics : ArrayLike
    num_iterations : float
    modes_to_optimize : List[int]
    zernike_mode_names : List[str]
    save_dir_path : Optional[Path], optional
        Path to save figure, by default None
    show_fig : Optional[bool], optional
        whether to display figure, by default False
    """
    import matplotlib.pyplot as plt
    if not show_fig:
        import matplotlib
        matplotlib.use('Agg')
        
    plt.rcParams.update({
        "font.size": 14,        # base font size
        "axes.titlesize": 18,   # title size
        "axes.labelsize": 15,   # x/y label size
        "xtick.labelsize": 15,  # x-tick label size
        "ytick.labelsize": 15,  # y-tick label size
        "legend.fontsize": 14   # legend size
    })
    num_modes = len(modes_to_optimize)
    samples_per_mode = len(all_metrics) // num_iterations / len(modes_to_optimize)
    modes_to_use_names = [zernike_mode_names[i] for i in modes_to_optimize]

    # Ignore starting metric
    metrics = np.array(all_metrics[1:])

    # Reshape into: (iterations, modes, samples)
    metrics_by_iteration = metrics.reshape(
        num_iterations, num_modes, int(samples_per_mode)
    )
    zero_metrics_by_iteration = metrics_by_iteration[:, :, int(samples_per_mode//2)]
        
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and markers for each iteration
    colors = ['b', 'g', 'r', 'c', 'm']
    markers = ['x', 'o', '^', 's', '*']

    # Loop over iterations and plot each series
    for ii in range(num_iterations):
        ax.plot(
            np.linspace(0, num_modes-1, num_modes),
            zero_metrics_by_iteration[ii],
            color=colors[ii],
            label=f"iteration {ii}", 
            marker=markers[ii],
            linestyle="--", 
            linewidth=1
        )

    ax.set(
        title='Metric Progress per Iteration',
        ylabel='Metric',
        xticks=np.arange(len(modes_to_use_names))
    )
    ax.set_xticklabels(modes_to_use_names, rotation=60, ha="right", fontsize=16) 
    ax.legend()
    plt.tight_layout()
    
    if show_fig:
        plt.show()
    if save_dir_path:
        fig.savefig(save_dir_path / Path("ao_metrics.png"))

def plot_phase(phase: Dict,
               save_dir_path: Path = None,
               show_fig: bool = False
) -> None:
    """Plot the 2d Phase for a given set of modal coeffs

    Parameters
    ----------
    phase : Dict
        _description_
    save_dir_path : Path, optional
        Path to save figure, by default None
    showfig : bool, optional
        whether to display figure, by default False
    """
    
    import matplotlib.pyplot as plt
    import matplotlib
    if not show_fig:
        matplotlib.use('Agg')
    # --- Set rcParams (this affects all plots until you change/reset it) ---
    plt.rcParams.update({
        "font.size": 14,        # base font size
        "axes.titlesize": 18,   # title size
        "axes.labelsize": 16,   # x/y label size
        "xtick.labelsize": 14,  # x-tick label size
        "ytick.labelsize": 14,  # y-tick label size
        "legend.fontsize": 14   # legend size
    })
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    vrange = np.max([np.abs(phase['min']), np.abs(phase['max'])])
    im = ax.imshow(
        phase['phase'],
        cmap='seismic',
        vmin=-vrange,
        vmax=vrange
    )
    cbar = plt.colorbar(im)
    ax.set_title('Wavefront Phase')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    if show_fig:
        plt.show()
    if save_dir_path:
        fig.savefig(save_dir_path / Path("phase.png"))
        
def plot_2d_localization_fit_summary(
    fit_results,
    img,
    coords_2d,
    save_dir_path: Path = None,
    showfig: bool = False
):
    """Generate a figure showing the localization an fit results

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
    
    to_keep = fit_results['to_keep']
    sxy = fit_results['fit_params'][to_keep, 4]
    amp = fit_results['fit_params'][to_keep, 0]
    bg = fit_results['fit_params'][to_keep, 6]
    centers = fit_results['fit_params'][to_keep][:, (3, 2, 1)]
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
        weights=[fit_results['fit_params'][to_keep, 4]],
        color_lists=['autumn'],
        color_limits=[[0.05,0.5]],
        cbar_labels=[r'$\sigma_{xy}$'],
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
    if center is None:
        center = get_image_center(image,threshold=1000)

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

def metric_r_power_integral(
    img: np.ndarray,
    integration_radius: int = 40,
    power: int = 2
) -> float:
    """TODO

    Parameters
    ----------
    img : np.ndarray
        _description_
    integration_radius : int, optional
        _description_, by default 40
    power : int, optional
        _description_, by default 2

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    h, w = img.shape
    if min(h, w) < 2 * integration_radius:
        raise ValueError("Radius too large for image size")

    bg = np.percentile(img, 99)
    roi_binary = np.zeros(img.shape)
    roi_binary[img > bg] = 1

    cy, cx = center_of_mass(roi_binary)
    #print(f"Center of mass {cy},{cx}")

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x + 0.5 - cx)**2 + (y + 0.5 - cy)**2)
    mask = (dist <= integration_radius).astype(int)

    weighted = img * mask * (dist**power)
    return float(weighted.sum() / mask.sum())

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

def normalize_roi(roi, bg_percentile=25.0, debug_mode=False):
    """
    Normalize input image (roi) to [0,1] between the defined (low) percentile and the maximum.

    Parameters
    ----------
    roi, ndrray of image (ROI)
    metric_settings, named tuple

    Returns
    ----------
    roi_normalized, ndarray of normalized ROI
    """
    assert len(roi.shape) == 2, "Error: ROI captured by camera must be 2D."
    bg = np.percentile(roi, bg_percentile)
    peak = roi.max()
    roi_normalized = (roi - bg)/(peak - bg)
    roi_normalized = np.clip(roi_normalized, 0, 1)
    if debug_mode:
        print('normalize_roi() values (low_percentile, background, peak):'
              + str(bg_percentile) + ',  ' + "{0:2.3f}".format(bg) + ',  ' + "{0:2.3f}".format(peak))
    return roi_normalized

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
    """TODO

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
    threshold = localize_psf_filters['threshold']
    amp_bounds = localize_psf_filters['amp_bounds']
    sxy_bounds = localize_psf_filters['sxy_bounds']
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
    percentile: Optional[float] = None,
    image_center: Optional[int] = None,
    return_image: Optional[bool] = False
) -> float:
    """Compute weighted metric for 2D Gaussian.

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

    if percentile:
        image_perc = np.percentile(image, percentile)
        max_pixels = image[image >= image_perc]
        image_max = np.mean(max_pixels)
    else:
        image_max = np.max(image)

    if return_image:
        return float(image_max), image
    else:
        return float(image_max)

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
    threshold: Optional[float] = 1000,
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
    image = image / np.max(image) # normalize_roi(image, bg_percentile=25.0)
    image = image.astype(np.float32)
    
    # create coord. grid for fitting 
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)
    
    # fitting assumes a single bead in FOV....
    initial_guess = (image.max(), image.shape[1] // 2, 
                     image.shape[0] // 2, 3, 3, image.min())
    fit_bounds = [[0,0,0,1.0,1.0,0],
                  [1.5,image.shape[1],image.shape[0],100,100,5000]]
    try:
        popt, pcov = curve_fit(
            gauss2d, 
            (x, y),
            image.ravel(), 
            p0=initial_guess,
            bounds=fit_bounds,
            maxfev=1000
        )
        
        amplitude, center_x, center_y, sigma_x, sigma_y, offset = popt

        weighted_metric = (1 - np.abs((sigma_x-sigma_y) / (sigma_x+sigma_y))) + 1 / (sigma_x+sigma_y)  + np.exp(-1 * (sigma_x+sigma_y-1)**2)
        # weighted_metric = np.exp(-1 * np.abs(sigma_x - sigma_y)) + 2 / (sigma_y + sigma_x)

        # SJS: From old laser_optmization code 
        # weighted_metric = (1 - np.abs((sigma_x-sigma_y)/(sigma_x+sigma_y)))* (1/(sigma_x+sigma_y)) * np.exp(-1*(sigma_x+sigma_y-4)**2)
        # weight_amp = 0
        # weight_sigma_x = 2
        # weight_sigma_y = 2
        # weighted_metric = weight_amp * amplitude + weight_sigma_x / sigma_x + weight_sigma_y / sigma_y + np.exp(-1*(sigma_x+sigma_y-4)**2)
        
        if (weighted_metric <= 0) or (weighted_metric > 100):
            weighted_metric = 1e-12 
    except Exception:
        weighted_metric = 1e-12
        
        
    if return_image:
        return weighted_metric, image
    else:
        return weighted_metric

def metric_localize_gauss2d(image: ArrayLike) -> float:
    """TODO

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
        
        to_keep = fit_results['to_keep']
        sxy = fit_results['fit_params'][to_keep, 4]
        metric = 1.0 / np.median(sxy)
    except Exception as e:
        print(f"2d localization and fit exceptions: {e}")
        metric = 0
        
    return metric

#-------------------------------------------------#
# Helper function for generating grid
#-------------------------------------------------#

def run_ao_grid_mapping(
    ao_dict: dict,
    stage_positions: List,
    num_tile_positions: int = 1,
    num_scan_positions: int = 1,
    save_dir_path: Path = None,
    verbose: bool = False,
) -> bool:
    """Given a set of stage positions, generate a grid to run A.O.
    then interpolate to stage positions.

    Parameters
    ----------
    stage_positions : list
        Experimental stage positions. Optimized for stage scan acquisitions
    ao_dict : dict
        A dictionary containing AO optimization parameters
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
            (pos['z'], pos['y'], pos['x']) for pos in stage_positions
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
    
    if save_dir_path:
        # save stage positions
        import json
        ao_positions_path = save_dir_path / Path('ao_stage_positions.json')
        exp_positions_path = save_dir_path / Path('exp_stage_positions.json')
        with open(ao_positions_path, 'w') as file:
                json.dump(ao_stage_positions, file, indent=4)
        with open(exp_positions_path, 'w') as file:
                json.dump(stage_positions, file, indent=4)
                
    # Visit AO stage positions and measure best WFC positions
    for ao_pos_idx in range(num_ao_pos):
        if verbose:
            print(f"\nMoving stage to: {ao_stage_positions[ao_pos_idx]}")
            
        target_x = ao_stage_positions[ao_pos_idx]['x']
        target_y = ao_stage_positions[ao_pos_idx]['y']
        target_z = ao_stage_positions[ao_pos_idx]['z']
        
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
        
        if ao_pos_idx==0:
            mirror_state = ao_dict['mirror_state']
        else:
            mirror_state = 'last_optimized'
        current_save_dir = save_dir_path / Path(f"grid_pos_{int(ao_pos_idx)}")
        current_save_dir.mkdir(exist_ok=True)
        run_ao_optimization(
            starting_mirror_state=mirror_state,
            metric_to_use=ao_dict['metric'],
            channel_states=ao_dict['channel_states'],
            exposure_ms=ao_dict['exposure_ms'],
            daq_mode=ao_dict['daq_mode'],
            image_mirror_range_um=ao_dict['image_mirror_range_um'],
            num_iterations=ao_dict['iterations'],
            init_delta_range=ao_dict['modal_delta'],
            delta_range_alpha_per_iter=ao_dict['modal_alpha'],
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
    ao_stage_positions_array = np.array([(pos['z'], pos['y'], pos['x']) for pos in ao_stage_positions])
    stage_positions_array = np.array([(pos['z'], pos['y'], pos['x']) for pos in stage_positions])

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

        if DEBUGGING:
            print(
                f"\n\n ++++ AO grid position: {ao_stage_positions[ao_grid_idx]} ++++",
                f"\n ++++ Exp. stage position: {stage_positions[pos_idx]} ++++"
            )
    aoMirror_local.positions_modal_array = position_wfc_coeffs
    aoMirror_local.positions_voltage_array = position_wfc_positions
    
    if verbose:
        print("AO Grid complete!")
    return True

#-------------------------------------------------#
# Helper functions for saving optmization results
#-------------------------------------------------#

def save_optimization_results(
    all_images: ArrayLike,
    all_metrics: ArrayLike,
    images_per_iteration: ArrayLike,
    metrics_per_iteration: ArrayLike,
    optimal_coeffs: ArrayLike,
    modes_to_optimize: List[int],
    # optimized_phase: Dict,
    metadata: Dict,
    save_dir_path: Path
) -> None:
    """Save the results from running AO-optimize

    Parameters
    ----------
    all_images : ArrayLike
        all images acquired during optimization
    all_metrics : ArrayLike
        all metrics measured during optimization
    images_per_iteration : ArrayLike
        optimal images per interation, including the starting image
    metrics_per_iteration : ArrayLike
        optmial metrics per iteration, including the starting metric
    optimal_coeffs : ArrayLike
        optimal coefficients per iteration
    modes_to_optimize : List[int]
        The modes optimized, in order
    metadata : Dict
        run_optimization parameters
    save_dir_path : Path
        zarr destination path
    """

    # Create the Zarr directory if it doesn't exist
    store = zarr.DirectoryStore(str(save_dir_path / Path("ao_results.zarr")))
    root = zarr.group(store=store)

    # Create datasets in the Zarr store
    root.create_dataset("all_images", data=all_images, overwrite=True)
    root.create_dataset("all_metrics", data=all_metrics, overwrite=True)
    root.create_dataset("images_per_iteration", data=images_per_iteration, overwrite=True)
    root.create_dataset("metrics_per_iteration", data=metrics_per_iteration, overwrite=True)
    root.create_dataset("optimal_coeffs", data=optimal_coeffs, overwrite=True)
    root.create_dataset("modes_to_optimize", data=modes_to_optimize, overwrite=True)
    # root.create_dataset("optimized_phase", data=optimized_phase, overwrite=True)
    root.create_dataset("zernike_mode_names", data=np.array(mode_names, dtype="S"), overwrite=True)
    root.attrs.update(metadata)
    
    return
    
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
    
    all_images = results['all_images'][:]
    all_metrics = results['all_metrics'][:]
    images_per_iteration = results['images_per_iteration'][:]
    metrics_per_iteration = results['metrics_per_iteration'][:]
    optimal_coeffs = results['optimal_coeffs'][:]
    # optimized_phase = results['optimized_phase'][:]

    modes_to_optimize = results['modes_to_optimize'][:]
    zernike_mode_names = [name.decode("utf-8") for name in results['zernike_mode_names'][:]]
    
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