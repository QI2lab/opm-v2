"""Sensorless adaptive optics.

TO DO: 
- Load interaction matrix from disk
- Set and get Zernike mode amplitudes from mirror
- Might need HASO functions to do this, since Zernike need to be composed given the pupil

2024/12 DPS initial work
"""


import numpy as np
from time import sleep
import time
from typing import Optional, List
from typing import Optional
from pathlib import Path
from pymmcore_plus import CMMCorePlus
from datetime import datetime

import matplotlib.pyplot as plt
from opm_v2.hardware.AOMirror import AOMirror, plotDM
import opm_v2.utils.sensorless_ao as ao

METRIC_PERC_THRESHOLD = 1.0
MAXIMUM_MODE_DELTA = 0.5
SIGN_FIGS = 3

focusing_modes = [2,7,14,23]
spherical_modes = [7,14,23]
spherical_modes_first =  [7,14,23,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
spherical_modes_first_plus_defocus =  [7,14,23,2,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
all_modes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
all_modes_no_tilt = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
stationary_modes = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
custom = [3,4,5,6,10,11,12,13]
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
# Hardare configuration paths
#-------------------------------------------------#

mm_configPath = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\OPM_alignment_config.cfg")

wfc_config_file_path = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_configuration_files\WaveFrontCorrector_mirao52-e_0329.dat"
)
haso_config_file_path = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_configuration_files\WFS_HASO4_VIS_7635.dat"
)
wfc_correction_file_path = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_configuration_files\202508026_tilted_interaction_matrix.aoc"
)
wfc_flat_file_path = Path(
    r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_setup_scripts\20250827_141952_laser_brightness_brightness_from_wfs_opt_using_max\20250827_brightness_from_wfs_opt_max_wfc_voltage.wcs'
)
# wfc_flat_file_path = None
output_root_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\wfc_setup_scripts')

#-------------------------------------------------#
# Setup global variables
#-------------------------------------------------#

o3_stage_name = 'MCL NanoDrive Z Stage'
o3_start_position = 49.3
zstack_depth = 5.0
pixel_center = [1143, 1146] #[1150,1150]
image_crop = 56
image_type = 'max'

image_history = []
current_frame_index = 0
live_mode = True
user_is_dragging_slider = False 

#-------------------------------------------------#
# Setup save directories
#-------------------------------------------------#

# Setup saving paths 
name = 'brightness_from_wfs_opt'
ao_save_path_prefix = f'20250827_{name}_{image_type}'
save_dir_suffix = f'brightness_{name}_using_{image_type}'

# Create a unique directory to save results
now = datetime.now()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ao_mirror_output_path = output_root_path / Path(f"{timestamp}_laser_{save_dir_suffix}")
ao_mirror_output_path.mkdir(exist_ok=True) 

#-------------------------------------------------#
# Helper functions for running optimization locally
#-------------------------------------------------#

def get_metric(
    image,
    metric_to_use
):
    if "DCT" in metric_to_use:
        metric = ao.metric_shannon_dct(image)  
    elif "localize_gauss_2d" in metric_to_use:
        metric = ao.metric_localize_gauss2d(image)
    elif 'gauss_2d' in metric_to_use:
        metric = ao.metric_gauss2d(image)
    elif 'brightness' in metric_to_use:
        metric = ao.metric_brightness(image)
    else:
        return np.nan
    return metric

def acquire_zstack(
    image_type='max'
):
    # set MM focus stage to O3 piezo stage
    mmc.setFocusDevice(o3_stage_name)
    mmc.waitForDevice(o3_stage_name)

    # grab O3 focus stage position
    mmc.setPosition(o3_start_position)
    mmc.waitForDevice(o3_stage_name)

    # generate arrays
    O3_stage_step_size = .115
    n_O3_stage_steps= zstack_depth//O3_stage_step_size

    O3_stage_positions = np.round(np.arange(o3_start_position-(O3_stage_step_size*np.round(n_O3_stage_steps/2,0)),o3_start_position+(O3_stage_step_size*np.round(n_O3_stage_steps/2,0)),O3_stage_step_size),2).astype(np.float64)
    images = []
    for O3_stage_pos in O3_stage_positions:
        mmc.setPosition(O3_stage_pos)
        mmc.waitForDevice(o3_stage_name)
        images.append(mmc.snap())
    
    mmc.setPosition(o3_start_position)
    mmc.waitForDevice(o3_stage_name)
    
    images = np.asarray(images)
    if image_type=='max':
        image = np.max(images, axis=0)
    elif image_type=='sum':
        image = np.sum(images, axis=0)
        
    return image

def metric_from_fit(a, b, c, delta):
    return a * delta**2 + b * delta + c


def round_to_sigfigs(x):
    if x == 0:
        return 0
    else:
        digits = SIGN_FIGS - int(np.floor(np.log10(abs(x)))) - 1
        return np.round(x, digits)
    
#-------------------------------------------------#
# Main optimization function
#-------------------------------------------------#

def optimize_modes(
    exposure_ms: float,
    metric_to_use: Optional[str] = "brightness",
    num_iterations: Optional[int] = 3,
    num_mode_steps: Optional[int] = 3,
    init_delta_range: Optional[float] = 0.25,
    delta_range_alpha_per_iter: Optional[float] = 0.9,
    modes_to_optimize: Optional[List[int]] = spherical_modes_first,
    save_dir_path: Optional[Path] = None,
    verbose: Optional[bool] = True,
    display_data: Optional[bool] = True,
    image_crop: Optional[int] = None
):
    #---------------------------------------------#
    # Create & update hardware controller instances
    #---------------------------------------------#
    
    aoMirror_local = AOMirror.instance()
    mmc = CMMCorePlus.instance()

    # Enforce camera exposure
    mmc.setProperty("OrcaFusionBT", "Exposure", float(exposure_ms))
    mmc.waitForDevice("OrcaFusionBT")

    # Set & get the starting mirror state    
    aoMirror_local.apply_system_flat_voltage()
    starting_coeffs = aoMirror_local.current_coeffs.copy() 
    
    # Define the coefficient deltas to use each iteration
    mode_deltas = [(init_delta_range - k*init_delta_range*(1-delta_range_alpha_per_iter)) for k in range(num_iterations)]
    
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
    # Start AO optimization
    #---------------------------------------------#   
    
    if verbose:
        print(
            f'+++++ Starting A.O. optimization ++++++'
            f'\n   Metric: {metric_to_use}'            
        )

    try:
        starting_image = acquire_zstack()
        if image_crop:
            starting_image = ao.get_cropped_image(
                starting_image,
                crop_size=image_crop,
                center=pixel_center
            )
        starting_metric = get_metric(starting_image, metric_to_use)
    except Exception as e:
        print(
            'Failed to get metric!'
            f'{e}')
        return
    
    if display_data:
        import matplotlib as mpl
        mpl.use("QtAgg")
        plt.ion()

         # Setup Figure for showing perturbed images 
        fig2, ax2 = plt.subplots()
        ax2.set_title("all images")
        im2 = ax2.imshow(starting_image,cmap="gray")
        def update_image2(new_data):
            new_data = new_data / np.max(new_data)
            im2.set_data(new_data)
            im2.set_clim(vmin=0,vmax=1.1)
            fig2.canvas.draw()
            fig2.canvas.flush_events()
            plt.pause(0.5)

        # Setup Figure for showing optimal images 
        fig1, ax1 = plt.subplots()
        ax1.set_title(f"Optimal images: {starting_metric}")
        im1 = ax1.imshow(starting_image,cmap="gray")
        def update_image1(new_data, metric):
            new_data = new_data / np.max(new_data)
            ax1.set_title(f"Optimal image: {metric}")
            im1.set_data(new_data)
            im1.set_clim(vmin=0,vmax=1.05)
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            plt.pause(0.5)

    #----------------------------------------------
    # Setup for saving AO results
    #----------------------------------------------
    
    if save_dir_path:
        if verbose:
            print(f"Saving AO results at:\n  {save_dir_path}\n")
            
        # Define metadata to save
        metadata = {
            'stage_position':None,
            'opm_mode': None,
            'metric_name': metric_to_use,
            'num_iterations': num_iterations,
            'num_mode_steps': num_mode_steps,
            'mode_deltas': mode_deltas,
            'modes_to_optimize': modes_to_optimize,
            'channel_states': None,
            'channel_exposure_ms': exposure_ms,
            'image_mirror_range_um': None
        }
    
    # Start AO iterations 
    for k in range(num_iterations): 
        # initialize current values
        if k==0:
            all_images.append(starting_image)
            all_metrics.append(starting_metric)
            all_optimal_images.append(starting_image)
            all_optimal_metrics.append(starting_metric)
        
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
                    '\n-------------------------------------------------------------------\n'
                    f'\n++++ Perturbing mirror with Zernike mode: {mode_names[mode]} ++++'
                    )
            # Acquire images and metrics for a range of modal coeff deltas
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
                    image = acquire_zstack(image_type)
                    if image_crop:
                        image = ao.get_cropped_image(
                            image,
                            crop_size=image_crop,
                            center=pixel_center
                        )
                    # Measure the image metric
                    try:
                        metric = round_to_sigfigs(get_metric(image, metric_to_use))
                        metrics.append(metric)
                        all_metrics.append(metric)
                        all_images.append(image)
                    except Exception:
                        success = False
                    if not(success) or metric==np.nan or metric==0:
                        success = False
                        if verbose:
                            print('\n    ---- Metric failed! ----')
                    elif display_data:
                        update_image2(image)
                        
                if verbose:
                    print(f'    + Delta={delta:.4f}, Metric = {metric:.6f}, Success = {success} +')
                    
            #---------------------------------------------#
            # Fit metrics to determine optimal metric
            # Skip if there were any problems in metrics
            #---------------------------------------------#   
            if success:
                if len(current_optimal_metrics)==0:
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
                        
                    else:
                        # Fit and use the peak of quadratic fit
                        popt = ao.quadratic_fit(deltas, metrics)
                        a, b, c = popt
                        optimal_delta = -b / (2 * a)
                    
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
                optimal_image = acquire_zstack()
                if image_crop:
                    optimal_image = ao.get_cropped_image(
                        optimal_image,
                        crop_size=image_crop,
                        center=pixel_center
                    )
                optimal_metric = round_to_sigfigs(get_metric(optimal_image, metric_to_use))
                
                if display_data:
                    update_image1(optimal_image, optimal_delta)
                if verbose:
                    print(
                        f'\n   ++ Optimal delta from fit: {optimal_delta:.4f} ++'
                        f'\n   ++ Measured optimal metric: {optimal_metric:.1f} ++'
                    )
                # Check if the new metric is better than the best overall metric
                if round_to_sigfigs(optimal_metric)>=round_to_sigfigs(current_optimal_metrics[-1]*METRIC_PERC_THRESHOLD):
                    # Accept the change, and update current mode coeffs
                    update = True
                    current_coeffs[mode] = current_coeffs[mode] + optimal_delta
                    all_optimal_images.append(optimal_image)
                    all_optimal_metrics.append(optimal_metric)
                    current_optimal_metrics.append(optimal_metric)
                    if display_data:
                        update_image1(optimal_image, optimal_metric)
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
            f"\n  ++++ Starting Zernike mode amplitude: ++++ \n{starting_coeffs}"
            f"\n  ++++ Final optimized Zernike mode amplitude: ++++ \n{current_coeffs}"
            )
    
    if save_dir_path:
        aoMirror_local.save_current_state(prefix=ao_save_path_prefix)
        aoMirror_local.save_positions_array(prefix=ao_save_path_prefix)
    
        all_images = np.asarray(all_images)
        all_metrics = np.asarray(all_metrics)
        all_optimal_images = np.asarray(all_optimal_images)
        all_optimal_metrics = np.asarray(all_optimal_metrics)
        optimal_coeffs = np.asarray(optimal_coeffs)
        
        try:
            # save and produce
            ao.save_optimization_results(
                all_images=all_images,
                all_metrics=all_metrics,
                images_per_iteration=all_optimal_images,
                metrics_per_iteration=all_optimal_metrics,
                optimal_coeffs=optimal_coeffs,
                modes_to_optimize=modes_to_optimize,
                metadata=metadata,
                save_dir_path=save_dir_path / Path('')
            )     
            ao.plot_zernike_coeffs(
                optimal_coeffs=optimal_coeffs,
                num_iterations=num_iterations,
                zernike_mode_names=mode_names,
                save_dir_path=save_dir_path,
                show_fig=True,
                x_range=0.1
            )        
            ao.plot_metric_progress(
                all_metrics = all_metrics,
                modes_to_optimize = modes_to_optimize,
                num_iterations = num_iterations,
                zernike_mode_names = mode_names,
                save_dir_path = save_dir_path,
                show_fig = True,
            )
        except Exception as e:
            print(
                '\n  ----- Saving result had an exception! -----'
                f'\n {e}'
            )
        
    if display_data:
        plt.ioff()
        plt.close(fig1)
        plt.close(fig2)

if __name__ == "__main__":

    # Load ao_mirror controller
    # ao_mirror puts the mirror in the flat_position state to start.
    ao_mirror = AOMirror(
        wfc_config_file_path = wfc_config_file_path,
        haso_config_file_path = haso_config_file_path,
        interaction_matrix_file_path = wfc_correction_file_path,
        system_flat_file_path= wfc_flat_file_path,
        output_path=ao_mirror_output_path,
        control_mode='modal',
        mirror_flat_file_path=None,
        coeff_file_path=None,
        n_modes = 32,
        modes_to_ignore = []
    )
    
    # Load pymmcore-plus and connect to hardware
    mmc = CMMCorePlus.instance()
    mmc.loadSystemConfiguration(mm_configPath)

    # Run optimization
    optimize_modes(
        exposure_ms= 20.0,
        metric_to_use= 'brightness',
        num_iterations=2,
        num_mode_steps= 3,
        init_delta_range= 0.08,
        delta_range_alpha_per_iter= 0.80,
        modes_to_optimize = stationary_modes,
        save_dir_path= ao_mirror_output_path,
        verbose= True,
        image_crop=image_crop,
        display_data=True
    )

    plotDM(ao_mirror.current_voltage, show_fig=True)
    
    del ao_mirror