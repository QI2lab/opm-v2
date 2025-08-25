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

DEBUGGING = False
METRIC_PRECISION = 1
METRIC_PERC_THRESHOLD = 0.99
MAXIMUM_MODE_DELTA = 0.5

focusing_modes = [2,7,14,23]
spherical_modes = [7,14,23]
spherical_modes_first =  [7,14,23,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
all_modes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
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
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\WaveFrontCorrector_mirao52-e_0329.dat"
)
haso_config_file_path = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\WFS_HASO4_VIS_7635.dat"
)
wfc_correction_file_path = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\interaction_matrices\202508021_tilted_interaction_matrix.aoc"
)
wfc_flat_file_path = Path(
    r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_position_files\20250821_laser_tilted_curvature_closed_loop_output.wcs'
)
# wfc_flat_file_path = None
output_root_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_setup')

#-------------------------------------------------#
# Setup global variables
#-------------------------------------------------#

o3_stage_name = 'MCL NanoDrive Z Stage'
o3_start_position = 52.2
zstack_depth = 2.0
pixel_center = [1150,1150]
image_crop = 75
image_type = 'max'

image_history = []
current_frame_index = 0
live_mode = True
user_is_dragging_slider = False 

#-------------------------------------------------#
# Setup save directories
#-------------------------------------------------#

# Setup saving paths 
ao_save_path_prefix = f'20250821_brightness_from_wfc_nocurvature_using_{image_type}'
save_dir_suffix = f'brightness_from_wfc_nocurvature_using_{image_type}'

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
    starting_modal_coeffs = aoMirror_local.current_coeffs.copy() 
    
    # Define the coefficient deltas to use each iteration
    mode_deltas = [(init_delta_range - k*init_delta_range*(1-delta_range_alpha_per_iter)) for k in range(num_iterations)]

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
        best_metric = starting_metric
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
            plt.pause(0.001)

        # Setup Figure for showing optimal images 
        fig1, ax1 = plt.subplots()
        ax1.set_title("Optimal images")
        im1 = ax1.imshow(starting_image,cmap="gray")
        def update_image1(new_data):
            new_data = new_data / np.max(new_data)
            im1.set_data(new_data)
            im1.set_clim(vmin=0,vmax=1.1)
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            plt.pause(0.001)

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
        
        # Create empty lists for values to save
        all_images = []
        all_metrics = []
        all_mode_coeffs = []
        images_per_iteration = []
        metrics_per_iteration = [] 
        coeffs_per_iteration = []
        best_metrics = []

        # Add the starting values
        all_images.append(starting_image.copy())
        all_metrics.append(starting_metric)
        all_mode_coeffs.append(starting_modal_coeffs.copy())
            
    # Start AO iterations 
    for k in range(num_iterations): 
        # initialize current values
        if k==0:
            current_metric = starting_metric
            current_image = starting_image.copy()
            current_modal_coeffs = starting_modal_coeffs.copy()
        
        # update saved values 
        if save_dir_path:
            images_per_iteration.append(current_image)
            metrics_per_iteration.append(current_metric) 
            coeffs_per_iteration.append(current_modal_coeffs)

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

            metrics = []
            deltas = np.linspace(-mode_deltas[k], mode_deltas[k], num_mode_steps)
            for delta in deltas:
                # Create an array for to apply deltas and apply to the mirror
                active_modal_coeffs = current_modal_coeffs.copy()
                active_modal_coeffs[mode] += delta
                
                # Apply Zernike pertubation mode to the mirror
                success = aoMirror_local.set_modal_coefficients(active_modal_coeffs)
                sleep(0.500)
                if not(success):
                    metric = 0
                    image = np.zeros_like(starting_image)
                                            
                else:
                    image = acquire_zstack(image_type)
                    if image_crop:
                        image = ao.get_cropped_image(
                            image,
                            crop_size=image_crop,
                            center=pixel_center
                        )
                    if display_data:
                        update_image2(image)
                    try:
                        metric = get_metric(image, metric_to_use)
                    except Exception:
                        metric = np.nan

                    if metric==np.nan:
                        if verbose:
                            print('\n    ---- Metric failed == NAN ----')
                        success = False
                        metric = float(np.nan_to_num(metric))
                                        
                    if verbose:
                        print(f'    + Delta={delta:.4f}, Metric = {metric:.6f}, Success = {success} +')
                    
                metrics.append(metric)
                
                # update saved values
                if save_dir_path:
                    all_images.append(image.copy())
                    all_metrics.append(metric)
                    all_mode_coeffs.append(active_modal_coeffs.copy())

            #---------------------------------------------#
            # Fit metrics to determine optimal metric
            #---------------------------------------------#   
            
            # Do not accept any changes if not success
            if success:
                try:                  
                    # Test if metric samples have a peak to fit or use the maximum
                    is_increasing = all(x < y for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:]))
                    is_decreasing = all(x > y for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:]))
                    if is_increasing or is_decreasing:
                        optimal_delta = deltas[np.argmax(metrics)]
                        optimal_metric = np.max(metrics)
                        print(
                            f' --- Metrics increasing/decreasing using maximum: {optimal_delta} ---'
                        )
                        
                    else:
                        # Optimal metric is at the peak of quadratic 
                        popt = ao.quadratic_fit(deltas, metrics)
                        a, b, c = popt
                        optimal_delta = -b / (2 * a)
                    
                        # Check for reasons to reject or ignore fitting results
                        if a >=0:
                            optimal_delta = deltas[np.argmax(metrics)]
                            print(
                                f' --- Metrics have positive curvature, using maximum: {optimal_delta} ---'
                            )
                        elif np.abs(optimal_delta)>MAXIMUM_MODE_DELTA:
                            optimal_delta = 0
                            print(
                                ' --- Optimal delta is outside of range: {optimal_delta:.3f} ---'
                            )
                        
                        # Estimate the metric from fit parameters
                        optimal_metric = metric_from_fit(a, b, c, optimal_delta)

                        if verbose:
                            print(
                                f'\n ++ Optimal delta = {optimal_delta:3f} +++'
                                f'\n ++ Metric from fit: {optimal_metric:.6f} +++'
                            )
                    
                except Exception as e:
                    optimal_delta = 0
                    optimal_metric = 0
                    if verbose:
                        print(f' --- Exception in fit occurred, no changes accepted! ---\n      e:{e}')
            else:
                optimal_delta = 0
                optimal_metric = 0
                if verbose:
                    print(' --- Error occurred in acquiring image or metric, no changes accepted! ---')
           
            #---------------------------------------------#
            # Apply the kept optimized mirror modal coeffs
            #---------------------------------------------# 
            
            # TODO: do we accept all the deltas or run a check against the best metric?
            if (optimal_delta!=0) and (np.round(optimal_metric,METRIC_PRECISION)>=np.round(METRIC_PERC_THRESHOLD*best_metric,METRIC_PRECISION)):
            # if (optimal_delta!=0) and (np.round(optimal_metric,METRIC_PRECISION)>=np.round(METRIC_PERC_THRESHOLD*metrics[int(len(metrics)//2)],METRIC_PRECISION)):
                update = True

                if display_data:
                    # apply the new modal coefficient to the mirror and update current coefficients
                    temp_modal_coeff = current_modal_coeffs.copy()
                    temp_modal_coeff[mode] = current_modal_coeffs[mode] + optimal_delta
                    _ = aoMirror_local.set_modal_coefficients(temp_modal_coeff)
                    sleep(0.500)

                    # re-measure metric to verify new position
                    optimal_image = acquire_zstack()
                    if image_crop:
                        optimal_image = ao.get_cropped_image(
                            optimal_image,
                            crop_size=image_crop,
                            center=pixel_center
                        )
                    update_image1(optimal_image)

            else:
                optimal_delta = 0
                update = False
            if verbose:
                print(
                    f'\n   ++ Optimal metric: {optimal_metric:.6f} ++'
                    f'\n   ++ Best overall metric: {np.max(all_metrics):.6f} ++'
                    f'\n   ++ Was mirror updated: {update} ++'
                )
            
            if update:
                best_metric = optimal_metric

                # apply the new modal coefficient to the mirror and update current coefficients
                current_modal_coeffs[mode] = current_modal_coeffs[mode] + optimal_delta
                _ = aoMirror_local.set_modal_coefficients(current_modal_coeffs)
            
            """Loop back to top and do the next mode until all modes are done"""
                        
        """Loop back to top and do the next iteration"""
    
    #---------------------------------------------#
    # After all the iterations the mirror state will be optimized
    #---------------------------------------------# 
    aoMirror_local.update_optimized_array()
    if verbose:
        print(
            f"\n  ++++ Starting Zernike mode amplitude: ++++ \n{starting_modal_coeffs}"
            f"\n  ++++ Final optimized Zernike mode amplitude: ++++ \n{current_modal_coeffs}"
            )
    
    if save_dir_path:
        aoMirror_local.save_current_state(prefix=ao_save_path_prefix)
        aoMirror_local.save_positions_array(prefix=ao_save_path_prefix)

        all_images = np.asarray(all_images)
        all_metrics = np.asarray(all_metrics)
        all_mode_coeffs = np.asarray(all_mode_coeffs)
        best_metrics = np.asarray(best_metrics)
        images_per_iteration = np.asarray(images_per_iteration)
        metrics_per_iteration = np.asarray(metrics_per_iteration)
        coeffs_per_iteration = np.asarray(coeffs_per_iteration)
        
        # save and produce
        ao.save_optimization_results(
            all_images=all_images,
            all_metrics=all_metrics,
            best_metrics = best_metrics,
            images_per_iteration=images_per_iteration,
            metrics_per_iteration=metrics_per_iteration,
            coefficients_per_iteration=coeffs_per_iteration,
            modes_to_optimize=modes_to_optimize,
            metadata=metadata,
            save_dir_path=save_dir_path / Path('')
        )     
        ao.plot_zernike_coeffs(
            coefficients_per_iteration=coeffs_per_iteration,
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
        
    if display_data:
        plt.ioff()
        plt.close(fig1)
    

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
        exposure_ms= 10.0,
        metric_to_use= 'brightness',
        num_iterations= 2,
        num_mode_steps= 3,
        init_delta_range= 0.075,
        delta_range_alpha_per_iter= 0.5,
        modes_to_optimize = stationary_modes,
        save_dir_path= ao_mirror_output_path,
        verbose= True,
        image_crop=image_crop,
        display_data=True
    )

    plotDM(ao_mirror.current_voltage)
    
    del ao_mirror