"""Sensorless adaptive optics.

TO DO: 
- Load interaction matrix from disk
- Set and get Zernike mode amplitudes from mirror
- Might need HASO functions to do this, since Zernike need to be composed given the pupil

2024/12 DPS initial work
"""


import numpy as np
from time import sleep

from typing import Optional, List
from typing import Optional
from pathlib import Path
from pymmcore_plus import CMMCorePlus
from tifffile import imwrite
from datetime import datetime

import matplotlib.pyplot as plt
from opm_v2.hardware.AOMirror import AOMirror, plotDM
import opm_v2.utils.sensorless_ao as ao

DEBUGGING = False
METRIC_PRECISION = 3
METRIC_PERC_THRESHOLD = 0.95

focusing_modes = [2,7,14,23]
spherical_modes = [7,14,23]
spherical_modes_first =  [7,14,23,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
all_modes = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

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

mm_configPath = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\OPM_alignment_config.cfg")

wfc_config_file_path = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\WaveFrontCorrector_mirao52-e_0329.dat"
)
haso_config_file_path = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\WFS_HASO4_VIS_7635.dat"
)
wfc_correction_file_path = Path(
    r"C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_configuration\20250808_laser_interaction_matrix.aoc"
)
# wfc_flat_file_path = Path(
#     r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_position_files\20250815_laser_curvature_closed_loop_output.wcs'
# )

wfc_flat_file_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_setup\20250818_123817_laser_optimization\20250818_laser_wfc_voltage.wcs')
output_root_path = Path(r'C:\Users\qi2lab\Documents\github\opm_v2\src\opm_v2\hardware\ao_wfc_setup')

output_prefix = '20250818_laser'
now = datetime.now()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ao_mirror_output_path = output_root_path / Path(f"{timestamp}_laser_optimization")
ao_mirror_output_path.mkdir(exist_ok=True) 

def optimize_modes(
    exposure_ms: float,
    metric_to_use: Optional[str] = "brightness",
    psf_radius_px: Optional[float] = 3,
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
    # Create hardware controller instances
    #---------------------------------------------#
    aoMirror_local = AOMirror.instance()
    mmc = CMMCorePlus.instance()

    # Enforce camera exposure
    mmc.setProperty("OrcaFusionBT", "Exposure", float(exposure_ms))
    mmc.waitForDevice("OrcaFusionBT")

    if display_data:
        setup_display=True
    
    #---------------------------------------------#
    # Setup metadata
    #---------------------------------------------#
    
    mode_deltas = [(init_delta_range - k*init_delta_range*(1-delta_range_alpha_per_iter)) for k in range(num_iterations)]
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

    #---------------------------------------------#
    # Setup Zernike modal coeff arrays
    #---------------------------------------------#
    # TODO: Do we start from system flat or current mirror state?
    aoMirror_local.apply_system_flat_voltage()
    starting_modal_coeffs = aoMirror_local.current_coeffs.copy() 
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
        best_metrics = []
        
    #---------------------------------------------#
    # Start AO optimization
    #---------------------------------------------#   
    if verbose:
        print(f"\nStarting A.O. optimization using {metric_to_use} metric")
    
    # Snap an image and calculate the starting metric.
    starting_image = mmc.snap()
    if image_crop:
        starting_image = ao.get_cropped_image(
            starting_image,
            crop_size=image_crop,
            center=None
        )
    if "DCT" in metric_to_use:
        starting_metric = ao.metric_shannon_dct(
            image=starting_image,
            psf_radius_px=psf_radius_px,
            crop_size=None
        )
    # TODO: test other information based metrics  
    elif "fourier_ratio" in metric_to_use:
        starting_metric = ao.metric_shannon_dct(
            image=starting_image,
            psf_radius_px=psf_radius_px,
            crop_size=None
        )
    elif "localize_gauss_2d" in metric_to_use:
        starting_metric =ao.metric_localize_gauss2d(
            image=starting_image
        )  
    elif 'brightness' in metric_to_use:
        starting_metric = ao.metric_brightness(
            image=starting_image,
            threshold=1000,
            image_center=None,
            return_image=False
        )
    else:
        print(f"Warning: AO metric '{metric_to_use}' not supported. Exiting function.")
        return  
        
    # update saved results
    if save_dir_path:
        all_images.append(starting_image.copy())
        all_metrics.append(starting_metric)
        all_mode_coeffs.append(starting_modal_coeffs.copy())
            
    best_metric = starting_metric
        
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
                    f'\n    Modal pertubation amplitude: {mode_deltas[k]:.3}'
                    f'\n    Perturbing mirror with Zernike mode: {mode+1}'
                    )
                
            metrics = []
            deltas = np.linspace(-mode_deltas[k], mode_deltas[k], num_mode_steps)
            for delta in deltas:
                # Create an array for applying deltas and sending to the mirror
                active_modal_coeffs = current_modal_coeffs.copy()
                active_modal_coeffs[mode] += delta
                
                # Apply Zernike pertubation mode to the mirror
                success = aoMirror_local.set_modal_coefficients(active_modal_coeffs)
                
                if not(success):
                    metric = 0
                    image = np.zeros_like(starting_image)
                                            
                else:
                    image = mmc.snap()
                    if image_crop:
                        image = ao.get_cropped_image(
                            image,
                            crop_size=image_crop,
                            center=None
                        )
                    if display_data:
                        if setup_display:
                            plt.ion()
                            fig, ax = plt.subplots()
                            im = ax.imshow(image,cmap="gray")
                            def update_image(new_data):
                                new_data = new_data / np.max(new_data)
                                im.set_data(new_data)
                                im.set_clim(vmin=0,vmax=1.1)
                                fig.canvas.flush_events()
                            update_image(image)
                            setup_display = False
                        else:
                            update_image(image)
                        sleep(1)

                    if "DCT" in metric_to_use:
                        metric = ao.metric_shannon_dct(
                            image=image,
                            psf_radius_px=psf_radius_px,
                            crop_size=None
                            )  
                    elif "localize_gauss_2d" in metric_to_use:
                        metric = ao.metric_localize_gauss2d(
                            image=image
                            )
                    elif 'brightness' in metric_to_use:
                        metric = ao.metric_brightness(
                            image=image,
                            crop_size=None,
                            threshold=1000,
                            image_center=None,
                            return_image=False
                        )

                    if metric==np.nan:
                        if verbose:
                            print('\n    ---- Metric failed == NAN ----')
                        success = False
                        metric = float(np.nan_to_num(metric))
                                        
                    if verbose:
                        print(f'      Delta={delta:.3f}, Metric = {metric:.6}, Success = {success}')

                    if DEBUGGING:
                        imwrite(Path(f"g:/ao/ao_{mode}_{delta}.tiff"),image)
                    
                metrics.append(metric)
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
                    # Test if metric samples have a peak to fit
                    is_increasing = all(x < y for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:]))
                    is_decreasing = all(x > y for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:]))
                    if is_increasing or is_decreasing:
                        raise Exception(f'Test metrics are monotonic and linear: {metrics}')
                    
                    # Optimal metric is at the peak of quadratic 
                    popt = ao.quadratic_fit(deltas, metrics)
                    a, b, c = popt
                    optimal_delta = -b / (2 * a)
                    optimal_metric = a*optimal_delta**2 + b*optimal_delta + c
                   
                    # compare to booth opt_delta
                    booth_delta = -mode_deltas[k] * (metrics[2] - metrics[0]) / (2*metrics[0] + 2*metrics[2] - 4*metrics[1])
                    if verbose:
                        print(
                            f'\n    ++   our delta: {optimal_delta}   ++',
                            f'\n    ++   Booth delta {booth_delta}   ++',
                            f'\n    ++++ Metric maximum at delta = {optimal_delta:3f} ++++'
                            f'\n    ++++ Maximum metric = {optimal_metric:.4f} ++++'
                        )

                    if a >=0:
                        raise Exception(f'Test metrics have a positive curvature: {metrics}')
                    elif (optimal_delta>mode_deltas[k]) or (optimal_delta<-mode_deltas[k]):
                        raise Exception(f'Result outside of range: opt_delta = {optimal_delta:.4f}')
                except Exception as e:
                    optimal_delta = 0
                    optimal_metric = 0
                    if verbose:
                        print(f'\n    ---- Exception in fit occurred, no changes accepted! ----\n      e:{e}')
            else:
                optimal_delta = 0
                optimal_metric = 0
                if verbose:
                    print('\n    ---- Error occurred in acquiring metric, no changes accepted! ----')
           
            #---------------------------------------------#
            # Apply the kept optimized mirror modal coeffs
            #---------------------------------------------# 
            # TODO: do we accept all the deltas or run a check against the best metric?
            if (optimal_delta!=0) and (np.round(optimal_metric,METRIC_PRECISION)>=METRIC_PERC_THRESHOLD*np.round(best_metric,METRIC_PRECISION)):
                best_metric = optimal_metric
                update = True
            else:
                optimal_delta = 0
                update = False
            if verbose:
                print(
                    f'\n    ++++ optimal metric from fit: {optimal_metric} ++++',
                    f'\n    ++++ best_metric: {best_metric} ++++',
                    f'\n    ++++ Was mirror updated: {update} ++++'
                )
            

            if update:
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
            f"\n++++ Starting Zernike mode amplitude: ++++ \n{starting_modal_coeffs}",
            f"\n++++ Final optimized Zernike mode amplitude: ++++ \n{current_modal_coeffs}"
            )
    
    if save_dir_path:
        aoMirror_local.save_current_state(prefix=output_prefix)
        aoMirror_local.save_positions_array(prefix=output_prefix)

        all_images = np.asarray(all_images)
        all_metrics = np.asarray(all_metrics)
        all_mode_coeffs = np.asarray(all_mode_coeffs)
        best_metrics = np.asarray(best_metrics)
        images_per_iteration = np.asarray(images_per_iteration)
        metrics_per_iteration = np.asarray(metrics_per_iteration)
        coeffs_per_iteration = np.asarray(coeffs_per_iteration)
        # optimized_phase = aoMirror_local.get_current_phase()
        
        # save and produce
        ao.save_optimization_results(
            all_images=all_images,
            all_metrics=all_metrics,
            best_metrics = best_metrics,
            images_per_iteration=images_per_iteration,
            metrics_per_iteration=metrics_per_iteration,
            coefficients_per_iteration=coeffs_per_iteration,
            # optimized_phase=optimized_phase,
            modes_to_optimize=modes_to_optimize,
            metadata=metadata,
            save_dir_path=save_dir_path / Path('')
        )     
        ao.plot_zernike_coeffs(
            optimal_coefficients=coeffs_per_iteration,
            zernike_mode_names=mode_names,
            save_dir_path = save_dir_path,
            show_fig = False,
            x_range = 0.05,
        )        
        ao.plot_metric_progress(
            all_metrics=all_metrics,
            num_iterations=num_iterations,
            modes_to_optimize=modes_to_optimize,
            zernike_mode_names=mode_names,
            save_dir_path=save_dir_path,
            show_fig=True
        )

    # if save_data:
    #     images = np.asarray(images,dtype=np.uint16)
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     directory_name = f"optimization_{timestamp}"
    #     save_data_directory_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao") / Path(directory_name)
    #     save_data_directory_path.mkdir(parents=True,exist_ok=True)
    #     save_data_path = save_data_directory_path / Path("optimization.tif")
    #     imwrite(
    #         save_data_path,
    #         images,
    #         imagej=True,
    #         resolution=(1.0 / .115, 1.0 / .115),
    #         metadata={'axes': 'TYX',}
    #     )
        
    #     save_data_path = save_data_directory_path / Path("final.tif")
    #     imwrite(
    #         save_data_path,
    #         final_image,
    #         imagej=True,
    #         resolution=(1.0 / .115, 1.0 / .115),
    #         metadata={'axes': 'YX',}
    #     )
        
    if display_data:
        plt.ioff()
        plt.close(fig)
    

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
        psf_radius_px = 3,
        num_iterations= 2,
        num_mode_steps= 5,
        init_delta_range= 0.1,
        delta_range_alpha_per_iter= 0.75,
        modes_to_optimize = spherical_modes_first,
        save_dir_path= ao_mirror_output_path,
        verbose= True,
        image_crop=56,
        display_data=True
    )

    plotDM(ao_mirror.current_positions)
    
    del ao_mirror