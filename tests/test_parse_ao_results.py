from pathlib import Path
import matplotlib.pyplot as plt
from opm_v2.utils import sensorless_ao as ao

showfig = False
root_dir = Path("/home/steven/Documents/qi2lab/projects/OPM/ao_optimization_images")

for _d in root_dir.iterdir():
    if "_ao_optimize" in _d.name:
        print(_d)
        ao_data_path = _d / Path("ao_results.zarr")

        ao_results = ao.load_optimization_results(ao_data_path)

        # ao.plot_metric_progress(ao_results["metrics_per_mode"],
        #                         ao_results["metrics_per_iteration"],
        #                         ao_results["modes_to_optimize"],
        #                         ao_results["mode_names"],
        #                         _d,
        #                         showfig)

        # ao.plot_zernike_coeffs(ao_results["coefficients_per_iteration"],
        #                     ao_results["mode_names"],
        #                     _d, 
        #                     showfig)
        
        fig, axs = plt.subplots(len(ao_results["images_per_mode"]), 1, sharex=True, sharey=True)
        for ii, img in enumerate(ao_results["images_per_mode"]):
            axs[ii].imshow(img)
            
        plt.show()
            