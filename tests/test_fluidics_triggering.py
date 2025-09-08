from opm_v2.hardware.ElveFlow import OB1Controller
from opm_v2.hardware.OPMNIDAQ import OPMNIDAQ
from opm_v2.hardware.AOMirror import AOMirror
from opm_v2.utils.elveflow_control import run_fluidic_program

from pathlib import Path
import json
from tqdm import trange
import time

def main():
    # load hardware configuration file
    config_path = Path(r"C:\Users\qi2lab\Documents\github\opm_v2\opm_config_20250218.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    
    # Start the mirror in the flat_position position.
    opmAOmirror = AOMirror(
        wfc_config_file_path = Path(config["AOMirror"]["wfc_config_path"]),
        haso_config_file_path = Path(config["AOMirror"]["haso_config_path"]),
        interaction_matrix_file_path = Path(config["AOMirror"]["wfc_correction_path"]),
        flat_positions_file_path = Path(config["AOMirror"]["wfc_flat_path"]),
        n_modes = 32,
        n_positions=1,
        modes_to_ignore = []
    )
    
    opmAOmirror.apply_system_flat_voltage()

    # load OPM NIDAQ and OPM AO mirror classes
    opmNIDAQ = OPMNIDAQ(
        name = str(config["NIDAQ"]["name"]),
        scan_type = str(config["NIDAQ"]["scan_type"]),
        exposure_ms = float(config["Camera"]["exposure_ms"]),
        laser_blanking = bool(config["NIDAQ"]["laser_blanking"]),
        image_mirror_calibration = float(str(config["NIDAQ"]["image_mirror_calibration"])),
        projection_mirror_calibration = float(str(config["NIDAQ"]["projection_mirror_calibration"])),
        image_mirror_step_um = float(str(config["NIDAQ"]["image_mirror_step_um"])),
        verbose = bool(config["NIDAQ"]["verbose"])
    )
    opmNIDAQ.reset()
    
    # Initialize ElveFlow OB1 Controller
    opmOB1 = OB1Controller(
        port=config["OB1"]["port"],
        to_OB1_pin=config["OB1"]["to_OB1_pin"],
        from_OB1_pin=config["OB1"]["from_OB1_pin"]
    )
    
    #------------------------------------------------------#
    # Run test
    imaging_time = 1.0
    dt = imaging_time * 60 * 60 # s
    total_run = 12 * 60 * 60 # s
    n_r = int(total_run//dt)
    for r in trange(n_r, leave=True,desc="Imaging rounds"):
        # Run fluidics by opening an instance of the OB1controller in the method.
        run_fluidic_program(True)
        
        # Initiate fake imaging round
        for ii in trange(0, int(dt), leave=True,desc=f"Fake {imaging_time:.1f} hr acquisition"):
            time.sleep(1.0)
    
    #------------------------------------------------------#
    print("Testing complete!")
    opmOB1.close_board()

    
if __name__=="__main__":
    main()
    

